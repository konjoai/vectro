//! Okapi BM25 full-text search index.
//!
//! Implements BM25 with Robertson–Zaragoza IDF:
//!   IDF(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
//!
//! Default hyperparameters: k1 = 1.5, b = 0.75.
//!
//! # Example
//! ```rust
//! use vectro_lib::index::bm25::BM25Index;
//!
//! let ids   = vec!["a", "b", "c"];
//! let texts = vec!["the quick brown fox", "the lazy dog", "fox over the dog"];
//! let idx   = BM25Index::build_from_texts(&ids, &texts);
//! let results = idx.top_k("fox dog", 2);
//! assert_eq!(results.len(), 2);
//! ```

use std::collections::HashMap;

/// Okapi BM25 inverted-index scorer.
///
/// Stores per-document term-frequency maps, IDF scores precomputed at build
/// time, document lengths, and corpus statistics.  All `String` storage uses
/// owned Strings so the index is self-contained and can later be serialised.
#[derive(Debug, Clone)]
pub struct BM25Index {
    /// Ordered document IDs (parallel to `doc_term_freqs` / `doc_lengths`).
    doc_ids: Vec<String>,
    /// Per-document raw TF map: term → raw count.
    doc_term_freqs: Vec<HashMap<String, u32>>,
    /// Inverted index: term → list of `(doc_idx, tf)` postings. Lets `top_k`
    /// traverse only the documents that contain a query term (the whole point of
    /// BM25) instead of scoring every document in the corpus.
    postings: HashMap<String, Vec<(u32, u32)>>,
    /// Token length of each document (number of tokens after normalisation).
    doc_lengths: Vec<f32>,
    /// Average document length across the corpus.
    avg_dl: f32,
    /// Precomputed IDF per term.  Only contains terms that appear ≥ 1 doc.
    idf: HashMap<String, f32>,
    /// Total number of documents in the index.
    n_docs: usize,
    /// Term-frequency saturation parameter (Robertson default = 1.5).
    k1: f32,
    /// Document-length normalisation weight (Robertson default = 0.75).
    b: f32,
}

// ─────────────────────────────── helpers ────────────────────────────────────

/// Normalise and tokenise a text string.
///
/// * Lowercases all characters.
/// * Splits on whitespace.
/// * Strips leading/trailing ASCII punctuation from each token so "fox!" and
///   "fox" map to the same term.
/// * Drops empty tokens.
fn tokenise(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| c.is_ascii_punctuation())
                .to_owned()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

/// Robertson–Zaragoza smoothed IDF (always ≥ 0).
///
/// `idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)`
#[inline]
fn robertson_idf(df: u32, n_docs: usize) -> f32 {
    let n = n_docs as f32;
    let df = df as f32;
    ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
}

// ──────────────────────────── implementation ────────────────────────────────

impl BM25Index {
    /// Build a BM25 index from parallel slices of document IDs and text bodies.
    ///
    /// `ids` and `texts` must have the same length; duplicate IDs are allowed
    /// (they are treated as independent documents).
    ///
    /// Uses Robertson defaults: k1 = 1.5, b = 0.75.
    ///
    /// # Panics
    /// Panics if `ids.len() != texts.len()`.
    pub fn build_from_texts(ids: &[&str], texts: &[&str]) -> Self {
        Self::build_with_params(ids, texts, 1.5, 0.75)
    }

    /// Build with explicit BM25 hyperparameters.
    ///
    /// - `k1`: TF saturation (typical range 1.2–2.0).
    /// - `b`:  Length normalisation weight (0.0 = off, 0.75 = Robertson default).
    ///
    /// # Panics
    /// Panics if `ids.len() != texts.len()`.
    pub fn build_with_params(ids: &[&str], texts: &[&str], k1: f32, b: f32) -> Self {
        assert_eq!(
            ids.len(),
            texts.len(),
            "ids and texts must have the same length"
        );

        let n_docs = ids.len();
        let doc_ids: Vec<String> = ids.iter().map(|&s| s.to_owned()).collect();

        // ── Step 1: tokenise every document, collect TF maps and lengths ─────
        let mut doc_term_freqs: Vec<HashMap<String, u32>> = Vec::with_capacity(n_docs);
        let mut doc_lengths: Vec<f32> = Vec::with_capacity(n_docs);
        let mut df: HashMap<String, u32> = HashMap::new();

        for text in texts {
            let tokens = tokenise(text);
            let mut tf: HashMap<String, u32> = HashMap::new();
            for tok in &tokens {
                *tf.entry(tok.clone()).or_insert(0) += 1;
            }
            // Update document-frequency counts
            for term in tf.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            doc_lengths.push(tokens.len() as f32);
            doc_term_freqs.push(tf);
        }

        // ── Step 2: corpus statistics ─────────────────────────────────────────
        let avg_dl = if n_docs == 0 {
            1.0
        } else {
            doc_lengths.iter().sum::<f32>() / n_docs as f32
        };

        // ── Step 3: precompute IDF for every seen term ────────────────────────
        let idf: HashMap<String, f32> = df
            .iter()
            .map(|(term, &df_count)| (term.clone(), robertson_idf(df_count, n_docs)))
            .collect();

        // ── Step 4: build the inverted index (term → [(doc_idx, tf)]) ──────────
        // Pre-sized per term from the document-frequency counts so the posting
        // Vecs never reallocate.
        let mut postings: HashMap<String, Vec<(u32, u32)>> = df
            .iter()
            .map(|(t, &c)| (t.clone(), Vec::with_capacity(c as usize)))
            .collect();
        for (doc_idx, tf_map) in doc_term_freqs.iter().enumerate() {
            for (term, &tf) in tf_map {
                if let Some(p) = postings.get_mut(term) {
                    p.push((doc_idx as u32, tf));
                }
            }
        }

        Self {
            doc_ids,
            doc_term_freqs,
            postings,
            doc_lengths,
            avg_dl,
            idf,
            n_docs,
            k1,
            b,
        }
    }

    /// BM25 score for a single document identified by its internal index.
    ///
    /// `query_tokens` must be **pre-normalised** (lowercase, punctuation
    /// stripped) — i.e. already passed through [`tokenise`] logic.
    /// Returns 0.0 for an out-of-range `doc_idx` or empty query.
    pub fn score_doc(&self, query_tokens: &[String], doc_idx: usize) -> f32 {
        if doc_idx >= self.n_docs || query_tokens.is_empty() {
            return 0.0;
        }
        let tf_map = &self.doc_term_freqs[doc_idx];
        let dl = self.doc_lengths[doc_idx];
        let norm_factor = 1.0 - self.b + self.b * (dl / self.avg_dl.max(1.0));

        query_tokens
            .iter()
            .filter_map(|term| {
                let idf = *self.idf.get(term.as_str())?;
                let tf = *tf_map.get(term.as_str()).unwrap_or(&0) as f32;
                if tf == 0.0 {
                    return None;
                }
                let bm25_tf = (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm_factor);
                Some(idf * bm25_tf)
            })
            .sum()
    }

    /// Return the top-`k` documents by BM25 score for `query`.
    ///
    /// Results are sorted descending by score.  Documents with score = 0.0 are
    /// excluded.  If `k` exceeds the number of matching documents, fewer than
    /// `k` results are returned.
    pub fn top_k<'a>(&'a self, query: &str, k: usize) -> Vec<(&'a str, f32)> {
        if k == 0 || self.n_docs == 0 {
            return Vec::new();
        }
        let tokens = tokenise(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        // Collapse repeated query terms to a per-term query frequency: scoring a
        // term's postings once and scaling by `qf` is identical to summing the
        // contribution `qf` times (what the per-document path does), but visits
        // each posting list once.
        let mut qtf: HashMap<&str, u32> = HashMap::new();
        for t in &tokens {
            *qtf.entry(t.as_str()).or_insert(0) += 1;
        }

        // Accumulate BM25 scores by walking only the postings of query terms —
        // documents containing no query term are never touched.
        let mut acc: HashMap<u32, f32> = HashMap::new();
        for (&term, &qf) in &qtf {
            let idf = match self.idf.get(term) {
                Some(&v) => v,
                None => continue,
            };
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let qf = qf as f32;
            for &(doc_idx, tf) in postings {
                let dl = self.doc_lengths[doc_idx as usize];
                let norm_factor = 1.0 - self.b + self.b * (dl / self.avg_dl.max(1.0));
                let tf = tf as f32;
                let bm25_tf = (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm_factor);
                *acc.entry(doc_idx).or_insert(0.0) += idf * bm25_tf * qf;
            }
        }

        let mut scores: Vec<(&str, f32)> = acc
            .into_iter()
            .map(|(idx, s)| (self.doc_ids[idx as usize].as_str(), s))
            .collect();

        // Partial sort: select the top-k, then fully sort just those k.
        if k < scores.len() {
            scores.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scores.truncate(k);
        }
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Number of documents in the index.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_docs
    }

    /// `true` if no documents have been indexed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_docs == 0
    }

    /// Return the IDF score for `term`, or `None` if the term is not in the
    /// vocabulary.  Useful for diagnostics and unit tests.
    pub fn idf(&self, term: &str) -> Option<f32> {
        self.idf.get(&term.to_lowercase()).copied()
    }

    /// All document IDs in index order.
    pub fn doc_ids(&self) -> &[String] {
        &self.doc_ids
    }
}

// ─────────────────────────────────── tests ──────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_index() -> BM25Index {
        let ids = vec!["doc0", "doc1", "doc2", "doc3"];
        let texts = vec![
            "the quick brown fox jumps over the lazy dog",
            "the lazy dog sat quietly under the tree",
            "a fox and a crow were sitting on a tree",
            "machine learning embeddings vector search",
        ];
        BM25Index::build_from_texts(&ids, &texts)
    }

    #[test]
    fn test_build_and_len() {
        let idx = small_index();
        assert_eq!(idx.len(), 4);
        assert!(!idx.is_empty());
    }

    #[test]
    fn test_idf_present() {
        let idx = small_index();
        // "fox" appears in doc0 and doc2 → should have a positive IDF
        let idf_fox = idx.idf("fox").expect("'fox' should be in vocabulary");
        assert!(idf_fox > 0.0, "IDF for 'fox' should be positive");
        // term not in any document → None
        assert!(idx.idf("zzzyyyy").is_none());
    }

    #[test]
    fn test_score_doc_nonzero() {
        let idx = small_index();
        let tokens = vec!["fox".to_owned()];
        // doc0 and doc2 contain "fox" — both should score > 0
        assert!(idx.score_doc(&tokens, 0) > 0.0);
        assert!(idx.score_doc(&tokens, 2) > 0.0);
        // doc3 has no Fox → score must be 0
        assert_eq!(idx.score_doc(&tokens, 3), 0.0);
    }

    #[test]
    fn test_top_k_ordering() {
        let idx = small_index();
        let results = idx.top_k("fox tree", 3);
        assert!(!results.is_empty(), "should return at least one result");
        // Scores must be non-increasing
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "results must be sorted descending: {} < {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn test_top_k_limit() {
        let idx = small_index();
        let results = idx.top_k("dog", 2);
        assert!(results.len() <= 2, "must return at most k results");
    }

    #[test]
    fn test_empty_query_returns_nothing() {
        let idx = small_index();
        assert!(idx.top_k("", 10).is_empty());
    }

    #[test]
    fn test_out_of_vocabulary_query() {
        let idx = small_index();
        // A query whose tokens are all OOV → no document can score > 0
        let results = idx.top_k("xyzzy quux frobnitz", 5);
        assert!(results.is_empty(), "OOV query should return empty results");
    }

    #[test]
    fn test_single_document_index() {
        let idx = BM25Index::build_from_texts(&["only"], &["hello world"]);
        assert_eq!(idx.len(), 1);
        let results = idx.top_k("hello", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "only");
    }

    #[test]
    fn test_empty_index() {
        let idx = BM25Index::build_from_texts(&[], &[]);
        assert!(idx.is_empty());
        assert!(idx.top_k("anything", 10).is_empty());
    }

    #[test]
    fn test_score_doc_out_of_range() {
        let idx = small_index();
        let tokens = vec!["fox".to_owned()];
        // Index 999 is OOB — should not panic and return 0
        assert_eq!(idx.score_doc(&tokens, 999), 0.0);
    }

    #[test]
    fn test_punctuation_stripped() {
        // "fox!" and "fox" should match the same term
        let idx = BM25Index::build_from_texts(&["a"], &["the fox! sat."]);
        let results = idx.top_k("fox", 1);
        assert_eq!(results.len(), 1, "trailing punctuation should be stripped");
    }

    /// Deterministic synthetic corpus: each doc is a bag of terms drawn from a
    /// vocabulary with a Zipf-ish skew (low ids common, high ids rare).
    fn synth_corpus(n_docs: usize, vocab: usize, doc_len: usize, seed: u64) -> Vec<String> {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut next = move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 33) as f64 / (1u64 << 31) as f64
        };
        (0..n_docs)
            .map(|_| {
                let words: Vec<String> = (0..doc_len)
                    .map(|_| {
                        // Skew toward low term ids: term = floor(vocab * r^3).
                        let r = next();
                        let t = ((vocab as f64) * r * r * r) as usize;
                        format!("t{}", t.min(vocab - 1))
                    })
                    .collect();
                words.join(" ")
            })
            .collect()
    }

    fn reference_top_k<'a>(idx: &'a BM25Index, query: &str, k: usize) -> Vec<(&'a str, f32)> {
        let tokens = tokenise(query);
        let mut scores: Vec<(&str, f32)> = (0..idx.len())
            .filter_map(|i| {
                let s = idx.score_doc(&tokens, i);
                (s > 0.0).then(|| (idx.doc_ids()[i].as_str(), s))
            })
            .collect();
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    #[test]
    fn top_k_inverted_matches_full_scan() {
        let texts = synth_corpus(3000, 400, 30, 7);
        let ids: Vec<String> = (0..texts.len()).map(|i| format!("d{i}")).collect();
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let idx = BM25Index::build_from_texts(&id_refs, &text_refs);

        for q in ["t0", "t1 t5", "t2 t2 t10", "t0 t1 t2 t3 t50"] {
            let got = idx.top_k(q, 10);
            let want = reference_top_k(&idx, q, 10);
            // Same score multiset (tie order may differ between HashMap iteration
            // and doc-order), and the same top score.
            let mut gs: Vec<f32> = got.iter().map(|&(_, s)| s).collect();
            let mut ws: Vec<f32> = want.iter().map(|&(_, s)| s).collect();
            assert_eq!(gs.len(), ws.len(), "result count mismatch for {q:?}");
            gs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (a, b) in gs.iter().zip(&ws) {
                assert!(
                    (a - b).abs() <= 1e-4,
                    "score mismatch for {q:?}: {a} vs {b}"
                );
            }
        }
    }

    /// `cargo test -p vectro_lib --release bm25_top_k_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bm25_top_k_bench() {
        use std::time::Instant;
        let texts = synth_corpus(200_000, 5000, 40, 1);
        let ids: Vec<String> = (0..texts.len()).map(|i| format!("d{i}")).collect();
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let idx = BM25Index::build_from_texts(&id_refs, &text_refs);
        // Selective query: rare-ish terms (high ids appear in few docs).
        let queries = ["t1200 t3400", "t800 t2500 t4100", "t4999 t4800"];

        let bench = |f: &dyn Fn(&str) -> usize| -> f64 {
            for q in &queries {
                std::hint::black_box(f(q));
            }
            let mut best = f64::INFINITY;
            for _ in 0..20 {
                let t = Instant::now();
                let mut s = 0usize;
                for q in &queries {
                    s += f(q);
                }
                std::hint::black_box(s);
                best = best.min(t.elapsed().as_secs_f64());
            }
            best * 1e6 / queries.len() as f64
        };
        let inv_us = bench(&|q| idx.top_k(q, 10).len());
        let scan_us = bench(&|q| reference_top_k(&idx, q, 10).len());
        println!(
            "bm25 top_k n={} vocab=5000: full-scan={scan_us:.1}us inverted={inv_us:.1}us ({:.1}x)",
            idx.len(),
            scan_us / inv_us
        );
    }

    #[test]
    fn test_build_with_params() {
        // k1=0 → TF is completely ignored; every matched doc gets the same IDF
        let idx =
            BM25Index::build_with_params(&["d0", "d1"], &["fox fox fox", "fox once"], 0.0, 0.75);
        let tokens = vec!["fox".to_owned()];
        let s0 = idx.score_doc(&tokens, 0);
        let s1 = idx.score_doc(&tokens, 1);
        // With k1=0, BM25-TF = (f * 1) / f = 1 → scores identical
        assert!(
            (s0 - s1).abs() < 1e-5,
            "k1=0 should produce equal scores regardless of TF"
        );
    }
}
