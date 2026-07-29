pub mod index;
pub mod quant;

#[cfg(target_arch = "wasm32")]
mod wasm;

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Embedding {
    pub id: String,
    pub vector: Vec<f32>,
}

impl Embedding {
    pub fn new(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingDataset {
    pub embeddings: Vec<Embedding>,
}

impl Default for EmbeddingDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingDataset {
    pub fn new() -> Self {
        Self { embeddings: vec![] }
    }

    pub fn add(&mut self, e: Embedding) {
        self.embeddings.push(e);
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let mut f = File::create(path)?;
        let data = bincode::serialize(self)?;
        f.write_all(&data)?;
        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let mut f = File::open(path)?;
        // detect if file is our streaming format by checking header
        let header = b"VECTRO+STREAM1\n";
        let qheader = b"VECTRO+QSTREAM1\n";
        let pqheader = b"VECTRO+PQSTREAM1\n";
        let rqheader = b"VECTRO+RQSTREAM1\n";
        let max_len = [header.len(), qheader.len(), pqheader.len(), rqheader.len()]
            .into_iter()
            .max()
            .unwrap_or(16);
        let mut sig = vec![0u8; max_len];
        let n = f.read(&mut sig)?;
        // reset cursor so each branch can read from the start as needed
        f.seek(SeekFrom::Start(0))?;
        if n >= header.len() && sig.as_slice()[..header.len()] == *header {
            // streaming format: multiple length-prefixed bincode Embedding entries
            let mut embeddings = Vec::new();
            // consume header
            let mut _hdr = vec![0u8; header.len()];
            let _ = f.read_exact(&mut _hdr);
            loop {
                let mut lenbuf = [0u8; 4];
                match f.read_exact(&mut lenbuf) {
                    Ok(_) => {
                        let len = u32::from_le_bytes(lenbuf) as usize;
                        let mut buf = vec![0u8; len];
                        f.read_exact(&mut buf)?;
                        let e: Embedding = bincode::deserialize(&buf)?;
                        embeddings.push(e);
                    }
                    Err(_) => break,
                }
            }
            return Ok(EmbeddingDataset { embeddings });
        }

        // maybe quantized stream
        // rewind and check for qheader
        f.seek(SeekFrom::Start(0))?;
        let mut qsig = vec![0u8; qheader.len()];
        if f.read_exact(&mut qsig).is_ok() && qsig.as_slice() == qheader {
            // we've consumed the header already; proceed (tables follow)
            // quantized stream layout: u32(table_count) u32(dim) [tables serialized as bincode] then repeated len-prefixed records: bincode((id:String, qvec:Vec<u8>))
            let mut buf4 = [0u8; 4];
            f.read_exact(&mut buf4)?;
            let _table_count = u32::from_le_bytes(buf4) as usize;
            f.read_exact(&mut buf4)?;
            let _dim = u32::from_le_bytes(buf4) as usize;
            // read tables blob length
            f.read_exact(&mut buf4)?;
            let tables_len = u32::from_le_bytes(buf4) as usize;
            let mut tblbuf = vec![0u8; tables_len];
            f.read_exact(&mut tblbuf)?;
            let _tables: Vec<crate::search::quant::QuantTable> = bincode::deserialize(&tblbuf)?;

            // now read quantized entries
            let mut embeddings = Vec::new();
            loop {
                let mut lenbuf = [0u8; 4];
                match f.read_exact(&mut lenbuf) {
                    Ok(_) => {
                        let len = u32::from_le_bytes(lenbuf) as usize;
                        let mut buf = vec![0u8; len];
                        f.read_exact(&mut buf)?;
                        let rec: (String, Vec<u8>) = bincode::deserialize(&buf)?;
                        // dequantize
                        let id = rec.0;
                        let qv = rec.1;
                        // naive dequantize each value using tables sequentially
                        // if tables length mismatch, we'll skip
                        let mut v = Vec::with_capacity(qv.len());
                        for (i, &b) in qv.iter().enumerate() {
                            let val = _tables[i].dequantize(b);
                            v.push(val);
                        }
                        embeddings.push(Embedding::new(id, v));
                    }
                    Err(_) => break,
                }
            }
            return Ok(EmbeddingDataset { embeddings });
        }

        // PQSTREAM1: codebook blob + per-vector bincode((id: String, code: Vec<u8>))
        if n >= pqheader.len() && sig[..pqheader.len()] == *pqheader {
            let mut f2 = File::open(path)?;
            let mut _hdr = vec![0u8; pqheader.len()];
            f2.read_exact(&mut _hdr)?;
            let mut buf4 = [0u8; 4];
            f2.read_exact(&mut buf4)?;
            let cb_len = u32::from_le_bytes(buf4) as usize;
            let mut cb_buf = vec![0u8; cb_len];
            f2.read_exact(&mut cb_buf)?;
            let codebook: crate::quant::pq::PQCodebook = bincode::deserialize(&cb_buf)?;
            let mut embeddings: Vec<Embedding> = Vec::new();
            loop {
                let mut lenbuf = [0u8; 4];
                match f2.read_exact(&mut lenbuf) {
                    Ok(_) => {
                        let len = u32::from_le_bytes(lenbuf) as usize;
                        let mut buf = vec![0u8; len];
                        f2.read_exact(&mut buf)?;
                        let (id, code): (String, Vec<u8>) = bincode::deserialize(&buf)?;
                        let decoded = crate::quant::pq::pq_decode(&[code], &codebook);
                        if let Some(v) = decoded.into_iter().next() {
                            embeddings.push(Embedding::new(id, v));
                        }
                    }
                    Err(_) => break,
                }
            }
            return Ok(EmbeddingDataset { embeddings });
        }

        // RQSTREAM1: codebook blob + per-vector bincode((id: String, flat_code: Vec<u8>))
        if n >= rqheader.len() && sig[..rqheader.len()] == *rqheader {
            let mut f2 = File::open(path)?;
            let mut _hdr = vec![0u8; rqheader.len()];
            f2.read_exact(&mut _hdr)?;
            let mut buf4 = [0u8; 4];
            f2.read_exact(&mut buf4)?;
            let cb_len = u32::from_le_bytes(buf4) as usize;
            let mut cb_buf = vec![0u8; cb_len];
            f2.read_exact(&mut cb_buf)?;
            let codebook: crate::quant::rq::RQCodebook = bincode::deserialize(&cb_buf)?;
            let mut embeddings: Vec<Embedding> = Vec::new();
            loop {
                let mut lenbuf = [0u8; 4];
                match f2.read_exact(&mut lenbuf) {
                    Ok(_) => {
                        let len = u32::from_le_bytes(lenbuf) as usize;
                        let mut buf = vec![0u8; len];
                        f2.read_exact(&mut buf)?;
                        let (id, flat_code): (String, Vec<u8>) = bincode::deserialize(&buf)?;
                        let decoded = crate::quant::rq::rq_decode_flat(&codebook, &[flat_code]);
                        if let Some(v) = decoded.into_iter().next() {
                            embeddings.push(Embedding::new(id, v));
                        }
                    }
                    Err(_) => break,
                }
            }
            return Ok(EmbeddingDataset { embeddings });
        }

        // fallback: rewind and read whole-file bincode
        f.seek(SeekFrom::Start(0))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let ds: EmbeddingDataset = bincode::deserialize(&buf)?;
        Ok(ds)
    }
}

/// Choose the binary format that best satisfies `target_cosine` and
/// `target_compression`, returning one of `"int8"`, `"nf4"`, `"pq"`, or `"rq"`.
pub fn auto_select_format(target_cosine: f32, target_compression: f32) -> &'static str {
    if target_cosine >= 0.9999 {
        "int8"
    } else if target_cosine >= 0.98 && target_compression <= 8.0 {
        "nf4"
    } else if target_compression <= 16.0 {
        "pq"
    } else {
        "rq"
    }
}

/// Search utilities
pub mod search {
    use crate::Embedding;
    use rayon::prelude::*;

    /// Compute dot product between two same-length slices
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Compute L2 norm of a vector
    fn norm(a: &[f32]) -> f32 {
        a.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Cosine similarity between two vectors (returns -1..1)
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return -1.0;
        }
        let denom = norm(a) * norm(b);
        if denom == 0.0 {
            return -1.0;
        }
        dot(a, b) / denom
    }

    /// Naive top-k nearest neighbors by cosine similarity.
    /// Returns a Vec of (id, score) sorted by descending score.
    pub fn top_k<'a>(dataset: &'a [Embedding], query: &[f32], k: usize) -> Vec<(&'a str, f32)> {
        let mut scores: Vec<(&str, f32)> = dataset
            .par_iter()
            .map(|e| (e.id.as_str(), cosine(&e.vector, query)))
            .collect();

        // sort descending by score
        scores.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores.into_iter().take(k).collect()
    }

    /// A simple search index that caches normalized vectors for fast cosine scoring.
    /// It owns a normalized copy of all vectors and the ids.
    pub struct SearchIndex {
        ids: Vec<String>,
        normalized: Vec<Vec<f32>>,
        dim: usize,
    }

    impl SearchIndex {
        /// Build an index from an embedding slice by normalizing each vector.
        pub fn from_dataset(dataset: &[Embedding]) -> Self {
            let mut ids = Vec::with_capacity(dataset.len());
            let mut normalized = Vec::with_capacity(dataset.len());
            let mut dim = 0usize;

            for e in dataset {
                if dim == 0 {
                    dim = e.vector.len();
                }
                ids.push(e.id.clone());
                // normalize; handle zero-norm vectors
                let n = norm(&e.vector);
                if n == 0.0 {
                    normalized.push(vec![0.0; e.vector.len()]);
                } else {
                    normalized.push(e.vector.iter().map(|v| v / n).collect());
                }
            }

            Self {
                ids,
                normalized,
                dim,
            }
        }

        /// Single query top-k using the cached normalized vectors. Query will be normalized.
        pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(&str, f32)> {
            if query.len() != self.dim {
                return vec![];
            }
            let qnorm = norm(query);
            if qnorm == 0.0 {
                return vec![];
            }
            let q: Vec<f32> = query.iter().map(|v| v / qnorm).collect();

            let mut scores: Vec<(&str, f32)> = self
                .normalized
                .par_iter()
                .zip(self.ids.par_iter())
                .map(|(vec, id)| (id.as_str(), dot(vec, &q)))
                .collect();

            scores.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.into_iter().take(k).collect()
        }

        /// Batch top-k: accept multiple queries and return a Vec per query.
        pub fn batch_top_k(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<(&str, f32)>> {
            // Parallelize across queries
            queries.par_iter().map(|q| self.top_k(q, k)).collect()
        }
    }

    /// Scalar quantization (per-dimension min/max -> u8)
    pub mod quant {
        use serde::{Deserialize, Serialize};
        /// Quantization table per-dimension
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub struct QuantTable {
            pub min: f32,
            pub max: f32,
        }

        impl QuantTable {
            pub fn new(min: f32, max: f32) -> Self {
                Self { min, max }
            }

            /// Quantize a float in [min, max] to u8
            pub fn quantize(&self, v: f32) -> u8 {
                if self.max <= self.min {
                    return 0u8;
                }
                let t = ((v - self.min) / (self.max - self.min)).clamp(0.0, 1.0);
                (t * 255.0).round() as u8
            }

            /// Dequantize a u8 back to float
            pub fn dequantize(&self, q: u8) -> f32 {
                if self.max <= self.min {
                    return self.min;
                }
                let t = (q as f32) / 255.0;
                self.min + t * (self.max - self.min)
            }
        }

        /// Quantizes a dataset of vectors per-dimension using min/max across dataset
        pub fn quantize_dataset(vectors: &[Vec<f32>]) -> (Vec<QuantTable>, Vec<Vec<u8>>) {
            if vectors.is_empty() {
                return (vec![], vec![]);
            }
            let dim = vectors[0].len();
            let mut mins = vec![f32::INFINITY; dim];
            let mut maxs = vec![f32::NEG_INFINITY; dim];
            for v in vectors {
                for (i, x) in v.iter().enumerate() {
                    if *x < mins[i] {
                        mins[i] = *x
                    }
                    if *x > maxs[i] {
                        maxs[i] = *x
                    }
                }
            }
            let tables: Vec<QuantTable> = mins
                .into_iter()
                .zip(maxs)
                .map(|(min, max)| QuantTable::new(min, max))
                .collect();

            let qvecs: Vec<Vec<u8>> = vectors
                .iter()
                .map(|v| {
                    v.iter()
                        .enumerate()
                        .map(|(i, x)| tables[i].quantize(*x))
                        .collect()
                })
                .collect();

            (tables, qvecs)
        }
    }

    /// Quantized index that stores u8 vectors with per-dimension quant tables.
    pub struct QuantizedIndex {
        ids: Vec<String>,
        tables: Vec<quant::QuantTable>,
        qvecs: Vec<Vec<u8>>,
        dim: usize,
        // optional cache of normalized dequantized vectors
        normalized_cache: Option<Vec<Vec<f32>>>,
    }

    impl QuantizedIndex {
        pub fn from_dataset(dataset: &[Embedding]) -> Self {
            let ids: Vec<String> = dataset.iter().map(|e| e.id.clone()).collect();
            let vectors: Vec<Vec<f32>> = dataset.iter().map(|e| e.vector.clone()).collect();
            let (tables, qvecs) = quant::quantize_dataset(&vectors);
            let dim = tables.len();
            Self {
                ids,
                tables,
                qvecs,
                dim,
                normalized_cache: None,
            }
        }

        /// Dequantize a u8 vector into f32 vector
        fn dequantize_vec(&self, q: &[u8]) -> Vec<f32> {
            q.iter()
                .enumerate()
                .map(|(i, &b)| self.tables[i].dequantize(b))
                .collect()
        }

        /// Top-k: dequantize vectors lazily and compute cosine with normalized query
        pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(&str, f32)> {
            if query.len() != self.dim {
                return vec![];
            }
            let qnorm = norm(query);
            if qnorm == 0.0 {
                return vec![];
            }
            let qnormed: Vec<f32> = query.iter().map(|v| v / qnorm).collect();

            let mut scores: Vec<(&str, f32)> = match &self.normalized_cache {
                Some(cache) => cache
                    .par_iter()
                    .zip(self.ids.par_iter())
                    .map(|(v, id)| (id.as_str(), dot(v, &qnormed)))
                    .collect(),
                None => self
                    .qvecs
                    .par_iter()
                    .zip(self.ids.par_iter())
                    .map(|(qv, id)| {
                        let v = self.dequantize_vec(qv);
                        // normalize dequantized vector
                        let n = norm(&v);
                        let score = if n == 0.0 {
                            -1.0
                        } else {
                            dot(&v, &qnormed) / n
                        };
                        (id.as_str(), score)
                    })
                    .collect(),
            };

            scores.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.into_iter().take(k).collect()
        }

        pub fn batch_top_k(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<(&str, f32)>> {
            queries.par_iter().map(|q| self.top_k(q, k)).collect()
        }

        /// Precompute and cache normalized dequantized vectors to accelerate scoring.
        pub fn precompute_normalized(&mut self) {
            let cache: Vec<Vec<f32>> = self
                .qvecs
                .iter()
                .map(|qv| {
                    let v = self.dequantize_vec(qv);
                    let n = norm(&v);
                    if n == 0.0 {
                        v.into_iter().map(|_| 0.0).collect()
                    } else {
                        v.into_iter().map(|x| x / n).collect()
                    }
                })
                .collect();
            self.normalized_cache = Some(cache);
        }
    }

    /// Hybrid BM25 + dense cosine search.
    ///
    /// Combines a dense vector search with an Okapi BM25 full-text search using
    /// min-max normalisation followed by a weighted linear combination:
    ///
    /// ```text
    /// combined = alpha * dense_norm + (1 - alpha) * bm25_norm
    /// ```
    ///
    /// - `alpha = 1.0` → pure dense cosine search.
    /// - `alpha = 0.0` → pure BM25.
    /// - `alpha = 0.7` (default) → dense-dominant hybrid (good for semantic tasks).
    ///
    /// Both score vectors are independently normalised to `[0, 1]` across the full
    /// corpus before combination, so neither modality dominates by scale.
    ///
    /// Returns the top-`k` document IDs sorted by descending combined score.
    pub fn hybrid_search<'a>(
        dataset: &'a [Embedding],
        bm25: &crate::index::BM25Index,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
        alpha: f32,
    ) -> Vec<(&'a str, f32)> {
        if k == 0 || dataset.is_empty() {
            return Vec::new();
        }
        let alpha = alpha.clamp(0.0, 1.0);
        let n = dataset.len();

        // ── Dense cosine scores ────────────────────────────────────────────
        let q_norm = norm(query_vector);
        let dense_raw: Vec<f32> = if q_norm == 0.0 {
            vec![0.0; n]
        } else {
            dataset
                .iter()
                .map(|e| {
                    let denom = q_norm * norm(&e.vector);
                    if denom == 0.0 {
                        0.0_f32
                    } else {
                        dot(query_vector, &e.vector) / denom
                    }
                })
                .collect()
        };

        // ── BM25 scores (sparse — build id→score map) ─────────────────────
        let bm25_hits: std::collections::HashMap<&str, f32> =
            bm25.top_k(query_text, n).into_iter().collect();
        let bm25_raw: Vec<f32> = dataset
            .iter()
            .map(|e| *bm25_hits.get(e.id.as_str()).unwrap_or(&0.0))
            .collect();

        // ── Min-max normalise both score vectors to [0, 1] ────────────────
        let normalise = |scores: &[f32]| -> Vec<f32> {
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = scores.iter().cloned().fold(f32::INFINITY, f32::min);
            let range = max - min;
            if range == 0.0 {
                return vec![0.0; scores.len()];
            }
            scores.iter().map(|&s| (s - min) / range).collect()
        };

        let dense_norm = normalise(&dense_raw);
        let bm25_norm = normalise(&bm25_raw);

        // ── Combine and return top-k ───────────────────────────────────────
        let mut combined: Vec<(&str, f32)> = dataset
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let score = alpha * dense_norm[i] + (1.0 - alpha) * bm25_norm[i];
                (e.id.as_str(), score)
            })
            .collect();

        combined
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        combined.truncate(k);
        combined
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn roundtrip_save_load() {
        let mut ds = EmbeddingDataset::new();
        ds.add(Embedding::new("one", vec![0.1, 0.2]));
        ds.add(Embedding::new("two", vec![1.0, 2.0]));

        let tmp = NamedTempFile::new().expect("create temp file");
        let path = tmp.path().to_str().unwrap().to_string();
        ds.save(&path).expect("save");

        let loaded = EmbeddingDataset::load(&path).expect("load");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.embeddings[0].id, "one");
    }

    #[test]
    fn test_embedding_dataset_new_and_len() {
        let ds = EmbeddingDataset::new();
        assert_eq!(ds.len(), 0);
        assert!(ds.is_empty());

        let mut ds2 = EmbeddingDataset::new();
        ds2.add(Embedding::new("test", vec![1.0, 2.0, 3.0]));
        assert_eq!(ds2.len(), 1);
        assert!(!ds2.is_empty());
    }

    #[test]
    fn test_embedding_dataset_default() {
        let ds: EmbeddingDataset = Default::default();
        assert_eq!(ds.len(), 0);
        assert!(ds.is_empty());
    }

    #[test]
    fn test_streaming_format_load() {
        let tmp = NamedTempFile::new().expect("create temp file");
        let path = tmp.path().to_str().unwrap().to_string();

        // Write streaming format manually
        let mut f = std::fs::File::create(&path).expect("create file");
        f.write_all(b"VECTRO+STREAM1\n").expect("write header");

        // Write first embedding
        let e1 = Embedding::new("test1", vec![1.0, 2.0]);
        let bytes1 = bincode::serialize(&e1).expect("serialize");
        let len1 = (bytes1.len() as u32).to_le_bytes();
        f.write_all(&len1).expect("write len");
        f.write_all(&bytes1).expect("write bytes");

        // Write second embedding
        let e2 = Embedding::new("test2", vec![3.0, 4.0]);
        let bytes2 = bincode::serialize(&e2).expect("serialize");
        let len2 = (bytes2.len() as u32).to_le_bytes();
        f.write_all(&len2).expect("write len");
        f.write_all(&bytes2).expect("write bytes");

        drop(f);

        let loaded = EmbeddingDataset::load(&path).expect("load");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.embeddings[0].id, "test1");
        assert_eq!(loaded.embeddings[1].id, "test2");
    }

    #[test]
    fn test_quantized_stream_format_load() {
        let tmp = NamedTempFile::new().expect("create temp file");
        let path = tmp.path().to_str().unwrap().to_string();

        // Create embeddings
        let e1 = Embedding::new("qtest1", vec![1.0, 2.0, 3.0]);
        let e2 = Embedding::new("qtest2", vec![4.0, 5.0, 6.0]);
        let vectors = vec![e1.vector.clone(), e2.vector.clone()];

        // Quantize
        let (tables, qvecs) = search::quant::quantize_dataset(&vectors);

        // Write quantized stream format
        let mut f = std::fs::File::create(&path).expect("create file");
        f.write_all(b"VECTRO+QSTREAM1\n").expect("write header");

        // Write table count and dim
        let table_count = (tables.len() as u32).to_le_bytes();
        let dim = (tables.len() as u32).to_le_bytes();
        f.write_all(&table_count).expect("write table count");
        f.write_all(&dim).expect("write dim");

        // Serialize and write tables
        let tables_blob = bincode::serialize(&tables).expect("serialize tables");
        let tables_len = (tables_blob.len() as u32).to_le_bytes();
        f.write_all(&tables_len).expect("write tables len");
        f.write_all(&tables_blob).expect("write tables");

        // Write quantized embeddings
        for (i, qv) in qvecs.iter().enumerate() {
            let id = if i == 0 { "qtest1" } else { "qtest2" };
            let rec = (id.to_string(), qv.clone());
            let bytes = bincode::serialize(&rec).expect("serialize rec");
            let len = (bytes.len() as u32).to_le_bytes();
            f.write_all(&len).expect("write len");
            f.write_all(&bytes).expect("write bytes");
        }

        drop(f);

        let loaded = EmbeddingDataset::load(&path).expect("load");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.embeddings[0].id, "qtest1");
        assert_eq!(loaded.embeddings[1].id, "qtest2");
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        use crate::search::cosine;

        // Different lengths
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine(&a, &b), -1.0);

        // Zero vectors
        let zero = vec![0.0, 0.0];
        let nonzero = vec![1.0, 2.0];
        assert_eq!(cosine(&zero, &nonzero), -1.0);
        assert_eq!(cosine(&nonzero, &zero), -1.0);
        assert_eq!(cosine(&zero, &zero), -1.0);
    }

    #[test]
    fn test_searchindex_zero_norm_query() {
        use crate::search::SearchIndex;

        let a = Embedding::new("a", vec![1.0, 2.0]);
        let ds = vec![a];
        let idx = SearchIndex::from_dataset(&ds);

        // Query with zero norm
        let zero_query = vec![0.0, 0.0];
        let results = idx.top_k(&zero_query, 1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_searchindex_from_zero_norm_vectors() {
        use crate::search::SearchIndex;

        let a = Embedding::new("zero", vec![0.0, 0.0]);
        let b = Embedding::new("nonzero", vec![1.0, 2.0]);
        let ds = vec![a, b];
        let idx = SearchIndex::from_dataset(&ds);

        let query = vec![1.0, 0.0];
        let results = idx.top_k(&query, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_quantized_index_zero_norm() {
        use crate::search::QuantizedIndex;

        let a = Embedding::new("zero", vec![0.0, 0.0]);
        let b = Embedding::new("nonzero", vec![1.0, 2.0]);
        let ds = vec![a, b];
        let idx = QuantizedIndex::from_dataset(&ds);

        let query = vec![1.0, 0.0];
        let results = idx.top_k(&query, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_quantized_index_zero_query() {
        use crate::search::QuantizedIndex;

        let a = Embedding::new("a", vec![1.0, 2.0]);
        let ds = vec![a];
        let idx = QuantizedIndex::from_dataset(&ds);

        let zero_query = vec![0.0, 0.0];
        let results = idx.top_k(&zero_query, 1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_quantized_index_dim_mismatch() {
        use crate::search::QuantizedIndex;

        let a = Embedding::new("a", vec![1.0, 2.0]);
        let ds = vec![a];
        let idx = QuantizedIndex::from_dataset(&ds);

        let wrong_query = vec![1.0, 2.0, 3.0];
        let results = idx.top_k(&wrong_query, 1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_quantized_index_with_precompute() {
        use crate::search::QuantizedIndex;

        let a = Embedding::new("a", vec![1.0, 0.0]);
        let b = Embedding::new("b", vec![0.0, 1.0]);
        let ds = vec![a, b];
        let mut idx = QuantizedIndex::from_dataset(&ds);

        // Test without precompute
        let query = vec![1.0, 0.0];
        let results1 = idx.top_k(&query, 1);
        assert_eq!(results1[0].0, "a");

        // Test with precompute
        idx.precompute_normalized();
        let results2 = idx.top_k(&query, 1);
        assert_eq!(results2[0].0, "a");
    }

    #[test]
    fn test_quant_table_edge_cases() {
        use crate::search::quant::QuantTable;

        // Min equals max
        let table = QuantTable::new(1.0, 1.0);
        assert_eq!(table.quantize(1.0), 0u8);
        assert_eq!(table.dequantize(128), 1.0);

        // Normal range
        let table2 = QuantTable::new(0.0, 10.0);
        let q = table2.quantize(5.0);
        let deq = table2.dequantize(q);
        assert!((deq - 5.0).abs() < 0.5); // Should be close
    }

    #[test]
    fn test_quantize_empty_dataset() {
        use crate::search::quant::quantize_dataset;

        let empty: Vec<Vec<f32>> = vec![];
        let (tables, qvecs) = quantize_dataset(&empty);
        assert!(tables.is_empty());
        assert!(qvecs.is_empty());
    }

    #[test]
    fn cosine_and_topk() {
        use crate::search;

        let a = Embedding::new("a", vec![1.0, 0.0]);
        let b = Embedding::new("b", vec![0.0, 1.0]);
        let c = Embedding::new("c", vec![0.707, 0.707]);

        let ds = vec![a.clone(), b.clone(), c.clone()];

        // a vs c ~ 0.707
        let sim_ac = search::cosine(&a.vector, &c.vector);
        assert!(sim_ac > 0.7 && sim_ac < 0.72);

        let top2 = search::top_k(&ds, &a.vector, 2);
        assert_eq!(top2.len(), 2);
        // first should be 'a' itself with score 1.0
        assert_eq!(top2[0].0, "a");
        assert!((top2[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn searchindex_topk_and_batch() {
        use crate::search::SearchIndex;

        let a = Embedding::new("a", vec![1.0, 0.0]);
        let b = Embedding::new("b", vec![0.0, 1.0]);
        let c = Embedding::new("c", vec![0.707, 0.707]);
        let ds = vec![a.clone(), b.clone(), c.clone()];

        let idx = SearchIndex::from_dataset(&ds);

        let q1 = vec![1.0, 0.0];
        let q2 = vec![0.0, 1.0];

        let single = idx.top_k(&q1, 2);
        let batch = idx.batch_top_k(&[q1.clone(), q2.clone()], 2);

        assert_eq!(single.len(), 2);
        assert_eq!(batch.len(), 2);
        assert_eq!(single[0].0, batch[0][0].0);
        assert!((single[0].1 - batch[0][0].1).abs() < 1e-6);
    }

    #[test]
    fn searchindex_dim_mismatch() {
        use crate::search::SearchIndex;

        let a = Embedding::new("a", vec![1.0, 0.0]);
        let ds = vec![a.clone()];
        let idx = SearchIndex::from_dataset(&ds);

        let q = vec![1.0, 0.0, 0.0];
        let res = idx.top_k(&q, 1);
        assert!(res.is_empty());
    }

    #[test]
    fn quantize_roundtrip_and_topk() {
        use crate::search::{quant, QuantizedIndex, SearchIndex};

        let a = Embedding::new("a", vec![1.0, 0.0]);
        let b = Embedding::new("b", vec![0.0, 1.0]);
        let c = Embedding::new("c", vec![0.6, 0.8]);
        let ds = vec![a.clone(), b.clone(), c.clone()];

        // Quantize dataset
        let vectors: Vec<Vec<f32>> = ds.iter().map(|e| e.vector.clone()).collect();
        let (tables, qvecs) = quant::quantize_dataset(&vectors);
        assert_eq!(tables.len(), 2);
        assert_eq!(qvecs.len(), 3);

        // dequantize first vector
        let q0 = &qvecs[0];
        let deq0: Vec<f32> = q0
            .iter()
            .enumerate()
            .map(|(i, &b)| tables[i].dequantize(b))
            .collect();
        assert!((deq0[0] - 1.0).abs() < 1e-2 || (deq0[0] - 1.0).abs() < 1e-1);

        // Compare top-k between float index and quantized index
        let float_idx = SearchIndex::from_dataset(&ds);
        let q_idx = QuantizedIndex::from_dataset(&ds);

        let query = vec![0.6f32, 0.8f32];
        let ftop = float_idx.top_k(&query, 3);
        let qtop = q_idx.top_k(&query, 3);

        // top-1 should likely be the same (c)
        assert_eq!(ftop[0].0, qtop[0].0);
    }
}
