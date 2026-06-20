//! Generic HNSW index over any [`Quantizer`].
//!
//! `QuantHnswIndex<Q>` mirrors `HnswIndex` in structure and public API but
//! stores compressed `Q::Encoded` vectors instead of raw `Vec<f32>`.
//!
//! # Asymmetric Distance Quantization (ADQ)
//! The query is always kept as plain f32 throughout beam search, while stored
//! nodes are compared via `Q::dist_to_query`.  This avoids accumulating
//! quantization error on both sides of every comparison (the "symmetric"
//! error) and closely matches the FAISS ADC approach.
//!
//! # Graph construction
//! New nodes are inserted while normalised to unit length; then encoded via
//! `Q::encode`.  During neighbor scoring (both candidate selection and
//! reverse-link pruning) the current node is the f32 query and stored
//! neighbors are decoded on-the-fly with `Q::dist_to_query`, so construction
//! is fully asymmetric too.
//!
//! # Convenience type aliases
//! | Alias             | Quantizer        | bits/dim |
//! |-------------------|------------------|----------|
//! | `Bf16HnswIndex`   | `Bf16Quantizer`  | 16       |
//! | `Int8HnswIndex`   | `Int8Quantizer`  | 8        |
//! | `Nf4HnswIndex`    | `Nf4Quantizer`   | 4        |
//! | `Sq3HnswIndex`    | `Sq3Quantizer`   | 3        |
//! | `Sq2HnswIndex`    | `Sq2Quantizer`   | 2        |
//! | `BinaryHnswIndex` | `BinaryQuantizer`| 1        |

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashSet};

use super::shuffled_order;

use crate::quant::{
    Bf16Quantizer, BinaryQuantizer, Int8Quantizer, Nf4Quantizer, Quantizer, Sq2Quantizer,
    Sq3Quantizer,
};

// ── Convenience type aliases ──────────────────────────────────────────────────

/// HNSW backed by 16-bit BFloat representations.
pub type Bf16HnswIndex = QuantHnswIndex<Bf16Quantizer>;
/// HNSW backed by 8-bit INT8 abs-max symmetric quantization.
pub type Int8HnswIndex = QuantHnswIndex<Int8Quantizer>;
/// HNSW backed by 4-bit NF4 normal-float quantization.
pub type Nf4HnswIndex = QuantHnswIndex<Nf4Quantizer>;
/// HNSW backed by 3-bit uniform scalar quantization.
pub type Sq3HnswIndex = QuantHnswIndex<Sq3Quantizer>;
/// HNSW backed by 2-bit uniform scalar quantization.
pub type Sq2HnswIndex = QuantHnswIndex<Sq2Quantizer>;
/// HNSW backed by 1-bit sign (binary) quantization.
pub type BinaryHnswIndex = QuantHnswIndex<BinaryQuantizer>;

/// Per-layer candidate neighbour lists for one node: `(layer, [(dist, id), …])`,
/// the read-only output of the search phase handed to the commit phase.
type LayerCandidates = Vec<(usize, Vec<(f32, usize)>)>;

/// Reused per-thread "visited" set for beam search: an epoch-tagged array
/// instead of a fresh `HashSet` per `search_layer` call. `begin` bumps the
/// epoch in O(1) (no clear), so a node is "visited this call" iff its tag equals
/// the current epoch. Thread-local, so it's contention-free under the parallel
/// build and parallel batch search (each worker thread gets its own).
struct SearchScratch {
    tags: Vec<u32>,
    epoch: u32,
    cands: BinaryHeap<(std::cmp::Reverse<OrdF32>, usize)>,
    window: BinaryHeap<(OrdF32, usize)>,
}

impl SearchScratch {
    /// Prepare for one `search_layer` call: grow the tag array to `n`, bump the
    /// epoch (O(1) "clear"), and empty the heaps while keeping their capacity.
    fn begin(&mut self, n: usize) -> u32 {
        if self.tags.len() < n {
            self.tags.resize(n, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // Wrapped after 4B searches on this thread — clear and restart.
            self.tags.iter_mut().for_each(|t| *t = 0);
            self.epoch = 1;
        }
        self.cands.clear();
        self.window.clear();
        self.epoch
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<SearchScratch> = const {
        std::cell::RefCell::new(SearchScratch {
            tags: Vec::new(),
            epoch: 0,
            cands: BinaryHeap::new(),
            window: BinaryHeap::new(),
        })
    };
}

// ── OrdF32 (private total-order wrapper for BinaryHeap) ───────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ── QuantHnswIndex ────────────────────────────────────────────────────────────

/// Generic HNSW index storing quantized vectors of type `Q::Encoded`.
///
/// Build with [`QuantHnswIndex::new`], insert with [`add`] / [`add_batch`],
/// query with [`search`].  Supports soft deletion, filtered search,
/// save/load, and vacuum.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize   = "Q::Encoded: Serialize",
    deserialize = "Q::Encoded: for<'de2> Deserialize<'de2>"
))]
pub struct QuantHnswIndex<Q: Quantizer> {
    m: usize,
    /// Max links at layer 0 = 2 * m.
    m0: usize,
    ef_construction: usize,
    /// Level multiplier = 1 / ln(m).
    ml: f64,
    /// Per-node quantized representations.
    encoded: Vec<Q::Encoded>,
    /// `neighbors[node][layer] = [neighbor_id, ...]`
    neighbors: Vec<Vec<Vec<usize>>>,
    entry_point: Option<usize>,
    max_level: usize,
    /// Soft-deletion tombstones; always aligned to `encoded.len()`.
    #[serde(default)]
    deleted: Vec<bool>,
    /// Optional centering vector (mean of the first batch's unit vectors).
    ///
    /// Auto-enabled for 1-bit (binary) quantization, where un-centered sign
    /// bits on real embeddings (which share a large mean direction) collapse
    /// to near-identical codes and destroy graph navigability. Subtracting the
    /// mean before sign-quantization restores informative bits. `None` for all
    /// other quantizers and for indexes built before this field existed.
    #[serde(default)]
    center: Option<Vec<f32>>,
    /// Transient full-precision (centered) vectors held *only during graph
    /// construction*. The HNSW graph is built using exact f32 distances — graph
    /// links are integers, so the topology costs nothing at rest — then these
    /// are dropped (`finalize`) leaving only the quantized codes in memory.
    /// Searching a coarse 1-bit graph built from 1-bit distances collapses
    /// recall (the graph isn't navigable); building from f32 and searching the
    /// quantized codes recovers it. Never serialized; empty after load.
    #[serde(skip)]
    build_vectors: Vec<Vec<f32>>,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<Q>,
}

impl<Q: Quantizer> QuantHnswIndex<Q> {
    /// Create a new empty index.
    ///
    /// * `m`               — max bidirectional links per node in layers ≥ 1.
    /// * `ef_construction` — beam width used during insertion.
    pub fn new(m: usize, ef_construction: usize) -> Self {
        assert!(m >= 2, "m must be >= 2");
        assert!(ef_construction >= m, "ef_construction must be >= m");
        let ml = 1.0 / (m as f64).ln();
        Self {
            m,
            m0: 2 * m,
            ef_construction,
            ml,
            encoded: Vec::new(),
            neighbors: Vec::new(),
            entry_point: None,
            max_level: 0,
            deleted: Vec::new(),
            center: None,
            build_vectors: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Number of vectors currently stored (including soft-deleted).
    pub fn len(&self) -> usize {
        self.encoded.len()
    }

    /// True when the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.encoded.is_empty()
    }

    // ─────────────────────── private helpers ─────────────────────────────────

    fn normalize(v: &[f32]) -> Vec<f32> {
        let sq: f32 = v.iter().map(|x| x * x).sum();
        if sq == 0.0 {
            return v.to_vec();
        }
        let inv = 1.0 / sq.sqrt();
        v.iter().map(|x| x * inv).collect()
    }

    /// Whether mean-centering is applied for this quantizer.
    ///
    /// Enabled only for 1-bit (binary) quantization — see the `center` field.
    #[inline]
    fn centering_enabled() -> bool {
        Q::bits_per_dim() == 1
    }

    /// Apply the stored centering transform to an already-unit-normalised vector:
    /// subtract the mean direction and re-normalise so the result stays unit-norm
    /// in the centered space. No-op when no center is set (e.g. non-binary
    /// quantizers, or before the first batch has established a center).
    fn apply_center(&self, normalized: &[f32]) -> Vec<f32> {
        match &self.center {
            Some(c) if c.len() == normalized.len() => {
                let centered: Vec<f32> =
                    normalized.iter().zip(c.iter()).map(|(x, m)| x - m).collect();
                Self::normalize(&centered)
            }
            _ => normalized.to_vec(),
        }
    }

    /// Compute the mean of the unit-normalised input vectors. Used to establish
    /// the centering vector from the first batch when centering is enabled.
    fn compute_center(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
        let dim = vectors.first().map(|v| v.len())?;
        if dim == 0 {
            return None;
        }
        let mut mean = vec![0.0f32; dim];
        for v in vectors {
            let nv = Self::normalize(v);
            for (m, x) in mean.iter_mut().zip(nv.iter()) {
                *m += x;
            }
        }
        let inv = 1.0 / vectors.len() as f32;
        for m in mean.iter_mut() {
            *m *= inv;
        }
        Some(mean)
    }

    #[inline]
    fn is_deleted(&self, id: usize) -> bool {
        self.deleted.get(id).copied().unwrap_or(false)
    }

    /// Deterministic geometric-distribution level for a given node id.
    ///
    /// Seeded purely by `id` (not insertion-time mutable state) so the parallel
    /// build assigns each node the *same* level the serial path would.
    fn level_of(id: usize, ml: f64) -> usize {
        let mut r = (id as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        r ^= r >> 33;
        r = r.wrapping_mul(0xff51afd7ed558ccd);
        r ^= r >> 33;
        let frac = (r >> 11) as f64 / (1u64 << 53) as f64;
        let frac = frac.max(1e-15);
        ((-frac.ln()) * ml) as usize
    }

    /// Deterministic geometric-distribution level via LCG of node count.
    fn random_level(&self) -> usize {
        Self::level_of(self.encoded.len(), self.ml)
    }

    /// Distance from a stored node to an f32 query.
    ///
    /// During construction (`use_f32 = true`) the transient full-precision
    /// `build_vectors` are used for an exact distance, yielding a high-quality
    /// graph. At query time (`use_f32 = false`, or after `finalize`) the
    /// asymmetric quantized distance `Q::dist_to_query` is used over the codes.
    #[inline]
    fn node_dist(&self, id: usize, query: &[f32], use_f32: bool) -> f32 {
        if use_f32 {
            if let Some(v) = self.build_vectors.get(id) {
                if !v.is_empty() {
                    // build_vectors and the construction query are both unit-norm.
                    return crate::quant::cosine_dist_unit(v, query);
                }
            }
        }
        Q::dist_to_query(&self.encoded[id], query)
    }

    /// Core beam search with an optional per-node inclusion predicate.
    ///
    /// `use_f32` selects the construction-time exact distance vs. the query-time
    /// asymmetric quantized distance (see [`node_dist`]). Deleted nodes are
    /// always excluded from the result window regardless of `filter`.  Excluded
    /// nodes are still *traversed* so connectivity is preserved.
    fn search_layer_impl<F: Fn(usize) -> bool>(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        use_f32: bool,
        filter: F,
    ) -> Vec<(f32, usize)> {
        SCRATCH.with(|cell| {
            let s = &mut *cell.borrow_mut();
            let epoch = s.begin(self.encoded.len());

            for &ep in entry_points {
                let d = self.node_dist(ep, query, use_f32);
                s.tags[ep] = epoch;
                s.cands.push((std::cmp::Reverse(OrdF32(d)), ep));
                if !self.is_deleted(ep) && filter(ep) {
                    s.window.push((OrdF32(d), ep));
                }
            }

            while let Some((std::cmp::Reverse(OrdF32(d_c)), c)) = s.cands.pop() {
                let worst = s.window.peek().map(|e| e.0 .0).unwrap_or(f32::INFINITY);
                if d_c > worst && s.window.len() >= ef {
                    break;
                }
                // Borrow the neighbour slice instead of cloning — search never
                // mutates the graph, so this avoids a Vec<usize> allocation per
                // expanded candidate (hot path).
                let nbrs: &[usize] = self
                    .neighbors[c]
                    .get(layer)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                for &nb in nbrs {
                    if s.tags[nb] == epoch {
                        continue;
                    }
                    s.tags[nb] = epoch;
                    let d_nb = self.node_dist(nb, query, use_f32);
                    let worst2 = s.window.peek().map(|e| e.0 .0).unwrap_or(f32::INFINITY);
                    if d_nb < worst2 || s.window.len() < ef {
                        s.cands.push((std::cmp::Reverse(OrdF32(d_nb)), nb));
                        if !self.is_deleted(nb) && filter(nb) {
                            s.window.push((OrdF32(d_nb), nb));
                            if s.window.len() > ef {
                                s.window.pop();
                            }
                        }
                    }
                }
            }

            let mut result: Vec<(f32, usize)> =
                s.window.drain().map(|(d, id)| (d.0, id)).collect();
            result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            result
        })
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        use_f32: bool,
    ) -> Vec<(f32, usize)> {
        self.search_layer_impl(query, entry_points, ef, layer, use_f32, |_| true)
    }

    fn select_neighbors(candidates: &[(f32, usize)], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|&(_, id)| id).collect()
    }

    // ─────────────────────────── public API ─────────────────────────────────

    /// Insert a single vector (normalised internally; then encoded).
    ///
    /// The vector is unit-normalised, then mean-centered (binary only), and the
    /// *same* transformed vector is used both as the stored code and as the f32
    /// query that drives graph construction — keeping build and search in one
    /// consistent metric space.
    pub fn add(&mut self, vector: &[f32]) {
        let norm_vec = self.apply_center(&Self::normalize(vector));
        let node_id = self.encoded.len();
        let node_level = self.random_level();

        // Retain the full-precision vector for graph construction only while the
        // transient buffer is still aligned (i.e. not yet finalized). After
        // `finalize`, post-hoc inserts fall back to the quantized distance.
        let building = self.build_vectors.len() == node_id;

        self.encoded.push(Q::encode(&norm_vec));
        if building {
            self.build_vectors.push(norm_vec.clone());
        }
        self.neighbors.push(vec![vec![]; node_level + 1]);
        self.deleted.push(false);

        match self.entry_point {
            None => {
                self.entry_point = Some(node_id);
                self.max_level = node_level;
            }
            Some(ep) => {
                let mut curr_ep = vec![ep];
                let max_l = self.max_level;

                // Greedy descent from top down to node_level + 1 (ef = 1).
                for lc in (node_level + 1..=max_l).rev() {
                    let res = self.search_layer(&norm_vec, &curr_ep, 1, lc, building);
                    if !res.is_empty() {
                        curr_ep = vec![res[0].1];
                    }
                }

                // ef_construction-width search from min(node_level, max_l) → 0.
                for lc in (0..=node_level.min(max_l)).rev() {
                    let candidates =
                        self.search_layer(&norm_vec, &curr_ep, self.ef_construction, lc, building);
                    let max_m = if lc == 0 { self.m0 } else { self.m };
                    let nbrs = Self::select_neighbors(&candidates, max_m);

                    self.neighbors[node_id][lc] = nbrs.clone();
                    curr_ep = candidates.into_iter().map(|(_, id)| id).collect();

                    // Add reverse links and prune if over max_m.
                    for nb_id in nbrs {
                        if lc < self.neighbors[nb_id].len() {
                            self.neighbors[nb_id][lc].push(node_id);
                            if self.neighbors[nb_id][lc].len() > max_m {
                                // Score nb's neighbors from nb's own viewpoint.
                                // Prefer nb's full-precision vector (exact pruning)
                                // and fall back to its decoded code post-finalize.
                                let nb_query: Vec<f32> = self
                                    .build_vectors
                                    .get(nb_id)
                                    .filter(|v| !v.is_empty())
                                    .cloned()
                                    .unwrap_or_else(|| Q::decode(&self.encoded[nb_id], 0));
                                let mut scored: Vec<(f32, usize)> = self.neighbors[nb_id][lc]
                                    .iter()
                                    .map(|&n| (self.node_dist(n, &nb_query, building), n))
                                    .collect();
                                scored.sort_by(|a, b| {
                                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                                });
                                self.neighbors[nb_id][lc] = scored
                                    .into_iter()
                                    .take(max_m)
                                    .map(|(_, id)| id)
                                    .collect();
                            }
                        }
                    }
                }

                if node_level > max_l {
                    self.max_level = node_level;
                    self.entry_point = Some(node_id);
                }
            }
        }
    }

    /// Minimum batch size into an empty index that triggers the parallel build.
    /// Below this the serial path avoids rayon/setup overhead.
    const PARALLEL_BUILD_THRESHOLD: usize = 512;

    /// Chunk size for the parallel build: a node can't link to its chunk-mates
    /// (they search the same frozen graph), so the chunk is bounded to ~n/64 to
    /// keep the fraction of unlinkable true-neighbours small, clamped to a
    /// [64, 1024] band for sane parallelism. Combined with a shuffled build
    /// order (see [`build_parallel`]) this keeps recall within noise of serial
    /// regardless of input ordering.
    fn parallel_build_chunk(n: usize) -> usize {
        (n / 64).clamp(64, 1024)
    }

    /// Insert a batch of vectors.
    ///
    /// On the first batch into an empty index, when centering is enabled
    /// (binary), the mean of the batch's unit vectors is computed and stored as
    /// the centering vector before any insertion. This is the recommended bulk
    /// path for binary indexes — pure incremental `add()` cannot establish a
    /// center and will store un-centered codes.
    ///
    /// A large first batch is built in parallel (see [`build_parallel`]).
    pub fn add_batch(&mut self, vectors: &[Vec<f32>]) {
        if Self::centering_enabled()
            && self.center.is_none()
            && self.encoded.is_empty()
            && !vectors.is_empty()
        {
            self.center = Self::compute_center(vectors);
        }

        if self.encoded.is_empty() && vectors.len() >= Self::PARALLEL_BUILD_THRESHOLD {
            self.build_parallel(vectors);
        } else {
            for v in vectors {
                self.add(v);
            }
        }
        // Reclaim the transient f32 build buffer so the index keeps only the
        // quantized codes at rest — this is what delivers the memory saving.
        self.finalize();
    }

    /// Parallel graph construction for a fresh index.
    ///
    /// Strategy: **parallel search + serial commit**, in chunks. The expensive
    /// part — searching the graph for each node's candidate neighbours — is
    /// purely read-only and runs concurrently across a chunk; the cheap part —
    /// stitching forward/reverse links and pruning — is applied serially, so no
    /// locks or `unsafe` are needed and the on-disk layout is unchanged.
    ///
    /// Nodes within one chunk search the graph frozen at the chunk boundary, so
    /// they can't link to each other; a modest chunk size keeps the recall hit
    /// negligible while exposing ~Ncores of parallelism on the search phase.
    fn build_parallel(&mut self, vectors: &[Vec<f32>]) {
        let n = vectors.len();

        // Phase 0 — encode + center every vector in parallel (read-only).
        let transformed: Vec<Vec<f32>> = vectors
            .par_iter()
            .map(|v| self.apply_center(&Self::normalize(v)))
            .collect();
        let encoded: Vec<Q::Encoded> = transformed.par_iter().map(|t| Q::encode(t)).collect();
        let levels: Vec<usize> = (0..n).map(|id| Self::level_of(id, self.ml)).collect();

        self.encoded = encoded;
        self.build_vectors = transformed;
        self.deleted = vec![false; n];
        self.neighbors = levels.iter().map(|&lv| vec![Vec::new(); lv + 1]).collect();

        // Shuffled processing order: chunk-mates must not be true neighbours,
        // and adjacent input rows are often correlated, so process in a
        // deterministic pseudo-random order. Node `order[0]` bootstraps as the
        // initial entry point.
        let order = shuffled_order(n);
        self.entry_point = Some(order[0]);
        self.max_level = levels[order[0]];

        let chunk = Self::parallel_build_chunk(n);
        let mut start = 1;
        while start < n {
            let end = (start + chunk).min(n);
            let ids = &order[start..end];
            // Parallel, read-only: each node searches the frozen graph.
            let found: Vec<LayerCandidates> = ids
                .par_iter()
                .map(|&id| self.find_candidates(id, levels[id]))
                .collect();
            // Serial: commit links in processing order.
            for (&id, per_layer) in ids.iter().zip(found.into_iter()) {
                self.commit_node(id, levels[id], per_layer);
            }
            start = end;
        }
    }

    /// Read-only neighbour search for `node_id` against the current graph,
    /// returning the per-layer candidate lists (closest-first). Mirrors the
    /// search portion of [`add`] but commits nothing. Safe to call concurrently.
    fn find_candidates(&self, node_id: usize, node_level: usize) -> LayerCandidates {
        let q = &self.build_vectors[node_id];
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };
        let max_l = self.max_level;
        let mut curr_ep = vec![ep];

        // Greedy descent (ef = 1) down to node_level + 1, using exact f32 dist.
        for lc in (node_level + 1..=max_l).rev() {
            let res = self.search_layer(q, &curr_ep, 1, lc, true);
            if !res.is_empty() {
                curr_ep = vec![res[0].1];
            }
        }

        let mut per_layer = Vec::new();
        for lc in (0..=node_level.min(max_l)).rev() {
            let candidates = self.search_layer(q, &curr_ep, self.ef_construction, lc, true);
            curr_ep = candidates.iter().map(|&(_, id)| id).collect();
            per_layer.push((lc, candidates));
        }
        per_layer
    }

    /// Serially commit a node's links from its pre-computed per-layer candidates:
    /// set forward links, add reverse links, prune over-full lists, and promote
    /// the entry point when the node introduces a new top layer.
    fn commit_node(&mut self, node_id: usize, node_level: usize, per_layer: LayerCandidates) {
        for (lc, candidates) in per_layer {
            let max_m = if lc == 0 { self.m0 } else { self.m };
            let nbrs = Self::select_neighbors(&candidates, max_m);
            self.neighbors[node_id][lc] = nbrs.clone();

            for nb_id in nbrs {
                if lc < self.neighbors[nb_id].len() {
                    self.neighbors[nb_id][lc].push(node_id);
                    if self.neighbors[nb_id][lc].len() > max_m {
                        let nb_query: Vec<f32> = self
                            .build_vectors
                            .get(nb_id)
                            .filter(|v| !v.is_empty())
                            .cloned()
                            .unwrap_or_else(|| Q::decode(&self.encoded[nb_id], 0));
                        let mut scored: Vec<(f32, usize)> = self.neighbors[nb_id][lc]
                            .iter()
                            .map(|&n| (self.node_dist(n, &nb_query, true), n))
                            .collect();
                        scored.sort_by(|a, b| {
                            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        self.neighbors[nb_id][lc] =
                            scored.into_iter().take(max_m).map(|(_, id)| id).collect();
                    }
                }
            }
        }

        if node_level > self.max_level {
            self.max_level = node_level;
            self.entry_point = Some(node_id);
        }
    }

    /// Drop the transient full-precision construction buffer, leaving only the
    /// quantized codes in memory. Called automatically at the end of
    /// [`add_batch`]; idempotent. After finalization the graph is fixed and
    /// further `add` calls construct against the quantized distance.
    pub fn finalize(&mut self) {
        self.build_vectors = Vec::new();
        self.build_vectors.shrink_to_fit();
    }

    /// Approximate k-nearest-neighbour search.
    ///
    /// Returns `Vec<(node_id, cosine_distance)>` sorted ascending by distance.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        let ep = match self.entry_point {
            None => return vec![],
            Some(ep) => ep,
        };
        let ef = ef.max(k);
        let q = self.apply_center(&Self::normalize(query));
        let mut curr_ep = vec![ep];

        // Greedy descent to layer 1.
        for lc in (1..=self.max_level).rev() {
            let res = self.search_layer(&q, &curr_ep, 1, lc, false);
            if !res.is_empty() {
                curr_ep = vec![res[0].1];
            }
        }

        // Full beam search at layer 0.
        let res = self.search_layer(&q, &curr_ep, ef, 0, false);
        res.into_iter().take(k).map(|(d, id)| (id, d)).collect()
    }

    /// Batch k-NN search over a row-major `[q, d]` flat query buffer, run in
    /// parallel across queries (rayon). Search is `&self`/read-only, so queries
    /// are independent — scales with cores for batched/serving workloads.
    pub fn search_batch_flat(
        &self,
        flat: &[f32],
        d: usize,
        k: usize,
        ef: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        if d == 0 {
            return Vec::new();
        }
        let q = flat.len() / d;
        (0..q)
            .into_par_iter()
            .map(|i| self.search(&flat[i * d..(i + 1) * d], k, ef))
            .collect()
    }

    /// Compute mean recall@k over a set of queries.
    pub fn recall_at_k(
        &self,
        queries: &[Vec<f32>],
        ground_truth: &[Vec<usize>],
        k: usize,
        ef: usize,
    ) -> f32 {
        assert_eq!(queries.len(), ground_truth.len(), "queries/gt length mismatch");
        let total: f32 = queries
            .iter()
            .zip(ground_truth.iter())
            .map(|(q, gt)| {
                let results = self.search(q, k, ef);
                let found: HashSet<usize> = results.iter().map(|&(id, _)| id).collect();
                let hits = gt.iter().take(k).filter(|&&id| found.contains(&id)).count();
                hits as f32 / k.min(gt.len()).max(1) as f32
            })
            .sum();
        total / queries.len() as f32
    }

    /// Soft-delete a vector by ID.
    pub fn delete(&mut self, id: usize) {
        if id < self.encoded.len() {
            if self.deleted.len() < self.encoded.len() {
                self.deleted.resize(self.encoded.len(), false);
            }
            self.deleted[id] = true;
        }
    }

    /// Approximate k-nearest-neighbour search with a predicate filter.
    ///
    /// Only nodes where `predicate(id)` returns `true` are eligible for the
    /// result set.  Filtered-out nodes are still traversed to maintain graph
    /// connectivity.
    pub fn search_filtered<F: Fn(usize) -> bool>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        predicate: F,
    ) -> Vec<(usize, f32)> {
        let ep = match self.entry_point {
            None => return vec![],
            Some(ep) => ep,
        };
        let ef = ef.max(k);
        let q = self.apply_center(&Self::normalize(query));
        let mut curr_ep = vec![ep];

        for lc in (1..=self.max_level).rev() {
            let res = self.search_layer(&q, &curr_ep, 1, lc, false);
            if !res.is_empty() {
                curr_ep = vec![res[0].1];
            }
        }

        let res = self.search_layer_impl(&q, &curr_ep, ef, 0, false, predicate);
        res.into_iter().take(k).map(|(d, id)| (id, d)).collect()
    }

    /// Persist the index to a file using bincode serialization.
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load an index previously saved with [`QuantHnswIndex::save`].
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let idx: Self = bincode::deserialize(&bytes)?;
        Ok(idx)
    }

    /// Compact the index by permanently removing all soft-deleted nodes and
    /// rebuilding the HNSW graph from the decoded survivors.
    ///
    /// Survivors are decoded from their quantized representation, re-normalised,
    /// and re-inserted.  Graph quality reflects the quantization error in the
    /// decoded vectors.  Returns the number of nodes removed.  No-op when no
    /// nodes are deleted.
    pub fn vacuum(&mut self) -> usize {
        let deleted_count = self.deleted.iter().filter(|&&d| d).count();
        if deleted_count == 0 {
            return 0;
        }

        // Decode surviving vectors and re-normalise.
        let survivors: Vec<Vec<f32>> = self
            .encoded
            .iter()
            .zip(self.deleted.iter())
            .filter(|(_, &d)| !d)
            .map(|(enc, _)| {
                let dec = Q::decode(enc, 0);
                Self::normalize(&dec)
            })
            .collect();

        let mut new_idx = QuantHnswIndex::<Q>::new(self.m, self.ef_construction);
        new_idx.add_batch(&survivors);
        *self = new_idx;
        deleted_count
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{
        Bf16Quantizer, BinaryQuantizer, Int8Quantizer, Nf4Quantizer, Sq2Quantizer, Sq3Quantizer,
    };

    /// Synthetic test vectors: deterministic sin-based, NOT unit-normalised
    /// (the index normalises internally).
    fn make_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..d).map(|j| ((i * d + j) as f32 * 0.017 + 0.1).sin()).collect())
            .collect()
    }

    /// Brute-force cosine nearest-neighbours for ground-truth recall computation.
    fn brute_force_gt(vecs: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
        queries
            .iter()
            .map(|q| {
                let q_n: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
                let mut scores: Vec<(usize, f32)> = vecs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let v_n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let dot: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                        let cos = if q_n * v_n > 0.0 { dot / (q_n * v_n) } else { -1.0 };
                        (i, cos)
                    })
                    .collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scores.into_iter().take(k).map(|(i, _)| i).collect()
            })
            .collect()
    }

    // ── smoke tests (build + basic search) ───────────────────────────────────

    macro_rules! smoke_test {
        ($name:ident, $Q:ty) => {
            #[test]
            fn $name() {
                let mut idx = QuantHnswIndex::<$Q>::new(8, 50);
                let vecs = make_vecs(50, 32);
                idx.add_batch(&vecs);
                assert_eq!(idx.len(), 50);

                // Distance-to-self should be very small.
                let results = idx.search(&vecs[0], 1, 50);
                assert_eq!(results.len(), 1);
                assert!(
                    results[0].1 < 0.15,
                    "dist-to-self[0] too large: {}",
                    results[0].1
                );
            }
        };
    }

    smoke_test!(smoke_bf16,   Bf16Quantizer);
    smoke_test!(smoke_int8,   Int8Quantizer);
    smoke_test!(smoke_nf4,    Nf4Quantizer);
    smoke_test!(smoke_sq3,    Sq3Quantizer);
    smoke_test!(smoke_sq2,    Sq2Quantizer);
    smoke_test!(smoke_binary, BinaryQuantizer);

    // ── recall@10 ≥ min_recall for each quantizer ─────────────────────────────

    macro_rules! recall_test {
        ($name:ident, $Q:ty, $min_recall:expr) => {
            #[test]
            fn $name() {
                let min_recall: f32 = $min_recall;
                let vecs = make_vecs(200, 64);
                let mut idx = QuantHnswIndex::<$Q>::new(8, 40);
                idx.add_batch(&vecs);
                let queries = &vecs[..20];
                let k = 10;
                let gt = brute_force_gt(&vecs, queries, k);
                let recall = idx.recall_at_k(queries, &gt, k, 80);
                assert!(
                    recall >= min_recall,
                    "recall@{k} = {recall:.3} < {min_recall}"
                );
            }
        };
    }

    recall_test!(recall_bf16,   Bf16Quantizer,   0.95);
    recall_test!(recall_int8,   Int8Quantizer,   0.90);
    recall_test!(recall_nf4,    Nf4Quantizer,    0.75);
    recall_test!(recall_sq3,    Sq3Quantizer,    0.70);
    recall_test!(recall_sq2,    Sq2Quantizer,    0.55);
    // Binary (1-bit) has very limited angular resolution in 64-d; realistic target.
    recall_test!(recall_binary, BinaryQuantizer, 0.20);

    #[test]
    fn parallel_build_recall_int8() {
        // n above PARALLEL_BUILD_THRESHOLD (512) exercises build_parallel.
        let vecs = make_vecs(800, 48);
        let queries = &vecs[..40];
        let k = 10;
        let gt = brute_force_gt(&vecs, queries, k);

        let mut par = QuantHnswIndex::<Int8Quantizer>::new(8, 64);
        par.add_batch(&vecs); // parallel path
        let par_recall = par.recall_at_k(queries, &gt, k, 80);

        let mut ser = QuantHnswIndex::<Int8Quantizer>::new(8, 64);
        for v in &vecs {
            ser.add(v); // serial path
        }
        let ser_recall = ser.recall_at_k(queries, &gt, k, 80);

        // Parallel quality must stay within noise of serial (the meaningful
        // property; absolute recall depends on the quantizer + this data).
        assert!(
            par_recall >= ser_recall - 0.05,
            "parallel int8 recall {par_recall:.3} regressed vs serial {ser_recall:.3}"
        );
        // Search must still work end-to-end.
        assert_eq!(par.search(&vecs[0], 1, 64).len(), 1);
    }

    // ── save / load round-trip ────────────────────────────────────────────────

    macro_rules! save_load_test {
        ($name:ident, $Q:ty) => {
            #[test]
            fn $name() {
                let mut idx = QuantHnswIndex::<$Q>::new(8, 40);
                for v in make_vecs(30, 32) {
                    idx.add(&v);
                }
                let dir = tempfile::TempDir::new().unwrap();
                let path = dir.path().join("qhnsw.bin");
                idx.save(&path).expect("save failed");
                let loaded = QuantHnswIndex::<$Q>::load(&path).expect("load failed");
                assert_eq!(loaded.len(), 30);

                let q = make_vecs(1, 32).remove(0);
                let r1 = idx.search(&q, 5, 40);
                let r2 = loaded.search(&q, 5, 40);
                let ids1: Vec<usize> = r1.iter().map(|&(id, _)| id).collect();
                let ids2: Vec<usize> = r2.iter().map(|&(id, _)| id).collect();
                assert_eq!(ids1, ids2, "search results differ after save/load");
            }
        };
    }

    save_load_test!(save_load_bf16,   Bf16Quantizer);
    save_load_test!(save_load_int8,   Int8Quantizer);
    save_load_test!(save_load_nf4,    Nf4Quantizer);
    save_load_test!(save_load_sq3,    Sq3Quantizer);
    save_load_test!(save_load_sq2,    Sq2Quantizer);
    save_load_test!(save_load_binary, BinaryQuantizer);

    // ── delete + vacuum ───────────────────────────────────────────────────────

    #[test]
    fn delete_removes_from_results_int8() {
        let mut idx = QuantHnswIndex::<Int8Quantizer>::new(8, 40);
        let vecs = make_vecs(20, 32);
        idx.add_batch(&vecs);
        idx.delete(0);
        let results = idx.search(&vecs[0], 5, 40);
        let ids: Vec<usize> = results.iter().map(|&(id, _)| id).collect();
        assert!(
            !ids.contains(&0),
            "deleted node 0 appeared in results: {:?}",
            ids
        );
    }

    #[test]
    fn vacuum_rebuilds_bf16() {
        let mut idx = QuantHnswIndex::<Bf16Quantizer>::new(8, 40);
        let vecs = make_vecs(30, 32);
        idx.add_batch(&vecs);
        for id in [2, 5, 10, 18, 24] {
            idx.delete(id);
        }
        let removed = idx.vacuum();
        assert_eq!(removed, 5, "vacuum should report 5 removed");
        assert_eq!(idx.len(), 25, "25 survivors after vacuum");
    }

    // ── filtered search ───────────────────────────────────────────────────────

    #[test]
    fn search_filtered_sq2() {
        let mut idx = QuantHnswIndex::<Sq2Quantizer>::new(8, 40);
        let vecs = make_vecs(30, 32);
        idx.add_batch(&vecs);
        let results = idx.search_filtered(&vecs[0], 5, 40, |id| id % 2 == 0);
        for &(id, _) in &results {
            assert_eq!(id % 2, 0, "odd id {id} in filtered results");
        }
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn empty_index_returns_nothing() {
        let idx = QuantHnswIndex::<Int8Quantizer>::new(4, 16);
        assert!(idx.search(&[1.0f32, 0.0, 0.0], 5, 20).is_empty());
    }

    #[test]
    fn single_element_int8() {
        let mut idx = QuantHnswIndex::<Int8Quantizer>::new(4, 16);
        idx.add(&[1.0f32, 0.0, 0.0]);
        let r = idx.search(&[1.0, 0.0, 0.0], 1, 4);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
        assert!(r[0].1 < 0.1, "dist to self = {}", r[0].1);
    }

    #[test]
    fn delete_oob_does_not_panic() {
        let mut idx = QuantHnswIndex::<Sq2Quantizer>::new(4, 16);
        idx.add(&[1.0f32, 0.0]);
        idx.delete(999); // out-of-bounds — must not panic
    }
}
