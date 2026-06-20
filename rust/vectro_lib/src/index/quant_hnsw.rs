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

use super::neighbor_store::{neighbors_serde, NeighborList, NodeId};
use super::{key_dist, key_id, pack_key, shuffled_order};
use crate::quant::{
    Bf16Quantizer, BinaryQuantizer, Int8Quantizer, Nf4Quantizer, Quantizer, Sq2Quantizer,
    Sq3Quantizer,
};

/// Per-layer candidate neighbour lists for one node: `(layer, [(dist, id), …])`,
/// the read-only output of the parallel search phase handed to the serial commit.
type LayerCandidates = Vec<(usize, Vec<(f32, usize)>)>;

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
    /// `neighbors[node][layer] = [neighbor_id, ...]`; `u32` ids in
    /// inline-capable lists (see [`super::neighbor_store`]).
    #[serde(with = "neighbors_serde")]
    neighbors: Vec<Vec<NeighborList>>,
    entry_point: Option<usize>,
    max_level: usize,
    /// Soft-deletion tombstones; always aligned to `encoded.len()`.
    #[serde(default)]
    deleted: Vec<bool>,
    /// Optional centering vector (mean of the first batch's unit vectors).
    /// Auto-enabled for 1-bit (binary) quantization, where un-centered sign bits
    /// on real embeddings collapse to near-identical codes and destroy graph
    /// navigability. `None` for other quantizers / older indexes.
    #[serde(default)]
    center: Option<Vec<f32>>,
    /// Transient full-precision (centered) vectors held *only during graph
    /// construction*. The graph is built from exact f32 distances (links are
    /// integers, free at rest), then dropped via `finalize`, leaving only the
    /// quantized codes. Building a coarse graph from quantized distances
    /// collapses recall; building from f32 and searching the codes recovers it.
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

    #[inline]
    fn is_deleted(&self, id: usize) -> bool {
        self.deleted.get(id).copied().unwrap_or(false)
    }

    /// Deterministic geometric-distribution level for a given node id, seeded
    /// purely by `id` so the parallel build matches the serial path's levels.
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

    fn random_level(&self) -> usize {
        Self::level_of(self.encoded.len(), self.ml)
    }

    /// Mean-centering is enabled only for 1-bit (binary) quantization.
    #[inline]
    fn centering_enabled() -> bool {
        Q::bits_per_dim() == 1
    }

    /// Subtract the stored centering vector from an already-unit-normalised
    /// vector and re-normalise. No-op when no center is set.
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

    /// Mean of the unit-normalised input vectors (the centering vector).
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

    /// Distance from a stored node to an f32 query. During construction
    /// (`use_f32 = true`) the transient full-precision `build_vectors` give an
    /// exact distance (high-quality graph); at query time (`use_f32 = false`,
    /// or after `finalize`) the asymmetric quantized distance is used.
    #[inline]
    fn node_dist(&self, id: usize, query: &[f32], use_f32: bool) -> f32 {
        if use_f32 {
            if let Some(v) = self.build_vectors.get(id) {
                if !v.is_empty() {
                    return crate::quant::cosine_dist_unit(v, query);
                }
            }
        }
        Q::dist_to_query(&self.encoded[id], query)
    }

    /// Core beam search with an optional per-node inclusion predicate.
    ///
    /// Uses **asymmetric distance**: `Q::dist_to_query(encoded_node, f32_query)`.
    /// Deleted nodes are always excluded from the result window regardless of
    /// `filter`.  Excluded nodes are still *traversed* so connectivity is
    /// preserved.
    fn search_layer_impl<F: Fn(usize) -> bool>(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        use_f32: bool,
        filter: F,
    ) -> Vec<(f32, usize)> {
        // Reusable thread-local epoch visited set (see `super::scratch`): O(1)
        // mark/check, allocated once per thread instead of once per layer call.
        super::scratch::with_visited(self.encoded.len(), |visited| {
            // Packed-u64 heaps (see super::pack_key): native integer ordering,
            // no per-comparison f32 branch.
            let mut cands: BinaryHeap<std::cmp::Reverse<u64>> = BinaryHeap::new();
            let mut window: BinaryHeap<u64> = BinaryHeap::new();

            for &ep in entry_points {
                let d = self.node_dist(ep, query, use_f32);
                visited.visit(ep);
                cands.push(std::cmp::Reverse(pack_key(d, ep)));
                if !self.is_deleted(ep) && filter(ep) {
                    window.push(pack_key(d, ep));
                }
            }

            while let Some(std::cmp::Reverse(ck)) = cands.pop() {
                let d_c = key_dist(ck);
                let worst = window.peek().map(|&k| key_dist(k)).unwrap_or(f32::INFINITY);
                if d_c > worst && window.len() >= ef {
                    break;
                }
                let c = key_id(ck);
                if layer >= self.neighbors[c].len() {
                    continue;
                }
                // Iterate adjacency by reference — `neighbors` and `encoded` are
                // distinct shared borrows of `self`, so no clone is needed here.
                for &nb in &self.neighbors[c][layer] {
                    let nb = nb as usize;
                    if !visited.visit(nb) {
                        continue;
                    }
                    let d_nb = self.node_dist(nb, query, use_f32);
                    let worst2 = window.peek().map(|&k| key_dist(k)).unwrap_or(f32::INFINITY);
                    if d_nb < worst2 || window.len() < ef {
                        cands.push(std::cmp::Reverse(pack_key(d_nb, nb)));
                        if !self.is_deleted(nb) && filter(nb) {
                            window.push(pack_key(d_nb, nb));
                            if window.len() > ef {
                                window.pop();
                            }
                        }
                    }
                }
            }

            window
                .into_sorted_vec()
                .into_iter()
                .map(|k| (key_dist(k), key_id(k)))
                .collect()
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

    /// Distance between two stored nodes (for the selection heuristic). Uses the
    /// exact f32 build vectors during construction (`use_f32`), else the decoded
    /// codes.
    #[inline]
    fn node_to_node_dist(&self, a: usize, b: usize, use_f32: bool) -> f32 {
        if use_f32 {
            if let (Some(va), Some(vb)) = (self.build_vectors.get(a), self.build_vectors.get(b)) {
                if !va.is_empty() && !vb.is_empty() {
                    return crate::quant::cosine_dist_unit(va, vb);
                }
            }
        }
        let vb = Q::decode(&self.encoded[b], 0);
        Q::dist_to_query(&self.encoded[a], &vb)
    }

    /// Heuristic neighbour selection (HNSW Algorithm 4): keep a candidate only if
    /// it is closer to the query than to every already-selected neighbour, so the
    /// links stay diverse. `candidates` sorted ascending by distance to the query.
    fn select_heuristic(&self, candidates: &[(f32, usize)], m: usize, use_f32: bool) -> Vec<usize> {
        if candidates.len() <= m {
            return candidates.iter().map(|&(_, id)| id).collect();
        }
        let mut result: Vec<usize> = Vec::with_capacity(m);
        for &(dist_eq, e) in candidates {
            if result.len() >= m {
                break;
            }
            let diverse = result
                .iter()
                .all(|&r| self.node_to_node_dist(e, r, use_f32) >= dist_eq);
            if diverse {
                result.push(e);
            }
        }
        result
    }

    /// Set `node_id`'s forward links at `lc` (heuristic-selected from
    /// `candidates`), then add reverse links to each neighbour, re-applying the
    /// heuristic when a neighbour's list grows past `max_m`.
    fn connect(
        &mut self,
        node_id: usize,
        lc: usize,
        max_m: usize,
        candidates: &[(f32, usize)],
        use_f32: bool,
    ) {
        let nbrs = self.select_heuristic(candidates, max_m, use_f32);
        self.neighbors[node_id][lc] = nbrs.iter().map(|&id| id as NodeId).collect();
        for &nb_id in &nbrs {
            if lc >= self.neighbors[nb_id].len() {
                continue;
            }
            self.neighbors[nb_id][lc].push(node_id as NodeId);
            if self.neighbors[nb_id][lc].len() > max_m {
                let nb_query: Vec<f32> = self
                    .build_vectors
                    .get(nb_id)
                    .filter(|v| !v.is_empty())
                    .cloned()
                    .unwrap_or_else(|| Q::decode(&self.encoded[nb_id], 0));
                let mut scored: Vec<(f32, usize)> = self.neighbors[nb_id][lc]
                    .iter()
                    .map(|&n| (self.node_dist(n as usize, &nb_query, use_f32), n as usize))
                    .collect();
                scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let kept = self.select_heuristic(&scored, max_m, use_f32);
                self.neighbors[nb_id][lc] = kept.iter().map(|&id| id as NodeId).collect();
            }
        }
    }

    // ─────────────────────────── public API ─────────────────────────────────

    /// Insert a single vector (normalised, mean-centered for binary, then encoded).
    pub fn add(&mut self, vector: &[f32]) {
        let norm_vec = self.apply_center(&Self::normalize(vector));
        let node_id = self.encoded.len();
        let node_level = self.random_level();
        // Retain the f32 vector for graph construction while the buffer is still
        // aligned (not finalized); post-finalize inserts use the quantized dist.
        let building = self.build_vectors.len() == node_id;

        self.encoded.push(Q::encode(&norm_vec));
        if building {
            self.build_vectors.push(norm_vec.clone());
        }
        self.neighbors.push(vec![NeighborList::new(); node_level + 1]);
        self.deleted.push(false);

        match self.entry_point {
            None => {
                self.entry_point = Some(node_id);
                self.max_level = node_level;
            }
            Some(ep) => {
                let mut curr_ep = vec![ep];
                let max_l = self.max_level;

                for lc in (node_level + 1..=max_l).rev() {
                    let res = self.search_layer(&norm_vec, &curr_ep, 1, lc, building);
                    if !res.is_empty() {
                        curr_ep = vec![res[0].1];
                    }
                }

                for lc in (0..=node_level.min(max_l)).rev() {
                    let candidates =
                        self.search_layer(&norm_vec, &curr_ep, self.ef_construction, lc, building);
                    let max_m = if lc == 0 { self.m0 } else { self.m };
                    self.connect(node_id, lc, max_m, &candidates, building);
                    curr_ep = candidates.into_iter().map(|(_, id)| id).collect();
                }

                if node_level > max_l {
                    self.max_level = node_level;
                    self.entry_point = Some(node_id);
                }
            }
        }
    }

    /// Minimum batch size into an empty index that triggers the parallel build.
    const PARALLEL_BUILD_THRESHOLD: usize = 512;
    /// Chunk size for the parallel build, bounded to ~n/64 (clamped [64,1024]).
    fn parallel_build_chunk(n: usize) -> usize {
        (n / 512).clamp(32, 256)
    }

    /// Insert a batch of vectors. On the first batch into an empty index the
    /// centering vector (binary only) is computed, and a large batch is built
    /// in parallel; the transient f32 buffer is dropped afterward.
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
        self.finalize();
    }

    /// Drop the transient f32 build buffer, leaving only the quantized codes.
    pub fn finalize(&mut self) {
        self.build_vectors = Vec::new();
        self.build_vectors.shrink_to_fit();
    }

    /// Parallel graph construction (parallel search + serial commit), building
    /// from f32 distances and storing quantized codes. Shuffled order + bounded
    /// chunk keep recall at parity with serial regardless of input ordering.
    fn build_parallel(&mut self, vectors: &[Vec<f32>]) {
        let n = vectors.len();
        let transformed: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| self.apply_center(&Self::normalize(v)))
            .collect();
        let encoded: Vec<Q::Encoded> = transformed.par_iter().map(|t| Q::encode(t)).collect();
        let levels: Vec<usize> = (0..n).map(|id| Self::level_of(id, self.ml)).collect();

        self.encoded = encoded;
        self.build_vectors = transformed;
        self.deleted = vec![false; n];
        self.neighbors = levels
            .iter()
            .map(|&lv| vec![NeighborList::new(); lv + 1])
            .collect();

        let order = shuffled_order(n);
        self.entry_point = Some(order[0]);
        self.max_level = levels[order[0]];

        let chunk = Self::parallel_build_chunk(n);
        let mut start = 1;
        while start < n {
            let end = (start + chunk).min(n);
            let ids = &order[start..end];
            let found: Vec<LayerCandidates> = ids
                .par_iter()
                .map(|&id| self.find_candidates(id, levels[id]))
                .collect();
            for (&id, per_layer) in ids.iter().zip(found.into_iter()) {
                self.commit_node(id, levels[id], per_layer);
            }
            start = end;
        }
    }

    /// Read-only neighbour search (uses exact f32 build distance). Safe to call
    /// concurrently; commits nothing.
    fn find_candidates(&self, node_id: usize, node_level: usize) -> LayerCandidates {
        let q = &self.build_vectors[node_id];
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };
        let max_l = self.max_level;
        let mut curr_ep = vec![ep];

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

    /// Serially commit a node's links from pre-computed per-layer candidates.
    fn commit_node(&mut self, node_id: usize, node_level: usize, per_layer: LayerCandidates) {
        for (lc, candidates) in per_layer {
            let max_m = if lc == 0 { self.m0 } else { self.m };
            self.connect(node_id, lc, max_m, &candidates, true);
        }
        if node_level > self.max_level {
            self.max_level = node_level;
            self.entry_point = Some(node_id);
        }
    }

    /// Batch k-NN search over a row-major `[q, d]` flat buffer, parallel over
    /// queries (rayon). Search is `&self`/read-only.
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
