//! HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index.
//!
//! Port of the Python reference in `python/hnsw_api.py`, which implements
//! Malkov & Yashunin 2018 (arXiv:1603.09320).
//!
//! Distance metric: cosine distance (1 − cosine_similarity).
//! All stored vectors are pre-normalised to unit length so the inner product
//! equals the cosine similarity directly.
//!
//! Recall@10 parity target (PLAN.md Phase 16): ≥ 0.97.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "aarch64"))]
use simsimd::SpatialSimilarity;
use std::collections::{BinaryHeap, HashSet};
use std::sync::{PoisonError, RwLock};

use super::graph::{graph_serde, Graph};
use super::neighbor_store::{NeighborList, NodeId};
use super::{key_dist, key_id, pack_key};

/// Inlined NEON f32 dot product — `Σ a[i]*b[i]` — for the HNSW search hot loop.
/// Four independent `f32x4` accumulators break the reduction dependency chain.
///
/// # Safety
/// Requires NEON (mandated on AArch64-v8). Reads only `min(a, b)` lanes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let chunks = n / 16;
    for i in 0..chunks {
        let o = i * 16;
        acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(o)), vld1q_f32(bp.add(o)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(o + 4)), vld1q_f32(bp.add(o + 4)));
        acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(o + 8)), vld1q_f32(bp.add(o + 8)));
        acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(o + 12)), vld1q_f32(bp.add(o + 12)));
    }
    let sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut total = vaddvq_f32(sum);
    for i in chunks * 16..n {
        total += a[i] * b[i];
    }
    total
}



/// HNSW approximate nearest-neighbour index.
///
/// Build with [`HnswIndex::new`], insert vectors with [`HnswIndex::add`] /
/// [`HnswIndex::add_batch`], then query with [`HnswIndex::search`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndex {
    m: usize,
    m0: usize,            // 2 * m — max links at layer 0
    ef_construction: usize,
    ml: f64,              // level multiplier = 1 / ln(m)
    /// Unit-norm stored vectors in **one contiguous buffer**, row `i` at
    /// `vectors[i*dim .. (i+1)*dim]`. Flat layout (vs `Vec<Vec<f32>>`) removes a
    /// pointer-chase per distance eval and lets the hardware prefetcher stream.
    vectors: Vec<f32>,
    /// Vector dimensionality (0 until the first insert).
    #[serde(default)]
    dim: usize,
    /// Graph adjacency in the compact flat-layer-0 store (see [`super::graph`]).
    #[serde(with = "graph_serde")]
    neighbors: Graph,
    entry_point: Option<usize>,
    max_level: usize,
    /// Soft-deletion tombstones; index aligns with `vectors`.
    #[serde(default)]
    deleted: Vec<bool>,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    ///
    /// # Arguments
    /// * `m`               — max bidirectional links per node in layers ≥ 1 (layer 0 uses `2*m`).
    /// * `ef_construction` — beam width used while building (≥ m, larger → better recall, slower build).
    pub fn new(m: usize, ef_construction: usize) -> Self {
        assert!(m >= 2, "m must be >= 2");
        assert!(ef_construction >= m, "ef_construction must be >= m");
        let ml = 1.0 / (m as f64).ln();
        Self {
            m,
            m0: 2 * m,
            ef_construction,
            ml,
            vectors: Vec::new(),
            dim: 0,
            neighbors: Graph::new(2 * m),
            entry_point: None,
            max_level: 0,
            deleted: Vec::new(),
        }
    }

    /// Number of vectors currently stored.
    pub fn len(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.vectors.len() / self.dim
        }
    }

    /// True when the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Borrow stored vector `id` from the flat buffer (no allocation/indirection).
    #[inline]
    fn vec(&self, id: usize) -> &[f32] {
        let base = id * self.dim;
        &self.vectors[base..base + self.dim]
    }

    // ─────────────────────────── internal helpers ────────────────────────

    #[inline]
    fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
        // Stored vectors are pre-normalised; dot product == cosine similarity.
        // On aarch64 use a directly-inlined NEON f32 dot — at small dims the
        // SimSIMD path's per-call overhead (runtime dispatch, f64 accumulator,
        // `Option` unwrap) is a large fraction of the tiny compute. Elsewhere
        // SimSIMD's runtime dispatch is the portable best.
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandated on AArch64-v8; the helper reads in-bounds lanes.
        let dot = unsafe { dot_f32_neon(a, b) };
        #[cfg(not(target_arch = "aarch64"))]
        let dot = <f32 as SpatialSimilarity>::dot(a, b).unwrap_or(-1.0) as f32;
        (1.0 - dot).max(0.0)
    }

    /// Software-prefetch a stored vector into L1, hiding the ~100 ns DRAM miss
    /// that dominates beam search once the vector buffer (473 MB at 1M × d=100)
    /// dwarfs cache. A pure hint — no correctness or aliasing effect.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn prefetch_vec(&self, id: usize) {
        let p = self.vectors.as_ptr().add(id * self.dim);
        core::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p, options(nostack, preserves_flags, readonly));
    }

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
    /// purely by `id` so the parallel build assigns the same levels as serial.
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
        Self::level_of(self.len(), self.ml)
    }

    /// Core beam search with an optional per-node inclusion filter.
    ///
    /// `filter(id)` controls whether a node may appear in the result window.
    /// Deleted nodes (via [`HnswIndex::delete`]) are always excluded regardless
    /// of `filter`.  Excluded nodes are still _traversed_ so graph connectivity
    /// is preserved for non-excluded neighbours.
    ///
    /// Returns up to `ef` nearest eligible nodes as `(cosine_dist, node_id)` sorted ascending.
    fn search_layer_impl<F: Fn(usize) -> bool>(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        filter: F,
    ) -> Vec<(f32, usize)> {
        // Reusable thread-local epoch visited set (see `super::scratch`): O(1)
        // mark/check, allocated once per thread instead of once per layer call.
        super::scratch::with_visited(self.len(), |visited| {
            // Packed-u64 heaps (see pack_key): native integer ordering, no
            // per-comparison f32 branch. cands = min-heap on dist (Reverse),
            // window = max-heap on dist (pop worst to keep size <= ef).
            let mut cands: BinaryHeap<std::cmp::Reverse<u64>> = BinaryHeap::new();
            let mut window: BinaryHeap<u64> = BinaryHeap::new();

            for &ep in entry_points {
                let d = Self::cosine_dist(query, self.vec(ep));
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

                if layer >= self.neighbors.num_layers(c) {
                    continue;
                }
                // Iterate the flat layer-0 slice (or upper-layer list) by ref —
                // `neighbors` and `vectors` are distinct shared borrows of `self`.
                let nbrs = self.neighbors.neighbors(c, layer);
                // Kick off all of this node's vector loads up front so the DRAM
                // latency overlaps the distance computations below (the loads are
                // independent random reads; issuing them together hides the miss).
                #[cfg(target_arch = "aarch64")]
                for &nb in nbrs {
                    // SAFETY: ids are in-bounds; prefetch is a no-effect hint.
                    unsafe { self.prefetch_vec(nb as usize) };
                }
                for &nb in nbrs {
                    let nb = nb as usize;
                    if !visited.visit(nb) {
                        continue;
                    }
                    let d_nb = Self::cosine_dist(query, self.vec(nb));
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

            // window is a max-heap by packed key; into_sorted_vec yields ascending.
            window
                .into_sorted_vec()
                .into_iter()
                .map(|k| (key_dist(k), key_id(k)))
                .collect()
        })
    }

    /// Beam search on a single layer (no filter, deletion-aware).
    ///
    /// Returns up to `ef` nearest nodes as `(cosine_dist, node_id)` sorted ascending.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<(f32, usize)> {
        self.search_layer_impl(query, entry_points, ef, layer, |_| true)
    }

    /// Heuristic neighbour selection (Malkov & Yashunin 2018, Algorithm 4 — the
    /// `getNeighborsByHeuristic2` of hnswlib/FAISS).
    ///
    /// `candidates` must be sorted ascending by distance to the query node.
    /// A candidate `e` is kept only if it is closer to the query than to every
    /// already-selected neighbour `r` (`dist(e, r) >= dist(e, q)`). This keeps
    /// the chosen links *diverse* (pointing in different directions) instead of
    /// clustered in the single nearest direction — which is what makes the graph
    /// navigable at high recall (R@0.99). Naive top-m maxes out around R@0.98.
    fn select_heuristic(&self, candidates: &[(f32, usize)], m: usize) -> Vec<usize> {
        if candidates.len() <= m {
            return candidates.iter().map(|&(_, id)| id).collect();
        }
        let mut result: Vec<usize> = Vec::with_capacity(m);
        for &(dist_eq, e) in candidates {
            if result.len() >= m {
                break;
            }
            let e_vec = self.vec(e);
            let diverse = result
                .iter()
                .all(|&r| Self::cosine_dist(e_vec, self.vec(r)) >= dist_eq);
            if diverse {
                result.push(e);
            }
        }
        result
    }

    /// Set `node_id`'s forward links at `lc` (heuristic-selected from
    /// `candidates`), then add reverse links to each neighbour, re-applying the
    /// heuristic when a neighbour's list grows past `max_m`.
    fn connect(&mut self, node_id: usize, lc: usize, max_m: usize, candidates: &[(f32, usize)]) {
        let nbrs = self.select_heuristic(candidates, max_m);
        let fwd: Vec<NodeId> = nbrs.iter().map(|&id| id as NodeId).collect();
        self.neighbors.set(node_id, lc, &fwd);

        for &nb_id in &nbrs {
            if lc >= self.neighbors.num_layers(nb_id) {
                continue;
            }
            self.neighbors.push(nb_id, lc, node_id as NodeId);
            if self.neighbors.len_at(nb_id, lc) > max_m {
                let nb_vec = self.vec(nb_id).to_vec();
                let mut scored: Vec<(f32, usize)> = self
                    .neighbors
                    .neighbors(nb_id, lc)
                    .iter()
                    .map(|&n| (Self::cosine_dist(&nb_vec, self.vec(n as usize)), n as usize))
                    .collect();
                scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let kept: Vec<NodeId> =
                    self.select_heuristic(&scored, max_m).iter().map(|&id| id as NodeId).collect();
                self.neighbors.set(nb_id, lc, &kept);
            }
        }
    }

    // ─────────────────────────── public API ─────────────────────────────

    /// Insert a single vector into the index (normalised internally).
    pub fn add(&mut self, vector: &[f32]) {
        let norm_vec = Self::normalize(vector);
        let node_id = self.len();
        let node_level = self.random_level();

        if self.dim == 0 {
            self.dim = norm_vec.len();
        }
        self.vectors.extend_from_slice(&norm_vec);
        self.neighbors.add_node(node_level);
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
                    let res = self.search_layer(&norm_vec, &curr_ep, 1, lc);
                    if !res.is_empty() {
                        curr_ep = vec![res[0].1];
                    }
                }

                // ef_construction-width search from min(node_level, max_l) → 0.
                for lc in (0..=node_level.min(max_l)).rev() {
                    let candidates =
                        self.search_layer(&norm_vec, &curr_ep, self.ef_construction, lc);
                    let max_m = if lc == 0 { self.m0 } else { self.m };
                    self.connect(node_id, lc, max_m, &candidates);
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

    /// Insert a batch of vectors.
    ///
    /// Generic over `AsRef<[f32]>` so callers can pass borrowed row slices
    /// (e.g. `&[&[f32]]` over a contiguous buffer) without an owning copy. A
    /// large first batch into an empty index is built with the concurrent
    /// insertion path (see [`build_concurrent`]).
    pub fn add_batch<V: AsRef<[f32]>>(&mut self, vectors: &[V]) {
        if self.vectors.is_empty() && vectors.len() >= Self::PARALLEL_BUILD_THRESHOLD {
            self.build_concurrent(vectors);
        } else {
            for v in vectors {
                self.add(v.as_ref());
            }
        }
    }

    // ───────────────────── concurrent-insertion build ───────────────────
    //
    // The chunked `build_parallel` freezes the graph at each chunk boundary, so
    // chunk-mates can't link to each other — a residual ≈ chunk/n that caps recall
    // around 0.997. This path instead inserts every node against the **live**
    // graph behind per-node locks (hnswlib-style): full visibility recovers
    // serial-quality recall (R@10 → 1.000) at parallel speed. Wiring is
    // schedule-dependent (not bit-reproducible) but node *levels* stay seeded.

    /// Number of nodes to seed serially before the concurrent insert phase.
    /// Sized to dominate the early-graph region (where concurrent inserts would
    /// otherwise see a sparse graph) while staying a small fraction of the build.
    fn concurrent_seed(n: usize) -> usize {
        (n / 20).clamp(256, 4096).min(n)
    }

    /// Concurrent graph construction: insert all nodes in parallel against a
    /// live, per-node-locked adjacency. No thread ever holds two node locks at
    /// once (forward links touch only this node's own list; each reverse link is
    /// locked, mutated, released independently) — so the build is deadlock-free.
    fn build_concurrent<V: AsRef<[f32]>>(&mut self, vectors: &[V]) {
        let n = vectors.len();
        self.dim = vectors.first().map(|v| v.as_ref().len()).unwrap_or(0);
        let mut flat = Vec::with_capacity(n * self.dim);
        for v in vectors {
            flat.extend_from_slice(&Self::normalize(v.as_ref()));
        }
        self.vectors = flat;
        self.deleted = vec![false; n];

        let levels: Vec<usize> = (0..n).map(|id| Self::level_of(id, self.ml)).collect();
        // Per-node, per-layer locked adjacency lists.
        let graph: Vec<Vec<RwLock<NeighborList>>> = levels
            .iter()
            .map(|&lv| (0..=lv).map(|_| RwLock::new(NeighborList::new())).collect())
            .collect();
        // (entry_point, max_level) — written only when a node raises the top level.
        let ep_state = RwLock::new((0usize, levels[0]));

        // Serial seed: build a high-quality core first. Without it the first node
        // of each thread's range searches a near-empty graph (every other thread
        // has barely started) and gets poor links — the residual that otherwise
        // caps recall below 1.0. Once `seed` nodes are well-connected, every
        // concurrent insert searches a fully-formed graph, recovering serial
        // recall (R@10 → 1.0) at ~parallel speed.
        let seed = Self::concurrent_seed(n);
        for (id, &lv) in levels.iter().enumerate().take(seed).skip(1) {
            self.insert_concurrent(id, lv, &graph, &ep_state);
        }
        (seed..n)
            .into_par_iter()
            .for_each(|id| self.insert_concurrent(id, levels[id], &graph, &ep_state));

        // Drain the locks into the compact flat-layer-0 store.
        let layered: Vec<Vec<NeighborList>> = graph
            .into_iter()
            .map(|node| {
                node.into_iter()
                    .map(|lock| lock.into_inner().unwrap_or_else(PoisonError::into_inner))
                    .collect()
            })
            .collect();
        self.neighbors = Graph::from_layered(layered, self.m0);
        let (ep, max_level) = ep_state.into_inner().unwrap_or_else(PoisonError::into_inner);
        self.entry_point = Some(ep);
        self.max_level = max_level;
    }

    /// Insert one node against the live locked graph (mirrors [`add`], but every
    /// graph read/write goes through the per-node `RwLock`s in `graph`).
    fn insert_concurrent(
        &self,
        node_id: usize,
        node_level: usize,
        graph: &[Vec<RwLock<NeighborList>>],
        ep_state: &RwLock<(usize, usize)>,
    ) {
        let q = self.vec(node_id);
        let (ep, max_l) = *ep_state.read().unwrap_or_else(PoisonError::into_inner);
        let mut curr_ep = vec![ep];

        // Greedy descent from the (snapshot) top down to node_level + 1.
        for lc in (node_level + 1..=max_l).rev() {
            let res = self.search_layer_locked(q, &curr_ep, 1, lc, graph);
            if !res.is_empty() {
                curr_ep = vec![res[0].1];
            }
        }

        // ef_construction beam search + connect on each owned layer.
        for lc in (0..=node_level.min(max_l)).rev() {
            let candidates = self.search_layer_locked(q, &curr_ep, self.ef_construction, lc, graph);
            let max_m = if lc == 0 { self.m0 } else { self.m };
            self.connect_locked(node_id, lc, max_m, &candidates, graph);
            curr_ep = candidates.into_iter().map(|(_, id)| id).collect();
        }

        // Raise the entry point if this node introduced a new top level.
        if node_level > max_l {
            let mut st = ep_state.write().unwrap_or_else(PoisonError::into_inner);
            if node_level > st.1 {
                *st = (node_id, node_level);
            }
        }
    }

    /// Beam search on one layer of the live locked graph. Each adjacency list is
    /// snapshotted under a read lock and released *before* the distance evals, so
    /// no lock is ever held across compute.
    fn search_layer_locked(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        graph: &[Vec<RwLock<NeighborList>>],
    ) -> Vec<(f32, usize)> {
        super::scratch::with_visited(self.len(), |visited| {
            let mut cands: BinaryHeap<std::cmp::Reverse<u64>> = BinaryHeap::new();
            let mut window: BinaryHeap<u64> = BinaryHeap::new();

            for &ep in entry_points {
                let d = Self::cosine_dist(query, self.vec(ep));
                visited.visit(ep);
                cands.push(std::cmp::Reverse(pack_key(d, ep)));
                window.push(pack_key(d, ep));
            }

            while let Some(std::cmp::Reverse(ck)) = cands.pop() {
                let d_c = key_dist(ck);
                let worst = window.peek().map(|&k| key_dist(k)).unwrap_or(f32::INFINITY);
                if d_c > worst && window.len() >= ef {
                    break;
                }
                let c = key_id(ck);
                if layer >= graph[c].len() {
                    continue;
                }
                let nbrs: NeighborList = graph[c][layer]
                    .read()
                    .unwrap_or_else(PoisonError::into_inner)
                    .clone();
                for &nb in &nbrs {
                    let nb = nb as usize;
                    if !visited.visit(nb) {
                        continue;
                    }
                    let d_nb = Self::cosine_dist(query, self.vec(nb));
                    let worst2 = window.peek().map(|&k| key_dist(k)).unwrap_or(f32::INFINITY);
                    if d_nb < worst2 || window.len() < ef {
                        cands.push(std::cmp::Reverse(pack_key(d_nb, nb)));
                        window.push(pack_key(d_nb, nb));
                        if window.len() > ef {
                            window.pop();
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

    /// Locked equivalent of [`connect`]: heuristic-select forward links (write
    /// this node's own list), then add a reverse link to each neighbour,
    /// re-applying the heuristic when its list overflows `max_m`. Each neighbour
    /// lock is taken and dropped in turn — never two at once.
    fn connect_locked(
        &self,
        node_id: usize,
        lc: usize,
        max_m: usize,
        candidates: &[(f32, usize)],
        graph: &[Vec<RwLock<NeighborList>>],
    ) {
        // Exclude self. Under concurrent insertion a node can become reachable
        // (via a reverse link from a concurrent inserter) before its own forward
        // links are set, so the beam search can return `node_id` itself — which
        // must never be a neighbour (a self-loop wastes a slot and corrupts the
        // graph). The serial `add` path can't hit this; the locked path can.
        let nbrs: Vec<usize> = self
            .select_heuristic(candidates, max_m)
            .into_iter()
            .filter(|&id| id != node_id)
            .collect();
        {
            let mut fwd = graph[node_id][lc]
                .write()
                .unwrap_or_else(PoisonError::into_inner);
            *fwd = nbrs.iter().map(|&id| id as NodeId).collect();
        }

        for &nb_id in &nbrs {
            if lc >= graph[nb_id].len() {
                continue;
            }
            let mut lock = graph[nb_id][lc]
                .write()
                .unwrap_or_else(PoisonError::into_inner);
            lock.push(node_id as NodeId);
            if lock.len() > max_m {
                let nb_vec = self.vec(nb_id);
                let mut scored: Vec<(f32, usize)> = lock
                    .iter()
                    .map(|&nbr| (Self::cosine_dist(nb_vec, self.vec(nbr as usize)), nbr as usize))
                    .collect();
                scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let kept = self.select_heuristic(&scored, max_m);
                *lock = kept.iter().map(|&id| id as NodeId).collect();
            }
        }
    }

    /// Batch k-NN search over a row-major `[q, d]` flat query buffer, run in
    /// parallel across queries (rayon). Search is `&self`/read-only.
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
    /// Returns `Vec<(node_index, cosine_distance)>` sorted ascending by distance.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        let ep = match self.entry_point {
            None => return vec![],
            Some(ep) => ep,
        };
        let ef = ef.max(k);
        let q = Self::normalize(query);
        let mut curr_ep = vec![ep];

        // Greedy descent to layer 1.
        for lc in (1..=self.max_level).rev() {
            let res = self.search_layer(&q, &curr_ep, 1, lc);
            if !res.is_empty() {
                curr_ep = vec![res[0].1];
            }
        }

        // Full beam search at layer 0.
        let res = self.search_layer(&q, &curr_ep, ef, 0);
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
    ///
    /// The vector is excluded from all future search results but stays in the
    /// graph structure to maintain connectivity for its non-deleted neighbours.
    pub fn delete(&mut self, id: usize) {
        if id < self.len() {
            // Backfill tombstone vec in case this index was loaded from a file
            // saved before the `deleted` field was introduced.
            if self.deleted.len() < self.len() {
                self.deleted.resize(self.len(), false);
            }
            self.deleted[id] = true;
        }
    }

    /// Approximate k-nearest-neighbour search with a predicate filter.
    ///
    /// Only nodes where `predicate(id) == true` are eligible for the result
    /// set. Filtered-out nodes are still traversed to find non-filtered
    /// neighbours further in the graph.
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
        let q = Self::normalize(query);
        let mut curr_ep = vec![ep];

        // Greedy descent through upper layers without filter (structural path-finding).
        for lc in (1..=self.max_level).rev() {
            let res = self.search_layer(&q, &curr_ep, 1, lc);
            if !res.is_empty() {
                curr_ep = vec![res[0].1];
            }
        }

        // Full ef-width beam search at layer 0 applying the user predicate.
        let res = self.search_layer_impl(&q, &curr_ep, ef, 0, predicate);
        res.into_iter().take(k).map(|(d, id)| (id, d)).collect()
    }

    /// Persist the index to a file using bincode serialization.
    ///
    /// Restore with [`HnswIndex::load`].
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load an index previously saved with [`HnswIndex::save`].
    ///
    /// Streams from a buffered reader rather than slurping the whole file into a
    /// `Vec<u8>` first — at 1M× d=100 the index is ~650 MB, so the buffer would
    /// otherwise double peak load memory.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let idx: Self = bincode::deserialize_from(reader)?;
        Ok(idx)
    }

    /// Compact the index by permanently removing all soft-deleted nodes and
    /// rebuilding the HNSW graph from scratch.
    ///
    /// This is more expensive than [`delete`] (O(n log n) insert cost) but
    /// restores full graph quality and reclaims memory.  Returns the number
    /// of nodes removed.  If no nodes are deleted this is a cheap no-op.
    pub fn vacuum(&mut self) -> usize {
        let deleted_count = self.deleted.iter().filter(|&&d| d).count();
        if deleted_count == 0 {
            return 0;
        }

        // Collect surviving original vectors (already unit-normalised by `add`).
        let survivors: Vec<Vec<f32>> = (0..self.len())
            .filter(|&i| !self.deleted.get(i).copied().unwrap_or(false))
            .map(|i| self.vec(i).to_vec())
            .collect();

        // Rebuild with the same construction parameters.
        let mut new_idx = HnswIndex::new(self.m, self.ef_construction);
        new_idx.add_batch(&survivors);
        *self = new_idx;
        deleted_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_key_orders_by_distance_then_id() {
        // Non-negative f32 bits are monotonic, so packed keys must sort by
        // distance first, then id — the property the beam heaps rely on.
        let a = pack_key(0.0, 7);
        let b = pack_key(0.5, 3);
        let c = pack_key(0.5, 9);
        let d = pack_key(2.0, 0);
        assert!(a < b && b < c && c < d, "distance/id ordering broken");
        // round-trip
        assert_eq!(key_dist(b), 0.5);
        assert_eq!(key_id(c), 9);
        // INFINITY sentinel must be the largest (it's the "no worst yet" value).
        assert!(pack_key(f32::INFINITY, 0) > d);
    }

    fn make_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..d).map(|j| ((i * d + j) as f32 * 0.017 + 0.1).sin()).collect())
            .collect()
    }

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

    #[test]
    fn build_and_search_smoke() {
        // Use generous parameters so exact-self search is always found.
        let mut idx = HnswIndex::new(8, 50);
        let vecs = make_vecs(50, 16);
        idx.add_batch(&vecs);
        assert_eq!(idx.len(), 50);

        // Query every stored vector against itself; it must be returned as the
        // nearest neighbour (distance ≈ 0).
        for (i, v) in vecs.iter().enumerate() {
            let results = idx.search(v, 1, 50);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, i, "nearest to vec[{i}] should be vec[{i}]");
            assert!(results[0].1 < 1e-4, "dist to self[{i}] = {}", results[0].1);
        }
    }

    #[test]
    fn search_empty_index_returns_empty() {
        let idx = HnswIndex::new(4, 16);
        assert!(idx.search(&[1.0f32, 0.0, 0.0], 5, 20).is_empty());
    }

    #[test]
    fn single_element_exact_match() {
        let mut idx = HnswIndex::new(4, 16);
        idx.add(&[1.0f32, 0.0, 0.0]);
        let r = idx.search(&[1.0, 0.0, 0.0], 1, 4);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
        assert!(r[0].1 < 1e-5, "dist to self = {}", r[0].1);
    }

    #[test]
    fn opposite_vectors_are_far() {
        let mut idx = HnswIndex::new(4, 16);
        idx.add(&[1.0f32, 0.0, 0.0]);
        idx.add(&[-1.0f32, 0.0, 0.0]);
        let r = idx.search(&[1.0, 0.0, 0.0], 2, 10);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].0, 0, "vec[0] should be closest to [1,0,0]");
        assert!(r[0].1 < 1e-4, "dist to self = {}", r[0].1);
        assert!(r[1].1 > 1.9, "dist to opposite = {}", r[1].1);
    }

    #[test]
    fn recall_at_k_reasonable() {
        let vecs = make_vecs(200, 32);
        let mut idx = HnswIndex::new(8, 40);
        idx.add_batch(&vecs);
        let queries = &vecs[..20];
        let k = 5;
        let gt = brute_force_gt(&vecs, queries, k);
        let recall = idx.recall_at_k(queries, &gt, k, 60);
        assert!(recall >= 0.80, "recall@{k} = {recall:.3} < 0.80");
    }

    #[test]
    fn concurrent_build_high_recall() {
        // A batch over PARALLEL_BUILD_THRESHOLD routes through `build_concurrent`
        // (serial seed + parallel live-graph insertion). It must reach near-exact
        // recall — the whole point of the concurrent path over the chunked one.
        let n = 1500;
        assert!(n > HnswIndex::PARALLEL_BUILD_THRESHOLD);
        let vecs = make_vecs(n, 32);
        let mut idx = HnswIndex::new(16, 200);
        idx.add_batch(&vecs);
        assert_eq!(idx.len(), n);

        let queries = &vecs[..50];
        let k = 10;
        let gt = brute_force_gt(&vecs, queries, k);
        // The concurrent build is schedule-dependent, so assert a robust floor
        // (real data reaches ~0.998; this synthetic set fluctuates a little run
        // to run). The point is "high recall", not an exact tie.
        let recall = idx.recall_at_k(queries, &gt, k, 200);
        assert!(recall >= 0.93, "concurrent-build recall@{k} = {recall:.4} < 0.93");
    }

    #[test]
    fn concurrent_build_graph_is_valid() {
        // Every node's adjacency must reference in-bounds ids and contain no
        // self-loops — a structural invariant the locked build must keep. The
        // self-loop race surfaces only on some thread schedules, so build several
        // times to make a regression reliably visible.
        let n = 800;
        let vecs = make_vecs(n, 16);
        for attempt in 0..8 {
            let mut idx = HnswIndex::new(8, 64);
            idx.add_batch(&vecs);
            for node in 0..idx.len() {
                for layer in 0..idx.neighbors.num_layers(node) {
                    for &nb in idx.neighbors.neighbors(node, layer) {
                        let nb = nb as usize;
                        assert!(nb < n, "neighbour id {nb} out of bounds (n={n})");
                        assert_ne!(nb, node, "self-loop at node {node} (attempt {attempt})");
                    }
                }
            }
        }
    }

    #[test]
    fn k_capped_at_index_size() {
        let mut idx = HnswIndex::new(4, 16);
        for v in make_vecs(3, 8) {
            idx.add(&v);
        }
        let r = idx.search(&[0.1f32; 8], 10, 20);
        assert!(r.len() <= 3, "got {} results for 3-element index", r.len());
    }

    #[test]
    fn save_load_roundtrip() {
        let mut idx = HnswIndex::new(8, 40);
        for v in make_vecs(30, 16) {
            idx.add(&v);
        }
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("index.bin");
        idx.save(&path).expect("save failed");
        let loaded = HnswIndex::load(&path).expect("load failed");
        assert_eq!(loaded.len(), idx.len());

        // Verify search results are identical after round-trip.
        let q = make_vecs(1, 16).remove(0);
        let r1 = idx.search(&q, 5, 40);
        let r2 = loaded.search(&q, 5, 40);
        let ids1: Vec<usize> = r1.iter().map(|&(id, _)| id).collect();
        let ids2: Vec<usize> = r2.iter().map(|&(id, _)| id).collect();
        assert_eq!(ids1, ids2, "search results differ after save/load");
    }

    #[test]
    fn delete_removes_from_results() {
        let mut idx = HnswIndex::new(8, 40);
        let vecs = make_vecs(20, 16);
        for v in &vecs {
            idx.add(v);
        }
        // Deleting a vector must exclude it from its own self-query.
        idx.delete(0);
        let results = idx.search(&vecs[0], 5, 40);
        let ids: Vec<usize> = results.iter().map(|&(id, _)| id).collect();
        assert!(!ids.contains(&0), "deleted node 0 appeared in search results: {:?}", ids);
    }

    #[test]
    fn search_filtered_respects_predicate() {
        let mut idx = HnswIndex::new(8, 40);
        let vecs = make_vecs(30, 16);
        for v in &vecs {
            idx.add(v);
        }
        // Allow only even-indexed nodes.
        let results = idx.search_filtered(&vecs[0], 5, 40, |id| id % 2 == 0);
        for &(id, _) in &results {
            assert_eq!(id % 2, 0, "odd id {id} appeared in filtered results");
        }
        assert!(!results.is_empty(), "filtered search returned no results");
    }

    #[test]
    fn delete_does_not_panic_on_out_of_bounds() {
        let mut idx = HnswIndex::new(4, 16);
        idx.add(&[1.0f32, 0.0]);
        // Deleting out-of-bounds id must not panic.
        idx.delete(999);
    }

    #[test]
    fn vacuum_removes_deleted_and_rebuilds() {
        let mut idx = HnswIndex::new(8, 40);
        let vecs = make_vecs(30, 16);
        idx.add_batch(&vecs);
        assert_eq!(idx.len(), 30);

        // Soft-delete 5 nodes.
        for id in [2, 5, 10, 18, 24] {
            idx.delete(id);
        }
        let removed = idx.vacuum();
        assert_eq!(removed, 5, "vacuum should report 5 removed nodes");
        assert_eq!(idx.len(), 25, "25 survivors after vacuum");
        // No tombstones in the rebuilt index.
        assert!(!idx.deleted.iter().any(|&d| d));

        // A second vacuum call on a clean index is a cheap no-op.
        assert_eq!(idx.vacuum(), 0);

        // The rebuilt index must still return useful results.
        let q = vecs[0].clone();
        let res = idx.search(&q, 1, 40);
        assert!(!res.is_empty());
    }
}
