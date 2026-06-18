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

use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;
use std::collections::{BinaryHeap, HashSet};

use super::neighbor_store::{neighbors_serde, NeighborList, NodeId};

/// Newtype wrapping f32 with a total order so we can use a standard
/// `BinaryHeap` without an external crate.
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
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
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
    vectors: Vec<Vec<f32>>, // unit-norm stored vectors
    /// `neighbors[node][layer] = [node_id, ...]`; `u32` ids in inline-capable
    /// lists (see [`super::neighbor_store`]).
    #[serde(with = "neighbors_serde")]
    neighbors: Vec<Vec<NeighborList>>,
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
            neighbors: Vec::new(),
            entry_point: None,
            max_level: 0,
            deleted: Vec::new(),
        }
    }

    /// Number of vectors currently stored.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// True when the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    // ─────────────────────────── internal helpers ────────────────────────

    #[inline]
    fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
        // Stored vectors are pre-normalised; dot product == cosine similarity.
        // SimSIMD dispatches to NEON/SVE on ARM or AVX2/AVX-512 on x86 at runtime.
        let dot: f64 = <f32 as SpatialSimilarity>::dot(a, b).unwrap_or(-1.0);
        (1.0 - dot as f32).max(0.0)
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

    fn random_level(&self) -> usize {
        // Deterministic geometric-distribution level via LCG hash of node id.
        let id = self.vectors.len() as u64;
        let mut r = id
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        r ^= r >> 33;
        r = r.wrapping_mul(0xff51afd7ed558ccd);
        r ^= r >> 33;
        let frac = (r >> 11) as f64 / (1u64 << 53) as f64;
        let frac = frac.max(1e-15);
        ((-frac.ln()) * self.ml) as usize
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
        let mut visited: HashSet<usize> = HashSet::with_capacity(ef * 4);

        // cands: min-heap on dist — pop closest first.
        let mut cands: BinaryHeap<(std::cmp::Reverse<OrdF32>, usize)> = BinaryHeap::new();
        // window W: max-heap on dist — pop worst to maintain size <= ef.
        let mut window: BinaryHeap<(OrdF32, usize)> = BinaryHeap::new();

        for &ep in entry_points {
            let d = Self::cosine_dist(query, &self.vectors[ep]);
            visited.insert(ep);
            cands.push((std::cmp::Reverse(OrdF32(d)), ep));
            if !self.is_deleted(ep) && filter(ep) {
                window.push((OrdF32(d), ep));
            }
        }

        while let Some((std::cmp::Reverse(OrdF32(d_c)), c)) = cands.pop() {
            let worst = window.peek().map(|e| e.0 .0).unwrap_or(f32::INFINITY);
            if d_c > worst && window.len() >= ef {
                break;
            }

            if layer >= self.neighbors[c].len() {
                continue;
            }
            // Iterate adjacency by reference — `neighbors` and `vectors` are
            // distinct shared borrows of `self`, so no clone is needed here.
            for &nb in &self.neighbors[c][layer] {
                let nb = nb as usize;
                if !visited.insert(nb) {
                    continue;
                }
                let d_nb = Self::cosine_dist(query, &self.vectors[nb]);
                let worst2 = window.peek().map(|e| e.0 .0).unwrap_or(f32::INFINITY);
                if d_nb < worst2 || window.len() < ef {
                    cands.push((std::cmp::Reverse(OrdF32(d_nb)), nb));
                    if !self.is_deleted(nb) && filter(nb) {
                        window.push((OrdF32(d_nb), nb));
                        if window.len() > ef {
                            window.pop();
                        }
                    }
                }
            }
        }

        let mut result: Vec<(f32, usize)> =
            window.into_iter().map(|(d, id)| (d.0, id)).collect();
        result.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
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

    fn select_neighbors(candidates: &[(f32, usize)], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|&(_, id)| id).collect()
    }

    // ─────────────────────────── public API ─────────────────────────────

    /// Insert a single vector into the index (normalised internally).
    pub fn add(&mut self, vector: &[f32]) {
        let norm_vec = Self::normalize(vector);
        let node_id = self.vectors.len();
        let node_level = self.random_level();

        self.vectors.push(norm_vec.clone());
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
                    let nbrs = Self::select_neighbors(&candidates, max_m);

                    self.neighbors[node_id][lc] =
                        nbrs.iter().map(|&id| id as NodeId).collect();
                    curr_ep = candidates.into_iter().map(|(_, id)| id).collect();

                    // Add reverse links and prune if over max_m.
                    for &nb_id in &nbrs {
                        if lc < self.neighbors[nb_id].len() {
                            self.neighbors[nb_id][lc].push(node_id as NodeId);
                            if self.neighbors[nb_id][lc].len() > max_m {
                                let nb_vec = self.vectors[nb_id].clone();
                                let mut scored: Vec<(f32, NodeId)> = self.neighbors[nb_id][lc]
                                    .iter()
                                    .map(|&n| {
                                        (Self::cosine_dist(&nb_vec, &self.vectors[n as usize]), n)
                                    })
                                    .collect();
                                scored.sort_by(|a, b| {
                                    a.0.partial_cmp(&b.0)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
                                self.neighbors[nb_id][lc] =
                                    scored.into_iter().take(max_m).map(|(_, id)| id).collect();
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

    /// Insert a batch of vectors.
    ///
    /// Generic over `AsRef<[f32]>` so callers can pass borrowed row slices
    /// (e.g. `&[&[f32]]` over a contiguous buffer) without an owning copy.
    pub fn add_batch<V: AsRef<[f32]>>(&mut self, vectors: &[V]) {
        for v in vectors {
            self.add(v.as_ref());
        }
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
        if id < self.vectors.len() {
            // Backfill tombstone vec in case this index was loaded from a file
            // saved before the `deleted` field was introduced.
            if self.deleted.len() < self.vectors.len() {
                self.deleted.resize(self.vectors.len(), false);
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
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let idx: Self = bincode::deserialize(&bytes)?;
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
        let survivors: Vec<Vec<f32>> = self
            .vectors
            .iter()
            .zip(self.deleted.iter())
            .filter(|(_, &d)| !d)
            .map(|(v, _)| v.clone())
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
