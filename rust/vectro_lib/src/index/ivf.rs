//! IVF-Flat (Inverted File Index) — coarse cluster search over raw float32 vectors.
//!
//! Algorithm summary
//! -----------------
//! **Build:** train `n_lists` centroids via Lloyd's k-means (same implementation
//! used by PQ, seeded deterministically).  Each inserted vector is assigned to
//! its nearest centroid and stored in that centroid's posting list.
//!
//! **Search:** for a query `q`, score all `n_lists` centroids, keep the closest
//! `n_probe` ones, scan only those posting lists by exact cosine distance, and
//! return the global top-k.
//!
//! Complexity
//! ----------
//! - Build:   O(n · d · n_lists · max_iter)
//! - Add:     O(n_lists · d)  per vector
//! - Search:  O(n_lists · d + n_probe · cluster_size · d)
//!
//! When the dataset is clustered (`n_probe` << `n_lists`) this is dramatically
//! faster than brute-force and uses far less RAM than HNSW for large n.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Internal k-means helpers (identical LCG seed + Lloyd's used by pq.rs)
// ---------------------------------------------------------------------------

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    // Route through the shared SIMD kernel (NEON / AVX2) rather than a scalar
    // reduction — k-means++ init and Lloyd assignment call this over the whole
    // dataset per iteration.
    crate::index::simd::l2_sq(a, b)
}

/// k-means++ centroid initialisation (D²-weighted sampling, LCG RNG).
fn kmeans_pp_init(data: &[&[f32]], k: usize, d: usize, seed: u64) -> Vec<f32> {
    let n = data.len();
    debug_assert!(n >= k);

    let mut state = seed.wrapping_add(1_442_695_040_888_963_407);
    let mut lcg = || -> f64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut cents = vec![0.0f32; k * d];
    let first = (lcg() * n as f64) as usize % n;
    cents[..d].copy_from_slice(data[first]);

    let mut min_d2 = vec![f32::INFINITY; n];

    for ki in 1..k {
        let new_cent = &cents[(ki - 1) * d..ki * d];
        for (i, v) in data.iter().enumerate() {
            let dist = l2_sq(v, new_cent);
            if dist < min_d2[i] {
                min_d2[i] = dist;
            }
        }
        let total: f64 = min_d2.iter().map(|&x| x as f64).sum();
        let chosen = if total == 0.0 {
            ki % n
        } else {
            let r = lcg() * total;
            let mut acc = 0.0f64;
            let mut idx = n - 1;
            for (i, &d2) in min_d2.iter().enumerate() {
                acc += d2 as f64;
                if acc >= r {
                    idx = i;
                    break;
                }
            }
            idx
        };
        cents[ki * d..(ki + 1) * d].copy_from_slice(data[chosen]);
    }
    cents
}

/// Lloyd's k-means; returns flat [k * d] centroid array.
fn kmeans_lloyd(data: &[&[f32]], k: usize, d: usize, max_iter: usize, seed: u64) -> Vec<f32> {
    let n = data.len();
    let mut cents = kmeans_pp_init(data, k, d, seed);
    let mut assignments = vec![0usize; n];

    // Flatten the data once into a contiguous [n, d] buffer for the GEMM
    // assignment (centroids are already flat [k, d]).
    let data_flat: Vec<f32> = data.iter().flat_map(|v| v.iter().copied()).collect();

    for _ in 0..max_iter {
        // Assignment step — one GEMM + parallel argmax (squared-L2 nearest).
        let new_asgn =
            super::kmeans::assign_nearest(&data_flat, &cents, n, k, d, super::kmeans::Metric::L2);

        if new_asgn == assignments {
            break;
        }
        assignments = new_asgn;

        // Update step.
        let mut sums = vec![0.0f32; k * d];
        let mut counts = vec![0usize; k];
        for (v, &ki) in data.iter().zip(assignments.iter()) {
            let start = ki * d;
            for (s, &x) in sums[start..start + d].iter_mut().zip(v.iter()) {
                *s += x;
            }
            counts[ki] += 1;
        }
        for (ki, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let inv = 1.0 / count as f32;
            let start = ki * d;
            // Centroid = mean of assigned points. (Earlier revisions had a
            // redundant reset+accumulate pass here that this single write
            // already subsumes.)
            for (c, s) in cents[start..start + d].iter_mut().zip(sums[start..start + d].iter()) {
                *c = s * inv;
            }
        }
    }
    cents
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// IVF-Flat index.
///
/// Stores the full `f32` vectors inside hashed posting lists partitioned by
/// coarse centroid assignment.  Best suited for datasets of up to ~50 M vectors
/// where memory is not the primary constraint.  For larger scales or memory
/// pressure, use [`crate::index::ivf_pq::IvfPqIndex`] which stores PQ codes
/// instead of raw vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "IvfIndexWire", into = "IvfIndexWire")]
pub struct IvfIndex {
    /// Number of Voronoi cells (coarse centroids).
    pub n_lists: usize,
    /// Default probe width: how many posting lists to scan on `search`.
    pub n_probe: usize,
    /// Flat array of coarse centroids, shape [n_lists * dim].
    centroids: Vec<f32>,
    /// Per-list global vector IDs.
    posting_lists: Vec<Vec<usize>>,
    /// All unit-norm vectors in **one contiguous buffer**, row `i` at
    /// `store[i*dim .. (i+1)*dim]`. Flat layout (was `Vec<Vec<f32>>`): removes a
    /// pointer-chase + per-vector `Vec` header per candidate and lets the search
    /// loop software-prefetch the next vector while scoring the current one.
    store: Vec<f32>,
    /// Soft-deletion tombstones, one per stored vector.
    #[serde(default)]
    deleted: Vec<bool>,
    /// Vector dimension (set at training time).
    dim: usize,
    /// Whether the index has been trained.
    trained: bool,
}

/// On-disk shadow of [`IvfIndex`] that preserves the original nested-`Vec`
/// store layout as the wire format. `IvfIndex` serializes *into* this and
/// deserializes *from* it (see the `#[serde(from/into)]` above), so the bincode
/// bytes are byte-identical to pre-flat-store releases — old `.bin` files load
/// unchanged, with no schema-version bump.
#[derive(Serialize, Deserialize)]
struct IvfIndexWire {
    n_lists: usize,
    n_probe: usize,
    centroids: Vec<f32>,
    posting_lists: Vec<Vec<usize>>,
    store: Vec<Vec<f32>>,
    #[serde(default)]
    deleted: Vec<bool>,
    dim: usize,
    trained: bool,
}

impl From<IvfIndexWire> for IvfIndex {
    fn from(w: IvfIndexWire) -> Self {
        // Flatten the nested per-vector rows into one contiguous buffer.
        let mut store = Vec::with_capacity(w.store.len() * w.dim);
        for row in &w.store {
            store.extend_from_slice(row);
        }
        Self {
            n_lists: w.n_lists,
            n_probe: w.n_probe,
            centroids: w.centroids,
            posting_lists: w.posting_lists,
            store,
            deleted: w.deleted,
            dim: w.dim,
            trained: w.trained,
        }
    }
}

impl From<IvfIndex> for IvfIndexWire {
    fn from(idx: IvfIndex) -> Self {
        // Re-chunk the flat store back into per-vector rows for the wire format.
        let rows = if idx.dim == 0 { 0 } else { idx.store.len() / idx.dim };
        let mut store = Vec::with_capacity(rows);
        for i in 0..rows {
            store.push(idx.store[i * idx.dim..(i + 1) * idx.dim].to_vec());
        }
        Self {
            n_lists: idx.n_lists,
            n_probe: idx.n_probe,
            centroids: idx.centroids,
            posting_lists: idx.posting_lists,
            store,
            deleted: idx.deleted,
            dim: idx.dim,
            trained: idx.trained,
        }
    }
}

impl IvfIndex {
    /// Create an empty, untrained index.
    ///
    /// Call [`train`] with representative data before inserting vectors.
    pub fn new(n_lists: usize, n_probe: usize) -> Self {
        assert!(n_lists >= 1, "n_lists must be >= 1");
        assert!(n_probe >= 1 && n_probe <= n_lists, "n_probe must be in [1, n_lists]");
        Self {
            n_lists,
            n_probe,
            centroids: Vec::new(),
            posting_lists: vec![Vec::new(); n_lists],
            store: Vec::new(),
            deleted: Vec::new(),
            dim: 0,
            trained: false,
        }
    }

    /// Whether the index has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Number of vectors currently stored.
    pub fn len(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.store.len() / self.dim
        }
    }

    /// True when no vectors have been inserted after training.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    // ──────────────────────────── internal helpers ───────────────────────────

    fn normalize(v: &[f32]) -> Vec<f32> {
        let sq: f32 = v.iter().map(|x| x * x).sum();
        if sq == 0.0 {
            return v.to_vec();
        }
        let inv = 1.0 / sq.sqrt();
        v.iter().map(|x| x * inv).collect()
    }

    #[inline]
    fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
        // Use the directly-compiled SIMD dot (NEON / AVX2) instead of SimSIMD's
        // per-call dispatch — this runs over every centroid in the coarse scan
        // and every candidate in the full scan.
        (1.0 - crate::index::simd::dot_f32(a, b)).max(0.0)
    }

    /// Find the nearest centroid to `v`; returns `(centroid_id, distance)`.
    fn nearest_centroid(&self, v: &[f32]) -> (usize, f32) {
        let mut best = 0;
        let mut best_d = f32::INFINITY;
        for ci in 0..self.n_lists {
            let c = &self.centroids[ci * self.dim..(ci + 1) * self.dim];
            let d = Self::cosine_dist(v, c);
            if d < best_d {
                best_d = d;
                best = ci;
            }
        }
        (best, best_d)
    }

    /// Top-`n_probe` centroid IDs for `v`, ascending by cosine distance.
    fn top_centroids(&self, v: &[f32], n_probe: usize) -> Vec<usize> {
        let mut scores: Vec<(usize, f32)> = (0..self.n_lists)
            .map(|ci| {
                let c = &self.centroids[ci * self.dim..(ci + 1) * self.dim];
                (ci, Self::cosine_dist(v, c))
            })
            .collect();
        // NaN-safe ordering: a NaN distance (e.g. a non-finite query) sorts as
        // equal rather than panicking, matching HnswIndex / IvfPqIndex.
        scores.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.into_iter().take(n_probe).map(|(ci, _)| ci).collect()
    }

    #[inline]
    fn is_deleted(&self, id: usize) -> bool {
        self.deleted.get(id).copied().unwrap_or(false)
    }

    /// Borrow stored vector `id` from the flat store.
    #[inline]
    fn vec(&self, id: usize) -> &[f32] {
        &self.store[id * self.dim..(id + 1) * self.dim]
    }

    /// Prefetch every cache line spanning stored vector `id` into L1. The
    /// candidate scan is cold-memory-bound at high dimension; issuing the whole
    /// span a couple of vectors ahead keeps the next vector streaming while the
    /// current cosine computes (mirrors the HNSW search path).
    #[inline(always)]
    fn prefetch_vec(&self, id: usize) {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: callers only pass posting-list ids, so `id < len()` and
        // `id*dim .. id*dim+dim` is in-bounds; `prfm` never faults regardless.
        unsafe {
            let base = self.store.as_ptr().add(id * self.dim);
            let lines = self.dim.div_ceil(16); // 64-byte lines = 16 f32 each
            for l in 0..lines {
                let p = base.add(l * 16);
                core::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p, options(nostack, preserves_flags, readonly));
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        let _ = id;
    }

    // ──────────────────────────── public API ────────────────────────────────

    /// Train the coarse quantizer on `training_data`.
    ///
    /// Must be called exactly once before [`add`] / [`search`].
    ///
    /// # Errors
    /// Returns an error string when `training_data.len() < n_lists`.
    pub fn train<V: AsRef<[f32]>>(
        &mut self,
        training_data: &[V],
        max_iter: usize,
        seed: u64,
    ) -> Result<(), String> {
        if training_data.len() < self.n_lists {
            return Err(format!(
                "training_data.len()={} must be >= n_lists={}",
                training_data.len(),
                self.n_lists
            ));
        }
        let d = training_data[0].as_ref().len();
        if d == 0 {
            return Err("vector dimension must be > 0".into());
        }

        let norms: Vec<Vec<f32>> =
            training_data.iter().map(|v| Self::normalize(v.as_ref())).collect();
        let refs: Vec<&[f32]> = norms.iter().map(|v| v.as_slice()).collect();
        self.centroids = kmeans_lloyd(&refs, self.n_lists, d, max_iter, seed);
        self.dim = d;
        self.trained = true;
        Ok(())
    }

    /// Insert a single vector; returns its global ID.
    ///
    /// # Panics
    /// Panics when the index has not been trained.
    pub fn add(&mut self, vector: &[f32]) -> usize {
        assert!(self.trained, "IvfIndex must be trained before calling add()");
        let norm_vec = Self::normalize(vector);
        let id = self.len();
        let (ci, _) = self.nearest_centroid(&norm_vec);
        self.posting_lists[ci].push(id);
        self.store.extend_from_slice(&norm_vec);
        self.deleted.push(false);
        id
    }

    /// Insert a batch of vectors; returns their global IDs.
    ///
    /// Generic over `AsRef<[f32]>` so callers can pass borrowed row slices
    /// (e.g. `&[&[f32]]` over a contiguous buffer) without an owning copy.
    pub fn add_batch<V: AsRef<[f32]>>(&mut self, vectors: &[V]) -> Vec<usize> {
        vectors.iter().map(|v| self.add(v.as_ref())).collect()
    }

    /// Approximate k-nearest-neighbour search.
    ///
    /// Scans the `n_probe` closest posting lists.  Uses `self.n_probe` by
    /// default; override with [`search_with_probe`].
    ///
    /// Returns `Vec<(global_id, cosine_distance)>` sorted ascending by distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.search_with_probe(query, k, self.n_probe)
    }

    /// Like [`search`] but with an explicit `n_probe` override.
    pub fn search_with_probe(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<(usize, f32)> {
        assert!(self.trained, "IvfIndex must be trained before calling search()");
        let n_probe = n_probe.min(self.n_lists);
        let q = Self::normalize(query);
        let probe_lists = self.top_centroids(&q, n_probe);

        // Collect candidates from all probed posting lists. Within each list we
        // software-prefetch a couple of vectors ahead so the next cold vector is
        // streaming in while the current cosine computes (flat store + prefetch).
        const PREFETCH_AHEAD: usize = 2;
        let mut candidates: Vec<(usize, f32)> = probe_lists
            .par_iter()
            .flat_map(|&ci| {
                let list = &self.posting_lists[ci];
                let mut out: Vec<(usize, f32)> = Vec::with_capacity(list.len());
                for (j, &id) in list.iter().enumerate() {
                    if let Some(&nid) = list.get(j + PREFETCH_AHEAD) {
                        self.prefetch_vec(nid);
                    }
                    if self.is_deleted(id) {
                        continue;
                    }
                    out.push((id, Self::cosine_dist(&q, self.vec(id))));
                }
                out
            })
            .collect();

        candidates.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.dedup_by_key(|e| e.0);
        candidates.truncate(k);
        candidates
    }

    /// Soft-delete a vector by global ID.
    pub fn delete(&mut self, id: usize) {
        if id < self.store.len() {
            if self.deleted.len() < self.store.len() {
                self.deleted.resize(self.store.len(), false);
            }
            self.deleted[id] = true;
        }
    }

    /// Compute mean recall@k against brute-force ground truth.
    pub fn recall_at_k(
        &self,
        queries: &[Vec<f32>],
        ground_truth: &[Vec<usize>],
        k: usize,
        n_probe: usize,
    ) -> f32 {
        use std::collections::HashSet;
        let total: f32 = queries
            .iter()
            .zip(ground_truth.iter())
            .map(|(q, gt)| {
                let res = self.search_with_probe(q, k, n_probe);
                let found: HashSet<usize> = res.iter().map(|&(id, _)| id).collect();
                let hits = gt.iter().take(k).filter(|&&id| found.contains(&id)).count();
                hits as f32 / k.min(gt.len()).max(1) as f32
            })
            .sum();
        total / queries.len() as f32
    }

    /// Persist to a file (bincode).
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load from a file saved with [`save`].
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        Ok(bincode::deserialize(&bytes)?)
    }

    /// Compact the index by permanently removing soft-deleted vectors and
    /// remapping all posting lists to contiguous global IDs.
    ///
    /// Returns the number of vectors removed.  If no vectors are deleted this
    /// is a cheap no-op.
    pub fn vacuum(&mut self) -> usize {
        let deleted_count = self.deleted.iter().filter(|&&d| d).count();
        if deleted_count == 0 {
            return 0;
        }

        // Build old_id → new_id mapping.
        let mut mapping: Vec<Option<usize>> = Vec::with_capacity(self.len());
        let mut new_id = 0usize;
        for &del in &self.deleted {
            if del {
                mapping.push(None);
            } else {
                mapping.push(Some(new_id));
                new_id += 1;
            }
        }

        // Compact the vector store into a fresh contiguous buffer.
        let survivors = self.len() - deleted_count;
        let mut new_store: Vec<f32> = Vec::with_capacity(survivors * self.dim);
        for id in 0..self.len() {
            if !self.deleted[id] {
                new_store.extend_from_slice(self.vec(id));
            }
        }

        // Remap all posting lists (filter deleted, translate IDs).
        for list in &mut self.posting_lists {
            *list = list.iter().filter_map(|&id| mapping[id]).collect();
        }

        self.store = new_store;
        self.deleted = vec![false; survivors];
        deleted_count
    }

    /// Filtered approximate k-nearest-neighbour search.
    ///
    /// Only vectors for which `filter(global_id) == true` are included in the
    /// result.  Uses `self.n_probe` posting lists.
    pub fn search_filtered<F: Fn(usize) -> bool>(
        &self,
        query: &[f32],
        k: usize,
        filter: F,
    ) -> Vec<(usize, f32)> {
        self.search_filtered_with_probe(query, k, self.n_probe, filter)
    }

    /// Like [`search_filtered`] but with an explicit `n_probe` override.
    pub fn search_filtered_with_probe<F: Fn(usize) -> bool>(
        &self,
        query: &[f32],
        k: usize,
        n_probe: usize,
        filter: F,
    ) -> Vec<(usize, f32)> {
        assert!(self.trained, "IvfIndex must be trained before calling search()");
        let n_probe = n_probe.min(self.n_lists);
        let q = Self::normalize(query);
        let probe_lists = self.top_centroids(&q, n_probe);

        let mut candidates: Vec<(usize, f32)> = probe_lists
            .iter()
            .flat_map(|&ci| {
                self.posting_lists[ci]
                    .iter()
                    .filter(|&&id| !self.is_deleted(id) && filter(id))
                    .map(|&id| (id, Self::cosine_dist(&q, self.vec(id))))
                    .collect::<Vec<_>>()
            })
            .collect();

        candidates.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.dedup_by_key(|e| e.0);
        candidates.truncate(k);
        candidates
    }

    /// Find the minimum `n_probe` that achieves at least `target_recall` for
    /// `query` relative to exhaustive search.
    ///
    /// Uses an exponential doubling probe schedule.  Returns
    /// `(results, n_probe_used)`.
    pub fn search_for_recall(
        &self,
        query: &[f32],
        k: usize,
        target_recall: f32,
    ) -> (Vec<(usize, f32)>, usize) {
        // Exhaustive ground truth.
        let exhaustive = self.search_with_probe(query, k, self.n_lists);
        let gt_ids: std::collections::HashSet<usize> =
            exhaustive.iter().map(|&(id, _)| id).collect();
        let gt_k = gt_ids.len().max(1);

        let mut n_probe = 1usize;
        loop {
            let results = self.search_with_probe(query, k, n_probe);
            let hits = results
                .iter()
                .filter(|(id, _)| gt_ids.contains(id))
                .count();
            let recall = hits as f32 / gt_k as f32;
            if recall >= target_recall || n_probe >= self.n_lists {
                return (results, n_probe);
            }
            n_probe = (n_probe * 2).min(self.n_lists);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..d).map(|j| ((i * d + j) as f32 * 0.031 + 0.1).sin()).collect())
            .collect()
    }

    fn brute_force_gt(vecs: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
        queries
            .iter()
            .map(|q| {
                let nq: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                let mut scores: Vec<(usize, f32)> = vecs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let nv: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                        let dot: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                        (i, dot / (nq * nv))
                    })
                    .collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scores.into_iter().take(k).map(|(i, _)| i).collect()
            })
            .collect()
    }

    #[test]
    fn train_and_add_smoke() {
        let vecs = make_vecs(100, 32);
        let mut idx = IvfIndex::new(8, 4);
        idx.train(&vecs, 20, 42).expect("train failed");
        for v in &vecs {
            idx.add(v);
        }
        assert_eq!(idx.len(), 100);
        // Every vector should appear exactly once across all posting lists.
        let total: usize = idx.posting_lists.iter().map(|l| l.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn train_borrowed_slices_matches_owned() {
        // Workstream A: the AsRef<[f32]> generic must produce bit-identical
        // centroids whether fed owned `Vec<f32>` rows or borrowed `&[f32]` rows
        // (the zero-copy path the PyO3 bindings take on contiguous arrays).
        let vecs = make_vecs(120, 32);

        let mut owned = IvfIndex::new(8, 4);
        owned.train(&vecs, 20, 42).expect("owned train failed");

        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let mut borrowed = IvfIndex::new(8, 4);
        borrowed.train(&refs, 20, 42).expect("borrowed train failed");

        assert_eq!(owned.centroids, borrowed.centroids);
        assert_eq!(owned.dim, borrowed.dim);
    }

    #[test]
    fn add_batch_borrowed_slices_matches_owned() {
        let vecs = make_vecs(120, 32);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let mut owned = IvfIndex::new(8, 4);
        owned.train(&vecs, 20, 42).unwrap();
        owned.add_batch(&vecs);

        let mut borrowed = IvfIndex::new(8, 4);
        borrowed.train(&refs, 20, 42).unwrap();
        borrowed.add_batch(&refs);

        assert_eq!(owned.len(), borrowed.len());
        assert_eq!(owned.store, borrowed.store);
        assert_eq!(owned.posting_lists, borrowed.posting_lists);
    }

    #[test]
    fn search_empty_after_train() {
        let train = make_vecs(10, 8);
        let mut idx = IvfIndex::new(4, 2);
        idx.train(&train, 10, 1).unwrap();
        let q = vec![0.1f32; 8];
        let res = idx.search(&q, 5);
        assert!(res.is_empty());
    }

    #[test]
    fn search_with_nan_query_does_not_panic() {
        // A non-finite query produces NaN cosine distances; the centroid /
        // candidate sorts must degrade gracefully (NaN-as-equal) rather than
        // panicking on `partial_cmp().unwrap()`.
        let vecs = make_vecs(40, 16);
        let mut idx = IvfIndex::new(4, 4);
        idx.train(&vecs, 10, 3).unwrap();
        idx.add_batch(&vecs);

        let mut q = vec![0.1f32; 16];
        q[3] = f32::NAN;
        q[7] = f32::INFINITY;
        // Must not panic on any search variant that sorts by distance.
        let _ = idx.search(&q, 5);
        let _ = idx.search_with_probe(&q, 5, 4);
        let _ = idx.search_filtered_with_probe(&q, 5, 4, |id| id % 2 == 0);
    }

    #[test]
    fn search_self_nearest() {
        let vecs = make_vecs(50, 16);
        let mut idx = IvfIndex::new(5, 5); // n_probe == n_lists → exact
        idx.train(&vecs, 20, 7).unwrap();
        for v in &vecs {
            idx.add(v);
        }
        for (i, v) in vecs.iter().enumerate() {
            let res = idx.search(v, 1);
            assert_eq!(res.len(), 1);
            // Distance to self (or a cosine-identical vector) must be ≈ 0.
            // Note: due to the periodic sin generator two vectors can share an
            // identical unit-normalised direction; we accept any match with
            // cosine distance < 1e-4 rather than requiring a specific id.
            assert!(
                res[0].1 < 1e-4,
                "vec[{i}] nearest dist={} (id={}); expected < 1e-4",
                res[0].1,
                res[0].0
            );
        }
    }

    #[test]
    fn n_probe_full_equals_exact() {
        // When n_probe == n_lists the result must equal brute-force.
        let vecs = make_vecs(80, 16);
        let queries = &vecs[..10];
        let k = 5;
        let gt = brute_force_gt(&vecs, queries, k);

        let mut idx = IvfIndex::new(4, 4);
        idx.train(&vecs, 20, 99).unwrap();
        for v in &vecs {
            idx.add(v);
        }
        let recall = idx.recall_at_k(queries, &gt, k, 4);
        assert!(recall >= 0.99, "full-probe recall={recall:.3}");
    }

    #[test]
    fn delete_excludes_from_search() {
        let vecs = make_vecs(20, 8);
        let mut idx = IvfIndex::new(4, 4);
        idx.train(&vecs, 10, 5).unwrap();
        for v in &vecs {
            idx.add(v);
        }
        idx.delete(0);
        let res = idx.search(&vecs[0], 5);
        let ids: Vec<usize> = res.iter().map(|&(id, _)| id).collect();
        assert!(!ids.contains(&0), "deleted id 0 appeared in results: {ids:?}");
    }

    #[test]
    fn delete_out_of_bounds_no_panic() {
        let vecs = make_vecs(5, 4);
        let mut idx = IvfIndex::new(2, 2);
        idx.train(&vecs, 5, 1).unwrap();
        idx.delete(9999); // must not panic
    }

    #[test]
    fn save_load_roundtrip() {
        let vecs = make_vecs(40, 16);
        let mut idx = IvfIndex::new(4, 4);
        idx.train(&vecs, 15, 13).unwrap();
        for v in &vecs {
            idx.add(v);
        }
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("ivf.bin");
        idx.save(&path).unwrap();
        let loaded = IvfIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        let q = vecs[0].clone();
        let r1 = idx.search(&q, 5);
        let r2 = loaded.search(&q, 5);
        assert_eq!(
            r1.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            r2.iter().map(|&(id, _)| id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn train_error_too_few_vectors() {
        let vecs = make_vecs(3, 4);
        let mut idx = IvfIndex::new(8, 2);
        let err = idx.train(&vecs, 10, 0);
        assert!(err.is_err());
    }

    #[test]
    fn untrained_add_panics() {
        let mut idx = IvfIndex::new(4, 2);
        let result = std::panic::catch_unwind(move || {
            idx.add(&[1.0f32, 0.0, 0.0]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn vacuum_compacts_deleted_vectors() {
        let vecs = make_vecs(40, 16);
        let mut idx = IvfIndex::new(4, 4);
        idx.train(&vecs, 10, 42).unwrap();
        for v in &vecs {
            idx.add(v);
        }
        // Delete 5 vectors.
        for id in [0, 5, 10, 15, 20] {
            idx.delete(id);
        }
        let removed = idx.vacuum();
        assert_eq!(removed, 5, "vacuum should report 5 removed vectors");
        assert_eq!(idx.len(), 35, "35 survivors after vacuum");
        // Posting lists must not contain any out-of-bounds IDs.
        let total: usize = idx.posting_lists.iter().map(|l| l.len()).sum();
        assert_eq!(total, 35);
        // No tombstones left.
        assert!(!idx.deleted.iter().any(|&d| d));
        // vacuum on already-clean index is a no-op.
        assert_eq!(idx.vacuum(), 0);
    }

    #[test]
    fn search_filtered_respects_allowlist() {
        let vecs = make_vecs(80, 16);
        let mut idx = IvfIndex::new(4, 4);
        idx.train(&vecs, 10, 42).unwrap();
        for v in &vecs {
            idx.add(v);
        }
        // Allow only even IDs.
        let results = idx.search_filtered(&vecs[0], 10, |id| id % 2 == 0);
        for &(id, _) in &results {
            assert_eq!(id % 2, 0, "odd id {id} in filtered results");
        }
        assert!(!results.is_empty(), "filtered search should have results");
    }

    #[test]
    fn search_for_recall_finds_reasonable_probe() {
        let vecs = make_vecs(200, 16);
        let mut idx = IvfIndex::new(8, 8);
        idx.train(&vecs, 20, 42).unwrap();
        for v in &vecs {
            idx.add(v);
        }
        let (results, n_probe) = idx.search_for_recall(&vecs[0], 5, 0.8);
        // n_probe must be within [1, n_lists].
        assert!((1..=8).contains(&n_probe));
        // Results must be non-empty.
        assert!(!results.is_empty());
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    fn make_idx(data: &[Vec<f32>], n_lists: usize) -> IvfIndex {
        let n_lists = n_lists.min(data.len()).max(1);
        let mut idx = IvfIndex::new(n_lists, n_lists);
        idx.train(data, 10, 42).unwrap();
        for v in data {
            idx.add(v);
        }
        idx
    }

    proptest! {
        /// With n_probe == n_lists every inserted vector must be its own nearest neighbour.
        #[test]
        fn self_is_nearest_full_probe(
            n in 4usize..20,
            d in 4usize..16usize,
        ) {
            let vecs: Vec<Vec<f32>> = (0..n)
                .map(|i| (0..d).map(|j| ((i * d + j) as f32 * 0.1 + 0.05).sin()).collect())
                .collect();
            let idx = make_idx(&vecs, 2);
            for (i, v) in vecs.iter().enumerate() {
                let res = idx.search(v, 1);
                prop_assert_eq!(res[0].0, i, "self-query failed at i={}", i);
            }
        }

        /// Search must never return a deleted ID.
        #[test]
        fn delete_never_returned(
            n in 8usize..30,
            del_id in 0usize..8,
        ) {
            let d = 8;
            let vecs: Vec<Vec<f32>> = (0..n)
                .map(|i| (0..d).map(|j| ((i + j) as f32 * 0.07).cos()).collect())
                .collect();
            let del = del_id % n;
            let mut idx = make_idx(&vecs, 2);
            idx.delete(del);
            for v in &vecs {
                let res = idx.search(v, n);
                for &(id, _) in &res {
                    prop_assert_ne!(id, del, "deleted id {} appeared in results", del);
                }
            }
        }
    }
}
