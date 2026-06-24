//! IVF-PQ index — coarse IVF clustering with Product Quantization residuals.
//!
//! Uses Asymmetric Distance Computation (ADC) so distances are computed from
//! the pre-built look-up table in O(M) rather than O(d) operations.
//!
//! Typical usage:
//! ```ignore
//! let mut idx = IvfPqIndex::new(64, 8);      // 64 lists, probe 8
//! idx.train(&data, 8, 16, 25, 42).unwrap();  // M=8 sub-spaces, K=16 centroids
//! for v in &data { idx.add(v); }
//! let results = idx.search(&query, 10);
//! ```

use crate::quant::pq::{pq_distance_table, train_pq_codebook, PQCodebook};
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ─── helpers ──────────────────────────────────────────────────────────────────

/// LCG parameters identical to pq.rs / ivf.rs for reproducible init.
#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

/// Indices of the `n_probe` centroids with the highest cosine similarity in a
/// query's coarse-scan row. Mirrors `top_coarse` (closest centroids first) but
/// reads pre-computed similarities from the batched GEMM. Uses
/// `select_nth_unstable` so the selection is O(n_lists) rather than a full sort.
fn top_probe_from_sims(sims: &[f32], n_probe: usize) -> Vec<usize> {
    let n = sims.len();
    let take = n_probe.min(n);
    if take == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    if take < n {
        // Partition so the `take` highest similarities sit first (descending).
        idx.select_nth_unstable_by(take - 1, |&a, &b| {
            sims[b].partial_cmp(&sims[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(take);
    }
    idx
}

/// Cosine distance (1 − cosine similarity) using SimSIMD.
#[inline]
pub(crate) fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    // Unit-norm vectors → cosine distance is 1 − dot. The shared SIMD kernel
    // avoids SimSIMD's per-call dispatch, which dominated the coarse-quantiser
    // scan (called over every centroid, per query) and k-means assignment.
    (1.0 - super::simd::dot_f32(a, b)).max(0.0)
}

/// K-means++ initialisation — returns `k` centroids from `data`.
///
/// Maintains a running per-point squared distance to the *nearest chosen*
/// centroid, updated against only the newly added centroid each round
/// (parallel). This is the standard O(n·k·d) algorithm; the previous version
/// recomputed distances to *all* chosen centroids every round — O(n·k²·d) and
/// serial — which dominated IVF training time (≈k/2 = hundreds× more work at
/// k=512).
fn kmeans_pp_init(data: &[Vec<f32>], k: usize, d: usize, seed: u64) -> Vec<f32> {
    let n = data.len();
    let mut rng = seed;
    let mut centroids = Vec::with_capacity(k * d);

    // First centroid: random pick.
    rng = lcg_next(rng);
    let first = (rng as usize) % n;
    centroids.extend_from_slice(&data[first]);

    // Running squared distance from each point to its nearest chosen centroid.
    let mut min_d2: Vec<f32> = data
        .par_iter()
        .map(|v| {
            let dd = cosine_dist(v, &data[first]);
            dd * dd
        })
        .collect();

    for _ in 1..k {
        let total: f32 = min_d2.iter().sum();
        rng = lcg_next(rng);
        let mut target = (rng as f64 / u64::MAX as f64) as f32 * total;
        let mut picked = n - 1;
        for (i, &d2) in min_d2.iter().enumerate() {
            target -= d2;
            if target <= 0.0 {
                picked = i;
                break;
            }
        }
        let new_cent = data[picked].clone();
        centroids.extend_from_slice(&new_cent);

        // Refresh the running minimum against only the new centroid (parallel).
        min_d2
            .par_iter_mut()
            .zip(data.par_iter())
            .for_each(|(m, v)| {
                let dd = cosine_dist(v, &new_cent);
                let d2 = dd * dd;
                if d2 < *m {
                    *m = d2;
                }
            });
    }
    centroids
}

/// Lloyd's k-means over `data` (unit-norm vectors, cosine distance).
///
/// Returns centroids as a flat `[k * d]` slice. `pub(crate)` so the coarse
/// quantiser is shared with [`super::pq4`]'s IVF-PQ4 fast-scan index.
pub(crate) fn kmeans_lloyd(
    data: &[Vec<f32>],
    k: usize,
    d: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<f32> {
    let _n = data.len();
    let mut centroids = kmeans_pp_init(data, k, d, seed);

    for _ in 0..max_iter {
        // Assignment step — parallelised
        let assignments: Vec<usize> = data
            .par_iter()
            .map(|v| {
                let mut best_c = 0usize;
                let mut best_d = f32::MAX;
                for ci in 0..k {
                    let cent = &centroids[ci * d..(ci + 1) * d];
                    let dist = cosine_dist(v, cent);
                    if dist < best_d {
                        best_d = dist;
                        best_c = ci;
                    }
                }
                best_c
            })
            .collect();

        // Update step
        let mut new_centroids = vec![0.0f32; k * d];
        let mut counts = vec![0usize; k];
        for (v, &ci) in data.iter().zip(assignments.iter()) {
            counts[ci] += 1;
            let base = ci * d;
            for (j, &x) in v.iter().enumerate() {
                new_centroids[base + j] += x;
            }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                let inv = 1.0 / counts[ci] as f32;
                let base = ci * d;
                for j in 0..d {
                    new_centroids[base + j] *= inv;
                }
            } else {
                // Empty cluster: re-seed from old centroid
                new_centroids[ci * d..(ci + 1) * d]
                    .copy_from_slice(&centroids[ci * d..(ci + 1) * d]);
            }
        }

        // Convergence check
        let moved: f32 = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        centroids = new_centroids;
        if moved < 1e-7 {
            break;
        }
    }
    centroids
}

// ─── IvfPqIndex ───────────────────────────────────────────────────────────────

/// IVF index with PQ-compressed residuals and ADC scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfPqIndex {
    /// Number of coarse clusters.
    n_lists: usize,
    /// Number of lists visited at query time.
    n_probe: usize,
    /// Coarse centroids, shape [n_lists * dim], unit-norm.
    coarse_centroids: Vec<f32>,
    /// Per-list global-id posting lists.
    posting_lists: Vec<Vec<usize>>,
    /// Trained PQ codebook.
    codebook: PQCodebook,
    /// PQ codes for every vector, stored flat and row-major as `[n_vectors * M]`
    /// (`M = codebook.n_subspaces`). A single contiguous buffer — rather than one
    /// heap `Vec<u8>` per vector — keeps a candidate's codes on one cache line and
    /// lets the ADC scan stream sequentially instead of pointer-chasing scattered
    /// allocations. The number of stored vectors is tracked by `deleted.len()`.
    pq_codes: Vec<u8>,
    /// Soft-deletion tombstones.
    #[serde(default)]
    deleted: Vec<bool>,
    /// Vector dimension.
    dim: usize,
    /// True after `train` succeeds.
    trained: bool,
}

impl IvfPqIndex {
    /// Create a new, untrained IvfPqIndex.
    ///
    /// * `n_lists`  — number of coarse clusters (typical: sqrt(N))
    /// * `n_probe`  — lists to visit at query time (typical: 8–64)
    pub fn new(n_lists: usize, n_probe: usize) -> Self {        Self {
            n_lists,
            n_probe,
            coarse_centroids: Vec::new(),
            posting_lists: vec![Vec::new(); n_lists],
            codebook: PQCodebook {
                n_subspaces: 0,
                n_centroids: 0,
                sub_dim: 0,
                centroids: Vec::new(),
            },
            pq_codes: Vec::new(),
            deleted: Vec::new(),
            dim: 0,
            trained: false,
        }
    }

    /// Number of coarse clusters.
    pub fn n_lists(&self) -> usize { self.n_lists }

    /// Whether the index has been trained.
    pub fn is_trained(&self) -> bool { self.trained }

    /// Train both the coarse quantizer and the PQ codebook.
    ///
    /// # Arguments
    /// * `training_data` — representative vectors (all same length)
    /// * `n_subspaces`   — PQ sub-spaces M; must divide `dim`
    /// * `n_centroids`   — PQ centroids K per sub-space; ≤ 256
    /// * `max_iter`      — Lloyd's iterations for both k-means passes
    /// * `seed`          — RNG seed
    pub fn train<V: AsRef<[f32]>>(
        &mut self,
        training_data: &[V],
        n_subspaces: usize,
        n_centroids: usize,
        max_iter: usize,
        seed: u64,
    ) -> Result<(), String> {
        if training_data.is_empty() {
            return Err("training_data is empty".into());
        }
        if training_data.len() < self.n_lists {
            return Err(format!(
                "need ≥ n_lists ({}) training vectors, got {}",
                self.n_lists,
                training_data.len()
            ));
        }
        let d = training_data[0].as_ref().len();
        if d == 0 {
            return Err("vector dimension is 0".into());
        }
        if !training_data.iter().all(|v| v.as_ref().len() == d) {
            return Err("training vectors have inconsistent lengths".into());
        }

        // --- Normalise training vectors ---
        let normed: Vec<Vec<f32>> = training_data
            .iter()
            .map(|v| {
                let v = v.as_ref();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                v.iter().map(|x| x / norm).collect()
            })
            .collect();

        // --- Coarse k-means ---
        self.coarse_centroids = kmeans_lloyd(&normed, self.n_lists, d, max_iter, seed);
        self.dim = d;

        // --- Train PQ codebook on the full normalised training set ---
        self.codebook = train_pq_codebook(&normed, n_subspaces, n_centroids, max_iter, seed)
            .map_err(|e| e.to_string())?;
        self.trained = true;
        Ok(())
    }

    /// Add a single vector; returns its global id.
    ///
    /// Panics if the index has not been trained.
    pub fn add(&mut self, vector: &[f32]) -> usize {
        assert!(self.trained, "IvfPqIndex must be trained before adding vectors");
        assert_eq!(
            vector.len(),
            self.dim,
            "vector dim {} ≠ index dim {}",
            vector.len(),
            self.dim
        );

        // Normalise
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let v_norm: Vec<f32> = vector.iter().map(|x| x / norm).collect();

        // Assign to nearest coarse centroid
        let list_id = self.nearest_coarse(&v_norm);

        // PQ-encode (encode_batch expects a slice of vecs)
        let codes = self.pq_encode_single(&v_norm);

        let global_id = self.deleted.len();
        self.pq_codes.extend_from_slice(&codes);
        self.deleted.push(false);
        self.posting_lists[list_id].push(global_id);
        global_id
    }

    /// The PQ code row for global id `gid`: a contiguous `M`-byte slice into the
    /// flat `pq_codes` buffer.
    #[inline]
    fn code_row(&self, gid: usize) -> &[u8] {
        let m = self.codebook.n_subspaces;
        &self.pq_codes[gid * m..gid * m + m]
    }

    /// Search for the `k` nearest neighbours using ADC scoring.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.search_with_probe(query, k, self.n_probe)
    }

    /// Search with explicit `n_probe` override.
    pub fn search_with_probe(
        &self,
        query: &[f32],
        k: usize,
        n_probe: usize,
    ) -> Vec<(usize, f32)> {
        if !self.trained || self.pq_codes.is_empty() {
            return Vec::new();
        }
        assert_eq!(query.len(), self.dim, "query dim mismatch");

        // Normalise query
        let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let q_norm: Vec<f32> = query.iter().map(|x| x / norm).collect();

        // Find top-n_probe coarse centroids, then ADC-rank the vectors they hold.
        let probe_lists = self.top_coarse(&q_norm, n_probe);
        self.adc_rank(&q_norm, &probe_lists, k)
    }

    /// ADC-rank the vectors in `probe_lists` against a normalised query, returning
    /// the top-`k` `(id, distance)` pairs ascending. Shared by the single-query
    /// (`search_with_probe`) and batch (`search_batch_flat`) paths so both compute
    /// distances identically — only how `probe_lists` is chosen differs.
    fn adc_rank(&self, q_norm: &[f32], probe_lists: &[usize], k: usize) -> Vec<(usize, f32)> {
        if k == 0 {
            return Vec::new();
        }
        // Build ADC table once for this query.
        let dist_table = pq_distance_table(q_norm, &self.codebook);
        let m = self.codebook.n_subspaces;
        let kc = self.codebook.n_centroids;

        // Scan posting lists — collect (dist, id) pairs.
        let mut candidates: Vec<(f32, usize)> = Vec::new();
        for &list_id in probe_lists {
            for &gid in &self.posting_lists[list_id] {
                if self.deleted[gid] {
                    continue;
                }
                let codes = self.code_row(gid);
                let adc_dist = crate::quant::pq::adc_distance(&dist_table, codes, m, kc);
                candidates.push((adc_dist, gid));
            }
        }

        // Top-k by ADC distance. Each vector belongs to exactly one posting list
        // and `probe_lists` is a distinct set, so a gid appears at most once —
        // no dedup needed. `select_nth_unstable_by` partitions the k smallest in
        // O(C) instead of fully sorting all C candidates (O(C log C)); only the
        // retained k are then sorted.
        if candidates.len() > k {
            candidates.select_nth_unstable_by(k - 1, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(k);
        }
        candidates.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.into_iter().map(|(d, id)| (id, d)).collect()
    }

    /// Batch search over a flat `[q * dim]` query buffer, parallelised across
    /// queries (rayon). `search_with_probe` is `&self`/read-only, so this is
    /// lock-free. Returns one `(id, dist)` result list per query row — the
    /// throughput path that avoids per-query Python/FFI call overhead.
    pub fn search_batch_flat(
        &self,
        flat: &[f32],
        dim: usize,
        k: usize,
        n_probe: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        if dim == 0 {
            return Vec::new();
        }
        let q = flat.len() / dim;
        if !self.trained || self.pq_codes.is_empty() {
            return vec![Vec::new(); q];
        }
        assert_eq!(dim, self.dim, "query dim mismatch");
        if q == 0 {
            return Vec::new();
        }

        // ── Batched coarse scan as one matrix multiply ──────────────────────────
        // Per query the coarse step compares the query to every centroid — at
        // d=768 those `q × n_lists` full-dim dots dominate the search. Computing
        // them as `Q · Cᵀ` (pure-Rust matrixmultiply via ndarray) reuses the
        // centroid matrix across all queries instead of re-streaming it per query,
        // the way FAISS routes its coarse quantiser through GEMM. Both queries and
        // centroids are unit-norm, so the dot product *is* cosine similarity.
        let mut qnorm = vec![0.0f32; q * dim];
        for i in 0..q {
            let row = &flat[i * dim..(i + 1) * dim];
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let inv = 1.0 / norm;
            for (j, &x) in row.iter().enumerate() {
                qnorm[i * dim + j] = x * inv;
            }
        }
        let cmat = ArrayView2::from_shape((self.n_lists, dim), &self.coarse_centroids)
            .expect("centroid shape");

        // ── Tile queries; each tile runs its own coarse GEMM + ADC in parallel ──
        // A whole-batch single GEMM would be cache-efficient but single-threaded,
        // surrendering the per-query rayon parallelism. Tiling keeps both: each
        // worker computes `Q_tile · Cᵀ` (centroid matrix reused across the tile)
        // and then ADC-ranks the tile's queries. CHUNK is sized so a tile's sim
        // block stays cache-resident.
        const CHUNK: usize = 32;
        let mut results: Vec<Vec<(usize, f32)>> = vec![Vec::new(); q];
        results
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(c, out)| {
                let lo = c * CHUNK;
                let rows = out.len();
                let qtile = ArrayView2::from_shape((rows, dim), &qnorm[lo * dim..(lo + rows) * dim])
                    .expect("query tile shape");
                let sims: Array2<f32> = qtile.dot(&cmat.t()); // (rows, n_lists)
                for (r, slot) in out.iter_mut().enumerate() {
                    let srow = sims.row(r);
                    let probe_lists =
                        top_probe_from_sims(srow.as_slice().expect("contig row"), n_probe);
                    let qi = lo + r;
                    *slot = self.adc_rank(&qnorm[qi * dim..(qi + 1) * dim], &probe_lists, k);
                }
            });
        results
    }

    /// Soft-delete a vector by global id.  Out-of-bounds ids are ignored.
    pub fn delete(&mut self, id: usize) {
        if id < self.deleted.len() {
            self.deleted[id] = true;
        }
    }

    /// Recall@K evaluation.
    ///
    /// For each (query, ground-truth-ids) pair, computes what fraction of
    /// `ground_truth_ids[i]` appear in the top-k search results.
    pub fn recall_at_k(
        &self,
        queries: &[Vec<f32>],
        ground_truth: &[Vec<usize>],
        k: usize,
        n_probe: usize,
    ) -> f32 {
        assert_eq!(queries.len(), ground_truth.len());
        if queries.is_empty() {
            return 0.0;
        }
        let sum: f32 = queries
            .iter()
            .zip(ground_truth.iter())
            .map(|(q, gt)| {
                let results = self.search_with_probe(q, k, n_probe);
                // Hash the ground-truth ids once (O(k)) rather than a linear
                // `gt.contains` per result (O(k²) per query).
                let gt_set: std::collections::HashSet<usize> = gt.iter().copied().collect();
                let found = results.iter().filter(|(id, _)| gt_set.contains(id)).count();
                found as f32 / gt.len() as f32
            })
            .sum();
        sum / queries.len() as f32
    }

    /// Serialize to a file at `path`.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let bytes = bincode::serialize(self).expect("serialization failed");
        std::fs::write(path, bytes)
    }

    /// Deserialize from a file at `path`.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let index: Self = bincode::deserialize(&bytes).expect("deserialization failed");
        Ok(index)
    }

    /// Compact the index by permanently removing soft-deleted vectors and
    /// remapping posting lists to contiguous global IDs.
    ///
    /// Returns the number of vectors removed.  If no vectors are deleted this
    /// is a cheap no-op.
    pub fn vacuum(&mut self) -> usize {
        let deleted_count = self.deleted.iter().filter(|&&d| d).count();
        if deleted_count == 0 {
            return 0;
        }

        // Build old_id → new_id mapping.
        let mut mapping: Vec<Option<usize>> = Vec::with_capacity(self.deleted.len());
        let mut new_id = 0usize;
        for &del in &self.deleted {
            if del {
                mapping.push(None);
            } else {
                mapping.push(Some(new_id));
                new_id += 1;
            }
        }

        // Compact the flat PQ-code buffer, copying surviving rows contiguously.
        let m = self.codebook.n_subspaces;
        let mut new_codes: Vec<u8> = Vec::with_capacity(new_id * m);
        for (gid, &del) in self.deleted.iter().enumerate() {
            if !del {
                new_codes.extend_from_slice(&self.pq_codes[gid * m..gid * m + m]);
            }
        }

        // Remap posting lists.
        for list in &mut self.posting_lists {
            *list = list.iter().filter_map(|&id| mapping[id]).collect();
        }

        self.pq_codes = new_codes;
        self.deleted = vec![false; new_id];
        deleted_count
    }

    /// Find the minimum `n_probe` that achieves at least `target_recall` for
    /// `query` relative to exhaustive ADC search.
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

    // ── private helpers ──────────────────────────────────────────────────────

    /// Index of the nearest coarse centroid (cosine distance on unit-norm `v`).
    fn nearest_coarse(&self, v: &[f32]) -> usize {
        let d = self.dim;
        let mut best_c = 0usize;
        let mut best_dist = f32::MAX;
        for ci in 0..self.n_lists {
            let cent = &self.coarse_centroids[ci * d..(ci + 1) * d];
            let dist = cosine_dist(v, cent);
            if dist < best_dist {
                best_dist = dist;
                best_c = ci;
            }
        }
        best_c
    }

    /// Top-n_probe coarse-centroid ids sorted by distance (closest first).
    fn top_coarse(&self, v: &[f32], n_probe: usize) -> Vec<usize> {
        let d = self.dim;
        let mut scored: Vec<(f32, usize)> = (0..self.n_lists)
            .map(|ci| {
                let cent = &self.coarse_centroids[ci * d..(ci + 1) * d];
                (cosine_dist(v, cent), ci)
            })
            .collect();
        // Partial selection: when n_probe ≪ n_lists (the usual case) a full
        // sort of all centroids is wasted work. `select_nth_unstable` is O(n),
        // then only the probed prefix is sorted.
        let probe = n_probe.min(self.n_lists);
        if probe == 0 {
            return Vec::new();
        }
        if probe < scored.len() {
            scored.select_nth_unstable_by(probe - 1, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(probe);
        }
        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(_, ci)| ci).collect()
    }

    /// PQ-encode a single (already-normalised) vector.
    fn pq_encode_single(&self, v: &[f32]) -> Vec<u8> {
        let m = self.codebook.n_subspaces;
        let sub_dim = self.codebook.sub_dim;
        let k = self.codebook.n_centroids;
        (0..m)
            .map(|mi| {
                let v_sub = &v[mi * sub_dim..(mi + 1) * sub_dim];
                let mut best_k = 0u8;
                let mut best_d = f32::MAX;
                for ki in 0..k {
                    let cent = self.codebook.centroid(mi, ki);
                    let dist: f32 = v_sub
                        .iter()
                        .zip(cent.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    if dist < best_d {
                        best_d = dist;
                        best_k = ki as u8;
                    }
                }
                best_k
            })
            .collect()
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn random_unit_vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..d)
                    .map(|_| {
                        state = lcg_next(state);
                        (state as i64 as f32) / i64::MAX as f32
                    })
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                v.into_iter().map(|x| x / norm).collect()
            })
            .collect()
    }

    #[test]
    fn train_and_add_smoke() {
        let data = random_unit_vecs(200, 32, 7);
        let mut idx = IvfPqIndex::new(8, 4);
        idx.train(&data, 4, 8, 10, 7).unwrap();
        for v in &data {
            idx.add(v);
        }
        assert_eq!(idx.deleted.len(), 200); // 200 vectors stored
        assert_eq!(idx.pq_codes.len(), 200 * idx.codebook.n_subspaces);
    }

    #[test]
    fn kmeans_pp_init_picks_k_distinct_data_points() {
        // Regression guard for the O(n·k·d) running-min rewrite: init must return
        // exactly k centroids, each an actual data row (k-means++ seeds from the
        // data), with no duplicate picks for well-separated points.
        let data = random_unit_vecs(500, 8, 3);
        let (k, d) = (32usize, 8usize);
        let cents = kmeans_pp_init(&data, k, d, 11);
        assert_eq!(cents.len(), k * d);
        let mut picks = Vec::new();
        for ci in 0..k {
            let c = &cents[ci * d..(ci + 1) * d];
            let found = data.iter().position(|v| v.as_slice() == c);
            assert!(found.is_some(), "centroid {ci} is not a data row");
            picks.push(found.unwrap());
        }
        picks.sort_unstable();
        picks.dedup();
        assert_eq!(picks.len(), k, "k-means++ picked duplicate seeds");
    }

    #[test]
    fn search_empty_returns_empty() {
        let data = random_unit_vecs(100, 32, 1);
        let mut idx = IvfPqIndex::new(4, 2);
        idx.train(&data, 4, 8, 5, 1).unwrap();
        // No vectors added → search must return empty
        let res = idx.search(&data[0], 5);
        assert!(res.is_empty());
    }

    #[test]
    fn search_k_zero_returns_empty() {
        // Guard the bounded top-k path: k == 0 must yield no results, not the
        // full candidate set (the select_nth_unstable fast path only runs k > 0).
        let data = random_unit_vecs(100, 32, 9);
        let mut idx = IvfPqIndex::new(4, 4);
        idx.train(&data, 4, 8, 5, 9).unwrap();
        for v in &data {
            idx.add(v);
        }
        assert!(idx.search(&data[0], 0).is_empty());
    }

    #[test]
    fn search_topk_matches_full_sort() {
        // The select_nth_unstable top-k must return exactly the same ids/order as
        // a full sort of every candidate's ADC distance (quality-neutral change).
        let data = random_unit_vecs(300, 32, 11);
        let mut idx = IvfPqIndex::new(8, 8); // full probe → deterministic candidate set
        idx.train(&data, 4, 16, 10, 11).unwrap();
        for v in &data {
            idx.add(v);
        }
        for v in data.iter().take(10) {
            let got: Vec<usize> = idx.search(v, 5).into_iter().map(|(id, _)| id).collect();
            // Reference: brute-force ADC over all vectors, full sort, take 5.
            let q_norm = {
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                v.iter().map(|x| x / n).collect::<Vec<f32>>()
            };
            let table = pq_distance_table(&q_norm, &idx.codebook);
            let m = idx.codebook.n_subspaces;
            let kc = idx.codebook.n_centroids;
            let mut all: Vec<(f32, usize)> = (0..data.len())
                .map(|gid| {
                    let codes = idx.code_row(gid);
                    let d = crate::quant::pq::adc_distance(&table, codes, m, kc);
                    (d, gid)
                })
                .collect();
            all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let want: Vec<usize> = all.into_iter().take(5).map(|(_, id)| id).collect();
            assert_eq!(got, want, "top-k selection diverged from full sort");
        }
    }

    #[test]
    fn search_batch_matches_single_query() {
        // The batched GEMM coarse scan must agree with the per-query path. Both
        // pick the n_probe highest-similarity centroids and ADC-rank identically;
        // f32 summation order differs between matrixmultiply and the SIMD dot, so
        // a handful of queries may break a centroid tie differently. Require
        // top-1 identical and ≥ k-1 of k overlap — looser than bit-equality, but
        // it catches any real divergence in the coarse selection or ADC path.
        let data = random_unit_vecs(2_000, 64, 21);
        let mut idx = IvfPqIndex::new(64, 8);
        idx.train(&data, 8, 64, 12, 21).unwrap();
        for v in &data {
            idx.add(v);
        }
        let k = 10;
        let flat: Vec<f32> = data.iter().take(50).flatten().copied().collect();
        let batch = idx.search_batch_flat(&flat, 64, k, 8);
        for (i, brow) in batch.iter().enumerate() {
            let single = idx.search_with_probe(&data[i], k, 8);
            assert_eq!(brow.len(), single.len(), "row {i} length differs");
            assert_eq!(brow[0].0, single[0].0, "row {i} top-1 differs");
            let bset: std::collections::HashSet<usize> = brow.iter().map(|&(id, _)| id).collect();
            let overlap = single.iter().filter(|&&(id, _)| bset.contains(&id)).count();
            assert!(
                overlap >= k - 1,
                "row {i}: batch/single overlap {overlap}/{k} too low"
            );
        }
    }

    #[test]
    fn search_batch_empty_and_untrained() {
        // Untrained index → one empty result per query, never a panic.
        let idx = IvfPqIndex::new(8, 4);
        let flat = vec![0.0f32; 3 * 16];
        let res = idx.search_batch_flat(&flat, 16, 5, 4);
        assert_eq!(res.len(), 3);
        assert!(res.iter().all(|r| r.is_empty()));
    }

    #[test]
    fn search_self_nearest_full_probe() {
        let data = random_unit_vecs(200, 32, 3);
        let mut idx = IvfPqIndex::new(8, 8); // n_probe = n_lists → full scan
        idx.train(&data, 4, 8, 10, 3).unwrap();
        for v in &data {
            idx.add(v);
        }
        // With full probe and PQ, the query vector itself should be top-1 most of the time
        let mut hits = 0usize;
        for (i, v) in data.iter().enumerate().take(20) {
            let res = idx.search(v, 1);
            if !res.is_empty() && res[0].0 == i {
                hits += 1;
            }
        }
        // PQ introduces quantisation noise; expect ≥ 80% self-recall
        assert!(
            hits >= 14,
            "only {hits}/20 self-nearest hits with full probe"
        );
    }

    #[test]
    fn delete_excludes_from_search() {
        let data = random_unit_vecs(100, 32, 5);
        let mut idx = IvfPqIndex::new(4, 4);
        idx.train(&data, 4, 8, 10, 5).unwrap();
        let ids: Vec<usize> = data.iter().map(|v| idx.add(v)).collect();
        // Delete all vectors
        for &id in &ids {
            idx.delete(id);
        }
        let res = idx.search(&data[0], 5);
        assert!(res.is_empty(), "expected empty search after all deletes");
    }

    #[test]
    fn delete_out_of_bounds_no_panic() {
        let data = random_unit_vecs(50, 16, 99);
        let mut idx = IvfPqIndex::new(4, 2);
        idx.train(&data, 2, 4, 5, 99).unwrap();
        idx.delete(9999); // should not panic
    }

    #[test]
    fn save_load_roundtrip() {
        let data = random_unit_vecs(100, 32, 11);
        let mut idx = IvfPqIndex::new(4, 4);
        idx.train(&data, 4, 8, 5, 11).unwrap();
        for v in &data {
            idx.add(v);
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ivfpq.bin").to_string_lossy().into_owned();
        idx.save(&path).unwrap();
        let loaded = IvfPqIndex::load(&path).unwrap();
        assert_eq!(loaded.pq_codes.len(), idx.pq_codes.len());
        assert_eq!(loaded.dim, idx.dim);
        assert_eq!(loaded.n_lists, idx.n_lists);
    }

    #[test]
    fn train_errors_on_too_few_vecs() {
        let data = random_unit_vecs(3, 16, 0);
        let mut idx = IvfPqIndex::new(8, 2);
        let err = idx.train(&data, 2, 4, 5, 0);
        assert!(err.is_err());
    }

    #[test]
    fn untrained_add_panics() {
        let mut idx = IvfPqIndex::new(4, 2);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add(&[0.0f32; 16]);
        }));
        assert!(result.is_err(), "expected panic for untrained add");
    }

    #[test]
    fn vacuum_compacts_deleted_codes() {
        let data = random_unit_vecs(80, 32, 7);
        let mut idx = IvfPqIndex::new(4, 4);
        idx.train(&data, 4, 8, 10, 7).unwrap();
        for v in &data {
            idx.add(v);
        }
        for id in [0, 3, 7, 12, 20] {
            idx.delete(id);
        }
        let removed = idx.vacuum();
        assert_eq!(removed, 5, "vacuum should report 5 removed");
        // Total entries in posting lists should equal survivors.
        let total: usize = idx.posting_lists.iter().map(|l| l.len()).sum();
        assert_eq!(total, 75);
        assert!(!idx.deleted.iter().any(|&d| d));
        assert_eq!(idx.vacuum(), 0, "second vacuum is a no-op");
    }

    #[test]
    fn search_for_recall_returns_valid_probe() {
        let data = random_unit_vecs(200, 32, 99);
        let mut idx = IvfPqIndex::new(4, 4);
        idx.train(&data, 4, 8, 10, 99).unwrap();
        for v in &data {
            idx.add(v);
        }
        let (results, n_probe) = idx.search_for_recall(&data[0], 5, 0.8);
        assert!((1..=4).contains(&n_probe));
        assert!(!results.is_empty());
    }

    #[test]
    fn recall_reasonable() {
        // With n_probe = n_lists (full scan) recall@10 on 200 vecs / 4 lists should be ≥ 0.80
        let data = random_unit_vecs(200, 32, 17);
        let mut idx = IvfPqIndex::new(4, 4);
        idx.train(&data, 4, 8, 10, 17).unwrap();
        for v in &data {
            idx.add(v);
        }
        // Ground truth: each vector's nearest is itself (id == index in data)
        let queries: Vec<Vec<f32>> = data[..10].to_vec();
        let gt: Vec<Vec<usize>> = (0..10usize).map(|i| vec![i]).collect();
        let recall = idx.recall_at_k(&queries, &gt, 10, 4 /* full probe */);
        assert!(recall >= 0.70, "recall@10 = {recall}");
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn delete_never_returned(seed in 0u64..1000) {
            let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

            let mut data: Vec<Vec<f32>> = Vec::with_capacity(60);
            for _ in 0..60 {
                let v: Vec<f32> = (0..16).map(|_| {
                    state = lcg_next(state);
                    (state as i64 as f32) / i64::MAX as f32
                }).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                data.push(v.into_iter().map(|x| x / norm).collect());
            }

            let mut idx = IvfPqIndex::new(4, 4);
            prop_assume!(idx.train(&data, 2, 4, 5, seed).is_ok());
            let mut ids = Vec::new();
            for v in &data { ids.push(idx.add(v)); }

            // Delete every other vector
            let deleted: Vec<usize> = ids.iter().step_by(2).copied().collect();
            for &id in &deleted { idx.delete(id); }

            let results = idx.search(&data[0], data.len());
            for (id, _) in &results {
                prop_assert!(!deleted.contains(id), "deleted id {} appeared in results", id);
            }
        }

        #[test]
        fn adc_dist_non_negative(seed in 1u64..500) {
            let mut state = seed;
            let mut data: Vec<Vec<f32>> = Vec::with_capacity(40);
            for _ in 0..40 {
                let v: Vec<f32> = (0..16).map(|_| {
                    state = lcg_next(state);
                    (state as i64 as f32) / i64::MAX as f32
                }).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                data.push(v.into_iter().map(|x| x / norm).collect());
            }

            let mut idx = IvfPqIndex::new(4, 4);
            prop_assume!(idx.train(&data, 2, 4, 5, seed).is_ok());
            for v in &data { idx.add(v); }

            let results = idx.search(&data[0], 10);
            for (_, dist) in &results {
                prop_assert!(*dist >= 0.0, "ADC distance {} < 0", dist);
            }
        }
    }
}
