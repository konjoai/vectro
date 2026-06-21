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

use crate::quant::int8::encode_scalar_into;
use crate::quant::pq::{pq_distance_table, train_pq_codebook, PQCodebook};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;

// ─── helpers ──────────────────────────────────────────────────────────────────

/// LCG parameters identical to pq.rs / ivf.rs for reproducible init.
#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

/// Cosine distance (1 − cosine similarity) using SimSIMD.
#[inline]
fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = <f32 as SpatialSimilarity>::dot(a, b).unwrap_or(-1.0);
    (1.0 - dot as f32).max(0.0)
}

/// K-means++ initialisation — returns `k` centroids from `data`.
fn kmeans_pp_init(data: &[Vec<f32>], k: usize, d: usize, seed: u64) -> Vec<f32> {
    let n = data.len();
    let mut rng = seed;
    let mut centroids = Vec::with_capacity(k * d);

    // First centroid: random pick
    rng = lcg_next(rng);
    let first = (rng as usize) % n;
    centroids.extend_from_slice(&data[first]);

    for c_idx in 1..k {
        let chosen_centroids = &centroids[..c_idx * d];

        // Compute D² distances from each point to its nearest chosen centroid
        let dists: Vec<f32> = data
            .iter()
            .map(|v| {
                let mut best = f32::MAX;
                for ci in 0..c_idx {
                    let cent = &chosen_centroids[ci * d..(ci + 1) * d];
                    let dist = cosine_dist(v, cent);
                    if dist < best {
                        best = dist;
                    }
                }
                best * best
            })
            .collect();

        let total: f32 = dists.iter().sum();
        rng = lcg_next(rng);
        let mut target = (rng as f64 / u64::MAX as f64) as f32 * total;
        let mut picked = n - 1;
        for (i, &d2) in dists.iter().enumerate() {
            target -= d2;
            if target <= 0.0 {
                picked = i;
                break;
            }
        }
        centroids.extend_from_slice(&data[picked]);
    }
    centroids
}

/// Lloyd's k-means over `data` (unit-norm vectors, cosine distance).
///
/// Returns centroids as a flat `[k * d]` slice.
fn kmeans_lloyd(
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
    /// PQ codes keyed by global id, length M each.
    pq_codes: Vec<Vec<u8>>,
    /// Soft-deletion tombstones.
    #[serde(default)]
    deleted: Vec<bool>,
    /// Vector dimension.
    dim: usize,
    /// True after `train` succeeds.
    trained: bool,
    /// Opt-in rerank store: unit-normalised vectors kept as **INT8** (1 byte/dim,
    /// fixed 1/127 scale) alongside the PQ codes, so `search_rerank` can re-score
    /// PQ candidates with near-exact cosine. PQ alone tops out ~0.6 recall on hard
    /// data; reranking a wider pool (`k·k_factor`) against these codes lifts it to
    /// 0.95+ at 1 byte/dim — a quarter of the fp32 cost, which is what makes a
    /// high-recall 100M index fit on commodity RAM. Empty unless enabled.
    /// Per-vector **abs-max** scaling (not a fixed scale) so the full int8 range
    /// is used even for small-magnitude normalised components — near-fp32 recall.
    #[serde(default)]
    rerank_codes: Vec<i8>,
    /// Per-vector decode scale (`abs_max / 127`) for `rerank_codes`.
    #[serde(default)]
    rerank_scales: Vec<f32>,
    /// Whether `rerank_vectors` is being populated (set before the first add).
    #[serde(default)]
    rerank: bool,
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
            rerank_codes: Vec::new(),
            rerank_scales: Vec::new(),
            rerank: false,
        }
    }

    /// Enable the exact-rerank store. Must be called **before** any `add`, since
    /// it changes what each insert records. Trades memory (`N·dim·4` bytes for the
    /// normalised vectors) for a large recall gain via [`search_rerank`].
    pub fn enable_rerank(&mut self) {
        assert!(self.pq_codes.is_empty(), "enable_rerank must be called before adding vectors");
        self.rerank = true;
    }

    /// Whether the rerank store is active.
    pub fn has_rerank(&self) -> bool {
        self.rerank
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

        // --- Train PQ codebook on coarse residuals (not raw vectors) ---
        // Each training vector contributes `v − centroid[assigned]`; the codebook
        // then models the in-cell residual distribution, which is much tighter and
        // gives substantially higher recall at the same byte budget.
        let residuals: Vec<Vec<f32>> = normed
            .iter()
            .map(|v| {
                let list = self.nearest_coarse(v);
                self.residual(v, list)
            })
            .collect();
        self.codebook = train_pq_codebook(&residuals, n_subspaces, n_centroids, max_iter, seed)
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

        // PQ-encode the residual against the assigned centroid (see `residual`).
        let codes = self.pq_encode_single(&self.residual(&v_norm, list_id));

        let global_id = self.pq_codes.len();
        self.pq_codes.push(codes);
        self.deleted.push(false);
        self.posting_lists[list_id].push(global_id);
        if self.rerank {
            let mut codes = vec![0i8; self.dim];
            // abs-max int8 (range_factor 1.0): scale = abs_max; decode = code·scale/127.
            let scale = encode_scalar_into(&v_norm, &mut codes, 1.0);
            self.rerank_codes.extend_from_slice(&codes);
            self.rerank_scales.push(scale / 127.0);
        }
        global_id
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

        // Find top-n_probe coarse centroids
        let probe_lists = self.top_coarse(&q_norm, n_probe);

        let m = self.codebook.n_subspaces;
        let kc = self.codebook.n_centroids;

        // Scan posting lists — collect (dist, id) pairs. The ADC table is rebuilt
        // **per list** from the query residual `q − centroid[list]`, since codes
        // are residual-encoded (the residual depends on which cell we are in).
        let mut candidates: Vec<(f32, usize)> = Vec::new();

        for &list_id in &probe_lists {
            let q_res = self.residual(&q_norm, list_id);
            let dist_table = pq_distance_table(&q_res, &self.codebook);
            for &gid in &self.posting_lists[list_id] {
                if self.deleted[gid] {
                    continue;
                }
                let codes = &self.pq_codes[gid];
                let adc_dist: f32 = (0..m)
                    .map(|mi| dist_table[mi * kc + codes[mi] as usize])
                    .sum();
                candidates.push((adc_dist, gid));
            }
        }

        // Sort ascending by ADC distance, deduplicate, take top-k
        candidates.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.dedup_by_key(|(_, id)| *id);
        candidates.truncate(k);
        candidates.into_iter().map(|(d, id)| (id, d)).collect()
    }

    /// Search with PQ candidate generation **plus exact rerank**.
    ///
    /// Retrieves `k · k_factor` candidates by ADC (cheap, approximate), then
    /// re-scores them with the true cosine distance from the [`enable_rerank`]
    /// store and returns the best `k`. This recovers the recall PQ alone loses
    /// (≈0.6 → 0.95+); `k_factor` trades a wider, more accurate candidate pool
    /// for a little more rerank work. Requires `enable_rerank` before adding.
    pub fn search_rerank(
        &self,
        query: &[f32],
        k: usize,
        n_probe: usize,
        k_factor: usize,
    ) -> Vec<(usize, f32)> {
        if !self.rerank || self.pq_codes.is_empty() {
            // No rerank store — fall back to plain PQ search.
            return self.search_with_probe(query, k, n_probe);
        }
        let pool = (k * k_factor.max(1)).max(k);
        let cands = self.search_with_probe(query, pool, n_probe);

        let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let q_norm: Vec<f32> = query.iter().map(|x| x / norm).collect();
        let d = self.dim;
        // Re-score against the INT8 rerank codes: cosine ≈ 1 − scaleᵢ·Σ qⱼ·codeⱼ
        // (stored vectors are unit-norm, so a plain dot is the cosine).
        let mut rescored: Vec<(f32, usize)> = cands
            .into_iter()
            .map(|(gid, _)| {
                let codes = &self.rerank_codes[gid * d..(gid + 1) * d];
                let dot: f32 = q_norm.iter().zip(codes).map(|(q, &c)| q * (c as f32)).sum::<f32>()
                    * self.rerank_scales[gid];
                ((1.0 - dot).max(0.0), gid)
            })
            .collect();
        rescored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        rescored.truncate(k);
        rescored.into_iter().map(|(dist, id)| (id, dist)).collect()
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
                let found = results
                    .iter()
                    .filter(|(id, _)| gt.contains(id))
                    .count();
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
        let mut mapping: Vec<Option<usize>> = Vec::with_capacity(self.pq_codes.len());
        let mut new_id = 0usize;
        for &del in &self.deleted {
            if del {
                mapping.push(None);
            } else {
                mapping.push(Some(new_id));
                new_id += 1;
            }
        }

        // Compact PQ codes.
        let new_codes: Vec<Vec<u8>> = self
            .pq_codes
            .iter()
            .zip(self.deleted.iter())
            .filter(|(_, &d)| !d)
            .map(|(c, _)| c.clone())
            .collect();

        // Remap posting lists.
        for list in &mut self.posting_lists {
            *list = list.iter().filter_map(|&id| mapping[id]).collect();
        }

        // Compact the rerank store in lockstep (keyed by global id).
        if self.rerank {
            let d = self.dim;
            let mut new_rr = Vec::with_capacity(new_codes.len() * d);
            let mut new_sc = Vec::with_capacity(new_codes.len());
            for (gid, &del) in self.deleted.iter().enumerate() {
                if !del {
                    new_rr.extend_from_slice(&self.rerank_codes[gid * d..(gid + 1) * d]);
                    new_sc.push(self.rerank_scales[gid]);
                }
            }
            self.rerank_codes = new_rr;
            self.rerank_scales = new_sc;
        }

        self.pq_codes = new_codes;
        self.deleted = vec![false; self.pq_codes.len()];
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
    /// Residual of a (normalised) vector against its assigned coarse centroid:
    /// `v − centroid[list]`. PQ encodes this residual rather than the raw vector,
    /// so the codebook only has to model the spread *within* a coarse cell — far
    /// less variance than the whole space, which is what lifts recall.
    #[inline]
    fn residual(&self, v: &[f32], list_id: usize) -> Vec<f32> {
        let d = self.dim;
        let cent = &self.coarse_centroids[list_id * d..(list_id + 1) * d];
        v.iter().zip(cent).map(|(a, b)| a - b).collect()
    }

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
        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(n_probe.min(self.n_lists))
            .map(|(_, ci)| ci)
            .collect()
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
        assert_eq!(idx.pq_codes.len(), 200);
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
    fn rerank_beats_plain_pq_recall() {
        // Exact rerank of a wider PQ candidate pool must recover recall that plain
        // ADC loses. Compare self-recall@1 over full probe.
        let data = random_unit_vecs(800, 32, 11);
        let mut idx = IvfPqIndex::new(16, 16);
        idx.enable_rerank();
        idx.train(&data, 8, 16, 12, 11).unwrap();
        for v in &data {
            idx.add(v);
        }
        let n = 80;
        let plain = (0..n).filter(|&i| idx.search(&data[i], 1).first().map(|r| r.0) == Some(i)).count();
        let rr = (0..n)
            .filter(|&i| idx.search_rerank(&data[i], 1, 16, 16).first().map(|r| r.0) == Some(i))
            .count();
        assert!(rr >= plain, "rerank {rr} should be ≥ plain {plain}");
        assert!(rr >= n * 9 / 10, "rerank self-recall {rr}/{n} too low");
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
