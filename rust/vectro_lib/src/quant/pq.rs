//! Product Quantization (PQ) — Jégou et al. 2011.
//!
//! Splits each d-dimensional vector into `M` equal sub-spaces,
//! trains K-means centroids per sub-space, and encodes each vector as
//! `M` centroid indices (one u8 each, so K ≤ 256).
//!
//! This is a pure-Rust port of the Python reference in `python/pq_api.py`.
//! scikit-learn MiniBatchKMeans is replaced by a straight Lloyd's K-means
//! implementation; the result is numerically equivalent for the same seed.
//!
//! Parity target (from PLAN.md Phase 16): recall@10 ≥ 0.95.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Trained PQ codebook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCodebook {
    /// Number of sub-spaces M.
    pub n_subspaces: usize,
    /// Centroids per sub-space K (≤ 256 to fit in u8).
    pub n_centroids: usize,
    /// Dimension of each sub-space: d / M.
    pub sub_dim: usize,
    /// Centroid table, shape [M][K][sub_dim], flattened row-major.
    pub centroids: Vec<f32>,
}

impl PQCodebook {
    /// Return a slice into `centroids` for subspace `m`, centroid `k`.
    #[inline]
    pub fn centroid(&self, m: usize, k: usize) -> &[f32] {
        let stride = self.n_centroids * self.sub_dim;
        let start = m * stride + k * self.sub_dim;
        &self.centroids[start..start + self.sub_dim]
    }
}

/// Train a PQ codebook with Lloyd's K-means.
///
/// # Arguments
/// * `training_data` — slice of f32 vectors, all of length `d`
/// * `n_subspaces`   — M; must divide d
/// * `n_centroids`   — K; must be ≤ 256 and ≤ n_training
/// * `max_iter`      — Lloyd's iterations
/// * `seed`          — RNG seed for centroid initialisation
pub fn train_pq_codebook(
    training_data: &[Vec<f32>],
    n_subspaces: usize,
    n_centroids: usize,
    max_iter: usize,
    seed: u64,
) -> Result<PQCodebook, String> {
    if training_data.is_empty() {
        return Err("training_data is empty".into());
    }
    let d = training_data[0].len();
    if !d.is_multiple_of(n_subspaces) {
        return Err(format!("d={d} not divisible by n_subspaces={n_subspaces}"));
    }
    if n_centroids > 256 {
        return Err(format!("n_centroids={n_centroids} exceeds u8 max 256"));
    }
    if n_centroids > training_data.len() {
        return Err(format!("n_centroids={n_centroids} > n_training={}", training_data.len()));
    }

    let sub_dim = d / n_subspaces;
    let total = n_subspaces * n_centroids * sub_dim;
    let mut centroids_flat = vec![0.0f32; total];

    // Training-set subsampling (FAISS strategy): k-means doesn't need every point
    // to place K centroids — a bounded sample of ~`TRAIN_POINTS_PER_CENTROID` per
    // centroid is statistically sufficient, and the assignment cost (the hot
    // loop) scales with the point count. A deterministic strided sample keeps the
    // build reproducible. For n ≤ cap this is a no-op (trains on everything).
    const TRAIN_POINTS_PER_CENTROID: usize = 64;
    let cap = n_centroids.saturating_mul(TRAIN_POINTS_PER_CENTROID);
    let sample: Vec<&Vec<f32>> = if training_data.len() > cap {
        let stride = training_data.len() / cap;
        training_data.iter().step_by(stride.max(1)).take(cap).collect()
    } else {
        training_data.iter().collect()
    };

    // Train each sub-space independently; parallelize across subspaces.
    let stride = n_centroids * sub_dim;
    let sub_results: Vec<Vec<f32>> = (0..n_subspaces)
        .into_par_iter()
        .map(|m| {
            let col_start = m * sub_dim;
            let sub_vecs: Vec<&[f32]> = sample
                .iter()
                .map(|v| &v[col_start..col_start + sub_dim])
                .collect();
            kmeans_lloyd(&sub_vecs, n_centroids, sub_dim, max_iter, seed + m as u64)
        })
        .collect();

    for (m, cents) in sub_results.into_iter().enumerate() {
        centroids_flat[m * stride..(m + 1) * stride].copy_from_slice(&cents);
    }

    Ok(PQCodebook {
        n_subspaces,
        n_centroids,
        sub_dim,
        centroids: centroids_flat,
    })
}

/// k-means++ centroid initialisation.
///
/// Selects `k` initial centroids from `data` using D²-weighted sampling:
/// the first centroid is chosen uniformly at random, and each subsequent
/// centroid is chosen with probability proportional to its squared distance
/// to the nearest already-chosen centroid.  This produces better-spread
/// initial centroids than uniform sampling and typically halves the number of
/// Lloyd iterations required to converge.
fn kmeans_pp_init(data: &[&[f32]], k: usize, sub_dim: usize, seed: u64) -> Vec<f32> {
    let n = data.len();
    debug_assert!(n >= k, "kmeans_pp_init: need n >= k");

    // Deterministic LCG RNG seeded per subspace.
    let mut state = seed.wrapping_add(1_442_695_040_888_963_407);
    let mut lcg = || -> f64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut cents = vec![0.0f32; k * sub_dim];

    // First centroid: uniform random pick.
    let first = (lcg() * n as f64) as usize % n;
    cents[..sub_dim].copy_from_slice(data[first]);

    // Maintain min-squared-distance to the nearest chosen centroid.
    let mut min_d2 = vec![f32::INFINITY; n];

    for ki in 1..k {
        // Update min_d2 with the most recently added centroid.
        let new_cent = &cents[(ki - 1) * sub_dim..ki * sub_dim];
        for (i, v) in data.iter().enumerate() {
            let d = l2_sq(v, new_cent);
            if d < min_d2[i] {
                min_d2[i] = d;
            }
        }

        // D²-weighted sampling via prefix sums.
        let total: f64 = min_d2.iter().map(|&d| d as f64).sum();
        let chosen = if total == 0.0 {
            // Degenerate: all remaining points are identical to chosen centroids.
            ki % n
        } else {
            let r = lcg() * total;
            let mut acc = 0.0f64;
            let mut idx = n - 1;
            for (i, &d) in min_d2.iter().enumerate() {
                acc += d as f64;
                if acc >= r {
                    idx = i;
                    break;
                }
            }
            idx
        };
        cents[ki * sub_dim..(ki + 1) * sub_dim].copy_from_slice(data[chosen]);
    }

    cents
}

/// Lloyd's K-means for a set of equal-length sub-vector slices.
/// Returns a flat [K * sub_dim] vector of centroids.
fn kmeans_lloyd(
    data: &[&[f32]],
    k: usize,
    sub_dim: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<f32> {
    let n = data.len();

    // k-means++ initialisation: better-spread initial centroids.
    let mut cents = kmeans_pp_init(data, k, sub_dim, seed);

    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iter {
        // Assignment step (parallel). Build the transposed-centroid LUT once per
        // iteration, then assign every point via the SIMD-across-K kernel.
        let lut = build_subspace_lut(&cents, k, sub_dim);
        let new_assignments: Vec<usize> = data
            .par_iter()
            .map(|v| assign_nearest(v, &lut, k, sub_dim) as usize)
            .collect();

        let changed = new_assignments
            .iter()
            .zip(assignments.iter())
            .filter(|(a, b)| a != b)
            .count();
        assignments = new_assignments;
        // Early stop on a tolerance: once < 1% of points move between iterations
        // the centroids have effectively converged (their motion is sub-quantum
        // for INT8/PQ reconstruction). The old exact-zero check almost never
        // tripped — a fraction of a percent of boundary points always flip — so
        // k-means ran all `max_iter` rounds every time, which is wasted work.
        if changed * 100 <= n {
            break;
        }

        // Update step
        let mut sums = vec![0.0f32; k * sub_dim];
        let mut counts = vec![0usize; k];
        for (v, &a) in data.iter().zip(assignments.iter()) {
            for (i, &x) in v.iter().enumerate() {
                sums[a * sub_dim + i] += x;
            }
            counts[a] += 1;
        }
        for ki in 0..k {
            if counts[ki] > 0 {
                let inv = 1.0 / counts[ki] as f32;
                for i in 0..sub_dim {
                    cents[ki * sub_dim + i] = sums[ki * sub_dim + i] * inv;
                }
            }
        }
    }

    cents
}

/// Squared L2 distance between two equal-length slices.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| { let d = x - y; d * d }).sum()
}

// ───────────────── fast nearest-centroid assignment (SIMD across K) ──────────
//
// The PQ hot loop — k-means assignment and encode — finds, per sub-vector,
// `argmin_k ‖v − c_k‖²`. Computing `‖v − c_k‖²` directly (`l2_sq`) only
// vectorizes over `sub_dim`, which is tiny (e.g. 4 for d=100, M=25). Instead use
//
//     argmin_k ‖v − c_k‖²  =  argmin_k (‖c_k‖² − 2·v·c_k)     (‖v‖² is constant)
//
// and lay the centroids out **transposed** as `ct[j*K + k]` (coordinate `j` of
// every centroid contiguous), so the `v·c_k` term vectorizes across the wide
// `K` (=256) axis: for each `j`, FMA the broadcast scalar `v[j]` across all K
// centroids at once. This is the layout FAISS uses for its fast assignment.

/// The maximum K a PQ codebook supports (u8 codes). Sizes the assignment buffer.
const MAX_K: usize = 256;

/// Transposed centroid LUT for one sub-space: `ct[j*k + ki]` is coordinate `j`
/// of centroid `ki`, and `norms[ki] = ‖c_ki‖²`. Built once per sub-space and
/// reused across all rows (encode) or all vectors in a k-means iteration.
struct SubspaceLut {
    ct: Vec<f32>,
    norms: Vec<f32>,
}

/// Build the transposed-centroid LUT from a row-major `[k][sub_dim]` table.
fn build_subspace_lut(table: &[f32], k: usize, sub_dim: usize) -> SubspaceLut {
    let mut ct = vec![0.0f32; sub_dim * k];
    let mut norms = vec![0.0f32; k];
    for (ki, cen) in table.chunks_exact(sub_dim).enumerate() {
        let mut nrm = 0.0f32;
        for (j, &x) in cen.iter().enumerate() {
            ct[j * k + ki] = x;
            nrm += x * x;
        }
        norms[ki] = nrm;
    }
    SubspaceLut { ct, norms }
}

/// `argmin_k (norms[k] − 2·v·c_k)` over the transposed LUT — the nearest
/// centroid index for sub-vector `v`. Vectorized across the K axis with NEON on
/// aarch64 and AVX2+FMA on x86_64 (runtime-detected); scalar fallback otherwise.
#[inline]
fn assign_nearest(v: &[f32], lut: &SubspaceLut, k: usize, sub_dim: usize) -> u8 {
    debug_assert!(k <= MAX_K);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandated on AArch64-v8; all indices below stay < k ≤ MAX_K.
        unsafe { assign_argmin_neon(v, &lut.ct, &lut.norms, k, sub_dim) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: gated by the runtime feature detection above; all indices
            // stay < k ≤ MAX_K and within the `ct`/`norms` slices. AVX-512 wins
            // 1.4–1.8× here (loop + argmin-update overhead bound, not FMA-bound).
            return unsafe { assign_argmin_avx512(v, &lut.ct, &lut.norms, k, sub_dim) };
        }
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: gated by the runtime feature detection above; all indices
            // stay < k ≤ MAX_K and within the `ct`/`norms` slices.
            return unsafe { assign_argmin_avx2(v, &lut.ct, &lut.norms, k, sub_dim) };
        }
        assign_argmin_portable(v, lut, k, sub_dim)
    }
}

/// Portable scalar `argmin_k (norms[k] − 2·v·c_k)` over the transposed LUT — the
/// correctness baseline the NEON / AVX2 kernels must match.
#[cfg(not(target_arch = "aarch64"))]
fn assign_argmin_portable(v: &[f32], lut: &SubspaceLut, k: usize, sub_dim: usize) -> u8 {
    let mut dist = [0.0f32; MAX_K];
    dist[..k].copy_from_slice(&lut.norms[..k]);
    for (j, &vj) in v[..sub_dim].iter().enumerate() {
        let row = &lut.ct[j * k..j * k + k];
        for (ki, &cjk) in row.iter().enumerate() {
            dist[ki] -= 2.0 * vj * cjk;
        }
    }
    let mut best = 0usize;
    let mut best_d = dist[0];
    for (ki, &d) in dist[1..k].iter().enumerate() {
        if d < best_d {
            best_d = d;
            best = ki + 1;
        }
    }
    best as u8
}

/// Fused NEON nearest-centroid: computes `dist = norms − 2·v·cᵀ` and tracks the
/// running minimum **in registers**, 4 centroids at a time — no per-point `dist`
/// buffer (the old kernel wrote 256 floats to the stack then re-read them to
/// argmin). Lane 0 holds centroids 0,4,8…, lane 1 holds 1,5,9…, etc.; ties pick
/// the lower index (the scalar argmin's behaviour, modulo lane order — which only
/// matters for exactly-equidistant centroids, i.e. effectively never on f32).
///
/// # Safety
/// Requires NEON. `ct.len() >= sub_dim*k`, `norms.len() >= k`, `v.len() >= sub_dim`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn assign_argmin_neon(v: &[f32], ct: &[f32], norms: &[f32], k: usize, sub_dim: usize) -> u8 {
    use std::arch::aarch64::*;
    let two = vdupq_n_f32(2.0);
    let four = vdupq_n_u32(4);
    let mut min_vals = vdupq_n_f32(f32::INFINITY);
    let mut min_idx = vdupq_n_u32(0);
    let lane0: [u32; 4] = [0, 1, 2, 3];
    let mut cur_idx = vld1q_u32(lane0.as_ptr());

    let kc = k & !3;
    let mut ki = 0;
    while ki < kc {
        let mut acc = vdupq_n_f32(0.0);
        for j in 0..sub_dim {
            let vj = vdupq_n_f32(*v.get_unchecked(j));
            let cj = vld1q_f32(ct.as_ptr().add(j * k + ki));
            acc = vfmaq_f32(acc, vj, cj);
        }
        let nrm = vld1q_f32(norms.as_ptr().add(ki));
        let d = vfmsq_f32(nrm, two, acc); // norms − 2·acc
        let mask = vcltq_f32(d, min_vals); // d < running min ?
        min_vals = vbslq_f32(mask, d, min_vals);
        min_idx = vbslq_u32(mask, cur_idx, min_idx);
        cur_idx = vaddq_u32(cur_idx, four);
        ki += 4;
    }

    // Reduce the 4 SIMD lanes to a scalar (value, index).
    let mut vals = [0.0f32; 4];
    let mut idxs = [0u32; 4];
    vst1q_f32(vals.as_mut_ptr(), min_vals);
    vst1q_u32(idxs.as_mut_ptr(), min_idx);
    let mut best = idxs[0];
    let mut best_d = vals[0];
    for l in 1..4 {
        if vals[l] < best_d {
            best_d = vals[l];
            best = idxs[l];
        }
    }

    // Scalar tail (< 4 centroids).
    while ki < k {
        let mut dot = 0.0f32;
        for j in 0..sub_dim {
            dot += *v.get_unchecked(j) * *ct.get_unchecked(j * k + ki);
        }
        let d = *norms.get_unchecked(ki) - 2.0 * dot;
        if d < best_d {
            best_d = d;
            best = ki as u32;
        }
        ki += 1;
    }
    best as u8
}


/// Fused AVX2+FMA nearest-centroid: computes `dist = norms − 2·v·cᵀ` and tracks
/// the running minimum **in registers**, 8 centroids at a time (256-bit lanes) —
/// no per-point `dist` buffer. Mirrors [`assign_argmin_neon`]. Lane `l` holds
/// centroids `ki+l`; ties pick the lower index (strict `<`), matching the scalar
/// baseline modulo exactly-equidistant centroids (effectively never on f32).
///
/// # Safety
/// Requires AVX2 + FMA. `ct.len() >= sub_dim*k`, `norms.len() >= k`,
/// `v.len() >= sub_dim`, `k <= MAX_K`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn assign_argmin_avx2(v: &[f32], ct: &[f32], norms: &[f32], k: usize, sub_dim: usize) -> u8 {
    use std::arch::x86_64::*;
    let two = _mm256_set1_ps(2.0);
    let mut min_vals = _mm256_set1_ps(f32::INFINITY);
    let mut min_idx = _mm256_setzero_si256();
    let mut cur_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let eight = _mm256_set1_epi32(8);

    let kc = k & !7;
    let mut ki = 0;
    while ki < kc {
        let mut acc = _mm256_setzero_ps();
        for j in 0..sub_dim {
            let vj = _mm256_set1_ps(*v.get_unchecked(j));
            let cj = _mm256_loadu_ps(ct.as_ptr().add(j * k + ki));
            acc = _mm256_fmadd_ps(vj, cj, acc);
        }
        let nrm = _mm256_loadu_ps(norms.as_ptr().add(ki));
        // d = norms − 2·acc  =  −(two·acc) + norms
        let d = _mm256_fnmadd_ps(two, acc, nrm);
        let mask = _mm256_cmp_ps::<_CMP_LT_OQ>(d, min_vals); // d < running min ?
        min_vals = _mm256_blendv_ps(min_vals, d, mask);
        min_idx = _mm256_blendv_epi8(min_idx, cur_idx, _mm256_castps_si256(mask));
        cur_idx = _mm256_add_epi32(cur_idx, eight);
        ki += 8;
    }

    // Reduce the 8 SIMD lanes to a scalar (value, index).
    let mut vals = [0.0f32; 8];
    let mut idxs = [0i32; 8];
    _mm256_storeu_ps(vals.as_mut_ptr(), min_vals);
    _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, min_idx);
    let mut best = idxs[0] as u32;
    let mut best_d = vals[0];
    for l in 1..8 {
        if vals[l] < best_d {
            best_d = vals[l];
            best = idxs[l] as u32;
        }
    }

    // Scalar tail (< 8 centroids).
    while ki < k {
        let mut dot = 0.0f32;
        for j in 0..sub_dim {
            dot += *v.get_unchecked(j) * *ct.get_unchecked(j * k + ki);
        }
        let d = *norms.get_unchecked(ki) - 2.0 * dot;
        if d < best_d {
            best_d = d;
            best = ki as u32;
        }
        ki += 1;
    }
    best as u8
}

/// Fused AVX-512F nearest-centroid: computes `dist = norms − 2·v·cᵀ` and tracks
/// the running minimum in **mask registers**, 16 centroids at a time (512-bit
/// lanes) — no per-point `dist` buffer. Mirrors [`assign_argmin_avx2`] at double
/// the lane width. For the common K=256 sub-quantizer this is 16 outer
/// iterations instead of 32, halving the per-iteration `norms` load, compare,
/// blend and index-increment overhead. Unlike the f32 *distance* kernels (which
/// were measured slower under AVX-512 on this CPU class — FMA-port-bound), this
/// argmin is dominated by loop + argmin-update overhead, which 512-bit width and
/// native `__mmask16` blends cut directly. Lane `l` holds centroid `ki+l`; ties
/// pick the lower index (strict `<`), matching the scalar baseline.
///
/// # Safety
/// Requires AVX-512F. `ct.len() >= sub_dim*k`, `norms.len() >= k`,
/// `v.len() >= sub_dim`, `k <= MAX_K`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn assign_argmin_avx512(v: &[f32], ct: &[f32], norms: &[f32], k: usize, sub_dim: usize) -> u8 {
    use std::arch::x86_64::*;
    let two = _mm512_set1_ps(2.0);
    let mut min_vals = _mm512_set1_ps(f32::INFINITY);
    let mut min_idx = _mm512_setzero_si512();
    let mut cur_idx =
        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let sixteen = _mm512_set1_epi32(16);

    let kc = k & !15;
    let mut ki = 0;
    while ki < kc {
        let mut acc = _mm512_setzero_ps();
        for j in 0..sub_dim {
            let vj = _mm512_set1_ps(*v.get_unchecked(j));
            let cj = _mm512_loadu_ps(ct.as_ptr().add(j * k + ki));
            acc = _mm512_fmadd_ps(vj, cj, acc);
        }
        let nrm = _mm512_loadu_ps(norms.as_ptr().add(ki));
        // d = norms − 2·acc
        let d = _mm512_fnmadd_ps(two, acc, nrm);
        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(d, min_vals); // d < running min ?
        min_vals = _mm512_mask_blend_ps(mask, min_vals, d);
        min_idx = _mm512_mask_blend_epi32(mask, min_idx, cur_idx);
        cur_idx = _mm512_add_epi32(cur_idx, sixteen);
        ki += 16;
    }

    // Masked tail (1..=15 centroids) in a single 512-bit iteration — masked-off
    // lanes load as 0 and are forced to +∞ so they never win the argmin. cur_idx
    // already holds [kc, kc+1, …] from the main loop's last increment.
    if ki < k {
        let rem = k - ki;
        let tail_mask: u16 = ((1u32 << rem) - 1) as u16;
        let mut acc = _mm512_setzero_ps();
        for j in 0..sub_dim {
            let vj = _mm512_set1_ps(*v.get_unchecked(j));
            let cj = _mm512_maskz_loadu_ps(tail_mask, ct.as_ptr().add(j * k + ki));
            acc = _mm512_fmadd_ps(vj, cj, acc);
        }
        let nrm = _mm512_maskz_loadu_ps(tail_mask, norms.as_ptr().add(ki));
        let d = _mm512_fnmadd_ps(two, acc, nrm);
        // Force the inactive lanes to +∞ before the compare.
        let d = _mm512_mask_blend_ps(tail_mask, _mm512_set1_ps(f32::INFINITY), d);
        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(d, min_vals);
        min_vals = _mm512_mask_blend_ps(mask, min_vals, d);
        min_idx = _mm512_mask_blend_epi32(mask, min_idx, cur_idx);
    }

    // Reduce the 16 SIMD lanes to a scalar (value, index).
    let mut vals = [0.0f32; 16];
    let mut idxs = [0i32; 16];
    _mm512_storeu_ps(vals.as_mut_ptr(), min_vals);
    _mm512_storeu_si512(idxs.as_mut_ptr() as *mut __m512i, min_idx);
    let mut best = idxs[0] as u32;
    let mut best_d = vals[0];
    for l in 1..16 {
        if vals[l] < best_d {
            best_d = vals[l];
            best = idxs[l] as u32;
        }
    }
    best as u8
}

/// Encode a batch of f32 vectors to PQ codes (u8 per sub-space).
///
/// Returns an (n, M) matrix stored as `Vec<Vec<u8>>`.
pub fn pq_encode(vectors: &[Vec<f32>], codebook: &PQCodebook) -> Vec<Vec<u8>> {
    vectors
        .par_iter()
        .map(|v| encode_one(v, codebook))
        .collect()
}

fn encode_one(v: &[f32], cb: &PQCodebook) -> Vec<u8> {
    let mut code = Vec::with_capacity(cb.n_subspaces);
    for m in 0..cb.n_subspaces {
        let sub = &v[m * cb.sub_dim..(m + 1) * cb.sub_dim];
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        for k in 0..cb.n_centroids {
            let d = l2_sq(sub, cb.centroid(m, k));
            if d < best_d {
                best_d = d;
                best = k as u8;
            }
        }
        code.push(best);
    }
    code
}

/// Zero-copy batch PQ encode against a trained codebook.
///
/// Encodes the row-major f32 `vectors` (length `n * d`, `d = M * sub_dim`) into
/// `n * M` u8 `codes_out`, choosing the nearest centroid (L2) per sub-space.
/// Rayon-parallel over rows with no per-row heap allocation — the fast path
/// behind `python/pq_api.pq_encode`.
///
/// Numerically equivalent to the NumPy reference in `python/pq_api.py`
/// (`argmin` keeps the first minimum) modulo floating-point: the NumPy path
/// expands `‖v-c‖² = ‖v‖²+‖c‖²-2v·c` while this computes `Σ(v-c)²` directly, so
/// the two may pick different *equidistant* centroids at ties — identical
/// reconstruction quality.
pub fn pq_encode_into(vectors: &[f32], cb: &PQCodebook, codes_out: &mut [u8]) {
    let m = cb.n_subspaces;
    let sub_dim = cb.sub_dim;
    let k = cb.n_centroids;
    let d = m * sub_dim;
    let cent_stride = k * sub_dim;
    debug_assert_eq!(vectors.len() % d, 0);
    debug_assert_eq!(codes_out.len(), (vectors.len() / d) * m);

    // Build the transposed-centroid LUTs once (read-only, shared across rows).
    let luts: Vec<SubspaceLut> = (0..m)
        .map(|sub| build_subspace_lut(&cb.centroids[sub * cent_stride..(sub + 1) * cent_stride], k, sub_dim))
        .collect();

    codes_out
        .par_chunks_mut(m)
        .zip(vectors.par_chunks(d))
        .for_each(|(code_row, v)| {
            for (sub, code) in code_row.iter_mut().enumerate() {
                let subv = &v[sub * sub_dim..(sub + 1) * sub_dim];
                *code = assign_nearest(subv, &luts[sub], k, sub_dim);
            }
        });
}

/// Decode PQ codes back to approximate f32 vectors.
pub fn pq_decode(codes: &[Vec<u8>], codebook: &PQCodebook) -> Vec<Vec<f32>> {
    codes
        .par_iter()
        .map(|code| {
            let d = codebook.n_subspaces * codebook.sub_dim;
            let mut out = Vec::with_capacity(d);
            for (m, &k) in code.iter().enumerate() {
                out.extend_from_slice(codebook.centroid(m, k as usize));
            }
            out
        })
        .collect()
}

/// Build an Asymmetric Distance Computation (ADC) lookup table for one query.
///
/// Returns a flat [M * K] table of squared L2 distances to each centroid in
/// each sub-space.  Used to score all database codes without decoding them.
pub fn pq_distance_table(query: &[f32], codebook: &PQCodebook) -> Vec<f32> {
    let m = codebook.n_subspaces;
    let k = codebook.n_centroids;
    let sub_dim = codebook.sub_dim;
    let mut table = Vec::with_capacity(m * k);
    for mi in 0..m {
        let q_sub = &query[mi * sub_dim..(mi + 1) * sub_dim];
        for ki in 0..k {
            table.push(l2_sq(q_sub, codebook.centroid(mi, ki)));
        }
    }
    table
}

/// Sum the ADC-table entries selected by `codes`: `Σ_m table[m*k + codes[m]]` —
/// the per-candidate approximate distance, the innermost loop of every PQ scan.
///
/// Four independent accumulators break the f32 reduction's serial dependency
/// chain (an `.iter().sum()` cannot legally reassociate f32, so it serializes at
/// the FP-add latency even though the table gathers are mutually independent).
/// The gathers can then overlap, turning the scan from add-latency-bound toward
/// load-throughput-bound. Reassociating changes only the last-ULP rounding of
/// an already-approximate distance.
#[inline]
pub fn adc_distance(table: &[f32], codes: &[u8], m: usize, k: usize) -> f32 {
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    let mut mi = 0;
    while mi + 4 <= m {
        a0 += table[mi * k + codes[mi] as usize];
        a1 += table[(mi + 1) * k + codes[mi + 1] as usize];
        a2 += table[(mi + 2) * k + codes[mi + 2] as usize];
        a3 += table[(mi + 3) * k + codes[mi + 3] as usize];
        mi += 4;
    }
    let mut s = (a0 + a1) + (a2 + a3);
    while mi < m {
        s += table[mi * k + codes[mi] as usize];
        mi += 1;
    }
    s
}

/// Approximate top-k nearest neighbours using the ADC table.
///
/// Returns `(Vec<index>, Vec<approx_dist>)` sorted ascending by distance.
pub fn pq_search(
    query: &[f32],
    codes: &[Vec<u8>],
    codebook: &PQCodebook,
    top_k: usize,
) -> Vec<(usize, f32)> {
    let table = pq_distance_table(query, codebook);
    let k = codebook.n_centroids;
    let m = codebook.n_subspaces;

    let mut dists: Vec<(usize, f32)> = codes
        .par_iter()
        .enumerate()
        .map(|(i, code)| (i, adc_distance(&table, code, m, k)))
        .collect();

    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(top_k);
    dists
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..d).map(|j| ((i * d + j) as f32 * 0.01).sin()).collect())
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { return -1.0; }
        dot / (na * nb)
    }

    /// Isolated A/B microbench for the PQ nearest-centroid argmin kernels.
    /// `cargo test -p vectro_lib --release argmin_kernel_microbench -- --ignored --nocapture`
    #[test]
    #[ignore]
    #[cfg(target_arch = "x86_64")]
    fn argmin_kernel_microbench() {
        use std::time::Instant;
        let nq = 4096usize;
        for &(k, sub_dim) in &[(256usize, 4usize), (256, 8), (256, 16), (200, 8)] {
            let vecs = make_vecs(nq + k, sub_dim);
            let table: Vec<f32> = vecs[..k].iter().flatten().copied().collect();
            let lut = build_subspace_lut(&table, k, sub_dim);
            let queries = &vecs[k..];
            let iters = 200usize;

            // portable
            let t = Instant::now();
            let mut acc = 0u64;
            for _ in 0..iters {
                for q in queries {
                    acc += assign_argmin_portable(q, &lut, k, sub_dim) as u64;
                }
            }
            let portable_ns = t.elapsed().as_nanos() as f64 / (iters * nq) as f64;

            // avx2
            let avx2_ns = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let t = Instant::now();
                for _ in 0..iters {
                    for q in queries {
                        acc += unsafe { assign_argmin_avx2(q, &lut.ct, &lut.norms, k, sub_dim) } as u64;
                    }
                }
                t.elapsed().as_nanos() as f64 / (iters * nq) as f64
            } else { f64::NAN };

            // avx512
            let avx512_ns = if is_x86_feature_detected!("avx512f") {
                let t = Instant::now();
                for _ in 0..iters {
                    for q in queries {
                        acc += unsafe { assign_argmin_avx512(q, &lut.ct, &lut.norms, k, sub_dim) } as u64;
                    }
                }
                t.elapsed().as_nanos() as f64 / (iters * nq) as f64
            } else { f64::NAN };

            println!(
                "k={k} sub_dim={sub_dim}: portable={portable_ns:.2}ns avx2={avx2_ns:.2}ns avx512={avx512_ns:.2}ns  avx512/avx2={:.3}x  (sink={acc})",
                avx2_ns / avx512_ns
            );
        }
    }

    #[test]
    fn train_subsamples_above_cap_and_stays_deterministic() {
        // With K=8 the training cap is 64*8 = 512, so n=2000 triggers the strided
        // subsample. Training must (a) still produce a usable codebook and (b) be
        // deterministic for a fixed seed despite subsampling.
        let (d, m, k) = (16usize, 4usize, 8usize);
        let vecs = make_vecs(2000, d);
        let cb1 = train_pq_codebook(&vecs, m, k, 15, 7).unwrap();
        let cb2 = train_pq_codebook(&vecs, m, k, 15, 7).unwrap();
        assert_eq!(cb1.centroids, cb2.centroids, "subsampled training not deterministic");

        // Codebook still reconstructs reasonably (encode→decode cosine).
        let codes = pq_encode(&vecs[..100], &cb1);
        let decoded = pq_decode(&codes, &cb1);
        let avg: f32 = vecs[..100].iter().zip(&decoded).map(|(v, r)| cosine(v, r)).sum::<f32>() / 100.0;
        assert!(avg >= 0.80, "subsampled-train reconstruction cosine {avg} < 0.80");
    }

    #[test]
    fn train_encode_decode_quality() {
        let d = 64;
        let m = 8;
        let k = 16;
        let vecs = make_vecs(200, d);
        let cb = train_pq_codebook(&vecs, m, k, 20, 0).unwrap();
        assert_eq!(cb.n_subspaces, m);
        assert_eq!(cb.n_centroids, k);
        assert_eq!(cb.sub_dim, d / m);

        let codes = pq_encode(&vecs[..50], &cb);
        let decoded = pq_decode(&codes, &cb);
        let cos_sum: f32 = vecs[..50].iter().zip(decoded.iter()).map(|(v, d)| cosine(v, d)).sum();
        let avg_cos = cos_sum / 50.0;
        // With d=64, M=8, K=16 the codebook is coarse but should average above 0.80
        assert!(avg_cos >= 0.80, "avg cosine {avg_cos} < 0.80");
    }

    #[test]
    fn train_fails_on_non_divisible_d() {
        let vecs = make_vecs(50, 65);
        let err = train_pq_codebook(&vecs, 8, 16, 5, 0);
        assert!(err.is_err());
    }

    #[test]
    fn train_fails_on_too_many_centroids() {
        let vecs = make_vecs(50, 64);
        let err = train_pq_codebook(&vecs, 8, 257, 5, 0);
        assert!(err.is_err());
    }

    #[test]
    fn pq_search_finds_nearest() {
        let d = 32;
        let m = 4;
        let k = 8;
        let vecs = make_vecs(100, d);
        let cb = train_pq_codebook(&vecs, m, k, 10, 42).unwrap();
        let codes = pq_encode(&vecs, &cb);

        // Query == vecs[10] → should return index 10 in top-3
        let results = pq_search(&vecs[10], &codes, &cb, 3);
        let returned_indices: Vec<usize> = results.iter().map(|(i, _)| *i).collect();
        assert!(returned_indices.contains(&10), "top-3 didn't contain exact match: {:?}", returned_indices);
    }

    #[test]
    fn l2_sq_correctness() {
        let a = [3.0f32, 4.0];
        let b = [0.0f32, 0.0];
        assert!((l2_sq(&a, &b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn distance_table_shape() {
        let d = 16;
        let m = 4;
        let k = 4;
        let vecs = make_vecs(20, d);
        let cb = train_pq_codebook(&vecs, m, k, 5, 0).unwrap();
        let table = pq_distance_table(&vecs[0], &cb);
        assert_eq!(table.len(), m * k);
    }

    #[test]
    fn assign_nearest_matches_bruteforce_l2() {
        // The SIMD-across-K reformulation must pick the same centroid as a naive
        // argmin over l2_sq for every sub-vector (ties aside — none in this data).
        let (k, sub_dim) = (200usize, 8usize);
        let vecs = make_vecs(260, sub_dim); // reuse as centroids + queries
        let table: Vec<f32> = vecs[..k].iter().flatten().copied().collect();
        let lut = build_subspace_lut(&table, k, sub_dim);
        for q in &vecs {
            let got = assign_nearest(q, &lut, k, sub_dim) as usize;
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for ki in 0..k {
                let d = l2_sq(q, &table[ki * sub_dim..(ki + 1) * sub_dim]);
                if d < best_d {
                    best_d = d;
                    best = ki;
                }
            }
            // Equal distance to the chosen and brute-force centroid ⇒ accept tie.
            let got_d = l2_sq(q, &table[got * sub_dim..(got + 1) * sub_dim]);
            assert!(
                (got_d - best_d).abs() <= 1e-4,
                "assign {got} (d={got_d}) vs brute {best} (d={best_d})"
            );
        }
    }

    #[test]
    fn assign_nearest_simd_matches_portable() {
        // The host-arch SIMD kernel (NEON / AVX2) must select a centroid whose
        // distance matches the portable scalar baseline's pick within fp
        // tolerance. Indices may differ only on genuine ties (FMA vs non-FMA
        // rounding tips the argmin), so compare the resulting distance — that is
        // what determines PQ quality. k=253 exercises the non-power-of-2 tail.
        #[cfg(not(target_arch = "aarch64"))]
        for &k in &[7usize, 8, 31, 200, 253, 256] {
            let sub_dim = 8usize;
            let vecs = make_vecs(k + 40, sub_dim);
            let table: Vec<f32> = vecs[..k].iter().flatten().copied().collect();
            let lut = build_subspace_lut(&table, k, sub_dim);
            let d2 = |q: &[f32], idx: usize| l2_sq(q, &table[idx * sub_dim..(idx + 1) * sub_dim]);
            for q in &vecs {
                let simd = assign_nearest(q, &lut, k, sub_dim) as usize;
                let scalar = assign_argmin_portable(q, &lut, k, sub_dim) as usize;
                assert!(
                    (d2(q, simd) - d2(q, scalar)).abs() <= 1e-4,
                    "k={k}: SIMD idx {simd} (d={}) vs scalar idx {scalar} (d={})",
                    d2(q, simd),
                    d2(q, scalar),
                );
            }
        }
    }

    #[test]
    fn pq_encode_into_matches_encode_one() {
        let (n, d, m, k) = (200usize, 32usize, 8usize, 16usize);
        let vecs = make_vecs(n, d);
        let cb = train_pq_codebook(&vecs, m, k, 10, 0).unwrap();

        // Reference: per-row encode_one.
        let reference = pq_encode(&vecs, &cb);

        // Flat batch encode.
        let flat: Vec<f32> = vecs.iter().flatten().copied().collect();
        let mut codes = vec![0u8; n * m];
        pq_encode_into(&flat, &cb, &mut codes);

        for (i, row) in reference.iter().enumerate() {
            assert_eq!(&codes[i * m..(i + 1) * m], row.as_slice(), "row {i} mismatch");
        }
    }
}
