//! GEMM-based k-means assignment — the Lloyd assignment step expressed as one
//! matrix multiply instead of a per-centroid distance scan.
//!
//! The scalar assignment loop is `parallel-over-points, serial-over-k`: each
//! point streams the whole `[k, d]` centroid matrix once per Lloyd iteration,
//! `k` separate dot loops. That is `O(n·k·d)` with poor centroid-matrix reuse —
//! the step the CHANGELOG flags as ~3.5× slower than FAISS at high `n_lists`.
//! FAISS instead computes every point-vs-centroid similarity as a single tiled
//! GEMM `[n, d]·[d, k] → [n, k]` (the centroid matrix is streamed once for the
//! whole batch, not once per point) and takes the per-row argmin.
//!
//! [`assign_nearest`] does exactly that: `data · centroidsᵀ` via ndarray's
//! pure-Rust `matrixmultiply` backend (the same call style as
//! `IvfPqIndex::search_batch_flat`'s coarse GEMM), then a parallel per-row
//! argmax of the metric score. It is a drop-in for the assignment step of every
//! `kmeans_lloyd` in the crate. Inputs are flat row-major `[n·d]` / `[k·d]`
//! slices so callers pass their data directly without building ndarray types.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Points per rayon tile. Each worker runs one `[CHUNK, d] · [d, k]` GEMM, so
/// this trades GEMM-setup amortisation (larger is better) against keeping the
/// `[CHUNK, k]` similarity block cache-resident (smaller is better).
const CHUNK: usize = 256;

/// Which distance the assignment minimises.
///
/// Both reduce to a per-row **argmax** of a similarity score, so the GEMM output
/// is consumed the same way in either case:
/// * `Cosine` — vectors are unit-norm, so `argmin (1 − v·c) = argmax v·c`; the
///   score is the raw dot product.
/// * `L2` — `argmin ‖v − c‖² = argmax (v·c − ½‖c‖²)` since `‖v‖²` is constant
///   across centroids for a fixed point; the per-centroid `½‖c‖²` offset is
///   subtracted from the dot. Used for the (non-unit-norm) PQ sub-vectors.
#[derive(Clone, Copy)]
pub(crate) enum Metric {
    Cosine,
    L2,
}

/// The per-centroid score offset for a metric: `½‖c_j‖²` for L2 (subtracted
/// from the dot so its argmax is the squared-distance argmin), zero for Cosine.
#[inline]
fn score_offset(centroid: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => 0.0,
        Metric::L2 => 0.5 * centroid.iter().map(|&x| x * x).sum::<f32>(),
    }
}

/// Assign each of the `n` row-major `data` vectors (`[n·d]`) to its nearest of
/// the `k` `centroids` (`[k·d]`), returning the `n` centroid indices.
///
/// Computes all `n·k` similarities as one `data · centroidsᵀ` GEMM, then takes
/// the per-row argmax of the metric score in parallel. Ties resolve to the
/// lowest centroid index (strict `>` keeps the first max), matching the
/// sequential scan so assignment stays deterministic.
///
/// `data` must be `n·d` long and `centroids` `k·d` long. A length mismatch is a
/// caller bug; rather than panic, it is logged and every point is assigned to
/// centroid 0 (all in-crate callers uphold the invariant).
pub(crate) fn assign_nearest(
    data: &[f32],
    centroids: &[f32],
    n: usize,
    k: usize,
    d: usize,
    metric: Metric,
) -> Vec<usize> {
    if k == 0 || n == 0 {
        return vec![0; n];
    }
    if data.len() != n * d || centroids.len() != k * d {
        tracing::warn!(
            data_len = data.len(),
            centroids_len = centroids.len(),
            n,
            k,
            d,
            "assign_nearest: slice length mismatch; assigning all points to centroid 0"
        );
        return vec![0; n];
    }

    // Lengths are checked above, so the centroid view cannot fail; the `else`
    // arm is an unreachable panic-free safety net.
    let Ok(cent_mat) = ArrayView2::from_shape((k, d), centroids) else {
        return vec![0; n];
    };
    let offsets: Vec<f32> = (0..k)
        .map(|j| score_offset(&centroids[j * d..(j + 1) * d], metric))
        .collect();

    // Tile the points across rayon workers; each worker runs its own
    // `[chunk, d] · [d, k]` GEMM (`matrixmultiply` is single-threaded, so the
    // parallelism has to come from tiling, exactly like `search_batch_flat`)
    // and argmaxes each row immediately. Tiling also caps the live similarity
    // block at `[chunk, k]` instead of materialising the full `[n, k]` matrix.
    let mut out = vec![0usize; n];
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(c, slot)| {
        let lo = c * CHUNK;
        let rows = slot.len();
        if let Ok(dtile) = ArrayView2::from_shape((rows, d), &data[lo * d..(lo + rows) * d]) {
            let sims: Array2<f32> = dtile.dot(&cent_mat.t()); // [rows, k]
            for (r, dst) in slot.iter_mut().enumerate() {
                if let Some(row) = sims.row(r).as_slice() {
                    *dst = argmax_offset(row, &offsets);
                }
            }
        }
    });
    out
}

/// Index of the largest `row[j] − offsets[j]`, ties to the lowest index.
#[inline]
fn argmax_offset(row: &[f32], offsets: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (j, (&s, &off)) in row.iter().zip(offsets.iter()).enumerate() {
        let score = s - off;
        if score > best_score {
            best_score = score;
            best = j;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference: the same argmax without the GEMM — the oracle the GEMM
    /// path is validated against.
    fn assign_scalar(
        data: &[f32],
        centroids: &[f32],
        n: usize,
        k: usize,
        d: usize,
        metric: Metric,
    ) -> Vec<usize> {
        (0..n)
            .map(|i| {
                let v = &data[i * d..(i + 1) * d];
                let mut best = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for j in 0..k {
                    let c = &centroids[j * d..(j + 1) * d];
                    let dot: f32 = v.iter().zip(c).map(|(&a, &b)| a * b).sum();
                    let score = dot - score_offset(c, metric);
                    if score > best_score {
                        best_score = score;
                        best = j;
                    }
                }
                best
            })
            .collect()
    }

    /// Pseudo-random f32 in [-1, 1) via an LCG — varied enough to avoid ties.
    fn rand_vecs(n: usize, d: usize, seed: u64, unit: bool) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut out = vec![0.0f32; n * d];
        for row in out.chunks_mut(d) {
            for x in row.iter_mut() {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *x = (s >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
            }
            if unit {
                let nrm = row.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
                for x in row.iter_mut() {
                    *x /= nrm;
                }
            }
        }
        out
    }

    /// The GEMM assignment must agree with the scalar oracle on both metrics
    /// (random non-degenerate data has no ties, so agreement is exact).
    fn assert_matches_scalar(unit: bool, metric: Metric, seed: u64) {
        let (n, k, d) = (256usize, 19usize, 24usize);
        let data = rand_vecs(n, d, seed, unit);
        let cents = rand_vecs(k, d, seed ^ 0xabc, unit);
        let got = assign_nearest(&data, &cents, n, k, d, metric);
        let want = assign_scalar(&data, &cents, n, k, d, metric);
        assert_eq!(got.len(), n);
        assert_eq!(got, want);
    }

    #[test]
    fn cosine_matches_scalar_reference() {
        assert_matches_scalar(true, Metric::Cosine, 1);
    }

    #[test]
    fn l2_matches_scalar_reference() {
        // Non-unit-norm vectors exercise the ½‖c‖² offset (the PQ sub-vector case).
        assert_matches_scalar(false, Metric::L2, 3);
    }

    #[test]
    fn degenerate_and_mismatched_shapes() {
        // No centroids → one (index-0) assignment per point; no data → empty.
        assert_eq!(
            assign_nearest(&[0.0; 12], &[], 3, 0, 4, Metric::L2),
            vec![0; 3]
        );
        assert!(assign_nearest(&[], &[0.0; 20], 0, 5, 4, Metric::Cosine).is_empty());
        // Length mismatch is guarded (no panic) → all assigned to centroid 0.
        assert_eq!(
            assign_nearest(&[1.0, 2.0], &[1.0, 0.0, 0.0, 1.0], 4, 2, 4, Metric::Cosine),
            vec![0; 4]
        );
    }
}
