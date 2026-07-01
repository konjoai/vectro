//! Shared GEMM-based nearest-centroid assignment for Lloyd's k-means.
//!
//! `ivf_pq.rs` and `ivf.rs` each ran the Lloyd assignment step as "parallel
//! across points, serial across centroids" — one distance call per centroid
//! per point (`O(n·k·d)` with `k` separate dot-product loops re-streaming the
//! centroid matrix for every point). FAISS instead computes the whole
//! `[n,d]×[d,k]` similarity as a single tiled GEMM (reusing the centroid
//! matrix across all points) and argmaxes each row. This module is that
//! shared assignment step; each site keeps its own init/update/convergence
//! logic and only swaps in [`assign_nearest`].

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Distance metric the assignment argmax is computed under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Metric {
    /// Unit-norm vectors: max dot product = min cosine distance.
    Cosine,
    /// Arbitrary vectors: `argmin_j ‖v_i − c_j‖² = argmax_j(v_i·c_j − ½‖c_j‖²)`.
    L2,
}

/// Row-tile width for the assignment GEMM (see [`assign_nearest`]).
const CHUNK: usize = 256;

/// Nearest-centroid assignment for every row of `data` against `centroids`.
///
/// Tiles `data` into row-chunks processed in parallel with rayon; each worker
/// computes its tile's `data_tile · centroidsᵀ` GEMM (reusing the centroid
/// matrix across the tile, FAISS-style) and argmaxes its rows. A single
/// whole-dataset GEMM would be cache-efficient but runs on one thread — that
/// would surrender the per-point parallelism the sequential per-centroid loop
/// had; tiling keeps both the GEMM's reuse of the centroid matrix and
/// rayon-parallel throughput across points, mirroring the coarse-scan tiling
/// in [`super::ivf_pq::IvfPqIndex::search_batch_flat`].
///
/// Ties (equal score) resolve to the lowest centroid index, matching the
/// deterministic behaviour of the sequential per-centroid loop this replaces.
pub(crate) fn assign_nearest(
    data: ArrayView2<f32>,
    centroids: ArrayView2<f32>,
    metric: Metric,
) -> Vec<usize> {
    let n = data.nrows();
    let d = data.ncols();
    let k = centroids.nrows();

    let bias: Vec<f32> = match metric {
        Metric::Cosine => vec![0.0; k],
        Metric::L2 => centroids
            .outer_iter()
            .map(|c| 0.5 * c.iter().map(|&x| x * x).sum::<f32>())
            .collect(),
    };

    let flat = data.as_standard_layout();
    // `as_standard_layout()` always yields a C-contiguous array, so `as_slice()`
    // cannot fail; `.expect()` is banned outside tests by this crate's lint
    // config, so make the invariant explicit instead.
    let flat = flat
        .as_slice()
        .unwrap_or_else(|| unreachable!("as_standard_layout() is always contiguous"));

    let mut assignments = vec![0usize; n];
    assignments
        .par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(c, out)| {
            let lo = c * CHUNK;
            let rows = out.len();
            // `rows * d == (lo + rows) * d - lo * d` always matches the tile slice
            // length, so this shape construction cannot fail.
            let tile = ArrayView2::from_shape((rows, d), &flat[lo * d..(lo + rows) * d])
                .unwrap_or_else(|_| unreachable!("tile length always matches (rows, d)"));
            let sims: Array2<f32> = tile.dot(&centroids.t()); // [rows, k]
            for (r, slot) in out.iter_mut().enumerate() {
                let row = sims.row(r);
                let mut best = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for (j, &s) in row.iter().enumerate() {
                    let score = s - bias[j];
                    if score > best_score {
                        best_score = score;
                        best = j;
                    }
                }
                *slot = best;
            }
        });
    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        1.0 - dot
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn random_vectors(n: usize, d: usize, seed: u64, unit_norm: bool) -> Vec<f32> {
        let mut state = seed;
        let mut out = vec![0.0f32; n * d];
        for row in out.chunks_mut(d) {
            for x in row.iter_mut() {
                *x = (lcg(&mut state) - 0.5) as f32 * 2.0;
            }
            if unit_norm {
                let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
                for x in row.iter_mut() {
                    *x /= norm;
                }
            }
        }
        out
    }

    #[test]
    fn matches_scalar_cosine_argmax() {
        let d = 16;
        let n = 200;
        let k = 12;
        let data = random_vectors(n, d, 1, true);
        let cents = random_vectors(k, d, 2, true);

        let got = assign_nearest(
            ArrayView2::from_shape((n, d), &data).unwrap(),
            ArrayView2::from_shape((k, d), &cents).unwrap(),
            Metric::Cosine,
        );

        for (i, &gi) in got.iter().enumerate() {
            let v = &data[i * d..(i + 1) * d];
            let mut best = 0usize;
            let mut best_d = f32::MAX;
            for ci in 0..k {
                let c = &cents[ci * d..(ci + 1) * d];
                let dist = cosine_dist(v, c);
                if dist < best_d {
                    best_d = dist;
                    best = ci;
                }
            }
            assert_eq!(gi, best, "row {i} mismatched cosine assignment");
        }
    }

    #[test]
    fn matches_scalar_l2_argmin() {
        let d = 24;
        let n = 150;
        let k = 9;
        let data = random_vectors(n, d, 3, false);
        let cents = random_vectors(k, d, 4, false);

        let got = assign_nearest(
            ArrayView2::from_shape((n, d), &data).unwrap(),
            ArrayView2::from_shape((k, d), &cents).unwrap(),
            Metric::L2,
        );

        for (i, &gi) in got.iter().enumerate() {
            let v = &data[i * d..(i + 1) * d];
            let mut best = 0usize;
            let mut best_d = f32::MAX;
            for ci in 0..k {
                let c = &cents[ci * d..(ci + 1) * d];
                let dist = l2_sq(v, c);
                if dist < best_d {
                    best_d = dist;
                    best = ci;
                }
            }
            assert_eq!(gi, best, "row {i} mismatched L2 assignment");
        }
    }

    #[test]
    fn ties_prefer_lowest_index() {
        // Two identical centroids at index 0 and 1: point equidistant must pick 0.
        let d = 4;
        let data = vec![1.0f32, 0.0, 0.0, 0.0];
        let cents = vec![0.5f32, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
        let got = assign_nearest(
            ArrayView2::from_shape((1, d), &data).unwrap(),
            ArrayView2::from_shape((2, d), &cents).unwrap(),
            Metric::L2,
        );
        assert_eq!(got, vec![0]);
    }
}
