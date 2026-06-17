//! Residual Quantization (RQ): chains `n_passes` Product Quantization codebooks,
//! training each pass on the residual produced by all previous passes.
//!
//! # Encoding
//! ```text
//! residual₀ = x
//! for l in 0..n_passes:
//!     codes[l] = PQ_l.encode(residual_l)
//!     residual_{l+1} = residual_l − PQ_l.decode(codes[l])
//! ```
//!
//! # Decoding
//! ```text
//! x̂ = Σ  PQ_l.decode(codes[l])   for l in 0..n_passes
//! ```
//!
//! On-disk flat-code layout: `n_passes × n_subspaces` bytes per vector
//! (fixed record size, no nested-Vec overhead).

use crate::quant::pq::{pq_decode, pq_encode, train_pq_codebook, PQCodebook};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ──────────────────────────── public types ────────────────────────────────────

/// A chained Residual Quantization codebook consisting of `n_passes` PQ codebooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RQCodebook {
    /// Number of residual passes (L).
    pub n_passes: usize,
    /// PQ subspaces per pass (M).
    pub n_subspaces: usize,
    /// Centroids per subspace (K ≤ 256).
    pub n_centroids: usize,
    /// One `PQCodebook` per pass, in order.
    pub codebooks: Vec<PQCodebook>,
}

// ──────────────────────────── training ────────────────────────────────────────

/// Train an RQ codebook from `data` (rows of `d`-dim `f32` vectors).
///
/// * `n_passes` — number of residual passes (L ≥ 1)
/// * `n_subspaces` — PQ subspaces per pass (M, must divide `d`)
/// * `n_centroids` — centroids per subspace (K ≤ 256)
/// * `max_iter` — Lloyd's iterations per pass
/// * `seed` — deterministic seed; pass `l` uses `seed + l`
pub fn train_rq_codebook(
    data: &[Vec<f32>],
    n_passes: usize,
    n_subspaces: usize,
    n_centroids: usize,
    max_iter: usize,
    seed: u64,
) -> Result<RQCodebook, String> {
    if data.is_empty() {
        return Err("empty training set".into());
    }
    if n_passes == 0 {
        return Err("n_passes must be ≥ 1".into());
    }

    let mut codebooks: Vec<PQCodebook> = Vec::with_capacity(n_passes);
    let mut residuals: Vec<Vec<f32>> = data.to_vec();

    for pass in 0..n_passes {
        let cb = train_pq_codebook(
            &residuals,
            n_subspaces,
            n_centroids,
            max_iter,
            seed + pass as u64,
        )?;

        // Encode current residuals, decode approximation, subtract → next residuals
        let codes = pq_encode(&residuals, &cb);
        let recon = pq_decode(&codes, &cb);

        residuals = residuals
            .par_iter()
            .zip(recon.par_iter())
            .map(|(r, rec)| r.iter().zip(rec.iter()).map(|(a, b)| a - b).collect())
            .collect();

        codebooks.push(cb);
    }

    Ok(RQCodebook {
        n_passes,
        n_subspaces,
        n_centroids,
        codebooks,
    })
}

// ──────────────────────────── encoding ────────────────────────────────────────

/// Encode `vecs` using the RQ codebook.
///
/// Returns `[n_vecs][n_passes][n_subspaces]`.
pub fn rq_encode(codebook: &RQCodebook, vecs: &[Vec<f32>]) -> Vec<Vec<Vec<u8>>> {
    let n = vecs.len();
    let mut all_codes: Vec<Vec<Vec<u8>>> = vec![vec![vec![]; codebook.n_passes]; n];
    let mut residuals: Vec<Vec<f32>> = vecs.to_vec();

    for (pass, cb) in codebook.codebooks.iter().enumerate() {
        let codes = pq_encode(&residuals, cb);
        let recon = pq_decode(&codes, cb);

        for (i, code) in codes.into_iter().enumerate() {
            all_codes[i][pass] = code;
        }

        residuals = residuals
            .par_iter()
            .zip(recon.par_iter())
            .map(|(r, rec)| r.iter().zip(rec.iter()).map(|(a, b)| a - b).collect())
            .collect();
    }

    all_codes
}

/// Encode `vecs` and return flat codes: `[n_vecs][n_passes * n_subspaces]`.
///
/// This is the preferred on-disk format (fixed record size).
pub fn rq_encode_flat(codebook: &RQCodebook, vecs: &[Vec<f32>]) -> Vec<Vec<u8>> {
    rq_encode(codebook, vecs)
        .into_iter()
        .map(|passes| passes.into_iter().flatten().collect())
        .collect()
}

// ──────────────────────────── decoding ────────────────────────────────────────

/// Decode flat codes `[n_vecs][n_passes * n_subspaces]` → `[n_vecs][dim]`.
pub fn rq_decode_flat(codebook: &RQCodebook, codes: &[Vec<u8>]) -> Vec<Vec<f32>> {
    codes.par_iter().map(|flat| decode_one(codebook, flat)).collect()
}

/// Decode nested codes `[n_vecs][n_passes][n_subspaces]` → `[n_vecs][dim]`.
pub fn rq_decode(codebook: &RQCodebook, codes: &[Vec<Vec<u8>>]) -> Vec<Vec<f32>> {
    codes
        .par_iter()
        .map(|passes| {
            let flat: Vec<u8> = passes.iter().flatten().copied().collect();
            decode_one(codebook, &flat)
        })
        .collect()
}

// ── internal ──────────────────────────────────────────────────────────────────

/// Decode one flat-code vector: sum per-pass PQ reconstructions.
///
/// Accumulates centroid slices directly into `result` rather than allocating a
/// fresh `Vec` (and spawning a single-element rayon job) per pass — the per-pass
/// summation order is unchanged, so the result is numerically identical.
fn decode_one(codebook: &RQCodebook, flat: &[u8]) -> Vec<f32> {
    let m = codebook.n_subspaces;
    let sub_dim = codebook.codebooks[0].sub_dim;
    let mut result = vec![0.0f32; sub_dim * m];

    for (pass, cb) in codebook.codebooks.iter().enumerate() {
        let code_slice = &flat[pass * m..(pass + 1) * m];
        for (mi, &k) in code_slice.iter().enumerate() {
            let cent = cb.centroid(mi, k as usize);
            let off = mi * sub_dim;
            for (r, &c) in result[off..off + sub_dim].iter_mut().zip(cent.iter()) {
                *r += c;
            }
        }
    }

    result
}

// ─────────────────────────────── tests ────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        let mut lcg = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        (0..n).map(|_| (0..d).map(|_| lcg()).collect()).collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return -1.0;
        }
        dot / (na * nb)
    }

    #[test]
    fn rq_codebook_shape() {
        let vecs = rand_vecs(200, 64, 1);
        let cb = train_rq_codebook(&vecs, 2, 8, 16, 10, 42).unwrap();
        assert_eq!(cb.n_passes, 2);
        assert_eq!(cb.codebooks.len(), 2);
        assert_eq!(cb.n_subspaces, 8);
        assert_eq!(cb.n_centroids, 16);
    }

    #[test]
    fn rq_flat_code_shape() {
        let vecs = rand_vecs(100, 64, 2);
        let cb = train_rq_codebook(&vecs, 2, 8, 16, 10, 42).unwrap();
        let codes = rq_encode_flat(&cb, &vecs);
        assert_eq!(codes.len(), 100);
        // n_passes * n_subspaces bytes per vector
        assert_eq!(codes[0].len(), 2 * 8);
    }

    #[test]
    fn rq_round_trip_shape() {
        let vecs = rand_vecs(50, 64, 3);
        let cb = train_rq_codebook(&vecs, 2, 8, 16, 10, 42).unwrap();
        let codes = rq_encode_flat(&cb, &vecs);
        let recon = rq_decode_flat(&cb, &codes);
        assert_eq!(recon.len(), 50);
        assert_eq!(recon[0].len(), 64);
    }

    #[test]
    fn rq_quality_exceeds_single_pass() {
        // 2-pass RQ on 300 training vectors must achieve ≥ 0.90 avg cosine
        let vecs = rand_vecs(300, 64, 42);
        let cb = train_rq_codebook(&vecs, 2, 8, 16, 25, 42).unwrap();
        let codes = rq_encode_flat(&cb, &vecs);
        let recon = rq_decode_flat(&cb, &codes);

        let avg: f32 = vecs
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| cosine(a, b))
            .sum::<f32>()
            / vecs.len() as f32;

        assert!(avg >= 0.90, "avg cosine {avg:.4} < 0.90");
    }

    #[test]
    fn rq_nested_flat_decode_match() {
        let vecs = rand_vecs(50, 64, 5);
        let cb = train_rq_codebook(&vecs, 2, 8, 16, 20, 42).unwrap();
        let nested = rq_encode(&cb, &vecs);
        let flat = rq_encode_flat(&cb, &vecs);
        let from_nested = rq_decode(&cb, &nested);
        let from_flat = rq_decode_flat(&cb, &flat);

        for (a, b) in from_nested.iter().zip(from_flat.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-5, "nested/flat mismatch: {x} vs {y}");
            }
        }
    }

    #[test]
    fn rq_fails_on_empty_data() {
        let res = train_rq_codebook(&[], 2, 8, 16, 10, 42);
        assert!(res.is_err());
    }

    #[test]
    fn rq_fails_on_zero_passes() {
        let vecs = rand_vecs(50, 64, 6);
        let res = train_rq_codebook(&vecs, 0, 8, 16, 10, 42);
        assert!(res.is_err());
    }
}
