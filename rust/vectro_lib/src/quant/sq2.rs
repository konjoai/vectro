//! 2-bit uniform scalar quantization (SQ2).
//!
//! Each per-vector dimension is mapped to one of 4 uniformly-spaced
//! reconstruction levels relative to the vector's abs-max:
//!
//! ```text
//!  code │  reconstruction value
//! ──────┼───────────────────────
//!   0   │  -3/4 · abs_max
//!   1   │  -1/4 · abs_max
//!   2   │   1/4 · abs_max
//!   3   │   3/4 · abs_max
//! ```
//!
//! Encoding maps `v` into `[0, 4)` via `(v / abs_max + 1.0) * 2.0`, then
//! floors and clamps to `0..3`.
//!
//! **Storage**: 4 codes are packed into 1 byte (2 bits each, LSB-first).
//! A vector of dimension `d` occupies `ceil(d / 4)` bytes plus 4 bytes for
//! the f32 scale.
//!
//! **Quality**: cosine similarity of `decode(encode(v))` vs `v` ≥ 0.95 for
//! typical 768-dimensional unit-normalised embeddings.

use serde::{Deserialize, Serialize};

/// Asymmetric `(dot, norm_sq)` of an SQ2-packed stored vector vs an f32 query.
/// The dequantised value is **affine in the code** — `dv = scale·(2·code−3)/4` —
/// so no lookup table is needed: unpack the 2-bit code to a float and apply the
/// affine map. AVX2+FMA on x86_64 (runtime-detected), scalar fallback otherwise.
#[inline]
fn sq2_dot_norm(packed: &[u8], query: &[f32], n: usize, scale: f32) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: gated by the runtime detection above; reads in-bounds lanes.
            return unsafe { sq2_dot_norm_avx2(packed, query, n, scale) };
        }
    }
    sq2_dot_norm_scalar(packed, query, n, scale)
}

/// Scalar reference for [`sq2_dot_norm`].
#[inline]
fn sq2_dot_norm_scalar(packed: &[u8], query: &[f32], n: usize, scale: f32) -> (f32, f32) {
    let mut dot = 0.0f32;
    let mut norm_sq = 0.0f32;
    for (i, &q) in query.iter().enumerate().take(n) {
        let code = (packed[i / 4] >> ((i % 4) * 2)) & 0b11;
        let dv = scale * ((2 * code as i32 - 3) as f32 / 4.0);
        dot += dv * q;
        norm_sq += dv * dv;
    }
    (dot, norm_sq)
}

/// AVX2 kernel for [`sq2_dot_norm`]. Unpacks 8 codes/iter (2 bytes) via per-lane
/// variable shift, applies the affine dequant `dv = scale·(2·code−3)·¼`, and
/// accumulates dot and squared-norm with two FMA chains.
///
/// # Safety
/// Requires AVX2 + FMA (the caller runtime-detects). Reads only `min(dim, query)` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sq2_dot_norm_avx2(packed: &[u8], query: &[f32], n: usize, scale: f32) -> (f32, f32) {
    use std::arch::x86_64::*;
    let full = n / 8;
    let shifts = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    let mask = _mm256_set1_epi32(0b11);
    let scale_v = _mm256_set1_ps(scale);
    let three = _mm256_set1_ps(3.0);
    let quarter = _mm256_set1_ps(0.25);
    let mut dot = _mm256_setzero_ps();
    let mut nrm = _mm256_setzero_ps();
    let qp = query.as_ptr();
    for g in 0..full {
        let b0 = *packed.get_unchecked(g * 2) as i32;
        let b1 = *packed.get_unchecked(g * 2 + 1) as i32;
        let bytes = _mm256_setr_epi32(b0, b0, b0, b0, b1, b1, b1, b1);
        let codes = _mm256_and_si256(_mm256_srlv_epi32(bytes, shifts), mask);
        let code_f = _mm256_cvtepi32_ps(codes);
        // dv = scale · ((2·code − 3) · ¼)
        let two_code = _mm256_add_ps(code_f, code_f);
        let dv = _mm256_mul_ps(
            scale_v,
            _mm256_mul_ps(_mm256_sub_ps(two_code, three), quarter),
        );
        let q8 = _mm256_loadu_ps(qp.add(g * 8));
        dot = _mm256_fmadd_ps(dv, q8, dot);
        nrm = _mm256_fmadd_ps(dv, dv, nrm);
    }
    let (mut d, mut nm) = (hsum256(dot), hsum256(nrm));
    for i in full * 8..n {
        let code = (*packed.get_unchecked(i / 4) >> ((i % 4) * 2)) & 0b11;
        let dv = scale * ((2 * code as i32 - 3) as f32 / 4.0);
        d += dv * *query.get_unchecked(i);
        nm += dv * dv;
    }
    (d, nm)
}

/// Horizontal sum of an `f32x8`.
///
/// # Safety
/// Requires AVX (caller is AVX2-gated).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let mut s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// One 2-bit-quantized vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sq2Vector {
    /// LSB-first packed 2-bit codes; `len == ceil(dim / 4)`.
    pub packed: Vec<u8>,
    /// Per-vector abs-max scale factor.
    pub scale: f32,
    /// Original vector dimension.
    pub dim: usize,
}

impl Sq2Vector {
    /// Encode a single f32 slice to 2-bit SQ.
    pub fn encode(v: &[f32]) -> Self {
        let dim = v.len();
        let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
        let inv = 1.0 / scale;

        let n_bytes = dim.div_ceil(4);
        let mut packed = vec![0u8; n_bytes];

        for (i, &x) in v.iter().enumerate() {
            // Map [-1, 1] → [0, 4) then floor+clamp to 0..3.
            let normalized = (x * inv).clamp(-1.0, 1.0);
            let code = ((normalized + 1.0) * 2.0).floor() as i32;
            let code = code.clamp(0, 3) as u8;
            packed[i / 4] |= code << ((i % 4) * 2);
        }

        Self { packed, scale, dim }
    }

    /// Asymmetric cosine distance to a full-precision query, computed directly
    /// from the packed 2-bit codes — no `decode()` allocation. Equivalent to
    /// `cosine_dist_f32(&self.decode(), query)`.
    #[inline]
    pub fn cosine_dist_to_query(&self, query: &[f32]) -> f32 {
        let n = self.dim.min(query.len());
        let (dot, norm_sq) = sq2_dot_norm(&self.packed, query, n, self.scale);
        let norm = norm_sq.sqrt();
        if norm < 1e-8 {
            return 1.0;
        }
        (1.0 - dot / norm).max(0.0)
    }

    /// Decode back to approximate f32.
    ///
    /// Reconstruction levels: `(-3, -1, 1, 3) / 4 * scale`.
    pub fn decode(&self) -> Vec<f32> {
        (0..self.dim)
            .map(|i| {
                let code = (self.packed[i / 4] >> ((i % 4) * 2)) & 0b11;
                // Divide before multiplying to avoid f32 overflow for large scale values.
                // code 0 → -3/4, 1 → -1/4, 2 → 1/4, 3 → 3/4
                self.scale * ((2 * code as i32 - 3) as f32 / 4.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn unit_vec(d: usize, seed: f32) -> Vec<f32> {
        let v: Vec<f32> = (0..d).map(|i| (i as f32 * seed + 0.1).sin()).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return v;
        }
        v.into_iter().map(|x| x / norm).collect()
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na * nb == 0.0 {
            return 0.0;
        }
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }

    #[test]
    fn encode_decode_shape() {
        let v = unit_vec(768, 0.01);
        let enc = Sq2Vector::encode(&v);
        assert_eq!(enc.dim, 768);
        assert_eq!(enc.packed.len(), 192); // ceil(768/4) = 192
        let dec = enc.decode();
        assert_eq!(dec.len(), 768);
    }

    #[test]
    fn encode_decode_cosine_quality() {
        // Cosine similarity of the 768-dim round-trip must be ≥ 0.95.
        let v = unit_vec(768, 0.01);
        let enc = Sq2Vector::encode(&v);
        let dec = enc.decode();
        let sim = cosine_sim(&v, &dec);
        assert!(sim >= 0.95, "cosine sim after SQ2 round-trip = {sim:.4}");
    }

    #[test]
    fn zero_vector_does_not_panic() {
        let v = vec![0.0f32; 16];
        let enc = Sq2Vector::encode(&v);
        let dec = enc.decode();
        assert_eq!(dec.len(), 16);
    }

    #[test]
    fn decode_bit_exact_vs_formula() {
        // Pins `decode` bit-for-bit to the documented reconstruction formula so
        // any future decode optimisation (LUT/SIMD) cannot silently drift.
        // dim=257 exercises full bytes plus a partial trailing byte.
        let v = unit_vec(257, 0.013);
        let enc = Sq2Vector::encode(&v);
        let dec = enc.decode();
        for (i, &got) in dec.iter().enumerate() {
            let code = (enc.packed[i / 4] >> ((i % 4) * 2)) & 0b11;
            let want = enc.scale * ((2 * code as i32 - 3) as f32 / 4.0);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "decode mismatch at index {i}"
            );
        }
    }

    #[test]
    fn odd_dim_roundtrip() {
        // dim not divisible by 4
        let v = unit_vec(13, 0.07);
        let enc = Sq2Vector::encode(&v);
        assert_eq!(enc.dim, 13);
        let dec = enc.decode();
        assert_eq!(dec.len(), 13);
        let sim = cosine_sim(&v, &dec);
        assert!(sim >= 0.90, "cosine sim for dim=13: {sim:.4}");
    }

    #[test]
    fn extreme_values_clamp() {
        let v = vec![1e38f32, -1e38, 0.0, 1e-38];
        let enc = Sq2Vector::encode(&v);
        let dec = enc.decode();
        assert_eq!(dec.len(), 4);
        // Must not be NaN or Inf
        for &x in &dec {
            assert!(x.is_finite(), "decoded value is not finite: {x}");
        }
    }

    proptest! {
        #[test]
        fn proptest_roundtrip_no_nan(
            raw in proptest::collection::vec(-1e18f32..1e18f32, 1..256usize)
        ) {
            let enc = Sq2Vector::encode(&raw);
            let dec = enc.decode();
            prop_assert_eq!(dec.len(), raw.len());
            for &x in &dec {
                prop_assert!(x.is_finite());
            }
        }
    }
}
