//! 3-bit uniform scalar quantization (SQ3).
//!
//! Each per-vector dimension is mapped to one of 8 uniformly-spaced
//! reconstruction levels relative to the vector's abs-max:
//!
//! ```text
//!  code │  reconstruction value
//! ──────┼───────────────────────
//!   0   │  -7/8 · abs_max
//!   1   │  -5/8 · abs_max
//!   2   │  -3/8 · abs_max
//!   3   │  -1/8 · abs_max
//!   4   │   1/8 · abs_max
//!   5   │   3/8 · abs_max
//!   6   │   5/8 · abs_max
//!   7   │   7/8 · abs_max
//! ```
//!
//! Encoding maps `v` into `[0, 8)` via `(v / abs_max + 1.0) * 4.0`, then
//! floors and clamps to `0..7`.
//!
//! **Storage**: codes are packed as a LSB-first bit stream (`ceil(d * 3 / 8)` bytes).
//! Each 3-bit code may span two consecutive bytes when its starting bit offset
//! is ≥ 6 mod 8; the packing is handled uniformly via a general bit-stream loop.
//!
//! **Quality**: cosine similarity of `decode(encode(v))` vs `v` ≥ 0.99 for
//! typical 768-dimensional unit-normalised embeddings.

use serde::{Deserialize, Serialize};

/// Asymmetric `(dot, norm_sq)` of an SQ3-packed stored vector vs an f32 query.
/// Like SQ2 the dequant is **affine** — `dv = scale·(2·code−7)/8` — so no LUT is
/// needed. The AVX2 path unpacks 8 codes/iter: 8 × 3-bit codes span exactly 24
/// bits, so one little-endian `u32` load + a per-lane variable shift extracts
/// them all (no byte-straddle branch). Scalar fallback otherwise.
#[inline]
fn sq3_dot_norm(packed: &[u8], query: &[f32], n: usize, scale: f32) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: gated by the runtime detection above; reads in-bounds lanes.
            return unsafe { sq3_dot_norm_avx2(packed, query, n, scale) };
        }
    }
    sq3_dot_norm_scalar(packed, query, n, scale)
}

/// Scalar reference for [`sq3_dot_norm`] (the original byte-straddle unpack).
#[inline]
fn sq3_dot_norm_scalar(packed: &[u8], query: &[f32], n: usize, scale: f32) -> (f32, f32) {
    let mut dot = 0.0f32;
    let mut norm_sq = 0.0f32;
    for (i, &q) in query.iter().enumerate().take(n) {
        let bit_pos = i * 3;
        let byte_idx = bit_pos / 8;
        let bit_shift = bit_pos % 8;
        let code = if bit_shift <= 5 {
            (packed[byte_idx] >> bit_shift) & 0x7
        } else {
            let lo = packed[byte_idx] >> bit_shift;
            let hi = packed[byte_idx + 1] << (8 - bit_shift);
            (lo | hi) & 0x7
        };
        let dv = scale * ((2 * code as i32 - 7) as f32 / 8.0);
        dot += dv * q;
        norm_sq += dv * dv;
    }
    (dot, norm_sq)
}

/// AVX2 kernel for [`sq3_dot_norm`]. Each group of 8 dims occupies 3 bytes (24
/// bits) starting at byte `g*3`; a `u32` load (needs `g*3 + 4 <= len`, the 4th
/// byte's bits are masked off) plus `srlv` by `[0,3,…,21]` & `7` yields all 8
/// codes. Affine dequant `dv = scale·(2·code−7)·⅛`, two FMA chains. Groups
/// without a 4th readable byte fall to the scalar tail.
///
/// # Safety
/// Requires AVX2 + FMA (the caller runtime-detects). Reads only in-bounds lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sq3_dot_norm_avx2(packed: &[u8], query: &[f32], n: usize, scale: f32) -> (f32, f32) {
    use std::arch::x86_64::*;
    let shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    let mask = _mm256_set1_epi32(0x7);
    let scale_v = _mm256_set1_ps(scale);
    let seven = _mm256_set1_ps(7.0);
    let eighth = _mm256_set1_ps(0.125);
    let mut dot = _mm256_setzero_ps();
    let mut nrm = _mm256_setzero_ps();
    let qp = query.as_ptr();
    // Number of 8-dim groups whose 3 code bytes + 1 load-slack byte are in range.
    let max_groups = if packed.len() >= 4 {
        (packed.len() - 4) / 3 + 1
    } else {
        0
    };
    let full = (n / 8).min(max_groups);
    for g in 0..full {
        let base = g * 3;
        let word = u32::from_le_bytes([
            *packed.get_unchecked(base),
            *packed.get_unchecked(base + 1),
            *packed.get_unchecked(base + 2),
            *packed.get_unchecked(base + 3),
        ]) as i32;
        let codes = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(word), shifts), mask);
        let code_f = _mm256_cvtepi32_ps(codes);
        // dv = scale · ((2·code − 7) · ⅛)
        let two_code = _mm256_add_ps(code_f, code_f);
        let dv = _mm256_mul_ps(
            scale_v,
            _mm256_mul_ps(_mm256_sub_ps(two_code, seven), eighth),
        );
        let q8 = _mm256_loadu_ps(qp.add(g * 8));
        dot = _mm256_fmadd_ps(dv, q8, dot);
        nrm = _mm256_fmadd_ps(dv, dv, nrm);
    }
    let (mut d, mut nm) = (hsum256(dot), hsum256(nrm));
    for i in full * 8..n {
        let bit_pos = i * 3;
        let byte_idx = bit_pos / 8;
        let bit_shift = bit_pos % 8;
        let code = if bit_shift <= 5 {
            (*packed.get_unchecked(byte_idx) >> bit_shift) & 0x7
        } else {
            let lo = *packed.get_unchecked(byte_idx) >> bit_shift;
            let hi = *packed.get_unchecked(byte_idx + 1) << (8 - bit_shift);
            (lo | hi) & 0x7
        };
        let dv = scale * ((2 * code as i32 - 7) as f32 / 8.0);
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

/// One 3-bit-quantized vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sq3Vector {
    /// LSB-first 3-bit codes packed into bytes; `len == ceil(dim * 3 / 8)`.
    pub packed: Vec<u8>,
    /// Per-vector abs-max scale factor.
    pub scale: f32,
    /// Original vector dimension.
    pub dim: usize,
}

impl Sq3Vector {
    /// Encode a single f32 slice to 3-bit SQ.
    pub fn encode(v: &[f32]) -> Self {
        let dim = v.len();
        let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
        let inv = 1.0 / scale;

        let n_bytes = (dim * 3).div_ceil(8);
        let mut packed = vec![0u8; n_bytes];

        for (i, &x) in v.iter().enumerate() {
            let normalized = (x * inv).clamp(-1.0, 1.0);
            // Map [-1, 1] → [0, 8) then floor+clamp to 0..7.
            let code = ((normalized + 1.0) * 4.0).floor() as i32;
            let code = code.clamp(0, 7) as u8;

            let bit_pos = i * 3;
            let byte_idx = bit_pos / 8;
            let bit_shift = bit_pos % 8;

            packed[byte_idx] |= code << bit_shift;
            if bit_shift > 5 {
                // Code spans into the next byte (bits 6+ of the current byte).
                packed[byte_idx + 1] |= code >> (8 - bit_shift);
            }
        }

        Self { packed, scale, dim }
    }

    /// Asymmetric cosine distance to a full-precision query, computed directly
    /// from the packed 3-bit codes — no `decode()` allocation. Equivalent to
    /// `cosine_dist_f32(&self.decode(), query)`.
    #[inline]
    pub fn cosine_dist_to_query(&self, query: &[f32]) -> f32 {
        let n = self.dim.min(query.len());
        let (dot, norm_sq) = sq3_dot_norm(&self.packed, query, n, self.scale);
        let norm = norm_sq.sqrt();
        if norm < 1e-8 {
            return 1.0;
        }
        (1.0 - dot / norm).max(0.0)
    }

    /// Decode back to approximate f32.
    ///
    /// Reconstruction levels: `(2 * code − 7) / 8 * scale`.
    pub fn decode(&self) -> Vec<f32> {
        (0..self.dim)
            .map(|i| {
                let bit_pos = i * 3;
                let byte_idx = bit_pos / 8;
                let bit_shift = bit_pos % 8;

                let code = if bit_shift <= 5 {
                    (self.packed[byte_idx] >> bit_shift) & 0x7
                } else {
                    let lo = self.packed[byte_idx] >> bit_shift;
                    let hi = self.packed[byte_idx + 1] << (8 - bit_shift);
                    (lo | hi) & 0x7
                };

                // Divide before multiplying to avoid f32 overflow for large scale values.
                // code 0 → -7/8, 1 → -5/8, ..., 7 → 7/8
                self.scale * ((2 * code as i32 - 7) as f32 / 8.0)
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
        let enc = Sq3Vector::encode(&v);
        assert_eq!(enc.dim, 768);
        assert_eq!(enc.packed.len(), 288); // ceil(768*3/8) = 288
        let dec = enc.decode();
        assert_eq!(dec.len(), 768);
    }

    #[test]
    fn encode_decode_cosine_quality() {
        let v = unit_vec(768, 0.01);
        let enc = Sq3Vector::encode(&v);
        let dec = enc.decode();
        let sim = cosine_sim(&v, &dec);
        assert!(sim >= 0.99, "cosine sim after SQ3 round-trip = {sim:.4}");
    }

    #[test]
    fn zero_vector_does_not_panic() {
        let v = vec![0.0f32; 16];
        let enc = Sq3Vector::encode(&v);
        let dec = enc.decode();
        assert_eq!(dec.len(), 16);
    }

    #[test]
    fn decode_bit_exact_vs_formula() {
        // Pins `decode` bit-for-bit to the documented reconstruction formula so
        // any future decode optimisation (LUT/SIMD) cannot silently drift,
        // including codes that straddle a byte boundary (dim=101 → 303 bits).
        let v = unit_vec(101, 0.029);
        let enc = Sq3Vector::encode(&v);
        let dec = enc.decode();
        for (i, &got) in dec.iter().enumerate() {
            let bit_pos = i * 3;
            let byte_idx = bit_pos / 8;
            let bit_shift = bit_pos % 8;
            let code = if bit_shift <= 5 {
                (enc.packed[byte_idx] >> bit_shift) & 0x7
            } else {
                let lo = enc.packed[byte_idx] >> bit_shift;
                let hi = enc.packed[byte_idx + 1] << (8 - bit_shift);
                (lo | hi) & 0x7
            };
            let want = enc.scale * ((2 * code as i32 - 7) as f32 / 8.0);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "decode mismatch at index {i}"
            );
        }
    }

    #[test]
    fn odd_dim_roundtrip() {
        // dims that are not multiples of 8* — exercises cross-byte boundary
        for d in [1, 3, 7, 9, 13, 17, 23, 100, 101] {
            let v = unit_vec(d, 0.07);
            let enc = Sq3Vector::encode(&v);
            assert_eq!(enc.dim, d);
            let dec = enc.decode();
            assert_eq!(dec.len(), d);
            if d >= 4 {
                let sim = cosine_sim(&v, &dec);
                assert!(sim >= 0.90, "cosine sim for dim={d}: {sim:.4}");
            }
        }
    }

    #[test]
    fn extreme_values_finite() {
        let v = vec![1e38f32, -1e38, 0.0, 1e-38];
        let enc = Sq3Vector::encode(&v);
        let dec = enc.decode();
        for &x in &dec {
            assert!(x.is_finite(), "decoded SQ3 value is not finite: {x}");
        }
    }

    proptest! {
        #[test]
        fn proptest_roundtrip_no_nan(
            raw in proptest::collection::vec(-1e18f32..1e18f32, 1..256usize)
        ) {
            let enc = Sq3Vector::encode(&raw);
            let dec = enc.decode();
            prop_assert_eq!(dec.len(), raw.len());
            for &x in &dec {
                prop_assert!(x.is_finite());
            }
        }
    }
}
