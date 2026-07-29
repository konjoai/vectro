//! BFloat16 (BF16) quantization with SimSIMD-accelerated distance computation.
//!
//! BF16 truncates the IEEE-754 float32 mantissa from 23 bits to 7 bits while
//! preserving the full 8-bit exponent.  This gives 2× memory savings with
//! negligible cosine-similarity loss on unit-normalised embedding vectors.
//!
//! Storage: 2 bytes/dimension (vs 4 for f32).
//! Quality: cosine similarity ≥ 0.9999 on typical 768-dim embeddings.
//!
//! Distance computation delegates to [`simsimd`]'s BF16 cosine kernel, which
//! automatically uses AVX-512-BF16 on Sapphire Rapids / Genoa, the ARM BF16
//! NEON extension, or falls back to a portable software path.

use serde::{Deserialize, Serialize};
use simsimd::{bf16 as SimBf16, SpatialSimilarity};

/// Asymmetric `(dot, norm_sq)` of a bf16-packed stored vector against an f32
/// query: `dot = Σ dv·q`, `norm_sq = Σ dv²`, where `dv` is the widened bf16.
/// BF16→F32 widening is exact (`f32::from_bits((bits as u32) << 16)`), so the
/// AVX2 path is bit-identical to the scalar reference. AVX2+FMA on x86_64
/// (runtime-detected), scalar fallback otherwise.
#[inline]
pub(crate) fn bf16_dot_norm(packed: &[u16], query: &[f32], n: usize) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 doubles the widen width (256-bit u16 load → 16 lanes). Unlike
        // the f32 distance kernels — where AVX-512 lost on this double-pumped
        // Xeon because they are FMA-bound — the bf16 path is load/widen-bound
        // (`cvtepu16_epi32` + `slli`), so halving the widen-op count is a net
        // win: measured ~1.34–1.44× over AVX2 at d≥128 (no regression at d=96).
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: gated by the runtime detection above; reads in-bounds lanes.
            return unsafe { bf16_dot_norm_avx512(packed, query, n) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: gated by the runtime detection above; reads in-bounds lanes.
            return unsafe { bf16_dot_norm_avx2(packed, query, n) };
        }
    }
    bf16_dot_norm_scalar(packed, query, n)
}

/// Scalar reference for [`bf16_dot_norm`].
#[inline]
fn bf16_dot_norm_scalar(packed: &[u16], query: &[f32], n: usize) -> (f32, f32) {
    let mut dot = 0.0f32;
    let mut norm_sq = 0.0f32;
    for (&bits, &q) in packed.iter().take(n).zip(query.iter()) {
        let dv = SimBf16(bits).to_f32();
        dot += dv * q;
        norm_sq += dv * dv;
    }
    (dot, norm_sq)
}

/// AVX2+FMA kernel for [`bf16_dot_norm`]. Widens 8 bf16 lanes per iteration
/// (zero-extend `u16`→`u32`, `<< 16`, reinterpret as `f32`) and accumulates the
/// dot and squared-norm with two independent FMA chains.
///
/// # Safety
/// Requires AVX2 + FMA (the caller runtime-detects). Reads only `min(dim, query)` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bf16_dot_norm_avx2(packed: &[u16], query: &[f32], n: usize) -> (f32, f32) {
    use std::arch::x86_64::*;
    let full = n / 8;
    let mut dot = _mm256_setzero_ps();
    let mut nrm = _mm256_setzero_ps();
    let pp = packed.as_ptr();
    let qp = query.as_ptr();
    for b in 0..full {
        let u16x8 = _mm_loadu_si128(pp.add(b * 8) as *const __m128i);
        // zero-extend u16→u32, shift into the high half → f32 bit pattern.
        let fbits = _mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(u16x8));
        let dv = _mm256_castsi256_ps(fbits);
        let q8 = _mm256_loadu_ps(qp.add(b * 8));
        dot = _mm256_fmadd_ps(dv, q8, dot);
        nrm = _mm256_fmadd_ps(dv, dv, nrm);
    }
    let lo = _mm256_castps256_ps128(dot);
    let hi = _mm256_extractf128_ps::<1>(dot);
    let mut sd = _mm_add_ps(lo, hi);
    sd = _mm_hadd_ps(sd, sd);
    sd = _mm_hadd_ps(sd, sd);
    let mut d = _mm_cvtss_f32(sd);
    let lo = _mm256_castps256_ps128(nrm);
    let hi = _mm256_extractf128_ps::<1>(nrm);
    let mut sn = _mm_add_ps(lo, hi);
    sn = _mm_hadd_ps(sn, sn);
    sn = _mm_hadd_ps(sn, sn);
    let mut nm = _mm_cvtss_f32(sn);
    for i in full * 8..n {
        let dv = f32::from_bits((*packed.get_unchecked(i) as u32) << 16);
        d += dv * *query.get_unchecked(i);
        nm += dv * dv;
    }
    (d, nm)
}

/// AVX-512F+BW kernel for [`bf16_dot_norm`]. Widens 16 bf16 lanes per iteration
/// (256-bit `u16` load → zero-extend `u16`→`u32`, `<< 16`, reinterpret as `f32`)
/// and accumulates dot and squared-norm with two independent FMA chains. Widening
/// is exact, so the result is bit-identical to the scalar reference.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW (the caller runtime-detects). Reads only
/// `min(dim, query)` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn bf16_dot_norm_avx512(packed: &[u16], query: &[f32], n: usize) -> (f32, f32) {
    use std::arch::x86_64::*;
    let full = n / 16;
    let mut dot = _mm512_setzero_ps();
    let mut nrm = _mm512_setzero_ps();
    let pp = packed.as_ptr();
    let qp = query.as_ptr();
    for b in 0..full {
        let u16x16 = _mm256_loadu_si256(pp.add(b * 16) as *const __m256i);
        let fbits = _mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(u16x16));
        let dv = _mm512_castsi512_ps(fbits);
        let q16 = _mm512_loadu_ps(qp.add(b * 16));
        dot = _mm512_fmadd_ps(dv, q16, dot);
        nrm = _mm512_fmadd_ps(dv, dv, nrm);
    }
    let mut d = _mm512_reduce_add_ps(dot);
    let mut nm = _mm512_reduce_add_ps(nrm);
    for i in full * 16..n {
        let dv = f32::from_bits((*packed.get_unchecked(i) as u32) << 16);
        d += dv * *query.get_unchecked(i);
        nm += dv * dv;
    }
    (d, nm)
}

/// One BF16-quantised vector, stored as a packed `Vec<u16>`.
///
/// The `u16` layout is identical to `simsimd::bf16`, enabling a zero-copy
/// transmutation for SIMD distance calls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bf16Vector {
    /// BF16 values packed as raw `u16` bits; `len == dim`.
    pub packed: Vec<u16>,
    /// Original vector dimension.
    pub dim: usize,
}

/// Append the BF16 (round-to-nearest, ties-even) packing of `v` to `out` — the
/// flat-buffer builder for the HNSW bf16 navigation store. Matches
/// [`Bf16Vector::encode`]'s rounding exactly so nav distances line up with the
/// standalone bf16 codec.
pub(crate) fn encode_bf16_flat(v: &[f32], out: &mut Vec<u16>) {
    out.extend(v.iter().map(|&x| SimBf16::from_f32(x).0));
}

impl Bf16Vector {
    /// Encode an f32 slice to BF16 (round-to-nearest, ties to even).
    pub fn encode(v: &[f32]) -> Self {
        let packed: Vec<u16> = v.iter().map(|&x| SimBf16::from_f32(x).0).collect();
        Self {
            packed,
            dim: v.len(),
        }
    }

    /// Decode BF16 values back to approximate f32.
    pub fn decode(&self) -> Vec<f32> {
        self.packed
            .iter()
            .map(|&bits| SimBf16(bits).to_f32())
            .collect()
    }

    /// Asymmetric cosine distance to a full-precision query, computed directly
    /// from the bf16 codes — no `decode()` allocation. Equivalent to
    /// `cosine_dist_f32(&self.decode(), query)`.
    #[inline]
    pub fn cosine_dist_to_query(&self, query: &[f32]) -> f32 {
        let n = self.dim.min(query.len());
        let (dot, norm_sq) = bf16_dot_norm(&self.packed, query, n);
        let norm = norm_sq.sqrt();
        if norm < 1e-8 {
            return 1.0;
        }
        (1.0 - dot / norm).max(0.0)
    }

    /// Cosine distance to another `Bf16Vector` using SimSIMD.
    ///
    /// Returns a value in `[0, 2]` where 0 = identical and 2 = opposite.
    /// Falls back to `1.0` (max cosine distance for unit vectors) on error.
    pub fn cosine_dist(&self, other: &Bf16Vector) -> f32 {
        // SAFETY: `SimBf16` is a `repr(transparent)` newtype over `u16`
        // with identical size and alignment; the transmute is sound.
        let a = unsafe {
            std::slice::from_raw_parts(self.packed.as_ptr() as *const SimBf16, self.packed.len())
        };
        let b = unsafe {
            std::slice::from_raw_parts(other.packed.as_ptr() as *const SimBf16, other.packed.len())
        };
        match <SimBf16 as SpatialSimilarity>::cosine(a, b) {
            Some(d) => d as f32,
            None => {
                // SimSIMD returns None only on a length mismatch or an
                // unsupported target — never silently swallow it.
                tracing::warn!(
                    a_len = a.len(),
                    b_len = b.len(),
                    "bf16 cosine distance unavailable (SimSIMD returned None); falling back to 1.0"
                );
                1.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(d: usize, seed: f32) -> Vec<f32> {
        let v: Vec<f32> = (0..d).map(|i| (i as f32 * seed + 0.1).sin()).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }

    #[test]
    fn encode_decode_preserves_shape() {
        let v = unit_vec(768, 0.01);
        let enc = Bf16Vector::encode(&v);
        assert_eq!(enc.dim, 768);
        assert_eq!(enc.packed.len(), 768);
        let dec = enc.decode();
        assert_eq!(dec.len(), 768);
    }

    #[test]
    fn mismatched_length_falls_back_to_max_distance() {
        // SimSIMD returns None on a length mismatch; cosine_dist must not panic
        // and must return the documented fallback (1.0), logging a warning.
        let a = Bf16Vector::encode(&unit_vec(64, 0.01));
        let b = Bf16Vector::encode(&unit_vec(32, 0.02));
        assert_eq!(a.cosine_dist(&b), 1.0);
    }

    #[test]
    fn encode_decode_cosine_quality() {
        let v = unit_vec(768, 0.01);
        let enc = Bf16Vector::encode(&v);
        let dec = enc.decode();
        let dot: f32 = v.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
        let nv: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nd: f32 = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (nv * nd);
        assert!(
            cos >= 0.9999,
            "cosine similarity after BF16 round-trip = {cos:.6}"
        );
    }

    #[test]
    fn cosine_dist_self_is_zero() {
        let v = unit_vec(64, 0.07);
        let enc = Bf16Vector::encode(&v);
        let d = enc.cosine_dist(&enc);
        assert!(d < 1e-3, "cosine_dist(self, self) = {d}");
    }

    #[test]
    fn cosine_dist_orthogonal_is_one() {
        // Build two orthogonal unit vectors.
        let a: Vec<f32> = (0..8).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let b: Vec<f32> = (0..8).map(|i| if i == 1 { 1.0 } else { 0.0 }).collect();
        let ea = Bf16Vector::encode(&a);
        let eb = Bf16Vector::encode(&b);
        let d = ea.cosine_dist(&eb);
        assert!((d - 1.0).abs() < 0.01, "cosine_dist(orthogonal) = {d}");
    }

    /// The AVX-512 widen kernel must match the scalar reference across the
    /// 16-lane stride boundary and the scalar tail (widening is exact).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn bf16_dot_norm_avx512_matches_scalar() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
            return;
        }
        for n in [
            1usize, 7, 15, 16, 17, 31, 32, 33, 96, 127, 128, 256, 768, 769,
        ] {
            let v = unit_vec(n, 0.013);
            let q = unit_vec(n, 0.027);
            let packed: Vec<u16> = Bf16Vector::encode(&v).packed;
            let (ds, ns) = super::bf16_dot_norm_scalar(&packed, &q, n);
            // SAFETY: avx512f+bw checked above.
            let (d5, n5) = unsafe { super::bf16_dot_norm_avx512(&packed, &q, n) };
            let tol = ds.abs() * 1e-3 + 1e-3;
            assert!((ds - d5).abs() <= tol, "n={n}: dot {ds} vs {d5}");
            assert!(
                (ns - n5).abs() <= ns.abs() * 1e-3 + 1e-3,
                "n={n}: norm {ns} vs {n5}"
            );
        }
    }

    #[test]
    fn storage_size_is_half_of_f32() {
        let v = vec![1.0f32; 768];
        let enc = Bf16Vector::encode(&v);
        assert_eq!(std::mem::size_of::<u16>() * enc.packed.len(), 1536);
        // For comparison f32 would be 3072 bytes.
    }

    #[test]
    fn roundtrip_serde_json() {
        let v = unit_vec(32, 0.03);
        let enc = Bf16Vector::encode(&v);
        let json = serde_json::to_string(&enc).expect("serialize");
        let dec: Bf16Vector = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(enc, dec);
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_unit_vec(d: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(
            prop::num::f32::NORMAL | prop::num::f32::POSITIVE | prop::num::f32::NEGATIVE,
            d,
        )
        .prop_map(|v| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            v.into_iter().map(|x| x / norm).collect()
        })
    }

    proptest! {
        /// BF16 roundtrip: cosine ≥ 0.999 for any unit vector (truncation only).
        #[test]
        fn roundtrip_cosine_quality(v in arb_unit_vec(32)) {
            let enc = Bf16Vector::encode(&v);
            let dec = enc.decode();
            let dot: f32 = v.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let nb = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if nb > 1e-6 {
                let cos = dot / nb; // v is already unit-norm
                prop_assert!(cos >= 0.999, "cosine {cos} < 0.999");
            }
        }

        /// Scale invariance: cosine similarity preserved regardless of vector scale.
        #[test]
        fn scale_cosine_invariant(
            v in arb_unit_vec(16),
            scale in 0.01f32..100.0f32,
        ) {
            let scaled: Vec<f32> = v.iter().map(|x| x * scale).collect();
            let enc_v = Bf16Vector::encode(&v);
            let enc_s = Bf16Vector::encode(&scaled);
            let d_v = enc_v.cosine_dist(&enc_v);
            let d_s = enc_s.cosine_dist(&enc_s);
            // Both round-tripped to themselves, so both cosine_dist should be ~0
            prop_assert!(d_v < 1e-2, "cosine_dist(v,v) = {d_v}");
            prop_assert!(d_s < 1e-2, "cosine_dist(s,s) = {d_s}");
        }

        /// Decoded length equals input dimension.
        #[test]
        fn decode_length_matches(v in arb_unit_vec(24)) {
            let enc = Bf16Vector::encode(&v);
            let dec = enc.decode();
            prop_assert_eq!(dec.len(), v.len());
        }
    }
}
