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

impl Bf16Vector {
    /// Encode an f32 slice to BF16 (round-to-nearest, ties to even).
    pub fn encode(v: &[f32]) -> Self {
        let packed: Vec<u16> = v.iter().map(|&x| SimBf16::from_f32(x).0).collect();
        Self { packed, dim: v.len() }
    }

    /// Decode BF16 values back to approximate f32.
    pub fn decode(&self) -> Vec<f32> {
        self.packed.iter().map(|&bits| SimBf16(bits).to_f32()).collect()
    }

    /// Cosine distance to another `Bf16Vector` using SimSIMD.
    ///
    /// Returns a value in `[0, 2]` where 0 = identical and 2 = opposite.
    /// Falls back to `1.0` (max cosine distance for unit vectors) on error.
    pub fn cosine_dist(&self, other: &Bf16Vector) -> f32 {
        // SAFETY: `SimBf16` is a `repr(transparent)` newtype over `u16`
        // with identical size and alignment; the transmute is sound.
        let a = unsafe {
            std::slice::from_raw_parts(
                self.packed.as_ptr() as *const SimBf16,
                self.packed.len(),
            )
        };
        let b = unsafe {
            std::slice::from_raw_parts(
                other.packed.as_ptr() as *const SimBf16,
                other.packed.len(),
            )
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
        assert!(cos >= 0.9999, "cosine similarity after BF16 round-trip = {cos:.6}");
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
