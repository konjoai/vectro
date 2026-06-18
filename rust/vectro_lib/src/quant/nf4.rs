//! NF4 (Normal Float 4-bit) quantization.
//!
//! Implements the NF4 encoding from Dettmers et al. 2023 ("QLoRA").
//! Each dimension is mapped to the nearest value in a 16-entry codebook whose
//! levels are the quantiles of the standard normal distribution, scaled per
//! vector by its abs-max.
//!
//! Storage: two 4-bit codes are packed into one u8 (low nibble = even dim,
//! high nibble = odd dim).  This gives exactly `ceil(d/2)` bytes per vector.
//!
//! Algorithm parity target (from PLAN.md Phase 16):
//!   cosine similarity of decode(encode(v)) vs v  ≥ 0.985

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// NF4 codebook — quantiles of N(0, 1), exactly reproducing the Python reference.
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.525_073,
    -0.3949003,
    -0.2844677,
    -0.1848745,
    -0.09105004,
    0.0,
    0.07958031,
    0.16093908,
    0.24611496,
    0.33791524,
    0.44070983,
    0.56266755,
    0.722_957_6,
    1.0,
];

/// Mid-points between adjacent NF4 levels, used for nearest-neighbour search.
const NF4_MIDS: [f32; 15] = {
    let mut m = [0.0f32; 15];
    let mut i = 0;
    while i < 15 {
        m[i] = (NF4_LEVELS[i] + NF4_LEVELS[i + 1]) * 0.5;
        i += 1;
    }
    m
};

/// Find the NF4 level index nearest to `x` via binary search on midpoints.
/// `x` must be in [-1, 1].
#[inline]
fn nearest_nf4(x: f32) -> u8 {
    // binary search: find first mid where x < mid → index before that
    let mut lo = 0usize;
    let mut hi = 15usize;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x < NF4_MIDS[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo as u8
}

/// One NF4-quantized vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nf4Vector {
    /// Packed 4-bit codes: low nibble = even index, high nibble = odd index.
    pub packed: Vec<u8>,
    /// Per-vector abs-max scale factor.
    pub scale: f32,
    /// Original dimension (needed for decode when d is odd).
    pub dim: usize,
}

impl Nf4Vector {
    /// Encode a single f32 slice to packed NF4.
    pub fn encode(v: &[f32]) -> Self {
        let dim = v.len();
        let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
        let inv = 1.0 / scale;

        let bytes_per_vec = dim.div_ceil(2);
        let mut packed = vec![0u8; bytes_per_vec];

        let mut i = 0;
        while i + 1 < dim {
            let lo = nearest_nf4((v[i] * inv).clamp(-1.0, 1.0));
            let hi = nearest_nf4((v[i + 1] * inv).clamp(-1.0, 1.0));
            packed[i / 2] = lo | (hi << 4);
            i += 2;
        }
        if dim % 2 == 1 {
            let lo = nearest_nf4((v[dim - 1] * inv).clamp(-1.0, 1.0));
            packed[bytes_per_vec - 1] = lo;
        }

        Self { packed, scale, dim }
    }

    /// Encode using a platform-optimised abs-max pass when available.
    ///
    /// On x86-64 with AVX2 the abs-max scan uses 256-bit SIMD (8-wide).
    /// The nibble quantisation loop stays scalar because it is a table lookup
    /// that doesn't benefit from float SIMD.  Falls back to `encode` on other
    /// targets.
    pub fn encode_fast(v: &[f32]) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: we checked avx2 at runtime.
                let abs_max = unsafe { avx2_abs_max(v) };
                return Self::encode_with_absmax(v, abs_max);
            }
        }
        // aarch64 NEON: use fold-based abs-max (compiler auto-vectorises well)
        #[cfg(target_arch = "aarch64")]
        {
            let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            return Self::encode_with_absmax(v, abs_max);
        }
        #[allow(unreachable_code)]
        Self::encode(v)
    }

    /// Internal: encode given a pre-computed abs-max.
    fn encode_with_absmax(v: &[f32], abs_max: f32) -> Self {
        let dim = v.len();
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
        let inv = 1.0 / scale;

        let bytes_per_vec = dim.div_ceil(2);
        let mut packed = vec![0u8; bytes_per_vec];

        let mut i = 0;
        while i + 1 < dim {
            let lo = nearest_nf4((v[i] * inv).clamp(-1.0, 1.0));
            let hi = nearest_nf4((v[i + 1] * inv).clamp(-1.0, 1.0));
            packed[i / 2] = lo | (hi << 4);
            i += 2;
        }
        if dim % 2 == 1 {
            let lo = nearest_nf4((v[dim - 1] * inv).clamp(-1.0, 1.0));
            packed[bytes_per_vec - 1] = lo;
        }

        Self { packed, scale, dim }
    }

    pub fn decode(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dim);
        let mut i = 0;
        while i + 1 < self.dim {
            let byte = self.packed[i / 2];
            out.push(NF4_LEVELS[(byte & 0x0F) as usize] * self.scale);
            out.push(NF4_LEVELS[((byte >> 4) & 0x0F) as usize] * self.scale);
            i += 2;
        }
        if self.dim % 2 == 1 {
            let byte = self.packed[self.packed.len() - 1];
            out.push(NF4_LEVELS[(byte & 0x0F) as usize] * self.scale);
        }
        out
    }
}

/// AVX2 horizontal abs-max over a f32 slice.
///
/// Processes 8 floats per iteration with 256-bit registers.
/// # Safety
/// Caller must ensure `avx2` is available (`is_x86_feature_detected!("avx2")`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_abs_max(v: &[f32]) -> f32 {
    let n = v.len();
    let ptr = v.as_ptr();

    // Sign-mask: clear sign bit of each f32 lane → |x|
    let sign_mask = _mm256_set1_ps(-0.0f32);

    let mut acc = _mm256_setzero_ps();
    let chunks = n / 8;
    for i in 0..chunks {
        let vals = _mm256_loadu_ps(ptr.add(i * 8));
        let abs_vals = _mm256_andnot_ps(sign_mask, vals); // |v[i]|
        acc = _mm256_max_ps(acc, abs_vals);
    }

    // Horizontal max across 8 lanes
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let max128 = _mm_max_ps(lo128, hi128);
    // Shuffle hi64 → lo64, take max
    let shuf = _mm_movehl_ps(max128, max128);
    let max64 = _mm_max_ps(max128, shuf);
    // max of two remaining lanes
    let shuf2 = _mm_shuffle_ps(max64, max64, 0x55);
    let max32 = _mm_max_ss(max64, shuf2);
    let mut result = _mm_cvtss_f32(max32);

    // Scalar tail
    let tail_start = chunks * 8;
    for &x in &v[tail_start..n] {
        let val = x.abs();
        if val > result {
            result = val;
        }
    }
    result
}

/// Encode a batch of f32 vectors to NF4 in parallel.
pub fn encode_batch(vectors: &[Vec<f32>]) -> Vec<Nf4Vector> {
    vectors.par_iter().map(|v| Nf4Vector::encode_fast(v)).collect()
}

/// Decode a batch of NF4 vectors back to f32 in parallel.
pub fn decode_batch(encoded: &[Nf4Vector]) -> Vec<Vec<f32>> {
    encoded.par_iter().map(|e| e.decode()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: cosine similarity between two equal-length slices.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { return -1.0; }
        dot / (na * nb)
    }

    #[test]
    fn roundtrip_cosine_quality() {
        // d=768, normally-distributed values — matches the primary benchmark vector shape
        let v: Vec<f32> = (0..768).map(|i| ((i as f32 * 0.007) - 2.688).sin() * 0.8).collect();
        let enc = Nf4Vector::encode(&v);
        let dec = enc.decode();
        assert_eq!(dec.len(), 768);
        let cos = cosine(&v, &dec);
        assert!(cos >= 0.985, "cosine {cos} < 0.985 (NF4 parity spec)");
    }

    #[test]
    fn zero_vector() {
        let v = vec![0.0f32; 64];
        let enc = Nf4Vector::encode(&v);
        assert_eq!(enc.scale, 1.0);
        let dec = enc.decode();
        // All decoded values should be NF4_LEVELS[nearest_nf4(0)] * 1.0 = 0.0
        for x in &dec { assert!(x.abs() < 1e-5); }
    }

    #[test]
    fn odd_dimension() {
        let v = vec![0.3f32, -0.7, 0.5, -0.1, 0.9];
        let enc = Nf4Vector::encode(&v);
        let dec = enc.decode();
        assert_eq!(dec.len(), 5);
        let cos = cosine(&v, &dec);
        assert!(cos >= 0.98, "odd-dim cosine {cos}");
    }

    #[test]
    fn packing_round_trip_short() {
        // Two elements → 1 byte.
        let v = vec![1.0f32, -1.0];
        let enc = Nf4Vector::encode(&v);
        assert_eq!(enc.packed.len(), 1);
        let dec = enc.decode();
        // level[0] * scale ≈ -1, level[15] * scale ≈ +1
        assert!((dec[0] - 1.0).abs() < 1e-5, "got {}", dec[0]);
        assert!((dec[1] + 1.0).abs() < 1e-5, "got {}", dec[1]);
    }

    #[test]
    fn batch_encode_decode() {
        let vecs: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..128).map(|j| ((i * j) as f32 * 0.01).sin()).collect())
            .collect();
        let encoded = encode_batch(&vecs);
        let decoded = decode_batch(&encoded);
        for (orig, dec) in vecs.iter().zip(decoded.iter()) {
            let cos = cosine(orig, dec);
            if orig.iter().any(|x| *x != 0.0) {
                assert!(cos >= 0.985, "batch cosine {cos} < 0.985");
            }
        }
    }

    #[test]
    fn binary_search_midpoints() {
        // nearest_nf4(0) should return index 7 (NF4_LEVELS[7] = 0.0)
        assert_eq!(nearest_nf4(0.0), 7);
        // nearest_nf4(-1.0) → index 0
        assert_eq!(nearest_nf4(-1.0), 0);
        // nearest_nf4(1.0) → index 15
        assert_eq!(nearest_nf4(1.0), 15);
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_nonzero_vec(d: usize) -> impl Strategy<Value = Vec<f32>> {
        // Use a range that covers wide dynamic range but keeps x² * d well below
        // f32::MAX (3.4e38).  With max |x| ≤ 1e18: sum(x²) ≤ d * 1e36 ≤ 3.2e37.
        prop::collection::vec(-1e18f32..1e18f32, d)
            .prop_filter("degenerate zero vector", |v| {
                v.iter().any(|x| x.abs() > 1e-6)
            })
    }

    proptest! {
        /// Encode then decode should yield cosine ≥ 0.97 with a normal vector.
        #[test]
        fn roundtrip_cosine_quality(v in arb_nonzero_vec(32)) {
            let enc = Nf4Vector::encode(&v);
            let dec = enc.decode();
            let dot: f32 = v.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let na = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na > 1e-6 && nb > 1e-6 {
                let cos = dot / (na * nb);
                prop_assert!(cos >= 0.97, "cosine {cos} < 0.97 for v={v:?}");
            }
        }

        /// Scale invariance: encoding v and α·v (α > 0) gives same nibbles.
        #[test]
        fn scale_invariance(
            v in arb_nonzero_vec(16),
            scale in 0.1f32..20.0f32,
        ) {
            // With max |v[i]| ≤ 1e18 and scale ≤ 20, max scaled value ≤ 2e19 << f32::MAX.
            let scaled: Vec<f32> = v.iter().map(|x| x * scale).collect();
            let enc1 = Nf4Vector::encode(&v);
            let enc2 = Nf4Vector::encode(&scaled);
            prop_assert_eq!(enc1.packed, enc2.packed,
                "packed bytes differ under scale factor {}", scale);
        }

        /// Decoded length always equals the original dimension.
        #[test]
        fn decode_length_matches(v in arb_nonzero_vec(24)) {
            let enc = Nf4Vector::encode(&v);
            let dec = enc.decode();
            prop_assert_eq!(dec.len(), v.len());
        }
    }
}
