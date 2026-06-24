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

/// Find the NF4 level index nearest to `x` via the midpoint thresholds.
/// `x` must be in [-1, 1].
///
/// `NF4_MIDS` is strictly increasing, so the binary-search result equals the
/// count of midpoints `x` meets or exceeds: `#{ i : x >= MIDS[i] }`. This
/// branchless form (compare → `setcc` → add, no data-dependent branches) avoids
/// the binary search's branch mispredictions and lets the per-element encode
/// loop auto-vectorize — a real win over the 4-comparison search.
#[inline]
fn nearest_nf4(x: f32) -> u8 {
    let mut idx = 0u8;
    let mut i = 0;
    while i < NF4_MIDS.len() {
        idx += u8::from(x >= NF4_MIDS[i]);
        i += 1;
    }
    idx
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

    /// Asymmetric cosine distance to a full-precision query, computed directly
    /// from the packed nibbles via codebook lookup — no `decode()` allocation.
    /// Equivalent to `cosine_dist_f32(&self.decode(), query)`.
    ///
    /// On x86-64 with AVX2+FMA this dispatches to an in-register codebook LUT
    /// (`vpermps` ×2 + blend) that scores 8 dims/iter — ~4–4.7× the scalar path
    /// (measured d=96..1024). NF4 was the last scalar quant distance; this brings
    /// it on par with INT8/SQ2/SQ3/BF16. Every other host uses the scalar path,
    /// which remains the correctness baseline.
    #[inline]
    pub fn cosine_dist_to_query(&self, query: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: gated by runtime AVX2+FMA detection; the kernel reads
                // `packed[..dim/2]` and `query[..dim]`, both in bounds.
                return unsafe { self.cosine_dist_to_query_avx2(query) };
            }
        }
        self.cosine_dist_to_query_scalar(query)
    }

    /// Portable scalar reference for [`Nf4Vector::cosine_dist_to_query`].
    #[inline]
    pub(crate) fn cosine_dist_to_query_scalar(&self, query: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_sq = 0.0f32;
        let mut i = 0;
        while i + 1 < self.dim {
            let byte = self.packed[i / 2];
            let lo = NF4_LEVELS[(byte & 0x0F) as usize] * self.scale;
            let hi = NF4_LEVELS[((byte >> 4) & 0x0F) as usize] * self.scale;
            dot += lo * query[i] + hi * query[i + 1];
            norm_sq += lo * lo + hi * hi;
            i += 2;
        }
        if self.dim % 2 == 1 {
            let byte = self.packed[self.packed.len() - 1];
            let lo = NF4_LEVELS[(byte & 0x0F) as usize] * self.scale;
            dot += lo * query[self.dim - 1];
            norm_sq += lo * lo;
        }
        let norm = norm_sq.sqrt();
        if norm < 1e-8 {
            return 1.0;
        }
        (1.0 - dot / norm).max(0.0)
    }

    /// AVX2+FMA NF4 asymmetric distance via an in-register 16-entry codebook LUT.
    ///
    /// Per 16 packed bytes (32 dims): unpack low/high nibbles, interleave back to
    /// dim order (`unpacklo/hi_epi8`), then for each group of 8 indices look up
    /// the level with two `permutevar8x32_ps` (low/high halves of `NF4_LEVELS`)
    /// blended on bit-3 of the index, scale, and accumulate dot (`level·query`)
    /// and squared norm (`level·level`) with FMA. Scalar tail for the remainder.
    ///
    /// # Safety
    /// Requires AVX2 + FMA (caller runtime-detects). Reads `packed[..dim/2]` and
    /// `query[..dim]`, both in bounds for a valid `Nf4Vector`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn cosine_dist_to_query_avx2(&self, query: &[f32]) -> f32 {
        let lo_t = _mm256_loadu_ps(NF4_LEVELS.as_ptr());
        let hi_t = _mm256_loadu_ps(NF4_LEVELS.as_ptr().add(8));
        let scale_v = _mm256_set1_ps(self.scale);
        let mut dacc = _mm256_setzero_ps();
        let mut nacc = _mm256_setzero_ps();
        let qptr = query.as_ptr();
        let pptr = self.packed.as_ptr();

        let pairs = self.dim / 2; // full (lo,hi) byte pairs
        let bytes16 = pairs / 16 * 16; // 16 bytes (32 dims) per outer iteration
        let mut b = 0usize;
        while b < bytes16 {
            let v = _mm_loadu_si128(pptr.add(b) as *const __m128i);
            let lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
            let hi = _mm_and_si128(_mm_srli_epi16(v, 4), _mm_set1_epi8(0x0F));
            // Interleave back to dim order: lo[0],hi[0],lo[1],hi[1],...
            let halves = [_mm_unpacklo_epi8(lo, hi), _mm_unpackhi_epi8(lo, hi)];
            for (half, dvec) in halves.iter().enumerate() {
                for g in 0..2 {
                    let idx8 = if g == 0 {
                        _mm256_cvtepu8_epi32(*dvec)
                    } else {
                        _mm256_cvtepu8_epi32(_mm_srli_si128(*dvec, 8))
                    };
                    let l = _mm256_permutevar8x32_ps(lo_t, idx8);
                    let h = _mm256_permutevar8x32_ps(hi_t, idx8);
                    // Select the high-half table when index ≥ 8 (bit-3 set).
                    let mask = _mm256_castsi256_ps(_mm256_slli_epi32(idx8, 28));
                    let lvl = _mm256_mul_ps(_mm256_blendv_ps(l, h, mask), scale_v);
                    let dimbase = b * 2 + half * 16 + g * 8;
                    let q = _mm256_loadu_ps(qptr.add(dimbase));
                    dacc = _mm256_fmadd_ps(lvl, q, dacc);
                    nacc = _mm256_fmadd_ps(lvl, lvl, nacc);
                }
            }
            b += 16;
        }

        let mut dot = hsum256(dacc);
        let mut norm_sq = hsum256(nacc);
        // Scalar tail: remaining byte-pairs below the 16-byte stride + odd dim.
        let mut i = b * 2;
        while i + 1 < self.dim {
            let byte = self.packed[i / 2];
            let lo = NF4_LEVELS[(byte & 0x0F) as usize] * self.scale;
            let hi = NF4_LEVELS[((byte >> 4) & 0x0F) as usize] * self.scale;
            dot += lo * query[i] + hi * query[i + 1];
            norm_sq += lo * lo + hi * hi;
            i += 2;
        }
        if self.dim % 2 == 1 {
            let byte = self.packed[self.packed.len() - 1];
            let lo = NF4_LEVELS[(byte & 0x0F) as usize] * self.scale;
            dot += lo * query[self.dim - 1];
            norm_sq += lo * lo;
        }
        let norm = norm_sq.sqrt();
        if norm < 1e-8 {
            return 1.0;
        }
        (1.0 - dot / norm).max(0.0)
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
/// Horizontal sum of an `f32x8` via the shuffle/add ladder (no `haddps`).
///
/// # Safety
/// Requires AVX (caller is `#[target_feature]`-gated).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let mut s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
    _mm_cvtss_f32(s)
}

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

/// Encode one vector's NF4 nibbles directly into `packed_out` (length
/// `ceil(d/2)`), returning the abs-max scale. No per-vector heap allocation —
/// the packing trick of [`Nf4Vector::encode_with_absmax`] writing straight into
/// a caller-owned slice, for the zero-copy batch FFI path.
#[inline]
pub fn encode_packed_into(v: &[f32], packed_out: &mut [u8]) -> f32 {
    let dim = v.len();
    #[cfg(target_arch = "x86_64")]
    let abs_max = if is_x86_feature_detected!("avx2") {
        // SAFETY: avx2 runtime-detected.
        unsafe { avx2_abs_max(v) }
    } else {
        v.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
    let inv = 1.0 / scale;
    let mut i = 0;
    while i + 1 < dim {
        let lo = nearest_nf4((v[i] * inv).clamp(-1.0, 1.0));
        let hi = nearest_nf4((v[i + 1] * inv).clamp(-1.0, 1.0));
        packed_out[i / 2] = lo | (hi << 4);
        i += 2;
    }
    if dim % 2 == 1 {
        let lo = nearest_nf4((v[dim - 1] * inv).clamp(-1.0, 1.0));
        packed_out[dim / 2] = lo;
    }
    scale
}

/// Encode a flat `[n, d]` row-major f32 batch to NF4 directly into caller-owned
/// strided buffers — `packed_out` is `[n, ceil(d/2)]` row-major u8, `scales_out`
/// is `[n]` f32. Parallel over rows; no per-row allocation. Backs the zero-copy
/// `quantize_nf4_batch` PyO3 entry (replaces the per-row `row.tolist()` FFI
/// loop). Every output byte/scale is written, so the caller may pass
/// uninitialised buffers.
pub fn batch_encode_packed_into(
    flat: &[f32],
    n: usize,
    d: usize,
    packed_out: &mut [u8],
    scales_out: &mut [f32],
) {
    if d == 0 || n == 0 {
        scales_out.iter_mut().for_each(|s| *s = 1.0);
        return;
    }
    let bpv = d.div_ceil(2);
    packed_out
        .par_chunks_mut(bpv)
        .zip(scales_out.par_iter_mut())
        .enumerate()
        .for_each(|(i, (prow, srow))| {
            *srow = encode_packed_into(&flat[i * d..i * d + d], prow);
        });
}

/// Decode a batch of NF4 vectors back to f32 in parallel.
pub fn decode_batch(encoded: &[Nf4Vector]) -> Vec<Vec<f32>> {
    encoded.par_iter().map(|e| e.decode()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flat zero-copy batch encode must produce byte-identical packing and
    /// scales to the per-vector `encode_fast`, across even and odd dims.
    #[test]
    fn batch_encode_packed_matches_per_vector() {
        for d in [1usize, 2, 15, 16, 31, 64, 127, 768, 769] {
            let n = 5;
            let flat: Vec<f32> = (0..n * d)
                .map(|i| ((i as f32 * 0.013) - 0.5).sin() * ((i % 7) as f32 + 1.0))
                .collect();
            let bpv = d.div_ceil(2);
            let mut packed = vec![0u8; n * bpv];
            let mut scales = vec![0.0f32; n];
            batch_encode_packed_into(&flat, n, d, &mut packed, &mut scales);
            for r in 0..n {
                let enc = Nf4Vector::encode_fast(&flat[r * d..r * d + d]);
                assert_eq!(&packed[r * bpv..r * bpv + bpv], &enc.packed[..], "d={d} row={r}");
                assert_eq!(scales[r], enc.scale, "d={d} row={r} scale");
            }
        }
    }

    /// The AVX2 LUT distance must match the scalar reference across the SIMD
    /// stride boundary (32 dims), partial blocks, and the odd-dim tail.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn cosine_dist_avx2_matches_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for dim in [1usize, 2, 15, 16, 31, 32, 33, 63, 64, 96, 127, 128, 129, 256, 768, 769] {
            let v: Vec<f32> = (0..dim).map(|i| ((i as f32 * 0.013) - 0.5).sin()).collect();
            let q: Vec<f32> = (0..dim).map(|i| ((i as f32 * 0.027) + 0.2).cos()).collect();
            let enc = Nf4Vector::encode_fast(&v);
            let scalar = enc.cosine_dist_to_query_scalar(&q);
            // SAFETY: avx2+fma checked above.
            let simd = unsafe { enc.cosine_dist_to_query_avx2(&q) };
            assert!((scalar - simd).abs() < 1e-4, "dim={dim}: scalar={scalar} simd={simd}");
        }
    }

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

        /// Scale invariance: encoding `v` and `α·v` (α > 0) yields NF4 codes
        /// that agree up to a single adjacent level.
        ///
        /// Exact byte equality does *not* hold in general.  The strategy admits
        /// magnitudes up to 1e18, where the float product `x·α` and the scaled
        /// abs-max round such that the normalised value `x·α / abs_max` lands
        /// ~1 ULP across an NF4 level boundary, flipping that nibble to the
        /// neighbouring level.  That perturbation (~1e-7 relative) is far below
        /// the ~0.04 minimum spacing between NF4 decision boundaries, so a code
        /// can shift by at most one level — which is the honest invariant here.
        #[test]
        fn scale_invariance(
            v in arb_nonzero_vec(16),
            scale in 0.1f32..20.0f32,
        ) {
            // With max |v[i]| ≤ 1e18 and scale ≤ 20, max scaled value ≤ 2e19 << f32::MAX.
            let scaled: Vec<f32> = v.iter().map(|x| x * scale).collect();
            let enc1 = Nf4Vector::encode(&v);
            let enc2 = Nf4Vector::encode(&scaled);
            prop_assert_eq!(enc1.packed.len(), enc2.packed.len());
            for (b, (&p1, &p2)) in enc1.packed.iter().zip(enc2.packed.iter()).enumerate() {
                // Each byte holds two 4-bit codes: low nibble then high nibble.
                for shift in [0u8, 4u8] {
                    let c1 = ((p1 >> shift) & 0x0F) as i16;
                    let c2 = ((p2 >> shift) & 0x0F) as i16;
                    prop_assert!(
                        (c1 - c2).abs() <= 1,
                        "NF4 code differs by >1 level at byte {b} (nibble shift {shift}) \
                         under scale {scale}: {c1} vs {c2}"
                    );
                }
            }
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
