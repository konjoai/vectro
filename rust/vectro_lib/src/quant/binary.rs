//! Binary (1-bit sign) quantization.
//!
//! Each dimension is encoded as its sign bit (positive → 1, negative/zero → 0).
//! Eight bits are packed into one byte (LSB-first: bit 0 = dim 0).
//!
//! Storage: `ceil(d/8)` bytes per vector.
//!
//! Nearest-neighbour search uses **Hamming distance** on the packed bytes, which
//! is proportional to the number of differing sign bits.
//!
//! When `normalize = true` (default), each vector is L2-normalized before
//! encoding.  This makes the Hamming distance a monotone proxy for cosine
//! distance on unit vectors: `cos(θ) ≈ 1 - 2·hamming/d`.
//!
//! Recall@10 parity target (from PLAN.md Phase 16): ≥ 0.95 after re-ranking.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use simsimd::BinarySimilarity;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Signed dot `Σ s_i·query_i` where `s_i = +1` if packed bit `i` is set else
/// `−1` — the unnormalised core of the binary asymmetric cosine distance.
///
/// AVX2 sign-flip kernel on x86_64 (runtime-detected), scalar fallback
/// otherwise. The SIMD path expands 8 packed bits to per-lane masks and flips
/// the sign of each `query` lane whose bit is clear (`q XOR signbit`), then
/// accumulates — 8 dims/iter with no per-element branch.
#[inline]
fn signed_dot(packed: &[u8], query: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: gated by the runtime detection above; reads in-bounds lanes.
            return unsafe { signed_dot_avx2(packed, query, n) };
        }
    }
    signed_dot_scalar(packed, query, n)
}

/// Scalar reference for [`signed_dot`] — byte-major, branchless
/// (`sign = 2*bit − 1`). The correctness baseline the SIMD kernel must match.
#[inline]
fn signed_dot_scalar(packed: &[u8], query: &[f32], n: usize) -> f32 {
    let full = n / 8;
    let mut dot = 0.0f32;
    for (b, &byte) in packed.iter().take(full).enumerate() {
        let base = b * 8;
        for k in 0..8 {
            let sign = (((byte >> k) & 1) as f32) * 2.0 - 1.0;
            dot += sign * query[base + k];
        }
    }
    // Tail (<8 elements): needs both the packed-bit index and query[i].
    #[allow(clippy::needless_range_loop)]
    for i in full * 8..n {
        let sign = (((packed[i / 8] >> (i % 8)) & 1) as f32) * 2.0 - 1.0;
        dot += sign * query[i];
    }
    dot
}

/// AVX2 sign-flip kernel for [`signed_dot`]. Processes one packed byte (8 dims)
/// per iteration: broadcast the byte, AND with per-lane bit selectors, compare
/// to build a set-mask, then XOR the query lanes whose bit is *clear* with the
/// sign bit (negating them) before accumulating.
///
/// # Safety
/// Requires AVX2 (the caller runtime-detects). Reads only `min(dim, query)` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn signed_dot_avx2(packed: &[u8], query: &[f32], n: usize) -> f32 {
    let full = n / 8;
    // LSB-first bit selectors: lane l tests bit l of the byte (dim base+l).
    let bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    let signbit = _mm256_set1_epi32(i32::MIN); // 0x8000_0000
    let mut acc = _mm256_setzero_ps();
    let qp = query.as_ptr();
    for b in 0..full {
        let byte = *packed.get_unchecked(b) as i32;
        let bcast = _mm256_set1_epi32(byte);
        let sel = _mm256_and_si256(bcast, bits);
        // 0xFFFF_FFFF in lanes whose bit is set, 0 elsewhere.
        let setmask = _mm256_cmpeq_epi32(sel, bits);
        // sign bit only in lanes whose bit is *clear* → those query lanes negate.
        let flip = _mm256_andnot_si256(setmask, signbit);
        let q8 = _mm256_loadu_ps(qp.add(b * 8));
        let signed = _mm256_xor_ps(q8, _mm256_castsi256_ps(flip));
        acc = _mm256_add_ps(acc, signed);
    }
    // Horizontal sum of the 8 lanes.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps::<1>(acc);
    let mut s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    let mut dot = _mm_cvtss_f32(s);
    for i in full * 8..n {
        let sign = (((*packed.get_unchecked(i / 8) >> (i % 8)) & 1) as f32) * 2.0 - 1.0;
        dot += sign * *query.get_unchecked(i);
    }
    dot
}

/// One binary-quantized vector (packed bits, original dimension stored).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryVector {
    /// LSB-first packed sign bits: len = ceil(dim/8).
    pub packed: Vec<u8>,
    /// Original vector dimension.
    pub dim: usize,
}

impl BinaryVector {
    /// Encode a single f32 slice.  L2-normalizes the input before sign-packing
    /// when `normalize` is true.
    pub fn encode(v: &[f32], normalize: bool) -> Self {
        let dim = v.len();
        let bytes_per_vec = dim.div_ceil(8);
        let mut packed = vec![0u8; bytes_per_vec];

        // L2-normalization scales every element by the same strictly-positive
        // factor, which can never flip a sign — so for sign-packing the
        // `normalize` flag is a no-op and the per-element divide it used to do
        // was dead work. The bit is set iff the raw value is positive.
        let _ = normalize;
        for (i, &x) in v.iter().enumerate() {
            if x > 0.0 {
                packed[i / 8] |= 1u8 << (i % 8);
            }
        }

        Self { packed, dim }
    }

    /// SIMD-accelerated sign-pack. Dispatches AVX-512F (16 signs → a 2-byte mask
    /// per `vcmpps`) → AVX2 (8 signs → 1 byte via `vmovmskps`) → scalar, and is
    /// bit-for-bit identical to [`encode`]. The scalar loop's per-element branch
    /// and `packed[i/8] |= 1<<(i%8)` scatter defeat autovectorisation, so this is
    /// a genuine win rather than a re-expression the compiler already finds.
    /// `normalize` is a no-op (a positive scale can't flip a sign), matching
    /// [`encode`].
    pub fn encode_fast(v: &[f32], normalize: bool) -> Self {
        let _ = normalize;
        let dim = v.len();
        let bytes_per_vec = dim.div_ceil(8);
        let mut packed = vec![0u8; bytes_per_vec];

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: guarded by runtime AVX-512F detection.
                unsafe { pack_signs_avx512(v, &mut packed) };
                return Self { packed, dim };
            }
            if is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by runtime AVX2 detection.
                unsafe { pack_signs_avx2(v, &mut packed) };
                return Self { packed, dim };
            }
        }

        // Scalar fallback (non-x86, or x86 without AVX2/AVX-512F).
        for (i, &x) in v.iter().enumerate() {
            if x > 0.0 {
                packed[i / 8] |= 1u8 << (i % 8);
            }
        }

        Self { packed, dim }
    }
    pub fn decode(&self) -> Vec<f32> {
        (0..self.dim)
            .map(|i| {
                if (self.packed[i / 8] >> (i % 8)) & 1 == 1 {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
            .collect()
    }

    /// Asymmetric cosine distance to a full-precision unit query, computed
    /// directly from the packed sign bits — no `decode()` allocation.
    ///
    /// Equivalent to `cosine_dist_f32(&self.decode(), query)`: decoded values
    /// are ±1 so the norm is `sqrt(dim)` and the dot is `Σ ±query_i`. Called
    /// per candidate during HNSW search, so avoiding the per-call `Vec<f32>` is
    /// a large win.
    #[inline]
    pub fn cosine_dist_to_query(&self, query: &[f32]) -> f32 {
        let n = self.dim.min(query.len());
        let dot = signed_dot(&self.packed, query, n);
        let norm = (self.dim as f32).sqrt();
        if norm < 1e-8 {
            return 1.0;
        }
        (1.0 - dot / norm).max(0.0)
    }

    /// Hamming distance to another BinaryVector of the same dimension.
    ///
    /// Uses SimSIMD's SIMD popcount path (NEON/SVE on ARM, Haswell/Ice on x86)
    /// with a scalar fallback for other targets.
    pub fn hamming(&self, other: &BinaryVector) -> u32 {
        match <u8 as BinarySimilarity>::hamming(&self.packed, &other.packed) {
            Some(h) => h as u32,
            None => {
                // A length mismatch yields None; falling back to 0 would rank
                // these vectors as identical, so surface it rather than hide it.
                tracing::warn!(
                    a_bytes = self.packed.len(),
                    b_bytes = other.packed.len(),
                    "binary Hamming distance unavailable (SimSIMD returned None); falling back to 0"
                );
                0
            }
        }
    }
}

/// AVX-512F sign-bit pack: `_mm512_cmp_ps_mask(x, 0, GT)` yields 16 sign bits as
/// a `__mmask16` (lane j → bit j), which is exactly 2 LSB-first packed bytes.
///
/// # Safety
/// Caller must ensure AVX-512F is available; writes `2` bytes per 16-lane chunk
/// into `packed`, whose length is `ceil(v.len()/8)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pack_signs_avx512(v: &[f32], packed: &mut [u8]) {
    let n = v.len();
    let ptr = v.as_ptr();
    let zero = _mm512_setzero_ps();
    let chunks = n / 16;
    for c in 0..chunks {
        let x = _mm512_loadu_ps(ptr.add(c * 16));
        // bit j set iff lane j > 0.0 (ordered; NaN and ≤0 → 0), matching `encode`.
        let mask: u16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(x, zero);
        let b = c * 2;
        packed[b] = (mask & 0xFF) as u8;
        packed[b + 1] = (mask >> 8) as u8;
    }
    for i in chunks * 16..n {
        if v[i] > 0.0 {
            packed[i / 8] |= 1u8 << (i % 8);
        }
    }
}

/// AVX2 sign-bit pack: `vmovmskps` of `vcmpps(x, 0, GT)` yields 8 sign bits as a
/// byte (lane j → bit j) — one LSB-first packed byte per 8-lane chunk.
///
/// # Safety
/// Caller must ensure AVX2 is available; writes `1` byte per 8-lane chunk into
/// `packed`, whose length is `ceil(v.len()/8)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_signs_avx2(v: &[f32], packed: &mut [u8]) {
    let n = v.len();
    let ptr = v.as_ptr();
    let zero = _mm256_setzero_ps();
    let chunks = n / 8;
    for (c, slot) in packed[..chunks].iter_mut().enumerate() {
        let x = _mm256_loadu_ps(ptr.add(c * 8));
        let cmp = _mm256_cmp_ps::<_CMP_GT_OQ>(x, zero);
        *slot = _mm256_movemask_ps(cmp) as u8;
    }
    for i in chunks * 8..n {
        if v[i] > 0.0 {
            packed[i / 8] |= 1u8 << (i % 8);
        }
    }
}

/// Encode a batch of f32 vectors to binary in parallel.
pub fn encode_batch(vectors: &[Vec<f32>], normalize: bool) -> Vec<BinaryVector> {
    vectors
        .par_iter()
        .map(|v| BinaryVector::encode_fast(v, normalize))
        .collect()
}

/// Decode a batch of BinaryVectors back to f32 in parallel.
pub fn decode_batch(encoded: &[BinaryVector]) -> Vec<Vec<f32>> {
    encoded.par_iter().map(|e| e.decode()).collect()
}

/// Compute Hamming distances from a single query to all database vectors.
///
/// Returns a Vec of (index, hamming_distance) sorted by ascending distance.
pub fn hamming_search(
    query: &BinaryVector,
    database: &[BinaryVector],
    top_k: usize,
) -> Vec<(usize, u32)> {
    let mut dists: Vec<(usize, u32)> = database
        .par_iter()
        .enumerate()
        .map(|(i, bv)| (i, query.hamming(bv)))
        .collect();
    // Partial selection: O(n) to isolate the top_k smallest, then sort only
    // that prefix — far cheaper than a full O(n log n) sort when top_k ≪ n.
    let k = top_k.min(dists.len());
    if k > 0 && k < dists.len() {
        dists.select_nth_unstable_by_key(k - 1, |&(_, d)| d);
        dists.truncate(k);
    }
    dists.sort_by_key(|&(_, d)| d);
    dists.truncate(top_k);
    dists
}

/// Full binary search pipeline: encode query, search by Hamming, return indices.
pub fn binary_search(
    query: &[f32],
    database: &[BinaryVector],
    top_k: usize,
    normalize: bool,
) -> Vec<(usize, u32)> {
    let q = BinaryVector::encode(query, normalize);
    hamming_search(&q, database, top_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `encode_fast` (AVX-512F / AVX2 sign-pack) must be bit-for-bit identical to
    /// the scalar `encode` across SIMD-width boundaries, odd tails, exact zeros
    /// (→ 0 bit), and large magnitudes. Runs on every target; the SIMD paths only
    /// activate where the host advertises the feature.
    #[test]
    fn encode_fast_matches_scalar() {
        for &d in &[
            0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 768,
        ] {
            let v: Vec<f32> = (0..d)
                .map(|i| ((i as f32 * 0.37).sin() - 0.1) * 1e3)
                .collect();
            let scalar = BinaryVector::encode(&v, true);
            let fast = BinaryVector::encode_fast(&v, true);
            assert_eq!(scalar.packed, fast.packed, "packed mismatch at d={d}");
            assert_eq!(scalar.dim, fast.dim, "dim mismatch at d={d}");
        }
        // Exact zero must clear its bit in both paths (sign = x > 0.0).
        let z = vec![0.0f32; 20];
        assert_eq!(BinaryVector::encode_fast(&z, true).packed, vec![0u8; 3]);
    }

    #[test]
    fn packing_basic() {
        // v = [1, -1, 1, -1, 0, 0, 0, 0] → bits 0,2 set → byte = 0b0000_0101 = 5
        let v = vec![1.0f32, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0];
        let bv = BinaryVector::encode(&v, false);
        assert_eq!(bv.packed.len(), 1);
        assert_eq!(bv.packed[0], 0b0000_0101);
    }

    #[test]
    fn mismatched_length_falls_back_to_zero() {
        // SimSIMD returns None when byte lengths differ; hamming must not panic
        // and returns the documented fallback (0), logging a warning.
        let a = BinaryVector::encode(&vec![1.0f32; 64], false);
        let b = BinaryVector::encode(&[1.0f32; 32], false);
        assert_eq!(a.hamming(&b), 0);
    }

    #[test]
    fn decode_round_trip() {
        let v = vec![1.0f32, -1.0, 0.5, -0.5, 0.0, 1.0, -0.1, 0.9];
        let bv = BinaryVector::encode(&v, false);
        let dec = bv.decode();
        // Should get sign pattern correct
        for (orig, &d) in v.iter().zip(dec.iter()) {
            if *orig > 0.0 {
                assert_eq!(d, 1.0);
            } else {
                assert_eq!(d, -1.0);
            }
        }
    }

    #[test]
    fn hamming_distance_self() {
        let v = vec![0.3f32, -0.5, 0.8, -0.2];
        let bv = BinaryVector::encode(&v, false);
        assert_eq!(bv.hamming(&bv), 0);
    }

    #[test]
    fn hamming_distance_opposite() {
        // All-positive vs all-negative → every bit different
        let pos = vec![1.0f32; 8];
        let neg = vec![-1.0f32; 8];
        let bvp = BinaryVector::encode(&pos, false);
        let bvn = BinaryVector::encode(&neg, false);
        assert_eq!(bvp.hamming(&bvn), 8);
    }

    #[test]
    fn odd_dimension() {
        let v: Vec<f32> = (0..13)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let bv = BinaryVector::encode(&v, false);
        assert_eq!(bv.packed.len(), 2); // ceil(13/8)
        let dec = bv.decode();
        assert_eq!(dec.len(), 13);
    }

    #[test]
    fn hamming_search_nearest_first() {
        // Build a small database and verify nearest returns first.
        let db_vecs: Vec<Vec<f32>> = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![-1.0, -1.0, -1.0, -1.0],
            vec![1.0, -1.0, 1.0, -1.0],
        ];
        let db: Vec<BinaryVector> = db_vecs
            .iter()
            .map(|v| BinaryVector::encode(v, false))
            .collect();
        let query = vec![1.0f32, 1.0, 1.0, 1.0];
        let results = binary_search(&query, &db, 1, false);
        assert_eq!(results[0].0, 0); // exact match → index 0
        assert_eq!(results[0].1, 0); // Hamming distance 0
    }

    #[test]
    fn normalize_flag() {
        // Scaling a vector shouldn't change its encoded bits when normalized
        let v = vec![2.0f32, -4.0, 6.0, -8.0];
        let v_scaled = vec![4.0f32, -8.0, 12.0, -16.0];
        let bv1 = BinaryVector::encode(&v, true);
        let bv2 = BinaryVector::encode(&v_scaled, true);
        assert_eq!(bv1.packed, bv2.packed);
    }

    #[test]
    fn batch_encode_decode() {
        let vecs: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                (0..32)
                    .map(|j| if (i + j) % 2 == 0 { 0.5f32 } else { -0.5f32 })
                    .collect()
            })
            .collect();
        let encoded = encode_batch(&vecs, true);
        let decoded = decode_batch(&encoded);
        assert_eq!(decoded.len(), 20);
        assert_eq!(decoded[0].len(), 32);
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy: non-zero f32 vector of fixed dimension d
    fn arb_nonzero_vec(d: usize) -> impl Strategy<Value = Vec<f32>> {
        // Use a range where x² * d stays well below f32::MAX (3.4e38).
        // With max |x| ≤ 1e18: sum(x²) ≤ d * 1e36 ≤ 3.2e37 for d=32.
        prop::collection::vec(-1e18f32..1e18f32, d).prop_filter("degenerate zero vector", |v| {
            v.iter().any(|x| x.abs() > 1e-10)
        })
    }

    proptest! {
        /// Hamming distance is symmetric.
        #[test]
        fn hamming_symmetry(
            v1 in arb_nonzero_vec(32),
            v2 in arb_nonzero_vec(32),
        ) {
            let a = BinaryVector::encode(&v1, false);
            let b = BinaryVector::encode(&v2, false);
            prop_assert_eq!(a.hamming(&b), b.hamming(&a));
        }

        /// Hamming distance of a vector with itself is 0.
        #[test]
        fn hamming_self_zero(v in arb_nonzero_vec(32)) {
            let enc = BinaryVector::encode(&v, false);
            prop_assert_eq!(enc.hamming(&enc), 0);
        }

        /// Complementing every element flips every bit → Hamming == dim.
        #[test]
        fn hamming_complement_equals_dim(v in arb_nonzero_vec(8)) {
            let d = v.len();
            let negated: Vec<f32> = v.iter().map(|x| -x).collect();
            let a = BinaryVector::encode(&v, false);
            let b = BinaryVector::encode(&negated, false);
            // Each sign flips → every bit differs
            prop_assert_eq!(a.hamming(&b) as usize, d);
        }

        /// Normalize flag: scaling doesn't change binary encoding.
        #[test]
        fn normalize_preserves_encoding(
            v in arb_nonzero_vec(16),
            scale in 0.1f32..10.0f32,
        ) {
            // With max |v[i]| ≤ 1e18 and scale ≤ 10, no overflow in f32.
            let scaled: Vec<f32> = v.iter().map(|x| x * scale).collect();
            let enc1 = BinaryVector::encode(&v, true);
            let enc2 = BinaryVector::encode(&scaled, true);
            prop_assert_eq!(enc1.packed, enc2.packed);
        }
    }
}
