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
        let bytes_per_vec = (dim + 7) / 8;
        let mut packed = vec![0u8; bytes_per_vec];

        // Use f64 for the norm to avoid f32 overflow on extreme-valued vectors.
        let norm_factor_f64: f64 = if normalize {
            let sq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
            if sq > 0.0 { sq.sqrt() } else { 1.0 }
        } else {
            1.0_f64
        };

        for (i, &x) in v.iter().enumerate() {
            if (x as f64) / norm_factor_f64 > 0.0 {
                packed[i / 8] |= 1u8 << (i % 8);
            }
        }

        Self { packed, dim }
    }

    /// Decode packed bits to {+1.0, -1.0} f32 values.
    pub fn decode(&self) -> Vec<f32> {
        (0..self.dim).map(|i| {
            if (self.packed[i / 8] >> (i % 8)) & 1 == 1 { 1.0f32 } else { -1.0f32 }
        }).collect()
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
        let mut dot = 0.0f32;
        for (i, &q) in query.iter().enumerate().take(self.dim) {
            let bit = (self.packed[i / 8] >> (i % 8)) & 1;
            dot += if bit == 1 { q } else { -q };
        }
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
        <u8 as BinarySimilarity>::hamming(&self.packed, &other.packed)
            .unwrap_or(0.0) as u32
    }
}

/// Encode a batch of f32 vectors to binary in parallel.
pub fn encode_batch(vectors: &[Vec<f32>], normalize: bool) -> Vec<BinaryVector> {
    vectors.par_iter().map(|v| BinaryVector::encode(v, normalize)).collect()
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

    #[test]
    fn packing_basic() {
        // v = [1, -1, 1, -1, 0, 0, 0, 0] → bits 0,2 set → byte = 0b0000_0101 = 5
        let v = vec![1.0f32, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0];
        let bv = BinaryVector::encode(&v, false);
        assert_eq!(bv.packed.len(), 1);
        assert_eq!(bv.packed[0], 0b0000_0101);
    }

    #[test]
    fn decode_round_trip() {
        let v = vec![1.0f32, -1.0, 0.5, -0.5, 0.0, 1.0, -0.1, 0.9];
        let bv = BinaryVector::encode(&v, false);
        let dec = bv.decode();
        // Should get sign pattern correct
        for (orig, &d) in v.iter().zip(dec.iter()) {
            if *orig > 0.0 { assert_eq!(d, 1.0); }
            else { assert_eq!(d, -1.0); }
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
        let v: Vec<f32> = (0..13).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
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
        let db: Vec<BinaryVector> = db_vecs.iter().map(|v| BinaryVector::encode(v, false)).collect();
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
            .map(|i| (0..32).map(|j| if (i + j) % 2 == 0 { 0.5f32 } else { -0.5f32 }).collect())
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
        prop::collection::vec(-1e18f32..1e18f32, d)
            .prop_filter("degenerate zero vector", |v| {
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
