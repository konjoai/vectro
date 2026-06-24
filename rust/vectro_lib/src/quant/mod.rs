//! Quantization algorithms — INT8, NF4, Binary, PQ, BF16, SQ2, and SQ3.
//!
//! The [`Quantizer`] trait provides a unified interface used by
//! [`crate::index::quant_hnsw::QuantHnswIndex`].

pub mod bf16;
pub mod binary;
pub mod int8;
pub mod nf4;
pub mod pq;
pub mod rq;
pub mod sq2;
pub mod sq3;

#[cfg(all(target_os = "macos", feature = "vectro_lib_accelerate"))]
pub mod accelerate;

use serde::{Deserialize, Serialize};

// ─────────────────────────── Quantizer trait ─────────────────────────────────

/// Unified quantizer interface for use with
/// [`crate::index::quant_hnsw::QuantHnswIndex`].
///
/// Each implementor is a **zero-sized type** (marker struct).  The encoded
/// per-vector representation is an associated type `Encoded`.
///
/// # Asymmetric distance
/// `dist_to_query` computes the distance between a stored encoded vector and a
/// raw f32 query *without* requiring the query to be encoded first.  This is
/// the standard ADQ (Asymmetric Distance Quantization) approach: the query
/// retains full f32 precision while only stored vectors are compressed.
pub trait Quantizer: Send + Sync + 'static {
    /// Per-vector encoded representation stored in the index.
    type Encoded: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync;

    /// Per-search prepared query.
    ///
    /// Built once at the start of a beam search via [`Quantizer::prepare`] and
    /// reused for every candidate via [`Quantizer::dist_to_prepared`]. Most
    /// codecs set this to the owned f32 query (no preprocessing). INT8
    /// specialises it to a once-quantised query so the per-candidate distance
    /// becomes a pure-integer VNNI dot product (no per-call i8→f32 widening).
    type Prepared: Send + Sync;

    /// Encode one f32 slice.
    fn encode(v: &[f32]) -> Self::Encoded;

    /// Decode back to approximate f32.
    fn decode(enc: &Self::Encoded, dim: usize) -> Vec<f32>;

    /// Asymmetric cosine distance: encoded stored vector vs plain f32 query.
    ///
    /// Both sides are expected to represent unit-normalised vectors.
    /// Returns a value in `[0, 2]` where 0 = identical direction, 2 = opposite.
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32;

    /// Build a [`Quantizer::Prepared`] query once per search. The default
    /// clones the query (the f32 fast path); INT8 quantises it for VNNI.
    fn prepare(query: &[f32]) -> Self::Prepared;

    /// Asymmetric distance against a prepared query. Must match
    /// [`Quantizer::dist_to_query`] within the codec's numerical tolerance.
    fn dist_to_prepared(enc: &Self::Encoded, prepared: &Self::Prepared) -> f32;

    /// Prefetch `enc`'s encoded bytes into cache (read hint).
    ///
    /// Default is a no-op; codecs whose `Encoded` holds a contiguous data buffer
    /// override it to prime that buffer. The beam search calls this a couple of
    /// neighbours ahead so each cold code row streams in while the current
    /// node's distance computes — the per-node code buffer is a separate heap
    /// allocation (array-of-structs), so without this every neighbour probe is a
    /// likely cache miss.
    #[inline]
    fn prefetch(enc: &Self::Encoded) {
        let _ = enc;
    }

    /// Bits used per dimension.
    fn bits_per_dim() -> u32;
}

/// Prefetch every 64-byte cache line spanning `[ptr, ptr + len_bytes)` into L1
/// (T0 read hint). A no-op on targets without a prefetch intrinsic.
///
/// `(lines - 1) * 64 < len_bytes` always holds, so the computed addresses stay
/// within the allocation; prefetch is a pure hint with no memory effect either
/// way.
#[inline]
pub(crate) fn prefetch_bytes(ptr: *const u8, len_bytes: usize) {
    if len_bytes == 0 {
        return;
    }
    let lines = len_bytes.div_ceil(64);
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        for l in 0..lines {
            // SAFETY: prefetch is a hint with no memory effect; address in-bounds.
            unsafe { _mm_prefetch::<_MM_HINT_T0>(ptr.add(l * 64) as *const i8) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        for l in 0..lines {
            // SAFETY: prefetch is a hint with no memory effect; address in-bounds.
            unsafe {
                let p = ptr.add(l * 64);
                core::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p, options(nostack, preserves_flags, readonly));
            }
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = lines;
    }
}

// ─────────────────────────── shared helper ────────────────────────────────────

/// Cosine distance between two vectors that are **both already unit-normalised**.
///
/// Uses SimSIMD's SIMD dot dispatch (NEON/SVE/AVX2/AVX-512). This is the hot
/// distance for graph *construction* in [`crate::index::quant_hnsw`], where the
/// transient build vectors and the query are unit-norm — skipping the per-call
/// norm recomputation of [`cosine_dist_f32`].
#[inline]
pub(crate) fn cosine_dist_unit(a: &[f32], b: &[f32]) -> f32 {
    use simsimd::SpatialSimilarity;
    let dot: f64 = <f32 as SpatialSimilarity>::dot(a, b).unwrap_or(-1.0);
    (1.0 - dot as f32).max(0.0)
}

// ─────────────────── Quantizer marker types + impls ───────────────────────────

/// BF16 quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bf16Quantizer;

impl Quantizer for Bf16Quantizer {
    type Encoded = bf16::Bf16Vector;
    type Prepared = Vec<f32>;
    fn encode(v: &[f32]) -> Self::Encoded { bf16::Bf16Vector::encode(v) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        // Direct from codes — no per-call decode allocation.
        enc.cosine_dist_to_query(query)
    }
    fn prepare(query: &[f32]) -> Vec<f32> { query.to_vec() }
    fn dist_to_prepared(enc: &Self::Encoded, prepared: &Vec<f32>) -> f32 {
        enc.cosine_dist_to_query(prepared)
    }
    fn prefetch(enc: &Self::Encoded) {
        prefetch_bytes(enc.packed.as_ptr() as *const u8, enc.packed.len() * 2);
    }
    fn bits_per_dim() -> u32 { 16 }
}

/// INT8 symmetric abs-max quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Int8Quantizer;

impl Quantizer for Int8Quantizer {
    type Encoded = int8::Int8Vector;
    type Prepared = int8::Int8Query;
    fn encode(v: &[f32]) -> Self::Encoded { int8::Int8Vector::encode_fast(v) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        // `dot_query` returns weighted dot product ≈ cosine similarity for
        // unit-normalised stored vectors.
        (1.0 - enc.dot_query(query)).max(0.0)
    }
    fn prepare(query: &[f32]) -> int8::Int8Query { int8::Int8Query::prepare(query) }
    fn dist_to_prepared(enc: &Self::Encoded, prepared: &int8::Int8Query) -> f32 {
        // VNNI integer dot when available; otherwise the exact f32 path.
        (1.0 - enc.dot_query_prepared(prepared)).max(0.0)
    }
    fn prefetch(enc: &Self::Encoded) {
        prefetch_bytes(enc.codes.as_ptr() as *const u8, enc.codes.len());
    }
    fn bits_per_dim() -> u32 { 8 }
}

/// NF4 4-bit normal-float quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Nf4Quantizer;

impl Quantizer for Nf4Quantizer {
    type Encoded = nf4::Nf4Vector;
    type Prepared = Vec<f32>;
    fn encode(v: &[f32]) -> Self::Encoded { nf4::Nf4Vector::encode_fast(v) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        enc.cosine_dist_to_query(query)
    }
    fn prepare(query: &[f32]) -> Vec<f32> { query.to_vec() }
    fn dist_to_prepared(enc: &Self::Encoded, prepared: &Vec<f32>) -> f32 {
        enc.cosine_dist_to_query(prepared)
    }
    fn prefetch(enc: &Self::Encoded) {
        prefetch_bytes(enc.packed.as_ptr(), enc.packed.len());
    }
    fn bits_per_dim() -> u32 { 4 }
}

/// Binary 1-bit sign quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryQuantizer;

impl Quantizer for BinaryQuantizer {
    type Encoded = binary::BinaryVector;
    type Prepared = Vec<f32>;
    fn encode(v: &[f32]) -> Self::Encoded { binary::BinaryVector::encode_fast(v, true) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        // Sign bits → asymmetric cosine, directly from the packed bits.
        enc.cosine_dist_to_query(query)
    }
    fn prepare(query: &[f32]) -> Vec<f32> { query.to_vec() }
    fn dist_to_prepared(enc: &Self::Encoded, prepared: &Vec<f32>) -> f32 {
        enc.cosine_dist_to_query(prepared)
    }
    fn prefetch(enc: &Self::Encoded) {
        prefetch_bytes(enc.packed.as_ptr(), enc.packed.len());
    }
    fn bits_per_dim() -> u32 { 1 }
}

/// 2-bit scalar quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sq2Quantizer;

impl Quantizer for Sq2Quantizer {
    type Encoded = sq2::Sq2Vector;
    type Prepared = Vec<f32>;
    fn encode(v: &[f32]) -> Self::Encoded { sq2::Sq2Vector::encode(v) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        enc.cosine_dist_to_query(query)
    }
    fn prepare(query: &[f32]) -> Vec<f32> { query.to_vec() }
    fn dist_to_prepared(enc: &Self::Encoded, prepared: &Vec<f32>) -> f32 {
        enc.cosine_dist_to_query(prepared)
    }
    fn prefetch(enc: &Self::Encoded) {
        prefetch_bytes(enc.packed.as_ptr(), enc.packed.len());
    }
    fn bits_per_dim() -> u32 { 2 }
}

/// 3-bit scalar quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sq3Quantizer;

impl Quantizer for Sq3Quantizer {
    type Encoded = sq3::Sq3Vector;
    type Prepared = Vec<f32>;
    fn encode(v: &[f32]) -> Self::Encoded { sq3::Sq3Vector::encode(v) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        enc.cosine_dist_to_query(query)
    }
    fn prepare(query: &[f32]) -> Vec<f32> { query.to_vec() }
    fn dist_to_prepared(enc: &Self::Encoded, prepared: &Vec<f32>) -> f32 {
        enc.cosine_dist_to_query(prepared)
    }
    fn prefetch(enc: &Self::Encoded) {
        prefetch_bytes(enc.packed.as_ptr(), enc.packed.len());
    }
    fn bits_per_dim() -> u32 { 3 }
}

#[cfg(test)]
mod dist_parity_tests {
    use super::*;

    /// Reference: cosine distance of a decoded vector vs the f32 query, the
    /// behaviour `dist_to_query` had before the alloc-free direct-from-codes
    /// rewrite. Each quantizer's `cosine_dist_to_query` must match this.
    fn ref_dist(decoded: &[f32], query: &[f32]) -> f32 {
        let norm: f32 = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-8 {
            return 1.0;
        }
        let dot: f32 = decoded.iter().zip(query).map(|(&x, &y)| x * y).sum();
        (1.0 - dot / norm).max(0.0)
    }

    fn make(d: usize, seed: u64) -> Vec<f32> {
        (0..d)
            .map(|i| (((i as u64 + seed).wrapping_mul(2654435761) % 1000) as f32 / 500.0) - 1.0)
            .collect()
    }

    macro_rules! parity {
        ($name:ident, $enc:expr) => {
            #[test]
            fn $name() {
                for d in [16usize, 31, 64, 127, 256] {
                    let v = make(d, 7);
                    let q = {
                        let raw = make(d, 99);
                        let n: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
                        raw.iter().map(|x| x / n).collect::<Vec<f32>>()
                    };
                    let enc = $enc(&v);
                    let direct = enc.cosine_dist_to_query(&q);
                    let reference = ref_dist(&enc.decode(), &q);
                    assert!(
                        (direct - reference).abs() < 1e-4,
                        "d={d}: direct={direct} reference={reference}"
                    );
                }
            }
        };
    }

    parity!(parity_binary, |v| binary::BinaryVector::encode(v, true));
    parity!(parity_nf4, |v| nf4::Nf4Vector::encode_fast(v));
    parity!(parity_sq2, |v| sq2::Sq2Vector::encode(v));
    parity!(parity_sq3, |v| sq3::Sq3Vector::encode(v));
    parity!(parity_bf16, |v| bf16::Bf16Vector::encode(v));
}
