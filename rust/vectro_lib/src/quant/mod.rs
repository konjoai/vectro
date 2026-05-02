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

    /// Encode one f32 slice.
    fn encode(v: &[f32]) -> Self::Encoded;

    /// Decode back to approximate f32.
    fn decode(enc: &Self::Encoded, dim: usize) -> Vec<f32>;

    /// Asymmetric cosine distance: encoded stored vector vs plain f32 query.
    ///
    /// Both sides are expected to represent unit-normalised vectors.
    /// Returns a value in `[0, 2]` where 0 = identical direction, 2 = opposite.
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32;

    /// Bits used per dimension.
    fn bits_per_dim() -> u32;
}

// ─────────────────────────── shared helper ────────────────────────────────────

/// Cosine distance between `a` and `b`.
///
/// `b` is assumed to be unit-normalised (e.g. the f32 query in
/// [`QuantHnswIndex`]).  `a` (the decoded stored vector) is normalised
/// here to handle quantizers whose decoded vectors are not exactly unit-norm
/// (e.g. binary, SQ2, SQ3, NF4).
#[inline]
pub(crate) fn cosine_dist_f32(a: &[f32], b: &[f32]) -> f32 {
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 {
        return 1.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    (1.0 - dot / norm_a).max(0.0)
}

// ─────────────────── Quantizer marker types + impls ───────────────────────────

/// BF16 quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bf16Quantizer;

impl Quantizer for Bf16Quantizer {
    type Encoded = bf16::Bf16Vector;
    fn encode(v: &[f32]) -> Self::Encoded { bf16::Bf16Vector::encode(v) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        cosine_dist_f32(&enc.decode(), query)
    }
    fn bits_per_dim() -> u32 { 16 }
}

/// INT8 symmetric abs-max quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Int8Quantizer;

impl Quantizer for Int8Quantizer {
    type Encoded = int8::Int8Vector;
    fn encode(v: &[f32]) -> Self::Encoded { int8::Int8Vector::encode_fast(v) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        // `dot_query` returns weighted dot product ≈ cosine similarity for
        // unit-normalised stored vectors.
        (1.0 - enc.dot_query(query)).max(0.0)
    }
    fn bits_per_dim() -> u32 { 8 }
}

/// NF4 4-bit normal-float quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Nf4Quantizer;

impl Quantizer for Nf4Quantizer {
    type Encoded = nf4::Nf4Vector;
    fn encode(v: &[f32]) -> Self::Encoded { nf4::Nf4Vector::encode_fast(v) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        cosine_dist_f32(&enc.decode(), query)
    }
    fn bits_per_dim() -> u32 { 4 }
}

/// Binary 1-bit sign quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryQuantizer;

impl Quantizer for BinaryQuantizer {
    type Encoded = binary::BinaryVector;
    fn encode(v: &[f32]) -> Self::Encoded { binary::BinaryVector::encode(v, true) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        // Decode to {+1, -1} then compute cosine against f32 query.
        cosine_dist_f32(&enc.decode(), query)
    }
    fn bits_per_dim() -> u32 { 1 }
}

/// 2-bit scalar quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sq2Quantizer;

impl Quantizer for Sq2Quantizer {
    type Encoded = sq2::Sq2Vector;
    fn encode(v: &[f32]) -> Self::Encoded { sq2::Sq2Vector::encode(v) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        cosine_dist_f32(&enc.decode(), query)
    }
    fn bits_per_dim() -> u32 { 2 }
}

/// 3-bit scalar quantizer marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sq3Quantizer;

impl Quantizer for Sq3Quantizer {
    type Encoded = sq3::Sq3Vector;
    fn encode(v: &[f32]) -> Self::Encoded { sq3::Sq3Vector::encode(v) }
    fn decode(enc: &Self::Encoded, _dim: usize) -> Vec<f32> { enc.decode() }
    fn dist_to_query(enc: &Self::Encoded, query: &[f32]) -> f32 {
        cosine_dist_f32(&enc.decode(), query)
    }
    fn bits_per_dim() -> u32 { 3 }
}
