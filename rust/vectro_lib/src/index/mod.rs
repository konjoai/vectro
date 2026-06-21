//! ANN index algorithms — HNSW, IVF-Flat, IVF-PQ, quantized HNSW variants,
//! and Okapi BM25 full-text search.

pub mod bm25;
pub mod hnsw;
pub mod ivf;
pub mod ivf_pq;
pub(crate) mod neighbor_store;
pub mod quant_hnsw;
pub(crate) mod scratch;

pub use bm25::BM25Index;

// ── Packed heap key: (distance, node id) in one u64 for branch-free ordering ──
//
// Beam search compares candidates by distance constantly. Wrapping f32 in an
// `Ord` newtype routes every comparison through `partial_cmp().unwrap_or()` — a
// branch per heap op. Since HNSW distances are always ≥ 0, an f32's raw bits are
// monotonic (`a < b ⟺ a.to_bits() < b.to_bits()` for non-negative floats), so we
// pack `(dist_bits << 32) | id` into a `u64` and let the heap use native integer
// ordering: no branch, 8-byte elements, sorts by distance then id.

#[inline]
pub(crate) fn pack_key(dist: f32, id: usize) -> u64 {
    ((dist.to_bits() as u64) << 32) | (id as u32 as u64)
}

#[inline]
pub(crate) fn key_dist(key: u64) -> f32 {
    f32::from_bits((key >> 32) as u32)
}

#[inline]
pub(crate) fn key_id(key: u64) -> usize {
    (key & 0xFFFF_FFFF) as usize
}

