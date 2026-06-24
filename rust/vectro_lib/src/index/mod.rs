//! ANN index algorithms — HNSW, IVF-Flat, IVF-PQ, quantized HNSW variants,
//! and Okapi BM25 full-text search.

pub mod bm25;
pub(crate) mod graph;
pub mod hnsw;
pub mod ivf;
pub mod ivf_pq;
pub mod ivf_pq4;
pub(crate) mod neighbor_store;
pub mod pq4;
pub mod quant_hnsw;
pub(crate) mod scratch;
pub(crate) mod simd;

pub use bm25::BM25Index;

// ── Packed heap key: (distance, node id) in one u64 for branch-free ordering ──
//
// Beam search compares candidates by distance constantly. Wrapping f32 in an
// `Ord` newtype routes every comparison through `partial_cmp().unwrap_or()` — a
// branch per heap op. Instead we map the f32 distance to an order-preserving u32
// and pack `(key_bits << 32) | id` into a `u64`, letting the heap use native
// integer ordering: no branch, 8-byte elements, sorts by distance then id.
//
// `float_order_bits` is the standard radix-sort float key: raw IEEE-754 bits are
// monotonic only for *non-negative* floats (negatives are bit-inverted by the
// sign bit). Flipping all bits for negatives and just the sign bit for
// non-negatives yields a total order over ALL finite floats — required because
// the `InnerProduct` metric produces negative distances (`-dot`). For
// non-negative inputs the high bit is simply set, preserving the previous
// ordering exactly (back-compatible with Cosine/L2).

#[inline]
fn float_order_bits(f: f32) -> u32 {
    let b = f.to_bits();
    // sign-bit set → flip all bits; sign-bit clear → flip just the sign bit.
    if b >> 31 == 1 { !b } else { b | 0x8000_0000 }
}

#[inline]
fn float_order_bits_inv(k: u32) -> f32 {
    // Exact inverse of `float_order_bits`.
    let b = if k >> 31 == 1 { k & 0x7FFF_FFFF } else { !k };
    f32::from_bits(b)
}

#[inline]
pub(crate) fn pack_key(dist: f32, id: usize) -> u64 {
    ((float_order_bits(dist) as u64) << 32) | (id as u32 as u64)
}

#[inline]
pub(crate) fn key_dist(key: u64) -> f32 {
    float_order_bits_inv((key >> 32) as u32)
}

#[inline]
pub(crate) fn key_id(key: u64) -> usize {
    (key & 0xFFFF_FFFF) as usize
}

