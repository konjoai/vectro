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

/// A deterministic permutation of `0..n` (seeded Fisher–Yates).
///
/// Used by the parallel HNSW build to process nodes in a pseudo-random order:
/// chunk-mates search the same frozen graph and therefore can't link to each
/// other, so a shuffled order keeps correlated (e.g. sorted) input rows out of
/// the same chunk. The fixed seed keeps builds reproducible.
pub(crate) fn shuffled_order(n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((state >> 33) as usize) % (i + 1);
        order.swap(i, j);
    }
    order
}
