//! ANN index algorithms — HNSW, IVF-Flat, IVF-PQ, quantized HNSW variants,
//! and Okapi BM25 full-text search.

pub mod bm25;
pub mod hnsw;
pub mod ivf;
pub mod ivf_pq;
pub mod quant_hnsw;

pub use bm25::BM25Index;

/// HNSW adjacency storage shared by `HnswIndex` and `QuantHnswIndex`.
///
/// Layer 0 — the hot, dense layer where the wide beam search runs — is stored as
/// a single contiguous `Vec<u32>` with a fixed stride of `m0 + 1` slots per node
/// (the `+1` holds the transient over-full entry before pruning back to `m0`).
/// This removes the per-node-per-layer `Vec` indirection of the old
/// `Vec<Vec<Vec<usize>>>` and packs 16 neighbour IDs per cache line (u32) instead
/// of 8 (usize). Sparse upper layers stay nested. IDs are `u32` (indexes are
/// well under 4 billion), halving link memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct NeighborGraph {
    m0: u32,
    /// Slots per node at layer 0 = `m0 + 1`.
    l0_stride: u32,
    /// Flat layer-0 links: node `i`'s links live in `l0[i*stride .. i*stride + l0_len[i]]`.
    l0: Vec<u32>,
    /// Number of valid layer-0 links per node.
    l0_len: Vec<u32>,
    /// Upper layers (≥ 1), sparse: `upper[node][lc - 1]` is node's layer-`lc` links.
    upper: Vec<Vec<Vec<u32>>>,
}

impl NeighborGraph {
    pub(crate) fn new(m0: usize) -> Self {
        Self {
            m0: m0 as u32,
            l0_stride: m0 as u32 + 1,
            l0: Vec::new(),
            l0_len: Vec::new(),
            upper: Vec::new(),
        }
    }

    /// Append a node with `level` layers above 0 (so layers `0..=level` exist).
    pub(crate) fn push_node(&mut self, level: usize) {
        self.l0.resize(self.l0.len() + self.l0_stride as usize, 0);
        self.l0_len.push(0);
        self.upper.push(vec![Vec::new(); level]);
    }

    /// Number of nodes.
    pub(crate) fn len(&self) -> usize {
        self.l0_len.len()
    }

    /// Read node `node`'s neighbours at `layer` (empty slice if the node has no
    /// such layer).
    #[inline]
    pub(crate) fn links(&self, node: usize, layer: usize) -> &[u32] {
        if layer == 0 {
            let base = node * self.l0_stride as usize;
            &self.l0[base..base + self.l0_len[node] as usize]
        } else {
            self.upper[node]
                .get(layer - 1)
                .map(Vec::as_slice)
                .unwrap_or(&[])
        }
    }

    /// True if `node` actually has `layer`.
    #[inline]
    pub(crate) fn has_layer(&self, node: usize, layer: usize) -> bool {
        layer == 0 || layer - 1 < self.upper[node].len()
    }

    /// Number of links at `(node, layer)`.
    #[inline]
    pub(crate) fn link_count(&self, node: usize, layer: usize) -> usize {
        if layer == 0 {
            self.l0_len[node] as usize
        } else {
            self.upper[node].get(layer - 1).map_or(0, Vec::len)
        }
    }

    /// Replace node's links at `layer` with `ids` (≤ `m0` at layer 0).
    pub(crate) fn set(&mut self, node: usize, layer: usize, ids: &[u32]) {
        if layer == 0 {
            let base = node * self.l0_stride as usize;
            self.l0[base..base + ids.len()].copy_from_slice(ids);
            self.l0_len[node] = ids.len() as u32;
        } else {
            self.upper[node][layer - 1] = ids.to_vec();
        }
    }

    /// Append `id` to node's `layer`. The layer-0 slot has room for one
    /// over-full entry (stride `m0 + 1`), so a caller pushes then prunes when
    /// `link_count` exceeds the cap.
    #[inline]
    pub(crate) fn push_link(&mut self, node: usize, layer: usize, id: u32) {
        if layer == 0 {
            let base = node * self.l0_stride as usize;
            let len = self.l0_len[node] as usize;
            self.l0[base + len] = id;
            self.l0_len[node] = len as u32 + 1;
        } else {
            self.upper[node][layer - 1].push(id);
        }
    }
}

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
