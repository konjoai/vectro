//! Compact HNSW adjacency storage.
//!
//! HNSW spends almost all of its memory on the graph, and ~99 % of that is
//! **layer 0** (the geometric level distribution puts only ~1/m of nodes above
//! it). The naive `Vec<Vec<SmallVec<[u32; m0]>>>` layout pays, per node:
//! an inner `Vec` header (24 B), a `SmallVec` that reserves `m0` ids *inline*
//! (128 B at m0=32) whether or not they're used, and a separate heap allocation
//! per node — so traversal is a chain of pointer chases with poor locality.
//!
//! [`Graph`] instead stores layer 0 as a single **flat, fixed-slot** array
//! (`l0[node*m0 .. node*m0 + m0]`, with a `u8` fill count per node) — the same
//! representation FAISS uses, which both halves the graph footprint and makes
//! the hot layer-0 neighbour scan a contiguous slice read. The rare upper
//! layers stay as per-node `Vec`s (tiny in aggregate).
//!
//! The on-disk format is unchanged: [`graph_serde`] (de)serialises through the
//! legacy `Vec<Vec<Vec<u64>>>` wire layout, so old indexes still load.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::neighbor_store::{NeighborList, NodeId};

/// Compact per-node HNSW adjacency: flat fixed-slot layer 0 + sparse upper.
#[derive(Debug, Clone)]
pub(crate) struct Graph {
    /// Logical max links at layer 0 (`2*m`).
    m0: usize,
    /// Physical slots per node at layer 0 = `m0 + 1`. The extra slot absorbs the
    /// transient `m0 + 1` state during `connect` (push a reverse link, then prune
    /// back to `m0`).
    stride: usize,
    /// Flat layer-0 adjacency: node `i`'s links live in `l0[i*stride ..][..len]`.
    l0: Vec<NodeId>,
    /// Number of valid layer-0 links per node (`≤ m0`).
    l0_len: Vec<u8>,
    /// Upper layers, `upper[node][layer-1]`. Empty for the common level-0 node.
    upper: Vec<Vec<NeighborList>>,
}

impl Graph {
    /// Empty graph; layer-0 nodes get `m0 + 1` slots each. `m0 ≤ 254` (a u8 fill
    /// count, +1 transient) — true for every sane HNSW `m` (default `m=16`).
    pub(crate) fn new(m0: usize) -> Self {
        assert!(
            m0 <= 254,
            "m0 must be ≤ 254 (m ≤ 127) for the flat layer-0 store"
        );
        Self {
            m0,
            stride: m0 + 1,
            l0: Vec::new(),
            l0_len: Vec::new(),
            upper: Vec::new(),
        }
    }

    /// Append a new node at `level` (its layers are `0..=level`), all empty.
    pub(crate) fn add_node(&mut self, level: usize) {
        self.l0.resize(self.l0.len() + self.stride, 0);
        self.l0_len.push(0);
        self.upper.push(vec![NeighborList::new(); level]);
    }

    /// Number of layers node `node` participates in (`1 + upper layers`).
    #[inline]
    pub(crate) fn num_layers(&self, node: usize) -> usize {
        1 + self.upper[node].len()
    }

    /// Borrow node `node`'s neighbour ids at `layer` (a contiguous slice).
    #[inline]
    pub(crate) fn neighbors(&self, node: usize, layer: usize) -> &[NodeId] {
        if layer == 0 {
            let base = node * self.stride;
            &self.l0[base..base + self.l0_len[node] as usize]
        } else {
            &self.upper[node][layer - 1]
        }
    }

    /// Replace node `node`'s `layer` adjacency with `ids` (`≤ m0` at layer 0).
    #[inline]
    pub(crate) fn set(&mut self, node: usize, layer: usize, ids: &[NodeId]) {
        if layer == 0 {
            debug_assert!(ids.len() <= self.m0);
            let base = node * self.stride;
            self.l0[base..base + ids.len()].copy_from_slice(ids);
            self.l0_len[node] = ids.len() as u8;
        } else {
            let list = &mut self.upper[node][layer - 1];
            list.clear();
            list.extend_from_slice(ids);
        }
    }

    /// Current neighbour count of node `node` at `layer`.
    #[inline]
    pub(crate) fn len_at(&self, node: usize, layer: usize) -> usize {
        if layer == 0 {
            self.l0_len[node] as usize
        } else {
            self.upper[node][layer - 1].len()
        }
    }

    /// Append `id` to node `node`'s `layer` adjacency.
    ///
    /// At layer 0 this fills the next free slot. The extra `+1` stride means a
    /// node at the `m0` limit can still take one reverse link transiently; the
    /// HNSW connect step prunes it back to `m0` immediately after.
    #[inline]
    pub(crate) fn push(&mut self, node: usize, layer: usize, id: NodeId) {
        if layer == 0 {
            let len = self.l0_len[node] as usize;
            debug_assert!(
                len < self.stride,
                "layer-0 slot overflow (len {len}, stride {})",
                self.stride
            );
            self.l0[node * self.stride + len] = id;
            self.l0_len[node] = (len + 1) as u8;
        } else {
            self.upper[node][layer - 1].push(id);
        }
    }

    /// Build a `Graph` from the per-node layered lists produced by the build
    /// phase (`layered[node][layer]`), with layer-0 budget `m0`.
    pub(crate) fn from_layered(layered: Vec<Vec<NeighborList>>, m0: usize) -> Self {
        let n = layered.len();
        let mut g = Graph::new(m0);
        g.l0 = vec![0; n * g.stride];
        g.l0_len = vec![0; n];
        g.upper = Vec::with_capacity(n);
        for (node, layers) in layered.into_iter().enumerate() {
            let mut it = layers.into_iter();
            if let Some(l0) = it.next() {
                let take = l0.len().min(m0);
                let base = node * g.stride;
                g.l0[base..base + take].copy_from_slice(&l0[..take]);
                g.l0_len[node] = take as u8;
            }
            g.upper.push(it.collect());
        }
        g
    }
}

/// (De)serialise [`Graph`] through a **compact flat** wire format: the layer-0
/// slot array (`u32`) and fill counts go out verbatim, so deserialise rebuilds
/// the flat store *directly* — no `Vec<Vec<SmallVec>>` intermediate (which both
/// halved on-disk size vs the old `u64`-per-link layout and removed the load-time
/// allocation spike that inflated resident memory). Upper layers are tiny and
/// ride along as plain nested vecs.
pub(crate) mod graph_serde {
    use super::{Deserialize, Deserializer, Graph, NeighborList, NodeId, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Wire {
        m0: usize,
        l0: Vec<NodeId>,
        l0_len: Vec<u8>,
        upper: Vec<Vec<Vec<NodeId>>>,
    }

    pub fn serialize<S>(graph: &Graph, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let upper: Vec<Vec<Vec<NodeId>>> = graph
            .upper
            .iter()
            .map(|layers| layers.iter().map(|l| l.to_vec()).collect())
            .collect();
        Wire {
            m0: graph.m0,
            l0: graph.l0.clone(),
            l0_len: graph.l0_len.clone(),
            upper,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Graph, D::Error>
    where
        D: Deserializer<'de>,
    {
        let w = Wire::deserialize(d)?;
        let upper: Vec<Vec<NeighborList>> = w
            .upper
            .into_iter()
            .map(|layers| layers.into_iter().map(NeighborList::from_vec).collect())
            .collect();
        Ok(Graph {
            m0: w.m0,
            stride: w.m0 + 1,
            l0: w.l0,
            l0_len: w.l0_len,
            upper,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_layer0_roundtrips_and_indexes() {
        let mut g = Graph::new(32);
        g.add_node(0); // node 0, layer 0 only
        g.add_node(2); // node 1, layers 0,1,2
        g.set(0, 0, &[1, 2, 3]);
        g.push(0, 0, 4);
        g.set(1, 0, &[0]);
        g.set(1, 1, &[0, 2]);
        g.set(1, 2, &[3]);

        assert_eq!(g.neighbors(0, 0), &[1, 2, 3, 4]);
        assert_eq!(g.len_at(0, 0), 4);
        assert_eq!(g.num_layers(0), 1);
        assert_eq!(g.num_layers(1), 3);
        assert_eq!(g.neighbors(1, 1), &[0, 2]);
        assert_eq!(g.neighbors(1, 2), &[3]);
    }

    #[test]
    fn from_layered_preserves_adjacency() {
        let layered: Vec<Vec<NeighborList>> = vec![
            vec![
                NeighborList::from_slice(&[1, 2, 300]),
                NeighborList::from_slice(&[2]),
            ],
            vec![NeighborList::from_slice(&[0])],
        ];
        let g = Graph::from_layered(layered, 32);
        assert_eq!(g.neighbors(0, 0), &[1, 2, 300]);
        assert_eq!(g.neighbors(0, 1), &[2]);
        assert_eq!(g.neighbors(1, 0), &[0]);
        assert_eq!(g.num_layers(1), 1);
    }

    #[test]
    fn serde_roundtrip_via_legacy_wire() {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Wrap(#[serde(with = "super::graph_serde")] Graph);

        let mut g = Graph::new(32);
        g.add_node(1);
        g.add_node(0);
        g.set(0, 0, &[1]);
        g.set(0, 1, &[1]);
        g.set(1, 0, &[0]);

        let bytes = bincode::serialize(&Wrap(g.clone())).unwrap();
        let back: Wrap = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.0.neighbors(0, 0), g.neighbors(0, 0));
        assert_eq!(back.0.neighbors(0, 1), g.neighbors(0, 1));
        assert_eq!(back.0.neighbors(1, 0), g.neighbors(1, 0));
    }
}
