//! Shared neighbour-storage primitives for the HNSW indexes.
//!
//! Both [`crate::index::hnsw::HnswIndex`] and
//! [`crate::index::quant_hnsw::QuantHnswIndex`] store their graph adjacency as
//! `neighbors[node][layer] = [neighbor_id, ...]`.  Two choices keep that hot,
//! memory-heavy structure cache-friendly:
//!
//! * **`u32` node ids** ([`NodeId`]) instead of `usize` halve adjacency memory
//!   and double cache-line density.  A single index supports up to `u32::MAX`
//!   (~4.3 billion) vectors — far beyond any realistic single-node working set.
//! * **`SmallVec`-backed lists** ([`NeighborList`]) keep each per-layer list
//!   inline up to the layer-0 link budget (`m0 = 2*m` for the default `m = 16`),
//!   so the dominant layer-0 lists never touch the heap.  Larger `m` spills to
//!   the heap transparently and stays correct.
//!
//! On-disk format is preserved: [`neighbors_serde`] (de)serializes the lists
//! through the legacy `u64` wire layout, so indexes saved before this migration
//! still load and freshly-saved indexes stay byte-compatible with the old
//! reader.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::SmallVec;

/// Graph node identifier.  `u32` (vs `usize`) halves adjacency memory.
pub(crate) type NodeId = u32;

/// Inline capacity for a per-(node, layer) neighbour list.
///
/// Equal to `m0 = 2 * m` for the default `m = 16`, so the overwhelmingly common
/// layer-0 lists stay inline; upper-layer lists (`≤ m`) always fit.
pub(crate) const NEIGHBOR_INLINE: usize = 32;

/// A single (node, layer) adjacency list.
pub(crate) type NeighborList = SmallVec<[NodeId; NEIGHBOR_INLINE]>;

/// (De)serialize the `neighbors` field through the legacy `Vec<Vec<Vec<u64>>>`
/// wire format.
///
/// `usize` was bincode-encoded as a fixed 8-byte integer, so emitting `u64`
/// here is byte-for-byte identical to the pre-migration layout: indexes saved
/// before the `u32`/`SmallVec` change still load, and new saves remain readable
/// by the old code path.
pub(crate) mod neighbors_serde {
    use super::{Deserialize, Deserializer, NeighborList, NodeId, Serialize, Serializer};

    pub fn serialize<S>(neighbors: &[Vec<NeighborList>], s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire: Vec<Vec<Vec<u64>>> = neighbors
            .iter()
            .map(|layers| {
                layers
                    .iter()
                    .map(|list| list.iter().map(|&id| id as u64).collect())
                    .collect()
            })
            .collect();
        wire.serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Vec<Vec<NeighborList>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = Vec::<Vec<Vec<u64>>>::deserialize(d)?;
        Ok(wire
            .into_iter()
            .map(|layers| {
                layers
                    .into_iter()
                    .map(|list| list.into_iter().map(|id| id as NodeId).collect())
                    .collect()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The custom serde path must round-trip and stay byte-identical to the
    /// legacy `Vec<Vec<Vec<u64>>>` (== old `Vec<Vec<Vec<usize>>>`) bincode
    /// layout, so previously-saved indexes still load.
    #[test]
    fn neighbors_wire_format_matches_legacy_u64() {
        // A small graph: 2 nodes, ragged layers, ids that exercise multi-byte.
        let graph: Vec<Vec<NeighborList>> = vec![
            vec![
                NeighborList::from_slice(&[1, 2, 300]),
                NeighborList::from_slice(&[2]),
            ],
            vec![NeighborList::from_slice(&[0])],
        ];

        // Serialize through the custom helper.
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct Wrap(#[serde(with = "super::neighbors_serde")] Vec<Vec<NeighborList>>);

        let via_helper = bincode::serialize(&Wrap(graph.clone())).unwrap();

        // Serialize the equivalent legacy structure directly.
        let legacy: Vec<Vec<Vec<u64>>> = graph
            .iter()
            .map(|ls| {
                ls.iter()
                    .map(|l| l.iter().map(|&x| x as u64).collect())
                    .collect()
            })
            .collect();
        let via_legacy = bincode::serialize(&legacy).unwrap();

        assert_eq!(
            via_helper, via_legacy,
            "wire format diverged from legacy u64 layout"
        );

        // And it round-trips back to the same in-memory graph.
        let back: Wrap = bincode::deserialize(&via_helper).unwrap();
        assert_eq!(back.0, graph);
    }
}
