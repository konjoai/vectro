"""Native Rust backend for :class:`python.hnsw_api.HNSWIndex`.

Wraps the compiled ``vectro_py.PyHnswIndex`` so the Python HNSW API can
delegate its hot build and search paths to Rust + SimSIMD kernels — roughly
20x faster graph construction and 18x faster query throughput at matched
recall — while the Python layer retains ownership of metadata, string-ID
upserts, tombstones and persistence.

Constraints
-----------
* **Cosine only.**  The native index stores unit-normalised vectors and uses
  ``1 - dot`` as the distance.  ``space="l2"`` falls back to pure Python.
* **Deterministic.**  Native level assignment is a pure function of the node
  ID (LCG hash), so rebuilding from the same vectors in the same order yields
  an identical graph — which makes rebuild-on-load and rebuild-after-update
  safe.

Pure-Python remains the correctness baseline; this backend must match it
numerically.  Node IDs are assigned in insertion order (0, 1, 2, …) so they
stay aligned with the positional metadata the Python layer stores alongside.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def rust_available() -> bool:
    """Return True if the compiled native HNSW index is importable."""
    try:
        import vectro_py
    except ImportError:
        return False
    return hasattr(vectro_py, "PyHnswIndex")


def normalize_rows(vecs: np.ndarray) -> np.ndarray:
    """Return unit-norm float32 rows; zero rows pass through unchanged.

    Accepts a 1-D ``(d,)`` or 2-D ``(n, d)`` array and preserves the input
    rank in the result.
    """
    arr = np.asarray(vecs, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr[np.newaxis, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    out = (arr / norms).astype(np.float32)
    return out[0] if single else out


class RustHnswBackend:
    """Thin, cosine-only wrapper over ``vectro_py.PyHnswIndex``.

    The wrapper owns only the native graph.  Vectors, metadata and tombstones
    remain in :class:`HNSWIndex`; this class is handed already-normalised data.
    """

    def __init__(self, m: int, ef_construction: int) -> None:
        import vectro_py

        self._vectro_py = vectro_py
        self.m = m
        self.ef_construction = ef_construction
        self._idx = vectro_py.PyHnswIndex(m, ef_construction)

    def add_many(self, normalized: np.ndarray) -> None:
        """Insert a batch of already-normalised rows in one native call."""
        if normalized.shape[0] == 0:
            return
        self._idx.add_np(np.ascontiguousarray(normalized, dtype=np.float32))

    def add_one(self, normalized_row: np.ndarray) -> None:
        """Insert a single already-normalised row."""
        row = np.ascontiguousarray(normalized_row[np.newaxis, :], dtype=np.float32)
        self._idx.add_np(row)

    def delete(self, node_id: int) -> None:
        """Soft-delete a node by ID (excluded from future results)."""
        self._idx.delete(node_id)

    def search(
        self,
        q_norm: np.ndarray,
        k: int,
        ef: int,
        allowed: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Return ``[(node_id, distance), …]`` ascending by distance.

        When *allowed* is given, only those node IDs are eligible for the
        result set (the graph is still traversed through other nodes).
        """
        q = np.ascontiguousarray(q_norm, dtype=np.float32)
        if allowed is None:
            res = self._idx.search_np(q, k, ef)
        else:
            res = self._idx.search_filtered_np(q, k, ef, allowed)
        return [(int(nid), float(dist)) for nid, dist in res]

    def search_arrays(
        self,
        q_norm: np.ndarray,
        k: int,
        ef: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-query search returning ``(ids int64, distances float32)`` numpy
        arrays directly from the native core, with the GIL released.

        Skips the per-query Python list-of-tuples allocation that bottlenecks the
        single-query hot path.
        """
        q = np.ascontiguousarray(q_norm, dtype=np.float32)
        return self._idx.search_arrays_np(q, k, ef)

    def search_batch(
        self,
        q_norm: np.ndarray,
        k: int,
        ef: int,
    ) -> List[List[Tuple[int, float]]]:
        """Batch search: one ``[(node_id, distance), …]`` list per query row.

        Delegates to the native ``search_batch_np``, which parallelises across
        queries with rayon and releases the GIL — far higher throughput than a
        per-query Python loop. Filtering is not supported on this path (the
        native batch entry takes no allow-list); callers needing a metadata
        filter fall back to per-query :meth:`search`.
        """
        q = np.ascontiguousarray(q_norm, dtype=np.float32)
        batch = self._idx.search_batch_np(q, k, ef)
        return [[(int(nid), float(dist)) for nid, dist in row] for row in batch]

    def rebuild(self, normalized_vectors: List[np.ndarray], deleted: "set[int]") -> None:
        """Rebuild the graph from scratch, then re-apply tombstones.

        Used after in-place vector updates and on load.  Deterministic: the
        native level assignment depends only on node ID, so the rebuilt graph
        is identical given the same insertion order.
        """
        self._idx = self._vectro_py.PyHnswIndex(self.m, self.ef_construction)
        if normalized_vectors:
            mat = np.ascontiguousarray(np.stack(normalized_vectors), dtype=np.float32)
            self._idx.add_np(mat)
        for nid in sorted(deleted):
            self._idx.delete(nid)

    def __len__(self) -> int:
        return len(self._idx)
