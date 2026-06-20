"""IVF (Inverted File) approximate nearest-neighbour indices (v7.0.0).

Thin Python wrappers around the Rust ``PyIvfIndex`` and ``PyIvfPqIndex``
PyO3 bindings, providing user-friendly names and numpy-first entry points.

Public API
----------
IVFIndex(n_lists, n_probe)
    .train(vectors, max_iter, seed)   -> None
    .train_np(array, max_iter, seed)  -> None
    .add(vector)                      -> int
    .add_np(array)                    -> None
    .search(query, k)                 -> list[(int, float)]
    .search_np(query, k)              -> list[(int, float)]
    .search_with_probe(query, k, n_probe) -> list[(int, float)]
    .search_filtered_np(query, k, allowed_ids) -> list[(int, float)]
    .search_for_recall(query, k, target_recall) -> (list[(int, float)], int)
    .delete(id)                       -> None
    .vacuum()                         -> int
    .save(path)                       -> None
    IVFIndex.load(path)               -> IVFIndex

IVFPQIndex(n_lists, n_probe)
    .train(vectors, n_subspaces, n_centroids, max_iter, seed)  -> None
    .train_np(array, n_subspaces, n_centroids, max_iter, seed) -> None
    .add(vector)                      -> int
    .add_np(array)                    -> None
    .search(query, k)                 -> list[(int, float)]
    .search_np(query, k)              -> list[(int, float)]
    .search_with_probe(query, k, n_probe) -> list[(int, float)]
    .search_for_recall(query, k, target_recall) -> (list[(int, float)], int)
    .delete(id)                       -> None
    .vacuum()                         -> int
    .save(path)                       -> None
    IVFPQIndex.load(path)             -> IVFPQIndex
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from vectro_py import PyIvfIndex as _PyIvfIndex  # type: ignore
    from vectro_py import PyIvfPqIndex as _PyIvfPqIndex  # type: ignore

    _BINDINGS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BINDINGS_AVAILABLE = False
    _PyIvfIndex = None  # type: ignore[assignment,misc]
    _PyIvfPqIndex = None  # type: ignore[assignment,misc]


def _require_bindings() -> None:
    if not _BINDINGS_AVAILABLE:
        raise ImportError(
            "vectro_py is required.  Build it with `maturin develop` or `pip install vectro` first."
        )


class IVFIndex:
    """IVF-Flat approximate nearest-neighbour index.

    Divides the vector space into *n_lists* Voronoi cells using a k-means
    coarse quantiser.  At search time, only *n_probe* cells are scanned,
    giving a configurable accuracy/speed trade-off.

    Parameters
    ----------
    n_lists : int
        Number of inverted lists (Voronoi cells).  A good starting point
        is ``int(sqrt(n_vectors))``.
    n_probe : int
        Number of cells to probe at search time.  Higher → better recall
        at the cost of latency.  Typically 4–32.
    """

    def __init__(self, n_lists: int, n_probe: int) -> None:
        _require_bindings()
        self._inner = _PyIvfIndex(n_lists, n_probe)
        self._count: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        vectors: List[List[float]],
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        """Train the coarse k-means quantiser from *vectors*."""
        self._inner.train(vectors, max_iter, seed)

    def train_np(
        self,
        array: np.ndarray,
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        """Zero-copy train from a 2-D float32 numpy array of shape ``(N, D)``."""
        arr = np.ascontiguousarray(array, dtype=np.float32)
        self._inner.train_np(arr, max_iter, seed)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, vector: List[float]) -> int:
        """Add a single vector; returns its global integer id."""
        result = self._inner.add(vector)
        self._count += 1
        return result

    def add_np(self, array: np.ndarray) -> None:
        """Zero-copy batch insert from a 2-D float32 numpy array ``(N, D)``."""
        arr = np.ascontiguousarray(array, dtype=np.float32)
        self._inner.add_np(arr)
        self._count += arr.shape[0]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: List[float], k: int) -> List[Tuple[int, float]]:
        """Return the *k* nearest neighbours as ``[(id, distance), ...]``."""
        return self._inner.search(query, k)

    def search_np(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Zero-copy search from a 1-D float32 numpy query vector."""
        q = np.ascontiguousarray(query, dtype=np.float32)
        return self._inner.search_np(q, k)

    def search_with_probe(
        self,
        query: List[float],
        k: int,
        n_probe: int,
    ) -> List[Tuple[int, float]]:
        """Search with an explicit *n_probe* override for this query."""
        return self._inner.search_with_probe(query, k, n_probe)

    def search_filtered_np(
        self,
        query: np.ndarray,
        k: int,
        allowed_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """Search restricted to *allowed_ids*; returns at most *k* results."""
        q = np.ascontiguousarray(query, dtype=np.float32)
        return self._inner.search_filtered_np(q, k, allowed_ids)

    def search_for_recall(
        self,
        query: List[float],
        k: int,
        target_recall: float,
    ) -> Tuple[List[Tuple[int, float]], int]:
        """Find the minimum n_probe achieving *target_recall*.

        Returns
        -------
        (results, n_probe_used) : tuple
        """
        return self._inner.search_for_recall(query, k, target_recall)

    # ------------------------------------------------------------------
    # Deletion / compaction
    # ------------------------------------------------------------------

    def delete(self, id: int) -> None:
        """Soft-delete a vector by global *id*."""
        self._inner.delete(id)

    def vacuum(self) -> int:
        """Permanently remove soft-deleted vectors.  Returns count removed."""
        removed = self._inner.vacuum()
        self._count -= removed
        return removed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the index to *path* (bincode format)."""
        self._inner.save(path)

    @classmethod
    def load(cls, path: str) -> "IVFIndex":
        """Load an index previously saved with :meth:`save`."""
        _require_bindings()
        obj = cls.__new__(cls)
        obj._inner = _PyIvfIndex.load(path)
        obj._count = len(obj._inner) if hasattr(obj._inner, "__len__") else 0
        return obj

    def __repr__(self) -> str:
        return repr(self._inner)

    def __len__(self) -> int:
        return self._count


class IVFPQIndex:
    """IVF-PQ approximate nearest-neighbour index with ADC scoring.

    Combines an IVF coarse quantiser with a Product Quantizer for compact
    storage and fast Asymmetric Distance Computation (ADC).  Suitable for
    very large-scale retrieval where IVF-Flat memory is prohibitive.

    Parameters
    ----------
    n_lists : int
        Number of IVF coarse quantiser cells.
    n_probe : int
        Number of cells to probe at search time.
    """

    def __init__(self, n_lists: int, n_probe: int) -> None:
        _require_bindings()
        self._inner = _PyIvfPqIndex(n_lists, n_probe)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        vectors: List[List[float]],
        n_subspaces: int,
        n_centroids: int,
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        """Train the coarse quantiser and PQ codebook.

        Parameters
        ----------
        n_subspaces : int
            Number of PQ sub-spaces (d must be divisible by this).
        n_centroids : int
            Number of centroids per sub-space (typically 256).
        """
        self._inner.train(vectors, n_subspaces, n_centroids, max_iter, seed)

    def train_np(
        self,
        array: np.ndarray,
        n_subspaces: int,
        n_centroids: int,
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        """Zero-copy train from a 2-D float32 numpy array ``(N, D)``."""
        arr = np.ascontiguousarray(array, dtype=np.float32)
        self._inner.train_np(arr, n_subspaces, n_centroids, max_iter, seed)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, vector: List[float]) -> int:
        """Add a single vector; returns its global integer id."""
        return self._inner.add(vector)

    def add_np(self, array: np.ndarray) -> None:
        """Zero-copy batch insert from a 2-D float32 numpy array ``(N, D)``."""
        arr = np.ascontiguousarray(array, dtype=np.float32)
        self._inner.add_np(arr)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: List[float], k: int) -> List[Tuple[int, float]]:
        """Return *k* nearest neighbours as ``[(id, distance), ...]``."""
        return self._inner.search(query, k)

    def search_np(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Zero-copy search from a 1-D float32 numpy query vector."""
        q = np.ascontiguousarray(query, dtype=np.float32)
        return self._inner.search_np(q, k)

    def search_with_probe(
        self,
        query: List[float],
        k: int,
        n_probe: int,
    ) -> List[Tuple[int, float]]:
        """Search with an explicit *n_probe* override for this query."""
        return self._inner.search_with_probe(query, k, n_probe)

    def search_for_recall(
        self,
        query: List[float],
        k: int,
        target_recall: float,
    ) -> Tuple[List[Tuple[int, float]], int]:
        """Find the minimum n_probe achieving *target_recall*.

        Returns ``(results, n_probe_used)``.
        """
        return self._inner.search_for_recall(query, k, target_recall)

    # ------------------------------------------------------------------
    # Deletion / compaction
    # ------------------------------------------------------------------

    def delete(self, id: int) -> None:
        """Soft-delete a vector by global *id*."""
        self._inner.delete(id)

    def vacuum(self) -> int:
        """Permanently remove soft-deleted vectors.  Returns count removed."""
        return self._inner.vacuum()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the index to *path*."""
        self._inner.save(path)

    @classmethod
    def load(cls, path: str) -> "IVFPQIndex":
        """Load an index previously saved with :meth:`save`."""
        _require_bindings()
        obj = cls.__new__(cls)
        obj._inner = _PyIvfPqIndex.load(path)
        return obj

    def __repr__(self) -> str:
        return repr(self._inner)
