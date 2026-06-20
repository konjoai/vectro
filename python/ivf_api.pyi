"""Type stubs for python/ivf_api.py — IVFIndex and IVFPQIndex."""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

class IVFIndex:
    """Inverted-file index backed by ``PyIvfIndex``.

    Parameters
    ----------
    n_lists : int
        Number of Voronoi cells (clusters).
    n_probe : int
        Number of cells probed at query time.
    """

    def __init__(self, n_lists: int, n_probe: int) -> None: ...

    # ── Training ────────────────────────────────────────────────────────────
    def train(self, vectors: List[List[float]]) -> None:
        """Train the quantizer on ``vectors``. Must be called before ``add``."""
        ...

    def train_np(self, array: np.ndarray) -> None:
        """Train from a 2-D float32 ndarray of shape ``(n, dim)``."""
        ...

    # ── Mutation ─────────────────────────────────────────────────────────────
    def add(self, id: str, vector: List[float]) -> int:
        """Insert a vector; returns the internal integer slot assigned."""
        ...

    def add_np(self, id: str, array: np.ndarray) -> int:
        """Insert from a 1-D float32 ndarray."""
        ...

    def delete(self, id: str) -> bool:
        """Soft-delete a vector by id. Returns ``True`` if found."""
        ...

    def vacuum(self) -> None:
        """Compact the index, removing soft-deleted entries."""
        ...

    # ── Search ───────────────────────────────────────────────────────────────
    def search(self, query: List[float], k: int) -> List[Tuple[str, float]]:
        """Return ``k`` nearest neighbours as ``[(id, score), …]``."""
        ...

    def search_np(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Same as ``search`` but accepts a 1-D float32 ndarray."""
        ...

    def search_with_probe(
        self, query: List[float], k: int, n_probe: int
    ) -> List[Tuple[str, float]]:
        """Search with a per-query probe count override."""
        ...

    def search_filtered_np(
        self,
        query: np.ndarray,
        k: int,
        allowed_ids: List[str],
    ) -> List[Tuple[str, float]]:
        """Search restricted to ``allowed_ids``; accepts float32 ndarray."""
        ...

    def search_for_recall(self, query: List[float], k: int) -> List[Tuple[str, float]]:
        """Search variant optimised for recall measurement."""
        ...

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Persist the index to ``path``."""
        ...

    @classmethod
    def load(cls, path: str) -> "IVFIndex":
        """Load a previously saved index from ``path``."""
        ...

    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...

class IVFPQIndex:
    """Inverted-file index with product quantization, backed by ``PyIvfPqIndex``.

    Parameters
    ----------
    n_lists : int
        Number of Voronoi cells (clusters).
    n_probe : int
        Number of cells probed at query time.
    """

    def __init__(self, n_lists: int, n_probe: int) -> None: ...

    # ── Training ─────────────────────────────────────────────────────────────
    def train(
        self,
        vectors: List[List[float]],
        n_subspaces: int = 8,
        n_centroids: int = 256,
        max_iter: int = 25,
        seed: int = 42,
    ) -> None:
        """Train the IVF+PQ quantizer.

        Parameters
        ----------
        vectors : list[list[float]]
            Training corpus.
        n_subspaces : int
            Number of PQ subspaces (must divide ``dim`` evenly).
        n_centroids : int
            Centroids per subspace (typically 256 for 8-bit codes).
        max_iter : int
            k-means iterations for PQ training.
        seed : int
            Random seed for reproducibility.
        """
        ...

    def train_np(
        self,
        array: np.ndarray,
        n_subspaces: int = 8,
        n_centroids: int = 256,
        max_iter: int = 25,
        seed: int = 42,
    ) -> None:
        """Train from a 2-D float32 ndarray."""
        ...

    # ── Mutation ─────────────────────────────────────────────────────────────
    def add(self, id: str, vector: List[float]) -> int: ...
    def add_np(self, id: str, array: np.ndarray) -> int: ...
    def delete(self, id: str) -> bool: ...
    def vacuum(self) -> None: ...

    # ── Search ───────────────────────────────────────────────────────────────
    def search(self, query: List[float], k: int) -> List[Tuple[str, float]]: ...
    def search_np(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]: ...
    def search_with_probe(
        self, query: List[float], k: int, n_probe: int
    ) -> List[Tuple[str, float]]: ...
    def search_for_recall(self, query: List[float], k: int) -> List[Tuple[str, float]]: ...

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "IVFPQIndex": ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
