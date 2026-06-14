"""api/store.py — in-memory vector index + numpy-only PCA / k-means.

The visualization endpoints in :mod:`api.app` and the live demo in
``demo/server.py`` both use these helpers, so the math (and therefore
the rendered scatter plot) is identical across entrypoints.

The PCA implementation is the textbook two-line SVD on centred data:

    Xc = X - mean(X)
    U, S, Vt = svd(Xc, full_matrices=False)
    coords = U[:, :2] * S[:2]    # === Xc @ Vt[:2].T

K-means uses k-means++ initialisation followed by Lloyd iterations.
Both functions accept an empty matrix and a single point without
raising — useful when the UI calls /project before any vectors are
added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Index data structure
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Index:
    """A named in-memory vector index — fp32 vectors + parallel id /
    metadata lists.  Vectors are stored as a list (not a 2-D array) so
    appends are O(1); :meth:`matrix` materialises the (N, dim) view on
    demand for the linear-algebra paths."""

    name: str
    dim: int
    vectors: List[np.ndarray] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    def matrix(self) -> np.ndarray:
        if not self.vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(self.vectors).astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.vectors)


class IndexStore:
    """Thread-safe ``name -> Index`` map.  The same instance backs both
    the FastAPI app (one per process) and the demo server."""

    def __init__(self) -> None:
        self._indexes: Dict[str, Index] = {}
        self._lock = RLock()

    def create(self, name: str, dim: int) -> Index:
        with self._lock:
            if name in self._indexes:
                raise ValueError(f"index already exists: {name!r}")
            if dim < 1:
                raise ValueError(f"dim must be >= 1, got {dim}")
            idx = Index(name=name, dim=int(dim))
            self._indexes[name] = idx
            return idx

    def get(self, name: str) -> Index:
        with self._lock:
            if name not in self._indexes:
                raise KeyError(name)
            return self._indexes[name]

    def get_or_create(self, name: str, dim: int) -> Index:
        with self._lock:
            if name not in self._indexes:
                return self.create(name, dim)
            return self._indexes[name]

    def delete(self, name: str) -> bool:
        with self._lock:
            return self._indexes.pop(name, None) is not None

    def names(self) -> List[str]:
        with self._lock:
            return list(self._indexes.keys())

    def reset(self) -> None:
        with self._lock:
            self._indexes.clear()


# ─────────────────────────────────────────────────────────────────────────
# PCA — SVD on centred data, top 2 components
# ─────────────────────────────────────────────────────────────────────────


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Project ``X`` (N, D) onto its top-2 principal components.

    Edge cases — all return float32:
      * N == 0  → shape (0, 2)
      * N == 1  → shape (1, 2) of zeros (one point sits at the centroid)
      * D == 1  → shape (N, 2) with zeros padded into the second column
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"pca_2d expects a 2-D matrix, got shape {X.shape}")
    n, d = X.shape
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n == 1:
        return np.zeros((1, 2), dtype=np.float32)

    Xc = X - X.mean(axis=0, keepdims=True)
    k = min(2, n, d)
    # SVD on the centred data: U @ diag(S) @ Vt = Xc.
    # Score on the i-th principal axis is U[:, i] * S[i].
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = U[:, :k] * S[:k]
    if k < 2:
        pad = np.zeros((n, 2 - k), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)
    return coords.astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────
# K-means — k-means++ init + Lloyd iterations
# ─────────────────────────────────────────────────────────────────────────


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    max_iter: int = 50,
    seed: int = 0,
    tol: float = 1e-6,
) -> np.ndarray:
    """Cluster the rows of ``X`` into ``k`` groups; return labels (N,).

    ``k`` is clamped to ``[1, N]``.  Empty input returns an empty
    int32 array.  Centres are initialised with k-means++ (probability
    proportional to squared distance from the nearest existing centre)
    and refined with up to ``max_iter`` Lloyd iterations; the loop
    breaks early once labels are stable or centroid movement falls
    below ``tol``.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"kmeans expects a 2-D matrix, got shape {X.shape}")
    n = X.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    k = max(1, min(int(k), n))
    rng = np.random.default_rng(int(seed))

    # k-means++ initialisation.
    centres = np.empty((k, X.shape[1]), dtype=np.float32)
    centres[0] = X[int(rng.integers(n))]
    for i in range(1, k):
        diff = X[:, None, :] - centres[None, :i, :]
        d2 = (diff * diff).sum(axis=-1).min(axis=1)
        total = float(d2.sum())
        if total <= 0.0:
            centres[i] = X[int(rng.integers(n))]
        else:
            probs = d2 / total
            centres[i] = X[int(rng.choice(n, p=probs))]

    labels = np.full((n,), -1, dtype=np.int32)
    for _ in range(max_iter):
        diff = X[:, None, :] - centres[None, :, :]
        d2 = (diff * diff).sum(axis=-1)
        new_labels = d2.argmin(axis=1).astype(np.int32)
        if (new_labels == labels).all():
            break
        labels = new_labels
        new_centres = centres.copy()
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centres[c] = X[mask].mean(axis=0)
        if float(np.linalg.norm(new_centres - centres)) < tol:
            centres = new_centres
            break
        centres = new_centres
    return labels


# ─────────────────────────────────────────────────────────────────────────
# Cosine search — used by the FastAPI /search route and demo viz
# ─────────────────────────────────────────────────────────────────────────


def cosine_topk(M: np.ndarray, q: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Return the top-``k`` rows of ``M`` by cosine similarity to ``q``.

    Empty matrix yields an empty result.  The returned list is sorted
    descending by score and capped at ``k`` (clamped to N).
    """
    M = np.asarray(M, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if M.shape[0] == 0:
        return []
    if q.shape[0] != M.shape[1]:
        raise ValueError(f"query dim {q.shape[0]} does not match index dim {M.shape[1]}")
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        q_norm = 1.0
    qn = q / q_norm
    m_norms = np.linalg.norm(M, axis=1)
    m_norms[m_norms == 0.0] = 1.0
    Mn = M / m_norms[:, None]
    scores = (Mn @ qn).astype(np.float32, copy=False)

    k = max(1, min(int(k), M.shape[0]))
    if k >= M.shape[0]:
        order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
    return [{"index": int(i), "score": float(scores[i])} for i in order]
