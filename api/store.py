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

import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence

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
# Cosine scoring — dense leg of the /search route (and demo viz)
# ─────────────────────────────────────────────────────────────────────────


def cosine_scores(M: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row of ``M`` to ``q`` — an (N,) float32
    array.  Zero-norm rows / queries are treated as unit norm so they score
    0 rather than dividing by zero.  Empty matrix yields an empty array."""
    M = np.asarray(M, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if M.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if q.shape[0] != M.shape[1]:
        raise ValueError(f"query dim {q.shape[0]} does not match index dim {M.shape[1]}")
    q_norm = float(np.linalg.norm(q)) or 1.0
    qn = q / q_norm
    m_norms = np.linalg.norm(M, axis=1)
    m_norms[m_norms == 0.0] = 1.0
    Mn = M / m_norms[:, None]
    return (Mn @ qn).astype(np.float32, copy=False)


def _topk_order(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest scores, sorted descending.  ``k`` is
    clamped to ``[1, len(scores)]``."""
    n = scores.shape[0]
    k = max(1, min(int(k), n))
    if k >= n:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


# ─────────────────────────────────────────────────────────────────────────
# BM25 + hybrid (dense ⊕ sparse) search — used by POST /index/{name}/search
# ─────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric word tokenization shared by the BM25 path."""
    return _TOKEN_RE.findall(text.lower())


def bm25_scores(
    docs: Sequence[str],
    query: str,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """Okapi BM25 relevance of each document in ``docs`` to ``query``.

    Returns an (N,) float32 array aligned to ``docs``; all-zero when the
    corpus is empty, every document is empty, or the query has no tokens.
    Pure-numpy with no external dependencies, matching the self-contained
    design of :func:`cosine_scores`.
    """
    n = len(docs)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    q_terms = set(tokenize(query))
    if not q_terms:
        return np.zeros((n,), dtype=np.float32)

    doc_tokens = [tokenize(d) for d in docs]
    doc_len = np.array([len(t) for t in doc_tokens], dtype=np.float32)
    avgdl = float(doc_len.mean())
    if avgdl == 0.0:
        return np.zeros((n,), dtype=np.float32)

    scores = np.zeros((n,), dtype=np.float32)
    for term in q_terms:
        tf = np.array([tokens.count(term) for tokens in doc_tokens], dtype=np.float32)
        df = float((tf > 0.0).sum())
        if df == 0.0:
            continue
        idf = float(np.log(1.0 + (n - df + 0.5) / (df + 0.5)))
        denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
        scores += idf * (tf * (k1 + 1.0)) / np.where(denom == 0.0, 1.0, denom)
    return scores.astype(np.float32, copy=False)


def _minmax(x: np.ndarray) -> np.ndarray:
    """Scale scores into [0, 1].  A constant vector maps to all-zeros — a
    component that cannot discriminate contributes nothing to the fusion."""
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32, copy=False)


def hybrid_topk(
    M: np.ndarray,
    docs: Sequence[str],
    *,
    query: Optional[np.ndarray] = None,
    text: Optional[str] = None,
    k: int = 5,
    alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    """Top-``k`` results fusing dense cosine and BM25 sparse rankings.

    * ``query`` only  → pure dense cosine.
    * ``text`` only   → pure BM25.
    * both            → ``alpha * minmax(dense) + (1 - alpha) * minmax(bm25)``
      where ``alpha = 1.0`` is dense-only and ``alpha = 0.0`` is BM25-only.

    Each hit carries the fused ``score`` plus the raw ``dense_score`` and
    ``bm25_score`` for transparency.  Raises ``ValueError`` when neither
    ``query`` nor ``text`` is supplied.
    """
    M = np.asarray(M, dtype=np.float32)
    n = M.shape[0]
    if n == 0:
        return []

    dense = cosine_scores(M, query) if query is not None else None
    sparse = bm25_scores(docs, text) if text is not None else None
    if dense is None and sparse is None:
        raise ValueError("hybrid_topk requires at least one of `query` or `text`")

    dense_raw = dense if dense is not None else np.zeros((n,), dtype=np.float32)
    sparse_raw = sparse if sparse is not None else np.zeros((n,), dtype=np.float32)

    if dense is not None and sparse is not None:
        a = float(min(1.0, max(0.0, alpha)))
        fused = a * _minmax(dense_raw) + (1.0 - a) * _minmax(sparse_raw)
    else:
        fused = dense_raw if dense is not None else sparse_raw

    order = _topk_order(fused.astype(np.float32, copy=False), k)
    return [
        {
            "index": int(i),
            "score": float(fused[i]),
            "dense_score": float(dense_raw[i]),
            "bm25_score": float(sparse_raw[i]),
        }
        for i in order
    ]
