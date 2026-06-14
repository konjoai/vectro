"""
Product Quantization (PQ) API for Vectro v3, Phase 3.

Implements K-means codebook training and PQ encode/decode entirely in NumPy/
scikit-learn so it works without the Mojo binary.

Key features:
  - `train_pq_codebook`: per-subspace K-means with scikit-learn MiniBatchKMeans
  - `pq_encode` / `pq_decode`: vectorised NumPy encode/decode
  - `pq_distance_table` + `pq_search`: Asymmetric Distance Computation (ADC)
  - `opq_rotation`: optional OPQ pre-rotation for +5-10 pp recall

Reference:
  Jégou et al., "Product Quantization for Nearest Neighbor Search",
  IEEE TPAMI 33(1), 2011.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from . import _mojo_bridge as _mojo_bridge
except Exception:
    try:
        import _mojo_bridge as _mojo_bridge  # type: ignore
    except Exception:
        _mojo_bridge = None


@dataclass
class PQCodebook:
    """Trained PQ codebook.

    Attributes:
        n_subspaces: Number of sub-spaces M.
        n_centroids: Centroids per sub-space K (must be <= 256).
        sub_dim:     Dimension of each sub-space (d // M).
        centroids:   Shape (M, K, sub_dim), float32.
        rotation:    Optional OPQ rotation matrix, shape (d, d), float32.
    """

    n_subspaces: int
    n_centroids: int
    sub_dim: int
    centroids: np.ndarray  # (M, K, sub_dim) float32
    rotation: Optional[np.ndarray] = None  # (d, d) float32


def train_pq_codebook(
    training_data: np.ndarray,
    n_subspaces: int = 96,
    n_centroids: int = 256,
    max_iter: int = 25,
    random_state: int = 0,
) -> PQCodebook:
    """Train a PQ codebook via per-subspace K-means.

    Args:
        training_data: Shape (n_train, d); d must be divisible by n_subspaces.
        n_subspaces:   M — number of sub-spaces.
        n_centroids:   K — centroids per sub-space (default 256 for uint8 codes).
        max_iter:      Lloyd's K-means iterations.
        random_state:  Random seed for reproducibility.

    Returns:
        Trained PQCodebook.

    Raises:
        ValueError: If d is not divisible by n_subspaces, or K > 256.
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        raise ImportError(
            "scikit-learn required for PQ codebook training: pip install scikit-learn"
        )

    training_data = np.ascontiguousarray(training_data, dtype=np.float32)
    n_train, d = training_data.shape

    if d % n_subspaces != 0:
        raise ValueError(f"d={d} must be divisible by n_subspaces={n_subspaces}")
    if n_centroids > 256:
        raise ValueError(f"n_centroids={n_centroids} exceeds uint8 max (256)")

    sub_dim = d // n_subspaces
    centroids = np.empty((n_subspaces, n_centroids, sub_dim), dtype=np.float32)

    for m in range(n_subspaces):
        sub_vecs = training_data[:, m * sub_dim : (m + 1) * sub_dim]
        km = MiniBatchKMeans(
            n_clusters=n_centroids,
            max_iter=max_iter,
            random_state=random_state + m,
            n_init=1,
        )
        km.fit(sub_vecs)
        centroids[m] = km.cluster_centers_.astype(np.float32)

    return PQCodebook(
        n_subspaces=n_subspaces,
        n_centroids=n_centroids,
        sub_dim=sub_dim,
        centroids=centroids,
    )


def pq_encode(
    vectors: np.ndarray,
    codebook: PQCodebook,
) -> np.ndarray:
    """Encode float32 vectors to PQ codes (uint8, one byte per sub-space).

    Args:
        vectors:  Shape (n, d), float32.
        codebook: Trained PQCodebook.

    Returns:
        Codes of shape (n, M), dtype uint8.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    n, d = vectors.shape
    M = codebook.n_subspaces
    K = codebook.n_centroids
    sub_dim = codebook.sub_dim

    if codebook.rotation is not None:
        vectors = vectors @ codebook.rotation

    # Preferred path: Mojo PQ kernels via pipe protocol.
    if _mojo_bridge is not None and _mojo_bridge.is_available() and K <= 256:
        try:
            return _mojo_bridge.pq_encode(vectors, codebook.centroids)
        except Exception:
            # Fall through to NumPy path for robustness.
            pass

    codes = np.empty((n, M), dtype=np.uint8)

    for m in range(M):
        sub_vecs = vectors[:, m * sub_dim : (m + 1) * sub_dim]  # (n, sub_dim)
        cen = codebook.centroids[m]  # (K, sub_dim)

        # L2 distance via broadcasting: ||v - c||^2 = ||v||^2 + ||c||^2 - 2 v·c^T
        v_sq = (sub_vecs**2).sum(axis=1, keepdims=True)  # (n, 1)
        c_sq = (cen**2).sum(axis=1)  # (K,)
        cross = sub_vecs @ cen.T  # (n, K)
        dists = v_sq + c_sq - 2 * cross  # (n, K)
        codes[:, m] = dists.argmin(axis=1).astype(np.uint8)

    return codes


def pq_decode(
    codes: np.ndarray,
    codebook: PQCodebook,
) -> np.ndarray:
    """Decode PQ codes back to approximate float32 vectors.

    Args:
        codes:    Shape (n, M), dtype uint8.
        codebook: Trained PQCodebook.

    Returns:
        Reconstructed vectors of shape (n, d), float32.
    """
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    n, M = codes.shape
    sub_dim = codebook.sub_dim
    d = M * sub_dim

    # Preferred path: Mojo PQ kernels via pipe protocol.
    if _mojo_bridge is not None and _mojo_bridge.is_available() and codebook.n_centroids <= 256:
        try:
            out = _mojo_bridge.pq_decode(codes, codebook.centroids, d=d)
        except Exception:
            out = np.empty((n, d), dtype=np.float32)
            for m in range(M):
                out[:, m * sub_dim : (m + 1) * sub_dim] = codebook.centroids[m][codes[:, m]]
    else:
        out = np.empty((n, d), dtype=np.float32)
        for m in range(M):
            out[:, m * sub_dim : (m + 1) * sub_dim] = codebook.centroids[m][codes[:, m]]

    if codebook.rotation is not None:
        out = out @ codebook.rotation.T

    return out


def pq_distance_table(
    query: np.ndarray,
    codebook: PQCodebook,
) -> np.ndarray:
    """Pre-compute per-query ADC table for fast batch approximate distance.

    Args:
        query:    Shape (d,), float32.
        codebook: Trained PQCodebook.

    Returns:
        Table of shape (M, K), float32; entry [m, k] = L2 dist^2 to centroid k in sub-space m.
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    if codebook.rotation is not None:
        query = query @ codebook.rotation

    M = codebook.n_subspaces
    K = codebook.n_centroids
    sub_dim = codebook.sub_dim
    table = np.empty((M, K), dtype=np.float32)

    for m in range(M):
        q_sub = query[m * sub_dim : (m + 1) * sub_dim]  # (sub_dim,)
        cen = codebook.centroids[m]  # (K, sub_dim)
        diff = cen - q_sub  # (K, sub_dim)
        table[m] = (diff**2).sum(axis=1)

    return table


def pq_search(
    query: np.ndarray,
    codes: np.ndarray,
    codebook: PQCodebook,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find approximate nearest neighbours using ADC.

    Args:
        query:    Shape (d,), float32.
        codes:    Shape (n, M), uint8 — encoded database.
        codebook: Trained PQCodebook.
        top_k:    Number of results to return.

    Returns:
        (indices, distances) — top_k results sorted by ascending ADC distance.
    """
    table = pq_distance_table(query, codebook)  # (M, K)
    # ADC: for each DB vector, sum table[m, codes[i, m]] over m
    dists = table[np.arange(codebook.n_subspaces), codes].sum(axis=1)  # (n,)
    idx = np.argpartition(dists, min(top_k, len(dists) - 1))[:top_k]
    idx = idx[np.argsort(dists[idx])]
    return idx, dists[idx]


def pq_compression_ratio(d: int, M: int) -> float:
    """Theoretical compression ratio vs FP32 for PQ.

    Args:
        d: Vector dimension.
        M: Sub-spaces (bytes per compressed vector).

    Returns:
        Compression ratio (e.g. 32.0 means 32× smaller).
    """
    return (d * 4) / M


def opq_rotation(
    training_data: np.ndarray,
    n_subspaces: int,
    n_iter: int = 10,
    random_state: int = 0,
) -> Tuple[np.ndarray, PQCodebook]:
    """Train an OPQ rotation matrix and codebook (alternating optimisation).

    Applies a random Haar matrix as the initial rotation, then alternates
    between updating the PQ codebook and refining the rotation via SVD.

    Args:
        training_data: Shape (n_train, d), float32.
        n_subspaces:   M.
        n_iter:        Alternating optimisation iterations.
        random_state:  Seed.

    Returns:
        (rotation_matrix, codebook) — rotation of shape (d, d).
    """
    rng = np.random.default_rng(random_state)
    n_train, d = training_data.shape

    # Random Haar initialisation
    R = np.linalg.qr(rng.standard_normal((d, d)))[0].astype(np.float32)

    rotated = training_data @ R
    cb = train_pq_codebook(rotated, n_subspaces=n_subspaces)

    for _ in range(n_iter - 1):
        codes = pq_encode(rotated, cb)
        recon = pq_decode(codes, cb)  # (n, d) in rotated space
        # SVD-based rotation update: R* = V @ U^T where X = U S V^T
        U, _, Vt = np.linalg.svd((recon.T @ training_data), full_matrices=False)
        R = (Vt.T @ U.T).astype(np.float32)
        rotated = training_data @ R
        cb = train_pq_codebook(rotated, n_subspaces=n_subspaces)

    cb.rotation = R
    return R, cb
