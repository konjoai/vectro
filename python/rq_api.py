"""Residual Quantizer (RQ) — Phase 7a Learned Quantization.

A Residual Quantizer chains multiple Product Quantizer (PQ) codebooks:
each pass encodes the *residual* (original minus current reconstruction)
left by all previous passes.  The final reconstruction is the sum of all
per-pass decodings.

Typical use
-----------
>>> rq = ResidualQuantizer(n_passes=3, n_subspaces=8, n_centroids=64)
>>> rq.train(train_data)                 # fits 3 PQ codebooks
>>> codes = rq.encode(vectors)           # list of 3 code arrays, each (n, M)
>>> recon = rq.decode(codes)             # float32 (n, d)
>>> sim   = rq.mean_cosine(vectors, recon)   # ≥ 0.85 on small data

Why it beats a single PQ
------------------------
Each pass "mops up" the residual variance that the previous codebook missed.
In practice 2–4 passes on top of a standard PQ yields cosine quality
comparable to 3–4× as many sub-spaces.
"""

from __future__ import annotations

from typing import List

import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans  # type: ignore
except ImportError:
    MiniBatchKMeans = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_sklearn() -> None:
    if MiniBatchKMeans is None:
        raise ImportError(
            "scikit-learn is required for ResidualQuantizer. Install with: pip install scikit-learn"
        )


def _fit_pq_codebook(
    data: np.ndarray,
    n_subspaces: int,
    n_centroids: int,
    seed: int = 0,
) -> List[np.ndarray]:
    """Fit one PQ codebook: return list of sub-codebooks (each n_centroids × sub_d).

    Parameters
    ----------
    data         : (n, d) float32
    n_subspaces  : number of sub-spaces M; d must be divisible by M
    n_centroids  : K — number of centroids per sub-space
    seed         : random state for reproducibility

    Returns
    -------
    codebooks : list of M arrays, each (K, sub_d)
    """
    n, d = data.shape
    if d % n_subspaces != 0:
        # Pad data so dims is divisible
        pad = n_subspaces - (d % n_subspaces)
        data = np.pad(data, ((0, 0), (0, pad)))
        d = data.shape[1]
    sub_d = d // n_subspaces

    codebooks: List[np.ndarray] = []
    for m in range(n_subspaces):
        sub = data[:, m * sub_d : (m + 1) * sub_d]
        km = MiniBatchKMeans(
            n_clusters=n_centroids,
            random_state=seed + m,
            n_init=3,
            max_iter=100,
        )
        km.fit(sub)
        codebooks.append(km.cluster_centers_.astype(np.float32))
    return codebooks


def _pq_encode_one(
    data: np.ndarray,
    codebooks: List[np.ndarray],
    n_subspaces: int,
) -> np.ndarray:
    """PQ encode (n, d) using fitted codebooks → uint8 (n, M)."""
    n, d_orig = data.shape
    d_pad = codebooks[0].shape[1] * n_subspaces
    if d_orig < d_pad:
        data = np.pad(data, ((0, 0), (0, d_pad - d_orig)))
    sub_d = codebooks[0].shape[1]
    codes = np.empty((n, n_subspaces), dtype=np.uint8)
    for m in range(n_subspaces):
        sub = data[:, m * sub_d : (m + 1) * sub_d]
        diffs = sub[:, np.newaxis, :] - codebooks[m][np.newaxis, :, :]  # (n, K, sub_d)
        codes[:, m] = np.argmin((diffs**2).sum(axis=2), axis=1).astype(np.uint8)
    return codes


def _pq_decode_one(
    codes: np.ndarray,
    codebooks: List[np.ndarray],
    n_subspaces: int,
    d_orig: int,
) -> np.ndarray:
    """PQ decode uint8 (n, M) → float32 (n, d_orig)."""
    n = codes.shape[0]
    sub_d = codebooks[0].shape[1]
    recon = np.empty((n, n_subspaces * sub_d), dtype=np.float32)
    for m in range(n_subspaces):
        recon[:, m * sub_d : (m + 1) * sub_d] = codebooks[m][codes[:, m].astype(int)]
    return recon[:, :d_orig]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ResidualQuantizer:
    """Multi-pass Residual Quantizer.

    Parameters
    ----------
    n_passes    : int  Number of PQ passes (default 3)
    n_subspaces : int  M — sub-spaces per PQ pass (default 8)
    n_centroids : int  K — centroids per sub-space (default 64)
    seed        : int  Random state for KMeans
    """

    def __init__(
        self,
        n_passes: int = 3,
        n_subspaces: int = 8,
        n_centroids: int = 64,
        seed: int = 0,
    ) -> None:
        _check_sklearn()
        self.n_passes = n_passes
        self.n_subspaces = n_subspaces
        self.n_centroids = n_centroids
        self.seed = seed
        self._codebooks: List[List[np.ndarray]] = []  # [pass][subspace] → centroids
        self._d_orig: int = 0
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data: np.ndarray) -> "ResidualQuantizer":
        """Fit n_passes PQ codebooks on the training data.

        Parameters
        ----------
        data : (n, d) float32 training vectors

        Returns
        -------
        self (for chaining)
        """
        data = np.ascontiguousarray(data, dtype=np.float32)
        self._d_orig = data.shape[1]
        residual = data.copy()
        self._codebooks = []

        for p in range(self.n_passes):
            cbs = _fit_pq_codebook(
                residual,
                self.n_subspaces,
                self.n_centroids,
                seed=self.seed + p * 100,
            )
            # Compute codes + reconstruction of this residual
            codes = _pq_encode_one(residual, cbs, self.n_subspaces)
            recon = _pq_decode_one(codes, cbs, self.n_subspaces, residual.shape[1])
            residual = residual - recon
            self._codebooks.append(cbs)

        self.is_trained = True
        return self

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, vectors: np.ndarray) -> List[np.ndarray]:
        """Encode vectors through all passes.

        Parameters
        ----------
        vectors : (n, d) float32

        Returns
        -------
        List[np.ndarray]  — n_passes arrays each of shape (n, M) uint8
        """
        if not self.is_trained:
            raise RuntimeError("ResidualQuantizer.train() must be called first.")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        all_codes: List[np.ndarray] = []
        residual = vectors.copy()

        for p, cbs in enumerate(self._codebooks):
            codes = _pq_encode_one(residual, cbs, self.n_subspaces)
            recon = _pq_decode_one(codes, cbs, self.n_subspaces, residual.shape[1])
            residual = residual - recon
            all_codes.append(codes)

        return all_codes

    def decode(self, codes_list: List[np.ndarray]) -> np.ndarray:
        """Reconstruct float32 vectors from multi-pass codes.

        Parameters
        ----------
        codes_list : List[np.ndarray]  — as returned by .encode()

        Returns
        -------
        np.ndarray, (n, d_orig) float32
        """
        if not self.is_trained:
            raise RuntimeError("ResidualQuantizer.train() must be called first.")
        n = codes_list[0].shape[0]
        result = np.zeros((n, self._d_orig), dtype=np.float32)
        for p, codes in enumerate(codes_list):
            recon = _pq_decode_one(codes, self._codebooks[p], self.n_subspaces, self._d_orig)
            result += recon
        return result

    # ------------------------------------------------------------------
    # Quality metric
    # ------------------------------------------------------------------

    def mean_cosine(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> float:
        """Return mean per-vector cosine similarity between original and reconstructed."""
        original = np.ascontiguousarray(original, dtype=np.float32)
        reconstructed = np.ascontiguousarray(reconstructed, dtype=np.float32)
        dots = (original * reconstructed).sum(axis=1)
        norms = np.linalg.norm(original, axis=1) * np.linalg.norm(reconstructed, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        return float((dots / norms).mean())

    def compression_ratio(self) -> float:
        """Return compression ratio vs float32 storage.

        Each vector is stored as n_passes × n_subspaces uint8 codes.
        Float32 storage is d × 4 bytes.  Code storage is n_passes × M bytes.
        """
        code_bytes = self.n_passes * self.n_subspaces  # uint8
        float_bytes = self._d_orig * 4 if self._d_orig > 0 else 1
        return float_bytes / max(code_bytes, 1)
