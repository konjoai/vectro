"""
Python API for NF4 (Normal Float 4-bit) quantization — Vectro v3, Phase 2.

Implements the NF4 encode/decode + optional mixed-precision (top-k FP16
outlier dims stored full-width, remainder NF4-encoded) entirely in NumPy
so it works without the Mojo binary.  The Mojo path (quantizer_simd.mojo)
is the performance-critical path; this module handles:
  - batch encode/decode via NumPy vectorised operations
  - mixed-precision support (flag top-k outlier dimensions per model)
  - quality validation: reconstruction cosine similarity
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

try:
    from . import _mojo_bridge
except ImportError:
    import importlib.util as _ilu
    import pathlib as _pl

    _spec = _ilu.spec_from_file_location(
        "_mojo_bridge", _pl.Path(__file__).parent / "_mojo_bridge.py"
    )
    _mojo_bridge = _ilu.module_from_spec(_spec)  # type: ignore[assignment]
    _spec.loader.exec_module(_mojo_bridge)  # type: ignore[union-attr]

# Optional Rust SIMD fast-path (vectro_py).  Used as a middle tier when the
# Mojo binary is absent but the PyO3 extension is installed.
try:
    import vectro_py as _vectro_py  # type: ignore[import]

    _HAS_VECTRO_PY = True
except ImportError:
    _vectro_py = None  # type: ignore[assignment]
    _HAS_VECTRO_PY = False


# NF4 codebook — quantiles of N(0,1), Dettmers et al. 2023
NF4_LEVELS: np.ndarray = np.array(
    [
        -1.0,
        -0.6961928,
        -0.5250730,
        -0.3949003,
        -0.2844677,
        -0.1848745,
        -0.09105004,
        0.0,
        0.07958031,
        0.16093908,
        0.24611496,
        0.33791524,
        0.44070983,
        0.56266755,
        0.72295761,
        1.0,
    ],
    dtype=np.float32,
)

# Half-levels for nearest-neighbour lookup (midpoints between adjacent levels)
_NF4_THRESHOLDS: np.ndarray = (NF4_LEVELS[:-1] + NF4_LEVELS[1:]) / 2.0


def quantize_nf4(
    vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode float32 vectors to NF4 packed bytes.

    Uses the compiled Mojo binary when available; falls back to NumPy.

    Args:
        vectors: Shape (n, d), dtype float32.

    Returns:
        packed: Shape (n, ceil(d/2)), dtype uint8.  Low nibble = even dim,
                high nibble = odd dim.
        scales: Shape (n,), dtype float32; per-vector abs-max factors.
    """
    if _mojo_bridge.is_available():
        return _mojo_bridge.nf4_encode(vectors)

    # Second-tier: Rust SIMD via vectro_py.encode_nf4_fast (per-vector loop).
    # Faster than pure NumPy for small-to-medium batches; falls through to
    # NumPy for environments without the extension.
    if _HAS_VECTRO_PY:
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis]
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = vectors.shape
        packed_rows = []
        scales = np.empty(n, dtype=np.float32)
        for i in range(n):
            packed_list, scale, _dim = _vectro_py.encode_nf4_fast(vectors[i].tolist())
            packed_rows.append(bytes(packed_list))
            scales[i] = scale
        packed = np.frombuffer(b"".join(packed_rows), dtype=np.uint8).reshape(n, -1)
        return packed, scales

    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n, d = vectors.shape

    scales = np.abs(vectors).max(axis=1, keepdims=True)  # (n, 1)
    scales_1d = scales.ravel().copy()
    safe_scales = np.where(scales == 0.0, 1.0, scales)
    normed = vectors / safe_scales  # (n, d) in [-1, 1]

    # Nearest NF4 index via searchsorted on midpoint thresholds
    flat = normed.ravel()
    indices = np.searchsorted(_NF4_THRESHOLDS, flat).astype(np.uint8)  # (n*d,)
    indices = indices.reshape(n, d)

    # Pack two nibbles per byte
    bytes_per_vec = (d + 1) // 2
    packed = np.zeros((n, bytes_per_vec), dtype=np.uint8)
    packed[:, : d // 2] = (indices[:, 1::2][:, : d // 2] << 4) | indices[:, 0::2][:, : d // 2]
    if d % 2 == 1:
        packed[:, d // 2] = indices[:, d - 1]

    return packed, scales_1d


def dequantize_nf4(
    packed: np.ndarray,
    scales: np.ndarray,
    d: int,
) -> np.ndarray:
    """Decode NF4 packed bytes back to float32.

    Uses the compiled Mojo binary when available; falls back to NumPy.

    Args:
        packed: Shape (n, ceil(d/2)), dtype uint8.
        scales: Shape (n,), dtype float32.
        d: Original vector dimension.

    Returns:
        Reconstructed float32 array of shape (n, d).
    """
    if _mojo_bridge.is_available():
        return _mojo_bridge.nf4_decode(packed, scales, d)

    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    n = packed.shape[0]
    bytes_per_vec = (d + 1) // 2

    out = np.empty((n, d), dtype=np.float32)

    d_even = (d // 2) * 2
    if d_even > 0:
        lo = (packed[:, : d_even // 2] & 0x0F).astype(np.int32)
        hi = ((packed[:, : d_even // 2] >> 4) & 0x0F).astype(np.int32)
        out[:, 0::2][:, : d_even // 2] = NF4_LEVELS[lo]
        out[:, 1::2][:, : d_even // 2] = NF4_LEVELS[hi]

    if d % 2 == 1:
        lo_last = (packed[:, bytes_per_vec - 1] & 0x0F).astype(np.int32)
        out[:, d - 1] = NF4_LEVELS[lo_last]

    out *= scales[:, np.newaxis]
    return out


# ────────────────────────────────────────────────────────────────────────────
# Mixed-precision: top-k outlier dims in FP16, remainder NF4
# ────────────────────────────────────────────────────────────────────────────


def select_outlier_dims(
    training_data: np.ndarray,
    k: int = 16,
) -> np.ndarray:
    """Return the k highest-variance dimension indices.

    Args:
        training_data: Shape (n_train, d), float32.
        k: Number of outlier dimensions to preserve.

    Returns:
        Sorted array of k dimension indices.
    """
    variances = training_data.var(axis=0)
    return np.sort(np.argpartition(variances, -k)[-k:])


def quantize_mixed(
    vectors: np.ndarray,
    outlier_dims: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mixed-precision encode: outlier dims as FP16, rest as NF4.

    Args:
        vectors: Shape (n, d), float32.
        outlier_dims: 1-D integer array of outlier dimension indices.

    Returns:
        fp16_vals: (n, k) float16 — outlier values.
        nf4_packed: (n, ceil((d-k)/2)) uint8 — NF4 bulk bytes.
        nf4_scales: (n,) float32 — per-vector NF4 scale factors.
        outlier_dims: the same outlier_dims array (for decode).
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n, d = vectors.shape

    mask = np.zeros(d, dtype=bool)
    mask[outlier_dims] = True
    bulk_dims = np.where(~mask)[0]

    fp16_vals = vectors[:, outlier_dims].astype(np.float16)
    bulk = vectors[:, bulk_dims]
    nf4_packed, nf4_scales = quantize_nf4(bulk)

    return fp16_vals, nf4_packed, nf4_scales, outlier_dims


def dequantize_mixed(
    fp16_vals: np.ndarray,
    nf4_packed: np.ndarray,
    nf4_scales: np.ndarray,
    outlier_dims: np.ndarray,
    d: int,
) -> np.ndarray:
    """Reconstruct float32 from mixed-precision encoding.

    Args:
        fp16_vals: (n, k) float16.
        nf4_packed: (n, ceil((d-k)/2)) uint8.
        nf4_scales: (n,) float32.
        outlier_dims: 1-D integer array of outlier dimension indices.
        d: Full vector dimension.

    Returns:
        Reconstructed float32 array of shape (n, d).
    """
    n = fp16_vals.shape[0]
    k = len(outlier_dims)
    bulk_d = d - k

    mask = np.zeros(d, dtype=bool)
    mask[outlier_dims] = True
    bulk_dims = np.where(~mask)[0]

    out = np.empty((n, d), dtype=np.float32)
    out[:, outlier_dims] = fp16_vals.astype(np.float32)
    out[:, bulk_dims] = dequantize_nf4(nf4_packed, nf4_scales, bulk_d)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Quality helpers
# ────────────────────────────────────────────────────────────────────────────


def nf4_cosine_sim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean cosine similarity between original and reconstructed vectors.

    Args:
        original: (n, d) float32.
        reconstructed: (n, d) float32.

    Returns:
        Mean cosine similarity as a Python float.
    """
    dot = (original * reconstructed).sum(axis=1)
    norm_o = np.linalg.norm(original, axis=1)
    norm_r = np.linalg.norm(reconstructed, axis=1)
    denom = norm_o * norm_r
    safe = denom > 0
    cos = np.where(safe, dot / np.where(safe, denom, 1.0), 0.0)
    return float(cos.mean())


def compression_ratio(d: int, k_outliers: int = 0) -> float:
    """Theoretical compression ratio vs FP32.

    Args:
        d: Vector dimension.
        k_outliers: Number of outlier dims stored as FP16 (0 = pure NF4).

    Returns:
        Compression ratio (e.g. 8.0 means 8× smaller than FP32).
    """
    fp32_bytes = d * 4
    if k_outliers == 0:
        nf4_bytes = (d + 1) // 2  # packed nibbles
        scale_bytes = 4
        return fp32_bytes / (nf4_bytes + scale_bytes)
    else:
        bulk_d = d - k_outliers
        nf4_bytes = (bulk_d + 1) // 2
        fp16_bytes = k_outliers * 2
        scale_bytes = 4
        return fp32_bytes / (nf4_bytes + fp16_bytes + scale_bytes)
