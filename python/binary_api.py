"""
Binary (1-bit) quantization Python API for Vectro v3, Phase 4.

sign(v) → 1-bit per dimension, 8 bits packed per byte.
Supports:
  - batch encode / decode
  - Hamming distance computation
  - top-k nearest neighbour search
  - optional L2 normalisation before encoding (recommended)
  - Matryoshka-aware encoding (encode at multiple prefix lengths)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List

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


def quantize_binary(
    vectors: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Encode float32 vectors to binary (sign bit), packed 8 per byte.

    Uses the compiled Mojo binary when available; falls back to NumPy.

    Args:
        vectors:   Shape (n, d), float32.
        normalize: If True, L2-normalise each vector before encoding.

    Returns:
        Packed binary array of shape (n, ceil(d/8)), dtype uint8.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    n, d = vectors.shape

    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe = np.where(norms == 0, 1.0, norms)
        vectors = vectors / safe

    if _mojo_bridge.is_available():
        return _mojo_bridge.bin_encode(vectors)

    bits = (vectors > 0.0).astype(np.uint8)  # (n, d)

    bytes_per_vec = (d + 7) // 8
    packed = np.zeros((n, bytes_per_vec), dtype=np.uint8)

    for bit_pos in range(8):
        col_start = np.arange(bytes_per_vec) * 8 + bit_pos
        valid = col_start < d
        packed[:, valid] |= (bits[:, col_start[valid]] << bit_pos).astype(np.uint8)

    return packed


def dequantize_binary(
    packed: np.ndarray,
    d: int,
) -> np.ndarray:
    """Decode binary packed bytes to {-1, +1} float32 vectors.

    Uses the compiled Mojo binary when available; falls back to NumPy.

    Args:
        packed: Shape (n, ceil(d/8)), dtype uint8.
        d:      Original vector dimension.

    Returns:
        Float32 array of shape (n, d); each element is +1.0 or -1.0.
    """
    if _mojo_bridge.is_available():
        return _mojo_bridge.bin_decode(packed, d)

    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    n, bytes_per_vec = packed.shape

    out = np.full((n, bytes_per_vec * 8), -1.0, dtype=np.float32)
    for bit_pos in range(8):
        cols = np.arange(bytes_per_vec) * 8 + bit_pos
        mask = ((packed >> bit_pos) & 1).astype(np.float32)  # (n, bytes)
        out[:, cols] = np.where(mask == 1, 1.0, -1.0)

    return out[:, :d]


def hamming_distance_batch(
    query_packed: np.ndarray,
    db_packed: np.ndarray,
) -> np.ndarray:
    """Compute Hamming distances from one query to n database vectors.

    Uses numpy XOR + unpackbits popcount.

    Args:
        query_packed: Shape (bytes_per_vec,), uint8.
        db_packed:    Shape (n, bytes_per_vec), uint8.

    Returns:
        Integer array of shape (n,) with Hamming distances.
    """
    query_packed = np.ascontiguousarray(query_packed, dtype=np.uint8).ravel()
    xor = db_packed ^ query_packed[np.newaxis]  # (n, B)
    bits = np.unpackbits(xor, axis=1, bitorder="little")  # (n, B*8)
    return bits.sum(axis=1).astype(np.int32)


def binary_search(
    query: np.ndarray,
    db_packed: np.ndarray,
    top_k: int = 10,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find approximate nearest neighbours by Hamming distance.

    Args:
        query:      Shape (d,), float32.
        db_packed:  Shape (n, bytes_per_vec), uint8 — encoded database.
        top_k:      Number of results.
        normalize:  L2-normalise query before encoding.

    Returns:
        (indices, distances) — top_k results sorted by ascending Hamming distance.
    """
    q_packed = quantize_binary(query[np.newaxis], normalize=normalize).ravel()
    dists = hamming_distance_batch(q_packed, db_packed)
    idx = np.argpartition(dists, min(top_k, len(dists) - 1))[:top_k]
    idx = idx[np.argsort(dists[idx])]
    return idx, dists[idx]


def matryoshka_encode(
    vectors: np.ndarray,
    dims: List[int],
    normalize: bool = True,
) -> dict:
    """Encode vectors at multiple Matryoshka prefix lengths.

    Args:
        vectors:   Shape (n, d_full), float32.
        dims:      List of prefix lengths to encode (e.g. [64, 128, 256, 512, 768]).
        normalize: L2-normalise each prefix independently.

    Returns:
        Dict mapping dim -> packed uint8 array of shape (n, ceil(dim/8)).
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    return {dim: quantize_binary(vectors[:, :dim], normalize=normalize) for dim in dims}


def binary_compression_ratio(d: int) -> float:
    """Theoretical compression ratio of binary vs FP32.

    Args:
        d: Vector dimension.

    Returns:
        Compression ratio (e.g. 32.0 for d=256).
    """
    return (d * 4) / ((d + 7) // 8)
