"""Scalar quantization (SQ2 / SQ3) — NumPy reference matching the Rust kernels.

SQ2 — 2-bit uniform: 4 reconstruction levels, 4 codes packed per byte (LSB-first).
SQ3 — 3-bit uniform: 8 reconstruction levels, packed as an LSB-first bit stream.

Each dimension is mapped to a uniform level relative to the vector's abs-max::

    code  = clamp(floor((clamp(x / abs_max, -1, 1) + 1) * (L / 2)), 0, L - 1)
    value = abs_max * (2 * code - (L - 1)) / L

where ``L = 2 ** bits``.  This is the Python-only correctness baseline
(CLAUDE.md): the bit layout and reconstruction levels exactly match
``vectro_lib::quant::sq2`` / ``sq3`` (4 codes/byte for SQ2; a 3-bit LSB-first
bit stream that may straddle byte boundaries for SQ3).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _abs_max_scales(vectors: np.ndarray) -> np.ndarray:
    """Per-vector abs-max scale; zero vectors map to 1.0 (mirrors the Rust kernel)."""
    abs_max = np.max(np.abs(vectors), axis=1)
    return np.where(abs_max == 0.0, np.float32(1.0), abs_max).astype(np.float32)


def _pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack ``(n, d)`` uint8 codes (each in ``[0, 2**bits)``) into an LSB-first
    bit stream of shape ``(n, ceil(d * bits / 8))``."""
    n, d = codes.shape
    planes = np.unpackbits(codes[:, :, None], axis=2, count=bits, bitorder="little")
    flat = planes.reshape(n, d * bits)
    return np.packbits(flat, axis=1, bitorder="little")


def _unpack_codes(packed: np.ndarray, dim: int, bits: int) -> np.ndarray:
    """Inverse of :func:`_pack_codes`; returns ``(n, dim)`` uint8 codes."""
    n = packed.shape[0]
    all_bits = np.unpackbits(packed, axis=1, bitorder="little")
    flat = all_bits[:, : dim * bits].reshape(n, dim, bits)
    weights = (1 << np.arange(bits)).astype(np.uint16)
    return (flat * weights).sum(axis=2).astype(np.uint8)


def _encode_scalar(vectors: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray]:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 1-D or 2-D float array")

    levels = 1 << bits
    scales = _abs_max_scales(vectors)
    inv = (np.float32(1.0) / scales)[:, None]
    normalized = np.clip(vectors * inv, -1.0, 1.0).astype(np.float32)
    codes = np.floor((normalized + np.float32(1.0)) * np.float32(levels / 2)).astype(np.int32)
    codes = np.clip(codes, 0, levels - 1).astype(np.uint8)
    return _pack_codes(codes, bits), scales


def _decode_scalar(packed: np.ndarray, scales: np.ndarray, dim: int, bits: int) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    if packed.ndim == 1:
        packed = packed[np.newaxis]
    scales = np.asarray(scales, dtype=np.float32).reshape(-1, 1)
    levels = 1 << bits
    codes = _unpack_codes(packed, dim, bits).astype(np.float32)
    values = scales * (2.0 * codes - (levels - 1)) / np.float32(levels)
    return values.astype(np.float32)


# ── public API ──────────────────────────────────────────────────────────────


def quantize_sq2(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode float32 vectors to 2-bit SQ.

    Returns ``(packed, scales)`` — ``packed`` is ``(n, ceil(d/4))`` uint8 and
    ``scales`` is ``(n,)`` float32 abs-max.
    """
    return _encode_scalar(vectors, 2)


def dequantize_sq2(packed: np.ndarray, scales: np.ndarray, dim: int) -> np.ndarray:
    """Decode 2-bit SQ codes back to ``(n, dim)`` float32."""
    return _decode_scalar(packed, scales, dim, 2)


def quantize_sq3(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode float32 vectors to 3-bit SQ.

    Returns ``(packed, scales)`` — ``packed`` is ``(n, ceil(d*3/8))`` uint8 and
    ``scales`` is ``(n,)`` float32 abs-max.
    """
    return _encode_scalar(vectors, 3)


def dequantize_sq3(packed: np.ndarray, scales: np.ndarray, dim: int) -> np.ndarray:
    """Decode 3-bit SQ codes back to ``(n, dim)`` float32."""
    return _decode_scalar(packed, scales, dim, 3)


_BITS_BY_MODE: Dict[str, int] = {"sq2": 2, "sq3": 3}


def scalar_quant_compression_ratio(dim: int, mode: str) -> float:
    """Theoretical compression ratio of SQ vs FP32 (including the 4-byte scale)."""
    bits = _BITS_BY_MODE[mode]
    comp = (dim * bits + 7) // 8 + 4  # packed codes + one f32 scale
    return (dim * 4) / comp


def scalar_quant_stats(vectors: np.ndarray, mode: str) -> Dict[str, float]:
    """Round-trip quality report for an SQ mode (the ``quantization_stats`` analog).

    Returns a dict with ``mode``, ``bits``, ``n``, ``dim``, ``compression_ratio``,
    and per-vector cosine-fidelity aggregates (``mean_cosine`` / ``min_cosine``).
    """
    if mode not in _BITS_BY_MODE:
        raise ValueError(f"unknown scalar-quant mode {mode!r}; expected 'sq2' or 'sq3'")
    bits = _BITS_BY_MODE[mode]

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    n, dim = vectors.shape

    packed, scales = _encode_scalar(vectors, bits)
    recon = _decode_scalar(packed, scales, dim, bits)

    dots = np.sum(vectors * recon, axis=1)
    na = np.linalg.norm(vectors, axis=1)
    nb = np.linalg.norm(recon, axis=1)
    denom = na * nb
    cos = np.where(denom == 0.0, 1.0, dots / np.where(denom == 0.0, 1.0, denom))

    orig_bytes = int(vectors.size * 4)
    comp_bytes = int(packed.nbytes + scales.nbytes)
    return {
        "mode": mode,
        "bits": float(bits),
        "n": float(n),
        "dim": float(dim),
        "compression_ratio": float(orig_bytes) / float(comp_bytes) if comp_bytes else 0.0,
        "mean_cosine": float(np.mean(cos)),
        "min_cosine": float(np.min(cos)),
    }
