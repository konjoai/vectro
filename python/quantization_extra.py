"""INT2 quantization and adaptive / outlier-aware scaling for Vectro.

INT2 (2-bit integer) quantization provides maximum compression (8× vs float32
for quantized data) at the cost of reduced precision.  It is best suited for
bulk storage of approximate nearest-neighbour indexes where exact reconstruction
is not required.

Adaptive scaling uses per-channel statistics (mean-absolute-deviation clipping)
to reduce quantization error for vectors with statistical outliers.

All functions operate on NumPy arrays and are backend-agnostic.

Public API
----------
quantize_int2(embeddings, group_size=32)  → (packed, scales, zeroes)
dequantize_int2(packed, scales, zeroes, group_size=32, vector_dim=None) → float32

quantize_adaptive(embeddings, bits=8, clip_ratio=3.0, group_size=0) → QuantizationResult
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .interface import QuantizationResult


# ---------------------------------------------------------------------------
# INT2
# ---------------------------------------------------------------------------

_INT2_LEVELS = 3  # [-1, 0, +1] symmetric ternary (best quality in 2 bits)
_INT2_MAX = 1  # clamped range


def quantize_int2(
    embeddings: np.ndarray,
    group_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize float32 embeddings to 2-bit integer representation.

    Uses symmetric ternary quantization (values mapped to {-1, 0, +1}).
    Four INT2 values are packed into each byte of the output.

    Args:
        embeddings: Float32 array of shape ``(n, d)``.
        group_size: Number of elements per quantization group.  Controls
            the granularity of scale factors — smaller → better accuracy.

    Returns:
        A 3-tuple ``(packed, scales, zeroes)``:

        * ``packed``  — ``uint8`` array of shape ``(n, ceil(d/4))``; each
          byte holds 4 × 2-bit values.
        * ``scales``  — ``float32`` array of shape ``(n, n_groups)``; per-group
          scale factors.
        * ``zeroes``  — ``float32`` array of shape ``(n, n_groups)``; per-group
          zero-points (mean, for asymmetric distributions).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2-D array of shape (n, d)")

    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    n, d = emb.shape
    group_size = min(group_size, d)
    n_groups = int(np.ceil(d / group_size))

    scales = np.zeros((n, n_groups), dtype=np.float32)
    zeroes = np.zeros((n, n_groups), dtype=np.float32)
    q_flat = np.zeros((n, d), dtype=np.int8)

    for g in range(n_groups):
        col_start = g * group_size
        col_end = min(col_start + group_size, d)
        block = emb[:, col_start:col_end]  # (n, gs)

        zero = block.mean(axis=1, keepdims=True)  # (n, 1)
        shifted = block - zero
        abs_max = np.max(np.abs(shifted), axis=1, keepdims=True)
        scale = np.where(abs_max == 0, 1.0, abs_max).astype(np.float32)

        q_block = np.round(shifted / scale).astype(np.int8)
        q_block = np.clip(q_block, -1, 1)  # ternary: {-1, 0, +1}

        q_flat[:, col_start:col_end] = q_block
        scales[:, g] = scale.ravel()
        zeroes[:, g] = zero.ravel()

    # Pack 4 ternary values per byte: map {-1→0, 0→1, 1→2}, store in 2 bits
    # Mapping: -1 → 00 (0), 0 → 01 (1), 1 → 10 (2)
    q_ternary = (q_flat + 1).astype(np.uint8)  # {0, 1, 2}
    packed = _pack_int2(q_ternary, d)

    return packed, scales, zeroes


def _pack_int2(q: np.ndarray, d: int) -> np.ndarray:
    """Pack a (n, d) uint8 array of values in [0,3] into (n, ceil(d/4)) uint8.

    Uses contiguous column stacks instead of strided slices to avoid cache
    misses caused by non-unit strides on large embedding matrices.
    """
    n = q.shape[0]
    pad = (-d) % 4
    if pad:
        # Pad with 1 (the zero-point mapping for ternary {-1→0, 0→1, 1→2})
        q = np.pad(q, ((0, 0), (0, pad)), mode="constant", constant_values=1)
    d_padded = q.shape[1]
    n_bytes = d_padded // 4

    # Reshape to (n, n_bytes, 4) — each group of 4 columns becomes a last dim.
    # This produces a contiguous layout so the bitwise ops below hit L1 cache.
    q_r = np.ascontiguousarray(q.reshape(n, n_bytes, 4), dtype=np.uint8)
    out = (
        (q_r[:, :, 0] & 0x3)
        | ((q_r[:, :, 1] & 0x3) << 2)
        | ((q_r[:, :, 2] & 0x3) << 4)
        | ((q_r[:, :, 3] & 0x3) << 6)
    )
    return out.astype(np.uint8)


def _unpack_int2(packed: np.ndarray, n_elements: int) -> np.ndarray:
    """Unpack (n, ceil(d/4)) uint8 into (n, n_elements) uint8 {0,1,2}.

    Contiguous reshape avoids the strided i::4 pattern that causes cache misses.
    """
    n, n_bytes = packed.shape
    # Decode all 4 slots per byte at once → (n, n_bytes, 4)
    slots = np.stack(
        [
            packed & 0x3,
            (packed >> 2) & 0x3,
            (packed >> 4) & 0x3,
            (packed >> 6) & 0x3,
        ],
        axis=2,
    )  # (n, n_bytes, 4)
    out = np.ascontiguousarray(slots.reshape(n, n_bytes * 4), dtype=np.uint8)
    return out[:, :n_elements]


def dequantize_int2(
    packed: np.ndarray,
    scales: np.ndarray,
    zeroes: np.ndarray,
    group_size: int = 32,
    vector_dim: Optional[int] = None,
) -> np.ndarray:
    """Reconstruct float32 embeddings from INT2 representation.

    Args:
        packed: ``uint8`` array of shape ``(n, ceil(d/4))`` from
            :func:`quantize_int2`.
        scales: ``float32`` array of shape ``(n, n_groups)``.
        zeroes: ``float32`` array of shape ``(n, n_groups)``.
        group_size: Must match the value used during quantization.
        vector_dim: Original vector dimension ``d`` (inferred from packed
            shape when *None*).

    Returns:
        ``float32`` array of shape ``(n, d)``.
    """
    if vector_dim is None:
        vector_dim = packed.shape[1] * 4

    q_ternary = _unpack_int2(packed, vector_dim)
    q_signed = q_ternary.astype(np.int8) - 1  # {0,1,2} → {-1,0,+1}

    n = q_signed.shape[0]
    n_groups = scales.shape[1]
    group_size = int(np.ceil(vector_dim / n_groups))

    out = np.zeros((n, vector_dim), dtype=np.float32)
    for g in range(n_groups):
        col_start = g * group_size
        col_end = min(col_start + group_size, vector_dim)
        s = scales[:, g : g + 1].astype(np.float32)
        z = zeroes[:, g : g + 1].astype(np.float32)
        out[:, col_start:col_end] = q_signed[:, col_start:col_end].astype(np.float32) * s + z

    return out


# ---------------------------------------------------------------------------
# Adaptive scaling (outlier-aware INT8)
# ---------------------------------------------------------------------------

from typing import Optional  # noqa: E402 — after INT2 block for readability


def quantize_adaptive(
    embeddings: np.ndarray,
    bits: int = 8,
    clip_ratio: float = 3.0,
    group_size: int = 0,
) -> QuantizationResult:
    """Outlier-aware INT8 quantization with MAD-based clipping.

    Standard min-max scaling is sensitive to outliers: one extreme value can
    force all other values into a tiny part of the quantization range.
    *Adaptive* scaling clips values to ``clip_ratio × MAD`` (Median Absolute
    Deviation) before computing scales, drastically reducing quantization
    error on embedding models that produce heavy-tailed distributions.

    Args:
        embeddings: Float32 array of shape ``(n, d)``.
        bits: Quantization bit width.  Only ``8`` is currently supported;
            this parameter is reserved for future INT4 adaptive support.
        clip_ratio: How many MADs above the median to clip.  A value of
            ``3.0`` clips approximately the top/bottom 0.3% of a normal
            distribution before quantization.
        group_size: If > 0, use per-group (sub-vector) scales.

    Returns:
        :class:`~vectro.interface.QuantizationResult` with ``precision_mode="int8"``
        and :attr:`~vectro.interface.QuantizationResult.group_size` set to the
        effective value used.
    """
    if bits != 8:
        raise ValueError("quantize_adaptive currently supports bits=8 only")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-D (n, d)")

    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    n, d = emb.shape
    effective_gs = group_size if (group_size > 0 and group_size < d) else d

    n_groups = int(np.ceil(d / effective_gs))
    scales = np.zeros((n, n_groups), dtype=np.float32)
    q_full = np.zeros((n, d), dtype=np.int8)
    _max_val = float(2 ** (bits - 1) - 1)  # 127 for INT8

    for g in range(n_groups):
        col_start = g * effective_gs
        col_end = min(col_start + effective_gs, d)
        block = emb[:, col_start:col_end].copy()  # (n, gs)

        # Per-row MAD clipping
        med = np.median(block, axis=1, keepdims=True)
        mad = np.median(np.abs(block - med), axis=1, keepdims=True)
        clip_val = np.where(
            mad == 0, np.max(np.abs(block), axis=1, keepdims=True), clip_ratio * mad
        )
        block = np.clip(block, -clip_val, clip_val)

        abs_max = np.max(np.abs(block), axis=1, keepdims=True)
        scale = np.where(abs_max == 0, 1.0, abs_max / _max_val).astype(np.float32)

        q_block = np.clip(np.round(block / scale), -_max_val, _max_val).astype(np.int8)
        q_full[:, col_start:col_end] = q_block
        scales[:, g] = scale.ravel()

    if n_groups == 1:
        scales_out = scales.ravel()  # (n,) for per-row
    else:
        scales_out = scales  # (n, n_groups)

    return QuantizationResult(
        quantized=q_full,
        scales=scales_out,
        dims=d,
        n=n,
        precision_mode="int8",
        group_size=effective_gs,
    )
