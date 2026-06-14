"""lora_api.py — LoRA adapter matrix compression for Vectro v3.

Compresses Low-Rank Adaptation (LoRA) A and B matrices using Vectro's
existing quantization backends (NF4, INT8, RQ).  LoRA matrices are treated
as batches of float32 row vectors, which maps directly onto the existing
per-vector quantization path.

Typical LoRA shapes
-------------------
    A : (rank, in_features)   e.g. (16, 768)
    B : (out_features, rank)  e.g. (768, 16)

Profiles
--------
"lora-nf4"   — NF4 4-bit per row, 8× compression, cosine ≥ 0.97
"lora-int8"  — INT8 per row, 4× compression, cosine ≥ 0.99
"lora-rq"    — Residual Quantizer 3-pass, 16-32× compression;
               auto-falls back to "lora-nf4" when n_rows < 32

Usage
-----
>>> A = np.random.randn(16, 768).astype(np.float32)
>>> B = np.random.randn(768, 16).astype(np.float32)
>>> result = compress_lora(A, B, profile="lora-nf4", target_module="q_proj")
>>> A_r, B_r = decompress_lora(result)

>>> adapter = {"q_proj": (A_q, B_q), "v_proj": (A_v, B_v)}
>>> compressed = compress_lora_adapter(adapter, profile="lora-nf4")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .nf4_api import quantize_nf4, dequantize_nf4, nf4_cosine_sim
from .interface import quantize_embeddings, reconstruct_embeddings, mean_cosine_similarity
from .rq_api import ResidualQuantizer

_VALID_LORA_PROFILES = frozenset({"lora-nf4", "lora-int8", "lora-rq"})

# Minimum number of rows required for a stable RQ codebook fit.
# Below this, RQ K-means may degenerate; fall back to NF4.
_RQ_MIN_ROWS = 32


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class LoRAResult:
    """Compressed LoRA adapter matrices for a single target module.

    Attributes
    ----------
    profile       : one of "lora-nf4", "lora-int8", "lora-rq"
    rank          : LoRA rank r (A.shape[0] == B.shape[1])
    target_module : name of the target module, e.g. "q_proj"
    A_data        : compressed A matrix payload (profile-dependent keys)
    B_data        : compressed B matrix payload (profile-dependent keys)
    A_shape       : original shape of A, (rank, in_features)
    B_shape       : original shape of B, (out_features, rank)
    cosine_sim_A  : per-row mean cosine similarity of A reconstruction
    cosine_sim_B  : per-row mean cosine similarity of B reconstruction
    """

    profile: str
    rank: int
    target_module: str
    A_data: Dict[str, Any]
    B_data: Dict[str, Any]
    A_shape: Tuple[int, int]
    B_shape: Tuple[int, int]
    cosine_sim_A: float = 0.0
    cosine_sim_B: float = 0.0

    def __repr__(self) -> str:
        return (
            f"LoRAResult(profile={self.profile!r}, rank={self.rank}, "
            f"module={self.target_module!r}, "
            f"A={self.A_shape}, B={self.B_shape}, "
            f"cos_A={self.cosine_sim_A:.4f}, cos_B={self.cosine_sim_B:.4f})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_lora_pair(A: np.ndarray, B: np.ndarray) -> None:
    """Validate shape, dtype, and rank consistency of an A/B pair."""
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    if B.ndim != 2:
        raise ValueError(f"B must be 2-D, got shape {B.shape}")
    if A.dtype != np.float32:
        raise ValueError(f"A must be float32, got {A.dtype}")
    if B.dtype != np.float32:
        raise ValueError(f"B must be float32, got {B.dtype}")
    if A.shape[0] != B.shape[1]:
        raise ValueError(f"Rank mismatch: A.shape[0]={A.shape[0]} != B.shape[1]={B.shape[1]}")


def _compress_matrix_nf4(mat: np.ndarray) -> Dict[str, Any]:
    """Compress a float32 matrix row-wise with NF4. Returns payload dict.

    Falls back to the inline NumPy implementation when the Mojo binary is
    present but fails at runtime (e.g. subprocess missing numpy).
    """
    try:
        packed, scales = quantize_nf4(mat)
    except (RuntimeError, OSError):
        # Mojo binary reported available but failed — use NumPy path directly.
        from .nf4_api import _NF4_THRESHOLDS

        n, d = mat.shape
        scales_2d = np.abs(mat).max(axis=1, keepdims=True)
        scales = scales_2d.ravel().copy()
        safe = np.where(scales_2d == 0.0, 1.0, scales_2d)
        normed = mat / safe
        indices = np.searchsorted(_NF4_THRESHOLDS, normed.ravel()).astype(np.uint8).reshape(n, d)
        bpv = (d + 1) // 2
        packed = np.zeros((n, bpv), dtype=np.uint8)
        packed[:, : d // 2] = (indices[:, 1::2][:, : d // 2] << 4) | indices[:, 0::2][:, : d // 2]
        if d % 2 == 1:
            packed[:, d // 2] = indices[:, d - 1]
    return {"packed": packed, "scales": scales, "d": mat.shape[1]}


def _decompress_matrix_nf4(data: Dict[str, Any]) -> np.ndarray:
    try:
        return dequantize_nf4(data["packed"], data["scales"], data["d"])
    except (RuntimeError, OSError):
        from .nf4_api import NF4_LEVELS

        packed, scales, d = data["packed"], data["scales"], data["d"]
        n = packed.shape[0]
        out = np.empty((n, d), dtype=np.float32)
        d_even = (d // 2) * 2
        if d_even > 0:
            lo = (packed[:, : d_even // 2] & 0x0F).astype(np.int32)
            hi = ((packed[:, : d_even // 2] >> 4) & 0x0F).astype(np.int32)
            out[:, 0::2][:, : d_even // 2] = NF4_LEVELS[lo]
            out[:, 1::2][:, : d_even // 2] = NF4_LEVELS[hi]
        if d % 2 == 1:
            bpv = (d + 1) // 2
            lo_last = (packed[:, bpv - 1] & 0x0F).astype(np.int32)
            out[:, d - 1] = NF4_LEVELS[lo_last]
        return out * scales[:, np.newaxis]


def _compress_matrix_int8(mat: np.ndarray) -> Dict[str, Any]:
    """Compress a float32 matrix row-wise with INT8. Returns payload dict."""
    result = quantize_embeddings(mat)
    return {"quantized": result.quantized, "scales": result.scales, "d": result.dims}


def _decompress_matrix_int8(data: Dict[str, Any]) -> np.ndarray:
    from .interface import QuantizationResult

    qr = QuantizationResult(
        quantized=data["quantized"],
        scales=data["scales"],
        dims=data["d"],
        n=data["quantized"].shape[0],
    )
    return reconstruct_embeddings(qr)


def _compress_matrix_rq(
    mat: np.ndarray, rq: Optional[ResidualQuantizer] = None
) -> Tuple[Dict[str, Any], ResidualQuantizer]:
    """Compress a float32 matrix row-wise with RQ. Trains codebook if not provided."""
    if rq is None:
        rq = ResidualQuantizer(n_passes=3, n_subspaces=8, n_centroids=64)
        rq.train(mat)
    codes = rq.encode(mat)
    return {"codes": codes, "n_passes": rq.n_passes, "d": mat.shape[1]}, rq


def _decompress_matrix_rq(data: Dict[str, Any], rq: ResidualQuantizer) -> np.ndarray:
    return rq.decode(data["codes"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compress_lora(
    A: np.ndarray,
    B: np.ndarray,
    profile: str = "lora-nf4",
    target_module: str = "",
) -> LoRAResult:
    """Compress a LoRA adapter (A, B) matrix pair.

    Parameters
    ----------
    A             : float32 array of shape (rank, in_features)
    B             : float32 array of shape (out_features, rank)
    profile       : "lora-nf4" | "lora-int8" | "lora-rq"
    target_module : human-readable name for the target layer, e.g. "q_proj"

    Returns
    -------
    LoRAResult with compressed payloads and per-matrix reconstruction quality.

    Notes
    -----
    When profile="lora-rq" and rank < _RQ_MIN_ROWS, automatically falls back
    to "lora-nf4" with a UserWarning. RQ requires enough rows for stable
    K-means codebook fitting.
    """
    if profile not in _VALID_LORA_PROFILES:
        raise ValueError(f"Unknown profile {profile!r}. Valid: {sorted(_VALID_LORA_PROFILES)}")
    _validate_lora_pair(A, B)

    rank = A.shape[0]
    effective_profile = profile

    # RQ fallback: if either matrix has too few rows, drop to NF4
    if profile == "lora-rq" and (rank < _RQ_MIN_ROWS or B.shape[0] < _RQ_MIN_ROWS):
        warnings.warn(
            f"lora-rq requires >= {_RQ_MIN_ROWS} rows for stable codebook fitting; "
            f"A has {rank} rows, B has {B.shape[0]} rows. "
            "Falling back to lora-nf4.",
            UserWarning,
            stacklevel=2,
        )
        effective_profile = "lora-nf4"

    if effective_profile == "lora-nf4":
        A_data = _compress_matrix_nf4(A)
        B_data = _compress_matrix_nf4(B)
        A_recon = _decompress_matrix_nf4(A_data)
        B_recon = _decompress_matrix_nf4(B_data)
    elif effective_profile == "lora-int8":
        A_data = _compress_matrix_int8(A)
        B_data = _compress_matrix_int8(B)
        A_recon = _decompress_matrix_int8(A_data)
        B_recon = _decompress_matrix_int8(B_data)
    else:  # lora-rq (with enough rows)
        A_data, rq_A = _compress_matrix_rq(A)
        B_data, rq_B = _compress_matrix_rq(B)
        A_data["_rq"] = rq_A
        B_data["_rq"] = rq_B
        A_recon = _decompress_matrix_rq(A_data, rq_A)
        B_recon = _decompress_matrix_rq(B_data, rq_B)

    # Quality metrics — guard against degenerate single-row matrices
    if A.shape[0] > 1:
        cos_A = float(mean_cosine_similarity(A, A_recon))
    else:
        cos_A = (
            float(nf4_cosine_sim(A, A_recon))
            if effective_profile == "lora-nf4"
            else float(mean_cosine_similarity(A, A_recon))
        )
    if B.shape[0] > 1:
        cos_B = float(mean_cosine_similarity(B, B_recon))
    else:
        cos_B = (
            float(nf4_cosine_sim(B, B_recon))
            if effective_profile == "lora-nf4"
            else float(mean_cosine_similarity(B, B_recon))
        )

    return LoRAResult(
        profile=effective_profile,
        rank=rank,
        target_module=target_module,
        A_data=A_data,
        B_data=B_data,
        A_shape=A.shape,
        B_shape=B.shape,
        cosine_sim_A=cos_A,
        cosine_sim_B=cos_B,
    )


def decompress_lora(result: LoRAResult) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct LoRA (A, B) float32 matrices from a LoRAResult.

    Returns
    -------
    A_recon : float32 array of shape result.A_shape
    B_recon : float32 array of shape result.B_shape
    """
    if result.profile == "lora-nf4":
        A = _decompress_matrix_nf4(result.A_data)
        B = _decompress_matrix_nf4(result.B_data)
    elif result.profile == "lora-int8":
        A = _decompress_matrix_int8(result.A_data)
        B = _decompress_matrix_int8(result.B_data)
    elif result.profile == "lora-rq":
        rq_A = result.A_data["_rq"]
        rq_B = result.B_data["_rq"]
        A = _decompress_matrix_rq(result.A_data, rq_A)
        B = _decompress_matrix_rq(result.B_data, rq_B)
    else:
        raise ValueError(f"Cannot decompress unknown profile {result.profile!r}")

    return A.astype(np.float32), B.astype(np.float32)


def compress_lora_adapter(
    adapter: Dict[str, Tuple[np.ndarray, np.ndarray]],
    profile: str = "lora-nf4",
) -> Dict[str, LoRAResult]:
    """Compress a full LoRA adapter (all target modules).

    Parameters
    ----------
    adapter : dict mapping module name → (A, B) float32 matrix pair
    profile : compression profile, applied uniformly to all modules

    Returns
    -------
    dict mapping module name → LoRAResult

    Example
    -------
    >>> adapter = {
    ...     "q_proj": (A_q, B_q),
    ...     "v_proj": (A_v, B_v),
    ...     "k_proj": (A_k, B_k),
    ... }
    >>> compressed = compress_lora_adapter(adapter, profile="lora-nf4")
    >>> for name, result in compressed.items():
    ...     print(name, result)
    """
    return {
        module_name: compress_lora(A, B, profile=profile, target_module=module_name)
        for module_name, (A, B) in adapter.items()
    }
