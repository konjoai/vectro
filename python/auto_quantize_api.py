"""Auto-Quantize — Phase 7c Learned Quantization.

Automatically selects the best quantization mode for a batch of embeddings
by trying multiple strategies in order and returning the first one that
meets the user's quality and compression constraints.

Strategy order
--------------
1. NF4           (4-bit, good for Gaussian-ish distributions)
2. NF4-mixed     (NF4 per-channel with outlier-aware clipping)
3. PQ-96         (Product Quantizer, 96-byte codes, highest quality)
4. PQ-48         (Product Quantizer, 48-byte codes, balanced)
5. Binary        (1-bit sign, maximum compression)

Routing heuristic
-----------------
Uses ``scipy.stats.kurtosis`` to detect heavy-tailed distributions.
High-kurtosis vectors are routed first to NF4-mixed (outlier-aware) before
the generic sequence.

Public API
----------
auto_quantize(embeddings, target_cosine, target_compression) -> dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

try:
    from scipy.stats import kurtosis as _scipy_kurtosis  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_sim_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-vector cosine similarity between a and b."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    dots = (a * b).sum(axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    return float((dots / norms).mean())


def _compute_kurtosis(embeddings: np.ndarray) -> float:
    """Return mean per-dimension excess kurtosis.

    Uses scipy if available, otherwise a fast pure-numpy estimator.
    """
    if _HAS_SCIPY:
        # Per-dimension kurtosis, then average
        return float(_scipy_kurtosis(embeddings, axis=0, fisher=True).mean())
    # Fallback: numpy excess kurtosis = μ₄/σ⁴ − 3
    m = embeddings.mean(axis=0)
    s = embeddings.std(axis=0) + 1e-8
    z = (embeddings - m) / s
    return float((z ** 4).mean(axis=0).mean() - 3.0)


def _compression_ratio_nf4(d: int) -> float:
    """NF4 stores 2 values per byte → 4-bit → ratio = 8×."""
    return 8.0


def _compression_ratio_pq(d: int, code_bytes: int) -> float:
    """PQ code_bytes codes per vector; float32 = d×4 bytes."""
    return (d * 4) / max(code_bytes, 1)


def _compression_ratio_binary(d: int) -> float:
    """Binary: bit-packing → d/8 bytes per vector."""
    return (d * 4) / max(d // 8, 1)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _try_nf4(embeddings: np.ndarray, mixed: bool = False) -> dict:
    """Attempt NF4 (or NF4-mixed) quantization.

    Parameters
    ----------
    embeddings : (n, d) float32
    mixed      : if True, uses outlier-aware mixed NF4 quantization

    Returns
    -------
    dict with keys: success, mode, cosine_sim, compression_ratio, result
    """
    mode_name = "nf4_mixed" if mixed else "nf4"
    try:
        from .nf4_api import (  # type: ignore[import]
            quantize_nf4,
            dequantize_nf4,
            quantize_mixed,
            dequantize_mixed,
            select_outlier_dims,
        )
    except ImportError:
        return {"success": False, "mode": mode_name, "error": "nf4_api not available"}

    d = embeddings.shape[1]
    try:
        if mixed:
            outlier_dims = select_outlier_dims(embeddings)
            result = quantize_mixed(embeddings, outlier_dims)
            fp16_vals, nf4_packed, nf4_scales, out_dims = result
            recon = dequantize_mixed(fp16_vals, nf4_packed, nf4_scales, out_dims, d)
        else:
            packed, scales = quantize_nf4(embeddings)
            result = (packed, scales)
            recon = dequantize_nf4(packed, scales, d)

        cosine = _cosine_sim_mean(embeddings, recon)
        ratio = _compression_ratio_nf4(d)
        return {
            "success": True,
            "mode": mode_name,
            "cosine_sim": cosine,
            "compression_ratio": ratio,
            "result": result,
        }
    except Exception as exc:
        return {"success": False, "mode": mode_name, "error": str(exc)}


def _try_pq(
    embeddings: np.ndarray,
    target_code_bytes: int,
) -> dict:
    """Attempt Product Quantizer at a given code size.

    Parameters
    ----------
    embeddings       : (n, d) float32
    target_code_bytes: desired code length in bytes (= n_subspaces, K=256)
    """
    try:
        from .pq_api import train_pq_codebook, pq_encode, pq_decode  # type: ignore[import]
    except ImportError:
        return {
            "success": False,
            "mode": f"pq_{target_code_bytes}",
            "error": "pq_api not available",
        }

    d = embeddings.shape[1]
    # n_subspaces = number of bytes per code (K=256 → uint8 per sub-space)
    M = min(target_code_bytes, d)
    while M > 1 and d % M != 0:
        M -= 1

    mode_name = f"pq_{target_code_bytes}"
    try:
        cb = train_pq_codebook(embeddings, n_subspaces=M, n_centroids=256)
        codes = pq_encode(embeddings, cb)
        recon = pq_decode(codes, cb)
        cosine = _cosine_sim_mean(embeddings, recon)
        ratio = _compression_ratio_pq(d, M)
        return {
            "success": True,
            "mode": mode_name,
            "cosine_sim": cosine,
            "compression_ratio": ratio,
            "result": {"codebook": cb, "codes": codes},
        }
    except Exception as exc:
        return {"success": False, "mode": mode_name, "error": str(exc)}


def _try_binary(embeddings: np.ndarray) -> dict:
    """Attempt binary sign quantization."""
    try:
        from .binary_api import quantize_binary, dequantize_binary  # type: ignore[import]
    except ImportError:
        return {"success": False, "mode": "binary", "error": "binary_api not available"}

    d = embeddings.shape[1]
    try:
        packed = quantize_binary(embeddings)       # (n, ceil(d/8)) uint8
        recon = dequantize_binary(packed, d)       # (n, d) float32
        cosine = _cosine_sim_mean(embeddings, recon)
        ratio = _compression_ratio_binary(d)
        return {
            "success": True,
            "mode": "binary",
            "cosine_sim": cosine,
            "compression_ratio": ratio,
            "result": {"packed": packed},
        }
    except Exception as exc:
        return {"success": False, "mode": "binary", "error": str(exc)}


# ---------------------------------------------------------------------------
# Fallback: simple INT8 quantizer (always available, no dependencies)
# ---------------------------------------------------------------------------

def _try_int8_fallback(embeddings: np.ndarray) -> dict:
    """Simple per-vector abs-max INT8 quantization as a last resort."""
    d = embeddings.shape[1]
    abs_max = np.abs(embeddings).max(axis=1)
    scales = np.where(abs_max > 0, abs_max / 127.0, 1.0)[:, np.newaxis]
    q = np.clip(np.round(embeddings / scales), -127, 127).astype(np.int8)
    recon = q.astype(np.float32) * scales
    cosine = _cosine_sim_mean(embeddings, recon)
    ratio = (d * 4) / d  # float32 → int8 = 4×
    return {
        "success": True,
        "mode": "int8_fallback",
        "cosine_sim": cosine,
        "compression_ratio": ratio,
        "result": {"quantized": q, "scales": scales.squeeze()},
    }


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def auto_quantize(
    embeddings: np.ndarray,
    target_cosine: float = 0.97,
    target_compression: float = 8.0,
    model_dir: "Union[str, Path, None]" = None,
) -> dict:
    """Select and apply the best quantization strategy for ``embeddings``.

    When *model_dir* is supplied, the function reads ``config.json`` from that
    directory and uses the model-family registry (:func:`python.profiles.get_profile`)
    to bypass the statistical heuristic with a deterministic method choice.
    This is faster and more accurate for known embedding model families.

    Tries strategies in order (highest quality first) and returns the first
    result that satisfies both ``target_cosine`` and ``target_compression``.
    If none meet both constraints, returns the strategy with the highest
    cosine similarity that still achieves the compression target; failing
    that, the best overall on cosine.

    Parameters
    ----------
    embeddings         : np.ndarray, shape (n, d), float32
    target_cosine      : float, minimum acceptable mean cosine similarity
    target_compression : float, minimum acceptable compression ratio (× float32)
    model_dir          : path to a HuggingFace model directory (optional).
                         When provided, the family registry overrides the
                         statistical heuristic for known families.

    Returns
    -------
    dict with keys:
        mode              : str   — chosen strategy name
        cosine_sim        : float — achieved mean cosine similarity
        compression_ratio : float — achieved compression ratio
        result            : Any   — quantizer-specific output
        kurtosis          : float — input distribution kurtosis (0.0 when skipped)
        tried             : list  — all strategies attempted (with outcomes)
        family            : str   — detected model family (present when model_dir given)
    """
    data = np.ascontiguousarray(embeddings, dtype=np.float32)

    # ── Fast path: model-family registry ──────────────────────────────────────────────────────────────────────────────
    if model_dir is not None:
        from .profiles import get_profile
        profile = get_profile(model_dir)
        if profile.method == "int8":
            result = _try_int8_fallback(data)
            result.update({"kurtosis": 0.0, "tried": [result], "family": profile.family})
            return result
        if profile.method == "nf4":
            result = _try_nf4(data, mixed=False)
            if not result.get("success"):
                result = _try_int8_fallback(data)
            result.update({"kurtosis": 0.0, "tried": [result], "family": profile.family})
            return result
        # profile.method == "auto" → fall through to statistical heuristic

    # ── Statistical heuristic ──────────────────────────────────────────────────────────────────────────────────────
    n, d = data.shape
    kurt = _compute_kurtosis(data)
    heavy_tailed = kurt > 1.5      # excess kurtosis threshold (Laplace ≈ 3, Gaussian ≈ 0)

    # Build candidate order
    # Heavy-tailed distributions benefit from NF4-mixed first
    if heavy_tailed:
        candidates = [
            lambda e=data: _try_nf4(e, mixed=True),
            lambda e=data: _try_nf4(e, mixed=False),
            lambda e=data: _try_pq(e, target_code_bytes=min(96, d)),
            lambda e=data: _try_pq(e, target_code_bytes=min(48, d)),
            lambda e=data: _try_binary(e),
            lambda e=data: _try_int8_fallback(e),
        ]
    else:
        candidates = [
            lambda e=data: _try_nf4(e, mixed=False),
            lambda e=data: _try_nf4(e, mixed=True),
            lambda e=data: _try_pq(e, target_code_bytes=min(96, d)),
            lambda e=data: _try_pq(e, target_code_bytes=min(48, d)),
            lambda e=data: _try_binary(e),
            lambda e=data: _try_int8_fallback(e),
        ]

    tried = []
    best_compression_ok: Optional[dict] = None   # meets compression, best cosine
    best_overall: Optional[dict] = None           # absolute best cosine

    for fn in candidates:
        r = fn()
        tried.append({
            "mode": r.get("mode"),
            "success": r.get("success"),
            "cosine_sim": r.get("cosine_sim"),
            "compression_ratio": r.get("compression_ratio"),
        })

        if not r.get("success"):
            continue

        cosine = r.get("cosine_sim", 0.0)
        ratio  = r.get("compression_ratio", 0.0)

        # Track best overall
        if best_overall is None or cosine > best_overall.get("cosine_sim", 0.0):
            best_overall = r

        # Check if this meets the target constraints
        if cosine >= target_cosine and ratio >= target_compression:
            # First strategy to satisfy both — return immediately
            return {
                "mode": r["mode"],
                "cosine_sim": cosine,
                "compression_ratio": ratio,
                "result": r.get("result"),
                "kurtosis": kurt,
                "tried": tried,
            }

        # Meets compression but not cosine (save for fallback)
        if ratio >= target_compression:
            if (
                best_compression_ok is None
                or cosine > best_compression_ok.get("cosine_sim", 0.0)
            ):
                best_compression_ok = r

    # No strategy met both constraints.
    # Prefer: meets compression (best cosine) > best overall cosine
    chosen = best_compression_ok or best_overall or tried[-1]
    # If chosen is a "tried" dict (no "result" key), reconstruct from int8 fallback
    if "result" not in chosen:
        chosen = _try_int8_fallback(data)

    return {
        "mode": chosen.get("mode", "int8_fallback"),
        "cosine_sim": chosen.get("cosine_sim", 0.0),
        "compression_ratio": chosen.get("compression_ratio", 0.0),
        "result": chosen.get("result"),
        "kurtosis": kurt,
        "tried": tried,
    }
