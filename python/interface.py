"""
Python fallback quantizer for Vectro MVP.
This provides a simple per-vector int8 quantizer and reconstruction.
API:
 - quantize_embeddings(embeddings: np.ndarray) -> QuantizationResult
 - reconstruct_embeddings(result: QuantizationResult) -> np.ndarray
 - mean_cosine_similarity(orig, recon) -> float
"""
from __future__ import annotations
import importlib
import numpy as np
import os
import subprocess
import tempfile
from typing import NamedTuple

from . import _mojo_bridge


class QuantizationResult(NamedTuple):
    """Result of quantization operation."""
    quantized: np.ndarray  # int8 array, shape (n, d)
    scales: np.ndarray     # float32 array, shape (n,)
    dims: int             # dimension d
    n: int                # number of vectors
    precision_mode: str = "int8"  # int8 | int4
    group_size: int = 0


# Try to import the high-performance backends if available
_cython_quant = None
try:
    # Try relative import first (inside the package)
    from . import quantizer_cython as _cython_quant  # type: ignore
except (ImportError, Exception):
    try:
        # Fallback for development/local install
        _cython_quant = importlib.import_module("quantizer_cython")
    except Exception:
        _cython_quant = None

# Rust-based squish_quant backend (highest throughput — built via maturin)
_squish_quant = None
try:
    import squish_quant as _squish_quant
except ImportError:
    _squish_quant = None

# vectro_py PyO3 batch backend — zero-copy INT8 batch quantize, ≥1 M vec/s
_vectro_py = None
try:
    import vectro_py as _vectro_py
except ImportError:
    _vectro_py = None

_mojo_available: bool = _mojo_bridge.is_available()
_mojo_binary: str | None = _mojo_bridge._binary_path


def _quantize_with_squish(embeddings: np.ndarray, group_size: int = 0) -> QuantizationResult:
    """Quantize using the Rust squish_quant extension (6+ GB/s on Apple Silicon).

    group_size=0  → per-row quantization (1 scale per row)
    group_size=N  → per-group-N quantization (higher accuracy, more scales)
    """
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    n, d = emb.shape

    if group_size <= 0 or group_size >= d:
        q, scales = _squish_quant.quantize_int8_f32(emb)       # scales: (n,)
    else:
        q, scales = _squish_quant.quantize_int8_grouped(emb, group_size)  # scales: (n, n_groups)

    return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)


def _dequantize_with_squish(result: QuantizationResult) -> np.ndarray:
    """Dequantize using the Rust squish_quant extension.

    Chooses grouped or per-row dequantize based on scales shape.
    Significantly faster than NumPy broadcast on large matrices.
    """
    q = np.ascontiguousarray(result.quantized, dtype=np.int8)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    s = np.asarray(result.scales, dtype=np.float32)
    if s.ndim == 0:
        s = s.reshape(1)

    if s.ndim == 1:
        return _squish_quant.dequantize_int8_f32(q, np.ascontiguousarray(s, dtype=np.float32))
    else:
        # scales is (n, n_groups) — derive group_size from shape
        group_size = result.dims // s.shape[1]
        return _squish_quant.dequantize_int8_grouped(q, np.ascontiguousarray(s, dtype=np.float32), group_size)


def _quantize_with_vectro_py(embeddings: np.ndarray) -> QuantizationResult:
    """Quantize using the vectro_py PyO3 SIMD-accelerated INT8 batch encoder.

    Calls ``vectro_py.quantize_int8_batch`` directly in-process — no subprocess
    spawn overhead (~45 ms eliminated).  Typical throughput ≥ 1 M vec/s on
    Apple Silicon (NEON) and x86-64 (AVX2).  Scales are stored as
    abs_max / 127.0, compatible with the numpy dequantise path.
    """
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    was_1d = emb.ndim == 1
    if was_1d:
        emb = emb.reshape(1, -1)
    n, d = emb.shape
    q, scales = _vectro_py.quantize_int8_batch(emb)
    if was_1d:
        q = q.reshape(-1)
        scales = scales.reshape(1)
    return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)


def _dequantize_with_vectro_py(result: QuantizationResult) -> np.ndarray:
    """Reconstruct float32 from INT8 using the vectro_py PyO3 batch dequantizer."""
    q = np.ascontiguousarray(result.quantized, dtype=np.int8)
    s = np.ascontiguousarray(result.scales, dtype=np.float32)
    # dequantize_int8_batch requires 2D codes [N, D]; a single stored vector
    # may arrive as 1D [D] — reshape transparently.
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if s.ndim == 0:
        s = s.reshape(1)
    return _vectro_py.dequantize_int8_batch(q, s)


def quantize_int4(
    embeddings: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """INT4 nibble-packed quantization — 50% disk vs INT8, requires squish_quant.

    Returns (packed_uint8, scales_float32) — packed has shape (n, d//2).
    Use dequantize_int4() to reconstruct.
    Requires squish_quant Rust extension (built with maturin).
    """
    if _squish_quant is None:
        raise RuntimeError("squish_quant Rust extension required for INT4.  Run: cd squish_quant_rs && python3 -m maturin build --release")
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    return _squish_quant.quantize_int4_grouped(emb, group_size)


def dequantize_int4(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 64,
) -> np.ndarray:
    """Reconstruct float32 from nibble-packed INT4 weights.

    packed: (n, d//2) uint8  — from quantize_int4()
    scales: (n, d//group_size) float32
    Returns: (n, d) float32
    """
    if _squish_quant is None:
        raise RuntimeError("squish_quant Rust extension required for INT4.")
    return _squish_quant.dequantize_int4_grouped(
        np.ascontiguousarray(packed, dtype=np.uint8),
        np.ascontiguousarray(scales, dtype=np.float32),
        group_size,
    )


def _quantize_vectorized(embeddings: np.ndarray, group_size: int = 0) -> QuantizationResult:
    """Fully vectorized per-row (or per-group) INT8 quantization.

    Replaces the original Python for-loop with numpy broadcast ops running in
    native C/BLAS — typically 20-50x faster for large weight matrices.

    group_size=0  → per-row quantization (1 scale per row, classic approach)
    group_size=N  → per-group-N quantization (better accuracy, more scales)
                    Each row is split into ceil(d/N) groups of ≤N elements.
    """
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    n, d = emb.shape

    if group_size <= 0 or group_size >= d:
        # ---- per-row -------------------------------------------------------
        row_max = np.max(np.abs(emb), axis=1)          # (n,)
        scales  = np.where(row_max == 0, 1.0, row_max / 127.0).astype(np.float32)
        q = np.clip(np.round(emb / scales[:, None]), -127, 127).astype(np.int8)
        return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)
    else:
        # ---- per-group -----------------------------------------------------
        # Pad columns to a multiple of group_size.
        pad = (-d) % group_size
        if pad:
            emb = np.pad(emb, ((0, 0), (0, pad)))
        n_groups = emb.shape[1] // group_size
        # Reshape to (n * n_groups, group_size) for vectorized scale computation.
        grouped = emb.reshape(n * n_groups, group_size)
        gmax   = np.max(np.abs(grouped), axis=1)       # (n*n_groups,)
        gscale = np.where(gmax == 0, 1.0, gmax / 127.0).astype(np.float32)
        q_groups = np.clip(
            np.round(grouped / gscale[:, None]), -127, 127
        ).astype(np.int8)
        # Trim padding, return flat (n, d) int8 + (n, n_groups) scales
        q = q_groups.reshape(n, -1)[:, :d]
        scales = gscale.reshape(n, n_groups).astype(np.float32)
        return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)


def _quantize_with_mojo(embeddings: np.ndarray) -> QuantizationResult:
    """Quantize using the compiled Mojo binary (INT8 symmetric abs-max)."""
    q, scales = _mojo_bridge.int8_quantize(embeddings)
    n, d = q.shape
    return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)


def _reconstruct_with_mojo(result: QuantizationResult) -> np.ndarray:
    """Reconstruct float32 from INT8 + scales using the compiled Mojo binary."""
    return _mojo_bridge.int8_reconstruct(result.quantized, result.scales, result.dims)


def get_backend_info():
    """Get information about available backends."""
    info = {
        "squish_quant_rust": _squish_quant is not None,
        "vectro_py": _vectro_py is not None,
        "mojo": _mojo_available,
        "cython": _cython_quant is not None,
        "numpy": True,  # Always available
    }
    if _mojo_available:
        info["mojo_binary"] = _mojo_binary
    return info


def quantize_embeddings(
    embeddings: np.ndarray,
    backend: str = "auto",
    precision_mode: str = "int8",
    group_size: int = 64,
) -> QuantizationResult:
    """Quantize embeddings to int8 with per-vector scale.

    embeddings: shape (n, d), dtype float32
    backend: 'auto', 'mojo', 'cython', or 'numpy'
    precision_mode: 'int8' or 'int4'
    group_size: group size used for grouped modes (currently int4)
    returns: QuantizationResult with quantized int8 array, scales, dims, n
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array of shape (n, d)")
    n, d = embeddings.shape

    precision_mode = precision_mode.lower()
    if precision_mode not in ("int8", "int4"):
        raise ValueError(f"Unsupported precision_mode: {precision_mode}")

    if precision_mode == "int4":
        packed, scales = quantize_int4(embeddings, group_size=group_size)
        return QuantizationResult(
            quantized=packed,
            scales=scales,
            dims=d,
            n=n,
            precision_mode="int4",
            group_size=group_size,
        )

    # Backend selection
    if backend == "auto":
        # Priority: squish_quant (Rust) > vectro_py (Rust/PyO3) > Mojo > Cython > NumPy
        if _squish_quant is not None:
            backend = "squish_quant"
        elif _vectro_py is not None:
            backend = "vectro_py"
        elif _mojo_available:
            backend = "mojo"
        elif _cython_quant is not None:
            backend = "cython"
        else:
            backend = "numpy"
    
    # Use selected backend
    if backend == "squish_quant" and _squish_quant is not None:
        return _quantize_with_squish(embeddings)
    elif backend == "vectro_py" and _vectro_py is not None:
        return _quantize_with_vectro_py(embeddings)
    elif backend == "mojo" and _mojo_available:
        return _quantize_with_mojo(embeddings)
    elif backend == "cython" and _cython_quant is not None:
        quantized, scales = _cython_quant.quantize_embeddings_cython(embeddings.astype(np.float32))
        return QuantizationResult(quantized=quantized, scales=scales, dims=d, n=n)
    elif backend == "numpy" or backend == "auto":
        return _quantize_vectorized(embeddings)
    else:
        raise ValueError(f"Unknown or unavailable backend: {backend}")


def reconstruct_embeddings(result: QuantizationResult, backend: str = "auto") -> np.ndarray:
    """Reconstruct embeddings from QuantizationResult.

    backend: 'auto', 'squish_quant', 'mojo', 'cython', or 'numpy'
    'auto' priority: squish_quant (Rust) > Cython > NumPy
    """
    if getattr(result, "precision_mode", "int8") == "int4":
        return dequantize_int4(
            result.quantized,
            result.scales,
            group_size=getattr(result, "group_size", 64) or 64,
        )

    # Backend selection
    if backend == "auto":
        if _squish_quant is not None:
            backend = "squish_quant"
        elif _vectro_py is not None:
            backend = "vectro_py"
        elif _mojo_available:
            backend = "mojo"
        elif _cython_quant is not None:
            backend = "cython"
        else:
            backend = "numpy"
    
    # Use selected backend
    if backend == "squish_quant" and _squish_quant is not None:
        return _dequantize_with_squish(result)
    elif backend == "vectro_py" and _vectro_py is not None:
        return _dequantize_with_vectro_py(result)
    elif backend == "mojo" and _mojo_available:
        return _reconstruct_with_mojo(result)
    elif backend == "cython" and _cython_quant is not None:
        return _cython_quant.reconstruct_embeddings_cython(result.quantized, result.scales.astype(np.float32))
    elif backend == "numpy" or backend == "auto":
        # Fallback to NumPy implementation
        q = result.quantized
        if q.ndim == 1:
            q = q.reshape(1, -1)

        q2 = q.astype(np.float32)
        scales = np.asarray(result.scales, dtype=np.float32)
        if scales.ndim == 0:
            scales = scales.reshape(1)
        if scales.shape[0] != result.n:
            raise ValueError("scales length must match number of vectors")
        recon = q2 * scales[:, None]
        return recon
    else:
        raise ValueError(f"Unknown or unavailable backend: {backend}")


def mean_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between rows of a and b."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if a.shape != b.shape:
        raise ValueError("shapes must match")

    # If Cython backend available, use it for similarity calculation
    if _cython_quant is not None:
        return _cython_quant.mean_cosine_similarity_cython(a, b)

    # Fallback to NumPy implementation
    # compute norms
    na = a
    nb = b
    norms_a = np.linalg.norm(na, axis=1)
    norms_b = np.linalg.norm(nb, axis=1)

    # prepare normalized arrays but avoid division by zero by temporarily replacing zeros with 1
    safe_a = na.copy()
    safe_b = nb.copy()
    norms_a_safe = norms_a.copy()
    norms_b_safe = norms_b.copy()
    norms_a_safe[norms_a_safe == 0] = 1.0
    norms_b_safe[norms_b_safe == 0] = 1.0
    safe_a = safe_a / norms_a_safe[:, None]
    safe_b = safe_b / norms_b_safe[:, None]

    dots = np.sum(safe_a * safe_b, axis=1)

    # handle zero-vector cases: if both are zero, define cosine = 1.0; if one is zero, define cosine = 0.0
    both_zero = (norms_a == 0) & (norms_b == 0)
    one_zero = ((norms_a == 0) ^ (norms_b == 0))
    dots[both_zero] = 1.0
    dots[one_zero] = 0.0

    return float(np.mean(dots))


if __name__ == "__main__":
    # Show available backends
    print("Available backends:")
    backend_info = get_backend_info()
    for name, available in backend_info.items():
        if name.endswith("_binary"):
            continue
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    print()
    
    # simple demo
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((100, 768)).astype(np.float32)
    result = quantize_embeddings(emb)
    recon = reconstruct_embeddings(result)
    print("Mean cosine:", mean_cosine_similarity(emb, recon))
