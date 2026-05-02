"""
_rust_bridge.py — Direct access to the vectro_py Rust/PyO3 SIMD backend.

Provides platform-aware INT8 and NF4 batch quantization through the Rust
extension, dispatching to:
  - NEON (AArch64 / Apple Silicon) — always active on arm64 builds
  - AVX2 (x86-64) — runtime-detected via is_x86_feature_detected!
  - Scalar fallback — all other targets

This module is the recommended entry point for benchmarking the Rust path
independently of the Mojo or NumPy fallback paths.  Use ``is_available()``
to gate any code that requires the extension.

See also:
  - ``_mojo_bridge.py`` — Mojo subprocess IPC path
  - ``interface.py``    — unified dispatch (Rust > Mojo > NumPy)
"""

from __future__ import annotations

import platform
from typing import Optional, Tuple

import numpy as np

_vectro_py = None
try:
    import vectro_py as _vectro_py
except ImportError:
    pass


def is_available() -> bool:
    return _vectro_py is not None


def simd_tier() -> str:
    """Return the SIMD tier active on this platform."""
    arch = platform.machine().lower()
    if "arm64" in arch or "aarch64" in arch:
        return "neon"
    if "x86_64" in arch or "amd64" in arch:
        import platform as _p
        os_type = _p.system()
        if os_type == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    flags = ""
                    for line in f:
                        if line.startswith("flags"):
                            flags = line
                            break
                if "avx512f" in flags:
                    return "avx512"
                if "avx2" in flags:
                    return "avx2"
            except OSError:
                pass
        elif os_type == "Darwin":
            import subprocess
            for flag, name in [("hw.optional.avx512f", "avx512"), ("hw.optional.avx2_0", "avx2")]:
                try:
                    r = subprocess.run(["sysctl", "-n", flag], capture_output=True, text=True, timeout=3)
                    if r.returncode == 0 and r.stdout.strip() == "1":
                        return name
                except Exception:
                    pass
        return "avx2"  # safe assumption for any modern x86
    return "scalar"


def quantize_int8_batch(
    vectors: np.ndarray,
    assume_normalized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize [N, D] float32 array to INT8 using Rust SIMD.

    Parameters
    ----------
    vectors : np.ndarray
        Float32 input matrix.  C-contiguous arrays are zero-copy; non-
        contiguous arrays are widened in Rust.
    assume_normalized : bool, default False
        Wave 1.3 — when True, dispatches to
        ``quantize_int8_batch_normalized`` which skips the abs-max scan.
        Caller asserts every row has ``||·||_2 ≤ 1``.  ~1.4× faster on
        memory-bandwidth-bound workloads at the cost of ~0.99 cosine
        recall floor instead of 0.9999.

    Returns
    -------
    codes  : np.ndarray, shape [N, D], dtype int8
    scales : np.ndarray, shape [N],    dtype float32  (abs_max per row)

    Raises
    ------
    RuntimeError if the Rust extension is not installed.
    """
    if _vectro_py is None:
        raise RuntimeError(
            "vectro_py Rust extension not installed. "
            "Build it with: cd rust && maturin develop --release"
        )
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(1, -1)
    if assume_normalized and hasattr(_vectro_py, "quantize_int8_batch_normalized"):
        codes, scales = _vectro_py.quantize_int8_batch_normalized(arr)
    else:
        codes, scales = _vectro_py.quantize_int8_batch(arr)
    if was_1d:
        codes = codes.reshape(-1)
        scales = scales.reshape(1)
    return codes, scales


def quantize_int8_batch_from_f16(
    vectors_f16: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wave 4 — Quantize a [N, D] float16 array to INT8 directly.

    Halves the input bandwidth versus a separate ``vectors.astype(f32)``
    widening pass; the f16→f32 widening is fused into the per-row encode.

    Returns the same ``(codes, scales)`` shape contract as
    :func:`quantize_int8_batch`.
    """
    if _vectro_py is None:
        raise RuntimeError("vectro_py Rust extension not installed.")
    if not hasattr(_vectro_py, "quantize_int8_batch_from_f16"):
        # Old extension built before Wave 4 — widen in Python.
        return quantize_int8_batch(vectors_f16.astype(np.float32))
    arr = np.ascontiguousarray(vectors_f16, dtype=np.float16)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(1, -1)
    codes, scales = _vectro_py.quantize_int8_batch_from_f16(arr)
    if was_1d:
        codes = codes.reshape(-1)
        scales = scales.reshape(1)
    return codes, scales


def dequantize_int8_batch(
    codes: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Reconstruct float32 from INT8 codes using Rust SIMD.

    Parameters
    ----------
    codes  : np.ndarray, shape [N, D], dtype int8
    scales : np.ndarray, shape [N],    dtype float32

    Returns
    -------
    np.ndarray, shape [N, D], dtype float32
    """
    if _vectro_py is None:
        raise RuntimeError("vectro_py Rust extension not installed.")
    c = np.ascontiguousarray(codes, dtype=np.int8)
    s = np.ascontiguousarray(scales, dtype=np.float32)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    return _vectro_py.dequantize_int8_batch(c, s)


def encode_nf4_batch(
    vectors: np.ndarray,
) -> list[Tuple[bytes, float, int]]:
    """Encode [N, D] float32 array to packed NF4 nibbles row by row.

    Returns a list of (packed_bytes, scale, dim) tuples.
    Uses ``encode_nf4_fast`` which dispatches to NEON or AVX2 abs-max scan.
    """
    if _vectro_py is None:
        raise RuntimeError("vectro_py Rust extension not installed.")
    results = []
    for row in vectors:
        packed, scale, dim = _vectro_py.encode_nf4_fast(row.tolist())
        results.append((bytes(packed), scale, dim))
    return results


def benchmark_int8_throughput(
    dim: int = 768,
    n_vectors: int = 50_000,
    warmup: int = 2,
    runs: int = 5,
) -> dict:
    """Measure Rust SIMD INT8 encode throughput.

    Returns a dict with keys: mean_vec_per_sec, std_vec_per_sec,
    min_vec_per_sec, max_vec_per_sec, simd_tier, platform, dim, n_vectors.
    """
    import time

    if _vectro_py is None:
        raise RuntimeError("vectro_py Rust extension not installed.")

    rng = np.random.default_rng(42)
    vectors = np.ascontiguousarray(
        rng.standard_normal((n_vectors, dim)).astype(np.float32)
    )

    for _ in range(warmup):
        _vectro_py.quantize_int8_batch(vectors[:min(1000, n_vectors)])

    throughputs = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _vectro_py.quantize_int8_batch(vectors)
        throughputs.append(n_vectors / (time.perf_counter() - t0))

    arr = np.array(throughputs)
    return {
        "mean_vec_per_sec": float(arr.mean()),
        "std_vec_per_sec": float(arr.std()),
        "min_vec_per_sec": float(arr.min()),
        "max_vec_per_sec": float(arr.max()),
        "simd_tier": simd_tier(),
        "platform": platform.machine(),
        "dim": dim,
        "n_vectors": n_vectors,
    }


if __name__ == "__main__":
    if not is_available():
        print("vectro_py not installed — build with: cd rust && maturin develop --release")
    else:
        print(f"SIMD tier: {simd_tier()}")
        for d in [128, 384, 768, 1536]:
            result = benchmark_int8_throughput(dim=d)
            print(
                f"  d={d:4d}: {result['mean_vec_per_sec']:>10.0f} ± "
                f"{result['std_vec_per_sec']:>7.0f} vec/s  ({result['simd_tier']})"
            )
