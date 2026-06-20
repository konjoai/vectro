"""
Mojo bridge — subprocess dispatch to the compiled ``vectro_quantizer`` binary.

All compute-intensive hot paths (INT8, NF4, Binary) are routed through this
module so that Python is a thin orchestration layer and Mojo performs every
quantization operation.

Data exchange protocol (pipe mode)
------------------------------------
- float32 : n × d × 4 bytes, little-endian IEEE 754 on stdin/stdout
- int8     : n × d × 1 bytes, signed two's complement
- uint8    : n × k × 1 bytes (k = ceil(d/2) for NF4, ceil(d/8) for binary)
- scales   : n × 4 bytes, little-endian IEEE 754 float32

All exposed functions raise ``RuntimeError`` if the binary is unavailable so
callers can fall back gracefully.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Tuple

import numpy as np

# ── binary discovery ─────────────────────────────────────────────────────────

_BINARY_NAME = "vectro_quantizer"


def _find_binary() -> str | None:
    """Return the absolute path to the compiled Mojo binary, or None."""
    candidates = [
        pathlib.Path(__file__).parent / _BINARY_NAME,  # installed wheel (bundled binary)
        pathlib.Path(__file__).parent.parent / _BINARY_NAME,  # repo root (dev)
        pathlib.Path(_BINARY_NAME),
        pathlib.Path.cwd() / _BINARY_NAME,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return str(p.resolve())
    return None


_binary_path: str | None = _find_binary()


def is_available() -> bool:
    """Return True if the ``vectro_quantizer`` binary is reachable."""
    return _binary_path is not None


def binary_path() -> str:
    """Return the path to the binary, raising RuntimeError if absent."""
    if _binary_path is None:
        raise RuntimeError(
            "vectro_quantizer binary not found.  Build it with:  pixi run build-mojo"
        )
    return _binary_path


# ── low-level I/O helpers ─────────────────────────────────────────────────────


def _write_f32(arr: np.ndarray, path: str) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(path)


def _read_f32(path: str, count: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f4", count=count)


def _write_i8(arr: np.ndarray, path: str) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.int8)
    arr.tofile(path)


def _read_i8(path: str, count: int) -> np.ndarray:
    return np.fromfile(path, dtype=np.int8, count=count)


def _write_u8(arr: np.ndarray, path: str) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    arr.tofile(path)


def _read_u8(path: str, count: int) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint8, count=count)


def _run(args: list[str]) -> None:
    """Execute the Mojo binary with ``args``.  Raises on non-zero exit."""
    cmd = [binary_path()] + args
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"vectro_quantizer exited with code {result.returncode}: " + " ".join(cmd)
        )


def _run_pipe(args: list[str], stdin_data: bytes) -> bytes:
    """Execute the Mojo binary in pipe mode.

    Passes ``stdin_data`` on stdin and returns stdout as bytes.
    Raises ``RuntimeError`` on non-zero exit.
    """
    cmd = [binary_path()] + args
    result = subprocess.run(cmd, input=stdin_data, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[:300]
        raise RuntimeError(
            f"vectro_quantizer exited with code {result.returncode}: "
            + " ".join(cmd)
            + (f"\nstderr: {stderr}" if stderr else "")
        )
    return result.stdout


# ── INT8 ──────────────────────────────────────────────────────────────────────


def int8_quantize(
    vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize float32 vectors to INT8 using the Mojo binary.

    Args:
        vectors: Shape (n, d), float32.

    Returns:
        q:      Shape (n, d), int8.
        scales: Shape (n,),   float32.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    n, d = vectors.shape

    stdout = _run_pipe(["pipe", "int8", "quantize", str(n), str(d)], vectors.tobytes())

    q = np.frombuffer(stdout[: n * d], dtype=np.int8).reshape(n, d).copy()
    scales = np.frombuffer(stdout[n * d : n * d + n * 4], dtype="<f4").copy()

    return q, scales


def int8_reconstruct(
    q: np.ndarray,
    scales: np.ndarray,
    d: int | None = None,
) -> np.ndarray:
    """Reconstruct float32 vectors from INT8 + scales using the Mojo binary.

    Args:
        q:      Shape (n, d) or (n*d,), int8.
        scales: Shape (n,), float32.
        d:      Dimension (inferred from q if 2-D).

    Returns:
        Reconstructed float32 array of shape (n, d).
    """
    q = np.ascontiguousarray(q, dtype=np.int8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    n = len(scales)
    if d is None:
        d = q.size // n
    q = q.reshape(n, d)

    stdin_data = q.tobytes() + scales.tobytes()
    stdout = _run_pipe(["pipe", "int8", "recon", str(n), str(d)], stdin_data)

    return np.frombuffer(stdout, dtype="<f4").reshape(n, d).copy()


# ── NF4 ───────────────────────────────────────────────────────────────────────


def nf4_encode(
    vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """NF4-encode float32 vectors using the Mojo binary.

    Args:
        vectors: Shape (n, d), float32.

    Returns:
        packed: Shape (n, ceil(d/2)), uint8.
        scales: Shape (n,), float32.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    n, d = vectors.shape
    half_d = (d + 1) // 2

    stdout = _run_pipe(["pipe", "nf4", "encode", str(n), str(d)], vectors.tobytes())

    packed = np.frombuffer(stdout[: n * half_d], dtype=np.uint8).reshape(n, half_d).copy()
    scales = np.frombuffer(stdout[n * half_d : n * half_d + n * 4], dtype="<f4").copy()

    return packed, scales


def nf4_decode(
    packed: np.ndarray,
    scales: np.ndarray,
    d: int,
) -> np.ndarray:
    """Decode NF4-packed bytes to float32 using the Mojo binary.

    Args:
        packed: Shape (n, ceil(d/2)), uint8.
        scales: Shape (n,), float32.
        d:      Original vector dimension.

    Returns:
        Reconstructed float32 array of shape (n, d).
    """
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    n = len(scales)

    stdin_data = packed.tobytes() + scales.tobytes()
    stdout = _run_pipe(["pipe", "nf4", "decode", str(n), str(d)], stdin_data)

    return np.frombuffer(stdout, dtype="<f4").reshape(n, d).copy()


# ── Binary ────────────────────────────────────────────────────────────────────


def bin_encode(
    vectors: np.ndarray,
) -> np.ndarray:
    """Binary (sign-bit) encode float32 vectors using the Mojo binary.

    Args:
        vectors: Shape (n, d), float32.  Positive values → bit 1.

    Returns:
        packed: Shape (n, ceil(d/8)), uint8.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    n, d = vectors.shape
    bpv = (d + 7) // 8

    stdout = _run_pipe(["pipe", "bin", "encode", str(n), str(d)], vectors.tobytes())

    return np.frombuffer(stdout, dtype=np.uint8).reshape(n, bpv).copy()


def bin_decode(
    packed: np.ndarray,
    d: int,
) -> np.ndarray:
    """Decode binary-packed bytes to {-1, +1} float32 using the Mojo binary.

    Args:
        packed: Shape (n, ceil(d/8)), uint8.
        d:      Original vector dimension.

    Returns:
        Float32 array of shape (n, d) with values in {-1.0, +1.0}.
    """
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    n = packed.shape[0]

    stdout = _run_pipe(["pipe", "bin", "decode", str(n), str(d)], packed.tobytes())

    return np.frombuffer(stdout, dtype="<f4").reshape(n, d).copy()


# ── Product Quantization (PQ) ───────────────────────────────────────────────


def pq_encode(
    vectors: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """PQ-encode float32 vectors using Mojo pipe protocol.

    Args:
        vectors: Shape (n, d), float32.
        centroids: Shape (M, K, sub_dim), float32 where d == M * sub_dim.

    Returns:
        codes: Shape (n, M), uint8.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape={vectors.shape}")

    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    if centroids.ndim != 3:
        raise ValueError(f"centroids must be 3D [M,K,sub_dim], got shape={centroids.shape}")

    n, d = vectors.shape
    M, K, sub_dim = centroids.shape
    if K > 256:
        raise ValueError(f"K={K} exceeds uint8 code limit (256)")
    if d != M * sub_dim:
        raise ValueError(f"dimension mismatch: d={d}, M*sub_dim={M * sub_dim}")

    stdin_data = vectors.tobytes() + centroids.tobytes()
    stdout = _run_pipe(["pipe", "pq", "encode", str(n), str(d), str(M), str(K)], stdin_data)

    return np.frombuffer(stdout, dtype=np.uint8).reshape(n, M).copy()


def pq_decode(
    codes: np.ndarray,
    centroids: np.ndarray,
    d: int | None = None,
) -> np.ndarray:
    """Decode PQ codes to approximate float32 vectors via Mojo pipe protocol.

    Args:
        codes: Shape (n, M), uint8.
        centroids: Shape (M, K, sub_dim), float32.
        d: Optional output dimension override (validated if provided).

    Returns:
        vectors: Shape (n, d), float32.
    """
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2D [n,M], got shape={codes.shape}")

    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    if centroids.ndim != 3:
        raise ValueError(f"centroids must be 3D [M,K,sub_dim], got shape={centroids.shape}")

    n, M_codes = codes.shape
    M, K, sub_dim = centroids.shape
    if K > 256:
        raise ValueError(f"K={K} exceeds uint8 code limit (256)")
    if M_codes != M:
        raise ValueError(f"codebook mismatch: codes M={M_codes}, centroids M={M}")

    out_d = M * sub_dim
    if d is not None and d != out_d:
        raise ValueError(f"d mismatch: provided {d}, inferred {out_d}")

    stdin_data = codes.tobytes() + centroids.tobytes()
    stdout = _run_pipe(["pipe", "pq", "decode", str(n), str(out_d), str(M), str(K)], stdin_data)

    return np.frombuffer(stdout, dtype="<f4").reshape(n, out_d).copy()
