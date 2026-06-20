#!/usr/bin/env python
"""
Real Embedding Dataset Benchmark v2 — GloVe-100 / SIFT1M

Downloads the real embedding dataset on first run, caches it as a .npy
file, then benchmarks all Vectro quantization modes (int8, nf4, binary,
auto) measuring throughput and reconstruction cosine similarity.

Replaces benchmark_real_embeddings.py which used synthetic data only.

Cache location:  ~/.cache/vectro_benchmarks/
GloVe download:  https://nlp.stanford.edu/data/glove.6B.zip  (862 MB)
SIFT1M download: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz  (500 MB)

Usage:
    python benchmarks/benchmark_real_embeddings_v2.py
    python benchmarks/benchmark_real_embeddings_v2.py --dataset glove-100
    python benchmarks/benchmark_real_embeddings_v2.py --dataset sift1m
    python benchmarks/benchmark_real_embeddings_v2.py --max-vectors 10000
    python benchmarks/benchmark_real_embeddings_v2.py --output results/re_v2.json
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = Path.home() / ".cache" / "vectro_benchmarks"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _show_progress(count: int, block: int, total: int) -> None:
    """urlretrieve progress callback."""
    pct = min(100.0, count * block / max(total, 1) * 100)
    print(f"\r  Downloading... {pct:.1f}%", end="", flush=True)


def load_glove_100(max_vectors: int = 100_000) -> np.ndarray:
    """Load GloVe 100-dimensional word vectors.

    Downloads the GloVe.6B zip (862 MB) on first call and caches the
    vectors as a .npy file at ``~/.cache/vectro_benchmarks/glove_100d.npy``.
    Subsequent calls load directly from cache.

    Args:
        max_vectors: Maximum number of vectors to return.
    Returns:
        float32 array of shape (min(n_available, max_vectors), 100).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = CACHE_DIR / "glove_100d.npy"

    if npy_path.exists():
        data = np.load(str(npy_path))
        return data[:max_vectors].astype(np.float32)

    # ── Download ─────────────────────────────────────────────────────────────
    zip_path = CACHE_DIR / "glove.6B.zip"
    if not zip_path.exists():
        print("  Downloading GloVe.6B zip (~862 MB) from Stanford...")
        print(f"  Cache: {zip_path}")
        urlretrieve(GLOVE_URL, str(zip_path), reporthook=_show_progress)
        print()  # newline after progress bar

    # ── Parse ─────────────────────────────────────────────────────────────────
    print("  Parsing glove.6B.100d.txt...", end="", flush=True)
    vecs: list[np.ndarray] = []
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        with zf.open("glove.6B.100d.txt") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                parts = line.rstrip().split(" ")
                if len(parts) < 101:
                    continue
                vecs.append(
                    np.fromiter(
                        (float(x) for x in parts[1:101]),
                        dtype=np.float32,
                        count=100,
                    )
                )
    print(f" {len(vecs):,} vectors parsed")

    data = np.stack(vecs, axis=0).astype(np.float32)
    np.save(str(npy_path), data)
    print(f"  Cached at {npy_path}")
    return data[:max_vectors]


def load_sift1m(max_vectors: int = 100_000) -> np.ndarray:
    """Load SIFT1M 128-dimensional SIFT feature vectors.

    Downloads the SIFT tarball via FTP (~500 MB) on first call.  The base
    vectors (sift_base.fvecs, 1M × 128 float32) are cached as a .npy file.

    Args:
        max_vectors: Maximum number of vectors to return.
    Returns:
        float32 array of shape (min(1_000_000, max_vectors), 128).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = CACHE_DIR / "sift1m_base.npy"

    if npy_path.exists():
        data = np.load(str(npy_path))
        return data[:max_vectors].astype(np.float32)

    tar_path = CACHE_DIR / "sift.tar.gz"
    if not tar_path.exists():
        print("  Downloading SIFT1M tarball (~500 MB) from IRISA FTP...")
        print(f"  Cache: {tar_path}")
        urlretrieve(SIFT_URL, str(tar_path), reporthook=_show_progress)
        print()

    import tarfile

    print("  Extracting sift_base.fvecs...", end="", flush=True)
    with tarfile.open(str(tar_path), "r:gz") as tf:
        member = next(m for m in tf.getmembers() if "sift_base.fvecs" in m.name)
        fvecs_bytes = tf.extractfile(member).read()
    print(" done")

    # fvecs format: [dim(int32), f0, f1, ..., f_{dim-1}] repeated
    print("  Parsing fvecs...", end="", flush=True)
    arr = np.frombuffer(fvecs_bytes, dtype=np.float32)
    # First element of each record is the dimension stored as float32 bits of int32=128.
    dim_as_float = arr[0]
    dim = struct.unpack("i", struct.pack("f", dim_as_float))[0]
    stride = dim + 1
    n_vecs = len(arr) // stride
    data = np.array(
        [arr[i * stride + 1 : i * stride + 1 + dim] for i in range(n_vecs)], dtype=np.float32
    )
    print(f" {n_vecs:,} vectors (d={dim})")

    np.save(str(npy_path), data)
    print(f"  Cached at {npy_path}")
    return data[:max_vectors]


# ---------------------------------------------------------------------------
# Quantization benchmark helpers
# ---------------------------------------------------------------------------


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-row cosine similarity between two (n, d) arrays."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float(np.mean((an * bn).sum(axis=1)))


def benchmark_mode(
    vectors: np.ndarray,
    mode: str,
    n_warmup: int = 2,
    n_iters: int = 5,
) -> dict[str, Any]:
    """Benchmark a single quantization mode.

    Args:
        vectors:  (n, d) float32 input.
        mode:     One of "int8", "nf4", "binary", "auto".
        n_warmup: Number of warm-up iterations (not timed).
        n_iters:  Number of timed iterations; best time is kept.
    Returns:
        Dict with throughput_vec_per_sec, mean_cosine_similarity, compression_ratio.
    """
    from python import compress_vectors, decompress_vectors  # type: ignore[import]

    # Warm up
    for _ in range(n_warmup):
        compress_vectors(vectors[:1000], profile=mode)

    # Time: best-of n_iters
    best_sec: float = float("inf")
    result = None
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = compress_vectors(vectors, profile=mode)
        elapsed = time.perf_counter() - t0
        if elapsed < best_sec:
            best_sec = elapsed

    throughput = len(vectors) / best_sec if best_sec > 0 else 0.0

    # Quality: reconstruct and measure cosine similarity
    try:
        reconstructed = decompress_vectors(result)
        cosine = _cosine_sim(vectors, reconstructed)
    except Exception:
        cosine = float("nan")

    compression_ratio = getattr(result, "compression_ratio", float("nan"))

    return {
        "throughput_vec_per_sec": int(throughput),
        "mean_cosine_similarity": round(cosine, 6),
        "compression_ratio": float(compression_ratio),
        "best_time_sec": round(best_sec, 4),
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    dataset: str = "glove-100",
    max_vectors: int = 100_000,
    modes: list[str] | None = None,
    output_path: Path = Path("results/real_embeddings_v2.json"),
) -> dict[str, Any]:
    """Load real embedding data and benchmark all quantization modes.

    Args:
        dataset:     "glove-100" or "sift1m".
        max_vectors: Maximum vectors to use from the dataset.
        modes:       List of modes to benchmark; defaults to all four.
        output_path: JSON output path.
    Returns:
        Results dict.
    """
    if modes is None:
        modes = ["fast", "binary"]

    print("=" * 70)
    print(f"Real Embedding Benchmark v2 — {dataset}  (max {max_vectors:,} vectors)")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    if dataset == "glove-100":
        vectors = load_glove_100(max_vectors)
    elif dataset == "sift1m":
        vectors = load_sift1m(max_vectors)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose: glove-100, sift1m")

    n, d = vectors.shape
    print(f"  Loaded {n:,} vectors × d={d}  dtype={vectors.dtype}")

    results: dict[str, Any] = {
        "benchmark": "real_embeddings_v2",
        "dataset": dataset,
        "n_vectors": n,
        "d": d,
        "modes": {},
    }

    # ── Benchmark each mode ───────────────────────────────────────────────────
    print(f"\n{'Mode':<10} {'Throughput':>14} {'Cosine':>8} {'Ratio':>7}")
    print("-" * 44)

    for mode in modes:
        print(f"  {mode:<8}", end="", flush=True)
        try:
            stats = benchmark_mode(vectors, mode)
            tput = stats["throughput_vec_per_sec"]
            cos = stats["mean_cosine_similarity"]
            ratio = stats["compression_ratio"]
            print(f"  {tput:>12,} vec/s  {cos:>7.4f}  {ratio:>5.1f}x")
            results["modes"][mode] = stats
        except Exception as exc:
            msg = str(exc)[:100]
            print(f"  ERROR: {msg}")
            results["modes"][mode] = {"status": "error", "message": msg}

    print()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = output_path.parent / (
        output_path.stem + f"_{dataset.replace('-', '')}" + output_path.suffix
    )
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Vectro quantization on real embedding datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="glove-100",
        choices=["glove-100", "sift1m"],
        help="Dataset to use (default: glove-100)",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=100_000,
        help="Maximum number of vectors to benchmark (default: 100000)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["fast", "ultra", "binary"],
        help="Quantization modes (fast=int8 4x, ultra=int4 8x d÷64 only, binary=1-bit 32x)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/real_embeddings_v2.json",
        help="Output JSON path (dataset name appended)",
    )
    args = parser.parse_args()
    run_benchmark(
        dataset=args.dataset,
        max_vectors=args.max_vectors,
        modes=args.modes,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
