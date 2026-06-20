#!/usr/bin/env python
"""
Benchmark script for Python/NumPy fallback mode.

This measures performance when the Mojo binary is not available,
using only Python + NumPy + optional squish_quant (Rust).

Usage:
    python benchmarks/benchmark_python_fallback.py [--output results.json]
"""

import json
import time
import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.v3_api import VectroV3
from python import compress_vectors


def benchmark_int8_throughput(
    dims: int,
    batch_sizes: List[int] = [100, 1000, 10000],
    num_vectors: int = 100000,
) -> Dict[str, Any]:
    """
    Benchmark INT8 quantization throughput at various batch sizes.

    Parameters
    ----------
    dims : int
        Embedding dimensionality (128, 384, 768, 1536)
    batch_sizes : list of int
        Batch sizes to test
    num_vectors : int
        Total number of vectors to process

    Returns
    -------
    dict with keys:
        - "dimension": int
        - "total_vectors": int
        - "results": list of dicts with "batch_size", "throughput_vec_per_sec", "time_sec"
    """
    print(f"\n▶ INT8 Throughput Benchmark (d={dims})")

    # Generate test data
    vectors = np.random.normal(size=(num_vectors, dims)).astype(np.float32)

    results = []
    for batch_size in batch_sizes:
        # Warm up
        _ = compress_vectors(vectors[: min(1000, batch_size)], profile="balanced")

        # Timed run
        start_time = time.time()
        for i in range(0, num_vectors, batch_size):
            batch = vectors[i : i + batch_size]
            compress_vectors(batch, profile="balanced")
        elapsed = time.time() - start_time

        throughput = num_vectors / elapsed
        results.append(
            {
                "batch_size": batch_size,
                "vectors_processed": num_vectors,
                "time_sec": round(elapsed, 3),
                "throughput_vec_per_sec": int(throughput),
            }
        )
        print(f"  Batch {batch_size:>5}: {throughput:>10.0f} vec/s ({elapsed:.2f}s)")

    return {
        "dimension": dims,
        "total_vectors": num_vectors,
        "results": results,
    }


def benchmark_quality_metrics(dims: int, num_vectors: int = 10000) -> Dict[str, Any]:
    """
    Measure cosine similarity and compression ratio for INT8, NF4, and Binary.

    Parameters
    ----------
    dims : int
        Embedding dimensionality
    num_vectors : int
        Number of vectors to test

    Returns
    -------
    dict with quality metrics by profile
    """
    print(f"\n▶ Quality Metrics (d={dims})")

    vectors = np.random.normal(size=(num_vectors, dims)).astype(np.float32)

    v3 = VectroV3()
    profiles = ["int8", "nf4", "binary"]
    results = {}

    for profile in profiles:
        print(f"  Testing {profile}...", end="", flush=True)
        try:
            v3 = VectroV3(profile=profile)
            result = v3.compress(vectors)
            decompressed = v3.decompress(result)

            # Calculate cosine similarity
            original_norm = np.linalg.norm(vectors, axis=1)
            decompressed_norm = np.linalg.norm(decompressed, axis=1)

            cosine_sims = np.sum(vectors * decompressed, axis=1) / (
                original_norm * decompressed_norm + 1e-8
            )
            mean_cosine = float(np.mean(cosine_sims))

            # Calculate compression ratio
            original_bytes = vectors.nbytes
            compressed_bytes = sum(v.nbytes for v in result.data.values())
            compression_ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0

            results[profile] = {
                "mean_cosine_similarity": round(mean_cosine, 6),
                "std_cosine_similarity": round(float(np.std(cosine_sims)), 6),
                "compression_ratio": round(compression_ratio, 2),
            }
            print(f" cosine_sim={mean_cosine:.5f}, ratio={compression_ratio:.1f}x")
        except Exception as e:
            print(f" ERROR: {e}")
            results[profile] = {"error": str(e)}

    return {
        "dimension": dims,
        "num_vectors": num_vectors,
        "profiles": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Vectro Python/NumPy fallback mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/benchmark_python_fallback.py
  python benchmarks/benchmark_python_fallback.py --output results/fallback_benchmark.json
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/fallback_benchmark.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", default=[768], help="Dimensions to benchmark (default: 768)"
    )
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VECTRO — Python/NumPy Fallback Benchmark")
    print("=" * 70)
    print(f"Output: {output_path}")

    all_results = {
        "benchmark_type": "python_fallback",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "throughput_results": [],
        "quality_results": [],
    }

    # Run throughput benchmarks for each dimension
    for dim in args.dims:
        result = benchmark_int8_throughput(dim)
        all_results["throughput_results"].append(result)

    # Run quality benchmarks for an example dimension
    if args.dims:
        result = benchmark_quality_metrics(args.dims[0])
        all_results["quality_results"].append(result)

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    return all_results


if __name__ == "__main__":
    main()
