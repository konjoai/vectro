#!/usr/bin/env python3
"""
Cross-Platform Benchmarking Framework for Vectro

Provides unified benchmarking across:
- Apple Silicon (M1, M2, M3) — Mojo + Rust NEON paths
- Intel macOS x86 — Rust AVX2 path
- Linux x86 — Rust AVX2 and AVX-512 paths

Usage:
    python benchmarks/cross_platform_benchmark.py --output results/
    python benchmarks/cross_platform_benchmark.py --mode quick --dimensions 768
    python benchmarks/cross_platform_benchmark.py --mode full --platforms all
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

# Ensure project root is importable
_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.platform_detection import detect_platform

try:
    from python._rust_bridge import (
        is_available as _rust_available,
        simd_tier as _rust_simd_tier,
        quantize_int8_batch as _rust_quantize_int8,
    )
except ImportError:

    def _rust_available() -> bool:
        return False  # type: ignore[misc]

    def _rust_simd_tier() -> str:
        return "unavailable"  # type: ignore[misc]

    def _rust_quantize_int8(v):
        raise RuntimeError("vectro_py not installed")  # type: ignore[misc]


@dataclass
class BenchmarkResult:
    """Single benchmark result."""

    benchmark_name: str
    dimension: int
    num_vectors: int
    warmup_runs: int
    measurement_runs: int
    mean_throughput_vec_per_sec: float
    std_throughput_vec_per_sec: float
    min_throughput_vec_per_sec: float
    max_throughput_vec_per_sec: float
    total_time_sec: float
    path_type: str  # 'python_fallback', 'rust_avx2', 'rust_avx512', 'rust_neon', 'mojo'
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrossPlatformBenchmark:
    """Unified benchmarking interface across platforms."""

    def __init__(self, output_dir: Path = None):
        self.platform = detect_platform()
        self.output_dir = output_dir or Path(__file__).parent / "results" / "cross_platform"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

        print(f"\n{'=' * 70}")
        print("VECTRO CROSS-PLATFORM BENCHMARK")
        print(f"{'=' * 70}")
        print(f"Platform:     {self.platform.os_type} {self.platform.os_version}")
        print(f"CPU:          {self.platform.cpu_generation or self.platform.cpu_model}")
        print(f"Cores:        {self.platform.cpu_cores}")
        print(
            f"Memory:       {self.platform.memory_gb:.1f} GB"
            if self.platform.memory_gb
            else "Memory:       Unknown"
        )
        print(f"SIMD:         {', '.join(self.platform.simd_capabilities)}")
        print(f"Mojo:         {'✓' if self.platform.mojo_available else '✗'}")
        print(f"Rust:         {'✓' if self.platform.rust_available else '✗'}")
        print(f"FAISS:        {'✓' if self.platform.faiss_available else '✗'}")
        print(f"{'=' * 70}\n")

    def benchmark_int8_throughput(
        self,
        dimensions: List[int] = [128, 384, 768, 1536],
        num_vectors: int = 100000,
        warmup_runs: int = 2,
        measurement_runs: int = 5,
    ) -> List[BenchmarkResult]:
        """Benchmark INT8 quantization throughput across dimensions."""
        print("\n▶ INT8 Throughput Benchmarks")
        print(f"  Vectors: {num_vectors:,}, Dimensions: {dimensions}")
        print(f"  Warmup: {warmup_runs}, Measurements: {measurement_runs}\n")

        results = []

        try:
            from python.batch_api import quantize_batch
        except ImportError:
            print("  ✗ batch_api not available")
            return results

        for dim in dimensions:
            print(f"  d={dim:4d}: ", end="", flush=True)

            # Generate test vectors
            vectors = np.random.normal(0, 1, size=(num_vectors, dim)).astype(np.float32)

            # Warmup runs
            for _ in range(warmup_runs):
                _ = quantize_batch(vectors[: min(1000, num_vectors)], profile="int8")

            # Measurement runs
            throughputs = []
            total_time = 0

            for run in range(measurement_runs):
                start = time.perf_counter()
                _ = quantize_batch(vectors, profile="int8")
                elapsed = time.perf_counter() - start

                throughput = num_vectors / elapsed
                throughputs.append(throughput)
                total_time += elapsed

                print(".", end="", flush=True)

            # Compute statistics
            throughputs = np.array(throughputs)
            mean_tp = float(np.mean(throughputs))
            std_tp = float(np.std(throughputs))
            min_tp = float(np.min(throughputs))
            max_tp = float(np.max(throughputs))

            # Detect path type
            path_type = self._detect_path_type()

            result = BenchmarkResult(
                benchmark_name="INT8 Quantization",
                dimension=dim,
                num_vectors=num_vectors,
                warmup_runs=warmup_runs,
                measurement_runs=measurement_runs,
                mean_throughput_vec_per_sec=mean_tp,
                std_throughput_vec_per_sec=std_tp,
                min_throughput_vec_per_sec=min_tp,
                max_throughput_vec_per_sec=max_tp,
                total_time_sec=total_time,
                path_type=path_type,
            )

            results.append(result)
            print(f" {mean_tp:>10.0f} ± {std_tp:>8.0f} vec/s")

        self.results.extend(results)
        return results

    def benchmark_nf4_throughput(
        self,
        dimensions: List[int] = [128, 384, 768, 1536],
        num_vectors: int = 100000,
        warmup_runs: int = 2,
        measurement_runs: int = 5,
    ) -> List[BenchmarkResult]:
        """Benchmark NF4 quantization throughput."""
        print("\n▶ NF4 Throughput Benchmarks")
        print(f"  Vectors: {num_vectors:,}, Dimensions: {dimensions}")
        print(f"  Warmup: {warmup_runs}, Measurements: {measurement_runs}\n")

        results = []

        try:
            from python.batch_api import quantize_batch
        except ImportError:
            print("  ✗ batch_api not available")
            return results

        for dim in dimensions:
            print(f"  d={dim:4d}: ", end="", flush=True)

            # Generate test vectors
            vectors = np.random.normal(0, 1, size=(num_vectors, dim)).astype(np.float32)

            # Warmup runs
            for _ in range(warmup_runs):
                _ = quantize_batch(vectors[: min(1000, num_vectors)], profile="nf4")

            # Measurement runs
            throughputs = []
            total_time = 0

            for run in range(measurement_runs):
                start = time.perf_counter()
                _ = quantize_batch(vectors, profile="nf4")
                elapsed = time.perf_counter() - start

                throughput = num_vectors / elapsed
                throughputs.append(throughput)
                total_time += elapsed

                print(".", end="", flush=True)

            # Compute statistics
            throughputs = np.array(throughputs)
            mean_tp = float(np.mean(throughputs))
            std_tp = float(np.std(throughputs))
            min_tp = float(np.min(throughputs))
            max_tp = float(np.max(throughputs))

            path_type = self._detect_path_type()

            result = BenchmarkResult(
                benchmark_name="NF4 Quantization",
                dimension=dim,
                num_vectors=num_vectors,
                warmup_runs=warmup_runs,
                measurement_runs=measurement_runs,
                mean_throughput_vec_per_sec=mean_tp,
                std_throughput_vec_per_sec=std_tp,
                min_throughput_vec_per_sec=min_tp,
                max_throughput_vec_per_sec=max_tp,
                total_time_sec=total_time,
                path_type=path_type,
            )

            results.append(result)
            print(f" {mean_tp:>10.0f} ± {std_tp:>8.0f} vec/s")

        self.results.extend(results)
        return results

    def benchmark_binary_throughput(
        self,
        dimensions: List[int] = [128, 384, 768, 1536],
        num_vectors: int = 100000,
        warmup_runs: int = 2,
        measurement_runs: int = 5,
    ) -> List[BenchmarkResult]:
        """Benchmark Binary quantization throughput."""
        print("\n▶ Binary Quantization Benchmarks")
        print(f"  Vectors: {num_vectors:,}, Dimensions: {dimensions}")
        print(f"  Warmup: {warmup_runs}, Measurements: {measurement_runs}\n")

        results = []

        try:
            from python.batch_api import quantize_batch
        except ImportError:
            print("  ✗ batch_api not available")
            return results

        for dim in dimensions:
            print(f"  d={dim:4d}: ", end="", flush=True)

            # Generate test vectors
            vectors = np.random.normal(0, 1, size=(num_vectors, dim)).astype(np.float32)

            # Warmup runs
            for _ in range(warmup_runs):
                _ = quantize_batch(vectors[: min(1000, num_vectors)], profile="binary")

            # Measurement runs
            throughputs = []
            total_time = 0

            for run in range(measurement_runs):
                start = time.perf_counter()
                _ = quantize_batch(vectors, profile="binary")
                elapsed = time.perf_counter() - start

                throughput = num_vectors / elapsed
                throughputs.append(throughput)
                total_time += elapsed

                print(".", end="", flush=True)

            # Compute statistics
            throughputs = np.array(throughputs)
            mean_tp = float(np.mean(throughputs))
            std_tp = float(np.std(throughputs))
            min_tp = float(np.min(throughputs))
            max_tp = float(np.max(throughputs))

            path_type = self._detect_path_type()

            result = BenchmarkResult(
                benchmark_name="Binary Quantization",
                dimension=dim,
                num_vectors=num_vectors,
                warmup_runs=warmup_runs,
                measurement_runs=measurement_runs,
                mean_throughput_vec_per_sec=mean_tp,
                std_throughput_vec_per_sec=std_tp,
                min_throughput_vec_per_sec=min_tp,
                max_throughput_vec_per_sec=max_tp,
                total_time_sec=total_time,
                path_type=path_type,
            )

            results.append(result)
            print(f" {mean_tp:>10.0f} ± {std_tp:>8.0f} vec/s")

        self.results.extend(results)
        return results

    def benchmark_rust_int8_throughput(
        self,
        dimensions: List[int] = [128, 384, 768, 1536],
        num_vectors: int = 50_000,
        warmup_runs: int = 2,
        measurement_runs: int = 5,
    ) -> List[BenchmarkResult]:
        """Benchmark INT8 throughput via the Rust/PyO3 SIMD path directly.

        This bypasses the Python batch_api dispatch chain and calls
        vectro_py.quantize_int8_batch directly, giving a clean measure of
        the NEON (AArch64) or AVX2 (x86-64) Rust path.
        """
        if not _rust_available():
            print("\n⚠ Rust extension (vectro_py) not available — skipping Rust path benchmarks")
            print("  Build with: cd rust && maturin develop --release")
            return []

        tier = _rust_simd_tier()
        print(f"\n▶ Rust SIMD INT8 Throughput Benchmarks  [{tier.upper()}]")
        print(f"  Vectors: {num_vectors:,}, Dimensions: {dimensions}")
        print(f"  Warmup: {warmup_runs}, Measurements: {measurement_runs}\n")

        results = []
        for dim in dimensions:
            print(f"  d={dim:4d}: ", end="", flush=True)
            rng = np.random.default_rng(42)
            vectors = np.ascontiguousarray(
                rng.standard_normal((num_vectors, dim)).astype(np.float32)
            )

            for _ in range(warmup_runs):
                _rust_quantize_int8(vectors[: min(1000, num_vectors)])

            throughputs = []
            total_time = 0.0
            for _ in range(measurement_runs):
                t0 = time.perf_counter()
                _rust_quantize_int8(vectors)
                elapsed = time.perf_counter() - t0
                throughputs.append(num_vectors / elapsed)
                total_time += elapsed
                print(".", end="", flush=True)

            arr = np.array(throughputs)
            result = BenchmarkResult(
                benchmark_name="Rust SIMD INT8",
                dimension=dim,
                num_vectors=num_vectors,
                warmup_runs=warmup_runs,
                measurement_runs=measurement_runs,
                mean_throughput_vec_per_sec=float(arr.mean()),
                std_throughput_vec_per_sec=float(arr.std()),
                min_throughput_vec_per_sec=float(arr.min()),
                max_throughput_vec_per_sec=float(arr.max()),
                total_time_sec=total_time,
                path_type=f"rust_{tier}",
                notes="vectro_py direct — no Python dispatch overhead",
            )
            results.append(result)
            print(f" {arr.mean():>10.0f} ± {arr.std():>8.0f} vec/s")

        self.results.extend(results)
        return results

    def _detect_path_type(self) -> str:
        """Detect which execution path is active for the current quantize_batch call."""
        if self.platform.mojo_available:
            return "mojo"
        if _rust_available():
            return f"rust_{_rust_simd_tier()}"
        return "python_fallback"

    def save_results(self, filename: str = None) -> Path:
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cpu_friendly = self.platform.cpu_generation or self.platform.cpu_model.replace(" ", "_")
            cpu_friendly = cpu_friendly.replace(",", "").replace("(", "").replace(")", "")
            filename = f"benchmark_{cpu_friendly}_{timestamp}.json"

        output_path = self.output_dir / filename

        data = {
            "platform": self.platform.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")
        return output_path

    def print_summary(self):
        """Print summary of all benchmark results."""
        print(f"\n{'=' * 70}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 70}\n")

        # Group by benchmark name
        by_name = {}
        for result in self.results:
            if result.benchmark_name not in by_name:
                by_name[result.benchmark_name] = []
            by_name[result.benchmark_name].append(result)

        for name, results in by_name.items():
            print(f"{name}:")
            for r in sorted(results, key=lambda x: x.dimension):
                print(
                    f"  d={r.dimension:4d}: {r.mean_throughput_vec_per_sec:>12.0f} ± "
                    f"{r.std_throughput_vec_per_sec:>8.0f} vec/s "
                    f"({r.path_type})"
                )
            print()


def main():
    parser = argparse.ArgumentParser(description="Vectro Cross-Platform Benchmarking Framework")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: benchmarks/results/cross_platform/)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[128, 384, 768, 1536],
        help="Embedding dimensions to benchmark (default: 128 384 768 1536)",
    )
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=100000,
        help="Number of vectors to benchmark (default: 100000)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["int8", "nf4", "binary"],
        choices=["int8", "nf4", "binary", "rust", "pq", "hnsw", "all"],
        help="Which benchmarks to run (default: int8 nf4 binary). "
        'Use "rust" to benchmark the vectro_py Rust SIMD path directly.',
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=2, help="Number of warmup runs (default: 2)"
    )
    parser.add_argument(
        "--measurement-runs", type=int, default=5, help="Number of measurement runs (default: 5)"
    )

    args = parser.parse_args()

    # Create benchmark instance
    bench = CrossPlatformBenchmark(output_dir=args.output_dir)

    # Expand 'all'
    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["int8", "nf4", "binary", "rust"]

    # Run requested benchmarks
    try:
        if "int8" in benchmarks:
            bench.benchmark_int8_throughput(
                dimensions=args.dimensions,
                num_vectors=args.num_vectors,
                warmup_runs=args.warmup_runs,
                measurement_runs=args.measurement_runs,
            )

        if "nf4" in benchmarks:
            bench.benchmark_nf4_throughput(
                dimensions=args.dimensions,
                num_vectors=args.num_vectors,
                warmup_runs=args.warmup_runs,
                measurement_runs=args.measurement_runs,
            )

        if "binary" in benchmarks:
            bench.benchmark_binary_throughput(
                dimensions=args.dimensions,
                num_vectors=args.num_vectors,
                warmup_runs=args.warmup_runs,
                measurement_runs=args.measurement_runs,
            )

        if "rust" in benchmarks:
            bench.benchmark_rust_int8_throughput(
                dimensions=args.dimensions,
                num_vectors=min(args.num_vectors, 50_000),
                warmup_runs=args.warmup_runs,
                measurement_runs=args.measurement_runs,
            )

    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")

    # Print summary and save results
    bench.print_summary()
    bench.save_results()


if __name__ == "__main__":
    main()
