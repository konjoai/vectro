"""Benchmark harness for Vectro — captures throughput, compression ratio,
and quality metrics across configurations and emits structured JSON/CSV reports.

Usage
-----
CLI::

    python -m python.benchmark --n 5000 --dim 384 --profiles fast balanced quality \
        --output results.json

Python API::

    from python.benchmark import BenchmarkSuite
    suite = BenchmarkSuite(n=5000, dim=384)
    report = suite.run()
    report.save("results.json")
    report.print_summary()
"""

from __future__ import annotations

import csv
import io
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkEntry:
    """Single benchmark data point."""

    profile: str
    n_vectors: int
    vector_dim: int
    precision_mode: str
    backend: str

    # Throughput
    throughput_vps: float  # vectors / second (median over trials)
    throughput_mbs: float  # MB / second (float32 input)

    # Compression
    compression_ratio: float
    compressed_mb: float
    original_mb: float

    # Quality
    mean_cosine_sim: float
    mean_absolute_error: float

    # Timing
    median_latency_ms: float
    p95_latency_ms: float

    # Environment
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    numpy_version: str = field(default_factory=lambda: np.__version__)


@dataclass
class BenchmarkReport:
    """Collection of benchmark entries with serialisation helpers."""

    entries: List[BenchmarkEntry]
    generated_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    vectro_version: str = "1.2.0"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "vectro_version": self.vectro_version,
            "generated_at": self.generated_at,
            "entries": [asdict(e) for e in self.entries],
        }

    def save(self, path: Union[str, Path], fmt: Optional[str] = None) -> None:
        """Save to *path*.  Format is inferred from file extension unless *fmt*
        is explicitly ``"json"`` or ``"csv"``."""
        path = Path(path)
        fmt = fmt or path.suffix.lstrip(".").lower() or "json"
        if fmt == "json":
            path.write_text(json.dumps(self.to_dict(), indent=2))
        elif fmt == "csv":
            self._save_csv(path)
        else:
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'json' or 'csv'.")

    def _save_csv(self, path: Path) -> None:
        if not self.entries:
            path.write_text("")
            return
        fields = list(asdict(self.entries[0]).keys())
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(asdict(entry))

    def to_csv_string(self) -> str:
        buf = io.StringIO()
        if not self.entries:
            return ""
        fields = list(asdict(self.entries[0]).keys())
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for entry in self.entries:
            writer.writerow(asdict(entry))
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a compact table of results to stdout."""
        header = (
            f"{'Profile':>10}  {'n×d':>10}  {'prec':>6}  {'kVec/s':>8}  {'ratio':>6}  {'cosSim':>8}"
        )
        print("=" * len(header))
        print(f"Vectro Benchmark  ({self.generated_at})")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for e in self.entries:
            print(
                f"{e.profile:>10}  "
                f"{e.n_vectors}×{e.vector_dim:>5}  "
                f"{e.precision_mode:>6}  "
                f"{e.throughput_vps / 1000:>7.1f}k  "
                f"{e.compression_ratio:>5.2f}×  "
                f"{e.mean_cosine_sim:>8.5f}"
            )
        print("=" * len(header))


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Run a configurable set of throughput, compression, and quality benchmarks.

    Args:
        n: Number of vectors per benchmark configuration.
        dim: Vector dimension.
        profiles: List of compression profiles to benchmark.
        trials: Number of timed repetitions per configuration (median is kept).
        seed: NumPy random seed for reproducible test data.
        backend: Quantization backend (``"auto"`` = fastest available).
    """

    def __init__(
        self,
        n: int = 2000,
        dim: int = 384,
        profiles: Optional[List[str]] = None,
        trials: int = 5,
        seed: int = 42,
        backend: str = "auto",
    ):
        self.n = n
        self.dim = dim
        self.profiles = profiles or ["fast", "balanced", "quality"]
        self.trials = trials
        self.seed = seed
        self.backend = backend

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_data(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.standard_normal((self.n, self.dim)).astype(np.float32)

    def _run_one(self, vectors: np.ndarray, profile: str) -> BenchmarkEntry:
        # Import here to avoid circular imports at module load time
        from .vectro import Vectro
        from .interface import get_backend_info

        vectro = Vectro(backend=self.backend)
        n, dim = vectors.shape
        original_mb = vectors.nbytes / (1024 * 1024)

        latencies: List[float] = []
        last_result = None
        for _ in range(self.trials):
            t0 = time.perf_counter()
            last_result = vectro.compress(vectors, profile=profile)
            latencies.append(time.perf_counter() - t0)

        latencies.sort()
        median_lat = latencies[self.trials // 2]
        p95_lat = latencies[int(self.trials * 0.95)]

        throughput_vps = n / median_lat
        throughput_mbs = original_mb / median_lat

        # Compression
        result = last_result
        compression_ratio = float(result.compression_ratio)
        compressed_mb = original_mb / compression_ratio

        # Quality
        reconstructed = vectro.decompress(result)
        dot = np.sum(vectors * reconstructed, axis=1)
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(reconstructed, axis=1)
        mean_cosine_sim = float(np.mean(dot / (norms + 1e-10)))
        mean_absolute_error = float(np.mean(np.abs(vectors - reconstructed)))

        # Backend detection
        binfo = get_backend_info()
        active_backend = (
            "squish_quant_rust"
            if binfo.get("squish_quant_rust")
            else "cython"
            if binfo.get("cython")
            else "numpy"
        )

        prec = getattr(result, "precision_mode", "int8")

        return BenchmarkEntry(
            profile=profile,
            n_vectors=n,
            vector_dim=dim,
            precision_mode=prec,
            backend=active_backend,
            throughput_vps=throughput_vps,
            throughput_mbs=throughput_mbs,
            compression_ratio=compression_ratio,
            compressed_mb=compressed_mb,
            original_mb=original_mb,
            mean_cosine_sim=mean_cosine_sim,
            mean_absolute_error=mean_absolute_error,
            median_latency_ms=median_lat * 1000,
            p95_latency_ms=p95_lat * 1000,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> BenchmarkReport:
        """Execute all configured benchmarks and return a :class:`BenchmarkReport`."""
        vectors = self._make_data()
        entries: List[BenchmarkEntry] = []
        for profile in self.profiles:
            entries.append(self._run_one(vectors, profile))
        return BenchmarkReport(entries=entries)


# ---------------------------------------------------------------------------
# CLI  (python -m python.benchmark)
# ---------------------------------------------------------------------------


def _main(argv: Optional[List[str]] = None) -> None:  # noqa: D401
    import argparse

    parser = argparse.ArgumentParser(description="Vectro benchmark harness")
    parser.add_argument(
        "--n", type=int, default=2000, metavar="N", help="Number of vectors (default: 2000)"
    )
    parser.add_argument(
        "--dim", type=int, default=384, metavar="D", help="Vector dimension (default: 384)"
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["fast", "balanced", "quality"],
        metavar="PROFILE",
        help="Profiles to benchmark",
    )
    parser.add_argument(
        "--trials", type=int, default=5, metavar="T", help="Timing trials per config (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save report to FILE (.json or .csv)",
    )
    parser.add_argument("--backend", default="auto", metavar="BACKEND", help="Quantization backend")

    args = parser.parse_args(argv)
    suite = BenchmarkSuite(
        n=args.n,
        dim=args.dim,
        profiles=args.profiles,
        trials=args.trials,
        backend=args.backend,
    )
    report = suite.run()
    report.print_summary()
    if args.output:
        report.save(args.output)
        print(f"\nSaved report to: {args.output}")


if __name__ == "__main__":
    _main()
