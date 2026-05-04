#!/usr/bin/env python3
"""vectro_paper_benchmark.py — reproducible bench harness for the v5.0.x paper.

This is the script referenced by:

  * pyproject.toml ``[tool.cibuildwheel].test-command``
  * reproduce_paper.sh + reproduce_paper.ps1
  * .github/workflows/bench-cross-platform.yml

It calls the **real Vectro library** at multiple (n × d) shapes and three
quantisation tables (INT8, NF4, binary) and emits one of two outputs:

  * ``--json``  — a single-line JSON object whose ``throughput`` field is
                  the best-of-N M vec/s for the requested table; consumed
                  by ``reproduce_paper.{sh,ps1}``.
  * (default)   — a human-readable text table.

Standard-library + numpy + the in-tree ``python`` package only.  No
external bench frameworks; warmup + best-of-N timing per shape per
table.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import python as vectro  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Shapes
# ─────────────────────────────────────────────────────────────────────────

# Quick mode runs only the (n=10k, d=768) shape — single sample sufficient
# for a CI wheel-test smoke check that exercises every code path.
QUICK_SHAPES: List[Tuple[int, int]] = [(10_000, 768)]

# Full mode covers the headline d values from the paper appendix.
FULL_SHAPES: List[Tuple[int, int]] = [
    (10_000,  128),
    (10_000,  384),
    (10_000,  768),
    (10_000, 1024),
    (10_000, 1536),
    (50_000,  768),
]

TABLES: Tuple[str, ...] = ("int8", "nf4", "binary")

# Map table name → Vectro compression profile.  INT8 ships under both
# the ``fast`` and ``balanced`` profiles; we use ``balanced`` as the
# headline, matching v5.0.0's PLAN.md throughput targets.
PROFILE_BY_TABLE = {
    "int8":   "balanced",
    "nf4":    "quality",
    "binary": "binary",
}


# ─────────────────────────────────────────────────────────────────────────
# Synthetic dataset
# ─────────────────────────────────────────────────────────────────────────

def _make_unit_vectors(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Deterministic L2-normalised f32 matrix.  Same input on every host
    so timing comparisons are like-for-like.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (raw / norms).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Per-table measurement
# ─────────────────────────────────────────────────────────────────────────

def _cosine_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1) + 1e-12
    nb = np.linalg.norm(b, axis=1) + 1e-12
    return np.einsum("ij,ij->i", a, b) / (na * nb)


def _bench_one(
    table: str,
    n: int,
    d: int,
    reps: int,
    warmup: int,
    vec: vectro.Vectro,
    data: np.ndarray,
) -> Dict[str, Any]:
    """Run ``reps`` timed encodes of *data* under *table*.

    Returns the raw timings + best/median throughput + reconstruction
    cosine + memory before/after.
    """
    profile = PROFILE_BY_TABLE[table]

    # Warmup so JIT / page faults / first-touch caches don't pollute
    # the first sample.
    for _ in range(warmup):
        vec.compress(data, profile=profile)

    samples_ms: List[float] = []
    last_result = None
    for _ in range(reps):
        t0 = time.perf_counter()
        last_result = vec.compress(data, profile=profile)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    assert last_result is not None
    best_ms = min(samples_ms)
    p50_ms = statistics.median(samples_ms)
    best_tput = (n / best_ms) * 1000.0
    p50_tput  = (n / p50_ms)  * 1000.0

    # Reconstruction quality — single mean over the whole batch.
    try:
        recon = last_result.reconstruct_batch()
        cos = float(np.mean(_cosine_per_row(data, recon)))
    except Exception:
        cos = float("nan")

    original_bytes   = int(data.nbytes)
    compressed_bytes = int(last_result.total_compressed_bytes)
    ratio            = float(last_result.compression_ratio)

    return {
        "table":            table,
        "profile":          profile,
        "n":                int(n),
        "d":                int(d),
        "reps":             int(reps),
        "warmup":           int(warmup),
        "samples_ms":       [round(s, 4) for s in samples_ms],
        "best_ms":          round(best_ms, 4),
        "p50_ms":           round(p50_ms,  4),
        "best_throughput_vec_s":   round(best_tput, 1),
        "p50_throughput_vec_s":    round(p50_tput,  1),
        "best_M_vec_s":     round(best_tput / 1.0e6, 4),
        "p50_M_vec_s":      round(p50_tput  / 1.0e6, 4),
        "original_bytes":   original_bytes,
        "compressed_bytes": compressed_bytes,
        "ratio":            round(ratio, 4),
        "mean_cosine":      round(cos, 6),
    }


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────

def run(
    table: str,
    quick: bool,
    n_override: Optional[int],
    d_override: Optional[int],
    reps: int,
    warmup: int,
) -> Dict[str, Any]:
    """Run the bench and return a dict suitable for JSON dump or pretty-print.

    Always returns a top-level ``throughput`` field (M vec/s) so the
    POSIX / PowerShell ``reproduce_paper`` parsers can extract a single
    headline number — they read whatever the user picked with ``--table``
    at the headline shape (n=10k d=768).
    """
    if table == "all":
        tables = list(TABLES)
    else:
        if table not in TABLES:
            raise SystemExit(f"unknown --table {table!r}; choose from {TABLES} or 'all'")
        tables = [table]

    shapes: List[Tuple[int, int]]
    if n_override is not None or d_override is not None:
        n = int(n_override) if n_override is not None else 10_000
        d = int(d_override) if d_override is not None else 768
        shapes = [(n, d)]
    else:
        shapes = list(QUICK_SHAPES if quick else FULL_SHAPES)

    vec = vectro.Vectro()
    rows: List[Dict[str, Any]] = []
    cache: Dict[Tuple[int, int], np.ndarray] = {}

    for (n, d) in shapes:
        if (n, d) not in cache:
            cache[(n, d)] = _make_unit_vectors(n, d)
        data = cache[(n, d)]
        for tbl in tables:
            row = _bench_one(tbl, n, d, reps=reps, warmup=warmup, vec=vec, data=data)
            rows.append(row)

    # Pick the headline number for downstream parsers: prefer
    # n=10_000, d=768 if present, else the first row.
    headline: Optional[Dict[str, Any]] = None
    for r in rows:
        if r["n"] == 10_000 and r["d"] == 768 and (
            table == "all" or r["table"] == table
        ):
            headline = r
            break
    if headline is None:
        headline = rows[0]

    record: Dict[str, Any] = {
        "schema":          "vectro/paper-benchmark/v1",
        "version":         vectro.__version__,
        "platform":        f"{platform.system()} / {platform.machine()}",
        "python":          platform.python_version(),
        "table":           table,
        "quick":           bool(quick),
        "rows":            rows,
        # ─── headline fields (consumed by reproduce_paper.{sh,ps1}) ───
        "throughput":      headline["best_M_vec_s"],
        "throughput_unit": "M vec/s",
        "headline_table":  headline["table"],
        "headline_n":      headline["n"],
        "headline_d":      headline["d"],
    }
    return record


# ─────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────

def _print_table(record: Dict[str, Any]) -> None:
    print()
    print(f"  Vectro paper benchmark — v{record['version']} on {record['platform']}")
    print(f"  Python {record['python']}  ·  {len(record['rows'])} measurements")
    print()
    cols = ["table", "n", "d", "best M/s", "p50 M/s", "ratio", "cosine", "best ms", "p50 ms"]
    widths = [9, 8, 6, 9, 9, 7, 7, 9, 9]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    print("  " + "  ".join("-" * w for w in widths))
    for r in record["rows"]:
        print(fmt.format(
            r["table"],
            f"{r['n']:,}",
            r["d"],
            f"{r['best_M_vec_s']:.3f}",
            f"{r['p50_M_vec_s']:.3f}",
            f"{r['ratio']:.2f}×",
            f"{r['mean_cosine']:.4f}",
            f"{r['best_ms']:.2f}",
            f"{r['p50_ms']:.2f}",
        ))
    print()
    print(f"  Headline ({record['headline_table']} @ n={record['headline_n']:,} "
          f"× d={record['headline_d']}):  {record['throughput']:.3f} M vec/s")
    print()


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Vectro paper benchmark — measures real INT8 / NF4 / binary "
                    "throughput on the host CPU.",
    )
    ap.add_argument("--quick", action="store_true",
                    help="Single shape (10k × 768) — for CI wheel-test smoke checks.")
    ap.add_argument("--table", default="int8",
                    help="One of: int8, nf4, binary, all  (default: int8)")
    ap.add_argument("--json", action="store_true",
                    help="Emit a single-line JSON record (consumed by reproduce_paper).")
    ap.add_argument("--n", type=int, default=None,
                    help="Override the number of vectors for a one-shape run.")
    ap.add_argument("--d", type=int, default=None,
                    help="Override the dimension for a one-shape run.")
    ap.add_argument("--reps", type=int, default=3,
                    help="Timed repetitions per shape per table (default: 3).")
    ap.add_argument("--warmup", type=int, default=1,
                    help="Warmup repetitions (default: 1).")
    args = ap.parse_args(argv)

    record = run(
        table=args.table.lower(),
        quick=args.quick,
        n_override=args.n,
        d_override=args.d,
        reps=max(1, args.reps),
        warmup=max(0, args.warmup),
    )

    if args.json:
        print(json.dumps(record))
    else:
        _print_table(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
