#!/usr/bin/env python3
"""Recall–QPS Pareto benchmark for HNSW: vectro vs hnswlib vs FAISS.

Why this exists
---------------
A single ``ef`` operating point is a misleading way to compare ANN indexes:
at a fixed ``ef`` two indexes reach *different* recall, so comparing their QPS
is apples-to-oranges. The field-standard methodology (ann-benchmarks) sweeps
``ef`` and compares **QPS at iso-recall** along the recall–QPS Pareto frontier.

This script does exactly that, with Konjo measurement rigor:

* median-of-N timed runs after warmup (``--reps`` / ``--warmup``),
* recall@k against an exact brute-force ground truth,
* full hardware + config metadata in the JSON output,
* timestamped results under ``benchmarks/results/`` (never overwritten),
* honest handling of absent competitors (``hnswlib`` / ``faiss`` optional).

Vectro is measured through its **batch** API (``HNSWIndex.search_batch``), the
rayon-parallel path; hnswlib and FAISS use their native multi-threaded batch
search — same threading model for all three.

Usage
-----
    python benchmarks/benchmark_hnsw_pareto.py \
        --hdf5 data/glove-100-angular.hdf5 --n 50000 --q 1000 \
        --ef 20,40,60,100,150,200
    # or synthetic:
    python benchmarks/benchmark_hnsw_pareto.py --n 50000 --d 100 --q 1000
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Allow `import python.*` when run as a script (repo root, not benchmarks/, on path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _unit(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (so inner product == cosine similarity)."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def load_data(
    hdf5: Optional[str], n: int, d: int, q: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train, queries)`` from an ann-benchmarks HDF5 file or synthetic.

    The HDF5 ``train``/``test`` datasets follow the ann-benchmarks layout.
    """
    if hdf5:
        import h5py  # type: ignore[import]

        with h5py.File(hdf5, "r") as f:
            train = f["train"][:n].astype(np.float32)
            queries = f["test"][:q].astype(np.float32)
        return train, queries
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n, d)).astype(np.float32),
        rng.standard_normal((q, d)).astype(np.float32),
    )


def brute_force_gt(train_u: np.ndarray, queries_u: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k cosine neighbours (both inputs already unit-normalised)."""
    gt = np.empty((len(queries_u), k), dtype=np.int64)
    for start in range(0, len(queries_u), 256):
        end = min(start + 256, len(queries_u))
        sims = queries_u[start:end] @ train_u.T
        gt[start:end] = np.argsort(-sims, axis=1)[:, :k]
    return gt


def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Mean fraction of each query's true top-k that appears in the prediction."""
    return float(np.mean([len(set(pred[i, :k]) & set(gt[i, :k])) / k for i in range(len(gt))]))


def _median_qps(
    query_fn: Callable[[], np.ndarray], n_queries: int, reps: int, warmup: int
) -> float:
    for _ in range(warmup):
        query_fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        query_fn()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return n_queries / med if med > 0 else 0.0


# ─────────────────────────── per-library sweeps ───────────────────────────


def sweep_vectro(
    train: np.ndarray,
    queries_u: np.ndarray,
    gt: np.ndarray,
    m: int,
    efc: int,
    k: int,
    efs: list[int],
    reps: int,
    warmup: int,
) -> Optional[list[dict[str, float]]]:
    try:
        from python.hnsw_api import HNSWIndex  # type: ignore[import]
    except ImportError:
        return None
    idx = HNSWIndex(M=m, ef_construction=efc, space="cosine")
    idx.add(train)
    out = []
    for ef in efs:
        pred, _ = idx.search_batch(queries_u, k=k, ef=ef)
        qps = _median_qps(
            lambda ef=ef: idx.search_batch(queries_u, k=k, ef=ef)[0], len(queries_u), reps, warmup
        )
        out.append({"ef": ef, "recall": recall_at_k(pred, gt, k), "qps": qps})
    return out


def sweep_hnswlib(
    train_u: np.ndarray,
    queries_u: np.ndarray,
    gt: np.ndarray,
    m: int,
    efc: int,
    k: int,
    efs: list[int],
    reps: int,
    warmup: int,
) -> Optional[list[dict[str, float]]]:
    try:
        import hnswlib  # type: ignore[import]
    except ImportError:
        return None
    idx = hnswlib.Index(space="cosine", dim=train_u.shape[1])
    idx.init_index(max_elements=len(train_u), ef_construction=efc, M=m)
    idx.add_items(train_u, np.arange(len(train_u)))
    out = []
    for ef in efs:
        idx.set_ef(ef)
        labels, _ = idx.knn_query(queries_u, k=k)  # native multi-threaded batch
        qps = _median_qps(lambda: idx.knn_query(queries_u, k=k)[0], len(queries_u), reps, warmup)
        out.append({"ef": ef, "recall": recall_at_k(np.asarray(labels), gt, k), "qps": qps})
    return out


def sweep_faiss(
    train_u: np.ndarray,
    queries_u: np.ndarray,
    gt: np.ndarray,
    m: int,
    efc: int,
    k: int,
    efs: list[int],
    reps: int,
    warmup: int,
) -> Optional[list[dict[str, float]]]:
    try:
        import faiss  # type: ignore[import]
    except ImportError:
        return None
    idx = faiss.IndexHNSWFlat(train_u.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = efc
    idx.add(train_u)
    out = []
    for ef in efs:
        idx.hnsw.efSearch = ef
        _, labels = idx.search(queries_u, k)
        qps = _median_qps(lambda: idx.search(queries_u, k)[1], len(queries_u), reps, warmup)
        out.append({"ef": ef, "recall": recall_at_k(np.asarray(labels), gt, k), "qps": qps})
    return out


def qps_at_recall(curve: list[dict[str, float]], target: float) -> Optional[float]:
    """Linear-interpolate QPS at a target recall along the (recall, qps) curve."""
    pts = sorted(curve, key=lambda p: p["recall"])
    if not pts or target < pts[0]["recall"] or target > pts[-1]["recall"]:
        return None
    for lo, hi in zip(pts, pts[1:]):
        if lo["recall"] <= target <= hi["recall"]:
            span = hi["recall"] - lo["recall"]
            if span <= 0:
                return lo["qps"]
            w = (target - lo["recall"]) / span
            return lo["qps"] + w * (hi["qps"] - lo["qps"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="HNSW recall–QPS Pareto: vectro vs hnswlib vs FAISS")
    ap.add_argument(
        "--hdf5", type=str, default=None, help="ann-benchmarks HDF5 dataset (else synthetic)"
    )
    ap.add_argument("--n", type=int, default=50_000, help="corpus size")
    ap.add_argument("--d", type=int, default=100, help="dim (synthetic only)")
    ap.add_argument("--q", type=int, default=1_000, help="query count")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=16, help="HNSW M")
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument(
        "--ef", type=str, default="20,40,60,100,150,200", help="comma-separated ef_search sweep"
    )
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    efs = [int(x) for x in args.ef.split(",")]
    train, queries = load_data(args.hdf5, args.n, args.d, args.q, seed=42)
    train_u, queries_u = _unit(train), _unit(queries)
    gt = brute_force_gt(train_u, queries_u, args.k)

    print("=" * 68)
    print(
        f"HNSW recall–QPS Pareto  n={len(train):,} d={train.shape[1]} q={len(queries):,} "
        f"k={args.k} M={args.m} ef_c={args.ef_construction}"
    )
    print(f"reps={args.reps} (median) warmup={args.warmup}  ef sweep={efs}")
    print("=" * 68)

    # vectro is fed raw vectors (it normalises internally); the C++ libs are fed
    # pre-normalised vectors so all three use cosine via inner product.
    results: dict[str, Any] = {}
    libs = {
        "vectro": sweep_vectro(
            train, queries_u, gt, args.m, args.ef_construction, args.k, efs, args.reps, args.warmup
        ),
        "hnswlib": sweep_hnswlib(
            train_u,
            queries_u,
            gt,
            args.m,
            args.ef_construction,
            args.k,
            efs,
            args.reps,
            args.warmup,
        ),
        "faiss": sweep_faiss(
            train_u,
            queries_u,
            gt,
            args.m,
            args.ef_construction,
            args.k,
            efs,
            args.reps,
            args.warmup,
        ),
    }
    for name, curve in libs.items():
        if curve is None:
            print(f"\n{name}: not installed")
            continue
        results[name] = curve
        print(f"\n{name}:")
        print(f"  {'ef':>4} {'recall@' + str(args.k):>10} {'QPS':>10}")
        for p in curve:
            print(f"  {p['ef']:>4} {p['recall']:>10.4f} {p['qps']:>10,.0f}")

    # Iso-recall comparison: QPS at common recall targets (vs vectro).
    present = {n: c for n, c in libs.items() if c}
    if "vectro" in present and len(present) > 1:
        print("\n── QPS at iso-recall (interpolated; ratio = vectro / competitor) ──")
        print(f"  {'recall':>8}" + "".join(f"{n:>12}" for n in present))
        for tgt in (0.85, 0.90, 0.93, 0.95):
            row = {n: qps_at_recall(c, tgt) for n, c in present.items()}
            cells = "".join(f"{(f'{row[n]:,.0f}' if row[n] else '—'):>12}" for n in present)
            print(f"  {tgt:>8.2f}{cells}")

    payload = {
        "benchmark": "hnsw_recall_qps_pareto",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n": len(train),
            "d": int(train.shape[1]),
            "q": len(queries),
            "k": args.k,
            "m": args.m,
            "ef_construction": args.ef_construction,
            "ef_sweep": efs,
            "reps": args.reps,
            "warmup": args.warmup,
            "dataset": args.hdf5 or "synthetic-gaussian(seed=42)",
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "results": results,
        "note": "Shared/cloud hosts add timing noise; ratios at iso-recall are the durable signal.",
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.output) if args.output else Path(f"benchmarks/results/hnsw_pareto_{ts}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
