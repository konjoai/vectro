#!/usr/bin/env python3
"""IVF-PQ at-scale benchmark — vectro's compression *fits the machine*.

The point of this benchmark is not to win a microsecond race at 50K vectors;
it is to show the regime where vectro structurally wins: **large corpora that
don't fit in RAM as float32**. At 100M×768, raw float32 is ~307 GB; the same
corpus as IVF-PQ codes (M=96 sub-quantisers, 1 byte each) is ~9.6 GB — a 32×
reduction that turns an impossible single-machine workload into a routine one.

This harness builds a vectro IVF-PQ index at a parametrised scale and reports,
with Konjo rigor:

* build time (train on a sample + encode all),
* **measured process RSS** (peak during build, steady-state after) and the
  analytic footprint vs a float32-flat / HNSW-float32 baseline,
* recall@k vs exact brute-force ground truth (computed in chunks; capped at
  ``--recall-cap`` so it stays tractable),
* QPS across an ``n_probe`` sweep,
* a projection of the memory model to 100M and 1B,
* optional FAISS ``IndexIVFPQ`` comparison (``--faiss``).

Results (timestamped JSON + full hw/config metadata) go to
``benchmarks/results/`` and are never overwritten.

Memory note: building needs the float32 corpus resident transiently (n·d·4
bytes). On a small box use ``--d 128`` and a few million rows; the analytic
projection covers 100M/1B which need a big-RAM or streaming build.

Usage
-----
    python benchmarks/benchmark_ivfpq_scale.py --n 1000000 --d 128
    python benchmarks/benchmark_ivfpq_scale.py --n 5000000 --d 128 --faiss
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────── memory helpers ───────────────────────────────


def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB (Linux KiB / macOS bytes)."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    scale = 1024 if sys.platform == "darwin" else 1024 * 1024
    return maxrss / scale


def current_rss_gb() -> float:
    """Current resident set size, in GB (reads /proc on Linux; else 0)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * resource.getpagesize() / 1e9
    except (OSError, IndexError, ValueError):
        return 0.0


# ─────────────────────────── data + ground truth ──────────────────────────


def load_hdf5(path: str, n: int, d: int) -> tuple[np.ndarray, int]:
    """Load+unit-normalise the first ``n`` rows of an ann-benchmarks ``train`` set."""
    import h5py  # type: ignore[import]

    with h5py.File(path, "r") as f:
        x = f["train"][:n].astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x, x.shape[1]


def make_clustered(n: int, d: int, n_clusters: int, seed: int) -> np.ndarray:
    """Unit-norm clustered vectors (realistic ANN structure, not pure noise).

    Generated in row blocks to avoid a second n·d float32 copy at peak.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32)
    out = np.empty((n, d), dtype=np.float32)
    block = 1_000_000
    for s in range(0, n, block):
        e = min(s + block, n)
        lbl = rng.integers(0, n_clusters, size=e - s)
        out[s:e] = centers[lbl] + 0.15 * rng.standard_normal((e - s, d)).astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out


def brute_force_gt(
    corpus_u: np.ndarray, queries_u: np.ndarray, k: int, chunk: int = 200_000
) -> np.ndarray:
    """Exact top-k cosine neighbours, computed in corpus chunks (bounded memory)."""
    q = len(queries_u)
    best_d = np.full((q, k), -np.inf, dtype=np.float32)
    best_i = np.full((q, k), -1, dtype=np.int64)
    for s in range(0, len(corpus_u), chunk):
        e = min(s + chunk, len(corpus_u))
        sims = queries_u @ corpus_u[s:e].T  # (q, b)
        cat_d = np.concatenate([best_d, sims], axis=1)
        cat_i = np.concatenate([best_i, np.arange(s, e)[None, :].repeat(q, axis=0)], axis=1)
        top = np.argpartition(-cat_d, k - 1, axis=1)[:, :k]
        rows = np.arange(q)[:, None]
        best_d = cat_d[rows, top]
        best_i = cat_i[rows, top]
    order = np.argsort(-best_d, axis=1)
    return best_i[np.arange(q)[:, None], order]


def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(pred[i, :k]) & set(gt[i, :k])) / k for i in range(len(gt))]))


# ─────────────────────────── footprint model ──────────────────────────────


def footprint_model(
    n: int, d: int, m_sub: int, n_lists: int, k_cent: int, hnsw_m: int
) -> dict[str, float]:
    """Analytic byte footprints (GB) for the corpus under each representation."""
    fp32_flat = n * d * 4
    hnsw_fp32 = fp32_flat + n * hnsw_m * 2 * 4  # vectors + ~2M int32 graph links/node
    ivfpq_codes = n * m_sub  # 1 byte per sub-quantiser
    centroids = n_lists * d * 4 + m_sub * k_cent * (d // m_sub) * 4
    ivfpq_total = ivfpq_codes + centroids + n * 8  # + id map
    return {
        "fp32_flat_gb": fp32_flat / 1e9,
        "hnsw_fp32_gb": hnsw_fp32 / 1e9,
        "ivfpq_gb": ivfpq_total / 1e9,
        "compression_x": fp32_flat / ivfpq_total if ivfpq_total else 0.0,
    }


# ─────────────────────────── benchmark core ───────────────────────────────


def run_vectro(
    corpus_u: np.ndarray,
    queries_u: np.ndarray,
    gt: Optional[np.ndarray],
    k: int,
    n_lists: int,
    m_sub: int,
    k_cent: int,
    probes: list[int],
    train_sample: int,
    reps: int,
) -> dict[str, Any]:
    import vectro_py  # type: ignore[import]

    n = len(corpus_u)
    idx = vectro_py.PyIvfPqIndex(n_lists, probes[0])
    sample = corpus_u[: min(train_sample, n)]
    t0 = time.perf_counter()
    idx.train_np(sample, m_sub, k_cent, 15, 0)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx.add_np(corpus_u)
    add_s = time.perf_counter() - t0
    peak = peak_rss_gb()

    sweep = []
    qlist = [list(map(float, queries_u[i])) for i in range(len(queries_u))]
    for npb in probes:
        # warmup
        for qq in qlist[: min(50, len(qlist))]:
            idx.search_with_probe(qq, k, npb)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            preds = [idx.search_with_probe(qq, k, npb) for qq in qlist]
            times.append(time.perf_counter() - t0)
        qps = len(qlist) / statistics.median(times)
        rec = None
        if gt is not None:
            pred = np.array(
                [[i for i, _ in r] + [-1] * (k - len(r)) for r in preds], dtype=np.int64
            )
            rec = recall_at_k(pred, gt, k)
        sweep.append({"n_probe": npb, "qps": qps, "recall": rec})
    return {"train_s": train_s, "add_s": add_s, "peak_rss_gb": peak, "sweep": sweep}


def run_faiss(
    corpus_u: np.ndarray,
    queries_u: np.ndarray,
    gt: Optional[np.ndarray],
    k: int,
    n_lists: int,
    m_sub: int,
    probes: list[int],
    train_sample: int,
    reps: int,
) -> Optional[dict[str, Any]]:
    try:
        import faiss  # type: ignore[import]
    except ImportError:
        return None
    d = corpus_u.shape[1]
    quant = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFPQ(quant, d, n_lists, m_sub, 8, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    idx.train(corpus_u[: min(train_sample, len(corpus_u))])
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx.add(corpus_u)
    add_s = time.perf_counter() - t0
    sweep = []
    for npb in probes:
        idx.nprobe = npb
        idx.search(queries_u, k)  # warmup
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _, labels = idx.search(queries_u, k)
            times.append(time.perf_counter() - t0)
        qps = len(queries_u) / statistics.median(times)
        rec = recall_at_k(np.asarray(labels), gt, k) if gt is not None else None
        sweep.append({"n_probe": npb, "qps": qps, "recall": rec})
    return {"train_s": train_s, "add_s": add_s, "sweep": sweep}


def main() -> None:
    ap = argparse.ArgumentParser(description="IVF-PQ at-scale benchmark (memory + recall + QPS)")
    ap.add_argument(
        "--hdf5", type=str, default=None, help="ann-benchmarks HDF5 (real data; else synthetic)"
    )
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--q", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-lists", type=int, default=0, help="IVF lists (default ≈ sqrt(n)·4)")
    ap.add_argument("--m", type=int, default=16, help="PQ sub-quantisers (d must divide m)")
    ap.add_argument("--k-cent", type=int, default=256, help="centroids per sub-quantiser")
    ap.add_argument("--n-probe", type=str, default="8,16,32,64")
    ap.add_argument("--clusters", type=int, default=0, help="synthetic clusters (default n//500)")
    ap.add_argument("--train-sample", type=int, default=200_000)
    ap.add_argument("--recall-cap", type=int, default=2_000_000, help="skip exact GT above this n")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--hnsw-m", type=int, default=16)
    ap.add_argument("--faiss", action="store_true")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    if args.d % args.m != 0:
        ap.error(f"d ({args.d}) must be divisible by m ({args.m})")
    n_lists = args.n_lists or max(64, int(math.isqrt(args.n) * 4))
    probes = [int(x) for x in args.n_probe.split(",")]
    clusters = args.clusters or max(16, args.n // 500)

    print("=" * 70)
    print(f"IVF-PQ at-scale  n={args.n:,} d={args.d} q={args.q} k={args.k}")
    print(f"n_lists={n_lists} M={args.m} K={args.k_cent} n_probe={probes} clusters={clusters:,}")
    print("=" * 70)

    fp = footprint_model(args.n, args.d, args.m, n_lists, args.k_cent, args.hnsw_m)
    print("\nFootprint (analytic):")
    print(f"  float32 flat      : {fp['fp32_flat_gb']:9.2f} GB")
    print(f"  HNSW (float32)    : {fp['hnsw_fp32_gb']:9.2f} GB")
    print(f"  vectro IVF-PQ     : {fp['ivfpq_gb']:9.2f} GB   ({fp['compression_x']:.0f}× smaller)")

    if args.hdf5:
        print(f"\nLoading {args.hdf5}…", flush=True)
        corpus_u, real_d = load_hdf5(args.hdf5, args.n, args.d)
        if real_d != args.d:
            ap.error(f"--d {args.d} != dataset dim {real_d}; pass --d {real_d}")
    else:
        print("\nGenerating clustered corpus…", flush=True)
        corpus_u = make_clustered(args.n, args.d, clusters, seed=42)
    queries_u = corpus_u[: args.q].copy()

    gt = None
    if args.n <= args.recall_cap:
        print("Computing exact ground truth (chunked)…", flush=True)
        gt = brute_force_gt(corpus_u, queries_u, args.k)
    else:
        print(f"n > recall-cap ({args.recall_cap:,}); skipping exact recall.")

    print("\nBuilding vectro IVF-PQ…", flush=True)
    vec = run_vectro(
        corpus_u,
        queries_u,
        gt,
        args.k,
        n_lists,
        args.m,
        args.k_cent,
        probes,
        args.train_sample,
        args.reps,
    )
    print(
        f"  train={vec['train_s']:.1f}s  add={vec['add_s']:.1f}s  peak_rss={vec['peak_rss_gb']:.2f} GB"
    )
    print(f"  {'n_probe':>7} {'recall@' + str(args.k):>10} {'QPS':>10}")
    for s in vec["sweep"]:
        r = f"{s['recall']:.4f}" if s["recall"] is not None else "—"
        print(f"  {s['n_probe']:>7} {r:>10} {s['qps']:>10,.0f}")

    faiss_res = None
    if args.faiss:
        print("\nBuilding FAISS IndexIVFPQ…", flush=True)
        faiss_res = run_faiss(
            corpus_u, queries_u, gt, args.k, n_lists, args.m, probes, args.train_sample, args.reps
        )
        if faiss_res is None:
            print("  faiss not installed")
        else:
            print(f"  train={faiss_res['train_s']:.1f}s  add={faiss_res['add_s']:.1f}s")
            for s in faiss_res["sweep"]:
                r = f"{s['recall']:.4f}" if s["recall"] is not None else "—"
                print(f"  n_probe={s['n_probe']:>4}  recall={r}  QPS={s['qps']:,.0f}")

    # Projections of the memory model to the headline scales.
    proj = {
        f"{tag}": footprint_model(nn, 768, 96, max(64, int(math.isqrt(nn) * 4)), 256, args.hnsw_m)
        for tag, nn in (("100M_d768", 100_000_000), ("1B_d768", 1_000_000_000))
    }
    print("\nProjection at d=768, M=96 (the headline regime):")
    for tag, f in proj.items():
        print(
            f"  {tag:>10}: float32 {f['fp32_flat_gb']:8.0f} GB  →  IVF-PQ {f['ivfpq_gb']:7.1f} GB"
            f"  ({f['compression_x']:.0f}×)"
        )

    payload = {
        "benchmark": "ivfpq_at_scale",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n": args.n,
            "d": args.d,
            "q": args.q,
            "k": args.k,
            "n_lists": n_lists,
            "m": args.m,
            "k_cent": args.k_cent,
            "n_probe": probes,
            "clusters": clusters,
            "train_sample": args.train_sample,
            "reps": args.reps,
        },
        "footprint_gb": fp,
        "vectro": vec,
        "faiss": faiss_res,
        "projection_d768_m96": proj,
        "current_rss_gb": current_rss_gb(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpus": __import__("os").cpu_count(),
        },
        "note": "fp32 corpus is resident transiently during build; codes are the steady state.",
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output) if args.output else Path(f"benchmarks/results/ivfpq_scale_{ts}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
