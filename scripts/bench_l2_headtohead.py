#!/usr/bin/env python3
"""Head-to-head L2 (Euclidean) benchmark: vectro vs faiss-HNSW.

ann-benchmarks methodology — precomputed `neighbors` ground truth, single
query thread, recall-vs-QPS Pareto. Validates vectro's new `l2` metric on a
real Euclidean dataset (a category vectro could not previously search
correctly, since only cosine existed).

Usage:
    python scripts/bench_l2_headtohead.py data/fashion-mnist-784-euclidean.hdf5
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import faiss
import h5py
import numpy as np

import vectro_py

M = 16
EF_CONSTRUCTION = 200
EF_QUERY = [10, 20, 40, 80, 160, 320]
K = 10


def recall(got: np.ndarray, gt: np.ndarray, k: int) -> float:
    hits = 0
    for g, t in zip(got, gt):
        hits += len(set(g[:k].tolist()) & set(t[:k].tolist()))
    return hits / (len(got) * k)


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def bench_vectro(train, test, gt, angular):
    metric = "cosine" if angular else "l2"
    idx = vectro_py.PyHnswIndex(M, EF_CONSTRUCTION, metric)
    t0 = time.perf_counter()
    idx.add_np(train)
    build = time.perf_counter() - t0
    rows = []
    for ef in EF_QUERY:
        t0 = time.perf_counter()
        res = [idx.search_np(q, K, ef) for q in test]
        dt = time.perf_counter() - t0
        got = np.array([[i for i, _ in r] + [-1] * (K - len(r)) for r in res])
        rows.append((ef, recall(got, gt, K), len(test) / dt))
    return build, rows


def bench_faiss(train, test, gt, angular):
    faiss.omp_set_num_threads(1)
    d = train.shape[1]
    if angular:
        # Cosine ≡ inner product on unit vectors.
        train = _normalize(train)
        test = _normalize(test)
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    t0 = time.perf_counter()
    index.add(train)
    build = time.perf_counter() - t0
    rows = []
    for ef in EF_QUERY:
        index.hnsw.efSearch = ef
        t0 = time.perf_counter()
        _, idxs = index.search(test, K)
        dt = time.perf_counter() - t0
        rows.append((ef, recall(idxs, gt, K), len(test) / dt))
    return build, rows


def qps_at_recall(rows, target):
    best = None
    for _, r, q in rows:
        if r >= target:
            best = q if best is None else max(best, q)
    return best


def main() -> int:
    path = Path(sys.argv[1])
    with h5py.File(path, "r") as f:
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        gt = np.ascontiguousarray(f["neighbors"][:], dtype=np.int64)
    n, d = train.shape
    angular = "angular" in path.stem
    metric = "cosine/angular" if angular else "L2/Euclidean"
    print(f"##### {path.stem}  N={n} D={d}  ({metric}) #####")

    vb, vr = bench_vectro(train, test, gt, angular)
    fb, fr = bench_faiss(train, test, gt, angular)
    print(f"vectro build={vb:.1f}s maxR={max(r for _, r, _ in vr):.4f}")
    print(f"faiss  build={fb:.1f}s maxR={max(r for _, r, _ in fr):.4f}")
    print(f"{'engine':8} {'build':>7} {'Q@.85':>9} {'Q@.90':>9} {'Q@.95':>9} {'Q@.99':>9}")
    for name, b, rows in (("vectro", vb, vr), ("faiss", fb, fr)):
        cells = [qps_at_recall(rows, t) for t in (0.85, 0.90, 0.95, 0.99)]
        cs = "  ".join(f"{int(c):>7,}" if c else f"{'--':>7}" for c in cells)
        print(f"{name:8} {b:6.1f}s  {cs}")

    out = {
        "dataset": path.stem,
        "n": n,
        "d": d,
        "metric": "cosine" if angular else "l2",
        "M": M,
        "ef_construction": EF_CONSTRUCTION,
        "k": K,
        "vectro": {"build_s": vb, "pareto": [{"ef": e, "recall": r, "qps": q} for e, r, q in vr]},
        "faiss": {"build_s": fb, "pareto": [{"ef": e, "recall": r, "qps": q} for e, r, q in fr]},
    }
    res_dir = Path("benchmarks/results")
    res_dir.mkdir(parents=True, exist_ok=True)
    dst = res_dir / f"headtohead_{path.stem}.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"WROTE {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
