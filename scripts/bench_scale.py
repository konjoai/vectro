#!/usr/bin/env python3
"""Scale benchmark for vectro past 1M — 10M / 100M / 1B.

Two data sources:
  --synthetic            stream-generate N seeded vectors in chunks (never holds
                         the full fp32 base in RAM; encodes/adds chunk by chunk)
  --format bigann ...    read big-ann-benchmarks .u8bin/.fbin base+query and an
                         .ibin/.fbin ground-truth file

Index types: vectro `hnsw` / `ivfpq`, and faiss `faiss-hnsw` / `faiss-ivfpq`.
At 100M+ only the PQ-compressed indexes fit on commodity RAM — see
benchmarks/BILLION_SCALE.md for the memory math.

Reports build time, peak RSS, on-disk size, and recall-vs-QPS; writes a
timestamped JSON to benchmarks/results/.

Examples:
    python scripts/bench_scale.py --synthetic --n 2_000_000 --dim 128 --index ivfpq
    python scripts/bench_scale.py --format bigann --base base.u8bin \
        --query query.u8bin --gt gt.ibin --n 10_000_000 --index hnsw
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

import vectro_py

CHUNK = 500_000  # vectors per streamed build chunk


def peak_rss_gb() -> float:
    """Peak resident set size. ru_maxrss is bytes on macOS, KiB on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    div = 1 << 30 if sys.platform == "darwin" else 1 << 20
    return raw / div


# ───────────────────────────── data sources ──────────────────────────────


def synth_chunk(start: int, count: int, dim: int, seed: int) -> np.ndarray:
    """Deterministic synthetic vectors [start, start+count) — clustered so ANN
    recall is meaningful (not uniform noise where all neighbours tie)."""
    rng = np.random.default_rng(seed + start)
    centers = np.random.default_rng(seed).standard_normal((256, dim)).astype(np.float32)
    cids = rng.integers(0, 256, size=count)
    return (centers[cids] + 0.35 * rng.standard_normal((count, dim)).astype(np.float32)).astype(
        np.float32
    )


def read_bin(path: Path, dtype: str, n: int | None = None) -> np.ndarray:
    """Read a big-ann-benchmarks .u8bin/.fbin: u32 npts, u32 dim, then data."""
    with open(path, "rb") as f:
        npts, dim = np.fromfile(f, dtype=np.uint32, count=2)
        take = npts if n is None else min(n, int(npts))
        data = np.fromfile(f, dtype=dtype, count=take * int(dim)).reshape(take, int(dim))
    return np.ascontiguousarray(data, dtype=np.float32)


def read_gt(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        nq, k = np.fromfile(f, dtype=np.uint32, count=2)
        ids = np.fromfile(f, dtype=np.int32, count=int(nq) * int(k)).reshape(int(nq), int(k))
    return ids.astype(np.int64)


# ────────────────────────── ground truth (synthetic) ──────────────────────


def _norm_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def brute_force_gt(
    queries: np.ndarray, gen, n: int, dim: int, seed: int, k: int, cosine: bool = False
) -> np.ndarray:
    """Chunked exact top-k over a streamed base — never holds all of it. With
    `cosine`, ranks by 1−cos (== L2 on unit vectors), matching the IVF-PQ index;
    otherwise squared-L2, matching the flat-HNSW L2 index."""
    nq = len(queries)
    q = _norm_rows(queries) if cosine else queries
    best_d = np.full((nq, k), np.inf, dtype=np.float32)
    best_i = np.full((nq, k), -1, dtype=np.int64)
    for start in range(0, n, CHUNK):
        cnt = min(CHUNK, n - start)
        chunk = gen(start, cnt, dim, seed)
        if cosine:
            chunk = _norm_rows(chunk)
        # (nq, cnt) squared-L2 via -2qx + |x|^2 (+|q|^2 const, irrelevant to argmin)
        d = (chunk * chunk).sum(1)[None, :] - 2.0 * q @ chunk.T
        idx = np.argpartition(d, min(k, cnt - 1), axis=1)[:, :k]
        cd = np.take_along_axis(d, idx, axis=1)
        ci = idx + start
        alld = np.concatenate([best_d, cd], axis=1)
        alli = np.concatenate([best_i, ci], axis=1)
        sel = np.argpartition(alld, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(alld, sel, axis=1)
        best_i = np.take_along_axis(alli, sel, axis=1)
    return best_i


def recall_at_k(got: np.ndarray, gt: np.ndarray, k: int) -> float:
    hits = sum(len(set(g[:k].tolist()) & set(t[:k].tolist())) for g, t in zip(got, gt))
    return hits / (len(got) * k)


# ───────────────────────────────── build ──────────────────────────────────


def build_index(args, gen, query: np.ndarray):
    n, dim = args.n, args.dim
    if args.index == "hnsw":
        idx = vectro_py.PyHnswIndex(args.m, args.ef_construction, "l2")
    elif args.index == "ivfpq":
        idx = vectro_py.PyIvfPqIndex(args.nlist, args.nprobe)
    else:
        raise SystemExit(f"unknown index {args.index}")

    t0 = time.perf_counter()
    if args.index == "ivfpq":
        train = gen(0, min(args.train, n), dim, args.seed)
        idx.train_np(train, args.pq_subspaces, 256, 25, args.seed)
        del train
    added = 0
    while added < n:
        cnt = min(CHUNK, n - added)
        idx.add_np(gen(added, cnt, dim, args.seed))
        added += cnt
        print(f"  added {added:,}/{n:,}  rss={peak_rss_gb():.2f} GB", flush=True)
    build_s = time.perf_counter() - t0
    return idx, build_s


def search_pareto(idx, kind: str, query: np.ndarray, gt: np.ndarray, k: int, params):
    """`params` sweeps ef (HNSW) or n_probe (IVF-PQ) — the recall/speed knob."""
    rows = []
    for pv in params:
        t0 = time.perf_counter()
        if kind == "ivfpq":
            res = [idx.search_with_probe(q, k, pv) for q in query]
        else:
            res = [idx.search_np(q, k, pv) for q in query]
        dt = time.perf_counter() - t0
        got = np.array([[i for i, _ in r] + [-1] * (k - len(r)) for r in res])
        rows.append((pv, recall_at_k(got, gt, k), len(query) / dt))
    return rows


# ────────────────────────────────── main ──────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--format", choices=["bigann"], help="real dataset format")
    p.add_argument("--base")
    p.add_argument("--query")
    p.add_argument("--gt")
    p.add_argument("--bin-dtype", default="uint8", help="bigann base dtype (uint8/float32)")
    p.add_argument("--n", type=int, default=2_000_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--index", choices=["hnsw", "ivfpq"], default="ivfpq")
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument("--pq-subspaces", type=int, default=16)
    p.add_argument("--train", type=int, default=200_000)
    p.add_argument("--nq", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--efs", default="16,32,64,128,256", help="ef (hnsw) / n_probe (ivfpq) sweep")
    args = p.parse_args()

    sweep = [int(x) for x in args.efs.split(",")]

    if args.format == "bigann":
        base_n = args.n
        query = read_bin(Path(args.query), args.bin_dtype, args.nq)
        gt = read_gt(Path(args.gt))[: len(query), : args.k]
        args.dim = query.shape[1]

        def gen(start, count, dim, seed):
            # Slice the on-disk base lazily per chunk.
            with open(args.base, "rb") as f:
                _npts, d = np.fromfile(f, dtype=np.uint32, count=2)
                f.seek(8 + start * int(d) * np.dtype(args.bin_dtype).itemsize)
                raw = np.fromfile(f, dtype=args.bin_dtype, count=count * int(d))
            return np.ascontiguousarray(raw.reshape(count, int(d)), dtype=np.float32)

    else:
        base_n = args.n
        query = synth_chunk(10**9, args.nq, args.dim, args.seed)  # disjoint query block
        gen = synth_chunk
        # IVF-PQ ranks by cosine (it normalises); HNSW here is L2. Match the GT.
        cosine_gt = args.index == "ivfpq"
        print(
            f"computing {'cosine' if cosine_gt else 'L2'} GT for {args.nq} queries "
            f"over {base_n:,}...",
            flush=True,
        )
        gt = brute_force_gt(query, gen, base_n, args.dim, args.seed, args.k, cosine=cosine_gt)

    bytes_per_vec = {"hnsw": args.dim * 4, "ivfpq": args.pq_subspaces}[args.index]
    print(
        f"##### scale={base_n:,} dim={args.dim} index={args.index} "
        f"(~{bytes_per_vec} B/vec codes) #####",
        flush=True,
    )
    idx, build_s = build_index(args, gen, query)
    rss = peak_rss_gb()

    dst_dir = Path("benchmarks/results")
    dst_dir.mkdir(parents=True, exist_ok=True)
    save_path = dst_dir / f"scale_{args.index}_{base_n}.idx"
    idx.save(str(save_path))
    on_disk_gb = save_path.stat().st_size / (1 << 30)

    rows = search_pareto(idx, args.index, query, gt, args.k, sweep)
    print(f"build={build_s:.1f}s  peakRSS={rss:.2f}GB  onDisk={on_disk_gb:.2f}GB")
    knob = "nprobe" if args.index == "ivfpq" else "ef"
    print(f"{knob:>6} {'recall':>8} {'qps':>10}")
    for ef, r, q in rows:
        print(f"{ef:>6} {r:>8.4f} {q:>10,.0f}")

    out = {
        "scale": base_n,
        "dim": args.dim,
        "index": args.index,
        "bytes_per_vec_codes": bytes_per_vec,
        "build_s": build_s,
        "peak_rss_gb": rss,
        "on_disk_gb": on_disk_gb,
        "k": args.k,
        "pareto": [{"ef": e, "recall": r, "qps": q} for e, r, q in rows],
        "params": vars(args),
    }
    dst = dst_dir / f"scale_{args.index}_{base_n}.json"
    dst.write_text(json.dumps(out, indent=2, default=str))
    print(f"WROTE {dst}")
    save_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
