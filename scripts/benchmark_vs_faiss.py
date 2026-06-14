#!/usr/bin/env python
"""
Vectro vs FAISS — Recall-Matched ANN Benchmark
===============================================

Compares vectro HNSWIndex against faiss.IndexHNSWFlat and faiss.IndexIVFFlat
on two ANN-benchmark datasets, at three index sizes, across matched Recall@10
targets (0.90 / 0.95 / 0.99).

Required packages (beyond the vectro dev environment)::

    pip install faiss-cpu h5py requests

Usage::

    python scripts/benchmark_vs_faiss.py          # full run
    python scripts/benchmark_vs_faiss.py --quick  # 5-minute sanity check

See the REPRODUCTION INSTRUCTIONS docstring at the bottom of this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────

K = 10  # Recall@K
HNSW_M = 16  # HNSW bidirectional links per node
HNSW_EF_CONSTRUCTION = 200
IVF_NLIST = 100  # IVF number of coarse clusters
N_QUERIES_EVAL = 200  # queries used for recall estimation + param search
N_WARMUP = 5  # warmup iterations discarded before timing
N_REPS = 5  # timed repetitions per batch size (median reported)

# Monotonically increasing sweeps — we stop at the first param that meets target
EF_SWEEP: List[int] = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048]
NPROBE_SWEEP: List[int] = [1, 2, 4, 5, 8, 10, 16, 20, 32, 50, 64, 100]

# ── dataset registry ──────────────────────────────────────────────────────────

DATASETS: Dict[str, Dict[str, Any]] = {
    "glove-100-angular": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "filename": "glove-100-angular.hdf5",
        "dim": 100,
        "n_train_full": 1_183_514,
    },
    "dbpedia-openai-1M-angular": {
        "url": "http://ann-benchmarks.com/dbpedia-openai-1M-angular.hdf5",
        "filename": "dbpedia-openai-1M-angular.hdf5",
        "dim": 1536,
        "n_train_full": 990_000,
    },
}

FULL_DATASETS = ["glove-100-angular", "dbpedia-openai-1M-angular"]
QUICK_DATASETS = ["glove-100-angular"]

FULL_INDEX_SIZES = [10_000, 100_000, 1_000_000]
QUICK_INDEX_SIZES = [10_000]

FULL_RECALL_TARGETS = [0.90, 0.95, 0.99]
QUICK_RECALL_TARGETS = [0.90]

BATCH_SIZES = [1, 10, 100]


# ── helpers ───────────────────────────────────────────────────────────────────


def _unit(v: np.ndarray) -> np.ndarray:
    """Row-normalise to unit length (float32)."""
    v = v.astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


def rss_kb() -> int:
    """Peak RSS in KiB (Linux reports KB; macOS reports bytes)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform != "darwin" else raw // 1024


def hardware_meta() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


# ── dataset download ──────────────────────────────────────────────────────────


def download_dataset(name: str, data_dir: Path) -> Path:
    """Download an ANN-benchmark HDF5 file if not already cached."""
    import requests  # noqa: PLC0415

    meta = DATASETS[name]
    out_path = data_dir / meta["filename"]
    if out_path.exists():
        log.info("Dataset cached: %s", out_path)
        return out_path

    data_dir.mkdir(parents=True, exist_ok=True)
    url = meta["url"]
    log.info("Downloading %s ...", url)

    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done // total
                    print(
                        f"\r  {name}: {pct:3d}%  ({done >> 20}/{total >> 20} MiB)    ",
                        end="",
                        flush=True,
                    )
    print()
    log.info("Saved to %s", out_path)
    return out_path


# ── dataset loading ───────────────────────────────────────────────────────────


def load_dataset(
    name: str,
    data_dir: Path,
    n_train: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train[:n_train], queries[:N_QUERIES_EVAL]) as float32 arrays."""
    import h5py  # noqa: PLC0415

    path = data_dir / DATASETS[name]["filename"]
    with h5py.File(path, "r") as fh:
        train = fh["train"][:n_train].astype(np.float32)
        n_q = min(N_QUERIES_EVAL, len(fh["test"]))
        queries = fh["test"][:n_q].astype(np.float32)

    log.info("Loaded: train %s  queries %s", train.shape, queries.shape)
    return train, queries


# ── brute-force ground truth ──────────────────────────────────────────────────


def compute_exact_gt(
    train: np.ndarray,
    queries: np.ndarray,
    k: int,
) -> np.ndarray:
    """Block-wise brute-force cosine top-k.  Peak memory ≈ 80 MB per block.

    Returns
    -------
    gt : ndarray, shape (n_queries, k), dtype int64
        Indices into *train*, sorted by decreasing cosine similarity.
    """
    n_train = len(train)
    n_q = len(queries)

    train_u = _unit(train)
    q_u = _unit(queries)

    best_sims = np.full((n_q, k), -np.inf, dtype=np.float32)
    best_ids = np.zeros((n_q, k), dtype=np.int64)

    blk = 100_000
    for ti in range(0, n_train, blk):
        chunk = train_u[ti : ti + blk]  # (B, d)
        sims = (q_u @ chunk.T).astype(np.float32)  # (n_q, B)
        blk_ids = np.arange(ti, ti + len(chunk), dtype=np.int64)

        all_sims = np.concatenate([best_sims, sims], axis=1)  # (n_q, k+B)
        all_ids = np.concatenate([best_ids, np.tile(blk_ids, (n_q, 1))], axis=1)  # (n_q, k+B)

        part_idx = np.argpartition(all_sims, -k, axis=1)[:, -k:]  # (n_q, k)
        rows = np.arange(n_q)[:, None]
        best_sims = all_sims[rows, part_idx]
        best_ids = all_ids[rows, part_idx]

    order = np.argsort(-best_sims, axis=1)
    rows = np.arange(n_q)[:, None]
    return best_ids[rows, order]


# ── recall ────────────────────────────────────────────────────────────────────


def batch_recall(results: List[np.ndarray], gt: np.ndarray, k: int) -> float:
    """Mean Recall@k over a list of per-query result arrays."""
    if not results:
        return 0.0
    hits = sum(
        len(set(int(x) for x in r[:k]) & set(int(x) for x in gt[i, :k]))
        for i, r in enumerate(results)
    )
    return hits / (k * len(results))


# ── index backends ────────────────────────────────────────────────────────────


class VectroHNSW:
    """Wraps python.hnsw_api.HNSWIndex with a uniform build/query interface."""

    label = "vectro-hnsw"
    param_name = "ef"
    param_sweep = EF_SWEEP

    def __init__(self) -> None:
        from python.hnsw_api import HNSWIndex  # noqa: PLC0415

        self._cls = HNSWIndex
        self.index: Optional[Any] = None

    def build(self, train: np.ndarray) -> None:
        idx = self._cls(M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, space="cosine")
        idx.add(train)
        self.index = idx

    def query_batch(self, queries: np.ndarray, k: int, param: int) -> List[np.ndarray]:
        assert self.index is not None
        return [self.index.search(q, k=k, ef=param)[0] for q in queries]


class FaissHNSW:
    """faiss.IndexHNSWFlat with matched M/ef_construction (inner-product / cosine)."""

    label = "faiss-hnsw"
    param_name = "ef"
    param_sweep = EF_SWEEP

    def __init__(self) -> None:
        import faiss  # noqa: PLC0415

        self._faiss = faiss
        self.index: Optional[Any] = None

    def build(self, train: np.ndarray) -> None:
        faiss = self._faiss
        d = int(train.shape[1])
        idx = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        idx.add(_unit(train))
        self.index = idx

    def query_batch(self, queries: np.ndarray, k: int, param: int) -> List[np.ndarray]:
        assert self.index is not None
        self.index.hnsw.efSearch = param
        _, I = self.index.search(_unit(queries), k)
        return [I[i] for i in range(len(queries))]


class FaissIVF:
    """faiss.IndexIVFFlat (nlist=IVF_NLIST, nprobe varied for recall target)."""

    label = "faiss-ivf"
    param_name = "nprobe"
    param_sweep = NPROBE_SWEEP

    def __init__(self, nlist: int = IVF_NLIST) -> None:
        import faiss  # noqa: PLC0415

        self._faiss = faiss
        self._nlist = nlist
        self.index: Optional[Any] = None

    def build(self, train: np.ndarray) -> None:
        faiss = self._faiss
        d = int(train.shape[1])
        quantizer = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFFlat(quantizer, d, self._nlist, faiss.METRIC_INNER_PRODUCT)
        train_u = _unit(train)
        idx.train(train_u)
        idx.add(train_u)
        self.index = idx

    def query_batch(self, queries: np.ndarray, k: int, param: int) -> List[np.ndarray]:
        assert self.index is not None
        self.index.nprobe = param
        _, I = self.index.search(_unit(queries), k)
        return [I[i] for i in range(len(queries))]


# ── parameter search ──────────────────────────────────────────────────────────


def find_param(
    query_fn: Callable[[np.ndarray, int, int], List[np.ndarray]],
    queries: np.ndarray,
    gt: np.ndarray,
    k: int,
    target: float,
    sweep: List[int],
) -> Optional[int]:
    """Return the smallest value in *sweep* that achieves *target* recall.

    Uses the first half of the eval queries so the full set remains unseen
    until the final verification step.
    """
    half = max(1, len(queries) // 2)
    q_small = queries[:half]
    gt_small = gt[:half]

    for param in sweep:
        results = query_fn(q_small, k, param)
        if batch_recall(results, gt_small, k) >= target:
            return param
    return None


# ── throughput ────────────────────────────────────────────────────────────────


def measure_throughput(
    query_fn: Callable[[np.ndarray, int, int], List[np.ndarray]],
    queries: np.ndarray,
    k: int,
    param: int,
    batch_sizes: List[int],
) -> Dict[int, float]:
    """QPS at each batch size.  Discards N_WARMUP runs, reports median of N_REPS."""
    query_fn(queries[:N_WARMUP], k, param)  # warmup

    result: Dict[int, float] = {}
    for bs in batch_sizes:
        q_bs = queries[:bs]
        times = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            query_fn(q_bs, k, param)
            times.append(time.perf_counter() - t0)
        median_s = float(np.median(times))
        result[bs] = bs / median_s if median_s > 0 else 0.0
    return result


# ── single-condition benchmark ────────────────────────────────────────────────


def run_one(
    backend: Any,
    train: np.ndarray,
    queries: np.ndarray,
    gt: np.ndarray,
    recall_targets: List[float],
    batch_sizes: List[int],
    k: int = K,
) -> Dict[str, Any]:
    """Build + sweep one index backend across all recall targets."""
    label = backend.label
    log.info("[%s] Building index  n=%d  d=%d", label, len(train), train.shape[1])

    t0 = time.perf_counter()
    backend.build(train)
    build_s = time.perf_counter() - t0
    log.info("[%s] Build %.2f s", label, build_s)

    out: Dict[str, Any] = {
        "label": label,
        "n": len(train),
        "d": int(train.shape[1]),
        "build_s": round(build_s, 3),
        "recalls": {},
    }

    for target in recall_targets:
        param = find_param(backend.query_batch, queries, gt, k, target, backend.param_sweep)
        if param is None:
            log.warning("[%s] Cannot reach recall %.2f — skipped", label, target)
            out["recalls"][str(target)] = {"skipped": True}
            continue

        # Verify on full eval set
        res_full = backend.query_batch(queries, k, param)
        actual_r = batch_recall(res_full, gt, k)
        rss_q = rss_kb()

        qps = measure_throughput(backend.query_batch, queries, k, param, batch_sizes)
        log.info(
            "[%s] R=%.2f  %s=%d  actual=%.4f  qps@1=%.0f  qps@100=%.0f",
            label,
            target,
            backend.param_name,
            param,
            actual_r,
            qps.get(1, 0),
            qps.get(100, 0),
        )

        out["recalls"][str(target)] = {
            backend.param_name: param,
            "actual_recall": round(actual_r, 4),
            "peak_rss_kb": rss_q,
            "qps": {str(bs): round(v, 1) for bs, v in qps.items()},
        }

    return out


# ── markdown table ────────────────────────────────────────────────────────────


def _qps_cell(val: float) -> str:
    """Format a QPS value compactly."""
    if val == 0:
        return "      —"
    if val >= 1_000_000:
        return f"{val / 1e6:>5.1f}Mqps"
    if val >= 1_000:
        return f"{val / 1e3:>5.1f}kqps"
    return f"{val:>7.1f}"


def md_section(
    dataset: str,
    n: int,
    conditions: List[Dict[str, Any]],
    recall_targets: List[float],
    batch_sizes: List[int],
) -> str:
    """Return a markdown section for one (dataset, n_train) run."""
    lines: List[str] = [
        f"\n## {dataset}  |  n={n:,}  |  d={conditions[0]['d']}",
        "",
    ]

    col_bs = [f"QPS@{bs:>3}" for bs in batch_sizes]
    header_parts = ["Method", "Target R@10", "Actual R@10", "Param"] + col_bs + ["RSS MB"]
    sep_parts = [":---", "---:", "---:", "---:"] + ["---:"] * len(batch_sizes) + ["---:"]

    def _row(cells: List[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    lines.append(_row(header_parts))
    lines.append(_row(sep_parts))

    for target in recall_targets:
        key = str(target)

        # Reference QPS from faiss-hnsw for speedup computation
        ref_qps: Dict[int, float] = {}
        for cond in conditions:
            if cond["label"] == "faiss-hnsw":
                rec = cond["recalls"].get(key, {})
                if not rec.get("skipped"):
                    ref_qps = {bs: rec["qps"].get(str(bs), 0.0) for bs in batch_sizes}

        for cond in conditions:
            rec = cond["recalls"].get(key, {})
            if rec.get("skipped"):
                cells = (
                    [cond["label"], f"{target:.2f}", "—", "—"] + ["—"] * len(batch_sizes) + ["—"]
                )
            else:
                param_val = rec.get("ef", rec.get("nprobe", rec.get("param", "—")))
                qps_d = rec.get("qps", {})
                qps_cells = [_qps_cell(qps_d.get(str(bs), 0.0)) for bs in batch_sizes]
                rss_mb = rec.get("peak_rss_kb", 0) // 1024
                cells = (
                    [
                        cond["label"],
                        f"{target:.2f}",
                        f"{rec.get('actual_recall', 0):.4f}",
                        str(param_val),
                    ]
                    + qps_cells
                    + [str(rss_mb)]
                )
            lines.append(_row(cells))

        # Speedup row: vectro vs faiss-hnsw
        vectro = next((c for c in conditions if c["label"] == "vectro-hnsw"), None)
        if vectro and ref_qps:
            vrec = vectro["recalls"].get(key, {})
            if not vrec.get("skipped"):
                vqps_d = vrec.get("qps", {})
                speedup_cells = [
                    f"{vqps_d.get(str(bs), 0.0) / max(ref_qps.get(bs, 1.0), 1e-9):.3f}x"
                    for bs in batch_sizes
                ]
                lines.append(_row(["  ↳ vectro/faiss-hnsw", "", "", ""] + speedup_cells + [""]))

        lines.append(_row([""] * len(header_parts)))  # blank separator row

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recall-matched vectro vs FAISS ANN benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only glove-100-angular at n=10k (~5 minutes)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(_PROJECT_ROOT / "data"),
        help="Directory for HDF5 dataset files (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_PROJECT_ROOT / "results"),
        help="Directory for JSON result files (default: results/)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = QUICK_DATASETS if args.quick else FULL_DATASETS
    index_sizes = QUICK_INDEX_SIZES if args.quick else FULL_INDEX_SIZES
    recall_targets = QUICK_RECALL_TARGETS if args.quick else FULL_RECALL_TARGETS

    try:
        import faiss  # noqa: F401, PLC0415

        has_faiss = True
        log.info("faiss %s available", getattr(faiss, "__version__", "?"))
    except ImportError:
        has_faiss = False
        log.warning("faiss not installed — only vectro-hnsw will run.  pip install faiss-cpu")

    try:
        import h5py  # noqa: F401, PLC0415
    except ImportError as exc:
        log.error("h5py is required.  pip install h5py")
        raise SystemExit(1) from exc

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_results: Dict[str, Any] = {
        "timestamp": timestamp,
        "quick": args.quick,
        "hardware": hardware_meta(),
        "config": {
            "K": K,
            "HNSW_M": HNSW_M,
            "HNSW_EF_CONSTRUCTION": HNSW_EF_CONSTRUCTION,
            "IVF_NLIST": IVF_NLIST,
            "N_QUERIES_EVAL": N_QUERIES_EVAL,
            "N_WARMUP": N_WARMUP,
            "N_REPS": N_REPS,
        },
        "runs": [],
    }

    md_sections: List[str] = []

    for ds_name in datasets:
        download_dataset(ds_name, data_dir)
        n_train_full = DATASETS[ds_name]["n_train_full"]

        for n_train in index_sizes:
            if n_train > n_train_full:
                log.info(
                    "Skipping n=%d for %s (only %d vectors available)",
                    n_train,
                    ds_name,
                    n_train_full,
                )
                continue

            log.info("── %s  n=%d ──", ds_name, n_train)
            train, queries = load_dataset(ds_name, data_dir, n_train)

            log.info(
                "Computing brute-force ground truth (n_train=%d, n_q=%d)...",
                len(train),
                len(queries),
            )
            gt = compute_exact_gt(train, queries, K)

            conditions: List[Dict[str, Any]] = []

            # vectro HNSW — always runs
            conditions.append(
                run_one(VectroHNSW(), train, queries, gt, recall_targets, BATCH_SIZES)
            )

            if has_faiss:
                conditions.append(
                    run_one(FaissHNSW(), train, queries, gt, recall_targets, BATCH_SIZES)
                )
                # Ensure nlist <= n_train and meets faiss training requirements (≥39 pts/cluster)
                safe_nlist = min(IVF_NLIST, max(1, n_train // 39))
                conditions.append(
                    run_one(
                        FaissIVF(nlist=safe_nlist),
                        train,
                        queries,
                        gt,
                        recall_targets,
                        BATCH_SIZES,
                    )
                )

            run_entry: Dict[str, Any] = {
                "dataset": ds_name,
                "n_train": n_train,
                "conditions": conditions,
            }
            all_results["runs"].append(run_entry)
            md_sections.append(
                md_section(ds_name, n_train, conditions, recall_targets, BATCH_SIZES)
            )

    out_path = out_dir / f"benchmark_{timestamp}.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    log.info("Results written to %s", out_path)

    print("\n" + "=" * 72)
    print("VECTRO vs FAISS — Recall-Matched Benchmark Results")
    print("=" * 72)
    for section in md_sections:
        print(section)
    print(f"\nFull results: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ═══════════════════════════════════════════════════════════════════════════════
# REPRODUCTION INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
_REPRO_DOC = """
How to Reproduce
================

Environment
-----------
1. Clone and set up the vectro repository::

       git clone https://github.com/konjoai/vectro && cd vectro
       cargo build --release
       pip install -e ".[dev,bench-ann]"

2. Install benchmark-specific dependencies::

       pip install faiss-cpu h5py requests

Quickstart (≈5 minutes)
-----------------------
Run only glove-100-angular at n=10k vectors::

    python scripts/benchmark_vs_faiss.py --quick

Full benchmark
--------------
Downloads two datasets (~6.5 GB total on first run), then runs all
conditions::

    python scripts/benchmark_vs_faiss.py

Expected wall time: 2–8 hours depending on hardware and whether vectro is
compiled (Rust backend) or running pure Python.

Results are written to ``results/benchmark_<timestamp>.json``.

Datasets
--------
Both come from the ANN-Benchmarks suite (https://ann-benchmarks.com):

glove-100-angular
    Source: GloVe word vectors (Pennington et al. 2014).
    Vectors: 1 183 514 train + 10 000 test, d=100, angular metric.
    File: ``data/glove-100-angular.hdf5`` (~500 MB)

dbpedia-openai-1M-angular
    Source: DBpedia entity descriptions embedded with OpenAI
    text-embedding-3-large.
    Vectors: ~990 000 train + 10 000 test, d=1 536, angular metric.
    File: ``data/dbpedia-openai-1M-angular.hdf5`` (~6 GB)

Methodology
-----------
All comparisons are **recall-matched**.  For each index type and target
recall level (0.90 / 0.95 / 0.99), the benchmark sweeps the search
parameter (``ef_search`` for HNSW, ``nprobe`` for IVF) and selects the
smallest value that meets the target on the first half of the eval queries.
Throughput is then measured at that parameter using the same query matrix
at batch sizes 1, 10, and 100 (N_WARMUP=5 discarded, N_REPS=5 timed,
median reported).

Ground truth for Recall@10 is always recomputed by brute-force cosine
search on the *indexed* training subset, so results are correct even when
the index uses fewer than the full dataset vectors.

Index configurations (matched between vectro and faiss):
  - HNSW: M=16, ef_construction=200
  - IVF:  nlist=100 (or less for small n), nprobe swept to hit target

Memory is measured via ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` after
the query phase and reported as peak RSS in MB.

References
----------
- HNSW: Malkov & Yashunin 2018, arXiv:1603.09320
- FAISS: Johnson et al. 2017, arXiv:1702.08734
- GloVe: Pennington et al. 2014, https://nlp.stanford.edu/projects/glove/
- ANN-Benchmarks: Aumuller et al. 2020, arXiv:1807.05614
"""
