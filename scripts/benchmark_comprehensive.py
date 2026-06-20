#!/usr/bin/env python
"""
Vectro Comprehensive Head-to-Head Benchmark
===========================================

Runs vectro against FAISS and hnswlib on **real** ANN-benchmark data along the
two axes that matter for an embedding-compression + search library:

1. **ANN search** — Recall@10 vs QPS (single-thread, recall-matched), build
   time, and serialized index size.  Backends: vectro-hnsw, faiss-hnsw,
   faiss-ivf, hnswlib, and an exact FAISS flat baseline (recall ceiling).

2. **Quantization** — encode throughput, compression ratio, and reconstruction
   cosine on the same real vectors.  Backends: vectro INT8 (the Rust SIMD
   kernel) and vectro PQ vs FAISS ScalarQuantizer(QT_8bit) and FAISS IndexPQ.

Fairness notes
--------------
* FAISS and hnswlib are pinned to a single thread (``faiss.omp_set_num_threads(1)``,
  ``num_threads=1``) so the comparison is single-thread vs vectro's single-thread
  Python query loop.
* All search backends use the same metric (cosine via L2-normalised inner
  product) and the same brute-force ground truth, recomputed on the indexed
  subset.
* The pip ``faiss-cpu`` wheel may load the generic (non-AVX2/AVX-512) build;
  this is recorded in the report's ``faiss_simd`` field.

Usage
-----
    pip install faiss-cpu hnswlib h5py requests matplotlib
    python scripts/benchmark_comprehensive.py --quick          # glove-25, n=10k
    python scripts/benchmark_comprehensive.py --dataset glove-100-angular --n 100000

Outputs JSON + markdown + PNG plots to
``benchmarks/results/<timestamp>_comprehensive/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse the validated dataset + recall machinery from the existing harness.
from scripts.benchmark_vs_faiss import (  # noqa: E402
    DATASETS,
    EF_SWEEP,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    NPROBE_SWEEP,
    FaissHNSW,
    FaissIVF,
    VectroHNSW,
    _unit,
    batch_recall,
    compute_exact_gt,
    download_dataset,
    hardware_meta,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_comprehensive")

K = 10
N_QUERIES = 200
N_WARMUP = 3
N_REPS = 5
RECALL_TARGETS = [0.90, 0.95, 0.99]


# ── single-thread fairness ──────────────────────────────────────────────────────


def pin_single_thread() -> str:
    """Pin FAISS to one OpenMP thread; return its SIMD build tag."""
    try:
        import faiss  # noqa: PLC0415

        faiss.omp_set_num_threads(1)
        # The loaded module name encodes the SIMD variant (avx512/avx2/generic).
        simd = "generic"
        for tag in ("avx512", "avx2"):
            if any(tag in m for m in sys.modules if "faiss" in m):
                simd = tag
                break
        return simd
    except ImportError:
        return "n/a"


# ── extra search backends (new competitors) ─────────────────────────────────────


class HnswlibHNSW:
    """hnswlib (the reference HNSW implementation), single-thread queries."""

    label = "hnswlib"
    param_name = "ef"
    param_sweep = EF_SWEEP

    def __init__(self) -> None:
        self.index: Optional[Any] = None

    def build(self, train: np.ndarray) -> None:
        import hnswlib  # noqa: PLC0415

        d = int(train.shape[1])
        idx = hnswlib.Index(space="cosine", dim=d)
        idx.init_index(max_elements=len(train), ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)
        idx.set_num_threads(1)
        idx.add_items(_unit(train), num_threads=1)
        self.index = idx

    def query_batch(self, queries: np.ndarray, k: int, param: int) -> List[np.ndarray]:
        assert self.index is not None
        self.index.set_ef(max(param, k))
        labels, _ = self.index.knn_query(_unit(queries), k=k, num_threads=1)
        return [labels[i] for i in range(len(queries))]


class ExactFlat:
    """FAISS exact flat (IndexFlatIP on unit vectors) — recall ceiling / QPS floor."""

    label = "exact-faiss"
    param_name = "—"
    param_sweep = [0]

    def __init__(self) -> None:
        self.index: Optional[Any] = None

    def build(self, train: np.ndarray) -> None:
        import faiss  # noqa: PLC0415

        idx = faiss.IndexFlatIP(int(train.shape[1]))
        idx.add(_unit(train))
        self.index = idx

    def query_batch(self, queries: np.ndarray, k: int, param: int) -> List[np.ndarray]:
        assert self.index is not None
        _, indices = self.index.search(_unit(queries), k)
        return [indices[i] for i in range(len(queries))]


# ── measurement helpers ─────────────────────────────────────────────────────────


def measure_index_size_mb(backend: Any) -> Optional[float]:
    """Serialise an index to a temp file and return its size in MB (or None)."""
    idx = getattr(backend, "index", None)
    if idx is None:
        return None
    module = type(idx).__module__
    fd, path = tempfile.mkstemp(suffix=".idx")
    os.close(fd)
    try:
        if "faiss" in module:
            import faiss  # noqa: PLC0415

            faiss.write_index(idx, path)
        elif "hnswlib" in module:
            idx.save_index(path)
        elif hasattr(idx, "save"):
            idx.save(path)  # vectro HNSWIndex
        else:
            return None
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    except (OSError, RuntimeError, ValueError):
        return None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def measure_qps(
    query_fn: Callable[[np.ndarray, int, int], List[np.ndarray]],
    queries: np.ndarray,
    k: int,
    param: int,
) -> float:
    """Single-thread QPS = n_queries / best-of-N batch time (warm-up discarded)."""
    for _ in range(N_WARMUP):
        query_fn(queries, k, param)
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        query_fn(queries, k, param)
        times.append(time.perf_counter() - t0)
    best = min(times)
    return len(queries) / best if best > 0 else 0.0


def sweep_pareto(
    backend: Any, queries: np.ndarray, gt: np.ndarray, k: int
) -> List[Dict[str, float]]:
    """Sweep the backend's param and return (recall, qps) operating points."""
    points: List[Dict[str, float]] = []
    for param in backend.param_sweep:
        res = backend.query_batch(queries, k, param)
        recall = batch_recall(res, gt, k)
        qps = measure_qps(backend.query_batch, queries, k, param)
        points.append({"param": float(param), "recall": round(recall, 4), "qps": round(qps, 1)})
        # Exact baseline has a single operating point; stop once recall saturates.
        if recall >= 0.999 and backend.param_name == "—":
            break
    return points


def qps_at_recall(points: List[Dict[str, float]], target: float) -> Optional[float]:
    """Best QPS among operating points meeting the recall target."""
    ok = [p["qps"] for p in points if p["recall"] >= target]
    return max(ok) if ok else None


# ── search head-to-head ──────────────────────────────────────────────────────────


def run_search(train: np.ndarray, queries: np.ndarray, gt: np.ndarray) -> List[Dict[str, Any]]:
    """Build every backend, sweep recall/QPS, record build time + index size."""
    backends: List[Any] = [VectroHNSW(), FaissHNSW(), FaissIVF(), HnswlibHNSW(), ExactFlat()]
    results: List[Dict[str, Any]] = []

    for backend in backends:
        label = backend.label
        try:
            t0 = time.perf_counter()
            backend.build(train)
            build_s = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 — a missing competitor must not abort the suite
            log.warning("[%s] build failed: %s", label, exc)
            continue

        size_mb = measure_index_size_mb(backend)
        points = sweep_pareto(backend, queries, gt, K)
        best_recall = max((p["recall"] for p in points), default=0.0)
        entry = {
            "label": label,
            "build_s": round(build_s, 3),
            "index_mb": size_mb,
            "max_recall": round(best_recall, 4),
            "pareto": points,
            "qps_at_recall": {f"{t:.2f}": qps_at_recall(points, t) for t in RECALL_TARGETS},
        }
        results.append(entry)
        log.info(
            "[%s] build=%.2fs  size=%sMB  maxR@10=%.3f  qps@R0.90=%s",
            label,
            build_s,
            size_mb,
            best_recall,
            entry["qps_at_recall"]["0.90"],
        )
    return results


# ── quantization head-to-head ────────────────────────────────────────────────────


def _mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float(np.mean(num / den))


def _best_throughput(fn: Callable[[], Any], n: int) -> float:
    for _ in range(2):
        fn()
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    return n / best if best > 0 else 0.0


def _bench_vectro_int8(vectors: np.ndarray) -> Dict[str, Any]:
    import vectro_py  # noqa: PLC0415

    v = np.ascontiguousarray(vectors, dtype=np.float32)
    codes, scales = vectro_py.quantize_int8_batch(v)
    recon = np.asarray(vectro_py.dequantize_int8_batch(codes, scales))
    ratio = v.nbytes / (np.asarray(codes).nbytes + np.asarray(scales).nbytes)
    tput = _best_throughput(lambda: vectro_py.quantize_int8_batch(v), len(v))
    return {
        "method": "vectro-int8 (rust simd)",
        "throughput_vec_s": int(tput),
        "compression_ratio": round(float(ratio), 2),
        "reconstruction_cosine": round(_mean_cosine(v, recon), 5),
    }


def _bench_faiss_sq(vectors: np.ndarray) -> Optional[Dict[str, Any]]:
    try:
        import faiss  # noqa: PLC0415
    except ImportError:
        return None
    v = np.ascontiguousarray(vectors, dtype=np.float32)
    d = int(v.shape[1])
    index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
    index.train(v[: min(10_000, len(v))])

    def _encode() -> None:
        index.reset()
        index.add(v)

    tput = _best_throughput(_encode, len(v))
    index.reset()
    index.add(v)
    recon = index.reconstruct_n(0, len(v))
    ratio = v.nbytes / (len(v) * d)  # QT_8bit = d bytes/vector
    return {
        "method": "faiss-scalarquantizer-int8",
        "throughput_vec_s": int(tput),
        "compression_ratio": round(float(ratio), 2),
        "reconstruction_cosine": round(_mean_cosine(v, recon), 5),
    }


def _bench_vectro_pq(vectors: np.ndarray, m: int) -> Optional[Dict[str, Any]]:
    try:
        from python.v3_api import PQCodebook  # noqa: PLC0415
    except ImportError:
        return None
    v = np.ascontiguousarray(vectors, dtype=np.float32)
    half = len(v) // 2
    cb = PQCodebook.train(v[:half], n_subspaces=m)
    codes = cb.encode(v[half:])
    recon = cb.decode(codes)
    ratio = v[half:].nbytes / np.asarray(codes).nbytes
    tput = _best_throughput(lambda: cb.encode(v[half:]), len(v) - half)
    return {
        "method": f"vectro-pq (M={m})",
        "throughput_vec_s": int(tput),
        "compression_ratio": round(float(ratio), 2),
        "reconstruction_cosine": round(_mean_cosine(v[half:], recon), 5),
    }


def _bench_faiss_pq(vectors: np.ndarray, m: int) -> Optional[Dict[str, Any]]:
    try:
        import faiss  # noqa: PLC0415
    except ImportError:
        return None
    v = np.ascontiguousarray(vectors, dtype=np.float32)
    d = int(v.shape[1])
    half = len(v) // 2
    index = faiss.IndexPQ(d, m, 8)
    index.train(v[:half])

    def _encode() -> None:
        index.reset()
        index.add(v[half:])

    tput = _best_throughput(_encode, len(v) - half)
    index.reset()
    index.add(v[half:])
    recon = index.reconstruct_n(0, len(v) - half)
    ratio = v[half:].nbytes / (len(v[half:]) * m)  # m bytes/vector at 8 bits
    return {
        "method": f"faiss-indexpq (M={m})",
        "throughput_vec_s": int(tput),
        "compression_ratio": round(float(ratio), 2),
        "reconstruction_cosine": round(_mean_cosine(v[half:], recon), 5),
    }


def run_quantization(vectors: np.ndarray) -> List[Dict[str, Any]]:
    """INT8 + PQ encode throughput / compression / quality head-to-head."""
    d = int(vectors.shape[1])
    # Pick a PQ subspace count that divides d (FAISS requires d % M == 0).
    m = next((cand for cand in (d // 8, d // 4, d // 2, 8, 4, 2) if cand >= 1 and d % cand == 0), 1)

    rows: List[Optional[Dict[str, Any]]] = [
        _bench_vectro_int8(vectors),
        _bench_faiss_sq(vectors),
        _bench_vectro_pq(vectors, m),
        _bench_faiss_pq(vectors, m),
    ]
    out = [r for r in rows if r is not None]
    for r in out:
        log.info(
            "[quant] %-30s %12d vec/s  ratio=%5.1fx  cosine=%.4f",
            r["method"],
            r["throughput_vec_s"],
            r["compression_ratio"],
            r["reconstruction_cosine"],
        )
    return out


# ── plots ────────────────────────────────────────────────────────────────────────


def plot_pareto(search: List[Dict[str, Any]], dataset: str, out_path: Path) -> bool:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(7, 5))
    for entry in search:
        pts = sorted(entry["pareto"], key=lambda p: p["recall"])
        recalls = [p["recall"] for p in pts]
        qps = [max(p["qps"], 1e-6) for p in pts]
        if entry["label"] == "exact-faiss":
            ax.scatter(recalls, qps, marker="*", s=160, label=entry["label"], zorder=5)
        else:
            ax.plot(recalls, qps, marker="o", label=entry["label"])
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (single-thread, log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Recall vs QPS — {dataset}")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_quant(quant: List[Dict[str, Any]], dataset: str, out_path: Path) -> bool:
    if not quant:
        return False
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return False

    labels = [r["method"] for r in quant]
    tputs = [r["throughput_vec_s"] for r in quant]
    colors = ["#2a9d8f" if "vectro" in m else "#e76f51" for m in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, tputs, color=colors)
    ax.set_xlabel("Encode throughput (vec/s, log scale)")
    ax.set_xscale("log")
    ax.set_title(f"Quantization encode throughput — {dataset}")
    for i, val in enumerate(tputs):
        ax.text(val, i, f" {val:,}", va="center", fontsize=8)
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ── report ───────────────────────────────────────────────────────────────────────


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    ds = report["dataset"]
    lines.append(
        f"# Vectro Comprehensive Benchmark — {ds} (n={report['n_train']:,}, d={report['d']})"
    )
    lines.append("")
    lines.append(
        f"_Single-thread. FAISS SIMD build: `{report['faiss_simd']}`. "
        f"{report['hardware']['platform']}_"
    )
    lines.append("")
    lines.append("## ANN Search — Recall@10 vs QPS (single-thread)")
    lines.append("")
    targets = sorted({t for e in report["search"] for t in e["qps_at_recall"]})

    def _f(x: Optional[float]) -> str:
        return f"{x:,.0f}" if x else "—"

    qps_cols = "".join(f" QPS@R{t} |" for t in targets)
    lines.append(f"| Backend | Build (s) | Index MB | Max R@10 |{qps_cols}")
    lines.append("|:--|--:|--:|--:|" + "--:|" * len(targets))
    for e in report["search"]:
        q = e["qps_at_recall"]
        qps_cells = "".join(f" {_f(q.get(t))} |" for t in targets)
        lines.append(
            f"| {e['label']} | {e['build_s']:.2f} | {e['index_mb'] or '—'} | "
            f"{e['max_recall']:.3f} |{qps_cells}"
        )
    lines.append("")
    lines.append("## Quantization — encode throughput / compression / quality")
    lines.append("")
    lines.append("| Method | Throughput (vec/s) | Compression | Reconstruction cosine |")
    lines.append("|:--|--:|--:|--:|")
    for r in report["quantization"]:
        lines.append(
            f"| {r['method']} | {r['throughput_vec_s']:,} | "
            f"{r['compression_ratio']:.1f}x | {r['reconstruction_cosine']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vectro comprehensive head-to-head benchmark")
    p.add_argument("--quick", action="store_true", help="glove-25-angular at n=10k")
    p.add_argument("--dataset", default="glove-25-angular", choices=sorted(DATASETS))
    p.add_argument("--n", type=int, default=50_000, help="number of train vectors to index")
    p.add_argument("--data-dir", default=str(_PROJECT_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_PROJECT_ROOT / "benchmarks" / "results"))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    dataset = "glove-25-angular" if args.quick else args.dataset
    n_train = 10_000 if args.quick else args.n
    n_train = min(n_train, DATASETS[dataset]["n_train_full"])

    faiss_simd = pin_single_thread()
    log.info("FAISS SIMD build: %s", faiss_simd)

    import h5py  # noqa: PLC0415

    data_dir = Path(args.data_dir)
    download_dataset(dataset, data_dir)
    with h5py.File(data_dir / DATASETS[dataset]["filename"], "r") as fh:
        train = fh["train"][:n_train].astype(np.float32)
        queries = fh["test"][: min(N_QUERIES, len(fh["test"]))].astype(np.float32)
    log.info("Loaded %s: train %s  queries %s", dataset, train.shape, queries.shape)

    log.info("Computing brute-force ground truth ...")
    gt = compute_exact_gt(train, queries, K)

    log.info("── ANN search head-to-head ──")
    search = run_search(train, queries, gt)
    log.info("── Quantization head-to-head ──")
    quant = run_quantization(train)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{timestamp}_comprehensive"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "timestamp": timestamp,
        "dataset": dataset,
        "n_train": int(n_train),
        "d": int(train.shape[1]),
        "k": K,
        "metric": "cosine",
        "faiss_simd": faiss_simd,
        "single_thread": True,
        "hardware": hardware_meta(),
        "config": {
            "HNSW_M": HNSW_M,
            "HNSW_EF_CONSTRUCTION": HNSW_EF_CONSTRUCTION,
            "N_QUERIES": N_QUERIES,
            "N_WARMUP": N_WARMUP,
            "N_REPS": N_REPS,
            "EF_SWEEP": EF_SWEEP,
            "NPROBE_SWEEP": NPROBE_SWEEP,
        },
        "search": search,
        "quantization": quant,
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    markdown = render_markdown(report)
    (out_dir / "report.md").write_text(markdown)
    if plot_pareto(search, dataset, out_dir / "recall_vs_qps.png"):
        log.info("Wrote recall_vs_qps.png")
    if plot_quant(quant, dataset, out_dir / "quant_throughput.png"):
        log.info("Wrote quant_throughput.png")

    print("\n" + markdown)
    print(f"\nArtifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
