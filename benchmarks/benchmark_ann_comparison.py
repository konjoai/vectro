#!/usr/bin/env python
"""
ANN Library Comparison — Vectro HNSW vs hnswlib vs annoy vs usearch

Measures recall@K and QPS for approximate nearest-neighbour search
across four libraries on the same synthetic dataset.  Each library is
imported inside a try/except; if it is not installed the benchmark
reports "not installed" and continues with the remaining libraries.

Ground truth is computed by exact brute-force (numpy L2 + argsort).

Usage:
    python benchmarks/benchmark_ann_comparison.py
    python benchmarks/benchmark_ann_comparison.py --output results/ann_comparison.json
    python benchmarks/benchmark_ann_comparison.py --n 50000 --d 128 --k 10

Dependencies (optional — install for full comparison):
    pip install "vectro[bench-ann]"
    # or individually:
    pip install hnswlib>=0.8.0 annoy>=1.17.3 usearch>=2.9.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Ground truth (brute-force exact search)
# ---------------------------------------------------------------------------


def brute_force_knn(
    corpus: np.ndarray,
    queries: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return exact k-NN indices for each query row (inner-product / cosine).

    Vectors are L2-normalised before dot-product, making this equivalent
    to cosine similarity search.

    Args:
        corpus:  (n, d) float32 corpus vectors.
        queries: (q, d) float32 query vectors.
        k:       Number of neighbours to return.
    Returns:
        (q, k) int64 array of corpus indices, nearest first.
    """
    # Normalise both matrices so dot-product == cosine similarity.
    corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True).clip(min=1e-8)
    queries_norms = np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-8)
    corpus_n = corpus / corpus_norms
    queries_n = queries / queries_norms

    # Compute similarity in batches to avoid OOM on large datasets.
    batch = 1000
    all_ids = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), batch):
        end = min(start + batch, len(queries))
        sims = queries_n[start:end] @ corpus_n.T  # (batch, n)
        partitioned = np.argpartition(sims, -k, axis=1)[:, -k:]
        for i, row_ids in enumerate(partitioned):
            sorted_ids = row_ids[np.argsort(sims[i, row_ids])[::-1]]
            all_ids[start + i] = sorted_ids
    return all_ids


def recall_at_k(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Compute Recall@k: fraction of queries where the true nearest neighbour
    appears in the top-k predicted results.

    Args:
        predictions:  (q, k) predicted indices.
        ground_truth: (q, 1) or (q, k) exact indices (use column 0 = nearest).
        k:            Top-k cutoff to evaluate.
    Returns:
        Scalar recall in [0, 1].
    """
    gt = ground_truth[:, 0:1]  # nearest only
    pred_k = predictions[:, :k]
    hits = (pred_k == gt).any(axis=1)
    return float(hits.mean())


# ---------------------------------------------------------------------------
# Per-library wrappers
# ---------------------------------------------------------------------------


def _build_vectro(
    corpus: np.ndarray,
    m: int,
    ef_construction: int,
) -> tuple[Any, int]:
    """Build a Vectro HNSW index. Returns (index, byte_size)."""
    from python.hnsw_api import HNSWIndex  # type: ignore[import]

    idx = HNSWIndex(M=m, ef_construction=ef_construction, space="cosine")
    idx.add(corpus)
    # Approximate memory: n * d * 4 bytes for float32 vectors + graph links
    byte_size = len(corpus) * corpus.shape[1] * 4 + len(corpus) * m * 2 * 8
    return idx, byte_size


def _query_vectro(idx: Any, queries: np.ndarray, k: int, ef_search: int) -> np.ndarray:
    """Query a Vectro HNSW index. Returns (q, k) int64 index array."""
    results = np.empty((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        indices, _ = idx.search(q, k=k, ef=ef_search)
        # pad with -1 if fewer than k results returned
        if len(indices) < k:
            pad = np.full(k - len(indices), -1, dtype=np.int64)
            indices = np.concatenate([indices, pad])
        results[i] = indices[:k]
    return results


def _build_hnswlib(
    corpus: np.ndarray,
    m: int,
    ef_construction: int,
) -> tuple[Any, int]:
    import hnswlib  # type: ignore[import]

    idx = hnswlib.Index(space="cosine", dim=corpus.shape[1])
    idx.init_index(max_elements=len(corpus), ef_construction=ef_construction, M=m)
    idx.add_items(corpus, list(range(len(corpus))))
    byte_size = (
        getattr(idx, "get_current_count", lambda: 0)() * corpus.shape[1] * 4 * 2
    )  # rough estimate
    return idx, byte_size


def _query_hnswlib(idx: Any, queries: np.ndarray, k: int, ef_search: int) -> np.ndarray:
    idx.set_ef(ef_search)
    labels, _ = idx.knn_query(queries, num_threads=1, k=k)
    return np.array(labels, dtype=np.int64)


def _build_annoy(
    corpus: np.ndarray,
    n_trees: int,
) -> tuple[Any, int]:
    from annoy import AnnoyIndex  # type: ignore[import]

    d = corpus.shape[1]
    idx = AnnoyIndex(d, "angular")
    for i, vec in enumerate(corpus):
        idx.add_item(i, vec.tolist())
    idx.build(n_trees)
    byte_size = n_trees * len(corpus) * 4  # rough estimate
    return idx, byte_size


def _query_annoy(idx: Any, queries: np.ndarray, k: int, search_k: int) -> np.ndarray:
    results = [idx.get_nns_by_vector(q.tolist(), k, search_k=search_k) for q in queries]
    arr = np.full((len(queries), k), -1, dtype=np.int64)
    for i, r in enumerate(results):
        arr[i, : len(r)] = r
    return arr


def _build_usearch(
    corpus: np.ndarray,
    m: int,
    ef_construction: int,
) -> tuple[Any, int]:
    from usearch.index import Index as UsearchIndex  # type: ignore[import]

    idx = UsearchIndex(
        ndim=corpus.shape[1], metric="cos", connectivity=m, expansion_add=ef_construction
    )
    idx.add(np.arange(len(corpus), dtype=np.int64), corpus)
    byte_size = getattr(idx, "memory_usage", 0)
    if callable(byte_size):
        byte_size = byte_size()
    return idx, byte_size


def _query_usearch(idx: Any, queries: np.ndarray, k: int, ef_search: int) -> np.ndarray:
    matches = idx.search(queries, count=k, expansion=ef_search)
    results = np.full((len(queries), k), -1, dtype=np.int64)
    for i, m in enumerate(matches):
        ids = np.array(m.keys, dtype=np.int64)
        results[i, : len(ids)] = ids
    return results


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    n: int = 100_000,
    d: int = 128,
    q: int = 1_000,
    k: int = 10,
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 100,
    n_trees: int = 10,
    output_path: Path = Path("results/ann_comparison.json"),
) -> dict[str, Any]:
    """Run the full ANN comparison benchmark.

    Args:
        n:              Number of corpus vectors.
        d:              Vector dimensionality.
        q:              Number of query vectors.
        k:              K for recall@K evaluation.
        m:              HNSW M parameter.
        ef_construction: HNSW ef_construction parameter.
        ef_search:      HNSW ef_search parameter.
        n_trees:        Annoy number of trees.
        output_path:    Where to write the JSON results.
    Returns:
        Results dict written to output_path.
    """
    print("=" * 70)
    print("ANN Library Comparison")
    print(f"n={n:,}  d={d}  q={q:,}  k={k}  M={m}  ef_construction={ef_construction}")
    print("=" * 70)

    rng = np.random.default_rng(42)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    queries = rng.standard_normal((q, d)).astype(np.float32)

    # ── Ground truth ──────────────────────────────────────────────────────────
    print("\nComputing brute-force ground truth...", end="", flush=True)
    gt_start = time.perf_counter()
    ground_truth = brute_force_knn(corpus, queries, k)
    gt_sec = time.perf_counter() - gt_start
    print(f" done ({gt_sec:.1f}s, exact search)")

    results: dict[str, Any] = {
        "benchmark": "ann_comparison",
        "config": {
            "n_corpus": n,
            "d": d,
            "n_queries": q,
            "k": k,
            "hnsw_M": m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "n_trees_annoy": n_trees,
        },
        "libraries": {},
    }

    # ── Library configurations ────────────────────────────────────────────────
    libraries: list[dict[str, Any]] = [
        {
            "name": "Vectro HNSW",
            "build": lambda: _build_vectro(corpus, m, ef_construction),
            "query": lambda idx: _query_vectro(idx, queries, k, ef_search),
        },
        {
            "name": "hnswlib",
            "build": lambda: _build_hnswlib(corpus, m, ef_construction),
            "query": lambda idx: _query_hnswlib(idx, queries, k, ef_search),
        },
        {
            "name": "annoy",
            "build": lambda: _build_annoy(corpus, n_trees),
            "query": lambda idx: _query_annoy(idx, queries, k, search_k=n_trees * k * 5),
        },
        {
            "name": "usearch",
            "build": lambda: _build_usearch(corpus, m, ef_construction),
            "query": lambda idx: _query_usearch(idx, queries, k, ef_search),
        },
    ]

    # ── Run each library ─────────────────────────────────────────────────────
    print(f"\n{'Library':<18} {'Build(s)':>8} {'QPS':>8} {'R@1':>6} {'R@5':>6} {'R@10':>6}")
    print("-" * 60)

    for lib in libraries:
        name = lib["name"]
        try:
            # Build
            t0 = time.perf_counter()
            idx, byte_size = lib["build"]()
            build_sec = time.perf_counter() - t0

            # Warmup query run (not timed)
            _ = lib["query"](idx)

            # Timed query run
            t0 = time.perf_counter()
            predictions = lib["query"](idx)
            query_sec = time.perf_counter() - t0
            qps = q / query_sec if query_sec > 0 else 0.0

            r1 = recall_at_k(predictions, ground_truth, 1)
            r5 = recall_at_k(predictions, ground_truth, 5)
            r10 = recall_at_k(predictions, ground_truth, 10)

            print(f"{name:<18} {build_sec:>8.2f} {qps:>8,.0f} {r1:>6.3f} {r5:>6.3f} {r10:>6.3f}")

            results["libraries"][name] = {
                "status": "ok",
                "build_sec": round(build_sec, 3),
                "index_bytes": byte_size,
                "qps": round(qps, 1),
                "recall_at_1": round(r1, 4),
                "recall_at_5": round(r5, 4),
                "recall_at_10": round(r10, 4),
            }

        except ImportError:
            print(f"{name:<18} {'not installed':>55}")
            results["libraries"][name] = {"status": "not_installed"}
        except Exception as exc:
            msg = str(exc)[:80]
            print(f"{name:<18} {'ERROR: ' + msg:>55}")
            results["libraries"][name] = {"status": "error", "message": msg}

    # ── Exact BF baseline line ────────────────────────────────────────────────
    bf_qps = q / gt_sec if gt_sec > 0 else 0.0
    print(f"{'Exact (BF)':<18} {'n/a':>8} {bf_qps:>8,.0f} {'1.000':>6} {'1.000':>6} {'1.000':>6}")
    results["exact_brute_force"] = {
        "build_sec": 0,
        "qps": round(bf_qps, 1),
        "recall_at_1": 1.0,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
    }

    print()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ANN library recall@K and QPS comparison")
    parser.add_argument("--n", type=int, default=100_000, help="Corpus size (default: 100000)")
    parser.add_argument("--d", type=int, default=128, help="Vector dimensions (default: 128)")
    parser.add_argument(
        "--q", type=int, default=1_000, help="Number of query vectors (default: 1000)"
    )
    parser.add_argument("--k", type=int, default=10, help="K for recall@K (default: 10)")
    parser.add_argument("--m", type=int, default=16, help="HNSW M parameter (default: 16)")
    parser.add_argument(
        "--ef-construction", type=int, default=200, help="HNSW ef_construction (default: 200)"
    )
    parser.add_argument("--ef-search", type=int, default=100, help="HNSW ef_search (default: 100)")
    parser.add_argument(
        "--n-trees", type=int, default=10, help="Annoy number of trees (default: 10)"
    )
    parser.add_argument("--output", type=str, default="results/ann_comparison.json")
    args = parser.parse_args()

    run_benchmark(
        n=args.n,
        d=args.d,
        q=args.q,
        k=args.k,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        n_trees=args.n_trees,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
