"""Parity and behaviour tests for the native Rust HNSW backend.

These verify that ``HNSWIndex(backend="auto"/"rust")`` — which delegates the
build and search hot paths to ``vectro_py.PyHnswIndex`` — matches the
pure-Python baseline numerically and preserves the full feature surface
(metadata filtering, soft-delete, upsert, persistence, trace/stats/compact).
"""

from __future__ import annotations

import numpy as np
import pytest

from python.hnsw_api import HNSWIndex
from python.hnsw_rust import rust_available

pytestmark = pytest.mark.skipif(
    not rust_available(), reason="compiled vectro_py extension not available"
)


def _data(n: int = 800, d: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def _brute_force_gt(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    def unit(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    sims = unit(queries) @ unit(corpus).T
    return np.argsort(-sims, axis=1)[:, :k]


def _recall(idx: HNSWIndex, queries: np.ndarray, gt: np.ndarray, k: int, ef: int) -> float:
    hits = 0
    for i, q in enumerate(queries):
        ids, _ = idx.search(q, k=k, ef=ef)
        hits += len(set(int(x) for x in ids) & set(int(x) for x in gt[i]))
    return hits / (len(queries) * k)


# ─────────────────────────── backend selection ────────────────────────────


def test_auto_selects_rust_for_cosine():
    assert HNSWIndex(space="cosine", backend="auto").backend == "rust"


def test_l2_falls_back_to_python():
    # The native core is cosine-only.
    assert HNSWIndex(space="l2", backend="auto").backend == "python"


def test_forced_python_backend():
    assert HNSWIndex(space="cosine", backend="python").backend == "python"


def test_rust_backend_rejects_l2():
    with pytest.raises(ValueError):
        HNSWIndex(space="l2", backend="rust")


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        HNSWIndex(backend="bogus")


# ─────────────────────────── build + search parity ────────────────────────


def test_build_search_recall_matches_python():
    corpus, queries = _data(800, 64, seed=1), _data(50, 64, seed=2)
    gt = _brute_force_gt(corpus, queries, k=10)

    rust = HNSWIndex(M=16, ef_construction=200, backend="rust")
    rust.add(corpus)
    py = HNSWIndex(M=16, ef_construction=200, backend="python")
    py.add(corpus)

    r_rust = _recall(rust, queries, gt, k=10, ef=200)
    r_py = _recall(py, queries, gt, k=10, ef=200)

    assert r_rust >= 0.90
    # Numerically on par with the baseline (same approximate algorithm).
    assert abs(r_rust - r_py) <= 0.05


def test_distances_are_nonnegative_and_sorted():
    corpus = _data(400, 32, seed=3)
    idx = HNSWIndex(backend="rust")
    idx.add(corpus)
    ids, dists = idx.search(corpus[0], k=10, ef=100)
    assert ids[0] == 0  # a vector is its own nearest neighbour
    assert dists[0] == pytest.approx(0.0, abs=1e-5)
    assert np.all(dists >= 0.0)
    assert np.all(np.diff(dists) >= -1e-6)  # ascending


# ─────────────────────────── feature surface ──────────────────────────────


def test_metadata_filter_on_rust():
    corpus = _data(300, 32, seed=4)
    idx = HNSWIndex(backend="rust")
    meta = [{"tag": "a" if i % 2 == 0 else "b"} for i in range(len(corpus))]
    idx.add(corpus, metadata=meta)

    ids, _ = idx.search(corpus[0], k=20, ef=100, filter={"tag": "a"})
    assert len(ids) > 0
    assert all(int(i) % 2 == 0 for i in ids)


def test_delete_excludes_from_results_on_rust():
    corpus = _data(300, 32, seed=5)
    idx = HNSWIndex(backend="rust")
    idx.add(corpus)
    idx.delete(0)
    ids, _ = idx.search(corpus[0], k=10, ef=100)
    assert 0 not in [int(i) for i in ids]


def test_add_batch_upsert_update_reflected_in_search():
    corpus = _data(200, 32, seed=6)
    idx = HNSWIndex(backend="rust")
    idx.add_batch(corpus, ids=[f"v{i}" for i in range(len(corpus))])

    # Overwrite v0 with a copy of v5's vector; querying v5 should now surface v0.
    res = idx.add_batch(corpus[5:6], ids=["v0"])
    assert res["updated"] == 1
    ids, _ = idx.search(corpus[5], k=5, ef=100)
    assert 0 in [int(i) for i in ids]


def test_trace_on_rust_backend_builds_python_graph():
    corpus = _data(200, 32, seed=7)
    idx = HNSWIndex(backend="rust")
    idx.add(corpus)
    ids, dists, trace = idx.search(corpus[0], k=10, ef=100, trace=True)
    assert len(ids) == 10
    assert trace.entry_point >= 0


def test_stats_on_rust_backend():
    corpus = _data(300, 32, seed=8)
    idx = HNSWIndex(backend="rust")
    idx.add(corpus)
    idx.delete(1)
    s = idx.stats()
    assert s["n_total"] == 300
    assert s["n_alive"] == 299
    assert s["n_deleted"] == 1


def test_compact_switches_to_python_backend():
    corpus = _data(200, 32, seed=9)
    idx = HNSWIndex(backend="rust")
    idx.add(corpus)
    idx.delete(0)
    idx.delete(1)
    idx.compact()
    assert idx.backend == "python"
    # Still searchable and deleted nodes stay gone.
    ids, _ = idx.search(corpus[2], k=10, ef=100)
    assert {0, 1}.isdisjoint(int(i) for i in ids)


def test_estimate_recall_on_rust():
    corpus = _data(500, 48, seed=10)
    idx = HNSWIndex(backend="rust")
    idx.add(corpus)
    out = idx.estimate_recall(sample_size=100, k=10, ef=200)
    assert out["recall"] >= 0.85


# ─────────────────────────── persistence ──────────────────────────────────


def test_save_load_roundtrip_rust(tmp_path):
    corpus, queries = _data(400, 32, seed=11), _data(20, 32, seed=12)
    idx = HNSWIndex(M=16, ef_construction=200, backend="rust")
    meta = [{"i": i} for i in range(len(corpus))]
    idx.add(corpus, metadata=meta)
    idx.delete(3)

    before = [idx.search(q, k=10, ef=150)[0].tolist() for q in queries]

    path = tmp_path / "index.vindex"
    idx.save(str(path))
    loaded = HNSWIndex.load(str(path))

    assert loaded.backend == "rust"
    assert len(loaded) == 400
    assert loaded._metadata[7] == {"i": 7}
    assert 3 in loaded._deleted

    after = [loaded.search(q, k=10, ef=150)[0].tolist() for q in queries]
    # Deterministic native graph → identical results after reload.
    assert before == after
