"""Smoke tests for the comprehensive head-to-head benchmark harness.

These exercise the search backends, quantization comparison, plotting, and
report rendering on tiny synthetic data — no network or dataset download — so
the harness can't silently drift out of sync with the vectro/FAISS APIs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import tests._path_setup as _path_setup  # noqa: F401

_path_setup.ensure_repo_root_on_path()

import scripts.benchmark_comprehensive as bc  # noqa: E402


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


requires_faiss = pytest.mark.skipif(not _has("faiss"), reason="faiss-cpu not installed")
requires_vectro_py = pytest.mark.skipif(
    not _has("vectro_py"), reason="vectro_py Rust extension not installed"
)


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(7)
    train = rng.standard_normal((2000, 32)).astype(np.float32)
    queries = rng.standard_normal((50, 32)).astype(np.float32)
    gt = bc.compute_exact_gt(train, queries, bc.K)
    return train, queries, gt


def test_pin_single_thread_returns_tag():
    tag = bc.pin_single_thread()
    assert isinstance(tag, str) and tag


def test_vectro_backend_recall_and_size(tiny):
    train, queries, gt = tiny
    backend = bc.VectroHNSW()
    backend.param_sweep = [16, 64, 256]  # short sweep keeps the pure-Python path fast in CI
    backend.build(train)
    pts = bc.sweep_pareto(backend, queries, gt, bc.K)
    assert pts, "expected at least one operating point"
    assert max(p["recall"] for p in pts) >= 0.80
    # vectro HNSW serialises, so index size must be measurable.
    assert bc.measure_index_size_mb(backend) is not None


@requires_faiss
def test_all_search_backends_build_and_recall(tiny):
    train, queries, gt = tiny
    results = bc.run_search(train, queries, gt)
    labels = {e["label"] for e in results}
    # Every competitor (incl. hnswlib + exact baseline) must run.
    assert {"vectro-hnsw", "faiss-hnsw", "faiss-ivf", "hnswlib", "exact-faiss"} <= labels
    exact = next(e for e in results if e["label"] == "exact-faiss")
    assert exact["max_recall"] >= 0.999  # exact flat == ground truth


@requires_vectro_py
def test_vectro_int8_quant_beats_baseline_quality(tiny):
    train, _, _ = tiny
    row = bc._bench_vectro_int8(train)
    assert row["throughput_vec_s"] > 0
    assert row["compression_ratio"] > 3.0
    assert row["reconstruction_cosine"] >= 0.99  # INT8 is near-lossless


@requires_faiss
def test_quantization_table_has_vectro_and_faiss(tiny):
    train, _, _ = tiny
    rows = bc.run_quantization(train)
    methods = {r["method"] for r in rows}
    assert any("vectro-int8" in m for m in methods)
    assert any("faiss-scalarquantizer" in m for m in methods)
    for r in rows:
        assert r["throughput_vec_s"] > 0
        assert 0.0 <= r["reconstruction_cosine"] <= 1.0001


def test_render_markdown_is_robust_to_target_set(tiny):
    train, queries, gt = tiny
    search = [
        {
            "label": "vectro-hnsw",
            "build_s": 1.0,
            "index_mb": 0.5,
            "max_recall": 0.95,
            "pareto": [{"param": 64, "recall": 0.95, "qps": 1000.0}],
            "qps_at_recall": {"0.90": 1000.0},  # only one target present
        }
    ]
    report = {
        "dataset": "synthetic",
        "n_train": 2000,
        "d": 32,
        "faiss_simd": "generic",
        "hardware": bc.hardware_meta(),
        "search": search,
        "quantization": [],
    }
    md = bc.render_markdown(report)
    assert "QPS@R0.90" in md and "vectro-hnsw" in md


def test_plots_write_files(tmp_path):
    # Ensure plotting produces PNG files when matplotlib is available.
    if not _has("matplotlib"):
        pytest.skip("matplotlib not installed")
    out = Path(tmp_path)
    search = [{"label": "a", "pareto": [{"recall": 0.9, "qps": 100.0}], "index_mb": 1.0}]
    quant = [
        {
            "method": "vectro-int8",
            "throughput_vec_s": 1000,
            "compression_ratio": 4.0,
            "reconstruction_cosine": 0.999,
        }
    ]
    assert bc.plot_pareto(search, "synthetic", out / "p.png")
    assert (out / "p.png").exists()
    assert bc.plot_quant(quant, "synthetic", out / "q.png")
    assert (out / "q.png").exists()
