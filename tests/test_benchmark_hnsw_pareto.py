"""Unit tests for the HNSW recall–QPS Pareto benchmark's pure helpers.

The benchmark's heavy paths (building indexes, timing) need optional native
deps and a dataset, so they are exercised by running the script. These tests
pin the deterministic, dependency-free helpers that the reported numbers depend
on: ground-truth correctness, recall computation, unit-norm, and the
iso-recall interpolation that drives the headline comparison.
"""

from __future__ import annotations

import numpy as np

from benchmarks.benchmark_hnsw_pareto import (
    _unit,
    brute_force_gt,
    qps_at_recall,
    recall_at_k,
)


def test_unit_rows_have_unit_norm():
    x = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    u = _unit(x)
    assert np.allclose(np.linalg.norm(u[0]), 1.0, atol=1e-5)
    assert np.linalg.norm(u[1]) == 0.0  # zero row stays zero (eps guard)


def test_brute_force_gt_is_exact():
    rng = np.random.default_rng(0)
    train = _unit(rng.standard_normal((200, 16)).astype(np.float32))
    queries = _unit(rng.standard_normal((10, 16)).astype(np.float32))
    gt = brute_force_gt(train, queries, k=5)
    assert gt.shape == (10, 5)
    # First column must be the exact argmax cosine for each query.
    sims = queries @ train.T
    assert (gt[:, 0] == np.argmax(sims, axis=1)).all()


def test_recall_at_k_bounds():
    gt = np.array([[1, 2, 3], [4, 5, 6]])
    # Perfect prediction → 1.0
    assert recall_at_k(gt.copy(), gt, 3) == 1.0
    # Disjoint prediction → 0.0
    bad = np.array([[7, 8, 9], [10, 11, 12]])
    assert recall_at_k(bad, gt, 3) == 0.0
    # Half overlap on row 0, none on row 1 → mean(1/2, 0) = 0.25
    part = np.array([[1, 99], [98, 97]])
    assert abs(recall_at_k(part, gt, 2) - 0.25) < 1e-9


def test_qps_at_recall_interpolates_and_clamps():
    curve = [
        {"ef": 20, "recall": 0.70, "qps": 100_000.0},
        {"ef": 60, "recall": 0.86, "qps": 50_000.0},
        {"ef": 100, "recall": 0.92, "qps": 25_000.0},
    ]
    # Exact endpoint.
    assert qps_at_recall(curve, 0.70) == 100_000.0
    # Midpoint interpolation between 0.86 and 0.92.
    mid = qps_at_recall(curve, 0.89)
    assert 25_000.0 < mid < 50_000.0
    # Out of range → None (cannot extrapolate).
    assert qps_at_recall(curve, 0.99) is None
    assert qps_at_recall(curve, 0.50) is None
    assert qps_at_recall([], 0.9) is None
