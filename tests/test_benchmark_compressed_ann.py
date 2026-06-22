"""Unit tests for the compressed-ANN tradeoff benchmark's pure helpers.

The build/measure paths need the native extension; these tests pin the
deterministic helpers the reported recall/memory numbers depend on.
"""

from __future__ import annotations

import numpy as np

from benchmarks.benchmark_compressed_ann import (
    _pad,
    brute_force_gt,
    graph_bytes,
    recall_at_k,
    unit,
)


def test_unit_normalises_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    u = unit(x)
    assert np.allclose(np.linalg.norm(u[0]), 1.0, atol=1e-5)
    assert np.linalg.norm(u[1]) == 0.0  # zero row stays zero


def test_brute_force_gt_chunked_equals_direct():
    rng = np.random.default_rng(0)
    corpus = unit(rng.standard_normal((500, 16)).astype(np.float32))
    queries = corpus[:25]
    gt = brute_force_gt(corpus, queries, k=5, chunk=64)  # tiny chunk → multi-pass merge
    direct = np.argsort(-(queries @ corpus.T), axis=1)[:, :5]
    assert (gt[:, 0] == np.arange(25)).all()  # self is nearest
    assert (gt == direct).all()


def test_recall_at_k():
    gt = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(gt.copy(), gt, 3) == 1.0
    assert recall_at_k(np.array([[7, 8, 9], [1, 2, 3]]), gt, 3) == 0.0
    assert abs(recall_at_k(np.array([[1, 99], [98, 97]]), gt, 2) - 0.25) < 1e-9


def test_pad_truncates_and_fills():
    res = [[(11, 0.1), (12, 0.2), (13, 0.3)], [(21, 0.1)]]
    out = _pad(res, k=3)
    assert out.shape == (2, 3)
    assert out[0].tolist() == [11, 12, 13]
    assert out[1].tolist() == [21, -1, -1]  # short row padded with -1


def test_graph_bytes_scales_with_m():
    assert graph_bytes(16) == 2 * 16 * 4
    assert graph_bytes(32) == 2 * graph_bytes(16) // 2 * 2  # linear in M
    assert graph_bytes(8) < graph_bytes(16) < graph_bytes(32)
