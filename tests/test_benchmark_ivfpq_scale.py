"""Unit tests for the IVF-PQ at-scale benchmark's pure helpers.

The build/measure paths need the native extension and large memory, so they are
exercised by running the script. These tests pin the deterministic helpers the
reported numbers rely on: the footprint model (the headline memory claim),
exact ground truth, and recall.
"""

from __future__ import annotations

import numpy as np

from benchmarks.benchmark_ivfpq_scale import (
    brute_force_gt,
    footprint_model,
    recall_at_k,
)


def test_footprint_model_100m_d768_matches_headline():
    # 100M × 768: float32 ≈ 307 GB; IVF-PQ (M=96, 1 byte/sub-q) ≈ 9.6 GB codes.
    fp = footprint_model(100_000_000, 768, m_sub=96, n_lists=65536, k_cent=256, hnsw_m=16)
    assert abs(fp["fp32_flat_gb"] - 307.2) < 1.0
    # codes dominate: 100M × 96 B = 9.6 GB, plus small centroid/id overhead.
    assert 9.5 < fp["ivfpq_gb"] < 11.0
    assert fp["compression_x"] > 28  # ~32× before overhead


def test_footprint_model_compression_monotonic_in_m():
    # Fewer sub-quantisers → smaller codes → higher compression.
    low_m = footprint_model(1_000_000, 128, 8, 4096, 256, 16)
    high_m = footprint_model(1_000_000, 128, 32, 4096, 256, 16)
    assert low_m["ivfpq_gb"] < high_m["ivfpq_gb"]
    assert low_m["compression_x"] > high_m["compression_x"]


def test_brute_force_gt_chunked_equals_direct():
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((1000, 16)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = corpus[:20]
    # Small chunk forces the multi-chunk merge path.
    gt = brute_force_gt(corpus, queries, k=5, chunk=128)
    direct = np.argsort(-(queries @ corpus.T), axis=1)[:, :5]
    # Each query's own vector is its nearest neighbour.
    assert (gt[:, 0] == np.arange(20)).all()
    assert (gt == direct).all()


def test_recall_at_k():
    gt = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(gt.copy(), gt, 3) == 1.0
    assert recall_at_k(np.array([[7, 8, 9], [1, 2, 3]]), gt, 3) == 0.0
    # row0 half overlap, row1 none → mean(1/2, 0) = 0.25
    assert abs(recall_at_k(np.array([[1, 99], [98, 97]]), gt, 2) - 0.25) < 1e-9
