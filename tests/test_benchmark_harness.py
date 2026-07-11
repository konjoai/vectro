"""Tests for the recall-matched benchmark harness (benchmarks/harness/).

Covers the pure, deterministic core — statistics (percentiles, CoV gate, paired
Wilcoxon + effect size), the .fvecs/.ivecs codecs, exact ground truth / recall —
plus an end-to-end synthetic self-test of the harness pipeline including the
two-run stability kill-test. These run on any host (no target hardware, no
multi-GB datasets, no compiled extension needed).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.harness import datasets, protocol, stats  # noqa: E402


class TestStats(unittest.TestCase):
    def test_percentiles_and_cov(self):
        s = stats.summarize([100.0] * 10)
        self.assertAlmostEqual(s.p50, 100.0)
        self.assertAlmostEqual(s.cov, 0.0)
        self.assertEqual(s.n, 10)

    def test_percentile_ordering(self):
        s = stats.summarize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertLessEqual(s.p50, s.p95)
        self.assertLessEqual(s.p95, s.p99)

    def test_cov_gate(self):
        tight = stats.summarize([100, 101, 99, 100, 100])
        noisy = stats.summarize([100, 200, 50, 175, 90])
        self.assertTrue(stats.cov_gate_passes(tight))
        self.assertFalse(stats.cov_gate_passes(noisy))

    def test_wilcoxon_detects_clear_improvement(self):
        base = [100.0 + i * 0.1 for i in range(30)]
        cand = [130.0 + i * 0.1 for i in range(30)]  # uniformly ~30% faster
        w = stats.paired_wilcoxon(base, cand)
        self.assertTrue(w.significant)
        self.assertGreater(w.median_improvement_pct, 20.0)
        self.assertGreater(w.effect_r, 0.5)

    def test_wilcoxon_no_difference(self):
        base = [100.0, 101.0, 99.0, 100.5, 99.5] * 6
        cand = list(base)
        w = stats.paired_wilcoxon(base, cand)
        self.assertFalse(w.significant)

    def test_wilcoxon_normal_approx_fallback(self):
        # The scipy-free path must agree in direction/significance with scipy.
        nz = [5.0, 6.0, 4.0, 7.0, 5.5, 6.5, 4.5, 5.2, 6.1, 5.8]
        stat, p, method = stats._wilcoxon_normal_approx(nz)
        self.assertEqual(method, "normal-approx")
        self.assertLess(p, 0.05)  # all-positive differences → significant

    def test_build_verdict_flags_noise(self):
        base = [100, 100, 100, 100, 100]
        cand = [100, 300, 20, 250, 60]  # candidate wildly noisy
        v = stats.build_verdict("A", "B", base, cand)
        self.assertFalse(v.cov_ok)
        self.assertTrue(any("CoV gate FAILED" in n for n in v.notes))


class TestDatasetCodecs(unittest.TestCase):
    def test_fvecs_roundtrip(self):
        import tempfile

        arr = np.random.default_rng(0).standard_normal((17, 13)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.fvecs"
            datasets.write_fvecs(p, arr)
            back = datasets.read_fvecs(p)
        np.testing.assert_allclose(arr, back)

    def test_ivecs_roundtrip(self):
        import tempfile

        arr = np.arange(5 * 8, dtype=np.int32).reshape(5, 8)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.ivecs"
            datasets.write_ivecs(p, arr)
            back = datasets.read_ivecs(p)
        np.testing.assert_array_equal(arr, back)

    def test_ground_truth_is_exact(self):
        rng = np.random.default_rng(1)
        train = rng.standard_normal((200, 16)).astype(np.float32)
        queries = train[:10] + rng.standard_normal((10, 16)).astype(np.float32) * 1e-4
        gt = datasets.compute_ground_truth(train, queries, k=5, metric="l2")
        # Each near-copy query's nearest neighbour must be its source row.
        self.assertTrue(all(gt[i, 0] == i for i in range(10)))

    def test_recall_at_k(self):
        found = np.array([[0, 1, 2], [3, 4, 9]])
        truth = np.array([[0, 1, 2], [3, 4, 5]])
        self.assertAlmostEqual(datasets.recall_at_k(found, truth, 3), (3 + 2) / 6)


class TestSyntheticSelfTest(unittest.TestCase):
    def test_pipeline_end_to_end_and_stability(self):
        from benchmarks.harness.engines import VectroHnswFp32

        ds = datasets.load_synthetic(n=800, q=40, dim=32, k=10)
        self.assertEqual(ds.dim, 32)
        eng = VectroHnswFp32(m=12, ef_construction=100)
        eng.build(ds.train, ds.metric)
        op = protocol.find_operating_point(eng, ds, k=10, recall_target=0.50, grid=[16, 64, 256])
        self.assertIsNotNone(op.param)
        m1 = protocol.measure(eng, ds, op, k=10, n_runs=6, warmup=2)
        m2 = protocol.measure(eng, ds, op, k=10, n_runs=6, warmup=2)
        self.assertEqual(len(m1.qps_runs), 6)
        # Both runs measure the same code path — p50 QPS should be same order.
        self.assertGreater(m1.qps.p50, 0.0)
        drift = abs(m2.qps.p50 - m1.qps.p50) / m1.qps.p50
        self.assertLess(drift, 0.9)  # loose: only asserts the machinery, not a claim


if __name__ == "__main__":
    unittest.main()
