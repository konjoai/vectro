"""Tests for python/quality_api.py — QualityMetrics, VectroQualityAnalyzer, QualityBenchmark, QualityReport."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.quality_api import (  # noqa: E402
    QualityBenchmark,
    QualityMetrics,
    QualityReport,
    VectroQualityAnalyzer,
    evaluate_quantization_quality,
    generate_quality_report,
)

_RNG = np.random.default_rng(0)


def _make_metrics(cosine_sim: float) -> QualityMetrics:
    """Construct a minimal QualityMetrics with a given mean cosine similarity."""
    return QualityMetrics(
        mean_absolute_error=0.01,
        mean_squared_error=0.0001,
        root_mean_squared_error=0.01,
        mean_cosine_similarity=cosine_sim,
        min_cosine_similarity=cosine_sim - 0.001,
        max_cosine_similarity=min(cosine_sim + 0.001, 1.0),
        percentile_errors={
            "error_p25": 0.005,
            "error_p50": 0.01,
            "error_p75": 0.02,
            "error_p95": 0.05,
            "error_p99": 0.08,
            "error_p99_9": 0.10,
        },
        signal_to_noise_ratio=30.0,
        peak_signal_to_noise_ratio=45.0,
        structural_similarity=0.99,
        compression_ratio=3.5,
    )


def _perfect_pair(n: int = 16, d: int = 8):
    vecs = _RNG.standard_normal((n, d)).astype(np.float32)
    return vecs, vecs.copy()


def _noisy_pair(n: int = 16, d: int = 8, noise_scale: float = 0.01):
    vecs = _RNG.standard_normal((n, d)).astype(np.float32)
    noisy = vecs + _RNG.standard_normal((n, d)).astype(np.float32) * noise_scale
    return vecs, noisy


# ---------------------------------------------------------------------------
# QualityMetrics — quality_grade and passes_quality_threshold
# ---------------------------------------------------------------------------


class TestQualityMetricsGrades(unittest.TestCase):
    def test_quality_grade_a_plus(self):
        m = _make_metrics(1.0)
        self.assertIn("A+", m.quality_grade())

    def test_quality_grade_a(self):
        m = _make_metrics(0.997)
        self.assertIn("A", m.quality_grade())
        self.assertNotIn("A+", m.quality_grade())

    def test_quality_grade_b_plus(self):
        m = _make_metrics(0.992)
        self.assertIn("B+", m.quality_grade())

    def test_quality_grade_b(self):
        m = _make_metrics(0.987)
        self.assertIn("B", m.quality_grade())
        self.assertNotIn("B+", m.quality_grade())

    def test_quality_grade_c_plus(self):
        m = _make_metrics(0.982)
        self.assertIn("C+", m.quality_grade())

    def test_quality_grade_c(self):
        m = _make_metrics(0.975)
        self.assertIn("C", m.quality_grade())
        self.assertNotIn("C+", m.quality_grade())

    def test_quality_grade_d(self):
        m = _make_metrics(0.96)
        self.assertIn("D", m.quality_grade())

    def test_passes_quality_threshold_above(self):
        m = _make_metrics(0.998)
        self.assertTrue(m.passes_quality_threshold(0.995))

    def test_passes_quality_threshold_below(self):
        m = _make_metrics(0.990)
        self.assertFalse(m.passes_quality_threshold(0.995))

    def test_quality_metrics_to_dict_contains_grade(self):
        m = _make_metrics(0.999)
        d = m.to_dict()
        self.assertIn("quality_grade", d)
        self.assertIn("mean_cosine_similarity", d)

    def test_to_dict_contains_percentile_errors(self):
        m = _make_metrics(0.999)
        d = m.to_dict()
        self.assertIn("error_p95", d)
        self.assertIn("error_p99", d)


# ---------------------------------------------------------------------------
# VectroQualityAnalyzer — evaluate_quality
# ---------------------------------------------------------------------------


class TestVectroQualityAnalyzer(unittest.TestCase):
    def test_evaluate_quality_returns_quality_metrics(self):
        orig, recon = _perfect_pair()
        result = VectroQualityAnalyzer.evaluate_quality(orig, recon)
        self.assertIsInstance(result, QualityMetrics)

    def test_evaluate_quality_perfect_reconstruction_mae_near_zero(self):
        orig, recon = _perfect_pair()
        result = VectroQualityAnalyzer.evaluate_quality(orig, recon)
        self.assertAlmostEqual(result.mean_absolute_error, 0.0, places=6)

    def test_evaluate_quality_cosine_sim_near_one_for_perfect(self):
        orig, recon = _perfect_pair()
        result = VectroQualityAnalyzer.evaluate_quality(orig, recon)
        self.assertAlmostEqual(result.mean_cosine_similarity, 1.0, places=5)

    def test_evaluate_quality_shape_mismatch_raises(self):
        orig = _RNG.standard_normal((4, 8)).astype(np.float32)
        recon = _RNG.standard_normal((5, 8)).astype(np.float32)
        with self.assertRaises(ValueError):
            VectroQualityAnalyzer.evaluate_quality(orig, recon)

    def test_evaluate_quality_handles_zero_vectors(self):
        orig = np.zeros((4, 8), dtype=np.float32)
        recon = np.zeros((4, 8), dtype=np.float32)
        result = VectroQualityAnalyzer.evaluate_quality(orig, recon)
        # Both all-zero: cosine similarity should be 1.0 (handled by implementation)
        self.assertAlmostEqual(result.mean_cosine_similarity, 1.0, places=5)

    def test_evaluate_quality_noisy_cosine_sim_less_than_one(self):
        orig, noisy = _noisy_pair(noise_scale=0.5)
        result = VectroQualityAnalyzer.evaluate_quality(orig, noisy)
        self.assertLess(result.mean_cosine_similarity, 1.0)

    def test_evaluate_quality_compression_ratio_estimated_when_none(self):
        orig, recon = _perfect_pair()
        result = VectroQualityAnalyzer.evaluate_quality(orig, recon, compression_ratio=None)
        self.assertGreater(result.compression_ratio, 0.0)

    def test_evaluate_quality_uses_provided_compression_ratio(self):
        orig, recon = _perfect_pair()
        result = VectroQualityAnalyzer.evaluate_quality(orig, recon, compression_ratio=5.5)
        self.assertAlmostEqual(result.compression_ratio, 5.5, places=6)


# ---------------------------------------------------------------------------
# QualityBenchmark
# ---------------------------------------------------------------------------


class TestQualityBenchmark(unittest.TestCase):
    def test_benchmark_dimensions_returns_dict_keyed_by_dim(self):
        dims = [8, 16]
        results = QualityBenchmark.benchmark_dimensions(dimensions=dims, num_vectors=20)
        self.assertEqual(set(results.keys()), {8, 16})
        for v in results.values():
            self.assertIsInstance(v, QualityMetrics)

    def test_benchmark_vector_counts_returns_dict_keyed_by_count(self):
        counts = [10, 50]
        results = QualityBenchmark.benchmark_vector_counts(vector_counts=counts, dimension=8)
        self.assertEqual(set(results.keys()), {10, 50})
        for v in results.values():
            self.assertIsInstance(v, QualityMetrics)


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------


class TestQualityReport(unittest.TestCase):
    def test_generate_quality_report_string_nonempty(self):
        m = _make_metrics(0.998)
        report = QualityReport.generate_report(m)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)

    def test_compare_configurations_table_sorted(self):
        configs = {
            "balanced": _make_metrics(0.995),
            "fast": _make_metrics(0.990),
            "quality": _make_metrics(0.999),
        }
        report = QualityReport.compare_configurations(configs)
        self.assertIsInstance(report, str)
        # "quality" has highest sim so should appear before "fast"
        quality_pos = report.find("quality")
        fast_pos = report.find("fast")
        self.assertLess(quality_pos, fast_pos)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


class TestModuleLevelQualityFunctions(unittest.TestCase):
    def test_module_level_evaluate_quantization_quality(self):
        orig, recon = _noisy_pair(noise_scale=0.01)
        result = evaluate_quantization_quality(orig, recon)
        self.assertIsInstance(result, QualityMetrics)

    def test_module_level_generate_quality_report(self):
        orig, recon = _noisy_pair(noise_scale=0.01)
        report = generate_quality_report(orig, recon)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)


if __name__ == "__main__":
    unittest.main()
