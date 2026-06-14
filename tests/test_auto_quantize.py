"""Tests for Phase 7c — auto_quantize_api."""

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.auto_quantize_api import (  # noqa: E402
    auto_quantize,
    _compute_kurtosis,
    _cosine_sim_mean,
    _try_int8_fallback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n=100, d=64, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtilities(unittest.TestCase):
    def test_cosine_sim_mean_identical(self):
        """Mean cosine of vector with itself = 1."""
        a = _make_data(20, 32)
        sim = _cosine_sim_mean(a, a)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_cosine_sim_mean_range(self):
        a = _make_data(20, 32)
        b = _make_data(20, 32, seed=99)
        sim = _cosine_sim_mean(a, b)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)

    def test_compute_kurtosis_gaussian(self):
        """Gaussian data should have excess kurtosis ≈ 0."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5000, 10)).astype(np.float32)
        kurt = _compute_kurtosis(data)
        # Allow liberal tolerance for small samples
        self.assertLess(abs(kurt), 2.0)

    def test_compute_kurtosis_heavy_tail(self):
        """Laplace distribution has excess kurtosis = 3 > 0."""
        rng = np.random.default_rng(0)
        data = rng.laplace(size=(2000, 10)).astype(np.float32)
        kurt = _compute_kurtosis(data)
        self.assertGreater(kurt, 0.0)

    def test_int8_fallback_shape(self):
        data = _make_data(50, 32)
        r = _try_int8_fallback(data)
        self.assertTrue(r["success"])
        self.assertEqual(r["result"]["quantized"].shape, (50, 32))

    def test_int8_fallback_cosine_quality(self):
        data = _make_data(50, 32)
        r = _try_int8_fallback(data)
        self.assertGreater(r["cosine_sim"], 0.99)


# ---------------------------------------------------------------------------
# auto_quantize — output structure
# ---------------------------------------------------------------------------


class TestAutoQuantizeStructure(unittest.TestCase):
    def setUp(self):
        self.data = _make_data(80, 48)

    def test_returns_dict(self):
        out = auto_quantize(self.data)
        self.assertIsInstance(out, dict)

    def test_required_keys(self):
        out = auto_quantize(self.data)
        for k in ("mode", "cosine_sim", "compression_ratio", "result", "kurtosis", "tried"):
            self.assertIn(k, out, f"Missing key: {k}")

    def test_mode_is_string(self):
        out = auto_quantize(self.data)
        self.assertIsInstance(out["mode"], str)
        self.assertGreater(len(out["mode"]), 0)

    def test_cosine_sim_in_range(self):
        out = auto_quantize(self.data)
        self.assertGreaterEqual(out["cosine_sim"], -1.0)
        self.assertLessEqual(out["cosine_sim"], 1.0 + 1e-5)

    def test_compression_ratio_positive(self):
        out = auto_quantize(self.data)
        self.assertGreater(out["compression_ratio"], 0)

    def test_tried_is_list(self):
        out = auto_quantize(self.data)
        self.assertIsInstance(out["tried"], list)
        self.assertGreater(len(out["tried"]), 0)

    def test_result_not_none(self):
        out = auto_quantize(self.data)
        self.assertIsNotNone(out["result"])

    def test_kurtosis_is_float(self):
        out = auto_quantize(self.data)
        self.assertIsInstance(out["kurtosis"], float)


# ---------------------------------------------------------------------------
# auto_quantize — quality
# ---------------------------------------------------------------------------


class TestAutoQuantizeQuality(unittest.TestCase):
    def test_cosine_achieves_fallback(self):
        """Even with tight targets the function returns something with cosine > 0.5."""
        data = _make_data(60, 32)
        out = auto_quantize(data, target_cosine=0.999, target_compression=100.0)
        self.assertGreater(out["cosine_sim"], 0.50)

    def test_loose_targets_met(self):
        """Loose quality & compression targets should always be met."""
        data = _make_data(100, 64)
        out = auto_quantize(data, target_cosine=0.50, target_compression=2.0)
        self.assertGreaterEqual(out["cosine_sim"], 0.50)
        self.assertGreaterEqual(out["compression_ratio"], 2.0)

    def test_no_nan_in_cosine(self):
        data = _make_data(50, 32)
        out = auto_quantize(data)
        self.assertFalse(np.isnan(out["cosine_sim"]))

    def test_no_nan_in_compression(self):
        data = _make_data(50, 32)
        out = auto_quantize(data)
        self.assertFalse(np.isnan(out["compression_ratio"]))


# ---------------------------------------------------------------------------
# auto_quantize — kurtosis routing
# ---------------------------------------------------------------------------


class TestAutoQuantizeKurtosisRouting(unittest.TestCase):
    def test_heavy_tail_routing(self):
        """Heavy-tailed data (Laplace) should still produce valid output."""
        rng = np.random.default_rng(0)
        data = rng.laplace(size=(80, 32)).astype(np.float32)
        out = auto_quantize(data)
        self.assertIsInstance(out["mode"], str)
        self.assertGreater(out["cosine_sim"], 0.0)

    def test_gaussian_routing(self):
        """Gaussian data should produce valid output."""
        rng = np.random.default_rng(1)
        data = rng.standard_normal((80, 32)).astype(np.float32)
        out = auto_quantize(data)
        self.assertIsInstance(out["mode"], str)

    def test_kurtosis_affects_tried_order(self):
        """Heavy-tailed data should list nf4_mixed before nf4 in tried list
        (or equivalently, the first tried entry mode should contain 'mixed')."""
        rng = np.random.default_rng(42)
        heavy = rng.laplace(size=(100, 32)).astype(np.float32)
        out_heavy = auto_quantize(heavy)
        # First in `tried` should be nf4_mixed for heavy-tailed
        first_mode = out_heavy["tried"][0]["mode"]
        self.assertIn("mixed", first_mode)

    def test_gaussian_first_tried_not_mixed(self):
        """Gaussian data: first tried entry should be plain nf4 (not mixed)."""
        rng = np.random.default_rng(0)
        # Use very large sample to ensure kurtosis ≈ 0
        gauss = rng.standard_normal((2000, 16)).astype(np.float32)
        out = auto_quantize(gauss)
        first_mode = out["tried"][0]["mode"]
        self.assertNotIn("mixed", first_mode)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAutoQuantizeEdgeCases(unittest.TestCase):
    def test_single_vector(self):
        data = _make_data(1, 32)
        out = auto_quantize(data)
        self.assertIsInstance(out, dict)
        self.assertGreater(out["cosine_sim"], 0.0)

    def test_high_dimensional(self):
        data = _make_data(50, 512)
        out = auto_quantize(data, target_cosine=0.90, target_compression=4.0)
        self.assertIsInstance(out, dict)

    def test_default_targets_reasonable(self):
        """Default targets (cosine=0.97, compression=8x) with d=64."""
        data = _make_data(100, 64)
        out = auto_quantize(data)
        # At minimum the result should be non-trivially compressed
        self.assertGreater(out["compression_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
