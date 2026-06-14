"""Tests for python/profiles_api.py — CompressionProfile, ProfileManager, CompressionOptimizer, ProfileComparison."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.profiles_api import (  # noqa: E402
    CompressionOptimizer,
    CompressionProfile,
    CompressionStrategy,
    ProfileComparison,
    ProfileManager,
    create_custom_profile,
    get_compression_profile,
)

_RNG = np.random.default_rng(7)
_VECS = _RNG.standard_normal((20, 16)).astype(np.float32)

# Ensure builtins are initialized before any test touches the class.
ProfileManager.initialize_builtin_profiles()

_BUILTIN_NAMES = {"fast", "balanced", "quality", "ultra", "binary"}

# ---------------------------------------------------------------------------
# ProfileManager — builtins
# ---------------------------------------------------------------------------


class TestProfileManagerBuiltins(unittest.TestCase):
    def test_list_profiles_contains_five_builtins(self):
        profiles = ProfileManager.list_profiles()
        for name in _BUILTIN_NAMES:
            self.assertIn(name, profiles)

    def test_get_fast_profile(self):
        p = ProfileManager.get_profile("fast")
        self.assertEqual(p.name, "fast")
        self.assertEqual(p.strategy, CompressionStrategy.FAST)
        self.assertEqual(p.quantization_bits, 8)

    def test_get_balanced_profile(self):
        p = ProfileManager.get_profile("balanced")
        self.assertEqual(p.name, "balanced")
        self.assertEqual(p.strategy, CompressionStrategy.BALANCED)

    def test_get_quality_profile(self):
        p = ProfileManager.get_profile("quality")
        self.assertEqual(p.name, "quality")
        self.assertEqual(p.strategy, CompressionStrategy.QUALITY)
        self.assertGreaterEqual(p.min_similarity_threshold, 0.995)

    def test_get_unknown_profile_raises_value_error(self):
        with self.assertRaises(ValueError):
            ProfileManager.get_profile("___nonexistent___")


# ---------------------------------------------------------------------------
# ProfileManager — custom profiles
# ---------------------------------------------------------------------------


def _make_custom(name: str = "__test_profile__") -> CompressionProfile:
    return CompressionProfile(
        name=name,
        strategy=CompressionStrategy.CUSTOM,
        quantization_bits=8,
        range_factor=0.95,
        clipping_percentile=99.0,
        adaptive_scaling=True,
        batch_optimization=True,
        precision_mode="int8",
        error_correction=True,
        simd_enabled=True,
        parallel_processing=True,
        memory_efficient=True,
        preserve_norms=True,
        preserve_angles=False,
        min_similarity_threshold=0.995,
    )


class TestProfileManagerCustom(unittest.TestCase):
    _TEST_NAME = "__unit_test_custom__"

    def tearDown(self):
        # Always clean up so class state is not polluted between tests.
        if self._TEST_NAME in ProfileManager._custom_profiles:
            del ProfileManager._custom_profiles[self._TEST_NAME]

    def test_add_and_remove_custom_profile(self):
        p = _make_custom(self._TEST_NAME)
        ProfileManager.add_custom_profile(p)
        fetched = ProfileManager.get_profile(self._TEST_NAME)
        self.assertEqual(fetched.name, self._TEST_NAME)
        ProfileManager.remove_custom_profile(self._TEST_NAME)
        with self.assertRaises(ValueError):
            ProfileManager.get_profile(self._TEST_NAME)

    def test_custom_profile_appears_in_list(self):
        p = _make_custom(self._TEST_NAME)
        ProfileManager.add_custom_profile(p)
        self.assertIn(self._TEST_NAME, ProfileManager.list_profiles())

    def test_remove_nonexistent_custom_profile_raises(self):
        with self.assertRaises(ValueError):
            ProfileManager.remove_custom_profile("___does_not_exist___")

    def test_save_and_load_custom_profiles(self):
        p = _make_custom(self._TEST_NAME)
        ProfileManager.add_custom_profile(p)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            ProfileManager.save_profiles(tmp_path)
            # Clear custom profiles and reload.
            del ProfileManager._custom_profiles[self._TEST_NAME]
            ProfileManager.load_profiles(tmp_path)
            loaded = ProfileManager.get_profile(self._TEST_NAME)
            self.assertEqual(loaded.name, self._TEST_NAME)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CompressionProfile — validation
# ---------------------------------------------------------------------------


class TestCompressionProfileValidation(unittest.TestCase):
    def _make(self, **overrides) -> CompressionProfile:
        defaults = dict(
            name="test",
            strategy=CompressionStrategy.CUSTOM,
            quantization_bits=8,
            range_factor=0.95,
            clipping_percentile=99.0,
            adaptive_scaling=True,
            batch_optimization=True,
            precision_mode="int8",
            error_correction=True,
            simd_enabled=True,
            parallel_processing=True,
            memory_efficient=True,
            preserve_norms=True,
            preserve_angles=False,
            min_similarity_threshold=0.995,
        )
        defaults.update(overrides)
        return CompressionProfile(**defaults)

    def test_compression_profile_validation_quantization_bits_out_of_range(self):
        with self.assertRaises(ValueError):
            self._make(quantization_bits=0)
        with self.assertRaises(ValueError):
            self._make(quantization_bits=9)

    def test_compression_profile_validation_range_factor_out_of_range(self):
        with self.assertRaises(ValueError):
            self._make(range_factor=-0.1)
        with self.assertRaises(ValueError):
            self._make(range_factor=1.1)

    def test_compression_profile_validation_clipping_percentile_out_of_range(self):
        with self.assertRaises(ValueError):
            self._make(clipping_percentile=-1.0)
        with self.assertRaises(ValueError):
            self._make(clipping_percentile=101.0)

    def test_compression_profile_to_and_from_dict_round_trip(self):
        p = _make_custom("rt_profile")
        d = p.to_dict()
        p2 = CompressionProfile.from_dict(d)
        self.assertEqual(p2.name, p.name)
        self.assertEqual(p2.strategy, p.strategy)
        self.assertAlmostEqual(p2.range_factor, p.range_factor)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleLevelProfileFunctions(unittest.TestCase):
    _TEST_NAME = "__unit_test_module_custom__"

    def tearDown(self):
        if self._TEST_NAME in ProfileManager._custom_profiles:
            del ProfileManager._custom_profiles[self._TEST_NAME]

    def test_get_compression_profile_module_func(self):
        p = get_compression_profile("balanced")
        self.assertEqual(p.name, "balanced")

    def test_create_custom_profile_module_func(self):
        p = create_custom_profile(self._TEST_NAME, quantization_bits=8, range_factor=0.9)
        self.assertIsInstance(p, CompressionProfile)
        self.assertEqual(p.name, self._TEST_NAME)


# ---------------------------------------------------------------------------
# CompressionOptimizer
# ---------------------------------------------------------------------------


class TestCompressionOptimizer(unittest.TestCase):
    def test_auto_optimize_returns_compression_profile(self):
        vecs = _RNG.standard_normal((30, 16)).astype(np.float32)
        result = CompressionOptimizer.auto_optimize_profile(
            vecs, target_similarity=0.995, target_compression=3.0
        )
        self.assertIsInstance(result, CompressionProfile)


# ---------------------------------------------------------------------------
# ProfileComparison
# ---------------------------------------------------------------------------


class TestProfileComparison(unittest.TestCase):
    def test_profile_comparison_compare_profiles_keys(self):
        results = ProfileComparison.compare_profiles(_VECS, profile_names=["fast", "balanced"])
        self.assertIn("fast", results)
        self.assertIn("balanced", results)

    def test_profile_comparison_generate_report_nonempty(self):
        results = ProfileComparison.compare_profiles(_VECS, profile_names=["fast", "balanced"])
        report = ProfileComparison.generate_comparison_report(results)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)


if __name__ == "__main__":
    unittest.main()
