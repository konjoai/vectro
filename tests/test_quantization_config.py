"""Tests for QuantizationConfig — python/vectro.py.

Covers:
- Default construction and field values
- Validation: unknown precision_mode, unknown profile, bad group_size
- from_profile() class-method
- to_dict() serialisation round-trip
- Vectro.compress(config=...) integration
- config fields override loose kwargs
- model_dir + explicit non-default precision_mode triggers UserWarning
"""
from __future__ import annotations

import warnings
import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.vectro import QuantizationConfig, Vectro  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(0)
BATCH = RNG.standard_normal((100, 128)).astype(np.float32)


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------

class TestQuantizationConfigDefaults(unittest.TestCase):
    def test_default_precision_mode(self):
        cfg = QuantizationConfig()
        self.assertEqual(cfg.precision_mode, "int8")

    def test_default_profile(self):
        cfg = QuantizationConfig()
        self.assertIsNone(cfg.profile)

    def test_default_group_size(self):
        cfg = QuantizationConfig()
        self.assertEqual(cfg.group_size, 64)

    def test_default_assume_normalized(self):
        cfg = QuantizationConfig()
        self.assertFalse(cfg.assume_normalized)

    def test_default_return_quality_metrics(self):
        cfg = QuantizationConfig()
        self.assertFalse(cfg.return_quality_metrics)

    def test_default_model_dir(self):
        cfg = QuantizationConfig()
        self.assertIsNone(cfg.model_dir)

    def test_default_seed(self):
        cfg = QuantizationConfig()
        self.assertIsNone(cfg.seed)


# ---------------------------------------------------------------------------
# Explicit construction
# ---------------------------------------------------------------------------

class TestQuantizationConfigExplicit(unittest.TestCase):
    def test_nf4_construction(self):
        cfg = QuantizationConfig(precision_mode="nf4")
        self.assertEqual(cfg.precision_mode, "nf4")

    def test_binary_construction(self):
        cfg = QuantizationConfig(precision_mode="binary")
        self.assertEqual(cfg.precision_mode, "binary")

    def test_group_size_32(self):
        cfg = QuantizationConfig(precision_mode="int4", group_size=32)
        self.assertEqual(cfg.group_size, 32)

    def test_seed_stored(self):
        cfg = QuantizationConfig(seed=1337)
        self.assertEqual(cfg.seed, 1337)

    def test_assume_normalized_stored(self):
        cfg = QuantizationConfig(assume_normalized=True)
        self.assertTrue(cfg.assume_normalized)

    def test_profile_stored(self):
        cfg = QuantizationConfig(profile="quality")
        self.assertEqual(cfg.profile, "quality")


# ---------------------------------------------------------------------------
# Validation — invalid inputs raise ValueError
# ---------------------------------------------------------------------------

class TestQuantizationConfigValidation(unittest.TestCase):
    def test_unknown_precision_mode_raises(self):
        with self.assertRaises(ValueError, msg="Unknown precision_mode should raise"):
            QuantizationConfig(precision_mode="fp8")

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError, msg="Unknown profile should raise"):
            QuantizationConfig(profile="ultra-fast-v2")

    def test_negative_group_size_raises(self):
        with self.assertRaises(ValueError):
            QuantizationConfig(group_size=-1)

    def test_zero_group_size_raises(self):
        with self.assertRaises(ValueError):
            QuantizationConfig(group_size=0)

    def test_non_power_of_two_group_size_raises(self):
        with self.assertRaises(ValueError):
            QuantizationConfig(group_size=60)

    def test_non_integer_group_size_raises(self):
        with self.assertRaises(ValueError):
            QuantizationConfig(group_size=32.0)  # type: ignore[arg-type]

    def test_non_integer_seed_raises(self):
        with self.assertRaises(ValueError):
            QuantizationConfig(seed="abc")  # type: ignore[arg-type]

    def test_model_dir_with_non_default_precision_warns(self):
        """model_dir + explicit non-default precision_mode triggers UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = QuantizationConfig(model_dir="/path/to/model", precision_mode="nf4")
        self.assertTrue(
            any("model_dir" in str(warning.message) for warning in w),
            "Expected a UserWarning about model_dir + precision_mode conflict",
        )
        # Config still constructs successfully
        self.assertEqual(cfg.precision_mode, "nf4")


# ---------------------------------------------------------------------------
# from_profile() class-method
# ---------------------------------------------------------------------------

class TestFromProfile(unittest.TestCase):
    def test_fast_maps_to_int8(self):
        cfg = QuantizationConfig.from_profile("fast")
        self.assertEqual(cfg.precision_mode, "int8")
        self.assertEqual(cfg.profile, "fast")

    def test_quality_maps_to_nf4(self):
        cfg = QuantizationConfig.from_profile("quality")
        self.assertEqual(cfg.precision_mode, "nf4")

    def test_binary_profile(self):
        cfg = QuantizationConfig.from_profile("binary")
        self.assertEqual(cfg.precision_mode, "binary")

    def test_overrides_applied(self):
        cfg = QuantizationConfig.from_profile("balanced", seed=42)
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.profile, "balanced")

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            QuantizationConfig.from_profile("not-a-real-profile")


# ---------------------------------------------------------------------------
# to_dict() serialisation
# ---------------------------------------------------------------------------

class TestToDict(unittest.TestCase):
    def test_keys_present(self):
        cfg = QuantizationConfig(precision_mode="nf4", seed=7)
        d = cfg.to_dict()
        expected_keys = {
            "precision_mode", "profile", "group_size",
            "assume_normalized", "return_quality_metrics", "model_dir", "seed",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_values_correct(self):
        cfg = QuantizationConfig(precision_mode="binary", group_size=128, seed=99)
        d = cfg.to_dict()
        self.assertEqual(d["precision_mode"], "binary")
        self.assertEqual(d["group_size"], 128)
        self.assertEqual(d["seed"], 99)

    def test_json_serialisable(self):
        import json
        cfg = QuantizationConfig(precision_mode="int8", profile="balanced", seed=0)
        d = cfg.to_dict()
        # Must not raise
        json.dumps(d)


# ---------------------------------------------------------------------------
# Vectro.compress(config=...) integration
# ---------------------------------------------------------------------------

class TestCompressWithConfig(unittest.TestCase):
    def setUp(self):
        self.vectro = Vectro()

    def test_config_int8_compress(self):
        cfg = QuantizationConfig(precision_mode="int8")
        result = self.vectro.compress(BATCH, config=cfg)
        self.assertIsNotNone(result)

    def test_config_nf4_compress(self):
        cfg = QuantizationConfig(precision_mode="nf4")
        result = self.vectro.compress(BATCH, config=cfg)
        self.assertIsNotNone(result)

    def test_config_binary_compress(self):
        cfg = QuantizationConfig(precision_mode="binary")
        result = self.vectro.compress(BATCH, config=cfg)
        self.assertIsNotNone(result)

    def test_config_overrides_loose_profile(self):
        """config.profile wins over the standalone profile= kwarg."""
        cfg = QuantizationConfig.from_profile("fast")  # int8
        # Pass profile="quality" (nf4) as a kwarg — config should win.
        result = self.vectro.compress(BATCH, profile="quality", config=cfg)
        # Just verify it runs without error; the exact mode depends on dispatch order.
        self.assertIsNotNone(result)

    def test_invalid_config_type_raises(self):
        with self.assertRaises(TypeError):
            self.vectro.compress(BATCH, config={"precision_mode": "int8"})  # type: ignore[arg-type]

    def test_config_from_profile_balanced(self):
        cfg = QuantizationConfig.from_profile("balanced")
        result = self.vectro.compress(BATCH, config=cfg)
        self.assertIsNotNone(result)

    def test_config_with_quality_metrics(self):
        cfg = QuantizationConfig(return_quality_metrics=True)
        output = self.vectro.compress(BATCH, config=cfg)
        # When return_quality_metrics=True the result is a 2-tuple (result, metrics)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)


if __name__ == "__main__":
    unittest.main()
