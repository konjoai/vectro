"""Tests for LoRA adapter compression — lora_api.py"""

import unittest
import warnings

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.lora_api import (  # noqa: E402
    compress_lora,
    decompress_lora,
    compress_lora_adapter,
    LoRAResult,
    _RQ_MIN_ROWS,
)
from python.interface import mean_cosine_similarity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

# Typical LoRA rank-16 adapter for a 768-d model
RANK = 16
IN_FEATURES = 768
OUT_FEATURES = 768

A_SMALL = RNG.standard_normal((RANK, IN_FEATURES)).astype(np.float32)
B_SMALL = RNG.standard_normal((OUT_FEATURES, RANK)).astype(np.float32)

# Large matrices sufficient for RQ training (rank=64)
RANK_LARGE = 64
A_LARGE = RNG.standard_normal((RANK_LARGE, IN_FEATURES)).astype(np.float32)
B_LARGE = RNG.standard_normal((OUT_FEATURES, RANK_LARGE)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoRANF4OutputShapes(unittest.TestCase):
    def test_result_fields(self):
        result = compress_lora(A_SMALL, B_SMALL, profile="lora-nf4", target_module="q_proj")
        self.assertIsInstance(result, LoRAResult)
        self.assertEqual(result.profile, "lora-nf4")
        self.assertEqual(result.rank, RANK)
        self.assertEqual(result.target_module, "q_proj")
        self.assertEqual(result.A_shape, A_SMALL.shape)
        self.assertEqual(result.B_shape, B_SMALL.shape)

    def test_packed_shapes(self):
        result = compress_lora(A_SMALL, B_SMALL, profile="lora-nf4")
        # NF4: packed is (n, ceil(d/2))
        self.assertEqual(result.A_data["packed"].shape, (RANK, IN_FEATURES // 2))
        self.assertEqual(result.B_data["packed"].shape, (OUT_FEATURES, RANK // 2))

    def test_scales_shapes(self):
        result = compress_lora(A_SMALL, B_SMALL, profile="lora-nf4")
        self.assertEqual(result.A_data["scales"].shape, (RANK,))
        self.assertEqual(result.B_data["scales"].shape, (OUT_FEATURES,))


class TestLoRANF4RoundtripQuality(unittest.TestCase):
    def setUp(self):
        self.result = compress_lora(A_SMALL, B_SMALL, profile="lora-nf4")
        self.A_r, self.B_r = decompress_lora(self.result)

    def test_reconstruction_dtypes(self):
        self.assertEqual(self.A_r.dtype, np.float32)
        self.assertEqual(self.B_r.dtype, np.float32)

    def test_reconstruction_shapes(self):
        self.assertEqual(self.A_r.shape, A_SMALL.shape)
        self.assertEqual(self.B_r.shape, B_SMALL.shape)

    def test_cosine_sim_A(self):
        self.assertGreaterEqual(
            self.result.cosine_sim_A, 0.97, f"cosine_sim_A={self.result.cosine_sim_A:.4f} < 0.97"
        )

    def test_cosine_sim_B(self):
        self.assertGreaterEqual(
            self.result.cosine_sim_B, 0.97, f"cosine_sim_B={self.result.cosine_sim_B:.4f} < 0.97"
        )

    def test_no_nan_inf(self):
        self.assertFalse(np.isnan(self.A_r).any(), "NaN in A reconstruction")
        self.assertFalse(np.isnan(self.B_r).any(), "NaN in B reconstruction")
        self.assertFalse(np.isinf(self.A_r).any(), "Inf in A reconstruction")
        self.assertFalse(np.isinf(self.B_r).any(), "Inf in B reconstruction")


class TestLoRAInt8RoundtripQuality(unittest.TestCase):
    def setUp(self):
        self.result = compress_lora(A_SMALL, B_SMALL, profile="lora-int8")
        self.A_r, self.B_r = decompress_lora(self.result)

    def test_profile_stored(self):
        self.assertEqual(self.result.profile, "lora-int8")

    def test_cosine_sim_A(self):
        self.assertGreaterEqual(
            self.result.cosine_sim_A, 0.99, f"cosine_sim_A={self.result.cosine_sim_A:.4f} < 0.99"
        )

    def test_cosine_sim_B(self):
        self.assertGreaterEqual(
            self.result.cosine_sim_B, 0.99, f"cosine_sim_B={self.result.cosine_sim_B:.4f} < 0.99"
        )

    def test_dtype_contract(self):
        self.assertEqual(self.A_r.dtype, np.float32)
        self.assertEqual(self.B_r.dtype, np.float32)


class TestLoRARQFallbackSmallRank(unittest.TestCase):
    def test_fallback_to_nf4_when_rank_too_small(self):
        """Rank < _RQ_MIN_ROWS must fall back to lora-nf4 with a warning."""
        self.assertLess(RANK, _RQ_MIN_ROWS)  # sanity: fixture rank is small
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compress_lora(A_SMALL, B_SMALL, profile="lora-rq")
        self.assertEqual(result.profile, "lora-nf4")
        self.assertTrue(any("lora-nf4" in str(warning.message) for warning in w))

    def test_fallback_result_still_decompresses(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = compress_lora(A_SMALL, B_SMALL, profile="lora-rq")
        A_r, B_r = decompress_lora(result)
        self.assertEqual(A_r.shape, A_SMALL.shape)
        self.assertEqual(B_r.shape, B_SMALL.shape)


class TestLoRARQFullRank(unittest.TestCase):
    def test_rq_succeeds_with_large_rank(self):
        """rank >= _RQ_MIN_ROWS should use RQ without fallback."""
        self.assertGreaterEqual(RANK_LARGE, _RQ_MIN_ROWS)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compress_lora(A_LARGE, B_LARGE, profile="lora-rq")
        self.assertEqual(result.profile, "lora-rq")
        self.assertFalse(
            any("lora-nf4" in str(warning.message) for warning in w),
            "Unexpected fallback warning for large-rank RQ",
        )

    def test_rq_roundtrip_quality(self):
        result = compress_lora(A_LARGE, B_LARGE, profile="lora-rq")
        A_r, B_r = decompress_lora(result)
        cos_A = mean_cosine_similarity(A_LARGE, A_r)
        cos_B = mean_cosine_similarity(B_LARGE, B_r)
        self.assertGreaterEqual(cos_A, 0.85, f"RQ cos_A={cos_A:.4f} < 0.85")
        self.assertGreaterEqual(cos_B, 0.85, f"RQ cos_B={cos_B:.4f} < 0.85")


class TestLoRAInputValidation(unittest.TestCase):
    def test_rank_mismatch_raises(self):
        A_bad = np.random.randn(16, 768).astype(np.float32)
        B_bad = np.random.randn(768, 32).astype(np.float32)  # rank mismatch
        with self.assertRaises(ValueError, msg="Expected ValueError for rank mismatch"):
            compress_lora(A_bad, B_bad)

    def test_wrong_dtype_raises(self):
        A_f64 = np.random.randn(16, 768).astype(np.float64)
        B_f64 = np.random.randn(768, 16).astype(np.float64)
        with self.assertRaises(ValueError):
            compress_lora(A_f64, B_f64)

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            compress_lora(A_SMALL, B_SMALL, profile="lora-unknown")

    def test_3d_input_raises(self):
        A_3d = np.random.randn(2, 16, 768).astype(np.float32)
        B_ok = np.random.randn(768, 16).astype(np.float32)
        with self.assertRaises(ValueError):
            compress_lora(A_3d, B_ok)


class TestCompressLoRAAdapter(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.adapter = {
            "q_proj": (
                rng.standard_normal((RANK, IN_FEATURES)).astype(np.float32),
                rng.standard_normal((OUT_FEATURES, RANK)).astype(np.float32),
            ),
            "v_proj": (
                rng.standard_normal((RANK, IN_FEATURES)).astype(np.float32),
                rng.standard_normal((OUT_FEATURES, RANK)).astype(np.float32),
            ),
            "k_proj": (
                rng.standard_normal((RANK, IN_FEATURES)).astype(np.float32),
                rng.standard_normal((OUT_FEATURES, RANK)).astype(np.float32),
            ),
        }

    def test_all_keys_preserved(self):
        compressed = compress_lora_adapter(self.adapter)
        self.assertEqual(set(compressed.keys()), set(self.adapter.keys()))

    def test_module_names_stored(self):
        compressed = compress_lora_adapter(self.adapter)
        for name, result in compressed.items():
            self.assertEqual(result.target_module, name)

    def test_all_results_decompressible(self):
        compressed = compress_lora_adapter(self.adapter)
        for name, result in compressed.items():
            A_r, B_r = decompress_lora(result)
            A_orig, B_orig = self.adapter[name]
            self.assertEqual(A_r.shape, A_orig.shape, f"{name}: A shape mismatch")
            self.assertEqual(B_r.shape, B_orig.shape, f"{name}: B shape mismatch")


class TestLoRADeltaApplication(unittest.TestCase):
    """The composed delta (B @ A) should be preserved after roundtrip."""

    def test_delta_roundtrip_nf4(self):
        rng = np.random.default_rng(99)
        A = rng.standard_normal((RANK, IN_FEATURES)).astype(np.float32)
        B = rng.standard_normal((OUT_FEATURES, RANK)).astype(np.float32)
        delta_orig = B @ A  # (out_features, in_features)

        result = compress_lora(A, B, profile="lora-nf4")
        A_r, B_r = decompress_lora(result)
        delta_recon = B_r @ A_r

        # Use cosine similarity on flattened delta — scale-invariant and
        # appropriate because NF4 quantization errors amplify multiplicatively
        # when A and B errors compose across the rank dimension.
        d1 = delta_orig.ravel().astype(np.float64)
        d2 = delta_recon.ravel().astype(np.float64)
        cos = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
        self.assertGreater(cos, 0.95, f"Delta cosine={cos:.4f} < 0.95")


if __name__ == "__main__":
    unittest.main()
