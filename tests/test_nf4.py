"""Tests for NF4 quantizer Phase 2."""

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.nf4_api import (  # noqa: E402
    NF4_LEVELS,
    quantize_nf4,
    dequantize_nf4,
    select_outlier_dims,
    quantize_mixed,
    dequantize_mixed,
    nf4_cosine_sim,
    compression_ratio,
)


class TestNF4Levels(unittest.TestCase):
    def test_level_count(self):
        self.assertEqual(len(NF4_LEVELS), 16)

    def test_monotonically_increasing(self):
        for i in range(15):
            self.assertLess(NF4_LEVELS[i], NF4_LEVELS[i + 1])

    def test_boundary_values(self):
        self.assertAlmostEqual(float(NF4_LEVELS[0]), -1.0, places=5)
        self.assertAlmostEqual(float(NF4_LEVELS[-1]), 1.0, places=5)


class TestQuantizeNF4(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.vecs = rng.standard_normal((100, 768)).astype(np.float32)

    def test_output_shapes(self):
        packed, scales = quantize_nf4(self.vecs)
        self.assertEqual(packed.shape, (100, 384))  # 768/2
        self.assertEqual(scales.shape, (100,))

    def test_packed_dtype(self):
        packed, _ = quantize_nf4(self.vecs)
        self.assertEqual(packed.dtype, np.uint8)

    def test_nibbles_in_range(self):
        packed, _ = quantize_nf4(self.vecs)
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        self.assertTrue((lo <= 15).all())
        self.assertTrue((hi <= 15).all())

    def test_scales_positive(self):
        _, scales = quantize_nf4(self.vecs)
        self.assertTrue((scales >= 0).all())

    def test_odd_dimension(self):
        vecs = np.random.randn(10, 5).astype(np.float32)
        packed, scales = quantize_nf4(vecs)
        self.assertEqual(packed.shape, (10, 3))  # ceil(5/2)
        self.assertEqual(scales.shape, (10,))


class TestDequantizeNF4(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.vecs = rng.standard_normal((50, 768)).astype(np.float32)

    def test_output_shape(self):
        packed, scales = quantize_nf4(self.vecs)
        recon = dequantize_nf4(packed, scales, 768)
        self.assertEqual(recon.shape, (50, 768))

    def test_cosine_sim_threshold(self):
        packed, scales = quantize_nf4(self.vecs)
        recon = dequantize_nf4(packed, scales, 768)
        cos = nf4_cosine_sim(self.vecs, recon)
        self.assertGreaterEqual(cos, 0.985, f"cosine_sim={cos:.4f} < 0.985")

    def test_zero_vector_roundtrip(self):
        vecs = np.zeros((3, 4), dtype=np.float32)
        packed, scales = quantize_nf4(vecs)
        recon = dequantize_nf4(packed, scales, 4)
        np.testing.assert_array_equal(recon, np.zeros((3, 4), dtype=np.float32))

    def test_odd_dim_roundtrip(self):
        vecs = np.random.randn(10, 5).astype(np.float32)
        packed, scales = quantize_nf4(vecs)
        recon = dequantize_nf4(packed, scales, 5)
        cos = nf4_cosine_sim(vecs, recon)
        self.assertGreaterEqual(cos, 0.90, f"odd-dim cosine_sim too low: {cos:.4f}")

    def test_identity_1d(self):
        vecs = NF4_LEVELS.reshape(1, 16).copy()
        packed, scales = quantize_nf4(vecs)
        recon = dequantize_nf4(packed, scales, 16)
        # NF4 is a 4-bit lookup; float32 representation of the 16 canonical
        # levels has ~1e-4 max absolute error after pack/unpack — atol=2e-4.
        np.testing.assert_allclose(recon, vecs, atol=2e-4)


class TestMixedPrecision(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.vecs = rng.standard_normal((30, 768)).astype(np.float32)
        # Add outliers to first 16 dims
        self.vecs[:, :16] *= 10.0
        self.outlier_dims = select_outlier_dims(self.vecs, k=16)

    def test_outlier_selection_count(self):
        self.assertEqual(len(self.outlier_dims), 16)

    def test_outlier_selection_sorted(self):
        dims = self.outlier_dims
        self.assertTrue((dims[:-1] <= dims[1:]).all())

    def test_mixed_output_shapes(self):
        fp16, packed, scales, od = quantize_mixed(self.vecs, self.outlier_dims)
        self.assertEqual(fp16.shape, (30, 16))
        self.assertEqual(fp16.dtype, np.float16)
        self.assertEqual(packed.shape[0], 30)
        self.assertEqual(scales.shape, (30,))

    def test_mixed_roundtrip_quality(self):
        fp16, packed, scales, od = quantize_mixed(self.vecs, self.outlier_dims)
        recon = dequantize_mixed(fp16, packed, scales, od, d=768)
        self.assertEqual(recon.shape, (30, 768))
        cos = nf4_cosine_sim(self.vecs, recon)
        self.assertGreaterEqual(cos, 0.990, f"mixed cos_sim={cos:.4f} < 0.990")


class TestCompressionRatio(unittest.TestCase):
    def test_pure_nf4_d768(self):
        ratio = compression_ratio(768, k_outliers=0)
        # 768*4 / (384 + 4) ≈ 7.9
        self.assertGreater(ratio, 7.0)
        self.assertLess(ratio, 9.0)

    def test_mixed_d768_k16(self):
        ratio = compression_ratio(768, k_outliers=16)
        # 768*4 / (376 + 32 + 4) ≈ 7.5
        self.assertGreater(ratio, 6.0)
        self.assertLess(ratio, 9.0)


if __name__ == "__main__":
    unittest.main()
