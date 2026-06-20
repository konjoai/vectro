"""Tests for Binary Quantization (Phase 4)."""

import unittest
import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.binary_api import (  # noqa: E402
    quantize_binary,
    dequantize_binary,
    hamming_distance_batch,
    binary_search,
    matryoshka_encode,
    binary_compression_ratio,
)


class TestQuantizeBinary(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.vecs = rng.standard_normal((50, 768)).astype(np.float32)

    def test_output_shape(self):
        packed = quantize_binary(self.vecs)
        self.assertEqual(packed.shape, (50, 96))  # 768/8

    def test_dtype(self):
        packed = quantize_binary(self.vecs)
        self.assertEqual(packed.dtype, np.uint8)

    def test_odd_dimension(self):
        vecs = np.random.randn(10, 7).astype(np.float32)
        packed = quantize_binary(vecs)
        self.assertEqual(packed.shape, (10, 1))  # ceil(7/8) = 1

    def test_single_vector(self):
        v = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], dtype=np.float32)
        packed = quantize_binary(v[np.newaxis], normalize=False)
        self.assertEqual(packed.shape, (1, 1))
        # bits: 1,0,1,0,1,0,1,0 => 0b01010101 = 85
        self.assertEqual(int(packed[0, 0]), 0b01010101)

    def test_all_positive(self):
        v = np.ones((1, 8), dtype=np.float32)
        packed = quantize_binary(v, normalize=False)
        self.assertEqual(int(packed[0, 0]), 0xFF)

    def test_all_negative(self):
        v = -np.ones((1, 8), dtype=np.float32)
        packed = quantize_binary(v, normalize=False)
        self.assertEqual(int(packed[0, 0]), 0x00)


class TestDequantizeBinary(unittest.TestCase):
    def test_roundtrip_shape(self):
        vecs = np.random.randn(20, 64).astype(np.float32)
        packed = quantize_binary(vecs, normalize=False)
        recon = dequantize_binary(packed, d=64)
        self.assertEqual(recon.shape, (20, 64))

    def test_values_are_pm1(self):
        vecs = np.random.randn(10, 64).astype(np.float32)
        packed = quantize_binary(vecs, normalize=False)
        recon = dequantize_binary(packed, d=64)
        unique = np.unique(recon)
        self.assertEqual(set(unique.tolist()), {-1.0, 1.0})

    def test_sign_preserved(self):
        vecs = np.random.randn(5, 8).astype(np.float32)
        packed = quantize_binary(vecs, normalize=False)
        recon = dequantize_binary(packed, d=8)
        expected = np.sign(vecs)
        np.testing.assert_array_equal(recon, expected)


class TestHammingDistance(unittest.TestCase):
    def test_identical_vectors_distance_zero(self):
        v = np.array([[0b10101010, 0b11001100]], dtype=np.uint8)
        dists = hamming_distance_batch(v.ravel(), v)
        self.assertEqual(int(dists[0]), 0)

    def test_flipped_all_bits(self):
        v = np.array([[0xFF, 0xFF]], dtype=np.uint8)
        q = np.array([[0x00, 0x00]], dtype=np.uint8)
        dists = hamming_distance_batch(q.ravel(), v)
        self.assertEqual(int(dists[0]), 16)

    def test_batch_shape(self):
        db = np.random.randint(0, 256, size=(100, 96), dtype=np.uint8)
        q = np.random.randint(0, 256, size=(96,), dtype=np.uint8)
        dists = hamming_distance_batch(q, db)
        self.assertEqual(dists.shape, (100,))


class TestBinarySearch(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20)
        self.db = rng.standard_normal((200, 768)).astype(np.float32)
        self.db_packed = quantize_binary(self.db)

    def test_returns_top_k(self):
        q = np.random.randn(768).astype(np.float32)
        idx, dists = binary_search(q, self.db_packed, top_k=10)
        self.assertEqual(len(idx), 10)

    def test_distances_ascending(self):
        q = np.random.randn(768).astype(np.float32)
        _, dists = binary_search(q, self.db_packed, top_k=10)
        self.assertTrue((np.diff(dists) >= 0).all())

    def test_self_search(self):
        """A vector's own encoding should recall itself with Hamming dist 0."""
        q = self.db[0].copy()
        idx, dists = binary_search(q, self.db_packed, top_k=1)
        self.assertEqual(int(idx[0]), 0)
        self.assertEqual(int(dists[0]), 0)


class TestMatryoshkaEncode(unittest.TestCase):
    def test_output_keys(self):
        vecs = np.random.randn(10, 768).astype(np.float32)
        dims = [64, 128, 256]
        result = matryoshka_encode(vecs, dims)
        self.assertEqual(set(result.keys()), {64, 128, 256})

    def test_output_shapes(self):
        vecs = np.random.randn(10, 768).astype(np.float32)
        result = matryoshka_encode(vecs, [64, 256, 768])
        self.assertEqual(result[64].shape, (10, 8))  # ceil(64/8)
        self.assertEqual(result[256].shape, (10, 32))  # ceil(256/8)
        self.assertEqual(result[768].shape, (10, 96))  # ceil(768/8)


class TestBinaryCompressionRatio(unittest.TestCase):
    def test_d768(self):
        ratio = binary_compression_ratio(768)
        # 768*4 / 96 = 32
        self.assertAlmostEqual(ratio, 32.0, places=3)

    def test_d256(self):
        ratio = binary_compression_ratio(256)
        # 256*4 / 32 = 32
        self.assertAlmostEqual(ratio, 32.0, places=3)


if __name__ == "__main__":
    unittest.main()
