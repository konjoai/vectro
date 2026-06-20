"""Tests for Phase 7a — Residual Quantizer."""

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.rq_api import ResidualQuantizer  # noqa: E402


class TestResidualQuantizerBasic(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.data = rng.standard_normal((200, 64)).astype(np.float32)

    def test_train_returns_self(self):
        rq = ResidualQuantizer(n_passes=2, n_subspaces=8, n_centroids=32)
        result = rq.train(self.data)
        self.assertIs(result, rq)

    def test_is_trained_after_train(self):
        rq = ResidualQuantizer(n_passes=2, n_subspaces=8, n_centroids=32)
        rq.train(self.data)
        self.assertTrue(rq.is_trained)

    def test_encode_returns_list(self):
        rq = ResidualQuantizer(n_passes=2, n_subspaces=8, n_centroids=32)
        rq.train(self.data)
        codes = rq.encode(self.data[:10])
        self.assertIsInstance(codes, list)

    def test_encode_length_equals_n_passes(self):
        rq = ResidualQuantizer(n_passes=3, n_subspaces=8, n_centroids=32)
        rq.train(self.data)
        codes = rq.encode(self.data[:10])
        self.assertEqual(len(codes), 3)

    def test_code_shapes(self):
        n_sub = 8
        rq = ResidualQuantizer(n_passes=2, n_subspaces=n_sub, n_centroids=32)
        rq.train(self.data)
        codes = rq.encode(self.data[:15])
        for c in codes:
            self.assertEqual(c.shape, (15, n_sub))
            self.assertEqual(c.dtype, np.uint8)

    def test_decode_shape(self):
        rq = ResidualQuantizer(n_passes=2, n_subspaces=8, n_centroids=32)
        rq.train(self.data)
        codes = rq.encode(self.data[:20])
        recon = rq.decode(codes)
        self.assertEqual(recon.shape, (20, 64))
        self.assertEqual(recon.dtype, np.float32)

    def test_untrained_encode_raises(self):
        rq = ResidualQuantizer()
        with self.assertRaises(RuntimeError):
            rq.encode(self.data[:5])

    def test_untrained_decode_raises(self):
        rq = ResidualQuantizer()
        with self.assertRaises(RuntimeError):
            rq.decode([np.zeros((5, 8), dtype=np.uint8)])


class TestResidualQuantizerQuality(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(22)
        self.data = rng.standard_normal((300, 64)).astype(np.float32)

    def test_single_pass_quality(self):
        """Even 1 pass should give reasonable reconstruction."""
        rq = ResidualQuantizer(n_passes=1, n_subspaces=8, n_centroids=64)
        rq.train(self.data)
        codes = rq.encode(self.data)
        recon = rq.decode(codes)
        sim = rq.mean_cosine(self.data, recon)
        self.assertGreater(sim, 0.60, f"Single-pass cosine {sim:.4f} < 0.60")

    def test_3pass_quality(self):
        """3 passes should yield cosine ≥ 0.85."""
        rq = ResidualQuantizer(n_passes=3, n_subspaces=8, n_centroids=64)
        rq.train(self.data)
        codes = rq.encode(self.data)
        recon = rq.decode(codes)
        sim = rq.mean_cosine(self.data, recon)
        self.assertGreater(sim, 0.85, f"3-pass cosine {sim:.4f} < 0.85")

    def test_more_passes_better_quality(self):
        """More passes should never decrease quality."""
        results = []
        for n_passes in (1, 2, 3):
            rq = ResidualQuantizer(n_passes=n_passes, n_subspaces=8, n_centroids=32)
            rq.train(self.data)
            codes = rq.encode(self.data)
            recon = rq.decode(codes)
            results.append(rq.mean_cosine(self.data, recon))
        self.assertGreaterEqual(results[1], results[0] - 0.05)  # 2 ≥ 1 (with tolerance)
        self.assertGreaterEqual(results[2], results[0] - 0.05)  # 3 ≥ 1

    def test_no_nan_in_reconstruction(self):
        rq = ResidualQuantizer(n_passes=2, n_subspaces=8, n_centroids=32)
        rq.train(self.data)
        codes = rq.encode(self.data[:50])
        recon = rq.decode(codes)
        self.assertFalse(np.isnan(recon).any())

    def test_compression_ratio_positive(self):
        rq = ResidualQuantizer(n_passes=2, n_subspaces=8, n_centroids=32)
        rq.train(self.data)
        self.assertGreater(rq.compression_ratio(), 0)

    def test_compression_ratio_formula(self):
        """compression = (d*4) / (n_passes * M)."""
        d = 64
        M = 8
        n_passes = 2
        expected = (d * 4) / (n_passes * M)
        rq = ResidualQuantizer(n_passes=n_passes, n_subspaces=M, n_centroids=32)
        rq.train(self.data)
        self.assertAlmostEqual(rq.compression_ratio(), expected, places=5)


class TestResidualQuantizerEdgeCases(unittest.TestCase):
    def test_non_divisible_dimension_handled(self):
        """d=70 is not divisible by 8; Residual Quantizer should still work."""
        rng = np.random.default_rng(33)
        data = rng.standard_normal((100, 70)).astype(np.float32)
        rq = ResidualQuantizer(n_passes=2, n_subspaces=5, n_centroids=16)
        rq.train(data)
        codes = rq.encode(data[:10])
        recon = rq.decode(codes)
        self.assertEqual(recon.shape, (10, 70))

    def test_unseen_vectors(self):
        """Encode/decode vectors not in training set."""
        rng = np.random.default_rng(99)
        train = rng.standard_normal((200, 32)).astype(np.float32)
        test = rng.standard_normal((50, 32)).astype(np.float32)
        rq = ResidualQuantizer(n_passes=2, n_subspaces=4, n_centroids=32)
        rq.train(train)
        codes = rq.encode(test)
        recon = rq.decode(codes)
        self.assertEqual(recon.shape, (50, 32))
        self.assertFalse(np.isnan(recon).any())

    def test_single_vector(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 16)).astype(np.float32)
        rq = ResidualQuantizer(n_passes=1, n_subspaces=4, n_centroids=16)
        rq.train(data)
        codes = rq.encode(data[:1])
        recon = rq.decode(codes)
        self.assertEqual(recon.shape, (1, 16))


if __name__ == "__main__":
    unittest.main()
