"""Tests for Product Quantization (Phase 3)."""

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

# Skip if sklearn not available
try:
    import sklearn  # noqa: F401

    _SKLEARN = True
except ImportError:
    _SKLEARN = False

from python.pq_api import (  # noqa: E402
    train_pq_codebook,
    pq_encode,
    pq_decode,
    pq_distance_table,
    pq_search,
    pq_compression_ratio,
)


@unittest.skipUnless(_SKLEARN, "scikit-learn required for PQ tests")
class TestPQCodebook(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        # Use small d=32, M=4 for fast training in CI
        self.vecs = rng.standard_normal((500, 32)).astype(np.float32)
        self.cb = train_pq_codebook(self.vecs, n_subspaces=4, n_centroids=16, max_iter=10)

    def test_codebook_shape(self):
        cb = self.cb
        self.assertEqual(cb.centroids.shape, (4, 16, 8))  # M=4, K=16, sub_dim=8

    def test_invalid_d_not_divisible(self):
        with self.assertRaises(ValueError):
            train_pq_codebook(np.zeros((10, 7), dtype=np.float32), n_subspaces=4)

    def test_invalid_k_too_large(self):
        with self.assertRaises(ValueError):
            train_pq_codebook(self.vecs, n_subspaces=4, n_centroids=300)


@unittest.skipUnless(_SKLEARN, "scikit-learn required for PQ tests")
class TestPQEncodeDecode(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.vecs = rng.standard_normal((200, 32)).astype(np.float32)
        self.cb = train_pq_codebook(self.vecs, n_subspaces=4, n_centroids=16, max_iter=10)

    def test_encode_shape(self):
        codes = pq_encode(self.vecs, self.cb)
        self.assertEqual(codes.shape, (200, 4))
        self.assertEqual(codes.dtype, np.uint8)

    def test_codes_in_range(self):
        codes = pq_encode(self.vecs, self.cb)
        self.assertTrue((codes < 16).all())

    def test_decode_shape(self):
        codes = pq_encode(self.vecs, self.cb)
        recon = pq_decode(codes, self.cb)
        self.assertEqual(recon.shape, (200, 32))

    def test_reconstruction_quality(self):
        """PQ roundtrip should preserve cosine similarity at a reasonable level."""
        codes = pq_encode(self.vecs, self.cb)
        recon = pq_decode(codes, self.cb)
        dot = (self.vecs * recon).sum(axis=1)
        n1 = np.linalg.norm(self.vecs, axis=1)
        n2 = np.linalg.norm(recon, axis=1)
        cos = (dot / (n1 * n2 + 1e-8)).mean()
        # K=16 centroids, d=32 sub-spaces of dim 8 — realistic floor is ~0.65
        # Production usage (K=256, M=96) achieves cosine_sim >= 0.95
        self.assertGreaterEqual(float(cos), 0.65, f"PQ cosine_sim={cos:.4f} < 0.65")


@unittest.skipUnless(_SKLEARN, "scikit-learn required for PQ tests")
class TestPQSearch(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.db = rng.standard_normal((100, 32)).astype(np.float32)
        self.cb = train_pq_codebook(self.db, n_subspaces=4, n_centroids=16, max_iter=10)
        self.codes = pq_encode(self.db, self.cb)

    def test_search_returns_k_results(self):
        q = np.random.randn(32).astype(np.float32)
        idx, dists = pq_search(q, self.codes, self.cb, top_k=5)
        self.assertEqual(len(idx), 5)
        self.assertEqual(len(dists), 5)

    def test_search_dist_ascending(self):
        q = np.random.randn(32).astype(np.float32)
        _, dists = pq_search(q, self.codes, self.cb, top_k=5)
        self.assertTrue((np.diff(dists) >= 0).all())

    def test_distance_table_shape(self):
        q = np.random.randn(32).astype(np.float32)
        table = pq_distance_table(q, self.cb)
        self.assertEqual(table.shape, (4, 16))


class TestPQCompressionRatio(unittest.TestCase):
    def test_d768_m96(self):
        ratio = pq_compression_ratio(768, 96)
        # 768*4 / 96 = 32
        self.assertAlmostEqual(ratio, 32.0, places=3)

    def test_d384_m48(self):
        ratio = pq_compression_ratio(384, 48)
        self.assertAlmostEqual(ratio, 32.0, places=3)


if __name__ == "__main__":
    unittest.main()
