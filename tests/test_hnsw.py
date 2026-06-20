"""Tests for HNSW Index (Phase 5)."""

import os
import tempfile
import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.hnsw_api import (  # noqa: E402
    HNSWIndex,
    build_hnsw_index,
    hnsw_search,
    recall_at_k,
    hnsw_compression_info,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N_SMALL = 200
D = 64


def _make_vecs(n: int = N_SMALL, d: int = D) -> np.ndarray:
    return RNG.standard_normal((n, d)).astype(np.float32)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestHNSWConstruction(unittest.TestCase):
    def test_default_params(self):
        idx = HNSWIndex()
        self.assertEqual(idx.M, 16)
        self.assertEqual(idx.M0, 32)
        self.assertEqual(idx.ef_construction, 200)
        self.assertEqual(idx.space, "cosine")

    def test_custom_params(self):
        idx = HNSWIndex(M=8, ef_construction=100, space="l2")
        self.assertEqual(idx.M, 8)
        self.assertEqual(idx.M0, 16)
        self.assertEqual(idx.space, "l2")

    def test_invalid_M(self):
        with self.assertRaises(ValueError):
            HNSWIndex(M=1)

    def test_invalid_ef(self):
        with self.assertRaises(ValueError):
            HNSWIndex(M=16, ef_construction=4)

    def test_invalid_space(self):
        with self.assertRaises(ValueError):
            HNSWIndex(M=16, ef_construction=200, space="hamming")

    def test_empty_index_len(self):
        idx = HNSWIndex()
        self.assertEqual(len(idx), 0)

    def test_single_vector_add(self):
        idx = HNSWIndex()
        v = np.ones(D, dtype=np.float32)
        idx.add(v)
        self.assertEqual(len(idx), 1)

    def test_batch_add(self):
        vecs = _make_vecs(50)
        idx = HNSWIndex()
        idx.add(vecs)
        self.assertEqual(len(idx), 50)

    def test_build_hnsw_index_helper(self):
        vecs = _make_vecs(100)
        idx = build_hnsw_index(vecs, M=8, ef_construction=50)
        self.assertEqual(len(idx), 100)
        self.assertIsInstance(idx, HNSWIndex)

    def test_repr(self):
        idx = HNSWIndex(M=8)
        idx.add(_make_vecs(10))
        r = repr(idx)
        self.assertIn("HNSWIndex", r)
        self.assertIn("n=10", r)


# ---------------------------------------------------------------------------
# Search correctness
# ---------------------------------------------------------------------------


class TestHNSWSearch(unittest.TestCase):
    def setUp(self):
        self.vecs = _make_vecs(N_SMALL, D)
        self.idx = build_hnsw_index(self.vecs, M=16, ef_construction=100)

    def test_search_returns_k_results(self):
        q = _make_vecs(1, D)[0]
        indices, distances = self.idx.search(q, k=10)
        self.assertEqual(len(indices), 10)
        self.assertEqual(len(distances), 10)

    def test_indices_dtype(self):
        q = _make_vecs(1, D)[0]
        indices, _ = self.idx.search(q, k=5)
        self.assertEqual(indices.dtype, np.int64)

    def test_distances_dtype(self):
        _, distances = self.idx.search(_make_vecs(1, D)[0], k=5)
        self.assertEqual(distances.dtype, np.float32)

    def test_indices_in_range(self):
        q = _make_vecs(1, D)[0]
        indices, _ = self.idx.search(q, k=10)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < N_SMALL).all())

    def test_distances_non_negative(self):
        q = _make_vecs(1, D)[0]
        _, distances = self.idx.search(q, k=10)
        self.assertTrue((distances >= -1e-5).all())

    def test_distances_sorted(self):
        q = _make_vecs(1, D)[0]
        _, distances = self.idx.search(q, k=10)
        self.assertTrue(np.all(distances[:-1] <= distances[1:] + 1e-6))

    def test_hnsw_search_helper(self):
        q = _make_vecs(1, D)[0]
        indices, distances = hnsw_search(self.idx, q, k=5)
        self.assertEqual(len(indices), 5)

    def test_search_empty_index(self):
        idx = HNSWIndex()
        q = np.ones(D, dtype=np.float32)
        indices, distances = idx.search(q, k=5)
        self.assertEqual(len(indices), 0)

    def test_k_larger_than_n(self):
        idx = HNSWIndex()
        idx.add(_make_vecs(5, D))
        q = _make_vecs(1, D)[0]
        indices, distances = idx.search(q, k=100)
        # Should return at most 5 results (all available vectors)
        self.assertLessEqual(len(indices), 5)

    def test_single_vector_self_search(self):
        """A 1-vector index should return itself for any query."""
        v = np.array([1.0] + [0.0] * (D - 1), dtype=np.float32)
        idx = HNSWIndex()
        idx.add(v)
        indices, distances = idx.search(v, k=1)
        self.assertEqual(int(indices[0]), 0)


# ---------------------------------------------------------------------------
# Self-recall: each indexed vector should retrieve itself
# ---------------------------------------------------------------------------


class TestHNSWSelfRecall(unittest.TestCase):
    def test_self_recall_cosine(self):
        """Every indexed vector should be in its own top-1 result."""
        vecs = _make_vecs(100, D)
        idx = build_hnsw_index(vecs, M=16, ef_construction=200)
        hits = 0
        for i, v in enumerate(vecs):
            indices, _ = idx.search(v, k=1, ef=100)
            if len(indices) > 0 and int(indices[0]) == i:
                hits += 1
        # Expect >= 95% self-recall (should be near 100%)
        self.assertGreaterEqual(hits / len(vecs), 0.95)

    def test_self_recall_l2(self):
        """Same property for L2 space."""
        vecs = _make_vecs(100, D)
        idx = build_hnsw_index(vecs, M=16, ef_construction=200, space="l2")
        hits = 0
        for i, v in enumerate(vecs):
            indices, _ = idx.search(v, k=1, ef=100)
            if len(indices) > 0 and int(indices[0]) == i:
                hits += 1
        self.assertGreaterEqual(hits / len(vecs), 0.95)


# ---------------------------------------------------------------------------
# Recall@k quality
# ---------------------------------------------------------------------------


class TestHNSWRecallAtK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(7)
        cls.n_db = 500
        cls.n_q = 50
        cls.d = 64
        cls.k = 10
        cls.db = cls.rng.standard_normal((cls.n_db, cls.d)).astype(np.float32)
        cls.queries = cls.rng.standard_normal((cls.n_q, cls.d)).astype(np.float32)

        # Normalise for cosine
        db_norm = cls.db / np.linalg.norm(cls.db, axis=1, keepdims=True)
        q_norm = cls.queries / np.linalg.norm(cls.queries, axis=1, keepdims=True)

        # Brute-force ground truth
        sims = q_norm @ db_norm.T  # (n_q, n_db)
        cls.gt = np.argsort(-sims, axis=1)[:, : cls.k]  # (n_q, k) IDs

        cls.idx = build_hnsw_index(cls.db, M=16, ef_construction=200)

    def test_recall_at_10(self):
        r = recall_at_k(self.idx, self.queries, self.gt, k=self.k, ef=64)
        # Target: >= 0.90 on this small dataset with M=16
        self.assertGreaterEqual(r, 0.90, msg=f"Recall@10 was {r:.3f}, expected >= 0.90")

    def test_high_ef_improves_recall(self):
        r_low = recall_at_k(self.idx, self.queries, self.gt, k=self.k, ef=10)
        r_high = recall_at_k(self.idx, self.queries, self.gt, k=self.k, ef=100)
        # Higher ef should not hurt recall
        self.assertGreaterEqual(r_high, r_low - 0.05)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestHNSWSaveLoad(unittest.TestCase):
    def _build_small(self) -> HNSWIndex:
        vecs = _make_vecs(80, D)
        return build_hnsw_index(vecs, M=8, ef_construction=50)

    def test_save_creates_file(self):
        idx = self._build_small()
        with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_load_restores_size(self):
        idx = self._build_small()
        with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            idx2 = HNSWIndex.load(path)
            self.assertEqual(len(idx2), len(idx))
        finally:
            os.unlink(path)

    def test_load_restores_params(self):
        idx = HNSWIndex(M=8, ef_construction=50, space="l2")
        idx.add(_make_vecs(20, D))
        with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            idx2 = HNSWIndex.load(path)
            self.assertEqual(idx2.M, 8)
            self.assertEqual(idx2.ef_construction, 50)
            self.assertEqual(idx2.space, "l2")
        finally:
            os.unlink(path)

    def test_search_after_load(self):
        vecs = _make_vecs(80, D)
        idx = build_hnsw_index(vecs, M=8, ef_construction=50)
        with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            idx2 = HNSWIndex.load(path)
            q = vecs[0]
            indices1, _ = idx.search(q, k=5)
            indices2, _ = idx2.search(q, k=5)
            # Results should be identical after load
            self.assertTrue(np.array_equal(indices1, indices2))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Compression info
# ---------------------------------------------------------------------------


class TestCompressionInfo(unittest.TestCase):
    def test_key_presence(self):
        info = hnsw_compression_info(768)
        for key in ("bytes_fp32", "bytes_int8", "bytes_graph", "bytes_total", "compression_ratio"):
            self.assertIn(key, info)

    def test_fp32_bytes(self):
        info = hnsw_compression_info(768)
        self.assertEqual(info["bytes_fp32"], 768 * 4)

    def test_compression_ratio_positive(self):
        info = hnsw_compression_info(768, M=16)
        self.assertGreater(info["compression_ratio"], 1.0)

    def test_higher_M_lower_ratio(self):
        r_low = hnsw_compression_info(768, M=8)["compression_ratio"]
        r_high = hnsw_compression_info(768, M=32)["compression_ratio"]
        # More links = more bytes = smaller ratio
        self.assertGreaterEqual(r_low, r_high)


if __name__ == "__main__":
    unittest.main()
