"""Unit tests for the shared mmr_select + cosine_scores utility."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.retrieval.mmr import cosine_scores, mmr_select  # noqa: E402


class TestCosineScores(unittest.TestCase):
    def test_unit_vectors_diagonal(self):
        mat = np.eye(4, dtype=np.float32)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        scores = cosine_scores(q, mat)
        self.assertAlmostEqual(float(scores[0]), 1.0, places=5)
        for i in range(1, 4):
            self.assertAlmostEqual(float(scores[i]), 0.0, places=5)

    def test_query_self_similarity_is_one(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal(64).astype(np.float32)
        mat = v[None, :]
        scores = cosine_scores(v, mat)
        self.assertAlmostEqual(float(scores[0]), 1.0, places=4)

    def test_orthogonal_vectors_score_zero(self):
        mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        q = np.array([1.0, 0.0], dtype=np.float32)
        scores = cosine_scores(q, mat)
        self.assertAlmostEqual(float(scores[0]), 1.0, places=5)
        self.assertAlmostEqual(float(scores[1]), 0.0, places=5)

    def test_zero_query_does_not_explode(self):
        mat = np.eye(3, dtype=np.float32)
        q = np.zeros(3, dtype=np.float32)
        scores = cosine_scores(q, mat)
        # All should be ~0, no NaN/Inf
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_shape_matches_n(self):
        mat = np.random.RandomState(0).randn(50, 16).astype(np.float32)
        q = np.random.RandomState(1).randn(16).astype(np.float32)
        scores = cosine_scores(q, mat)
        self.assertEqual(scores.shape, (50,))

    def test_scores_in_unit_range(self):
        rng = np.random.default_rng(7)
        mat = rng.standard_normal((20, 32)).astype(np.float32)
        q = rng.standard_normal(32).astype(np.float32)
        scores = cosine_scores(q, mat)
        self.assertTrue(np.all(scores >= -1.0001))
        self.assertTrue(np.all(scores <= 1.0001))


class TestMMRSelect(unittest.TestCase):
    def test_returns_k_indices(self):
        rng = np.random.default_rng(1)
        mat = rng.standard_normal((20, 16)).astype(np.float32)
        q = rng.standard_normal(16).astype(np.float32)
        idx = mmr_select(mat, q, k=5, fetch_k=10)
        self.assertEqual(len(idx), 5)

    def test_indices_are_unique(self):
        rng = np.random.default_rng(2)
        mat = rng.standard_normal((30, 16)).astype(np.float32)
        q = rng.standard_normal(16).astype(np.float32)
        idx = mmr_select(mat, q, k=8, fetch_k=20)
        self.assertEqual(len(set(idx.tolist())), len(idx))

    def test_lambda_one_selects_top_relevance_first(self):
        """lambda_mult=1.0 → first selection equals argmax(cosine)."""
        rng = np.random.default_rng(3)
        mat = rng.standard_normal((30, 16)).astype(np.float32)
        q = rng.standard_normal(16).astype(np.float32)

        scores = cosine_scores(q, mat)
        top1 = int(np.argmax(scores))

        idx = mmr_select(mat, q, k=1, fetch_k=10, lambda_mult=1.0)
        self.assertEqual(int(idx[0]), top1)

    def test_k_larger_than_n_clamped(self):
        rng = np.random.default_rng(4)
        mat = rng.standard_normal((5, 8)).astype(np.float32)
        q = rng.standard_normal(8).astype(np.float32)
        idx = mmr_select(mat, q, k=20, fetch_k=20)
        self.assertEqual(len(idx), 5)

    def test_fetch_k_clamped_to_n(self):
        rng = np.random.default_rng(5)
        mat = rng.standard_normal((6, 8)).astype(np.float32)
        q = rng.standard_normal(8).astype(np.float32)
        idx = mmr_select(mat, q, k=3, fetch_k=100)
        self.assertEqual(len(idx), 3)


if __name__ == "__main__":
    unittest.main()
