"""Tests for Phase 6 GPU API — cpu fallback path."""

import unittest
import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.gpu_api import (  # noqa: E402
    gpu_available,
    gpu_device_info,
    quantize_int8_batch,
    reconstruct_int8_batch,
    batch_cosine_similarity,
    batch_cosine_int8,
    batch_cosine_query,
    batch_topk_int8,
    gpu_benchmark,
)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


class TestDeviceDetection(unittest.TestCase):
    def test_gpu_available_returns_bool(self):
        result = gpu_available()
        self.assertIsInstance(result, bool)

    def test_device_info_keys(self):
        info = gpu_device_info()
        for key in ("backend", "device_name", "simd_width", "unified_memory"):
            self.assertIn(key, info)

    def test_device_info_backend_string(self):
        info = gpu_device_info()
        self.assertIsInstance(info["backend"], str)
        self.assertGreater(len(info["backend"]), 0)


# ---------------------------------------------------------------------------
# INT8 quantize / reconstruct
# ---------------------------------------------------------------------------


class TestQuantizeInt8(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.vecs = rng.standard_normal((32, 64)).astype(np.float32)

    def test_output_shapes(self):
        q, s = quantize_int8_batch(self.vecs)
        self.assertEqual(q.shape, (32, 64))
        self.assertEqual(s.shape, (32,))

    def test_output_dtypes(self):
        q, s = quantize_int8_batch(self.vecs)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(s.dtype, np.float32)

    def test_quantized_range(self):
        q, _ = quantize_int8_batch(self.vecs)
        self.assertTrue((q >= -127).all())
        self.assertTrue((q <= 127).all())

    def test_scales_positive(self):
        _, s = quantize_int8_batch(self.vecs)
        self.assertTrue((s > 0).all())

    def test_single_vector_input(self):
        v = self.vecs[0]  # 1-D
        q, s = quantize_int8_batch(v)
        self.assertEqual(q.shape, (1, 64))
        self.assertEqual(s.shape, (1,))

    def test_zero_vector(self):
        """Zero vector should not produce NaN scales."""
        v = np.zeros((1, 8), dtype=np.float32)
        q, s = quantize_int8_batch(v)
        self.assertFalse(np.isnan(s).any())
        self.assertFalse(np.isinf(s).any())

    def test_reconstruct_shapes(self):
        q, s = quantize_int8_batch(self.vecs)
        r = reconstruct_int8_batch(q, s)
        self.assertEqual(r.shape, (32, 64))
        self.assertEqual(r.dtype, np.float32)

    def test_reconstruct_cosine_quality(self):
        """Round-trip cosine similarity should be very high."""
        q, s = quantize_int8_batch(self.vecs)
        r = reconstruct_int8_batch(q, s)
        # Cosine sim per vector
        dots = (self.vecs * r).sum(axis=1)
        norms = np.linalg.norm(self.vecs, axis=1) * np.linalg.norm(r, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        cosines = dots / norms
        self.assertGreater(cosines.mean(), 0.99)


# ---------------------------------------------------------------------------
# Batch cosine similarity
# ---------------------------------------------------------------------------


class TestBatchCosineSimilarity(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.a = rng.standard_normal((16, 32)).astype(np.float32)
        self.b = rng.standard_normal((20, 32)).astype(np.float32)

    def test_output_shape(self):
        sim = batch_cosine_similarity(self.a, self.b)
        self.assertEqual(sim.shape, (16, 20))

    def test_output_dtype(self):
        sim = batch_cosine_similarity(self.a, self.b)
        self.assertEqual(sim.dtype, np.float32)

    def test_symmetry(self):
        sim_ab = batch_cosine_similarity(self.a[:5], self.a[:5])
        self.assertTrue(np.allclose(sim_ab, sim_ab.T, atol=1e-5))

    def test_self_similarity_diagonal(self):
        """Normalised vectors → self dot-product == 1."""
        sim = batch_cosine_similarity(self.a, self.a)
        diag = np.diag(sim)
        self.assertTrue(np.allclose(diag, 1.0, atol=1e-5))

    def test_values_in_range(self):
        sim = batch_cosine_similarity(self.a, self.b)
        self.assertTrue((sim >= -1.0 - 1e-5).all())
        self.assertTrue((sim <= 1.0 + 1e-5).all())

    def test_batch_cosine_int8_consistency(self):
        """batch_cosine_int8 should match batch_cosine_similarity on reconstructed."""
        q, s = quantize_int8_batch(self.a)
        recon = reconstruct_int8_batch(q, s)
        direct = batch_cosine_similarity(recon, recon)
        via_int8 = batch_cosine_int8(q, s)
        self.assertTrue(np.allclose(direct, via_int8, atol=1e-5))


# ---------------------------------------------------------------------------
# Top-k search
# ---------------------------------------------------------------------------


class TestBatchCosineQuery(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(99)
        self.db = rng.standard_normal((100, 32)).astype(np.float32)
        self.queries = rng.standard_normal((5, 32)).astype(np.float32)

    def test_output_shapes(self):
        idx, scores = batch_cosine_query(self.queries, self.db, top_k=10)
        self.assertEqual(idx.shape, (5, 10))
        self.assertEqual(scores.shape, (5, 10))

    def test_index_dtype(self):
        idx, _ = batch_cosine_query(self.queries, self.db, top_k=10)
        self.assertEqual(idx.dtype, np.int64)

    def test_scores_dtype(self):
        _, scores = batch_cosine_query(self.queries, self.db, top_k=10)
        self.assertEqual(scores.dtype, np.float32)

    def test_indices_in_range(self):
        idx, _ = batch_cosine_query(self.queries, self.db, top_k=10)
        self.assertTrue((idx >= 0).all())
        self.assertTrue((idx < len(self.db)).all())

    def test_scores_descending(self):
        """Scores within each query must be non-increasing."""
        _, scores = batch_cosine_query(self.queries, self.db, top_k=10)
        for row in scores:
            self.assertTrue(
                np.all(row[:-1] >= row[1:] - 1e-6),
                f"Scores not descending: {row}",
            )

    def test_top1_is_best(self):
        """The single top-1 result should have the highest cosine score."""
        full_sim = batch_cosine_similarity(self.queries, self.db)  # (5, 100)
        idx, scores = batch_cosine_query(self.queries, self.db, top_k=1)
        for i in range(len(self.queries)):
            best_score = float(full_sim[i].max())
            self.assertAlmostEqual(float(scores[i, 0]), best_score, places=5)

    def test_batch_topk_int8_consistency(self):
        """batch_topk_int8 indices should match float32 search."""
        q, s = quantize_int8_batch(self.db)
        idx_int8, _ = batch_topk_int8(q, s, self.queries, top_k=5)
        # Reconstruct db and run float search
        db_recon = reconstruct_int8_batch(q, s)
        idx_f32, _ = batch_cosine_query(self.queries, db_recon, top_k=5)
        # Top-1 should match between INT8 and float32 paths
        self.assertTrue(np.all(idx_int8[:, 0] == idx_f32[:, 0]))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class TestBenchmark(unittest.TestCase):
    def test_benchmark_runs(self):
        result = gpu_benchmark(n=200, d=32)
        self.assertIn("backend", result)
        self.assertIn("quantize_vecs_per_sec", result)
        self.assertIn("cosine_pairs_per_sec", result)

    def test_benchmark_positive_throughput(self):
        result = gpu_benchmark(n=200, d=32)
        self.assertGreater(result["quantize_vecs_per_sec"], 0)
        self.assertGreater(result["cosine_pairs_per_sec"], 0)


if __name__ == "__main__":
    unittest.main()
