"""Tests for python/batch_api.py — VectroBatchProcessor, BatchQuantizationResult, BatchCompressionAnalyzer."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.batch_api import (  # noqa: E402
    BatchCompressionAnalyzer,
    BatchQuantizationResult,
    VectroBatchProcessor,
    benchmark_batch_compression,
    quantize_embeddings_batch,
)

_RNG = np.random.default_rng(42)
_VECS = _RNG.standard_normal((32, 16)).astype(np.float32)


def _processor() -> VectroBatchProcessor:
    return VectroBatchProcessor(backend="python")


def _result(n: int = 8, d: int = 8, profile: str = "balanced") -> BatchQuantizationResult:
    vecs = _RNG.standard_normal((n, d)).astype(np.float32)
    return _processor().quantize_batch(vecs, profile)


# ---------------------------------------------------------------------------
# BatchQuantizationResult
# ---------------------------------------------------------------------------


class TestBatchQuantizationResult(unittest.TestCase):
    def test_quantize_batch_returns_correct_type(self):
        result = _result()
        self.assertIsInstance(result, BatchQuantizationResult)

    def test_quantize_batch_shape_correct(self):
        n, d = 12, 10
        vecs = _RNG.standard_normal((n, d)).astype(np.float32)
        result = _processor().quantize_batch(vecs)
        self.assertEqual(result.batch_size, n)
        self.assertEqual(result.vector_dim, d)
        self.assertEqual(len(result.quantized_vectors), n)
        self.assertEqual(len(result.scales), n)

    def test_quantize_batch_fast_profile(self):
        result = _result(profile="fast")
        self.assertIsInstance(result, BatchQuantizationResult)

    def test_quantize_batch_balanced_profile(self):
        result = _result(profile="balanced")
        self.assertIsInstance(result, BatchQuantizationResult)

    def test_quantize_batch_quality_profile(self):
        result = _result(profile="quality")
        self.assertIsInstance(result, BatchQuantizationResult)

    def test_unknown_profile_falls_back_to_balanced(self):
        """Unknown profile silently falls back to 'balanced' without raising."""
        n, d = 4, 8
        vecs = _RNG.standard_normal((n, d)).astype(np.float32)
        # Should not raise; must produce a valid result with the same shape
        result = _processor().quantize_batch(vecs, profile="nonexistent_profile")
        self.assertEqual(result.batch_size, n)
        self.assertEqual(result.vector_dim, d)

    def test_get_vector_returns_quantized_and_scale(self):
        result = _result()
        q, s = result.get_vector(0)
        self.assertIsInstance(q, np.ndarray)
        self.assertIsInstance(s, float)

    def test_get_vector_index_error_on_out_of_bounds(self):
        result = _result(n=4)
        with self.assertRaises(IndexError):
            result.get_vector(100)

    def test_reconstruct_batch_shape_matches(self):
        n, d = 6, 12
        result = _result(n=n, d=d)
        recon = result.reconstruct_batch()
        self.assertEqual(recon.shape, (n, d))

    def test_reconstruct_batch_float32_dtype(self):
        result = _result()
        recon = result.reconstruct_batch()
        self.assertEqual(recon.dtype, np.float32)

    def test_reconstruct_vector_matches_manual(self):
        """Reconstruct a single int8 vector manually and compare."""
        n, d = 4, 8
        vecs = _RNG.standard_normal((n, d)).astype(np.float32)
        result = _processor().quantize_batch(vecs)
        for i in range(n):
            q, s = result.get_vector(i)
            expected = q.astype(np.float32) * s
            actual = result.reconstruct_vector(i)
            np.testing.assert_array_almost_equal(actual, expected, decimal=6)

    def test_batch_compression_ratio_positive(self):
        result = _result()
        self.assertGreater(result.compression_ratio, 0.0)


# ---------------------------------------------------------------------------
# VectroBatchProcessor — streaming
# ---------------------------------------------------------------------------


class TestVectroBatchProcessorStreaming(unittest.TestCase):
    def test_quantize_streaming_chunk_count(self):
        n, chunk_size = 25, 10
        vecs = _RNG.standard_normal((n, 8)).astype(np.float32)
        results = _processor().quantize_streaming(vecs, chunk_size=chunk_size)
        # 25 vectors in chunks of 10: ceil(25/10) = 3 chunks
        import math

        self.assertEqual(len(results), math.ceil(n / chunk_size))

    def test_quantize_streaming_chunk_shapes(self):
        n, d, chunk_size = 20, 6, 7
        vecs = _RNG.standard_normal((n, d)).astype(np.float32)
        results = _processor().quantize_streaming(vecs, chunk_size=chunk_size)
        total = sum(r.batch_size for r in results)
        self.assertEqual(total, n)
        for r in results:
            self.assertEqual(r.vector_dim, d)

    def test_binary_profile_compression_ratio_approx_32x(self):
        """Binary profile must report ~32x compression (1 bit/dim, not ~4x INT8)."""
        d = 256
        n = 10
        vecs = _RNG.standard_normal((n, d)).astype(np.float32)
        result = _processor().quantize_batch(vecs, profile="binary")
        self.assertEqual(result.precision_mode, "binary")
        # 256 dims → ceil(256/8)=32 bytes/vec; ratio = 256*4/32 = 32.0x exactly
        self.assertAlmostEqual(result.compression_ratio, 32.0, places=1)
        self.assertEqual(result.batch_size, n)
        self.assertEqual(result.vector_dim, d)

    def test_binary_profile_packed_bytes_correct_shape(self):
        """Each binary-encoded vector must be ceil(d/8) uint8 bytes."""
        d = 768
        n = 4
        vecs = _RNG.standard_normal((n, d)).astype(np.float32)
        result = _processor().quantize_batch(vecs, profile="binary")
        expected_bytes_per_vec = (d + 7) // 8  # 96 for d=768
        for packed_vec in result.quantized_vectors:
            self.assertEqual(packed_vec.dtype, np.uint8)
            self.assertEqual(packed_vec.shape, (expected_bytes_per_vec,))

    def test_binary_profile_roundtrip_cosine_similarity(self):
        """Reconstructed vectors via binary must achieve cosine ≥ 0.75 (spec floor)."""
        d = 128
        n = 8
        np.random.seed(42)
        vecs = np.random.standard_normal((n, d)).astype(np.float32)
        result = _processor().quantize_batch(vecs, profile="binary")
        for i in range(n):
            reconstructed = result.reconstruct_vector(i)
            orig = vecs[i]
            cos = float(
                np.dot(orig, reconstructed)
                / (np.linalg.norm(orig) * np.linalg.norm(reconstructed) + 1e-9)
            )
            self.assertGreaterEqual(cos, 0.75, msg=f"Binary cosine below floor at vector {i}")


# ---------------------------------------------------------------------------
# BatchCompressionAnalyzer
# ---------------------------------------------------------------------------


class TestBatchCompressionAnalyzer(unittest.TestCase):
    def test_analyze_batch_result_keys_present(self):
        result = _result()
        analysis = BatchCompressionAnalyzer.analyze_batch_result(result)
        for key in (
            "compression_ratio",
            "space_savings_percent",
            "original_mb",
            "compressed_mb",
            "savings_mb",
            "vectors_processed",
            "vector_dimension",
        ):
            self.assertIn(key, analysis)

    def test_compare_profiles_returns_all_three_default_profiles(self):
        vecs = _RNG.standard_normal((10, 8)).astype(np.float32)
        comparison = BatchCompressionAnalyzer.compare_profiles(vecs)
        for profile in ("fast", "balanced", "quality"):
            self.assertIn(profile, comparison)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions(unittest.TestCase):
    def test_module_level_quantize_embeddings_batch(self):
        vecs = _RNG.standard_normal((8, 6)).astype(np.float32)
        result = quantize_embeddings_batch(vecs)
        self.assertIsInstance(result, BatchQuantizationResult)
        self.assertEqual(result.batch_size, 8)

    def test_module_level_benchmark_batch_compression_keys(self):
        vecs = _RNG.standard_normal((10, 8)).astype(np.float32)
        report = benchmark_batch_compression(vecs)
        for key in ("fast", "balanced", "quality"):
            self.assertIn(key, report)


if __name__ == "__main__":
    unittest.main()
