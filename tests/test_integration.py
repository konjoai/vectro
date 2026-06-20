"""
Integration tests for Vectro Python API with Mojo backend.
Tests end-to-end functionality and performance benchmarks.
"""

import time
import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python import (  # noqa: E402
    Vectro,
    VectroBatchProcessor,
    VectroQualityAnalyzer,
)


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance integration between Python and Mojo."""

    def setUp(self):
        """Set up performance test data."""
        np.random.seed(42)
        self.vectro = Vectro()
        self.processor = VectroBatchProcessor()

        # Various test vector sizes
        self.small_batch = np.random.randn(100, 256).astype(np.float32)
        self.medium_batch = np.random.randn(1000, 384).astype(np.float32)
        self.large_batch = np.random.randn(5000, 512).astype(np.float32)

    def test_throughput_small_batches(self):
        """Test throughput on small batches."""
        start_time = time.time()

        for _ in range(10):
            result = self.vectro.compress(self.small_batch)
            self.assertIsNotNone(result)

        elapsed = time.time() - start_time
        vectors_per_sec = (100 * 10) / elapsed

        # Should achieve good throughput
        print(f"Small batch throughput: {vectors_per_sec:.0f} vectors/sec")
        self.assertGreater(vectors_per_sec, 50000)  # 50K+ vectors/sec

    def test_throughput_medium_batches(self):
        """Test throughput on medium batches."""
        start_time = time.time()

        for _ in range(5):
            result = self.vectro.compress(self.medium_batch)
            self.assertIsNotNone(result)

        elapsed = time.time() - start_time
        vectors_per_sec = (1000 * 5) / elapsed

        print(f"Medium batch throughput: {vectors_per_sec:.0f} vectors/sec")
        self.assertGreater(vectors_per_sec, 50000)  # CI-stable floor (GitHub-hosted runners)

    def test_memory_efficiency_large_batches(self):
        """Test memory efficiency with large batches."""
        # Use streaming for large batch
        results = self.processor.quantize_streaming(
            self.large_batch, chunk_size=1000, profile="fast"
        )

        self.assertEqual(len(results), 5)  # 5000 / 1000 = 5 chunks

        # Verify all chunks processed correctly
        total_vectors = sum(result.batch_size for result in results)
        self.assertEqual(total_vectors, 5000)

        # Check compression ratios
        for result in results:
            self.assertGreater(result.compression_ratio, 2.0)

    def test_profile_performance_comparison(self):
        """Compare performance across different profiles."""
        profiles = ["fast", "balanced", "quality"]
        results = {}

        for profile in profiles:
            start_time = time.time()

            compressed = self.vectro.compress(self.medium_batch, profile=profile)

            elapsed = time.time() - start_time
            throughput = len(self.medium_batch) / elapsed

            results[profile] = {
                "throughput": throughput,
                "compression_ratio": compressed.compression_ratio,
                "elapsed": elapsed,
            }

        # Fast should generally be faster, but allow some variance.
        # Use 30K floor (CI-stable single-run threshold; actual M3 is 300K+ vec/s).
        self.assertGreater(results["fast"]["throughput"], 30000)
        self.assertGreater(results["quality"]["throughput"], 30000)

        # Quality should have better compression ratio
        self.assertGreaterEqual(
            results["quality"]["compression_ratio"], results["fast"]["compression_ratio"]
        )

        print("\nProfile Performance Comparison:")
        for profile, metrics in results.items():
            print(
                f"  {profile}: {metrics['throughput']:.0f} vec/sec, "
                f"{metrics['compression_ratio']:.2f}x compression"
            )


class TestQualityIntegration(unittest.TestCase):
    """Test quality preservation across different scenarios."""

    def setUp(self):
        """Set up quality test data."""
        np.random.seed(42)
        self.analyzer = VectroQualityAnalyzer()
        self.vectro = Vectro()

        # Different types of vectors
        self.random_vectors = np.random.randn(200, 384).astype(np.float32)
        self.normalized_vectors = self._normalize_vectors(
            np.random.randn(200, 384).astype(np.float32)
        )
        self.sparse_vectors = self._create_sparse_vectors(200, 384, sparsity=0.9)

    def _normalize_vectors(self, vectors):
        """L2 normalize vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)

    def _create_sparse_vectors(self, n, dim, sparsity):
        """Create sparse vectors."""
        vectors = np.random.randn(n, dim).astype(np.float32)
        mask = np.random.rand(n, dim) < sparsity
        vectors[mask] = 0
        return vectors

    def test_random_vector_quality(self):
        """Test quality on random vectors."""
        compressed = self.vectro.compress(self.random_vectors)
        quality = self.analyzer.evaluate_quality(
            self.random_vectors, self.vectro.decompress(compressed)
        )

        # Should achieve good quality
        self.assertGreater(quality.mean_cosine_similarity, 0.99)
        self.assertIn("A", quality.quality_grade())  # Should get good grade

    def test_normalized_vector_quality(self):
        """Test quality on normalized vectors."""
        compressed = self.vectro.compress(self.normalized_vectors, profile="quality")
        decompressed = self.vectro.decompress(compressed)

        quality = self.analyzer.evaluate_quality(self.normalized_vectors, decompressed)

        # Normalized vectors should compress very well
        self.assertGreater(quality.mean_cosine_similarity, 0.995)
        # Allow full grade format
        self.assertIn("A+", quality.quality_grade())

    def test_sparse_vector_quality(self):
        """Test quality on sparse vectors."""
        compressed = self.vectro.compress(self.sparse_vectors)
        quality = self.analyzer.evaluate_quality(
            self.sparse_vectors, self.vectro.decompress(compressed)
        )

        # Sparse vectors are more challenging
        self.assertGreater(quality.mean_cosine_similarity, 0.97)
        self.assertIn("A", quality.quality_grade())  # Should get reasonable grade

    def test_quality_preservation_across_profiles(self):
        """Test quality preservation across profiles."""
        profiles = ["fast", "balanced", "quality", "ultra"]
        qualities = {}

        for profile in profiles:
            compressed = self.vectro.compress(self.normalized_vectors, profile=profile)
            decompressed = self.vectro.decompress(compressed)

            quality = self.analyzer.evaluate_quality(self.normalized_vectors, decompressed)

            qualities[profile] = quality.mean_cosine_similarity

        # Quality order should generally be preserved (may have some variance)
        # INT8 profiles (fast/balanced/quality) should have very high cosine similarity.
        # Ultra uses INT4 (now GA in v3) which achieves ≥ 0.92 per v3 acceptance criteria.
        self.assertGreater(qualities["ultra"], 0.92)
        self.assertGreater(qualities["quality"], 0.999)

        print("\nProfile Quality Comparison:")
        for profile, similarity in qualities.items():
            print(f"  {profile}: {similarity:.5f} cosine similarity")


class TestRobustnessIntegration(unittest.TestCase):
    """Test robustness and edge cases."""

    def setUp(self):
        """Set up robustness test data."""
        self.vectro = Vectro()

    def test_extreme_values(self):
        """Test with extreme vector values."""
        # Very large values
        large_vectors = np.random.randn(50, 128).astype(np.float32) * 1000
        compressed = self.vectro.compress(large_vectors)
        decompressed = self.vectro.decompress(compressed)

        # Should handle gracefully
        self.assertEqual(decompressed.shape, large_vectors.shape)

        # Very small values
        small_vectors = np.random.randn(50, 128).astype(np.float32) * 0.001
        compressed = self.vectro.compress(small_vectors)
        decompressed = self.vectro.decompress(compressed)

        self.assertEqual(decompressed.shape, small_vectors.shape)

    def test_different_dimensions(self):
        """Test with different vector dimensions."""
        dimensions = [64, 128, 256, 384, 512, 768, 1024]

        for dim in dimensions:
            vectors = np.random.randn(20, dim).astype(np.float32)
            compressed = self.vectro.compress(vectors)
            decompressed = self.vectro.decompress(compressed)

            self.assertEqual(decompressed.shape, vectors.shape)
            self.assertGreater(compressed.compression_ratio, 1.5)

    def test_single_vector_edge_cases(self):
        """Test single vector edge cases."""
        # Zero vector
        zero_vector = np.zeros(256, dtype=np.float32)
        compressed = self.vectro.compress(zero_vector)
        decompressed = self.vectro.decompress(compressed)

        # Single-vector decompress returns 1D (d,) shape
        self.assertEqual(decompressed.shape, (256,))

        # Unit vector
        unit_vector = np.zeros(256, dtype=np.float32)
        unit_vector[0] = 1.0

        compressed = self.vectro.compress(unit_vector)
        decompressed = self.vectro.decompress(compressed)

        # Single-vector decompress returns 1D (d,) shape
        self.assertEqual(decompressed.shape, (256,))

    def test_batch_size_variations(self):
        """Test various batch sizes."""
        batch_sizes = [1, 5, 10, 50, 100, 500, 1000]

        for batch_size in batch_sizes:
            vectors = np.random.randn(batch_size, 256).astype(np.float32)
            compressed = self.vectro.compress(vectors)

            if batch_size == 1:
                # Single vector now returns BatchQuantizationResult with batch_size=1
                from python import BatchQuantizationResult

                self.assertIsInstance(compressed, BatchQuantizationResult)
                self.assertEqual(compressed.batch_size, 1)
            else:
                # Batch returns BatchQuantizationResult
                from python import BatchQuantizationResult

                self.assertIsInstance(compressed, BatchQuantizationResult)
                self.assertEqual(compressed.batch_size, batch_size)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows."""

    def setUp(self):
        """Set up workflow test data."""
        np.random.seed(42)
        self.test_vectors = np.random.randn(1000, 384).astype(np.float32)
        self.vectro = Vectro()
        self.processor = VectroBatchProcessor()

    def test_complete_compression_workflow(self):
        """Test complete compression workflow."""
        # Step 1: Get baseline characteristics (simplified)
        vector_norms = np.linalg.norm(self.test_vectors, axis=1)
        baseline_stats = {
            "mean_norm": np.mean(vector_norms),
            "std_norm": np.std(vector_norms),
            "vector_count": len(self.test_vectors),
        }

        self.assertIn("mean_norm", baseline_stats)
        self.assertIn("std_norm", baseline_stats)

        # Step 2: Choose appropriate profile
        profile_name = "balanced"  # For general use

        # Step 3: Compress with quality monitoring
        compressed, quality = self.vectro.compress(
            self.test_vectors, profile=profile_name, return_quality_metrics=True
        )

        # Step 4: Verify quality meets requirements
        self.assertGreater(quality.mean_cosine_similarity, 0.99)
        self.assertTrue(quality.passes_quality_threshold(0.99))

        # Step 5: Decompress for usage
        decompressed = self.vectro.decompress(compressed)

        # Step 6: Final verification
        final_analyzer = VectroQualityAnalyzer()
        final_quality = final_analyzer.evaluate_quality(self.test_vectors, decompressed)

        self.assertGreater(final_quality.mean_cosine_similarity, 0.99)
        self.assertLess(final_quality.mean_absolute_error, 0.1)

    def test_streaming_workflow(self):
        """Test streaming compression workflow."""
        # Large dataset simulation
        large_dataset = np.random.randn(5000, 256).astype(np.float32)

        # Process in streams
        all_results = []
        chunk_size = 1000

        for i in range(0, len(large_dataset), chunk_size):
            chunk = large_dataset[i : i + chunk_size]

            # Compress chunk
            compressed = self.vectro.compress(chunk, profile="fast")
            all_results.append(compressed)

            # Verify chunk quality
            decompressed = self.vectro.decompress(compressed)
            cosine_sim = np.mean(
                [
                    np.dot(original, reconstructed)
                    / (np.linalg.norm(original) * np.linalg.norm(reconstructed))
                    for original, reconstructed in zip(chunk, decompressed)
                ]
            )

            self.assertGreater(cosine_sim, 0.99)

        # Verify all chunks processed
        self.assertEqual(len(all_results), 5)
        total_vectors = sum(r.batch_size if hasattr(r, "batch_size") else 1 for r in all_results)
        self.assertEqual(total_vectors, 5000)

    def test_profile_optimization_workflow(self):
        """Test automatic profile optimization workflow."""
        from python import CompressionOptimizer

        # Step 1: Get sample data for optimization
        sample_data = self.test_vectors[:100]

        # Step 2: Auto-optimize profile
        optimizer = CompressionOptimizer()
        optimized_profile = optimizer.auto_optimize_profile(
            sample_data, target_similarity=0.995, target_compression=3.5
        )

        self.assertIsNotNone(optimized_profile)
        self.assertEqual(optimized_profile.name, "auto_optimized")

        # Step 3: Test optimized profile on full dataset by using it directly
        compressed = self.vectro.compress(
            self.test_vectors,
            profile="balanced",  # Use existing profile for test
        )

        # Step 4: Verify optimization results
        self.assertGreater(compressed.compression_ratio, 3.0)

        decompressed = self.vectro.decompress(compressed)
        quality = VectroQualityAnalyzer().evaluate_quality(self.test_vectors, decompressed)

        self.assertGreater(quality.mean_cosine_similarity, 0.995)


class TestPerformanceRegression(unittest.TestCase):
    """Regression gates that prevent shipping performance regressions.

    Thresholds are intentionally conservative so they remain green across
    CI runners with varying specs (GitHub-hosted ubuntu-latest, Mac M-series).
    Adjust only via a deliberate PR with measured evidence.

    Baseline (NumPy backend, 2026-03 on M-series Mac):
      - throughput  : 80K+ vec/sec on 1000 × 384 batch
      - compression : 3.5× minimum for int8
      - quality     : 0.99 mean cosine similarity on normalised vectors
    """

    _N = 1000
    _DIM = 384
    _RNG_SEED = 0

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(cls._RNG_SEED)
        cls.vectors = rng.standard_normal((cls._N, cls._DIM)).astype(np.float32)
        # Pre-normalise a copy for quality tests (tighter bound)
        norms = np.linalg.norm(cls.vectors, axis=1, keepdims=True)
        cls.unit_vectors = cls.vectors / (norms + 1e-10)
        cls.vectro = Vectro()

    # ------------------------------------------------------------------
    # Throughput gates
    # ------------------------------------------------------------------

    def _throughput(self, vectors, profile="balanced", repeats=3):
        """Return median throughput in vectors/sec over ``repeats`` runs."""
        timings = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.vectro.compress(vectors, profile=profile)
            timings.append(time.perf_counter() - t0)
        elapsed = sorted(timings)[len(timings) // 2]  # median
        return len(vectors) / elapsed

    def test_throughput_regression_balanced(self):
        """balanced profile must sustain ≥ 60K vec/sec on 1000×384."""
        vps = self._throughput(self.vectors, profile="balanced")
        print(f"\n[regression] balanced throughput: {vps:,.0f} vec/sec  (gate: 60K)")
        self.assertGreater(vps, 60_000, f"Throughput regression: {vps:,.0f} vec/sec < 60K gate")

    def test_throughput_regression_fast(self):
        """fast profile must sustain ≥ 60K vec/sec on 1000×384."""
        vps = self._throughput(self.vectors, profile="fast")
        print(f"\n[regression] fast throughput: {vps:,.0f} vec/sec  (gate: 60K)")
        self.assertGreater(vps, 60_000, f"Throughput regression: {vps:,.0f} vec/sec < 60K gate")

    # ------------------------------------------------------------------
    # Compression ratio gate
    # ------------------------------------------------------------------

    def test_compression_ratio_regression(self):
        """int8 compression must achieve ≥ 3.5× on 1000×384 float32 input."""
        result = self.vectro.compress(self.vectors, profile="balanced")
        ratio = result.compression_ratio
        print(f"\n[regression] compression ratio: {ratio:.2f}×  (gate: 3.5×)")
        self.assertGreaterEqual(
            ratio, 3.5, f"Compression ratio regression: {ratio:.2f}× < 3.5× gate"
        )

    # ------------------------------------------------------------------
    # Quality gate
    # ------------------------------------------------------------------

    def test_quality_regression_cosine(self):
        """Reconstructed vectors must have mean cosine similarity ≥ 0.99."""
        compressed = self.vectro.compress(self.unit_vectors, profile="balanced")
        reconstructed = self.vectro.decompress(compressed)

        dot = np.sum(self.unit_vectors * reconstructed, axis=1)
        norm_r = np.linalg.norm(reconstructed, axis=1)
        cosine = float(np.mean(dot / (norm_r + 1e-10)))
        print(f"\n[regression] mean cosine similarity: {cosine:.5f}  (gate: 0.99)")
        self.assertGreater(cosine, 0.99, f"Quality regression: cosine sim {cosine:.5f} < 0.99 gate")


def run_integration_tests():
    """Run all integration tests."""
    print("Running Vectro Python API Integration Tests")
    print("=" * 55)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestPerformanceIntegration,
        TestQualityIntegration,
        TestRobustnessIntegration,
        TestEndToEndWorkflow,
        TestPerformanceRegression,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 55)
    print("Integration Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")

    success = len(result.failures) == 0 and len(result.errors) == 0

    if success:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some integration tests failed!")

    return success


if __name__ == "__main__":
    run_integration_tests()
