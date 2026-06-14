"""
Comprehensive test suite for Vectro Python API.
Tests all new Python bindings and functionality.
"""

import json
import os
import tempfile
import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python import (  # noqa: E402
    Vectro,
    compress_vectors,
    decompress_vectors,
    analyze_compression_quality,
    generate_compression_report,
    QuantizationResult,
    BatchQuantizationResult,
    VectroBatchProcessor,
    VectroQualityAnalyzer,
    QualityMetrics,
    ProfileManager,
    CompressionProfile,
    CompressionStrategy,
    CompressionOptimizer,
    create_custom_profile,
    get_version_info,
    get_backend_info,
)


class TestVectroCore(unittest.TestCase):
    """Test core Vectro functionality."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.single_vector = np.random.randn(384).astype(np.float32)
        self.batch_vectors = np.random.randn(100, 384).astype(np.float32)
        self.vectro = Vectro()

    def test_single_vector_compression(self):
        """Test compressing a single vector."""
        result = self.vectro.compress(self.single_vector)

        self.assertIsInstance(result, QuantizationResult)
        self.assertEqual(result.dims, 384)
        self.assertEqual(result.n, 1)
        self.assertIsInstance(result.quantized, np.ndarray)
        self.assertEqual(result.quantized.dtype, np.int8)
        self.assertEqual(len(result.quantized), 384)

    def test_batch_compression(self):
        """Test compressing a batch of vectors."""
        result = self.vectro.compress(self.batch_vectors)

        self.assertIsInstance(result, BatchQuantizationResult)
        self.assertEqual(result.batch_size, 100)
        self.assertEqual(result.vector_dim, 384)
        self.assertEqual(len(result.quantized_vectors), 100)
        self.assertEqual(len(result.scales), 100)
        self.assertGreater(result.compression_ratio, 2.0)

    def test_compression_decompression_roundtrip(self):
        """Test compression followed by decompression."""
        # Test single vector
        compressed = self.vectro.compress(self.single_vector)
        decompressed = self.vectro.decompress(compressed)

        # Single vector decompression returns 1D (dim,) shape
        self.assertEqual(decompressed.shape, (self.single_vector.shape[0],))

        # Should have reasonable reconstruction quality
        mae = np.mean(np.abs(self.single_vector - decompressed))
        self.assertLess(mae, 0.1)  # Mean error < 0.1

        # Test batch
        compressed_batch = self.vectro.compress(self.batch_vectors)
        decompressed_batch = self.vectro.decompress(compressed_batch)

        self.assertEqual(decompressed_batch.shape, self.batch_vectors.shape)

        mae_batch = np.mean(np.abs(self.batch_vectors - decompressed_batch))
        self.assertLess(mae_batch, 0.1)

    def test_quality_analysis(self):
        """Test quality analysis functionality."""
        compressed = self.vectro.compress(self.batch_vectors)
        quality = self.vectro.analyze_quality(self.batch_vectors, compressed)

        self.assertIsInstance(quality, QualityMetrics)
        self.assertGreater(quality.mean_cosine_similarity, 0.99)
        self.assertLess(quality.mean_absolute_error, 0.1)
        self.assertGreater(quality.compression_ratio, 2.0)
        # Check that error percentiles are available (individual fields, not nested)
        quality_dict = quality.to_dict()
        self.assertIn("error_p95", quality_dict)

    def test_compression_with_quality_metrics(self):
        """Test compression with quality metrics returned."""
        result, quality = self.vectro.compress(self.batch_vectors, return_quality_metrics=True)

        self.assertIsInstance(result, BatchQuantizationResult)
        self.assertIsInstance(quality, QualityMetrics)
        self.assertGreater(quality.mean_cosine_similarity, 0.99)

    def test_single_vector_compression_with_quality_metrics(self):
        """Single-vector compress(..., return_quality_metrics=True) must not raise.

        Regression: the single-vector INT8 branch previously referenced an
        undefined ``result`` name, raising ``NameError`` whenever quality
        metrics were requested for a 1-D input.
        """
        result, quality = self.vectro.compress(self.single_vector, return_quality_metrics=True)

        self.assertIsInstance(result, QuantizationResult)
        self.assertEqual(result.n, 1)
        self.assertIsInstance(quality, QualityMetrics)
        self.assertGreater(quality.mean_cosine_similarity, 0.99)

    def test_ultra_profile_precision_mode(self):
        """Ultra profile falls through to INT4 if squish_quant is available, INT8 otherwise."""
        backend_info = get_backend_info()
        result = self.vectro.compress(self.batch_vectors, profile="ultra")
        if backend_info.get("squish_quant_rust", False):
            self.assertEqual(result.precision_mode, "int4")
        else:
            self.assertEqual(result.precision_mode, "int8")


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing functionality."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_vectors = np.random.randn(500, 256).astype(np.float32)
        self.processor = VectroBatchProcessor()

    def test_batch_quantization(self):
        """Test batch quantization."""
        result = self.processor.quantize_batch(self.test_vectors)

        self.assertIsInstance(result, BatchQuantizationResult)
        self.assertEqual(result.batch_size, 500)
        self.assertEqual(result.vector_dim, 256)
        self.assertGreater(result.compression_ratio, 2.0)

        # Test individual vector retrieval
        quantized, scale = result.get_vector(0)
        self.assertEqual(len(quantized), 256)
        self.assertIsInstance(scale, float)

    def test_streaming_quantization(self):
        """Test streaming quantization for large datasets."""
        large_vectors = np.random.randn(2500, 128).astype(np.float32)

        results = self.processor.quantize_streaming(large_vectors, chunk_size=500, profile="fast")

        self.assertEqual(len(results), 5)  # 2500 / 500 = 5 chunks

        # Each result should be valid
        for result in results:
            self.assertIsInstance(result, BatchQuantizationResult)
            self.assertLessEqual(result.batch_size, 500)
            self.assertEqual(result.vector_dim, 128)

    def test_performance_benchmarking(self):
        """Test performance benchmarking."""
        results = self.processor.benchmark_batch_performance(
            batch_sizes=[100, 200],
            vector_dims=[128, 256],
            num_trials=1,  # Keep test fast
        )

        self.assertEqual(len(results), 4)  # 2 x 2 combinations

        for key, metrics in results.items():
            self.assertIn("avg_throughput_vec_per_sec", metrics)
            self.assertGreater(metrics["avg_throughput_vec_per_sec"], 0)


class TestQualityAnalysis(unittest.TestCase):
    """Test quality analysis functionality."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.original = np.random.randn(200, 128).astype(np.float32)
        # Create slightly modified version to simulate quantization
        noise = np.random.randn(*self.original.shape) * 0.01
        self.reconstructed = self.original + noise.astype(np.float32)
        self.analyzer = VectroQualityAnalyzer()

    def test_quality_evaluation(self):
        """Test comprehensive quality evaluation."""
        metrics = self.analyzer.evaluate_quality(
            self.original, self.reconstructed, compression_ratio=3.5
        )

        self.assertIsInstance(metrics, QualityMetrics)
        self.assertGreater(metrics.mean_cosine_similarity, 0.99)
        self.assertEqual(metrics.compression_ratio, 3.5)
        self.assertIn("error_p95", metrics.percentile_errors)
        self.assertIn("A", metrics.quality_grade())  # Should get good grade

    def test_error_percentiles(self):
        """Test error percentile calculations."""
        metrics = self.analyzer.evaluate_quality(self.original, self.reconstructed)

        percentiles = metrics.percentile_errors
        self.assertIn("error_p25", percentiles)
        self.assertIn("error_p50", percentiles)
        self.assertIn("error_p75", percentiles)
        self.assertIn("error_p95", percentiles)
        self.assertIn("error_p99", percentiles)

        # Percentiles should be ordered
        self.assertLessEqual(percentiles["error_p25"], percentiles["error_p50"])
        self.assertLessEqual(percentiles["error_p50"], percentiles["error_p75"])
        self.assertLessEqual(percentiles["error_p75"], percentiles["error_p95"])

    def test_quality_threshold_checking(self):
        """Test quality threshold validation."""
        metrics = self.analyzer.evaluate_quality(self.original, self.reconstructed)

        # Should pass reasonable threshold
        self.assertTrue(metrics.passes_quality_threshold(0.99))

        # Might not pass very strict threshold
        result = metrics.passes_quality_threshold(0.9999)
        self.assertIsInstance(result, (bool, np.bool_))  # Accept numpy boolean

    def test_quality_report_generation(self):
        """Test quality report generation."""
        metrics = self.analyzer.evaluate_quality(self.original, self.reconstructed)
        report = generate_compression_report(self.original, compress_vectors(self.original))

        self.assertIsInstance(report, str)
        self.assertIn("Compression Quality Report", report)  # Actual header text
        self.assertIn("Cosine Similarity", report)
        self.assertIn("Error Metrics", report)


class TestCompressionProfiles(unittest.TestCase):
    """Test compression profiles functionality."""

    def setUp(self):
        """Set up test data."""
        ProfileManager.initialize_builtin_profiles()
        np.random.seed(42)
        self.test_vectors = np.random.randn(100, 128).astype(np.float32)

    def test_builtin_profiles(self):
        """Test built-in compression profiles."""
        profiles = ProfileManager.list_profiles()

        # Should have standard profiles
        expected_profiles = ["fast", "balanced", "quality", "ultra", "binary"]
        for profile in expected_profiles:
            self.assertIn(profile, profiles)

    def test_profile_retrieval(self):
        """Test retrieving compression profiles."""
        balanced = ProfileManager.get_profile("balanced")

        self.assertIsInstance(balanced, CompressionProfile)
        self.assertEqual(balanced.name, "balanced")
        self.assertEqual(balanced.strategy, CompressionStrategy.BALANCED)
        self.assertEqual(balanced.quantization_bits, 8)

    def test_custom_profile_creation(self):
        """Test creating custom profiles."""
        custom = create_custom_profile(
            "test_profile", quantization_bits=6, range_factor=0.92, min_similarity_threshold=0.993
        )

        self.assertEqual(custom.name, "test_profile")
        self.assertEqual(custom.quantization_bits, 6)
        self.assertEqual(custom.range_factor, 0.92)
        self.assertEqual(custom.min_similarity_threshold, 0.993)

    def test_profile_serialization(self):
        """Test profile serialization/deserialization."""
        custom = create_custom_profile("serialization_test")

        # Convert to dict and back
        profile_dict = custom.to_dict()
        restored = CompressionProfile.from_dict(profile_dict)

        self.assertEqual(custom.name, restored.name)
        self.assertEqual(custom.quantization_bits, restored.quantization_bits)
        self.assertEqual(custom.range_factor, restored.range_factor)

    def test_profile_optimization(self):
        """Test automatic profile optimization."""
        optimizer = CompressionOptimizer()

        optimized = optimizer.auto_optimize_profile(
            self.test_vectors, target_similarity=0.995, target_compression=3.0
        )

        self.assertIsInstance(optimized, CompressionProfile)
        self.assertEqual(optimized.name, "auto_optimized")
        self.assertEqual(optimized.strategy, CompressionStrategy.CUSTOM)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_vectors = np.random.randn(50, 256).astype(np.float32)

    def test_compress_decompress_vectors(self):
        """Test convenience compress/decompress functions."""
        compressed = compress_vectors(self.test_vectors, profile="fast")
        decompressed = decompress_vectors(compressed)

        self.assertEqual(decompressed.shape, self.test_vectors.shape)

        # Check quality
        mae = np.mean(np.abs(self.test_vectors - decompressed))
        self.assertLess(mae, 0.1)

    def test_quality_analysis_convenience(self):
        """Test convenience quality analysis function."""
        compressed = compress_vectors(self.test_vectors)
        quality = analyze_compression_quality(self.test_vectors, compressed)

        self.assertIsInstance(quality, QualityMetrics)
        self.assertGreater(quality.mean_cosine_similarity, 0.99)

    def test_report_generation(self):
        """Test report generation convenience function."""
        compressed = compress_vectors(self.test_vectors)
        report = generate_compression_report(self.test_vectors, compressed)

        self.assertIsInstance(report, str)
        self.assertIn("Compression Quality Report", report)

    def test_version_info(self):
        """Test version information."""
        version_info = get_version_info()

        self.assertIn("version", version_info)
        self.assertIn("author", version_info)
        self.assertIn("license", version_info)
        self.assertEqual(version_info["license"], "MIT")


class TestFileIO(unittest.TestCase):
    """Test file I/O operations."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_vectors = np.random.randn(100, 256).astype(np.float32)
        self.vectro = Vectro()

    def test_save_load_compressed(self):
        """Test saving and loading compressed vectors."""
        compressed = self.vectro.compress(self.test_vectors)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name

        try:
            # Save compressed data
            self.vectro.save_compressed(compressed, temp_path)

            # Load compressed data
            loaded = self.vectro.load_compressed(temp_path)

            # Verify data integrity
            self.assertIsInstance(loaded, BatchQuantizationResult)
            self.assertEqual(loaded.batch_size, compressed.batch_size)
            self.assertEqual(loaded.vector_dim, compressed.vector_dim)
            self.assertEqual(len(loaded.quantized_vectors), len(compressed.quantized_vectors))

            # Test decompression of loaded data
            original_decompressed = self.vectro.decompress(compressed)
            loaded_decompressed = self.vectro.decompress(loaded)

            np.testing.assert_array_almost_equal(
                original_decompressed, loaded_decompressed, decimal=6
            )

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_writes_versioned_metadata(self):
        """Saved artifacts should include v2 storage metadata."""
        compressed = self.vectro.compress(self.test_vectors)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name

        try:
            self.vectro.save_compressed(compressed, temp_path, metadata={"experiment": "unit-test"})
            data = np.load(temp_path)

            self.assertIn("storage_format_version", data)
            self.assertIn("storage_format", data)
            self.assertIn("metadata_json", data)
            self.assertEqual(int(data["storage_format_version"]), 2)
            self.assertEqual(str(data["storage_format"]), "vectro_npz")

            metadata = json.loads(str(data["metadata_json"]))
            self.assertEqual(metadata["storage_format_version"], 2)
            self.assertEqual(metadata["storage_format"], "vectro_npz")
            self.assertEqual(metadata["user_metadata"]["experiment"], "unit-test")
            self.assertEqual(str(data["precision_mode"]), compressed.precision_mode)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_legacy_artifact_without_format_version(self):
        """Loading v1-style artifacts should remain supported."""
        compressed = self.vectro.compress(self.test_vectors)
        quantized_array = np.array(compressed.quantized_vectors)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name

        try:
            # Legacy layout: no storage_format_version key.
            np.savez_compressed(
                temp_path,
                quantized=quantized_array,
                scales=compressed.scales,
                batch_size=compressed.batch_size,
                vector_dim=compressed.vector_dim,
                compression_ratio=compressed.compression_ratio,
                total_original_bytes=compressed.total_original_bytes,
                total_compressed_bytes=compressed.total_compressed_bytes,
            )

            loaded = self.vectro.load_compressed(temp_path)
            self.assertIsInstance(loaded, BatchQuantizationResult)
            self.assertEqual(loaded.batch_size, compressed.batch_size)
            self.assertEqual(loaded.vector_dim, compressed.vector_dim)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_rejects_newer_unsupported_storage_version(self):
        """Loading newer format versions should fail with a clear error."""
        compressed = self.vectro.compress(self.test_vectors)
        quantized_array = np.array(compressed.quantized_vectors)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name

        try:
            np.savez_compressed(
                temp_path,
                quantized=quantized_array,
                scales=compressed.scales,
                batch_size=compressed.batch_size,
                vector_dim=compressed.vector_dim,
                compression_ratio=compressed.compression_ratio,
                total_original_bytes=compressed.total_original_bytes,
                total_compressed_bytes=compressed.total_compressed_bytes,
                storage_format_version=99,
                storage_format="vectro_npz",
            )

            with self.assertRaises(ValueError):
                self.vectro.load_compressed(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up test data."""
        self.vectro = Vectro()

    def test_invalid_input_shapes(self):
        """Test handling of invalid input shapes."""
        # 3D array should raise error
        with self.assertRaises(ValueError):
            invalid_vectors = np.random.randn(10, 10, 10)
            self.vectro.compress(invalid_vectors)

        # Empty array should raise error
        with self.assertRaises(ValueError):
            empty_vectors = np.array([])
            self.vectro.compress(empty_vectors)

    def test_profile_not_found(self):
        """Test handling of non-existent profiles."""
        with self.assertRaises(ValueError):
            ProfileManager.get_profile("nonexistent_profile")

    def test_invalid_profile_parameters(self):
        """Test validation of profile parameters."""
        with self.assertRaises(ValueError):
            # Invalid quantization bits
            CompressionProfile(
                name="invalid",
                strategy=CompressionStrategy.CUSTOM,
                quantization_bits=16,  # Invalid
                range_factor=0.95,
                clipping_percentile=99.0,
                adaptive_scaling=True,
                batch_optimization=True,
                precision_mode="int8",
                error_correction=False,
                simd_enabled=True,
                parallel_processing=True,
                memory_efficient=True,
                preserve_norms=True,
                preserve_angles=False,
                min_similarity_threshold=0.995,
            )

    def test_mismatched_shapes_quality_analysis(self):
        """Test quality analysis with mismatched shapes."""
        original = np.random.randn(100, 128).astype(np.float32)
        mismatched = np.random.randn(100, 256).astype(np.float32)  # Wrong dimension

        analyzer = VectroQualityAnalyzer()
        with self.assertRaises(ValueError):
            analyzer.evaluate_quality(original, mismatched)


def run_all_tests():
    """Run all test suites."""
    print("Running Vectro Python API Test Suite")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestVectroCore,
        TestBatchProcessing,
        TestQualityAnalysis,
        TestCompressionProfiles,
        TestConvenienceFunctions,
        TestFileIO,
        TestErrorHandling,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")

    return success


if __name__ == "__main__":
    run_all_tests()
