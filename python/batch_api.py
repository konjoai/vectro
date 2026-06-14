"""
Python API for Vectro batch processing operations.
Provides high-level interface to Mojo batch_processor module.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from .interface import dequantize_int4
from . import _mojo_bridge


@dataclass
class BatchQuantizationResult:
    """Result from batch quantization operation."""

    quantized_vectors: List[np.ndarray]  # List of quantized int8 vectors
    scales: np.ndarray  # Float32 scale factors
    batch_size: int  # Number of vectors processed
    vector_dim: int  # Dimension of each vector
    compression_ratio: float  # Achieved compression ratio
    total_original_bytes: int  # Original size in bytes
    total_compressed_bytes: int  # Compressed size in bytes
    precision_mode: str = "int8"  # int8 | int4
    group_size: int = 0

    def get_vector(self, index: int) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """Get a specific quantized vector and its scale."""
        if index >= len(self.quantized_vectors):
            raise IndexError(f"Vector index {index} out of range")
        scale_value = self.scales[index]
        if np.isscalar(scale_value):
            return self.quantized_vectors[index], float(scale_value)
        return self.quantized_vectors[index], np.asarray(scale_value, dtype=np.float32)

    def reconstruct_vector(self, index: int) -> np.ndarray:
        """Reconstruct a specific vector from quantized form."""
        if index >= len(self.quantized_vectors):
            raise IndexError(f"Vector index {index} out of range")
        # Binary mode carries no per-vector scale — decode directly
        if self.precision_mode == "binary":
            from .binary_api import dequantize_binary

            return dequantize_binary(self.quantized_vectors[index].reshape(1, -1), self.vector_dim)[
                0
            ]
        quantized, scale = self.get_vector(index)
        if self.precision_mode == "int4":
            return dequantize_int4(
                np.asarray([quantized], dtype=np.uint8),
                np.asarray([self.scales[index]], dtype=np.float32),
                group_size=self.group_size or 64,
            )[0]
        return quantized.astype(np.float32) * scale

    def reconstruct_batch(self) -> np.ndarray:
        """Reconstruct all vectors in the batch."""
        if self.precision_mode == "int4":
            return dequantize_int4(
                np.asarray(self.quantized_vectors, dtype=np.uint8),
                np.asarray(self.scales, dtype=np.float32),
                group_size=self.group_size or 64,
            )

        if self.precision_mode == "binary":
            from .binary_api import dequantize_binary

            packed = np.stack(self.quantized_vectors)
            return dequantize_binary(packed, self.vector_dim)

        reconstructed = np.zeros((self.batch_size, self.vector_dim), dtype=np.float32)
        for i in range(self.batch_size):
            reconstructed[i] = self.reconstruct_vector(i)
        return reconstructed


class VectroBatchProcessor:
    """High-performance batch processing for vector quantization."""

    def __init__(self, backend: str = "auto"):
        """Initialize batch processor.

        Args:
            backend: Processing backend ("auto", "mojo", "python")
        """
        self.backend = backend
        self._mojo_binary = self._find_mojo_binary()

    def _find_mojo_binary(self) -> Optional[str]:
        """Find the Mojo binary for batch processing."""
        possible_paths = [
            Path(__file__).parent.parent / "vectro_quantizer",
            Path(__file__).parent.parent / "vectro-macos-arm64",
            Path("vectro_quantizer"),
        ]

        for path in possible_paths:
            if path.exists() and path.is_file():
                return str(path.absolute())
        return None

    def quantize_batch(
        self, vectors: Union[np.ndarray, List[np.ndarray]], profile: str = "balanced"
    ) -> BatchQuantizationResult:
        """Quantize a batch of vectors efficiently.

        Args:
            vectors: Array of shape (n, d) or list of vectors to quantize
            profile: Compression profile ("fast", "balanced", "quality", "binary").
                     "binary" encodes sign bits (1 bit/dim, ~32x), all others INT8 (~4x).

        Returns:
            BatchQuantizationResult with quantized data and metadata
        """
        # Convert input to standard format
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        elif not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be numpy array or list")

        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D array of shape (n, d)")

        vectors = vectors.astype(np.float32)
        batch_size, vector_dim = vectors.shape

        # Choose backend — binary profile bypasses Mojo (INT8-only) path
        if self.backend == "mojo" and self._mojo_binary and profile != "binary":
            return self._quantize_batch_mojo(vectors, profile)
        else:
            return self._quantize_batch_python(vectors, profile)

    def _quantize_batch_mojo(self, vectors: np.ndarray, profile: str) -> BatchQuantizationResult:
        """Quantize batch using the compiled Mojo binary (INT8 symmetric abs-max)."""
        batch_size, vector_dim = vectors.shape

        q, scales = _mojo_bridge.int8_quantize(vectors)

        original_bytes = batch_size * vector_dim * 4
        compressed_bytes = batch_size * vector_dim * 1 + batch_size * 4
        compression_ratio = original_bytes / compressed_bytes

        return BatchQuantizationResult(
            quantized_vectors=list(q),
            scales=scales,
            batch_size=batch_size,
            vector_dim=vector_dim,
            compression_ratio=compression_ratio,
            total_original_bytes=original_bytes,
            total_compressed_bytes=compressed_bytes,
        )

    def _quantize_batch_python(self, vectors: np.ndarray, profile: str) -> BatchQuantizationResult:
        """Quantize batch using Python implementation."""
        batch_size, vector_dim = vectors.shape

        # Binary profile: 1-bit sign encoding, ~32x compression
        # sign(v_i) packed 8 bits per byte; no per-vector scale needed.
        if profile == "binary":
            from .binary_api import quantize_binary

            packed = quantize_binary(vectors, normalize=True)  # (n, ceil(d/8)), uint8
            bytes_per_vec = (vector_dim + 7) // 8
            original_bytes = batch_size * vector_dim * 4  # float32
            compressed_bytes = batch_size * bytes_per_vec  # packed bits
            compression_ratio = original_bytes / compressed_bytes  # ≈ 32x
            return BatchQuantizationResult(
                quantized_vectors=[packed[i] for i in range(batch_size)],
                scales=np.array([], dtype=np.float32),
                batch_size=batch_size,
                vector_dim=vector_dim,
                compression_ratio=compression_ratio,
                total_original_bytes=original_bytes,
                total_compressed_bytes=compressed_bytes,
                precision_mode="binary",
            )

        # INT8 profiles
        int8_profiles = {
            "fast": {"range_factor": 1.0},
            "balanced": {"range_factor": 0.95},
            "quality": {"range_factor": 0.90},
        }

        if profile not in int8_profiles:
            profile = "balanced"

        range_factor = int8_profiles[profile]["range_factor"]

        max_abs = np.max(np.abs(vectors), axis=1)  # (n,)
        scales = np.where(max_abs == 0, 1.0, max_abs / (127.0 * range_factor)).astype(np.float32)
        quantized_matrix = np.clip(np.round(vectors / scales[:, np.newaxis]), -127, 127).astype(
            np.int8
        )
        quantized_vectors = list(quantized_matrix)

        # Calculate compression metrics
        original_bytes = batch_size * vector_dim * 4  # float32
        compressed_bytes = batch_size * vector_dim * 1 + batch_size * 4  # int8 + scales
        compression_ratio = original_bytes / compressed_bytes

        return BatchQuantizationResult(
            quantized_vectors=quantized_vectors,
            scales=scales,
            batch_size=batch_size,
            vector_dim=vector_dim,
            compression_ratio=compression_ratio,
            total_original_bytes=original_bytes,
            total_compressed_bytes=compressed_bytes,
        )

    def quantize_streaming(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        chunk_size: int = 1000,
        profile: str = "balanced",
    ) -> List[BatchQuantizationResult]:
        """Quantize vectors in streaming fashion for large datasets.

        Args:
            vectors: Large array of vectors
            chunk_size: Size of each processing chunk
            profile: Compression profile

        Returns:
            List of BatchQuantizationResult for each chunk
        """
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D array")

        num_vectors = vectors.shape[0]
        results = []

        # Process in chunks
        for start_idx in range(0, num_vectors, chunk_size):
            end_idx = min(start_idx + chunk_size, num_vectors)
            chunk = vectors[start_idx:end_idx]

            result = self.quantize_batch(chunk, profile)
            results.append(result)

        return results

    def benchmark_batch_performance(
        self, batch_sizes: List[int] = None, vector_dims: List[int] = None, num_trials: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark batch processing performance.

        Args:
            batch_sizes: List of batch sizes to test
            vector_dims: List of vector dimensions to test
            num_trials: Number of trials per configuration

        Returns:
            Performance metrics dictionary
        """
        if batch_sizes is None:
            batch_sizes = [100, 500, 1000, 2000]
        if vector_dims is None:
            vector_dims = [128, 384, 768, 1536]

        results = {}

        for batch_size in batch_sizes:
            for vector_dim in vector_dims:
                key = f"{batch_size}x{vector_dim}"

                times = []
                throughputs = []

                for trial in range(num_trials):
                    # Generate test data
                    vectors = np.random.randn(batch_size, vector_dim).astype(np.float32)

                    # Measure performance
                    import time

                    start_time = time.perf_counter()

                    self.quantize_batch(vectors, "balanced")

                    end_time = time.perf_counter()
                    duration = end_time - start_time

                    throughput = batch_size / duration  # vectors/sec

                    times.append(duration)
                    throughputs.append(throughput)

                # Calculate statistics
                avg_time = np.mean(times)
                avg_throughput = np.mean(throughputs)
                std_throughput = np.std(throughputs)

                results[key] = {
                    "batch_size": batch_size,
                    "vector_dim": vector_dim,
                    "avg_time_sec": avg_time,
                    "avg_throughput_vec_per_sec": avg_throughput,
                    "std_throughput": std_throughput,
                    "trials": num_trials,
                }

        return results


class BatchCompressionAnalyzer:
    """Analyze compression performance for batches."""

    @staticmethod
    def analyze_batch_result(result: BatchQuantizationResult) -> Dict[str, float]:
        """Analyze compression metrics for a batch result."""
        return {
            "compression_ratio": result.compression_ratio,
            "space_savings_percent": (1.0 - 1.0 / result.compression_ratio) * 100,
            "original_mb": result.total_original_bytes / (1024 * 1024),
            "compressed_mb": result.total_compressed_bytes / (1024 * 1024),
            "savings_mb": (result.total_original_bytes - result.total_compressed_bytes)
            / (1024 * 1024),
            "vectors_processed": result.batch_size,
            "vector_dimension": result.vector_dim,
        }

    @staticmethod
    def compare_profiles(
        vectors: np.ndarray, profiles: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare compression performance across different profiles."""
        if profiles is None:
            profiles = ["fast", "balanced", "quality"]

        processor = VectroBatchProcessor()
        results = {}

        for profile in profiles:
            batch_result = processor.quantize_batch(vectors, profile)
            analysis = BatchCompressionAnalyzer.analyze_batch_result(batch_result)

            # Calculate quality metrics
            original = vectors
            reconstructed = batch_result.reconstruct_batch()

            # Cosine similarity
            cos_sim = np.mean(
                [
                    np.dot(original[i], reconstructed[i])
                    / (np.linalg.norm(original[i]) * np.linalg.norm(reconstructed[i]))
                    for i in range(len(original))
                ]
            )

            # Mean absolute error
            mae = np.mean(np.abs(original - reconstructed))

            analysis.update(
                {"cosine_similarity": cos_sim, "mean_absolute_error": mae, "profile": profile}
            )

            results[profile] = analysis

        return results


# Convenience functions for common use cases
def quantize_embeddings_batch(
    embeddings: np.ndarray, profile: str = "balanced"
) -> BatchQuantizationResult:
    """Convenience function to quantize a batch of embeddings."""
    processor = VectroBatchProcessor()
    return processor.quantize_batch(embeddings, profile)


def benchmark_batch_compression(
    embeddings: np.ndarray, profiles: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """Convenience function to benchmark different compression profiles."""
    return BatchCompressionAnalyzer.compare_profiles(embeddings, profiles)


# Example usage and demo
if __name__ == "__main__":
    print("Vectro Batch Processing API Demo")
    print("=" * 40)

    # Create sample embeddings
    np.random.seed(42)
    embeddings = np.random.randn(1000, 384).astype(np.float32)

    print(f"Sample embeddings: {embeddings.shape}")
    print(f"Original size: {embeddings.nbytes / 1024:.1f} KB")

    # Test batch processing
    processor = VectroBatchProcessor()
    result = processor.quantize_batch(embeddings, "balanced")

    print("\nBatch Processing Results:")
    print(f"  Compression ratio: {result.compression_ratio:.2f}x")
    print(f"  Space savings: {(1 - 1 / result.compression_ratio) * 100:.1f}%")
    print(f"  Compressed size: {result.total_compressed_bytes / 1024:.1f} KB")

    # Test reconstruction quality
    reconstructed = result.reconstruct_batch()
    mae = np.mean(np.abs(embeddings - reconstructed))
    print(f"  Reconstruction MAE: {mae:.6f}")

    # Profile comparison
    print("\nProfile Comparison:")
    comparison = benchmark_batch_compression(embeddings[:100])  # Small sample for demo

    for profile, metrics in comparison.items():
        print(
            f"  {profile:>8}: {metrics['compression_ratio']:.2f}x compression, "
            f"MAE: {metrics['mean_absolute_error']:.6f}"
        )
