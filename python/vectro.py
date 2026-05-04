"""
Vectro - Ultra-High-Performance LLM Embedding Compressor
Unified Python API providing access to all Mojo-powered compression capabilities.

This module provides a clean, Pythonic interface to Vectro's high-performance
vector quantization capabilities implemented in Mojo.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings
import json
from datetime import datetime, timezone

# Import Vectro modules
from .interface import (
    QuantizationResult, 
    quantize_embeddings, 
    reconstruct_embeddings,
    mean_cosine_similarity,
    get_backend_info
)
from .batch_api import (
    VectroBatchProcessor,
    BatchQuantizationResult,
    BatchCompressionAnalyzer,
    quantize_embeddings_batch,
    benchmark_batch_compression
)
from .quality_api import (
    VectroQualityAnalyzer,
    QualityMetrics,
    QualityBenchmark,
    QualityReport,
    evaluate_quantization_quality,
    generate_quality_report
)
from .profiles_api import (
    ProfileManager,
    CompressionProfile,
    CompressionStrategy,
    CompressionOptimizer,
    ProfileComparison,
    get_compression_profile,
    create_custom_profile
)

__version__ = "5.0.1"
__author__ = "Wesley Scholl"
__license__ = "MIT"
__description__ = "Ultra-High-Performance LLM Embedding Compressor"

_STORAGE_FORMAT_VERSION = 2
_STORAGE_FORMAT_NAME = "vectro_npz"


class Vectro:
    """
    Main Vectro API class providing unified access to all compression capabilities.
    
    This class serves as the primary interface for vector quantization operations,
    combining high-performance Mojo backends with convenient Python APIs.
    """
    
    def __init__(
        self,
        backend: str = "auto",
        profile: str = "balanced",
        enable_batch_optimization: bool = True,
    ):
        """Initialize Vectro compressor.

        Args:
            backend: Processing backend ("auto", "mojo", "cython", "python")
            profile: Default compression profile ("fast", "balanced", "quality",
                     "ultra", "binary")
            enable_batch_optimization: Enable batch processing optimizations
        """
        self.backend = backend
        self.default_profile = profile
        self.enable_batch_optimization = enable_batch_optimization
        
        # Initialize components
        self.batch_processor = VectroBatchProcessor(backend=backend)
        self.quality_analyzer = VectroQualityAnalyzer()
        
        # Initialize profiles
        ProfileManager.initialize_builtin_profiles()
        
        # Cache for performance metrics
        self._performance_cache = {}
        
    def compress(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        profile: Optional[str] = None,
        precision_mode: Optional[str] = None,
        return_quality_metrics: bool = False,
        model_dir: Optional[str] = None,
    ) -> Union[QuantizationResult, BatchQuantizationResult, Tuple[Any, QualityMetrics]]:
        """Compress vectors using quantization.

        Args:
            vectors: Input vectors to compress (single vector or batch)
            profile: Compression profile to use (None = use default)
            precision_mode: Override quantization method (int8, nf4, binary, int4)
            return_quality_metrics: Return quality analysis alongside results
            model_dir: Path to a HuggingFace model directory.  When supplied,
                the model-family registry selects the optimal precision_mode
                for that embedding family (e.g. GTE → INT8, BGE → NF4).
                Ignored when precision_mode is explicitly set.

        Returns:
            Quantization result(s) and optionally quality metrics
        """
        if profile is None:
            profile = self.default_profile

        selected_profile = get_compression_profile(profile)
        requested_precision = precision_mode or selected_profile.precision_mode

        # Model-family registry: override precision when model_dir is given and
        # precision_mode was not explicitly specified by the caller.
        if model_dir is not None and precision_mode is None:
            from .profiles import get_profile as _get_profile
            _fp = _get_profile(model_dir)
            if _fp.method != "auto":
                requested_precision = _fp.method

        requested_precision = requested_precision.lower()
        group_size = 64

        # INT4 currently depends on squish_quant; fallback keeps API usable.
        if requested_precision == "int4" and not get_backend_info().get("squish_quant_rust", False):
            warnings.warn(
                "INT4 requested but squish_quant backend is unavailable. Falling back to INT8.",
                RuntimeWarning,
            )
            requested_precision = "int8"
            
        # Convert input to numpy array
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        elif not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be numpy array or list")

        if vectors.size == 0:
            raise ValueError("vectors must be non-empty")
            
        vectors = vectors.astype(np.float32)
        
        # Determine if single vector or batch
        if vectors.ndim == 1:
            # Single vector
            vectors = vectors.reshape(1, -1)
            if requested_precision == "binary":
                from .binary_api import quantize_binary
                packed = quantize_binary(vectors)   # shape (1, ceil(d/8)), uint8
                d = vectors.shape[1]
                single_result = QuantizationResult(
                    quantized=packed[0],
                    scales=np.ones(1, dtype=np.float32),
                    dims=d,
                    n=1,
                    precision_mode="binary",
                    group_size=0,
                )
                if return_quality_metrics:
                    reconstructed = self.decompress(single_result)
                    quality = self.quality_analyzer.evaluate_quality(vectors, reconstructed.reshape(1, -1))
                    return single_result, quality
                return single_result
            quantized_result = quantize_embeddings(
                vectors,
                backend=self.backend,
                precision_mode=requested_precision,
                group_size=group_size,
            )

            single_scales = quantized_result.scales[0:1] if quantized_result.scales.ndim == 1 else quantized_result.scales[0:1, :]
            single_quantized = quantized_result.quantized[0]

            single_result = QuantizationResult(
                quantized=single_quantized,
                scales=single_scales,
                dims=quantized_result.dims,
                n=1,
                precision_mode=quantized_result.precision_mode,
                group_size=quantized_result.group_size,
            )
            
            if return_quality_metrics:
                original = vectors
                reconstructed = result.reconstruct_batch()
                quality = self.quality_analyzer.evaluate_quality(original, reconstructed)
                return single_result, quality
            else:
                return single_result
                
        elif vectors.ndim == 2:
            # Batch of vectors
            if requested_precision == "binary":
                from .binary_api import quantize_binary
                packed = quantize_binary(vectors)   # shape (n, ceil(d/8)), uint8
                n, d = vectors.shape
                orig_bytes = n * d * 4
                comp_bytes = packed.nbytes
                result = BatchQuantizationResult(
                    quantized_vectors=[packed[i] for i in range(n)],
                    scales=np.ones(n, dtype=np.float32),
                    batch_size=n,
                    vector_dim=d,
                    compression_ratio=float(orig_bytes) / float(comp_bytes),
                    total_original_bytes=orig_bytes,
                    total_compressed_bytes=comp_bytes,
                    precision_mode="binary",
                    group_size=0,
                )
            elif self.enable_batch_optimization and len(vectors) > 1:
                result = self._compress_single_batch(vectors, profile, requested_precision)
            else:
                # Process individually for small batches
                result = self._compress_individually(vectors, profile, requested_precision)
            
            if return_quality_metrics:
                original = vectors
                reconstructed = result.reconstruct_batch()
                quality = self.quality_analyzer.evaluate_quality(original, reconstructed)
                return result, quality
            else:
                return result
        else:
            raise ValueError("vectors must be 1D or 2D array")
    
    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def compress_async(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        profile: Optional[str] = None,
        precision_mode: Optional[str] = None,
        model_dir: Optional[str] = None,
    ) -> Union[QuantizationResult, BatchQuantizationResult]:
        """Non-blocking compress — runs :meth:`compress` in a thread-pool.

        Identical to :meth:`compress` but returns an awaitable, making it
        safe to call from ``async def`` request handlers (FastAPI, aiohttp,
        etc.) without blocking the event loop.

        Example::

            result = await vectro.compress_async(vectors, profile="balanced")
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.compress(
                vectors,
                profile=profile,
                precision_mode=precision_mode,
                model_dir=model_dir,
            ),
        )

    async def decompress_async(
        self,
        result: Union[QuantizationResult, BatchQuantizationResult],
    ) -> np.ndarray:
        """Non-blocking decompress — runs :meth:`decompress` in a thread-pool."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.decompress(result))

    # ------------------------------------------------------------------

    def decompress(
        self,
        result: Union[QuantizationResult, BatchQuantizationResult]
    ) -> np.ndarray:
        """Decompress quantized vectors back to float32.
        
        Args:
            result: Quantization result to decompress
            
        Returns:
            Reconstructed vectors as float32 array
        """
        if isinstance(result, QuantizationResult):
            if getattr(result, 'precision_mode', '') == 'binary':
                from .binary_api import dequantize_binary
                packed = result.quantized
                if packed.ndim == 1:
                    packed = packed.reshape(1, -1)
                reconstructed = dequantize_binary(packed, result.dims)
                return reconstructed[0] if result.n == 1 else reconstructed
            recon = reconstruct_embeddings(result, backend=self.backend)
            # Single-vector QuantizationResult: squeeze (1, d) → (d,)
            if result.n == 1 and recon.ndim == 2:
                recon = recon[0]
            return recon
        elif isinstance(result, BatchQuantizationResult):
            return result.reconstruct_batch()
        else:
            raise ValueError("result must be QuantizationResult or BatchQuantizationResult")
    
    def analyze_quality(
        self,
        original: np.ndarray,
        compressed_result: Union[QuantizationResult, BatchQuantizationResult]
    ) -> QualityMetrics:
        """Analyze compression quality metrics.
        
        Args:
            original: Original vectors
            compressed_result: Compression result
            
        Returns:
            Comprehensive quality metrics
        """
        reconstructed = self.decompress(compressed_result)
        
        if isinstance(compressed_result, BatchQuantizationResult):
            compression_ratio = compressed_result.compression_ratio
        else:
            # Estimate compression ratio for single vector
            orig_bytes = original.size * 4
            if getattr(compressed_result, 'precision_mode', '') == 'binary':
                comp_bytes = ((compressed_result.dims + 7) // 8) * compressed_result.n
            else:
                comp_bytes = len(compressed_result.quantized) * 1 + 4
            compression_ratio = orig_bytes / comp_bytes
            
        return self.quality_analyzer.evaluate_quality(
            original, reconstructed, compression_ratio
        )
    
    def benchmark_performance(
        self,
        vector_dims: List[int] = None,
        batch_sizes: List[int] = None,
        profiles: List[str] = None,
        num_trials: int = 3
    ) -> Dict[str, Any]:
        """Comprehensive performance benchmarking.
        
        Args:
            vector_dims: Dimensions to benchmark
            batch_sizes: Batch sizes to test
            profiles: Profiles to compare
            num_trials: Number of trials per configuration
            
        Returns:
            Detailed performance metrics
        """
        if vector_dims is None:
            vector_dims = [128, 384, 768]
        if batch_sizes is None:
            batch_sizes = [100, 1000]
        if profiles is None:
            profiles = ["fast", "balanced", "quality"]
        
        results = {
            "throughput_benchmarks": {},
            "profile_comparisons": {},
            "quality_benchmarks": {},
            "system_info": get_backend_info()
        }
        
        # Throughput benchmarking
        results["throughput_benchmarks"] = self.batch_processor.benchmark_batch_performance(
            batch_sizes, vector_dims, num_trials
        )
        
        # Profile comparisons
        for dim in vector_dims:
            test_vectors = np.random.randn(500, dim).astype(np.float32)
            comparison = ProfileComparison.compare_profiles(test_vectors, profiles)
            results["profile_comparisons"][f"dim_{dim}"] = comparison
        
        # Quality benchmarking across dimensions
        quality_results = QualityBenchmark.benchmark_dimensions(
            vector_dims, num_vectors=200
        )
        results["quality_benchmarks"] = {
            dim: metrics.to_dict() for dim, metrics in quality_results.items()
        }
        
        return results
    
    def optimize_profile(
        self,
        sample_vectors: np.ndarray,
        target_similarity: float = 0.995,
        target_compression: float = 3.0
    ) -> CompressionProfile:
        """Automatically optimize a compression profile for specific data.
        
        Args:
            sample_vectors: Representative sample of your data
            target_similarity: Desired cosine similarity threshold
            target_compression: Desired compression ratio
            
        Returns:
            Optimized CompressionProfile
        """
        return CompressionOptimizer.auto_optimize_profile(
            sample_vectors, target_similarity, target_compression
        )
    
    def compare_profiles(
        self,
        vectors: np.ndarray,
        profile_names: List[str] = None
    ) -> str:
        """Compare compression profiles and generate a report.
        
        Args:
            vectors: Test vectors for comparison
            profile_names: Profiles to compare (None = default set)
            
        Returns:
            Formatted comparison report
        """
        comparison = ProfileComparison.compare_profiles(vectors, profile_names)
        return ProfileComparison.generate_comparison_report(comparison)
    
    def save_compressed(
        self,
        result: Union[QuantizationResult, BatchQuantizationResult],
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save compressed vectors to file.
        
        Args:
            result: Compression result to save
            filepath: Output file path (.npz format)
            metadata: Optional user metadata persisted in the artifact
        """
        user_metadata = metadata or {}
        file_metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "created_by": "vectro",
            "vectro_version": __version__,
            "storage_format": _STORAGE_FORMAT_NAME,
            "storage_format_version": _STORAGE_FORMAT_VERSION,
            "backend": self.backend,
            "profile": self.default_profile,
            "user_metadata": user_metadata,
        }

        if isinstance(result, QuantizationResult):
            np.savez_compressed(
                filepath,
                quantized=result.quantized,
                scales=result.scales,
                dims=result.dims,
                n=result.n,
                artifact_type="single",
                precision_mode=getattr(result, "precision_mode", "int8"),
                group_size=getattr(result, "group_size", 0),
                storage_format=_STORAGE_FORMAT_NAME,
                storage_format_version=_STORAGE_FORMAT_VERSION,
                metadata_json=json.dumps(file_metadata),
            )
        elif isinstance(result, BatchQuantizationResult):
            # Convert list of arrays to single array for storage
            quantized_array = np.array(result.quantized_vectors)
            np.savez_compressed(
                filepath,
                quantized=quantized_array,
                scales=result.scales,
                batch_size=result.batch_size,
                vector_dim=result.vector_dim,
                compression_ratio=result.compression_ratio,
                total_original_bytes=result.total_original_bytes,
                total_compressed_bytes=result.total_compressed_bytes,
                artifact_type="batch",
                precision_mode=result.precision_mode,
                group_size=result.group_size,
                storage_format=_STORAGE_FORMAT_NAME,
                storage_format_version=_STORAGE_FORMAT_VERSION,
                metadata_json=json.dumps(file_metadata),
            )
    
    def load_compressed(self, filepath: str) -> Union[QuantizationResult, BatchQuantizationResult]:
        """Load compressed vectors from file.
        
        Args:
            filepath: Input file path (.npz format)
            
        Returns:
            Loaded compression result
        """
        data = np.load(filepath)
        format_version = int(data["storage_format_version"]) if "storage_format_version" in data else 1

        if format_version > _STORAGE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported storage format version {format_version}. "
                f"This Vectro build supports up to version {_STORAGE_FORMAT_VERSION}."
            )
        
        if 'batch_size' in data:
            # BatchQuantizationResult
            quantized_list = [data['quantized'][i] for i in range(len(data['quantized']))]
            
            return BatchQuantizationResult(
                quantized_vectors=quantized_list,
                scales=data['scales'],
                batch_size=int(data['batch_size']),
                vector_dim=int(data['vector_dim']),
                compression_ratio=float(data['compression_ratio']),
                total_original_bytes=int(data['total_original_bytes']),
                total_compressed_bytes=int(data['total_compressed_bytes']),
                precision_mode=str(data['precision_mode']) if 'precision_mode' in data else "int8",
                group_size=int(data['group_size']) if 'group_size' in data else 0,
            )
        else:
            # QuantizationResult
            return QuantizationResult(
                quantized=data['quantized'],
                scales=data['scales'],
                dims=int(data['dims']),
                n=int(data['n']),
                precision_mode=str(data['precision_mode']) if 'precision_mode' in data else "int8",
                group_size=int(data['group_size']) if 'group_size' in data else 0,
            )
    
    def _compress_single_batch(
        self, 
        vectors: np.ndarray, 
        profile: str,
        precision_mode: str = "int8",
    ) -> BatchQuantizationResult:
        """Compress vectors as a single batch."""
        if precision_mode == "int4":
            result = quantize_embeddings(
                vectors,
                backend=self.backend,
                precision_mode="int4",
                group_size=64,
            )

            quantized_vectors = [result.quantized[i] for i in range(result.n)]
            original_bytes = vectors.size * 4
            compressed_bytes = int(result.quantized.nbytes + result.scales.nbytes)
            compression_ratio = float(original_bytes) / float(compressed_bytes) if compressed_bytes > 0 else 0.0

            return BatchQuantizationResult(
                quantized_vectors=quantized_vectors,
                scales=result.scales,
                batch_size=result.n,
                vector_dim=result.dims,
                compression_ratio=compression_ratio,
                total_original_bytes=original_bytes,
                total_compressed_bytes=compressed_bytes,
                precision_mode="int4",
                group_size=result.group_size,
            )

        return self.batch_processor.quantize_batch(vectors, profile)
    
    def _compress_individually(
        self,
        vectors: np.ndarray,
        profile: str,
        precision_mode: str = "int8",
    ) -> BatchQuantizationResult:
        """Compress vectors, dispatching to the fast batch path when possible.

        The old implementation always called quantize_embeddings one row at a
        time even for large arrays, which incurred per-call Python overhead for
        every single vector.  Now:
          - If batch_size > 1 and the batch path supports the precision mode,
            delegate to _compress_single_batch (single NumPy call, BLAS-level).
          - Only fall back to row-by-row processing for edge cases (e.g. a
            caller that explicitly needs per-vector error isolation).
        """
        batch_size, vector_dim = vectors.shape

        # Fast path: delegate to batch quantization for multi-vector inputs
        if batch_size > 1 and precision_mode in ("int8", "int4"):
            return self._compress_single_batch(vectors, profile, precision_mode)

        # Scalar path: single vector or unsupported precision
        quantized_vectors = []
        scales_rows: List[np.ndarray] = []
        scales = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            result = quantize_embeddings(
                vectors[i:i+1],
                backend=self.backend,
                precision_mode=precision_mode,
                group_size=64,
            )
            quantized_vectors.append(result.quantized[0])
            if precision_mode == "int4":
                scales_rows.append(np.asarray(result.scales[0], dtype=np.float32))
            else:
                scales[i] = result.scales[0]

        if precision_mode == "int4":
            scales_final = np.asarray(scales_rows, dtype=np.float32)
            original_bytes = batch_size * vector_dim * 4
            compressed_bytes = int(
                np.asarray(quantized_vectors, dtype=np.uint8).nbytes + scales_final.nbytes
            )
            compression_ratio = float(original_bytes) / float(compressed_bytes) if compressed_bytes > 0 else 0.0
            return BatchQuantizationResult(
                quantized_vectors=quantized_vectors,
                scales=scales_final,
                batch_size=batch_size,
                vector_dim=vector_dim,
                compression_ratio=compression_ratio,
                total_original_bytes=original_bytes,
                total_compressed_bytes=compressed_bytes,
                precision_mode="int4",
                group_size=64,
            )

        original_bytes = batch_size * vector_dim * 4
        compressed_bytes = batch_size * vector_dim * 1 + batch_size * 4
        compression_ratio = original_bytes / compressed_bytes

        return BatchQuantizationResult(
            quantized_vectors=quantized_vectors,
            scales=scales,
            batch_size=batch_size,
            vector_dim=vector_dim,
            compression_ratio=compression_ratio,
            total_original_bytes=original_bytes,
            total_compressed_bytes=compressed_bytes,
            precision_mode="int8",
            group_size=0,
        )
    
    @property
    def available_profiles(self) -> List[str]:
        """Get list of available compression profiles."""
        return ProfileManager.list_profiles()
    
    @property
    def backend_info(self) -> Dict[str, Any]:
        """Get information about available backends."""
        return get_backend_info()


# Convenience functions for quick access
def compress_vectors(
    vectors: Union[np.ndarray, List[np.ndarray]],
    profile: str = "balanced",
    backend: str = "auto"
) -> Union[QuantizationResult, BatchQuantizationResult]:
    """Quick vector compression with default settings."""
    compressor = Vectro(backend=backend, profile=profile)
    return compressor.compress(vectors)


def decompress_vectors(
    result: Union[QuantizationResult, BatchQuantizationResult]
) -> np.ndarray:
    """Quick vector decompression."""
    compressor = Vectro()
    return compressor.decompress(result)


def analyze_compression_quality(
    original: np.ndarray,
    result: Union[QuantizationResult, BatchQuantizationResult]
) -> QualityMetrics:
    """Quick quality analysis."""
    compressor = Vectro()
    return compressor.analyze_quality(original, result)


def generate_compression_report(
    original: np.ndarray,
    result: Union[QuantizationResult, BatchQuantizationResult]
) -> str:
    """Generate a comprehensive compression report."""
    quality = analyze_compression_quality(original, result)
    return QualityReport.generate_report(quality, "Compression Quality Report")


# Module-level configuration
def set_default_backend(backend: str):
    """Set the default backend for all operations."""
    import sys
    module = sys.modules[__name__]
    module._default_backend = backend


def get_version_info() -> Dict[str, str]:
    """Get version information."""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": __description__
    }


# Example usage demonstration
if __name__ == "__main__":
    print(f"Vectro {__version__} - {__description__}")
    print("=" * 50)
    
    # Create sample embeddings
    np.random.seed(42)
    embeddings = np.random.randn(1000, 384).astype(np.float32)
    
    print(f"Sample embeddings: {embeddings.shape}")
    print(f"Original size: {embeddings.nbytes / 1024:.1f} KB")
    
    # Initialize Vectro compressor
    vectro = Vectro(profile="balanced")
    
    # Compress vectors
    print("\nCompressing vectors...")
    compressed_result = vectro.compress(embeddings)
    
    # Display compression results
    print(f"Compression ratio: {compressed_result.compression_ratio:.2f}x")
    print(f"Compressed size: {compressed_result.total_compressed_bytes / 1024:.1f} KB")
    print(f"Space savings: {(1 - 1/compressed_result.compression_ratio)*100:.1f}%")
    
    # Analyze quality
    quality = vectro.analyze_quality(embeddings, compressed_result)
    print(f"\nQuality Analysis:")
    print(f"  Cosine similarity: {quality.mean_cosine_similarity:.6f}")
    print(f"  Quality grade: {quality.quality_grade()}")
    print(f"  Mean Absolute Error: {quality.mean_absolute_error:.6f}")
    
    # Compare profiles
    print(f"\nProfile Comparison:")
    sample_vectors = embeddings[:100]  # Small sample for demo
    comparison_report = vectro.compare_profiles(sample_vectors)
    print(comparison_report)
    
    print(f"\nAvailable backends: {vectro.backend_info}")
    print(f"Available profiles: {vectro.available_profiles}")
    
    # Demonstrate round-trip compression
    decompressed = vectro.decompress(compressed_result)
    reconstruction_error = np.mean(np.abs(embeddings - decompressed))
    print(f"\nRound-trip reconstruction error: {reconstruction_error:.6f}")
    
    print("\n✅ Vectro API demonstration completed successfully!")