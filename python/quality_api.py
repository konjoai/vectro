"""
Python API for Vectro quality metrics and analysis.
Provides comprehensive quality assessment tools for quantized embeddings.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for quantization assessment."""

    mean_absolute_error: float  # Average absolute reconstruction error
    mean_squared_error: float  # Average squared reconstruction error
    root_mean_squared_error: float  # Square root of MSE
    mean_cosine_similarity: float  # Average cosine similarity between original and reconstructed
    min_cosine_similarity: float  # Minimum cosine similarity observed
    max_cosine_similarity: float  # Maximum cosine similarity observed
    percentile_errors: Dict[str, float]  # Error percentiles (p25, p50, p75, p95, p99)
    signal_to_noise_ratio: float  # SNR in dB
    peak_signal_to_noise_ratio: float  # PSNR in dB
    structural_similarity: float  # SSIM-like metric for vectors
    compression_ratio: float  # Achieved compression ratio

    def quality_grade(self) -> str:
        """Return a quality grade based on cosine similarity."""
        if self.mean_cosine_similarity >= 0.999:
            return "A+ (Excellent)"
        elif self.mean_cosine_similarity >= 0.995:
            return "A (Very Good)"
        elif self.mean_cosine_similarity >= 0.99:
            return "B+ (Good)"
        elif self.mean_cosine_similarity >= 0.985:
            return "B (Acceptable)"
        elif self.mean_cosine_similarity >= 0.98:
            return "C+ (Fair)"
        elif self.mean_cosine_similarity >= 0.97:
            return "C (Poor)"
        else:
            return "D (Unacceptable)"

    def passes_quality_threshold(self, min_cosine_sim: float = 0.995) -> bool:
        """Check if quality meets minimum threshold."""
        return self.mean_cosine_similarity >= min_cosine_sim

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format."""
        result = {
            "mean_absolute_error": self.mean_absolute_error,
            "mean_squared_error": self.mean_squared_error,
            "root_mean_squared_error": self.root_mean_squared_error,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "min_cosine_similarity": self.min_cosine_similarity,
            "max_cosine_similarity": self.max_cosine_similarity,
            "signal_to_noise_ratio": self.signal_to_noise_ratio,
            "peak_signal_to_noise_ratio": self.peak_signal_to_noise_ratio,
            "structural_similarity": self.structural_similarity,
            "compression_ratio": self.compression_ratio,
            "quality_grade": self.quality_grade(),
        }
        result.update(self.percentile_errors)
        return result


class VectroQualityAnalyzer:
    """Advanced quality analysis for vector quantization."""

    @staticmethod
    def evaluate_quality(
        original: np.ndarray, reconstructed: np.ndarray, compression_ratio: Optional[float] = None
    ) -> QualityMetrics:
        """Comprehensive quality evaluation of quantized vectors.

        Args:
            original: Original vectors, shape (n, d)
            reconstructed: Reconstructed vectors, shape (n, d)
            compression_ratio: Optional compression ratio

        Returns:
            QualityMetrics with comprehensive assessment
        """
        if original.shape != reconstructed.shape:
            raise ValueError("Original and reconstructed arrays must have same shape")

        original = original.astype(np.float32)
        reconstructed = reconstructed.astype(np.float32)

        # Basic error metrics
        error = original - reconstructed
        abs_error = np.abs(error)
        squared_error = error**2

        mae = np.mean(abs_error)
        mse = np.mean(squared_error)
        rmse = np.sqrt(mse)

        # Cosine similarity metrics
        cosine_similarities = VectroQualityAnalyzer._compute_cosine_similarities(
            original, reconstructed
        )
        mean_cos_sim = np.mean(cosine_similarities)
        min_cos_sim = np.min(cosine_similarities)
        max_cos_sim = np.max(cosine_similarities)

        # Percentile error analysis
        percentiles = VectroQualityAnalyzer._compute_error_percentiles(abs_error)

        # Signal quality metrics
        snr = VectroQualityAnalyzer._compute_snr(original, error)
        psnr = VectroQualityAnalyzer._compute_psnr(original, mse)

        # Structural similarity (simplified for vectors)
        ssim = VectroQualityAnalyzer._compute_vector_ssim(original, reconstructed)

        # Estimate compression ratio if not provided
        if compression_ratio is None:
            # Assume float32 -> int8 + scale
            orig_bytes = original.size * 4
            compressed_bytes = original.size * 1 + original.shape[0] * 4
            compression_ratio = orig_bytes / compressed_bytes

        return QualityMetrics(
            mean_absolute_error=mae,
            mean_squared_error=mse,
            root_mean_squared_error=rmse,
            mean_cosine_similarity=mean_cos_sim,
            min_cosine_similarity=min_cos_sim,
            max_cosine_similarity=max_cos_sim,
            percentile_errors=percentiles,
            signal_to_noise_ratio=snr,
            peak_signal_to_noise_ratio=psnr,
            structural_similarity=ssim,
            compression_ratio=compression_ratio,
        )

    @staticmethod
    def _compute_cosine_similarities(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between corresponding vectors."""
        # Normalize vectors
        orig_norms = np.linalg.norm(original, axis=1, keepdims=True)
        recon_norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)

        # Handle zero vectors
        orig_norms = np.where(orig_norms == 0, 1, orig_norms)
        recon_norms = np.where(recon_norms == 0, 1, recon_norms)

        orig_normalized = original / orig_norms
        recon_normalized = reconstructed / recon_norms

        # Compute dot products (cosine similarities)
        similarities = np.sum(orig_normalized * recon_normalized, axis=1)

        # Handle cases where both vectors are zero
        both_zero = (np.linalg.norm(original, axis=1) == 0) & (
            np.linalg.norm(reconstructed, axis=1) == 0
        )
        similarities[both_zero] = 1.0  # Perfect similarity for zero vectors

        return similarities

    @staticmethod
    def _compute_error_percentiles(abs_error: np.ndarray) -> Dict[str, float]:
        """Compute error percentiles."""
        flat_error = abs_error.flatten()

        return {
            "error_p25": float(np.percentile(flat_error, 25)),
            "error_p50": float(np.percentile(flat_error, 50)),
            "error_p75": float(np.percentile(flat_error, 75)),
            "error_p95": float(np.percentile(flat_error, 95)),
            "error_p99": float(np.percentile(flat_error, 99)),
            "error_p99_9": float(np.percentile(flat_error, 99.9)),
        }

    @staticmethod
    def _compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio in dB."""
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)

        if noise_power == 0:
            return float("inf")

        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)

        return float(snr_db)

    @staticmethod
    def _compute_psnr(signal: np.ndarray, mse: float) -> float:
        """Compute Peak Signal-to-Noise Ratio in dB."""
        max_signal = np.max(signal)

        if mse == 0:
            return float("inf")

        psnr = 20 * np.log10(max_signal / np.sqrt(mse))

        return float(psnr)

    @staticmethod
    def _compute_vector_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute Structural Similarity Index adapted for vectors."""
        # Simplified SSIM for vectors
        mu_orig = np.mean(original, axis=1)
        mu_recon = np.mean(reconstructed, axis=1)

        var_orig = np.var(original, axis=1)
        var_recon = np.var(reconstructed, axis=1)

        covar = np.mean(
            (original - mu_orig[:, np.newaxis]) * (reconstructed - mu_recon[:, np.newaxis]), axis=1
        )

        # SSIM constants
        c1 = 0.01**2
        c2 = 0.03**2

        numerator = (2 * mu_orig * mu_recon + c1) * (2 * covar + c2)
        denominator = (mu_orig**2 + mu_recon**2 + c1) * (var_orig + var_recon + c2)

        # Avoid division by zero
        ssim_values = np.where(denominator != 0, numerator / denominator, 1.0)

        return float(np.mean(ssim_values))


class QualityBenchmark:
    """Benchmark quality metrics across different scenarios."""

    @staticmethod
    def benchmark_dimensions(
        dimensions: List[int] = None, num_vectors: int = 1000, quantization_func: callable = None
    ) -> Dict[int, QualityMetrics]:
        """Benchmark quality across different vector dimensions."""
        if dimensions is None:
            dimensions = [64, 128, 256, 384, 512, 768, 1024, 1536]

        results = {}

        for dim in dimensions:
            # Generate test vectors
            original = np.random.randn(num_vectors, dim).astype(np.float32)

            if quantization_func:
                reconstructed = quantization_func(original)
            else:
                # Default quantization for benchmarking
                reconstructed = QualityBenchmark._default_quantize(original)

            quality = VectroQualityAnalyzer.evaluate_quality(original, reconstructed)
            results[dim] = quality

        return results

    @staticmethod
    def benchmark_vector_counts(
        vector_counts: List[int] = None, dimension: int = 384, quantization_func: callable = None
    ) -> Dict[int, QualityMetrics]:
        """Benchmark quality with different numbers of vectors."""
        if vector_counts is None:
            vector_counts = [100, 500, 1000, 2000, 5000, 10000]

        results = {}

        for count in vector_counts:
            # Generate test vectors
            original = np.random.randn(count, dimension).astype(np.float32)

            if quantization_func:
                reconstructed = quantization_func(original)
            else:
                reconstructed = QualityBenchmark._default_quantize(original)

            quality = VectroQualityAnalyzer.evaluate_quality(original, reconstructed)
            results[count] = quality

        return results

    @staticmethod
    def _default_quantize(vectors: np.ndarray) -> np.ndarray:
        """Default quantization for benchmarking."""
        reconstructed = np.zeros_like(vectors)

        for i in range(len(vectors)):
            vec = vectors[i]
            max_abs = np.max(np.abs(vec))

            if max_abs == 0:
                scale = 1.0
            else:
                scale = max_abs / 127.0

            # Quantize to int8 and reconstruct
            quantized = np.round(vec / scale).astype(np.int8)
            quantized = np.clip(quantized, -127, 127)
            reconstructed[i] = quantized.astype(np.float32) * scale

        return reconstructed


class QualityReport:
    """Generate comprehensive quality assessment reports."""

    @staticmethod
    def generate_report(
        quality_metrics: QualityMetrics, title: str = "Vectro Quality Assessment"
    ) -> str:
        """Generate a formatted quality report."""
        report = f"""
{title}
{"=" * len(title)}

Overall Quality Grade: {quality_metrics.quality_grade()}
Quality Threshold: {"✅ PASS" if quality_metrics.passes_quality_threshold() else "❌ FAIL"}

Similarity Metrics:
  Mean Cosine Similarity: {quality_metrics.mean_cosine_similarity:.6f}
  Min Cosine Similarity:  {quality_metrics.min_cosine_similarity:.6f}  
  Max Cosine Similarity:  {quality_metrics.max_cosine_similarity:.6f}

Error Metrics:
  Mean Absolute Error:    {quality_metrics.mean_absolute_error:.6f}
  Mean Squared Error:     {quality_metrics.mean_squared_error:.6f}
  Root Mean Squared Error: {quality_metrics.root_mean_squared_error:.6f}

Error Percentiles:
  25th percentile:        {quality_metrics.percentile_errors["error_p25"]:.6f}
  50th percentile (median): {quality_metrics.percentile_errors["error_p50"]:.6f}
  75th percentile:        {quality_metrics.percentile_errors["error_p75"]:.6f}
  95th percentile:        {quality_metrics.percentile_errors["error_p95"]:.6f}
  99th percentile:        {quality_metrics.percentile_errors["error_p99"]:.6f}
  99.9th percentile:      {quality_metrics.percentile_errors["error_p99_9"]:.6f}

Signal Quality:
  Signal-to-Noise Ratio: {quality_metrics.signal_to_noise_ratio:.2f} dB
  Peak SNR:              {quality_metrics.peak_signal_to_noise_ratio:.2f} dB
  Structural Similarity:  {quality_metrics.structural_similarity:.6f}

Compression:
  Compression Ratio:      {quality_metrics.compression_ratio:.2f}x
  Space Savings:         {((1 - 1 / quality_metrics.compression_ratio) * 100):.1f}%
"""
        return report

    @staticmethod
    def compare_configurations(
        results: Dict[str, QualityMetrics], title: str = "Configuration Comparison"
    ) -> str:
        """Generate a comparison report for different configurations."""
        report = f"""
{title}
{"=" * len(title)}

Configuration Performance Summary:
"""

        # Sort by cosine similarity (descending)
        sorted_results = sorted(
            results.items(), key=lambda x: x[1].mean_cosine_similarity, reverse=True
        )

        report += f"{'Config':<15} {'Cos Sim':<10} {'MAE':<12} {'Compression':<12} {'Grade'}\n"
        report += f"{'-' * 15} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 10}\n"

        for config_name, metrics in sorted_results:
            report += f"{config_name:<15} "
            report += f"{metrics.mean_cosine_similarity:.6f} "
            report += f"{metrics.mean_absolute_error:.6f}   "
            report += f"{metrics.compression_ratio:.2f}x        "
            report += f"{metrics.quality_grade()}\n"

        return report


# Convenience functions
def evaluate_quantization_quality(
    original: np.ndarray, reconstructed: np.ndarray, compression_ratio: Optional[float] = None
) -> QualityMetrics:
    """Convenience function for quality evaluation."""
    return VectroQualityAnalyzer.evaluate_quality(original, reconstructed, compression_ratio)


def generate_quality_report(
    original: np.ndarray, reconstructed: np.ndarray, title: str = "Quality Assessment"
) -> str:
    """Convenience function to generate a quality report."""
    metrics = evaluate_quantization_quality(original, reconstructed)
    return QualityReport.generate_report(metrics, title)


# Example usage
if __name__ == "__main__":
    print("Vectro Quality Analysis API Demo")
    print("=" * 40)

    # Generate test data
    np.random.seed(42)
    original = np.random.randn(1000, 384).astype(np.float32)

    # Simulate quantization
    reconstructed = QualityBenchmark._default_quantize(original)

    # Analyze quality
    metrics = evaluate_quantization_quality(original, reconstructed)

    print("Quality Analysis Results:")
    print(f"  Cosine Similarity: {metrics.mean_cosine_similarity:.6f}")
    print(f"  Quality Grade: {metrics.quality_grade()}")
    print(f"  Mean Absolute Error: {metrics.mean_absolute_error:.6f}")
    print(f"  Compression Ratio: {metrics.compression_ratio:.2f}x")

    # Generate detailed report
    report = generate_quality_report(original, reconstructed, "Demo Analysis")
    print("\nDetailed Report:")
    print(report)
