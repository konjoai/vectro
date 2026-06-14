import numpy as np
from dataclasses import dataclass

@dataclass
class QualityMetrics:
    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    mean_cosine_similarity: float
    min_cosine_similarity: float
    max_cosine_similarity: float
    percentile_errors: dict[str, float]
    signal_to_noise_ratio: float
    peak_signal_to_noise_ratio: float
    structural_similarity: float
    compression_ratio: float
    def quality_grade(self) -> str: ...
    def passes_quality_threshold(self, min_cosine_sim: float = 0.995) -> bool: ...
    def to_dict(self) -> dict[str, float]: ...

class VectroQualityAnalyzer:
    @staticmethod
    def evaluate_quality(
        original: np.ndarray, reconstructed: np.ndarray, compression_ratio: float | None = None
    ) -> QualityMetrics: ...
    @staticmethod
    def _compute_cosine_similarities(
        original: np.ndarray, reconstructed: np.ndarray
    ) -> np.ndarray: ...
    @staticmethod
    def _compute_error_percentiles(abs_error: np.ndarray) -> dict[str, float]: ...
    @staticmethod
    def _compute_snr(signal: np.ndarray, noise: np.ndarray) -> float: ...
    @staticmethod
    def _compute_psnr(signal: np.ndarray, mse: float) -> float: ...
    @staticmethod
    def _compute_vector_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float: ...

class QualityBenchmark:
    @staticmethod
    def benchmark_dimensions(
        dimensions: list[int] = None, num_vectors: int = 1000, quantization_func: callable = None
    ) -> dict[int, QualityMetrics]: ...
    @staticmethod
    def benchmark_vector_counts(
        vector_counts: list[int] = None, dimension: int = 384, quantization_func: callable = None
    ) -> dict[int, QualityMetrics]: ...
    @staticmethod
    def _default_quantize(vectors: np.ndarray) -> np.ndarray: ...

class QualityReport:
    @staticmethod
    def generate_report(
        quality_metrics: QualityMetrics, title: str = "Vectro Quality Assessment"
    ) -> str: ...
    @staticmethod
    def compare_configurations(
        results: dict[str, QualityMetrics], title: str = "Configuration Comparison"
    ) -> str: ...

def evaluate_quantization_quality(
    original: np.ndarray, reconstructed: np.ndarray, compression_ratio: float | None = None
) -> QualityMetrics: ...
def generate_quality_report(
    original: np.ndarray, reconstructed: np.ndarray, title: str = "Quality Assessment"
) -> str: ...
