import numpy as np
from . import _mojo_bridge as _mojo_bridge
from .interface import QuantizationResult as QuantizationResult, dequantize_int4 as dequantize_int4
from _typeshed import Incomplete
from dataclasses import dataclass

@dataclass
class BatchQuantizationResult:
    quantized_vectors: list[np.ndarray]
    scales: np.ndarray
    batch_size: int
    vector_dim: int
    compression_ratio: float
    total_original_bytes: int
    total_compressed_bytes: int
    precision_mode: str = ...
    group_size: int = ...
    def get_vector(self, index: int) -> tuple[np.ndarray, float | np.ndarray]: ...
    def reconstruct_vector(self, index: int) -> np.ndarray: ...
    def reconstruct_batch(self) -> np.ndarray: ...

class VectroBatchProcessor:
    backend: Incomplete
    _mojo_binary: Incomplete
    def __init__(self, backend: str = "auto") -> None: ...
    def _find_mojo_binary(self) -> str | None: ...
    def quantize_batch(
        self, vectors: np.ndarray | list[np.ndarray], profile: str = "balanced"
    ) -> BatchQuantizationResult: ...
    def _quantize_batch_mojo(
        self, vectors: np.ndarray, profile: str
    ) -> BatchQuantizationResult: ...
    def _quantize_batch_python(
        self, vectors: np.ndarray, profile: str
    ) -> BatchQuantizationResult: ...
    def quantize_streaming(
        self,
        vectors: np.ndarray | list[np.ndarray],
        chunk_size: int = 1000,
        profile: str = "balanced",
    ) -> list[BatchQuantizationResult]: ...
    def benchmark_batch_performance(
        self, batch_sizes: list[int] = None, vector_dims: list[int] = None, num_trials: int = 3
    ) -> dict[str, dict[str, float]]: ...

class BatchCompressionAnalyzer:
    @staticmethod
    def analyze_batch_result(result: BatchQuantizationResult) -> dict[str, float]: ...
    @staticmethod
    def compare_profiles(
        vectors: np.ndarray, profiles: list[str] = None
    ) -> dict[str, dict[str, float]]: ...

def quantize_embeddings_batch(
    embeddings: np.ndarray, profile: str = "balanced"
) -> BatchQuantizationResult: ...
def benchmark_batch_compression(
    embeddings: np.ndarray, profiles: list[str] = None
) -> dict[str, dict[str, float]]: ...
