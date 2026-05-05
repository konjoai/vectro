"""Type stubs for python/vectro.py."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .batch_api import (
    BatchCompressionAnalyzer as BatchCompressionAnalyzer,
    BatchQuantizationResult as BatchQuantizationResult,
    VectroBatchProcessor as VectroBatchProcessor,
    benchmark_batch_compression as benchmark_batch_compression,
    quantize_embeddings_batch as quantize_embeddings_batch,
)
from .interface import (
    QuantizationResult as QuantizationResult,
    get_backend_info as get_backend_info,
    mean_cosine_similarity as mean_cosine_similarity,
    quantize_embeddings as quantize_embeddings,
    reconstruct_embeddings as reconstruct_embeddings,
)
from .profiles_api import (
    CompressionOptimizer as CompressionOptimizer,
    CompressionProfile as CompressionProfile,
    CompressionStrategy as CompressionStrategy,
    ProfileComparison as ProfileComparison,
    ProfileManager as ProfileManager,
    create_custom_profile as create_custom_profile,
    get_compression_profile as get_compression_profile,
)
from .quality_api import (
    QualityBenchmark as QualityBenchmark,
    QualityMetrics as QualityMetrics,
    QualityReport as QualityReport,
    VectroQualityAnalyzer as VectroQualityAnalyzer,
    evaluate_quantization_quality as evaluate_quantization_quality,
    generate_quality_report as generate_quality_report,
)

__version__: str
__author__: str
__license__: str
__description__: str

_STORAGE_FORMAT_VERSION: int
_STORAGE_FORMAT_NAME: str
_VALID_PRECISION_MODES: frozenset[str]
_VALID_BACKENDS: frozenset[str]
_VALID_PROFILES: frozenset[str]


class QuantizationConfig:
    """Validated configuration container for a single Vectro compression run.

    Provides a structured, type-checked alternative to the raw ``profile=`` /
    ``precision_mode=`` string arguments on :meth:`Vectro.compress`.

    Parameters
    ----------
    precision_mode : One of ``"int8"`` (default), ``"nf4"``, ``"binary"``,
        ``"int4"``, ``"int2"``.
    profile : Named compression profile (``"fast"``, ``"balanced"``, ``"quality"``,
        ``"ultra"``, ``"binary"``).  ``None`` means no named profile.
    group_size : Sub-vector group size; must be a positive power of 2.  Default 64.
    assume_normalized : Skip abs-max scan for L2-normalised inputs.
    return_quality_metrics : Return ``(result, QualityMetrics)`` tuple from compress.
    model_dir : HuggingFace model directory for auto-routing via the family registry.
    seed : Integer seed for stochastic operations, or ``None``.
    """

    precision_mode: str
    profile: Optional[str]
    group_size: int
    assume_normalized: bool
    return_quality_metrics: bool
    model_dir: Optional[str]
    seed: Optional[int]

    def __init__(
        self,
        precision_mode: str = "int8",
        profile: Optional[str] = None,
        group_size: int = 64,
        assume_normalized: bool = False,
        return_quality_metrics: bool = False,
        model_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None: ...

    def __post_init__(self) -> None: ...

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of all config fields."""
        ...

    @classmethod
    def from_profile(cls, profile: str, **overrides: Any) -> "QuantizationConfig":
        """Construct a config from a named profile with optional field overrides."""
        ...


class Vectro:
    """Main Vectro API class — unified access to all compression capabilities."""

    backend: str
    default_profile: str
    enable_batch_optimization: bool
    batch_processor: VectroBatchProcessor
    quality_analyzer: VectroQualityAnalyzer

    def __init__(
        self,
        backend: str = "auto",
        profile: str = "balanced",
        enable_batch_optimization: bool = True,
    ) -> None: ...

    def compress(
        self,
        vectors: Union[np.ndarray, list],
        profile: Optional[str] = None,
        precision_mode: Optional[str] = None,
        return_quality_metrics: bool = False,
        model_dir: Optional[str] = None,
        config: Optional[QuantizationConfig] = None,
    ) -> Union[QuantizationResult, BatchQuantizationResult, Tuple[Any, QualityMetrics]]: ...

    async def compress_async(
        self,
        vectors: Union[np.ndarray, list],
        profile: Optional[str] = None,
        precision_mode: Optional[str] = None,
    ) -> Union[QuantizationResult, BatchQuantizationResult]: ...

    def decompress(
        self,
        result: Union[QuantizationResult, BatchQuantizationResult],
    ) -> np.ndarray: ...

    async def decompress_async(
        self,
        result: Union[QuantizationResult, BatchQuantizationResult],
    ) -> np.ndarray: ...

    def analyze_quality(
        self,
        original: np.ndarray,
        compressed_result: Union[QuantizationResult, BatchQuantizationResult],
    ) -> QualityMetrics: ...

    def benchmark_performance(
        self,
        vector_dims: Optional[list] = None,
        batch_sizes: Optional[list] = None,
        profiles: Optional[list] = None,
        num_trials: int = 3,
    ) -> Dict[str, Any]: ...

    def optimize_profile(
        self,
        sample_vectors: np.ndarray,
        target_similarity: float = 0.995,
        target_compression: float = 3.0,
    ) -> CompressionProfile: ...

    def compare_profiles(
        self,
        vectors: np.ndarray,
        profile_names: Optional[list] = None,
    ) -> str: ...

    def save_compressed(
        self,
        result: Union[QuantizationResult, BatchQuantizationResult],
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def load_compressed(
        self,
        filepath: str,
    ) -> Union[QuantizationResult, BatchQuantizationResult]: ...

    @property
    def available_profiles(self) -> list: ...

    @property
    def backend_info(self) -> Dict[str, Any]: ...


def compress_vectors(
    vectors: Union[np.ndarray, list],
    profile: str = "balanced",
    backend: str = "auto",
) -> Union[QuantizationResult, BatchQuantizationResult]: ...

def decompress_vectors(
    result: Union[QuantizationResult, BatchQuantizationResult],
) -> np.ndarray: ...

def analyze_compression_quality(
    original: np.ndarray,
    result: Union[QuantizationResult, BatchQuantizationResult],
) -> QualityMetrics: ...

def generate_compression_report(
    original: np.ndarray,
    result: Union[QuantizationResult, BatchQuantizationResult],
) -> str: ...

def get_version_info() -> Dict[str, str]: ...
