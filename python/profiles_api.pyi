import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Any

class CompressionStrategy(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    CUSTOM = "custom"

@dataclass
class CompressionProfile:
    name: str
    strategy: CompressionStrategy
    quantization_bits: int
    range_factor: float
    clipping_percentile: float
    adaptive_scaling: bool
    batch_optimization: bool
    precision_mode: str
    error_correction: bool
    simd_enabled: bool
    parallel_processing: bool
    memory_efficient: bool
    preserve_norms: bool
    preserve_angles: bool
    min_similarity_threshold: float
    def __post_init__(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompressionProfile: ...

class ProfileManager:
    _builtin_profiles: dict[str, CompressionProfile]
    _custom_profiles: dict[str, CompressionProfile]
    @classmethod
    def initialize_builtin_profiles(cls) -> None: ...
    @classmethod
    def get_profile(cls, name: str) -> CompressionProfile: ...
    @classmethod
    def list_profiles(cls) -> list[str]: ...
    @classmethod
    def add_custom_profile(cls, profile: CompressionProfile): ...
    @classmethod
    def remove_custom_profile(cls, name: str): ...
    @classmethod
    def save_profiles(cls, filepath: str): ...
    @classmethod
    def load_profiles(cls, filepath: str): ...

class CompressionOptimizer:
    @staticmethod
    def auto_optimize_profile(
        sample_vectors: np.ndarray,
        target_similarity: float = 0.995,
        target_compression: float = 3.0,
        max_iterations: int = 10,
    ) -> CompressionProfile: ...
    @staticmethod
    def _evaluate_profile(
        profile: CompressionProfile,
        vectors: np.ndarray,
        target_similarity: float,
        target_compression: float,
    ) -> float: ...
    @staticmethod
    def _simulate_compression(vectors: np.ndarray, profile: CompressionProfile) -> np.ndarray: ...

class ProfileComparison:
    @staticmethod
    def compare_profiles(
        vectors: np.ndarray, profile_names: list[str] = None
    ) -> dict[str, dict[str, float]]: ...
    @staticmethod
    def generate_comparison_report(
        comparison_results: dict[str, dict[str, float]], title: str = "Profile Comparison Report"
    ) -> str: ...

def get_compression_profile(name: str) -> CompressionProfile: ...
def create_custom_profile(
    name: str, quantization_bits: int = 8, range_factor: float = 0.95, **kwargs
) -> CompressionProfile: ...
