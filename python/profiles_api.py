"""
Python API for Vectro compression profiles.
Provides configurable compression strategies optimized for different use cases.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import json


class CompressionStrategy(Enum):
    """Available compression strategies."""
    FAST = "fast"           # Maximum speed, minimal quality loss
    BALANCED = "balanced"   # Balance between speed and quality  
    QUALITY = "quality"     # Maximum quality preservation
    CUSTOM = "custom"       # User-defined parameters


@dataclass
class CompressionProfile:
    """Configuration for vector compression behavior."""
    name: str                    # Profile name
    strategy: CompressionStrategy  # Compression strategy
    quantization_bits: int      # Number of quantization bits (8, 4, 2, 1)
    range_factor: float          # Scaling factor for quantization range [0.0, 1.0]
    clipping_percentile: float   # Percentile for outlier clipping [0.0, 100.0]
    adaptive_scaling: bool       # Use adaptive per-vector scaling
    batch_optimization: bool     # Enable batch processing optimizations
    precision_mode: str          # 'int8', 'int4', 'binary'
    error_correction: bool       # Enable error correction techniques
    
    # Performance tuning
    simd_enabled: bool           # Enable SIMD optimizations
    parallel_processing: bool    # Enable parallel processing
    memory_efficient: bool       # Use memory-efficient algorithms
    
    # Quality preservation
    preserve_norms: bool         # Preserve vector norms
    preserve_angles: bool        # Preserve angular relationships
    min_similarity_threshold: float  # Minimum acceptable cosine similarity

    # Wave 1.3 — input invariant.  When True, the encoder asserts that
    # every row has ||·||_2 ≤ 1 (text-embedding-3-*, gte-large, bge-m3,
    # e5 are all L2-normalised by default) and skips the abs-max scan
    # entirely.  Trades a small amount of i8 dynamic range for ~1.4×
    # encode throughput.  Off by default; opt in per profile.
    assume_normalized: bool = False

    def __post_init__(self):
        """Validate profile parameters."""
        if not 1 <= self.quantization_bits <= 8:
            raise ValueError("quantization_bits must be between 1 and 8")
        if not 0.0 <= self.range_factor <= 1.0:
            raise ValueError("range_factor must be between 0.0 and 1.0")
        if not 0.0 <= self.clipping_percentile <= 100.0:
            raise ValueError("clipping_percentile must be between 0.0 and 100.0")
        if not 0.0 <= self.min_similarity_threshold <= 1.0:
            raise ValueError("min_similarity_threshold must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "quantization_bits": self.quantization_bits,
            "range_factor": self.range_factor,
            "clipping_percentile": self.clipping_percentile,
            "adaptive_scaling": self.adaptive_scaling,
            "batch_optimization": self.batch_optimization,
            "precision_mode": self.precision_mode,
            "error_correction": self.error_correction,
            "simd_enabled": self.simd_enabled,
            "parallel_processing": self.parallel_processing,
            "memory_efficient": self.memory_efficient,
            "preserve_norms": self.preserve_norms,
            "preserve_angles": self.preserve_angles,
            "min_similarity_threshold": self.min_similarity_threshold,
            "assume_normalized": self.assume_normalized,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressionProfile':
        """Create profile from dictionary."""
        strategy = CompressionStrategy(data["strategy"])
        return cls(
            name=data["name"],
            strategy=strategy,
            quantization_bits=data["quantization_bits"],
            range_factor=data["range_factor"],
            clipping_percentile=data["clipping_percentile"],
            adaptive_scaling=data["adaptive_scaling"],
            batch_optimization=data["batch_optimization"],
            precision_mode=data["precision_mode"],
            error_correction=data["error_correction"],
            simd_enabled=data["simd_enabled"],
            parallel_processing=data["parallel_processing"],
            memory_efficient=data["memory_efficient"],
            preserve_norms=data["preserve_norms"],
            preserve_angles=data["preserve_angles"],
            min_similarity_threshold=data["min_similarity_threshold"],
            assume_normalized=data.get("assume_normalized", False),
        )


class ProfileManager:
    """Manages compression profiles and provides built-in configurations."""
    
    _builtin_profiles: Dict[str, CompressionProfile] = {}
    _custom_profiles: Dict[str, CompressionProfile] = {}
    
    @classmethod
    def initialize_builtin_profiles(cls):
        """Initialize built-in compression profiles."""
        if cls._builtin_profiles:
            return  # Already initialized
        
        # Fast Profile - Maximum speed
        cls._builtin_profiles["fast"] = CompressionProfile(
            name="fast",
            strategy=CompressionStrategy.FAST,
            quantization_bits=8,
            range_factor=1.0,               # Use full int8 range
            clipping_percentile=100.0,      # No clipping
            adaptive_scaling=True,
            batch_optimization=True,
            precision_mode="int8",
            error_correction=False,
            simd_enabled=True,
            parallel_processing=True,
            memory_efficient=False,
            preserve_norms=False,
            preserve_angles=False,
            min_similarity_threshold=0.990
        )
        
        # Balanced Profile - Speed/quality balance
        cls._builtin_profiles["balanced"] = CompressionProfile(
            name="balanced",
            strategy=CompressionStrategy.BALANCED,
            quantization_bits=8,
            range_factor=0.95,              # Conservative range usage
            clipping_percentile=99.5,       # Clip extreme outliers
            adaptive_scaling=True,
            batch_optimization=True,
            precision_mode="int8",
            error_correction=True,
            simd_enabled=True,
            parallel_processing=True,
            memory_efficient=True,
            preserve_norms=True,
            preserve_angles=False,
            min_similarity_threshold=0.995
        )
        
        # Quality Profile - Maximum quality preservation
        cls._builtin_profiles["quality"] = CompressionProfile(
            name="quality",
            strategy=CompressionStrategy.QUALITY,
            quantization_bits=8,
            range_factor=0.90,              # Very conservative range
            clipping_percentile=99.0,       # Aggressive outlier clipping
            adaptive_scaling=True,
            batch_optimization=False,       # Prioritize quality over speed
            precision_mode="int8",
            error_correction=True,
            simd_enabled=False,             # Disable for accuracy
            parallel_processing=False,
            memory_efficient=False,
            preserve_norms=True,
            preserve_angles=True,
            min_similarity_threshold=0.997
        )
        
        # Ultra Profile - Extreme compression
        cls._builtin_profiles["ultra"] = CompressionProfile(
            name="ultra",
            strategy=CompressionStrategy.CUSTOM,
            quantization_bits=4,            # 4-bit quantization
            range_factor=0.85,
            clipping_percentile=98.0,
            adaptive_scaling=True,
            batch_optimization=True,
            precision_mode="int4",
            error_correction=True,
            simd_enabled=True,
            parallel_processing=True,
            memory_efficient=True,
            preserve_norms=True,
            preserve_angles=False,
            min_similarity_threshold=0.985
        )
        
        # Binary Profile - Maximum compression
        cls._builtin_profiles["binary"] = CompressionProfile(
            name="binary",
            strategy=CompressionStrategy.CUSTOM,
            quantization_bits=1,            # Binary quantization
            range_factor=1.0,
            clipping_percentile=95.0,
            adaptive_scaling=False,         # Use global threshold
            batch_optimization=True,
            precision_mode="binary",
            error_correction=False,
            simd_enabled=True,
            parallel_processing=True,
            memory_efficient=True,
            preserve_norms=False,
            preserve_angles=False,
            min_similarity_threshold=0.950
        )
    
    @classmethod
    def get_profile(cls, name: str) -> CompressionProfile:
        """Get a compression profile by name."""
        cls.initialize_builtin_profiles()
        
        # Check builtin profiles first
        if name in cls._builtin_profiles:
            return cls._builtin_profiles[name]
        
        # Check custom profiles
        if name in cls._custom_profiles:
            return cls._custom_profiles[name]
        
        raise ValueError(f"Profile '{name}' not found. Available profiles: {cls.list_profiles()}")
    
    @classmethod
    def list_profiles(cls) -> List[str]:
        """List all available profile names."""
        cls.initialize_builtin_profiles()
        builtin = list(cls._builtin_profiles.keys())
        custom = list(cls._custom_profiles.keys())
        return builtin + custom
    
    @classmethod
    def add_custom_profile(cls, profile: CompressionProfile):
        """Add a custom compression profile."""
        cls._custom_profiles[profile.name] = profile
    
    @classmethod
    def remove_custom_profile(cls, name: str):
        """Remove a custom compression profile."""
        if name in cls._custom_profiles:
            del cls._custom_profiles[name]
        else:
            raise ValueError(f"Custom profile '{name}' not found")
    
    @classmethod
    def save_profiles(cls, filepath: str):
        """Save custom profiles to file."""
        data = {
            name: profile.to_dict() 
            for name, profile in cls._custom_profiles.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_profiles(cls, filepath: str):
        """Load custom profiles from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for name, profile_data in data.items():
            profile = CompressionProfile.from_dict(profile_data)
            cls.add_custom_profile(profile)


class CompressionOptimizer:
    """Optimizes compression profiles for specific datasets."""
    
    @staticmethod
    def auto_optimize_profile(
        sample_vectors: np.ndarray,
        target_similarity: float = 0.995,
        target_compression: float = 3.0,
        max_iterations: int = 10
    ) -> CompressionProfile:
        """Automatically optimize a compression profile for a dataset.
        
        Args:
            sample_vectors: Representative sample of vectors to optimize for
            target_similarity: Desired cosine similarity threshold
            target_compression: Desired compression ratio
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized CompressionProfile
        """
        # Start with balanced profile
        base_profile = ProfileManager.get_profile("balanced")
        
        best_profile = base_profile
        best_score = float('-inf')
        
        # Optimization parameters to try
        range_factors = [0.80, 0.85, 0.90, 0.95, 1.0]
        clipping_percentiles = [95.0, 97.5, 99.0, 99.5, 100.0]
        
        for range_factor in range_factors:
            for clipping_percentile in clipping_percentiles:
                # Create test profile
                test_profile = CompressionProfile(
                    name=f"optimized_{range_factor}_{clipping_percentile}",
                    strategy=CompressionStrategy.CUSTOM,
                    quantization_bits=base_profile.quantization_bits,
                    range_factor=range_factor,
                    clipping_percentile=clipping_percentile,
                    adaptive_scaling=base_profile.adaptive_scaling,
                    batch_optimization=base_profile.batch_optimization,
                    precision_mode=base_profile.precision_mode,
                    error_correction=base_profile.error_correction,
                    simd_enabled=base_profile.simd_enabled,
                    parallel_processing=base_profile.parallel_processing,
                    memory_efficient=base_profile.memory_efficient,
                    preserve_norms=base_profile.preserve_norms,
                    preserve_angles=base_profile.preserve_angles,
                    min_similarity_threshold=target_similarity
                )
                
                # Test profile performance
                score = CompressionOptimizer._evaluate_profile(
                    test_profile, sample_vectors, target_similarity, target_compression
                )
                
                if score > best_score:
                    best_score = score
                    best_profile = test_profile
        
        best_profile.name = "auto_optimized"
        return best_profile
    
    @staticmethod
    def _evaluate_profile(
        profile: CompressionProfile,
        vectors: np.ndarray,
        target_similarity: float,
        target_compression: float
    ) -> float:
        """Evaluate a compression profile performance."""
        from .quality_api import VectroQualityAnalyzer
        
        # Apply compression (simplified simulation)
        reconstructed = CompressionOptimizer._simulate_compression(vectors, profile)
        
        # Evaluate quality
        quality = VectroQualityAnalyzer.evaluate_quality(vectors, reconstructed)
        
        # Calculate score based on similarity and compression
        similarity_score = quality.mean_cosine_similarity / target_similarity
        compression_score = quality.compression_ratio / target_compression
        
        # Weighted combined score (prioritize similarity)
        score = 0.7 * similarity_score + 0.3 * compression_score
        
        # Penalty if below minimum thresholds
        if quality.mean_cosine_similarity < target_similarity:
            score *= 0.5
        if quality.compression_ratio < target_compression:
            score *= 0.8
        
        return score
    
    @staticmethod
    def _simulate_compression(
        vectors: np.ndarray, 
        profile: CompressionProfile
    ) -> np.ndarray:
        """Simulate compression with given profile."""
        reconstructed = np.zeros_like(vectors)
        
        for i in range(len(vectors)):
            vec = vectors[i]
            
            # Apply clipping if specified
            if profile.clipping_percentile < 100.0:
                clip_threshold = np.percentile(np.abs(vec), profile.clipping_percentile)
                vec = np.clip(vec, -clip_threshold, clip_threshold)
            
            # Calculate scale factor
            max_abs = np.max(np.abs(vec))
            if max_abs == 0:
                scale = 1.0
            else:
                scale = max_abs / (127.0 * profile.range_factor)
            
            # Quantize based on bits
            if profile.quantization_bits == 8:
                quantized = np.round(vec / scale).astype(np.int8)
                quantized = np.clip(quantized, -127, 127)
                reconstructed[i] = quantized.astype(np.float32) * scale
            elif profile.quantization_bits == 4:
                quantized = np.round(vec / scale).astype(np.int8)
                quantized = np.clip(quantized, -7, 7)  # 4-bit range
                reconstructed[i] = quantized.astype(np.float32) * scale
            elif profile.quantization_bits == 1:
                # Binary quantization
                binary = np.sign(vec)
                reconstructed[i] = binary * scale
            
        return reconstructed


class ProfileComparison:
    """Compare performance of different compression profiles."""
    
    @staticmethod
    def compare_profiles(
        vectors: np.ndarray,
        profile_names: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple compression profiles on a dataset."""
        from .quality_api import VectroQualityAnalyzer
        
        if profile_names is None:
            profile_names = ["fast", "balanced", "quality", "ultra"]
        
        results = {}
        
        for profile_name in profile_names:
            try:
                profile = ProfileManager.get_profile(profile_name)
                
                # Apply compression
                reconstructed = CompressionOptimizer._simulate_compression(vectors, profile)
                
                # Evaluate quality
                quality = VectroQualityAnalyzer.evaluate_quality(vectors, reconstructed)
                
                results[profile_name] = {
                    "mean_cosine_similarity": quality.mean_cosine_similarity,
                    "mean_absolute_error": quality.mean_absolute_error,
                    "compression_ratio": quality.compression_ratio,
                    "quality_grade": quality.quality_grade(),
                    "passes_threshold": quality.passes_quality_threshold(
                        profile.min_similarity_threshold
                    ),
                    "profile_strategy": profile.strategy.value,
                    "quantization_bits": profile.quantization_bits
                }
                
            except Exception as e:
                results[profile_name] = {"error": str(e)}
        
        return results
    
    @staticmethod
    def generate_comparison_report(
        comparison_results: Dict[str, Dict[str, float]],
        title: str = "Profile Comparison Report"
    ) -> str:
        """Generate a formatted comparison report."""
        report = f"""
{title}
{'=' * len(title)}

Profile Performance Comparison:

{'Profile':<12} {'Similarity':<10} {'MAE':<12} {'Compression':<12} {'Grade':<15} {'Pass':<6}
{'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*15} {'-'*6}
"""
        
        for profile_name, metrics in comparison_results.items():
            if "error" in metrics:
                report += f"{profile_name:<12} ERROR: {metrics['error']}\n"
                continue
                
            report += f"{profile_name:<12} "
            report += f"{metrics['mean_cosine_similarity']:.6f} "
            report += f"{metrics['mean_absolute_error']:.6f}   "
            report += f"{metrics['compression_ratio']:.2f}x        "
            report += f"{metrics['quality_grade']:<15} "
            report += f"{'✅' if metrics['passes_threshold'] else '❌':<6}\n"
        
        return report


# Convenience functions
def get_compression_profile(name: str) -> CompressionProfile:
    """Get a compression profile by name."""
    return ProfileManager.get_profile(name)


def create_custom_profile(
    name: str,
    quantization_bits: int = 8,
    range_factor: float = 0.95,
    **kwargs
) -> CompressionProfile:
    """Create a custom compression profile with reasonable defaults."""
    defaults = {
        "strategy": CompressionStrategy.CUSTOM,
        "clipping_percentile": 99.0,
        "adaptive_scaling": True,
        "batch_optimization": True,
        "precision_mode": "int8",
        "error_correction": True,
        "simd_enabled": True,
        "parallel_processing": True,
        "memory_efficient": True,
        "preserve_norms": True,
        "preserve_angles": False,
        "min_similarity_threshold": 0.995
    }
    
    # Override defaults with provided kwargs
    defaults.update(kwargs)
    
    return CompressionProfile(
        name=name,
        quantization_bits=quantization_bits,
        range_factor=range_factor,
        **defaults
    )


# Example usage
if __name__ == "__main__":
    print("Vectro Compression Profiles API Demo")
    print("=" * 42)
    
    # Initialize profiles
    ProfileManager.initialize_builtin_profiles()
    
    # List available profiles
    profiles = ProfileManager.list_profiles()
    print(f"Available profiles: {', '.join(profiles)}")
    
    # Create test vectors
    np.random.seed(42)
    vectors = np.random.randn(1000, 384).astype(np.float32)
    
    # Compare profiles
    comparison = ProfileComparison.compare_profiles(vectors)
    report = ProfileComparison.generate_comparison_report(comparison)
    print(report)
    
    # Create custom profile
    custom_profile = create_custom_profile(
        "my_custom",
        quantization_bits=6,
        range_factor=0.92
    )
    print(f"\nCreated custom profile: {custom_profile.name}")
    print(f"  Strategy: {custom_profile.strategy.value}")
    print(f"  Quantization bits: {custom_profile.quantization_bits}")
    print(f"  Range factor: {custom_profile.range_factor}")