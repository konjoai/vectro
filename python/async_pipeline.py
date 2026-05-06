"""Async compression pipeline for vectro — v5.2.0.

compress_async(): runs Vectro.compress() in asyncio thread pool.
CompressionPipeline: chains multiple quantization stages in sequence.
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """One stage in a compression pipeline."""
    mode: str
    profile: Optional[str] = None
    group_size: Optional[int] = None

    def __post_init__(self):
        if not self.mode:
            raise ValueError("PipelineStage.mode must be non-empty")


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""
    stages: List[str]
    input_shape: tuple
    output_shape: tuple
    input_dtype: str
    output_dtype: str
    total_latency_ms: float
    stage_latencies_ms: List[float]
    compression_ratio: float


async def compress_async(vectro, vectors: np.ndarray, mode: str = "int8", **kwargs) -> Any:
    """Run vectro.compress() in a thread pool (non-blocking for the event loop)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: vectro.compress(vectors, precision_mode=mode, **kwargs),
    )


class CompressionPipeline:
    """Chain multiple vectro compression stages in sequence.

    Each stage operates on the output of the previous stage.
    """

    def __init__(self, stages: List[PipelineStage], vectro=None):
        if not stages:
            raise ValueError("Pipeline must have at least one stage")
        self._stages = stages
        self._vectro = vectro

    @property
    def stages(self) -> List[PipelineStage]:
        return self._stages

    def _get_vectro(self):
        if self._vectro is not None:
            return self._vectro
        from .vectro import Vectro
        return Vectro()

    def run(self, vectors: np.ndarray) -> tuple:
        """Run all stages synchronously. Returns (PipelineResult, final_array)."""
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        vectro = self._get_vectro()
        current = vectors.astype(np.float32)
        input_shape = current.shape
        input_bytes = current.nbytes
        stage_latencies: List[float] = []
        stage_names: List[str] = []

        for stage in self._stages:
            kwargs: dict = {}
            if stage.profile is not None:
                kwargs["profile"] = stage.profile
            if stage.group_size is not None:
                kwargs["group_size"] = stage.group_size
            t0 = time.perf_counter()
            result = vectro.compress(current, precision_mode=stage.mode, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            stage_latencies.append(elapsed_ms)
            stage_names.append(stage.mode)
            # compress() may return (array, metadata) or just array
            if isinstance(result, tuple):
                current = result[0]
            else:
                current = result
            if hasattr(current, 'astype') and current.dtype != np.float32:
                try:
                    current = current.astype(np.float32)
                except (ValueError, TypeError):
                    pass
            logger.info("Pipeline stage '%s': %.1f ms", stage.mode, elapsed_ms)

        output_bytes = current.nbytes if hasattr(current, 'nbytes') else input_bytes
        pr = PipelineResult(
            stages=stage_names,
            input_shape=input_shape,
            output_shape=current.shape if hasattr(current, 'shape') else (0,),
            input_dtype=str(vectors.dtype),
            output_dtype=str(current.dtype) if hasattr(current, 'dtype') else "unknown",
            total_latency_ms=sum(stage_latencies),
            stage_latencies_ms=stage_latencies,
            compression_ratio=input_bytes / max(output_bytes, 1),
        )
        return pr, current

    async def run_async(self, vectors: np.ndarray) -> tuple:
        """Run the pipeline asynchronously (thread pool)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run, vectors)
