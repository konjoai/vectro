"""Tests for async_pipeline — v5.2.0."""
import asyncio
import numpy as np
import pytest

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.async_pipeline import (
    PipelineStage, PipelineResult, CompressionPipeline, compress_async
)


def test_stage_valid():
    s = PipelineStage(mode="int8")
    assert s.mode == "int8"

def test_stage_empty_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        PipelineStage(mode="")

def test_stage_profile():
    s = PipelineStage(mode="int8", profile="speed")
    assert s.profile == "speed"

def test_pipeline_empty_raises():
    with pytest.raises(ValueError, match="stage"):
        CompressionPipeline([])

def test_pipeline_stages():
    p = CompressionPipeline([PipelineStage(mode="int8")])
    assert len(p.stages) == 1

def test_pipeline_run_int8():
    from python.vectro import Vectro
    vectro = Vectro()
    pipeline = CompressionPipeline([PipelineStage(mode="int8")], vectro=vectro)
    vectors = np.random.randn(10, 64).astype(np.float32)
    result, output = pipeline.run(vectors)
    assert output is not None
    assert result.stages == ["int8"]
    assert result.input_shape == (10, 64)

def test_pipeline_1d_raises():
    from python.vectro import Vectro
    pipeline = CompressionPipeline([PipelineStage(mode="int8")], vectro=Vectro())
    with pytest.raises(ValueError, match="2-D"):
        pipeline.run(np.ones(64))

def test_pipeline_latency_nonneg():
    from python.vectro import Vectro
    pipeline = CompressionPipeline([PipelineStage(mode="int8")], vectro=Vectro())
    result, _ = pipeline.run(np.random.randn(10, 64).astype(np.float32))
    assert result.total_latency_ms >= 0.0

def test_pipeline_compression_ratio():
    from python.vectro import Vectro
    pipeline = CompressionPipeline([PipelineStage(mode="int8")], vectro=Vectro())
    result, _ = pipeline.run(np.random.randn(10, 64).astype(np.float32))
    assert result.compression_ratio > 0.0

def test_pipeline_result_type():
    from python.vectro import Vectro
    pipeline = CompressionPipeline([PipelineStage(mode="int8")], vectro=Vectro())
    result, _ = pipeline.run(np.random.randn(5, 32).astype(np.float32))
    assert isinstance(result, PipelineResult)

def test_pipeline_input_dtype():
    from python.vectro import Vectro
    pipeline = CompressionPipeline([PipelineStage(mode="int8")], vectro=Vectro())
    result, _ = pipeline.run(np.random.randn(5, 32).astype(np.float32))
    assert result.input_dtype == "float32"

def test_pipeline_run_async():
    from python.vectro import Vectro
    async def _run():
        pipeline = CompressionPipeline([PipelineStage(mode="int8")], vectro=Vectro())
        return await pipeline.run_async(np.random.randn(10, 64).astype(np.float32))
    result, output = asyncio.run(_run())
    assert result.stages == ["int8"]

def test_compress_async_basic():
    from python.vectro import Vectro
    async def _run():
        return await compress_async(Vectro(), np.random.randn(10, 64).astype(np.float32), mode="int8")
    output = asyncio.run(_run())
    assert output is not None

def test_pipeline_stage_count():
    from python.vectro import Vectro
    pipeline = CompressionPipeline([PipelineStage(mode="int8")], vectro=Vectro())
    result, _ = pipeline.run(np.random.randn(5, 32).astype(np.float32))
    assert len(result.stage_latencies_ms) == 1
