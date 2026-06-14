"""Tests for Vectro.compress_async() and Vectro.decompress_async()."""

from __future__ import annotations

import asyncio

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.vectro import Vectro  # noqa: E402
from python.batch_api import BatchQuantizationResult  # noqa: E402
from python.interface import QuantizationResult  # noqa: E402

RNG = np.random.default_rng(42)


def _batch(n: int = 64, d: int = 128) -> np.ndarray:
    return RNG.standard_normal((n, d)).astype(np.float32)


def _single(d: int = 128) -> np.ndarray:
    return RNG.standard_normal(d).astype(np.float32)


# ---------------------------------------------------------------------------
# compress_async
# ---------------------------------------------------------------------------


class TestCompressAsync:
    def test_batch_compress_async(self):
        vectro = Vectro()
        vectors = _batch()

        async def _run():
            return await vectro.compress_async(vectors)

        result = asyncio.run(_run())
        assert isinstance(result, BatchQuantizationResult)
        assert result.batch_size == 64

    def test_single_vector_compress_async(self):
        vectro = Vectro()
        vec = _single()

        async def _run():
            return await vectro.compress_async(vec)

        result = asyncio.run(_run())
        assert isinstance(result, QuantizationResult)

    def test_async_profile_forwarded(self):
        vectro = Vectro()
        vectors = _batch()

        async def _run():
            return await vectro.compress_async(vectors, profile="quality")

        result = asyncio.run(_run())
        assert result.batch_size == 64

    def test_async_precision_mode_forwarded(self):
        vectro = Vectro()
        vectors = _batch(n=16, d=64)

        async def _run():
            return await vectro.compress_async(vectors, precision_mode="int8")

        result = asyncio.run(_run())
        assert result.precision_mode == "int8"

    def test_async_result_matches_sync(self):
        """Async and sync compress must produce numerically equivalent results."""
        vectro = Vectro()
        vectors = _batch(n=32, d=64)

        async def _run():
            return await vectro.compress_async(vectors, profile="fast")

        async_result = asyncio.run(_run())
        sync_result = vectro.compress(vectors, profile="fast")

        # Same shapes
        assert async_result.batch_size == sync_result.batch_size
        assert async_result.vector_dim == sync_result.vector_dim
        assert async_result.precision_mode == sync_result.precision_mode

    def test_async_model_dir_forwarded(self):
        from pathlib import Path

        vectro = Vectro()
        vectors = _batch(n=16, d=64)
        model_dir = str(Path(__file__).parent / "fixtures" / "gte")

        async def _run():
            return await vectro.compress_async(vectors, model_dir=model_dir)

        result = asyncio.run(_run())
        assert result.precision_mode == "int8"

    def test_concurrent_compress(self):
        """Multiple concurrent compress_async calls must not corrupt results."""
        vectro = Vectro()
        batches = [_batch(n=16, d=32) for _ in range(4)]

        async def _run():
            return await asyncio.gather(*[vectro.compress_async(b) for b in batches])

        results = asyncio.run(_run())
        assert len(results) == 4
        for r in results:
            assert isinstance(r, BatchQuantizationResult)
            assert r.batch_size == 16


# ---------------------------------------------------------------------------
# decompress_async
# ---------------------------------------------------------------------------


class TestDecompressAsync:
    def test_batch_decompress_async(self):
        vectro = Vectro()
        vectors = _batch()
        result = vectro.compress(vectors)

        async def _run():
            return await vectro.decompress_async(result)

        restored = asyncio.run(_run())
        assert restored.shape == vectors.shape
        assert restored.dtype == np.float32

    def test_single_decompress_async(self):
        vectro = Vectro()
        vec = _single()
        result = vectro.compress(vec)

        async def _run():
            return await vectro.decompress_async(result)

        restored = asyncio.run(_run())
        assert restored.ndim == 1
        assert restored.shape[0] == vec.shape[0]

    def test_roundtrip_cosine_async(self):
        """Async roundtrip must preserve INT8 cosine similarity floor (≥ 0.9999)."""
        vectro = Vectro()
        rng = np.random.default_rng(99)
        vectors = rng.standard_normal((32, 128)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        async def _run():
            result = await vectro.compress_async(vectors, precision_mode="int8")
            return await vectro.decompress_async(result)

        restored = asyncio.run(_run())

        norms_a = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        norms_b = np.linalg.norm(restored, axis=1, keepdims=True) + 1e-10
        cosines = ((vectors / norms_a) * (restored / norms_b)).sum(axis=1)
        assert float(cosines.mean()) >= 0.9999, (
            f"INT8 async roundtrip cosine {cosines.mean():.6f} < 0.9999 floor"
        )

    def test_concurrent_compress_and_decompress(self):
        vectro = Vectro()
        vecs_list = [_batch(n=8, d=32) for _ in range(4)]

        async def _run_one(v):
            r = await vectro.compress_async(v)
            return await vectro.decompress_async(r)

        async def _run():
            return await asyncio.gather(*[_run_one(v) for v in vecs_list])

        results = asyncio.run(_run())
        assert len(results) == 4
        for orig, restored in zip(vecs_list, results):
            assert orig.shape == restored.shape
