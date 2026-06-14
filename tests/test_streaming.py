"""Tests for python/streaming.py — StreamingDecompressor."""

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.batch_api import BatchQuantizationResult  # noqa: E402
from python.interface import QuantizationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_int8_batch(n: int = 50, dim: int = 32) -> BatchQuantizationResult:
    """Create a simple INT8 BatchQuantizationResult for testing."""
    rng = np.random.default_rng(7)
    floats = rng.standard_normal((n, dim)).astype(np.float32)
    scales = np.max(np.abs(floats), axis=1) / 127.0
    q = np.clip(np.round(floats / scales[:, None]), -127, 127).astype(np.int8)
    original = n * dim * 4
    compressed = q.nbytes + scales.nbytes
    return BatchQuantizationResult(
        quantized_vectors=list(q),
        scales=scales,
        batch_size=n,
        vector_dim=dim,
        compression_ratio=original / compressed,
        total_original_bytes=original,
        total_compressed_bytes=compressed,
    )


def _make_int8_quant_result(n: int = 50, dim: int = 32) -> QuantizationResult:
    """Create a QuantizationResult (single-block) for testing."""
    rng = np.random.default_rng(13)
    floats = rng.standard_normal((n, dim)).astype(np.float32)
    scales = np.max(np.abs(floats), axis=1) / 127.0
    q = np.clip(np.round(floats / scales[:, None]), -127, 127).astype(np.int8)
    return QuantizationResult(quantized=q, scales=scales, dims=dim, n=n)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamingDecompressorInit(unittest.TestCase):
    def test_invalid_chunk_size_raises(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_batch()
        with self.assertRaises(ValueError):
            StreamingDecompressor(result, chunk_size=0)

    def test_chunk_size_one_allowed(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_batch(n=5)
        sd = StreamingDecompressor(result, chunk_size=1)
        self.assertIsNotNone(sd)


class TestStreamingDecompressorLen(unittest.TestCase):
    def test_len_batch_result(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_batch(n=37)
        sd = StreamingDecompressor(result, chunk_size=10)
        self.assertEqual(len(sd), 37)

    def test_len_quant_result(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_quant_result(n=22)
        sd = StreamingDecompressor(result, chunk_size=10)
        self.assertEqual(len(sd), 22)


class TestStreamingDecompressorChunks(unittest.TestCase):
    def test_total_vectors_match(self):
        """All chunks together reconstruct exactly n vectors."""
        from python.streaming import StreamingDecompressor

        n, dim, chunk = 50, 32, 13
        result = _make_int8_batch(n=n, dim=dim)
        sd = StreamingDecompressor(result, chunk_size=chunk)
        total = sum(len(c) for c in sd)
        self.assertEqual(total, n)

    def test_chunk_shape(self):
        """Each chunk has shape (≤chunk_size, dim)."""
        from python.streaming import StreamingDecompressor

        n, dim, chunk_size = 27, 16, 10
        result = _make_int8_batch(n=n, dim=dim)
        for chunk in StreamingDecompressor(result, chunk_size=chunk_size):
            self.assertEqual(chunk.ndim, 2)
            self.assertEqual(chunk.shape[1], dim)
            self.assertLessEqual(chunk.shape[0], chunk_size)

    def test_last_chunk_correct_size(self):
        """Last chunk contains the remainder (n % chunk_size)."""
        from python.streaming import StreamingDecompressor

        n, dim, chunk_size = 25, 16, 7  # remainder = 4
        result = _make_int8_batch(n=n, dim=dim)
        chunks = list(StreamingDecompressor(result, chunk_size=chunk_size))
        self.assertEqual(len(chunks[-1]), n % chunk_size)

    def test_single_chunk_when_n_lt_chunk_size(self):
        """When n < chunk_size we get exactly one chunk."""
        from python.streaming import StreamingDecompressor

        n, dim = 5, 8
        result = _make_int8_batch(n=n, dim=dim)
        chunks = list(StreamingDecompressor(result, chunk_size=100))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].shape[0], n)

    def test_chunk_dtype_float32(self):
        """Reconstructed chunks are float32."""
        from python.streaming import StreamingDecompressor

        result = _make_int8_batch(n=10, dim=8)
        for chunk in StreamingDecompressor(result, chunk_size=5):
            self.assertEqual(chunk.dtype, np.float32)

    def test_reconstruction_accuracy(self):
        """Reconstructed values are approximately correct (INT8 quantization error)."""
        from python.streaming import StreamingDecompressor

        n, dim = 20, 32
        rng = np.random.default_rng(42)
        floats = rng.standard_normal((n, dim)).astype(np.float32)
        scales = np.max(np.abs(floats), axis=1) / 127.0
        q = np.clip(np.round(floats / scales[:, None]), -127, 127).astype(np.int8)
        original = n * dim * 4
        compressed = q.nbytes + scales.nbytes
        result = BatchQuantizationResult(
            quantized_vectors=list(q),
            scales=scales,
            batch_size=n,
            vector_dim=dim,
            compression_ratio=original / compressed,
            total_original_bytes=original,
            total_compressed_bytes=compressed,
        )
        recon = np.vstack(list(StreamingDecompressor(result, chunk_size=7)))

        dot = np.sum(floats * recon, axis=1)
        norms = np.linalg.norm(floats, axis=1) * np.linalg.norm(recon, axis=1)
        cos_sim = np.mean(dot / (norms + 1e-10))
        self.assertGreater(cos_sim, 0.99)

    def test_iterator_reuse(self):
        """Iterating twice produces the same total count."""
        from python.streaming import StreamingDecompressor

        result = _make_int8_batch(n=20, dim=8)
        sd = StreamingDecompressor(result, chunk_size=7)
        first_total = sum(len(c) for c in sd)
        second_total = sum(len(c) for c in sd)
        self.assertEqual(first_total, second_total)


class TestStreamingDecompressorQuantResult(unittest.TestCase):
    """Test the QuantizationResult path."""

    def test_len_quant_result(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_quant_result(n=15, dim=8)
        self.assertEqual(len(StreamingDecompressor(result)), 15)

    def test_total_vectors_quant_result(self):
        from python.streaming import StreamingDecompressor

        n = 15
        result = _make_int8_quant_result(n=n, dim=8)
        total = sum(len(c) for c in StreamingDecompressor(result, chunk_size=6))
        self.assertEqual(total, n)

    def test_chunk_dtype_quant_result(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_quant_result(n=10, dim=8)
        for chunk in StreamingDecompressor(result, chunk_size=4):
            self.assertEqual(chunk.dtype, np.float32)


# ---------------------------------------------------------------------------
# AsyncStreamingDecompressor tests
# ---------------------------------------------------------------------------


class TestAsyncStreamingDecompressor(unittest.TestCase):
    """Tests for AsyncStreamingDecompressor using asyncio.run()."""

    def _run(self, coro):
        """Helper: run a coroutine synchronously."""
        import asyncio

        return asyncio.run(coro)

    # ── BatchQuantizationResult path ──────────────────────────────────────

    def test_total_vectors_int8_batch(self):
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_batch(n=64, dim=16)

        async def go():
            count = 0
            async for chunk in AsyncStreamingDecompressor(result, chunk_size=10):
                count += len(chunk)
            return count

        self.assertEqual(self._run(go()), 64)

    def test_chunk_dtype_batch(self):
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_batch(n=20, dim=8)

        async def go():
            dtypes = []
            async for chunk in AsyncStreamingDecompressor(result, chunk_size=7):
                dtypes.append(chunk.dtype)
            return dtypes

        for dt in self._run(go()):
            self.assertEqual(dt, np.float32)

    def test_no_nan_batch(self):
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_batch(n=32, dim=8)

        async def go():
            chunks = []
            async for chunk in AsyncStreamingDecompressor(result, chunk_size=8):
                chunks.append(chunk)
            return np.vstack(chunks)

        out = self._run(go())
        self.assertFalse(np.isnan(out).any())

    # ── QuantizationResult path ───────────────────────────────────────────

    def test_total_vectors_quant_result(self):
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_quant_result(n=50, dim=16)

        async def go():
            count = 0
            async for chunk in AsyncStreamingDecompressor(result, chunk_size=12):
                count += len(chunk)
            return count

        self.assertEqual(self._run(go()), 50)

    def test_chunk_dtype_quant_result(self):
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_quant_result(n=10, dim=4)

        async def go():
            dtypes = []
            async for chunk in AsyncStreamingDecompressor(result, chunk_size=3):
                dtypes.append(chunk.dtype)
            return dtypes

        for dt in self._run(go()):
            self.assertEqual(dt, np.float32)

    def test_no_nan_quant_result(self):
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_quant_result(n=20, dim=8)

        async def go():
            chunks = []
            async for chunk in AsyncStreamingDecompressor(result, chunk_size=5):
                chunks.append(chunk)
            return np.vstack(chunks)

        out = self._run(go())
        self.assertFalse(np.isnan(out).any())

    def test_single_chunk_when_n_lt_chunk_size(self):
        """When n < chunk_size the entire result arrives in one chunk."""
        from python.streaming import AsyncStreamingDecompressor

        n, dim = 5, 8
        result = _make_int8_quant_result(n=n, dim=dim)

        async def go():
            chunks = []
            async for chunk in AsyncStreamingDecompressor(result, chunk_size=100):
                chunks.append(chunk)
            return chunks

        chunks = self._run(go())
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), n)

    def test_single_vector(self):
        """n=1 should yield exactly one chunk of length 1."""
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_quant_result(n=1, dim=4)

        async def go():
            chunks = []
            async for chunk in AsyncStreamingDecompressor(result, chunk_size=1):
                chunks.append(chunk)
            return chunks

        chunks = self._run(go())
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].shape, (1, 4))

    def test_len_matches_n(self):
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_quant_result(n=37, dim=8)
        asd = AsyncStreamingDecompressor(result, chunk_size=10)
        self.assertEqual(len(asd), 37)

    def test_invalid_chunk_size_raises(self):
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_quant_result(n=10, dim=4)
        with self.assertRaises(ValueError):
            AsyncStreamingDecompressor(result, chunk_size=0)

    def test_invalid_queue_size_raises(self):
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_quant_result(n=10, dim=4)
        with self.assertRaises(ValueError):
            AsyncStreamingDecompressor(result, queue_size=0)

    def test_sequential_runs_independent(self):
        """An AsyncStreamingDecompressor can be re-iterated without state leak."""
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_quant_result(n=16, dim=4)

        async def go():
            asd = AsyncStreamingDecompressor(result, chunk_size=4)
            c1 = sum(len(c) for c in [chunk async for chunk in asd])
            # A new iteration via __aiter__ should work identically.
            c2 = 0
            async for chunk in AsyncStreamingDecompressor(result, chunk_size=4):
                c2 += len(chunk)
            return c1, c2

        c1, c2 = self._run(go())
        self.assertEqual(c1, 16)
        self.assertEqual(c2, 16)

    def test_reconstruction_values_plausible(self):
        """Reconstructed vectors should be within float32 range and finite."""
        from python.streaming import AsyncStreamingDecompressor

        result = _make_int8_quant_result(n=30, dim=16)

        async def go():
            chunks = []
            async for chunk in AsyncStreamingDecompressor(result, chunk_size=10):
                chunks.append(chunk)
            return np.vstack(chunks)

        out = self._run(go())
        self.assertEqual(out.shape, (30, 16))
        self.assertTrue(np.all(np.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
