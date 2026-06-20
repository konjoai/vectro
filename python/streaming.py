"""Streaming decompression pipeline for large compressed vector sets.

Provides a memory-efficient iterator that reconstructs float32 vectors
from a compressed artifact one chunk at a time — no need to load the
entire decompressed dataset into RAM simultaneously.

Usage (synchronous)::

    vectro = Vectro()
    result = vectro.compress(large_matrix, profile="balanced")

    for chunk in StreamingDecompressor(result, chunk_size=500):
        # process 500 float32 vectors at a time
        process(chunk)

Usage (async)::

    async def ingest(result):
        async for chunk in AsyncStreamingDecompressor(result, chunk_size=128):
            await send_to_downstream(chunk)
"""

from __future__ import annotations

import asyncio
import threading
from typing import Generator, Iterator, Optional, Union

import numpy as np

from .interface import QuantizationResult, dequantize_int4
from .batch_api import BatchQuantizationResult


class StreamingDecompressor:
    """Memory-efficient streaming decompressor for compressed vector batches.

    Iterates over a :class:`~vectro.BatchQuantizationResult` (or
    :class:`~vectro.interface.QuantizationResult`) in fixed-size chunks,
    reconstructing float32 vectors on demand rather than all at once.

    Args:
        result: Compressed artifact from :meth:`~vectro.Vectro.compress`.
        chunk_size: Number of vectors to reconstruct per iteration step.
        backend: Reconstruction backend (``"auto"`` = fastest available).

    Example::

        compressor = Vectro()
        compressed = compressor.compress(embeddings, profile="quality")

        total = 0
        for chunk in StreamingDecompressor(compressed, chunk_size=256):
            total += len(chunk)          # chunk: np.ndarray (≤256, dim)
        assert total == len(embeddings)
    """

    def __init__(
        self,
        result: Union[BatchQuantizationResult, QuantizationResult],
        chunk_size: int = 1000,
        backend: str = "auto",
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be ≥ 1")
        self._result = result
        self._chunk_size = chunk_size
        self._backend = backend

    # ------------------------------------------------------------------
    # Python iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        return self._iter_chunks()

    def __len__(self) -> int:
        """Total number of vectors in the compressed artifact."""
        if isinstance(self._result, BatchQuantizationResult):
            return self._result.batch_size
        return self._result.n

    def _iter_chunks(self) -> Generator[np.ndarray, None, None]:
        result = self._result

        if isinstance(result, BatchQuantizationResult):
            yield from self._iter_batch(result)
        else:
            yield from self._iter_quant_result(result)

    # ------------------------------------------------------------------
    # BatchQuantizationResult path
    # ------------------------------------------------------------------

    def _iter_batch(self, result: BatchQuantizationResult) -> Generator[np.ndarray, None, None]:
        n = result.batch_size
        precision_mode = result.precision_mode
        group_size = result.group_size or 64

        # Build a contiguous array view once so slicing is cheap
        q_all = np.asarray(
            result.quantized_vectors,
            dtype=np.uint8 if precision_mode == "int4" else np.int8,
        )  # (n, dim_q)
        s_all = np.asarray(result.scales, dtype=np.float32)  # (n,) or (n, n_groups)

        for start in range(0, n, self._chunk_size):
            end = min(start + self._chunk_size, n)
            q_chunk = q_all[start:end]
            s_chunk = s_all[start:end]

            if precision_mode == "int4":
                yield dequantize_int4(q_chunk, s_chunk, group_size=group_size)
            else:
                # Vectorised broadcast: (chunk, dim) × (chunk, 1) or (chunk,)
                if s_chunk.ndim == 1:
                    yield q_chunk.astype(np.float32) * s_chunk[:, np.newaxis]
                else:
                    # grouped scales — repeat each group scale across group elements
                    yield _apply_grouped_scales(q_chunk, s_chunk, result.vector_dim)

    # ------------------------------------------------------------------
    # QuantizationResult path
    # ------------------------------------------------------------------

    def _iter_quant_result(self, result: QuantizationResult) -> Generator[np.ndarray, None, None]:
        n = result.n
        precision_mode = getattr(result, "precision_mode", "int8")
        group_size = getattr(result, "group_size", 0) or 64

        q_all = np.asarray(result.quantized)
        s_all = np.asarray(result.scales, dtype=np.float32)

        for start in range(0, n, self._chunk_size):
            end = min(start + self._chunk_size, n)
            q_chunk = q_all[start:end]
            s_chunk = s_all[start:end]

            if precision_mode == "int4":
                yield dequantize_int4(q_chunk, s_chunk, group_size=group_size)
            else:
                if s_chunk.ndim == 1:
                    yield q_chunk.astype(np.float32) * s_chunk[:, np.newaxis]
                else:
                    yield _apply_grouped_scales(q_chunk, s_chunk, result.dims)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _apply_grouped_scales(
    quantized: np.ndarray,
    scales: np.ndarray,
    vector_dim: int,
) -> np.ndarray:
    """Broadcast grouped scales back to per-element form and dequantize.

    Args:
        quantized: ``(n, d)`` int8 array.
        scales:    ``(n, n_groups)`` float32 scale array.
        vector_dim: Original (full) vector dimension ``d``.

    Returns:
        ``(n, d)`` float32 reconstructed vectors.
    """
    n, d = quantized.shape
    n_groups = scales.shape[1]
    group_size = vector_dim // n_groups
    # Repeat each scale across its group
    scales_expanded = np.repeat(scales, group_size, axis=1)[:, :d]
    return quantized.astype(np.float32) * scales_expanded


# ---------------------------------------------------------------------------
# Async streaming decompressor
# ---------------------------------------------------------------------------


class AsyncStreamingDecompressor:
    """Async iterator that wraps :class:`StreamingDecompressor` with backpressure.

    Reconstruction (numpy) runs in a background thread so the event loop is
    never blocked.  A bounded :class:`asyncio.Queue` between producer and
    consumer limits peak memory by stalling the producer when the consumer
    falls behind (backpressure).

    Args:
        result:     Compressed artifact (``QuantizationResult`` or
                    ``BatchQuantizationResult``).
        chunk_size: Number of vectors per chunk (default: 1000).
        backend:    Reconstruction backend (``"auto"`` = fastest available).
        queue_size: Maximum number of pre-computed chunks to buffer (default: 4).

    Example::

        async def ingest(result):
            async for chunk in AsyncStreamingDecompressor(result, chunk_size=128):
                await send(chunk)          # chunk: np.ndarray shape (≤chunk_size, dim)
    """

    _SENTINEL = object()  # poison-pill sentinel; never yielded to the caller

    def __init__(
        self,
        result: Union["BatchQuantizationResult", "QuantizationResult"],
        chunk_size: int = 1000,
        backend: str = "auto",
        queue_size: int = 4,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be ≥ 1")
        if queue_size < 1:
            raise ValueError("queue_size must be ≥ 1")
        self._result = result
        self._chunk_size = chunk_size
        self._backend = backend
        self._queue_size = queue_size
        self._queue: Optional[asyncio.Queue] = None

    # ------------------------------------------------------------------
    # Sizing helper
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of vectors in the compressed artifact."""
        return len(StreamingDecompressor(self._result, self._chunk_size))

    # ------------------------------------------------------------------
    # Async iterator protocol
    # ------------------------------------------------------------------

    def __aiter__(self) -> "AsyncStreamingDecompressor":
        loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue(maxsize=self._queue_size)

        def _produce() -> None:
            sync = StreamingDecompressor(
                self._result,
                chunk_size=self._chunk_size,
                backend=self._backend,
            )
            try:
                for chunk in sync:
                    # Block the producer thread if the queue is full (backpressure).
                    asyncio.run_coroutine_threadsafe(self._queue.put(chunk), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(self._queue.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(self._queue.put(self._SENTINEL), loop).result()

        thread = threading.Thread(target=_produce, daemon=True)
        thread.start()
        return self

    async def __anext__(self) -> np.ndarray:
        assert self._queue is not None, "__aiter__ must be called before __anext__"
        item = await self._queue.get()
        if item is self._SENTINEL:
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item
