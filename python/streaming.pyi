import asyncio
import numpy as np
from .batch_api import BatchQuantizationResult as BatchQuantizationResult
from .interface import (
    QuantizationResult as QuantizationResult,
    dequantize_int4 as dequantize_int4,
    reconstruct_embeddings as reconstruct_embeddings,
)
from _typeshed import Incomplete
from typing import Generator, Iterator

class StreamingDecompressor:
    _result: Incomplete
    _chunk_size: Incomplete
    _backend: Incomplete
    def __init__(
        self,
        result: BatchQuantizationResult | QuantizationResult,
        chunk_size: int = 1000,
        backend: str = "auto",
    ) -> None: ...
    def __iter__(self) -> Iterator[np.ndarray]: ...
    def __len__(self) -> int: ...
    def _iter_chunks(self) -> Generator[np.ndarray, None, None]: ...
    def _iter_batch(self, result: BatchQuantizationResult) -> Generator[np.ndarray, None, None]: ...
    def _iter_quant_result(
        self, result: QuantizationResult
    ) -> Generator[np.ndarray, None, None]: ...

def _apply_grouped_scales(
    quantized: np.ndarray, scales: np.ndarray, vector_dim: int
) -> np.ndarray: ...

class AsyncStreamingDecompressor:
    _SENTINEL: Incomplete
    _result: Incomplete
    _chunk_size: Incomplete
    _backend: Incomplete
    _queue_size: Incomplete
    _queue: asyncio.Queue | None
    def __init__(
        self,
        result: BatchQuantizationResult | QuantizationResult,
        chunk_size: int = 1000,
        backend: str = "auto",
        queue_size: int = 4,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __aiter__(self) -> AsyncStreamingDecompressor: ...
    async def __anext__(self) -> np.ndarray: ...
