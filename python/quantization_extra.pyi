import numpy as np
from .interface import QuantizationResult as QuantizationResult
from typing import Optional

_INT2_LEVELS: int
_INT2_MAX: int

def quantize_int2(
    embeddings: np.ndarray, group_size: int = 32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def _pack_int2(q: np.ndarray, d: int) -> np.ndarray: ...
def _unpack_int2(packed: np.ndarray, n_elements: int) -> np.ndarray: ...
def dequantize_int2(
    packed: np.ndarray,
    scales: np.ndarray,
    zeroes: np.ndarray,
    group_size: int = 32,
    vector_dim: Optional[int] = None,
) -> np.ndarray: ...
def quantize_adaptive(
    embeddings: np.ndarray, bits: int = 8, clip_ratio: float = 3.0, group_size: int = 0
) -> QuantizationResult: ...
