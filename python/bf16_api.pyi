"""Type stubs for python/bf16_api.py — Bf16Encoder."""

from __future__ import annotations

from typing import List
import numpy as np

class Bf16Encoder:
    """BFloat16 encoder for lossless (within BF16 precision) vector compression.

    Stores vectors internally as BF16, roughly halving the memory footprint
    compared to float32 while preserving cosine similarity ≥ 0.9999.

    Examples
    --------
    >>> enc = Bf16Encoder()
    >>> enc.encode([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> decoded = enc.decode()
    >>> len(decoded)
    2
    """

    def __init__(self) -> None: ...
    def encode(self, vectors: List[List[float]]) -> None:
        """Encode and store ``vectors`` as BF16.

        Parameters
        ----------
        vectors : list[list[float]]
            Each row is a float32-compatible embedding vector.
        """
        ...

    def encode_np(self, array: np.ndarray) -> None:
        """Encode from a 2-D float32 ndarray of shape ``(n, dim)``.

        The array is cast to C-contiguous float32 internally before encoding.
        """
        ...

    def decode(self) -> List[List[float]]:
        """Decode all stored vectors back to float32.

        Returns
        -------
        list[list[float]]
            Reconstructed vectors; same shape as the input to :meth:`encode`.
        """
        ...

    def cosine_dist(self, i: int, j: int) -> float:
        """Cosine *distance* between stored vector ``i`` and vector ``j``.

        Returns a value in ``[0, 2]`` where ``0`` means identical direction.
        """
        ...

    def __len__(self) -> int:
        """Number of vectors currently stored."""
        ...

    def __repr__(self) -> str: ...
