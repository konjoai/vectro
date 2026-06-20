"""BFloat16 encoder for memory-efficient vector storage (v7.0.0).

Thin Python wrapper around the Rust ``PyBf16Encoder`` PyO3 binding.
BF16 provides a 2× memory reduction over FP32 with minimal cosine-similarity
degradation (cosine ≥ 0.9999 on typical embedding distributions).

Public API
----------
Bf16Encoder()
    .encode(vectors)             -> None
    .encode_np(array)            -> None
    .decode()                    -> list[list[float]]
    .cosine_dist(i, j)           -> float
    len(encoder)                 -> int
"""

from __future__ import annotations

from typing import List

import numpy as np

try:
    from vectro_py import PyBf16Encoder as _PyBf16Encoder  # type: ignore

    _BINDINGS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BINDINGS_AVAILABLE = False
    _PyBf16Encoder = None  # type: ignore[assignment,misc]


class Bf16Encoder:
    """BFloat16 vector store with SimSIMD-accelerated cosine distance.

    Stores vectors in BFloat16 (2 bytes per dimension), halving memory
    consumption compared to FP32 whilst preserving ranking quality.

    Example
    -------
    >>> enc = Bf16Encoder()
    >>> enc.encode([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> enc.decode()
    [[...], [...]]
    >>> enc.cosine_dist(0, 1)
    0.007...
    """

    def __init__(self) -> None:
        if not _BINDINGS_AVAILABLE:
            raise ImportError(
                "vectro_py is required.  Build it with `maturin develop` or "
                "`pip install vectro` first."
            )
        self._inner = _PyBf16Encoder()

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, vectors: List[List[float]]) -> None:
        """Encode *vectors* (list of f32 lists) to BF16 in-place.

        Replaces any previously stored vectors.
        """
        self._inner.encode(vectors)

    def encode_np(self, array: np.ndarray) -> None:
        """Zero-copy encode from a 2-D float32 numpy array ``(N, D)``.

        Replaces any previously stored vectors.
        """
        arr = np.ascontiguousarray(array, dtype=np.float32)
        self._inner.encode_np(arr)

    def decode(self) -> List[List[float]]:
        """Decode all stored BF16 vectors back to float32 lists."""
        return self._inner.decode()

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    def cosine_dist(self, i: int, j: int) -> float:
        """Cosine *distance* (1 − cosine_similarity) between vectors *i* and *j*.

        Returns a value in ``[0, 2]``; 0 = identical direction.
        """
        return self._inner.cosine_dist(i, j)

    # ------------------------------------------------------------------
    # Python dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._inner)

    def __repr__(self) -> str:
        return repr(self._inner)
