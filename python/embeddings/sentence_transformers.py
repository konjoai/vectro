"""SentenceTransformers embedding provider.

Local, GPU-friendly embedding via the ``sentence-transformers`` package.
The underlying model already does its own internal batching, but we still
chunk by ``batch_size`` so cache lookups stay coherent and progress reporting
is per-chunk.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .base import BaseEmbeddingProvider


_ST_HINT = (
    "sentence-transformers is required for SentenceTransformersEmbeddings.  "
    "Install with: pip install sentence-transformers"
)


class SentenceTransformersEmbeddings(BaseEmbeddingProvider):
    """SentenceTransformers models with caching + auto-batching.

    Args:
        model: HuggingFace model id, e.g. ``"BAAI/bge-small-en-v1.5"``,
            ``"sentence-transformers/all-MiniLM-L6-v2"``,
            ``"intfloat/e5-base-v2"``.
        model_obj: Optional pre-loaded ``sentence_transformers.SentenceTransformer``
            (or stub).  Bypasses model construction — useful for sharing one
            model across multiple providers, or for tests.
        device: Optional torch device string (``"cpu"``, ``"cuda"``,
            ``"mps"``).  Forwarded to ``SentenceTransformer(device=...)``.
        batch_size: Encoder mini-batch size (forwarded to ``encode``).
        cache_dir, normalize, dimension: Forwarded to
            :class:`BaseEmbeddingProvider`.  ``normalize=True`` produces
            unit-norm vectors via ``encode(normalize_embeddings=True)``.
    """

    provider_name = "sentence-transformers"

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_obj: Optional[Any] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        normalize: bool = False,
        dimension: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            batch_size=batch_size,
            cache_dir=cache_dir,
            normalize=normalize,
            dimension=dimension,
        )
        self._device = device
        self._model_obj = model_obj

    def _ensure_model(self) -> Any:
        if self._model_obj is not None:
            return self._model_obj
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(_ST_HINT) from exc
        kwargs = {}
        if self._device is not None:
            kwargs["device"] = self._device
        self._model_obj = SentenceTransformer(self.model, **kwargs)
        return self._model_obj

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        model = self._ensure_model()
        # Note: we set normalize_embeddings=False here because the base class
        # applies normalisation uniformly *after* batching — keeping a single
        # source of truth.  Users get the same end result either way.
        out = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr


__all__ = ["SentenceTransformersEmbeddings"]
