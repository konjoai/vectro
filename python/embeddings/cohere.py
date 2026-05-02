"""Cohere embedding provider.

Wraps the ``cohere`` Python SDK.  Uses ``client.embed(texts=..., model=...,
input_type=...)`` and reads ``response.embeddings``.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .base import BaseEmbeddingProvider


_COHERE_HINT = (
    "cohere is required for CohereEmbeddings.  "
    "Install with: pip install cohere"
)


class CohereEmbeddings(BaseEmbeddingProvider):
    """Cohere ``embed-*`` models with caching + auto-batching.

    Args:
        model: e.g. ``"embed-english-v3.0"`` (1024 dim),
            ``"embed-multilingual-v3.0"`` (1024 dim),
            ``"embed-english-light-v3.0"`` (384 dim).
        api_key: Optional API key.  Falls back to ``COHERE_API_KEY``.
        client: Optional pre-built ``cohere.Client`` (or stub).
        input_type: Cohere v3 requires one of ``"search_document"``,
            ``"search_query"``, ``"classification"``, ``"clustering"``.
            Default ``"search_document"`` matches indexing semantics.
        batch_size: Cohere allows up to 96 inputs per request; default 96.
        cache_dir, normalize, dimension: Forwarded to
            :class:`BaseEmbeddingProvider`.
    """

    provider_name = "cohere"

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        input_type: str = "search_document",
        batch_size: int = 96,
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
        self._api_key = api_key
        self._client = client
        self.input_type = input_type

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import cohere  # type: ignore
        except ImportError as exc:
            raise ImportError(_COHERE_HINT) from exc
        if self._api_key is not None:
            self._client = cohere.Client(api_key=self._api_key)
        else:
            self._client = cohere.Client()
        return self._client

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        return self._embed_batch_with_type(texts, self.input_type)

    def _embed_batch_with_type(
        self, texts: List[str], input_type: str
    ) -> np.ndarray:
        client = self._ensure_client()
        resp = client.embed(
            texts=list(texts), model=self.model, input_type=input_type
        )
        embs = getattr(resp, "embeddings", None)
        if embs is None and isinstance(resp, dict):
            embs = resp["embeddings"]
        # Some cohere SDK versions return EmbedByTypeResponseEmbeddings — pick float
        if hasattr(embs, "float_"):
            embs = embs.float_
        elif hasattr(embs, "float"):
            embs = embs.float
        return np.asarray(embs, dtype=np.float32)

    # Cohere v3 is asymmetric: queries use input_type="search_query".
    def embed_query(self, text: str) -> List[float]:
        if self.input_type == "search_query":
            return super().embed_query(text)
        prev_provider = self.provider_name
        try:
            self.provider_name = f"{prev_provider}:query"
            key = self._cache_key(text)
            cached = self._cache_get([key])
            if key in cached:
                self._cache_hits += 1
                vec = cached[key]
            else:
                self._cache_misses += 1
                vec = self._embed_batch_with_type([text], "search_query")[0]
                if self.normalize:
                    vec = self._l2_normalize(vec)
                if self.dimension is None:
                    self.dimension = int(vec.size)
                self._cache_put({key: vec})
        finally:
            self.provider_name = prev_provider
        return vec.astype(np.float32).tolist()


__all__ = ["CohereEmbeddings"]
