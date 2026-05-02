"""Voyage AI embedding provider.

Wraps the official ``voyageai`` Python SDK.  Uses ``client.embed(texts,
model=..., input_type=...)`` per batch.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .base import BaseEmbeddingProvider


_VOYAGE_HINT = (
    "voyageai is required for VoyageEmbeddings.  "
    "Install with: pip install voyageai"
)


class VoyageEmbeddings(BaseEmbeddingProvider):
    """Voyage AI embedding models with caching + auto-batching.

    Args:
        model: e.g. ``"voyage-3"`` (1024 dim), ``"voyage-3-large"`` (2048 dim),
            ``"voyage-code-3"`` (1024 dim, code-specialised).
        api_key: Optional API key.  Falls back to ``VOYAGE_API_KEY``.
        client: Optional pre-built ``voyageai.Client`` (or stub).
        input_type: ``"document"`` (default — corpus indexing) or ``"query"``
            (asymmetric retrieval).  Per-call override available via
            :meth:`embed_query` (always uses ``"query"``).
        batch_size: Voyage allows up to 128 inputs per request; default 64.
        cache_dir, normalize, dimension: Forwarded to
            :class:`BaseEmbeddingProvider`.
    """

    provider_name = "voyage"

    def __init__(
        self,
        model: str = "voyage-3",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        input_type: str = "document",
        batch_size: int = 64,
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
            import voyageai  # type: ignore
        except ImportError as exc:
            raise ImportError(_VOYAGE_HINT) from exc
        kwargs = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        self._client = voyageai.Client(**kwargs)
        return self._client

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        return self._embed_batch_with_type(texts, self.input_type)

    def _embed_batch_with_type(
        self, texts: List[str], input_type: str
    ) -> np.ndarray:
        client = self._ensure_client()
        resp = client.embed(
            list(texts), model=self.model, input_type=input_type
        )
        # voyageai>=0.3: resp.embeddings is a list[list[float]]
        embs = getattr(resp, "embeddings", None)
        if embs is None and isinstance(resp, dict):
            embs = resp["embeddings"]
        return np.asarray(embs, dtype=np.float32)

    # Voyage retrieval is asymmetric — queries use input_type="query".
    def embed_query(self, text: str) -> List[float]:
        # Bypass cache only when input_type differs from the indexing one.
        if self.input_type == "query":
            return super().embed_query(text)
        # Use a query-typed cache key so document/query embeddings don't collide.
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
                vec = self._embed_batch_with_type([text], "query")[0]
                if self.normalize:
                    vec = self._l2_normalize(vec)
                if self.dimension is None:
                    self.dimension = int(vec.size)
                self._cache_put({key: vec})
        finally:
            self.provider_name = prev_provider
        return vec.astype(np.float32).tolist()


__all__ = ["VoyageEmbeddings"]
