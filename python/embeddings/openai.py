"""OpenAI embedding provider.

Wraps the official ``openai`` Python SDK (>= 1.0).  Calls
``client.embeddings.create(model=..., input=[...])`` per batch and assembles
the resulting vectors into a float32 matrix.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .base import BaseEmbeddingProvider


_OPENAI_HINT = (
    "openai is required for OpenAIEmbeddings.  "
    "Install with: pip install openai"
)


class OpenAIEmbeddings(BaseEmbeddingProvider):
    """OpenAI ``text-embedding-*`` models with caching + auto-batching.

    Args:
        model: e.g. ``"text-embedding-3-small"`` (1536 dim) or
            ``"text-embedding-3-large"`` (3072 dim).
        api_key: Optional API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable.
        client: Optional pre-built ``openai.OpenAI`` (or compatible) client.
            Used as-is if supplied; lets callers configure timeouts, retries,
            base URLs, or pass a stub during testing.
        batch_size: OpenAI accepts up to 2048 inputs per request — the
            default of 256 keeps individual requests well-conditioned.
        cache_dir, normalize, dimension: Forwarded to
            :class:`BaseEmbeddingProvider`.
    """

    provider_name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        batch_size: int = 256,
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

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError(_OPENAI_HINT) from exc
        kwargs = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        self._client = openai.OpenAI(**kwargs)
        return self._client

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        client = self._ensure_client()
        resp = client.embeddings.create(model=self.model, input=list(texts))
        # openai>=1.0: resp.data is a list of objects with .embedding (list[float])
        rows: List[List[float]] = []
        data = getattr(resp, "data", None) or resp["data"]  # both styles
        for item in data:
            emb = getattr(item, "embedding", None)
            if emb is None:
                emb = item["embedding"]
            rows.append(list(emb))
        return np.asarray(rows, dtype=np.float32)


__all__ = ["OpenAIEmbeddings"]
