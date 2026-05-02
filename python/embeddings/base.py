"""Base embedding provider — auto-batching + on-disk SQLite cache.

Subclasses implement :meth:`_embed_batch(texts)` to call out to OpenAI, Voyage,
Cohere, SentenceTransformers, etc.  The base class handles:

- **Auto-batching** — long input lists are split into chunks of size
  ``batch_size`` so the underlying API never sees more than its supported
  request size.
- **On-disk cache** — every text → embedding is keyed by SHA-256 of
  ``model + text`` and persisted in a single SQLite file (``cache.sqlite``)
  inside ``cache_dir``.  Re-embedding the same text is a single SQL round-trip.
- **Multi-protocol surface** — instances are simultaneously valid as a Vectro
  ``embed_fn`` callable (str | list → ``np.ndarray``), a LangChain
  ``Embeddings`` (``embed_query`` / ``embed_documents``), and a LlamaIndex
  ``BaseEmbedding`` (``_get_query_embedding`` / ``_get_text_embedding`` /
  ``_get_text_embeddings``).
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np


TextLike = Union[str, Sequence[str]]


class BaseEmbeddingProvider:
    """Common scaffolding for embedding-provider adapters.

    Args:
        model: Model identifier (e.g. ``"text-embedding-3-small"``).  Mixed
            into the cache key so swapping models invalidates the cache.
        batch_size: Maximum chunk size passed to :meth:`_embed_batch`.
        cache_dir: Directory holding ``cache.sqlite``.  ``None`` disables the
            cache entirely.
        normalize: If ``True``, returned vectors are L2-normalised in place.
            Cached values are stored *after* normalisation.
        dimension: Optional fixed dimension hint.  When omitted, set lazily
            from the first embedding call.
    """

    #: Provider identifier — set by subclasses (e.g. ``"openai"``).
    provider_name: str = "base"

    def __init__(
        self,
        model: str,
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
        normalize: bool = False,
        dimension: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model = model
        self.batch_size = int(batch_size)
        self.cache_dir = cache_dir
        self.normalize = bool(normalize)
        self.dimension = dimension

        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._conn: Optional[sqlite3.Connection] = None

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self._conn = sqlite3.connect(
                os.path.join(cache_dir, "cache.sqlite"),
                check_same_thread=False,
                isolation_level=None,  # autocommit
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "  k TEXT PRIMARY KEY, "
                "  v BLOB NOT NULL, "
                "  dim INTEGER NOT NULL"
                ")"
            )

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Return an ``(n, d)`` float32 matrix for *texts*.

        Subclasses must implement this.  Inputs are guaranteed to be a list
        of strings of length ``≤ batch_size``.  Caching and normalisation are
        applied by the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _embed_batch"
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        """SHA-256 of ``provider:model:text`` — collision-resistant cache key."""
        h = hashlib.sha256()
        h.update(self.provider_name.encode("utf-8"))
        h.update(b":")
        h.update(self.model.encode("utf-8"))
        h.update(b":")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, keys: List[str]) -> Dict[str, np.ndarray]:
        """Bulk fetch of cached vectors by key."""
        if self._conn is None or not keys:
            return {}
        with self._cache_lock:
            placeholders = ",".join("?" * len(keys))
            rows = self._conn.execute(
                f"SELECT k, v, dim FROM cache WHERE k IN ({placeholders})",
                keys,
            ).fetchall()
        out: Dict[str, np.ndarray] = {}
        for k, blob, dim in rows:
            arr = np.frombuffer(blob, dtype=np.float32).copy()
            if arr.size != dim:
                continue  # corrupt row — re-embed
            out[k] = arr
        return out

    def _cache_put(self, items: Dict[str, np.ndarray]) -> None:
        """Bulk insert of ``key → vector`` pairs."""
        if self._conn is None or not items:
            return
        rows = [(k, v.astype(np.float32).tobytes(), int(v.size))
                for k, v in items.items()]
        with self._cache_lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO cache (k, v, dim) VALUES (?, ?, ?)",
                rows,
            )

    def cache_stats(self) -> Dict[str, int]:
        """Return cache hit/miss counters and current row count."""
        size = 0
        if self._conn is not None:
            with self._cache_lock:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM cache"
                ).fetchone()
                size = int(row[0]) if row else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": size,
        }

    def clear_cache(self) -> None:
        """Remove every cached entry under this provider."""
        if self._conn is None:
            return
        with self._cache_lock:
            self._conn.execute("DELETE FROM cache")
            self._cache_hits = 0
            self._cache_misses = 0

    def close(self) -> None:
        if self._conn is not None:
            with self._cache_lock:
                self._conn.close()
                self._conn = None

    def __del__(self) -> None:  # best-effort
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core dispatch — auto-batching + cache lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        if mat.ndim == 1:
            n = float(np.linalg.norm(mat))
            return mat / n if n > 0 else mat
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def _embed_uncached(self, texts: List[str]) -> np.ndarray:
        """Call :meth:`_embed_batch` in chunks of ``batch_size``."""
        if not texts:
            d = self.dimension or 0
            return np.zeros((0, max(d, 1)), dtype=np.float32)

        chunks: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start:start + self.batch_size])
            arr = np.asarray(self._embed_batch(chunk), dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != len(chunk):
                raise ValueError(
                    f"_embed_batch returned shape {arr.shape}; "
                    f"expected ({len(chunk)}, dim)"
                )
            chunks.append(arr)

        mat = np.vstack(chunks)
        if self.normalize:
            mat = self._l2_normalize(mat)
        if self.dimension is None:
            self.dimension = int(mat.shape[1])
        elif mat.shape[1] != self.dimension:
            raise ValueError(
                f"_embed_batch returned dim {mat.shape[1]}, "
                f"expected dim {self.dimension}"
            )
        return mat

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Embed a list of strings with cache lookup + auto-batching."""
        text_list: List[str] = list(texts)
        n = len(text_list)
        if n == 0:
            d = self.dimension or 0
            return np.zeros((0, max(d, 1)), dtype=np.float32)

        keys = [self._cache_key(t) for t in text_list]
        cached = self._cache_get(keys)
        self._cache_hits += len(cached)

        miss_indices = [i for i, k in enumerate(keys) if k not in cached]
        self._cache_misses += len(miss_indices)

        if miss_indices:
            miss_texts = [text_list[i] for i in miss_indices]
            new_mat = self._embed_uncached(miss_texts)
            new_items = {keys[i]: new_mat[j]
                         for j, i in enumerate(miss_indices)}
            self._cache_put(new_items)
            cached.update(new_items)

        if self.dimension is None:
            # cached-only path — recover dim from the first vector
            self.dimension = int(next(iter(cached.values())).size)

        out = np.empty((n, self.dimension), dtype=np.float32)
        for i, k in enumerate(keys):
            out[i] = cached[k]
        return out

    def __call__(self, x: TextLike) -> np.ndarray:
        """Vectro ``embed_fn`` contract — str → 1D, list → 2D."""
        if isinstance(x, str):
            return self.embed_texts([x])[0]
        return self.embed_texts(x)

    # ------------------------------------------------------------------
    # LangChain Embeddings protocol
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts).tolist()

    async def aembed_query(self, text: str) -> List[float]:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    # ------------------------------------------------------------------
    # LlamaIndex BaseEmbedding protocol
    # ------------------------------------------------------------------

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self.aembed_query(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self.aembed_query(text)

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, "
            f"batch_size={self.batch_size}, "
            f"cache_dir={self.cache_dir!r}, normalize={self.normalize})"
        )


__all__ = ["BaseEmbeddingProvider", "TextLike"]
