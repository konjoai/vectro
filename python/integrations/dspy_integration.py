"""DSPy retriever backed by Vectro compression.

Provides a drop-in DSPy retrieval module that transparently compresses stored
passage embeddings to INT8 / NF4 — reducing in-memory footprint by 4–8× while
preserving recall above the standard DSPy quality bar.

Usage::

    import dspy
    from python.integrations.dspy_integration import VectroDSPyRetriever

    def my_embed(text):           # any embedding fn: str | List[str] -> np.ndarray
        return model.encode(text)

    rm = VectroDSPyRetriever(embed_fn=my_embed, k=5, compression_profile="balanced")
    rm.add_texts(
        passages=["Paris is the capital of France", "Berlin is cold in winter"],
    )

    # DSPy retrieval contract — text in, dspy.Prediction(passages=[...]) out
    prediction = rm("What is the capital of France?")
    print(prediction.passages)        # ["Paris is the capital of France", ...]

    # Or wire as the global retriever
    dspy.settings.configure(rm=rm)

The retriever follows the ``dspy.Retrieve`` duck-typing protocol — both
``forward(query_or_queries, k)`` and ``__call__`` are supported, returning a
``dspy.Prediction`` when DSPy is installed (otherwise a structurally equivalent
fallback object).
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from python.retrieval.mmr import cosine_scores as _cosine_scores_fn, mmr_select as _mmr_select


QueryLike = Union[str, Sequence[str]]
EmbedFn = Callable[[Union[str, List[str]]], Any]


_DSPY_HINT = (
    "dspy-ai is recommended for full DSPy integration. "
    "Install with: pip install dspy-ai"
)


def _make_prediction(passages: List[str], **fields: Any) -> Any:
    """Return a ``dspy.Prediction`` if DSPy is installed, else a fallback."""
    try:
        import dspy  # type: ignore
        return dspy.Prediction(passages=list(passages), **fields)
    except Exception:
        class _Prediction:
            def __init__(self, **kw: Any) -> None:
                self.__dict__.update(kw)

            def __repr__(self) -> str:
                return f"_Prediction({self.__dict__!r})"

        return _Prediction(passages=list(passages), **fields)


def _embed_one(embed_fn: EmbedFn, text: str) -> np.ndarray:
    """Call *embed_fn* on a single string and return a 1-D float32 vector."""
    out = embed_fn(text)
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(
            f"embed_fn must return a 1-D vector for a single string, got shape {arr.shape}"
        )
    return arr


def _embed_many(embed_fn: EmbedFn, texts: List[str]) -> np.ndarray:
    """Call *embed_fn* on a list and return a 2-D float32 matrix (n, d)."""
    out = embed_fn(texts)
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 1:
        # Per-call fallback — embed_fn does not batch
        arr = np.stack([_embed_one(embed_fn, t) for t in texts], axis=0)
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        raise ValueError(
            f"embed_fn returned shape {arr.shape}; expected ({len(texts)}, dim)"
        )
    return arr


class VectroDSPyRetriever:
    """DSPy retrieval module with Vectro embedding compression.

    Implements the DSPy ``Retrieve`` duck-typing protocol — both
    ``forward(query_or_queries, k)`` and ``__call__`` return a
    ``dspy.Prediction(passages=[...])`` (or a structurally identical fallback
    when DSPy is not installed).

    Args:
        embed_fn: Callable mapping a query string (or list of strings) to a
            numpy array of embeddings.  Used at query time to encode raw text;
            ``None`` is allowed if all retrieval calls pass ``query_embedding``
            directly.
        k: Default number of passages returned per query (DSPy convention).
        compression_profile: ``"fast"``, ``"balanced"`` (default),
            ``"quality"``, or ``"binary"``.
        model_dir: Optional HuggingFace model directory.  When supplied,
            the model-family registry selects the optimal quantization
            method (e.g. GTE → INT8, BGE → NF4).
    """

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = None,
        k: int = 3,
        compression_profile: str = "balanced",
        model_dir: Optional[str] = None,
    ) -> None:
        from python.vectro import Vectro

        self._embed_fn = embed_fn
        self._k = int(k)
        self._profile = compression_profile
        self._model_dir = model_dir
        self._vectro = Vectro()

        self._lock = threading.Lock()
        self._passages: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._compressed: Any = None
        self._n_dims: int = 0

    # ------------------------------------------------------------------
    # DSPy Retrieve attribute compatibility
    # ------------------------------------------------------------------

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        self._k = int(value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_embed_fn(self) -> EmbedFn:
        if self._embed_fn is None:
            raise ValueError(
                "embed_fn was not provided; pass query_embedding=... to forward(), "
                "or instantiate VectroDSPyRetriever(embed_fn=...)."
            )
        return self._embed_fn

    def _rebuild(self, new_embs: np.ndarray) -> None:
        """Append *new_embs* rows to the compressed store (caller holds lock)."""
        if self._compressed is None:
            self._compressed = self._vectro.compress(
                new_embs, profile=self._profile, model_dir=self._model_dir
            )
            self._n_dims = new_embs.shape[1]
        else:
            existing = self._compressed.reconstruct_batch()
            combined = np.vstack([existing, new_embs])
            self._compressed = self._vectro.compress(
                combined, profile=self._profile, model_dir=self._model_dir
            )

    def _cosine_scores(self, query_emb: np.ndarray) -> np.ndarray:
        return _cosine_scores_fn(query_emb, self._compressed.reconstruct_batch())

    @staticmethod
    def _normalize_query(query_or_queries: QueryLike) -> List[str]:
        if isinstance(query_or_queries, str):
            return [query_or_queries]
        return list(query_or_queries)

    @staticmethod
    def _matches_filters(meta: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        if not meta:
            return False
        return all(meta.get(k) == v for k, v in filters.items())

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def add_texts(
        self,
        passages: Sequence[str],
        embeddings: Optional[np.ndarray] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> int:
        """Add passages to the retriever.

        Args:
            passages: Sequence of text strings to index.
            embeddings: Optional pre-computed ``(n, d)`` embedding matrix.  If
                omitted, ``embed_fn`` is invoked on the passages.
            metadatas: Optional per-passage metadata dicts (used by ``filters``
                in :meth:`forward`).  Length must match ``passages``.

        Returns:
            Number of passages added.
        """
        passages = list(passages)
        if not passages:
            return 0

        if embeddings is None:
            embed_fn = self._ensure_embed_fn()
            emb_matrix = _embed_many(embed_fn, passages)
        else:
            emb_matrix = np.asarray(embeddings, dtype=np.float32)
            if emb_matrix.ndim != 2 or emb_matrix.shape[0] != len(passages):
                raise ValueError(
                    f"embeddings shape {emb_matrix.shape} does not match "
                    f"{len(passages)} passages"
                )

        if metadatas is None:
            metas: List[Dict[str, Any]] = [{} for _ in passages]
        else:
            metas = [dict(m) if m else {} for m in metadatas]
            if len(metas) != len(passages):
                raise ValueError(
                    f"metadatas length {len(metas)} != passages length {len(passages)}"
                )

        with self._lock:
            self._passages.extend(passages)
            self._metadatas.extend(metas)
            self._rebuild(emb_matrix)

        return len(passages)

    def clear(self) -> None:
        """Remove all passages from the retriever."""
        with self._lock:
            self._passages.clear()
            self._metadatas.clear()
            self._compressed = None

    # ------------------------------------------------------------------
    # DSPy Retrieve protocol
    # ------------------------------------------------------------------

    def forward(
        self,
        query_or_queries: QueryLike,
        k: Optional[int] = None,
        *,
        query_embedding: Optional[np.ndarray] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Retrieve the top-*k* passages for one or more queries.

        Args:
            query_or_queries: A single query string or a list of strings.  When
                a list is passed, scores are aggregated across queries (sum).
            k: Override the default ``k`` set at construction.
            query_embedding: Pre-computed query vector ``(d,)``.  Bypasses
                ``embed_fn`` — useful when DSPy is wired into a pipeline that
                already produced an embedding.  Only valid for single-string
                queries.
            filters: Optional metadata equality filters applied before ranking.

        Returns:
            ``dspy.Prediction(passages=[str, ...])`` — or a structurally
            equivalent fallback object when DSPy is not installed.
        """
        k_eff = self._k if k is None else int(k)
        if k_eff <= 0:
            return _make_prediction(passages=[])

        with self._lock:
            n_corpus = len(self._passages)
            if n_corpus == 0 or self._compressed is None:
                return _make_prediction(passages=[])
            passages = list(self._passages)
            metadatas = list(self._metadatas)
            mat = self._compressed.reconstruct_batch()

        if query_embedding is not None:
            q_arr = np.asarray(query_embedding, dtype=np.float32)
            if q_arr.ndim != 1:
                raise ValueError(
                    f"query_embedding must be 1-D, got shape {q_arr.shape}"
                )
            scores = _cosine_scores_fn(q_arr, mat)
        else:
            queries = self._normalize_query(query_or_queries)
            embed_fn = self._ensure_embed_fn()
            q_mat = _embed_many(embed_fn, queries)
            # Aggregate by summing per-query cosine scores (multi-query DSPy convention)
            scores = np.zeros(n_corpus, dtype=np.float32)
            for row in q_mat:
                scores += _cosine_scores_fn(row, mat)

        filtered_idx = [
            i for i in range(n_corpus)
            if self._matches_filters(metadatas[i], filters)
        ]
        if not filtered_idx:
            return _make_prediction(passages=[])

        k_eff = min(k_eff, len(filtered_idx))
        filtered_scores = scores[filtered_idx]
        top_local = np.argpartition(filtered_scores, -k_eff)[-k_eff:]
        top_local = top_local[np.argsort(filtered_scores[top_local])[::-1]]

        chosen = [filtered_idx[i] for i in top_local]
        chosen_passages = [passages[i] for i in chosen]
        chosen_scores = [float(scores[i]) for i in chosen]

        return _make_prediction(
            passages=chosen_passages,
            scores=chosen_scores,
            indices=list(chosen),
        )

    def __call__(
        self,
        query_or_queries: QueryLike,
        k: Optional[int] = None,
        *,
        query_embedding: Optional[np.ndarray] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self.forward(
            query_or_queries,
            k=k,
            query_embedding=query_embedding,
            filters=filters,
        )

    async def aforward(
        self,
        query_or_queries: QueryLike,
        k: Optional[int] = None,
        *,
        query_embedding: Optional[np.ndarray] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Non-blocking variant of :meth:`forward`."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.forward(
                query_or_queries,
                k=k,
                query_embedding=query_embedding,
                filters=filters,
            ),
        )

    # ------------------------------------------------------------------
    # MMR retrieval
    # ------------------------------------------------------------------

    def forward_mmr(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        query_embedding: Optional[np.ndarray] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Diversity-promoting retrieval using Maximal Marginal Relevance.

        Args:
            query: Single query string.
            k: Number of passages to return.
            fetch_k: Candidate pool size (≥ k).
            lambda_mult: Trade-off in [0, 1] — 1.0 pure relevance, 0.0 pure
                diversity.  Default 0.5.
            query_embedding: Pre-computed query vector.  Bypasses ``embed_fn``.
            filters: Optional metadata equality filters applied before MMR.

        Returns:
            ``dspy.Prediction(passages=[...])`` ordered by MMR selection.
        """
        with self._lock:
            n_corpus = len(self._passages)
            if n_corpus == 0 or self._compressed is None:
                return _make_prediction(passages=[])
            passages = list(self._passages)
            metadatas = list(self._metadatas)
            mat = self._compressed.reconstruct_batch()

        if query_embedding is not None:
            q_arr = np.asarray(query_embedding, dtype=np.float32)
        else:
            embed_fn = self._ensure_embed_fn()
            q_arr = _embed_one(embed_fn, query)

        filtered_idx = [
            i for i in range(n_corpus)
            if self._matches_filters(metadatas[i], filters)
        ]
        if not filtered_idx:
            return _make_prediction(passages=[])

        filtered_arr = np.array(filtered_idx)
        filtered_mat = mat[filtered_arr]

        mmr_local = _mmr_select(
            filtered_mat, q_arr, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        chosen = [int(filtered_arr[i]) for i in mmr_local]
        return _make_prediction(
            passages=[passages[i] for i in chosen],
            indices=chosen,
        )

    async def aforward_mmr(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        query_embedding: Optional[np.ndarray] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Non-blocking variant of :meth:`forward_mmr`."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.forward_mmr(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                query_embedding=query_embedding,
                filters=filters,
            ),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the retriever to *path* (a directory).

        Creates two files inside *path*:
        - ``meta.json`` — passages, metadatas, profile, dims
        - ``vectors.npy`` — reconstructed float32 embeddings

        ``embed_fn`` is **not** serialized — supply it again on :meth:`load`.
        """
        import json
        import os

        os.makedirs(path, exist_ok=True)

        with self._lock:
            n = len(self._passages)
            if n == 0:
                mat = np.zeros((0, max(self._n_dims, 1)), dtype=np.float32)
            else:
                mat = self._compressed.reconstruct_batch()
            passages = list(self._passages)
            metadatas = list(self._metadatas)

        np.save(os.path.join(path, "vectors.npy"), mat)
        meta = {
            "version": 1,
            "store_type": "dspy",
            "profile": self._profile,
            "model_dir": self._model_dir,
            "n_dims": self._n_dims,
            "k": self._k,
            "passages": passages,
            "metadatas": metadatas,
        }
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump(meta, fh)

    @classmethod
    def load(
        cls,
        path: str,
        embed_fn: Optional[EmbedFn] = None,
    ) -> "VectroDSPyRetriever":
        """Load a previously saved retriever from *path*.

        Args:
            path: Directory containing ``meta.json`` and ``vectors.npy``.
            embed_fn: Query-time embedding function — must be supplied again
                since callables cannot be JSON-serialized.

        Returns:
            A new ``VectroDSPyRetriever`` with all passages and compressed
            embeddings restored.
        """
        import json
        import os

        with open(os.path.join(path, "meta.json")) as fh:
            meta = json.load(fh)

        if meta.get("store_type") != "dspy":
            raise ValueError(
                f"meta.json store_type={meta.get('store_type')!r} is not 'dspy'."
            )

        rm = cls(
            embed_fn=embed_fn,
            k=int(meta.get("k", 3)),
            compression_profile=meta["profile"],
            model_dir=meta.get("model_dir"),
        )
        mat = np.load(os.path.join(path, "vectors.npy"))
        passages = meta["passages"]
        metadatas = meta.get("metadatas") or [{} for _ in passages]

        if len(mat) > 0 and len(passages) > 0:
            with rm._lock:
                rm._passages = list(passages)
                rm._metadatas = [dict(m) if m else {} for m in metadatas]
                rm._n_dims = int(meta["n_dims"])
                rm._compressed = rm._vectro.compress(
                    mat, profile=meta["profile"], model_dir=meta.get("model_dir")
                )
        return rm

    # ------------------------------------------------------------------
    # Vectro-specific helpers
    # ------------------------------------------------------------------

    @property
    def compression_stats(self) -> dict:
        """Return memory usage and compression ratio statistics."""
        with self._lock:
            n = len(self._passages)
            if n == 0 or self._compressed is None:
                return {"n_passages": 0, "compression_ratio": 1.0}
            d = self._n_dims
            original_mb = n * d * 4 / (1024 ** 2)
            compressed_mb = self._compressed.total_compressed_bytes / (1024 ** 2)
            return {
                "n_passages": n,
                "dimensions": d,
                "compression_profile": self._profile,
                "original_mb": round(original_mb, 3),
                "compressed_mb": round(compressed_mb, 3),
                "compression_ratio": round(self._compressed.compression_ratio, 2),
                "memory_saved_mb": round(original_mb - compressed_mb, 3),
            }

    def __len__(self) -> int:
        return len(self._passages)

    def __repr__(self) -> str:
        return (
            f"VectroDSPyRetriever(n={len(self)}, k={self._k}, "
            f"profile={self._profile!r}, dims={self._n_dims})"
        )
