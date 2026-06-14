"""Score-based re-ranking layer for Vectro RAG pipelines.

After an initial retrieval stage (dense, sparse, or RRF hybrid), a re-ranker
refines the ranking by computing cosine similarity between a (potentially
updated) query vector and the stored compressed document embeddings.

Two strategies are provided:
- ``"cosine"``  — pure cosine re-score, no original scores used.
- ``"rrf"``     — fuses original retrieval scores with re-scores via RRF
                  (Cormack 2009, k=60 default), providing a principled
                  multi-signal combination.

Example::

    from python.retrieval import VectroReranker

    store = LangChainVectorStore.from_texts(texts, embedding=embedder)

    # Initial retrieval (any method)
    initial = store.similarity_search_with_score("my query", k=20)

    # Re-rank against a refined query
    reranker = VectroReranker(store, strategy="rrf")
    results = reranker.rerank(
        refined_query_embedding,
        [(doc.metadata.get("id"), doc, score) for doc, score in initial],
        top_k=5,
    )

For LangChain ``BaseDocumentCompressor`` duck-typing::

    from python.retrieval import LangChainReranker

    reranker = LangChainReranker(store, embedding=embedder, top_k=5)
    docs = reranker.compress_documents(initial_docs, "refined query")
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core re-scoring helpers
# ---------------------------------------------------------------------------


def _cosine_rerank(
    query_vec: np.ndarray,
    candidates: Sequence[Tuple[str, Any, float]],
    store_mat: np.ndarray,
    id_to_row: Dict[str, int],
    top_k: int,
) -> List[Tuple[str, Any, float]]:
    """Re-score *candidates* by cosine similarity and return top-*top_k*."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    scored = []
    for doc_id, doc, _orig_score in candidates:
        row = id_to_row.get(doc_id)
        if row is None:
            continue
        vec = store_mat[row]
        norm = np.linalg.norm(vec) + 1e-10
        score = float((vec / norm) @ q)
        scored.append((doc_id, doc, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


def _rrf_rerank(
    query_vec: np.ndarray,
    candidates: Sequence[Tuple[str, Any, float]],
    store_mat: np.ndarray,
    id_to_row: Dict[str, int],
    top_k: int,
    k: int = 60,
) -> List[Tuple[str, Any, float]]:
    """Re-score via RRF fusion of original ranks and cosine re-scores."""
    # Build original ranking (sorted by descending original score)
    orig_ranked = sorted(candidates, key=lambda x: x[2], reverse=True)
    orig_rank = {t[0]: i for i, t in enumerate(orig_ranked)}

    # Build cosine re-score ranking
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    cosine_scores: List[Tuple[str, float]] = []
    for doc_id, _doc, _orig in candidates:
        row = id_to_row.get(doc_id)
        if row is None:
            continue
        vec = store_mat[row]
        norm = np.linalg.norm(vec) + 1e-10
        cosine_scores.append((doc_id, float((vec / norm) @ q)))
    cosine_scores.sort(key=lambda x: x[1], reverse=True)
    cosine_rank = {did: i for i, (did, _) in enumerate(cosine_scores)}

    # RRF fusion
    rrf: Dict[str, float] = defaultdict(float)
    all_ids = {t[0] for t in candidates}
    for doc_id in all_ids:
        r_orig = orig_rank.get(doc_id, len(candidates))
        r_cos = cosine_rank.get(doc_id, len(candidates))
        rrf[doc_id] += 1.0 / (k + r_orig + 1)
        rrf[doc_id] += 1.0 / (k + r_cos + 1)

    id_map = {t[0]: t for t in candidates}
    ranked = sorted(all_ids, key=lambda did: rrf[did], reverse=True)
    return [(did, id_map[did][1], rrf[did]) for did in ranked[:top_k] if did in id_map]


# ---------------------------------------------------------------------------
# VectroReranker
# ---------------------------------------------------------------------------


class VectroReranker:
    """Re-rank retrieved results using Vectro-compressed embeddings.

    Compatible with any Vectro store that exposes ``_compressed``,
    ``_doc_ids`` / ``_node_ids``, and ``_node_store`` / ``_doc_store``
    (LangChain, LlamaIndex, and Haystack adapters all qualify).

    Args:
        store: A Vectro vector store instance.
        strategy: ``"cosine"`` (default) or ``"rrf"``.
        rrf_k: The *k* constant in RRF denominator (default 60).
    """

    def __init__(
        self,
        store: Any,
        strategy: str = "cosine",
        rrf_k: int = 60,
    ) -> None:
        if strategy not in ("cosine", "rrf"):
            raise ValueError(f"strategy must be 'cosine' or 'rrf', got {strategy!r}")
        self._store = store
        self._strategy = strategy
        self._rrf_k = rrf_k

    def _snapshot(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """Return (matrix, id→row) from the store under its lock."""
        store = self._store
        lock = getattr(store, "_lock", None)

        def _read():
            compressed = getattr(store, "_compressed", None)
            if compressed is None:
                return np.zeros((0, 1), dtype=np.float32), {}
            mat = compressed.reconstruct_batch()
            # LangChain uses _ids; LlamaIndex uses _node_ids; Haystack uses _doc_ids
            ids: List[str] = (
                getattr(store, "_ids", None)
                or getattr(store, "_node_ids", None)
                or getattr(store, "_doc_ids", None)
                or []
            )
            return mat, {did: i for i, did in enumerate(ids)}

        if lock is not None:
            with lock:
                return _read()
        return _read()

    def rerank(
        self,
        query_embedding: Any,
        candidates: Sequence[Tuple[str, Any, float]],
        top_k: int = 5,
    ) -> List[Tuple[str, Any, float]]:
        """Re-rank *candidates* against *query_embedding*.

        Args:
            query_embedding: 1-D float32 array (or list) representing the
                (optionally refined) query vector.
            candidates: Sequence of ``(doc_id, document_object, original_score)``
                tuples from an initial retrieval stage.
            top_k: Number of results to return.

        Returns:
            List of ``(doc_id, document_object, new_score)`` tuples, sorted
            descending by re-ranked score.
        """
        if not candidates:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        mat, id_to_row = self._snapshot()
        if mat.shape[0] == 0 or not id_to_row:
            return list(candidates)[:top_k]

        if self._strategy == "rrf":
            return _rrf_rerank(q, candidates, mat, id_to_row, top_k, k=self._rrf_k)
        return _cosine_rerank(q, candidates, mat, id_to_row, top_k)

    async def arerank(
        self,
        query_embedding: Any,
        candidates: Sequence[Tuple[str, Any, float]],
        top_k: int = 5,
    ) -> List[Tuple[str, Any, float]]:
        """Async variant of :meth:`rerank` — runs in a thread-pool executor."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.rerank(query_embedding, candidates, top_k)
        )

    def __repr__(self) -> str:
        return f"VectroReranker(strategy={self._strategy!r}, rrf_k={self._rrf_k})"


# ---------------------------------------------------------------------------
# LangChainReranker — duck-typed BaseDocumentCompressor
# ---------------------------------------------------------------------------


class LangChainReranker:
    """LangChain ``BaseDocumentCompressor``-compatible re-ranker.

    Implements the ``compress_documents(documents, query)`` interface so it
    can be used in any ``ContextualCompressionRetriever`` pipeline without
    importing ``langchain_core``.

    Args:
        store: A Vectro LangChain vector store.
        embedding: LangChain-compatible embedding model (must have
            ``.embed_query(text) -> List[float]``).
        top_k: Number of documents to keep after re-ranking (default 5).
        strategy: ``"cosine"`` (default) or ``"rrf"``.
        rrf_k: RRF denominator constant (default 60).
    """

    def __init__(
        self,
        store: Any,
        embedding: Any,
        top_k: int = 5,
        strategy: str = "cosine",
        rrf_k: int = 60,
    ) -> None:
        self._store = store
        self._embedding = embedding
        self._top_k = top_k
        self._reranker = VectroReranker(store, strategy=strategy, rrf_k=rrf_k)

    def compress_documents(
        self,
        documents: List[Any],
        query: str,
        callbacks: Any = None,
    ) -> List[Any]:
        """Re-rank *documents* against *query*.

        Args:
            documents: List of LangChain ``Document`` objects from an initial
                retrieval stage.  Each must carry metadata with an ``"id"``
                key matching the store's internal ids, or the method falls
                back to positional order.
            query: The (optionally refined) user query string.
            callbacks: Ignored (present for interface compatibility).

        Returns:
            Re-ranked list of documents, truncated to *top_k*.
        """
        if not documents:
            return []
        q_vec = np.asarray(self._embedding.embed_query(query), dtype=np.float32)

        # Build candidates: (doc_id, document, score=0.0 as placeholder)
        ids = _extract_ids(self._store, documents)
        candidates = [(ids[i], documents[i], 0.0) for i in range(len(documents))]

        results = self._reranker.rerank(q_vec, candidates, top_k=self._top_k)
        return [doc for _did, doc, _score in results]

    async def acompress_documents(
        self,
        documents: List[Any],
        query: str,
        callbacks: Any = None,
    ) -> List[Any]:
        """Async variant of :meth:`compress_documents`."""
        import asyncio

        loop = asyncio.get_running_loop()
        q_vec = np.asarray(
            await loop.run_in_executor(None, lambda: self._embedding.embed_query(query)),
            dtype=np.float32,
        )
        ids = _extract_ids(self._store, documents)
        candidates = [(ids[i], documents[i], 0.0) for i in range(len(documents))]
        return [
            doc
            for _did, doc, _score in await self._reranker.arerank(
                q_vec, candidates, top_k=self._top_k
            )
        ]

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> List[Any]:
        """LangChain ``Runnable.invoke`` compatibility shim."""
        docs = input.get("documents", []) if isinstance(input, dict) else input
        query = input.get("query", "") if isinstance(input, dict) else kwargs.get("query", "")
        return self.compress_documents(docs, query)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> List[Any]:
        """LangChain ``Runnable.ainvoke`` compatibility shim."""
        docs = input.get("documents", []) if isinstance(input, dict) else input
        query = input.get("query", "") if isinstance(input, dict) else kwargs.get("query", "")
        return await self.acompress_documents(docs, query)

    def __repr__(self) -> str:
        return f"LangChainReranker(top_k={self._top_k}, strategy={self._reranker._strategy!r})"


# ---------------------------------------------------------------------------
# HaystackReranker — duck-typed Haystack 2.x component
# ---------------------------------------------------------------------------


class HaystackReranker:
    """Haystack 2.x component duck-type for re-ranking retrieved Documents.

    Implements the ``run()`` interface expected by Haystack pipelines.
    Returns ``{"documents": [...]}`` so it wires directly into any
    ``Pipeline.add_component`` chain.

    Args:
        store: A ``VectroDocumentStore`` instance.
        top_k: Number of documents to keep after re-ranking (default 5).
        strategy: ``"cosine"`` (default) or ``"rrf"``.
        rrf_k: RRF denominator constant (default 60).
    """

    def __init__(
        self,
        store: Any,
        top_k: int = 5,
        strategy: str = "cosine",
        rrf_k: int = 60,
    ) -> None:
        self._store = store
        self._top_k = top_k
        self._reranker = VectroReranker(store, strategy=strategy, rrf_k=rrf_k)

    def run(
        self,
        query_embedding: Any,
        documents: List[Any],
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Any]]:
        """Re-rank *documents* against *query_embedding*.

        Args:
            query_embedding: 1-D float32 array or list representing the query vector.
            documents: List of Haystack ``Document`` objects from an initial
                retrieval stage.
            top_k: Override the instance-level top_k for this call.

        Returns:
            ``{"documents": [re-ranked Document list]}``.
        """
        k = top_k if top_k is not None else self._top_k
        if not documents:
            return {"documents": []}

        q_vec = np.asarray(query_embedding, dtype=np.float32)
        ids = _extract_haystack_ids(self._store, documents)
        candidates = [(ids[i], documents[i], 0.0) for i in range(len(documents))]
        results = self._reranker.rerank(q_vec, candidates, top_k=k)
        return {"documents": [doc for _did, doc, _score in results]}

    async def async_run(
        self,
        query_embedding: Any,
        documents: List[Any],
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Any]]:
        """Async variant of :meth:`run`."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.run(query_embedding, documents, top_k))

    def __repr__(self) -> str:
        return f"HaystackReranker(top_k={self._top_k}, strategy={self._reranker._strategy!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_ids(store: Any, documents: List[Any]) -> List[str]:
    """Map LangChain *documents* to store-internal ids, falling back to positional."""
    ids: List[str] = (
        getattr(store, "_ids", None)
        or getattr(store, "_node_ids", None)
        or getattr(store, "_doc_ids", None)
        or []
    )
    id_set = set(ids)
    result = []
    for i, doc in enumerate(documents):
        meta = getattr(doc, "metadata", {}) or {}
        candidate = meta.get("id") or meta.get("doc_id") or getattr(doc, "id_", None)
        if candidate and candidate in id_set:
            result.append(str(candidate))
        else:
            result.append(ids[i] if i < len(ids) else f"__pos_{i}__")
    return result


def _extract_haystack_ids(store: Any, documents: List[Any]) -> List[str]:
    """Map Haystack *documents* to store-internal ids, falling back to positional."""
    store_ids: List[str] = getattr(store, "_doc_ids", None) or []
    id_set = set(store_ids)
    result = []
    for i, doc in enumerate(documents):
        # Haystack Document exposes .id directly
        candidate = getattr(doc, "id", None)
        if candidate and str(candidate) in id_set:
            result.append(str(candidate))
        else:
            result.append(store_ids[i] if i < len(store_ids) else f"__pos_{i}__")
    return result
