"""Reciprocal Rank Fusion (RRF) hybrid retriever — pure Python, zero extra deps.

Combines ranked result lists from multiple retrieval sources using the standard
RRF formula:

    score(d) = Σ_i  1 / (k + rank_i(d))

where *k* = 60 (Cormack et al. 2009 default) and rank is 0-indexed.

Typical use:
- Dense store (Vectro INT8) + keyword/BM25 store → fusion gives hybrid recall.
- Multiple Vectro stores with different compression profiles → ensemble quality.
- No external dependencies; works with any list of (doc_id, score) pairs.

Adapters provided:
- :class:`RRFRetriever`          — generic, framework-agnostic
- :class:`LangChainRRFRetriever` — duck-typed LangChain ``BaseRetriever``
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Core RRF algorithm
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
) -> Dict[str, float]:
    """Fuse multiple ranked lists of document ids via Reciprocal Rank Fusion.

    Args:
        rankings: Each element is an ordered sequence of document ids, best
            first.  Duplicate ids within a single ranking are ignored after
            their first appearance.
        k: RRF smoothing constant (default 60 — empirically optimal across
            many TREC collections).

    Returns:
        ``{doc_id: rrf_score}`` dict.  Higher score → better combined rank.
    """
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set = set()
        for rank, doc_id in enumerate(ranking):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank + 1)
    return dict(scores)


def rrf_top_k(
    rankings: Sequence[Sequence[str]],
    k: int,
    rrf_k: int = 60,
) -> List[Tuple[str, float]]:
    """Return top-k *(doc_id, rrf_score)* pairs after fusing *rankings*.

    Args:
        rankings: Ranked doc-id lists, best first per source.
        k: Number of results to return.
        rrf_k: RRF smoothing constant.

    Returns:
        List of ``(doc_id, score)`` sorted by descending RRF score.
    """
    scores = reciprocal_rank_fusion(rankings, k=rrf_k)
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return top[:k]


# ---------------------------------------------------------------------------
# Generic RRF retriever
# ---------------------------------------------------------------------------


class RRFRetriever:
    """Framework-agnostic hybrid retriever using Reciprocal Rank Fusion.

    Accepts a list of *retrieval functions* — each takes a query string and
    ``fetch_k`` and returns a list of ``(doc_id, text, score)`` tuples ordered
    best-first.  Fuses the rankings and returns top-k results.

    Args:
        retrievers: List of callables ``(query: str, fetch_k: int) ->
            List[Tuple[str, str, float]]`` where the tuple is
            ``(doc_id, text, score)``.
        k: Final number of results to return.
        fetch_k: Candidate pool to draw from each source (≥ k).
        rrf_k: RRF smoothing constant (default 60).

    Example::

        def keyword_search(query, fetch_k):
            # simple BM25 or keyword match returning (id, text, score) triples
            ...

        def dense_search(query, fetch_k):
            results = store.similarity_search_with_score(query, k=fetch_k)
            return [(doc.metadata["_vectro_id"], doc.page_content, score)
                    for doc, score in results]

        retriever = RRFRetriever([keyword_search, dense_search], k=5)
        results = retriever.retrieve("quantum computing")
    """

    def __init__(
        self,
        retrievers: List[Callable[[str, int], List[Tuple[str, str, float]]]],
        k: int = 4,
        fetch_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("RRFRetriever requires at least one retriever.")
        self._retrievers = retrievers
        self._k = k
        self._fetch_k = fetch_k
        self._rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run all retrievers and fuse results via RRF.

        Returns:
            List of ``{"id": str, "text": str, "score": float}`` dicts,
            sorted by descending RRF score.
        """
        k = k or self._k
        fetch_k = max(k, self._fetch_k)

        all_results: List[List[Tuple[str, str, float]]] = []
        for fn in self._retrievers:
            try:
                all_results.append(fn(query, fetch_k))
            except Exception:  # noqa: BLE001 — individual source failure is non-fatal
                all_results.append([])

        # Build id→text map and rankings
        id_text: Dict[str, str] = {}
        rankings: List[List[str]] = []
        for source_results in all_results:
            ranking: List[str] = []
            for doc_id, text, _ in source_results:
                id_text.setdefault(doc_id, text)
                ranking.append(doc_id)
            rankings.append(ranking)

        fused = rrf_top_k(rankings, k=k, rrf_k=self._rrf_k)
        return [
            {"id": doc_id, "text": id_text.get(doc_id, ""), "score": score}
            for doc_id, score in fused
        ]

    async def aretrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Async variant — runs blocking :meth:`retrieve` in a thread-pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.retrieve(query, k=k))


# ---------------------------------------------------------------------------
# LangChain-compatible RRF retriever
# ---------------------------------------------------------------------------


class LangChainRRFRetriever:
    """LangChain duck-typed ``BaseRetriever`` backed by RRF fusion.

    Implements ``get_relevant_documents`` and ``aget_relevant_documents`` —
    the two methods required by the LangChain retriever protocol — without
    hard-importing ``langchain_core`` at module load time.

    Args:
        stores: List of Vectro :class:`VectroVectorStore` instances (or any
            object that implements ``similarity_search_with_score``).
        k: Final number of results to return.
        fetch_k: Candidate pool per store (≥ k).
        rrf_k: RRF smoothing constant (default 60).

    Example::

        from python.integrations import LangChainVectorStore
        from python.retrieval.rrf_retriever import LangChainRRFRetriever

        dense = LangChainVectorStore.from_texts(texts, embedding=emb)
        retriever = LangChainRRFRetriever([dense], k=5)

        docs = retriever.get_relevant_documents("my query")
        docs = await retriever.aget_relevant_documents("my query")
    """

    def __init__(
        self,
        stores: List[Any],
        k: int = 4,
        fetch_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if not stores:
            raise ValueError("LangChainRRFRetriever requires at least one store.")
        self._stores = stores
        self._k = k
        self._fetch_k = fetch_k
        self._rrf_k = rrf_k

    def _store_to_retriever_fn(self, store: Any) -> Callable:
        def _fn(query: str, fetch_k: int) -> List[Tuple[str, str, float]]:
            results = store.similarity_search_with_score(query, k=fetch_k)
            out = []
            for doc, score in results:
                doc_id = doc.metadata.get("_vectro_id", "")
                text = getattr(doc, "page_content", "")
                out.append((doc_id, text, float(score)))
            return out

        return _fn

    def _run(self, query: str) -> List[Any]:
        """Internal: run fusion and return LangChain Document objects."""
        try:
            from langchain_core.documents import Document as _LCDoc
        except ImportError:
            _LCDoc = None  # type: ignore[assignment]

        fetch_k = max(self._k, self._fetch_k)
        id_text: Dict[str, str] = {}
        rankings: List[List[str]] = []

        for store in self._stores:
            try:
                results = store.similarity_search_with_score(query, k=fetch_k)
            except Exception:  # noqa: BLE001
                results = []
            ranking: List[str] = []
            for doc, _ in results:
                doc_id = doc.metadata.get("_vectro_id", "")
                id_text.setdefault(doc_id, getattr(doc, "page_content", ""))
                ranking.append(doc_id)
            rankings.append(ranking)

        fused = rrf_top_k(rankings, k=self._k, rrf_k=self._rrf_k)

        docs = []
        for doc_id, rrf_score in fused:
            text = id_text.get(doc_id, "")
            if _LCDoc is not None:
                docs.append(
                    _LCDoc(
                        page_content=text,
                        metadata={"_vectro_id": doc_id, "_rrf_score": rrf_score},
                    )
                )
            else:
                docs.append({"id": doc_id, "text": text, "score": rrf_score})
        return docs

    def get_relevant_documents(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Any]:
        """Return top-k documents fused across all stores via RRF."""
        return self._run(query)

    async def aget_relevant_documents(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Any]:
        """Async variant — delegates to :meth:`get_relevant_documents`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.get_relevant_documents(query))

    # LangChain also calls invoke/ainvoke in newer versions
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> List[Any]:
        query = input if isinstance(input, str) else str(input)
        return self.get_relevant_documents(query)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> List[Any]:
        query = input if isinstance(input, str) else str(input)
        return await self.aget_relevant_documents(query)

    @property
    def n_stores(self) -> int:
        return len(self._stores)

    def __repr__(self) -> str:
        return (
            f"LangChainRRFRetriever(stores={self.n_stores}, k={self._k}, "
            f"fetch_k={self._fetch_k}, rrf_k={self._rrf_k})"
        )
