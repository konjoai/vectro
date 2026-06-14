"""Tests for the Vectro re-ranking module.

Covers:
- _cosine_rerank: pure cosine re-scoring helper
- _rrf_rerank: RRF fusion helper
- VectroReranker.rerank (cosine + rrf strategies)
- VectroReranker.arerank (async)
- LangChainReranker.compress_documents + acompress_documents
- LangChainReranker.invoke / ainvoke
- Edge cases: empty candidates, unknown doc_ids, strategy validation
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Inject minimal LangChain stub so top-level import doesn't fail
# ---------------------------------------------------------------------------


def _ensure_lc_stub():
    if "langchain_core" not in sys.modules:
        sys.modules["langchain_core"] = types.ModuleType("langchain_core")
        sys.modules["langchain_core.vectorstores"] = types.ModuleType("langchain_core.vectorstores")


_ensure_lc_stub()

from python.retrieval.reranker import (  # noqa: E402
    VectroReranker,
    LangChainReranker,
    _cosine_rerank,
    _rrf_rerank,
)


# ---------------------------------------------------------------------------
# Minimal fake "document" object
# ---------------------------------------------------------------------------


class _Doc:
    def __init__(self, text, doc_id, meta=None):
        self.page_content = text
        self.metadata = {"id": doc_id, **(meta or {})}
        self.id_ = doc_id


# ---------------------------------------------------------------------------
# Minimal fake store (mirrors LangChain store structure)
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self, mat: np.ndarray, ids):
        from python.vectro import Vectro

        v = Vectro()
        self._compressed = v.compress(mat, profile="fast")
        self._ids = list(ids)
        import threading

        self._lock = threading.Lock()

    def similarity_search_with_score(self, query, k=4, **kwargs):
        return []


# ---------------------------------------------------------------------------
# _cosine_rerank unit tests
# ---------------------------------------------------------------------------


class TestCosineRerankHelper(unittest.TestCase):
    def _setup(self, n=6, dim=32):
        rng = np.random.default_rng(1)
        mat = rng.standard_normal((n, dim)).astype(np.float32)
        ids = [f"d{i}" for i in range(n)]
        id_to_row = {did: i for i, did in enumerate(ids)}
        q = rng.standard_normal(dim).astype(np.float32)
        docs = [_Doc(f"text {i}", f"d{i}") for i in range(n)]
        candidates = [(docs[i].id_, docs[i], float(i)) for i in range(n)]
        return mat, id_to_row, q, candidates

    def test_returns_top_k(self):
        mat, id_to_row, q, cands = self._setup()
        result = _cosine_rerank(q, cands, mat, id_to_row, top_k=3)
        self.assertEqual(len(result), 3)

    def test_result_sorted_descending(self):
        mat, id_to_row, q, cands = self._setup()
        result = _cosine_rerank(q, cands, mat, id_to_row, top_k=4)
        scores = [s for _, _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_scores_in_cosine_range(self):
        mat, id_to_row, q, cands = self._setup()
        result = _cosine_rerank(q, cands, mat, id_to_row, top_k=6)
        for _, _, score in result:
            self.assertGreaterEqual(score, -1.01)
            self.assertLessEqual(score, 1.01)

    def test_unknown_doc_id_skipped(self):
        mat, id_to_row, q, _ = self._setup(n=3)
        cands = [("unknown-id", _Doc("x", "unknown-id"), 0.5)]
        result = _cosine_rerank(q, cands, mat, id_to_row, top_k=5)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# _rrf_rerank unit tests
# ---------------------------------------------------------------------------


class TestRRFRerankHelper(unittest.TestCase):
    def _setup(self, n=6, dim=32):
        rng = np.random.default_rng(2)
        mat = rng.standard_normal((n, dim)).astype(np.float32)
        ids = [f"r{i}" for i in range(n)]
        id_to_row = {did: i for i, did in enumerate(ids)}
        q = rng.standard_normal(dim).astype(np.float32)
        docs = [_Doc(f"text {i}", f"r{i}") for i in range(n)]
        # Descending original scores
        candidates = [(docs[i].id_, docs[i], float(n - i)) for i in range(n)]
        return mat, id_to_row, q, candidates

    def test_returns_top_k(self):
        mat, id_to_row, q, cands = self._setup()
        result = _rrf_rerank(q, cands, mat, id_to_row, top_k=4)
        self.assertEqual(len(result), 4)

    def test_result_sorted_descending(self):
        mat, id_to_row, q, cands = self._setup()
        result = _rrf_rerank(q, cands, mat, id_to_row, top_k=6)
        scores = [s for _, _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rrf_scores_positive(self):
        mat, id_to_row, q, cands = self._setup()
        result = _rrf_rerank(q, cands, mat, id_to_row, top_k=3)
        for _, _, s in result:
            self.assertGreater(s, 0.0)

    def test_no_duplicates(self):
        mat, id_to_row, q, cands = self._setup()
        result = _rrf_rerank(q, cands, mat, id_to_row, top_k=6)
        ids = [did for did, _, _ in result]
        self.assertEqual(len(ids), len(set(ids)))


# ---------------------------------------------------------------------------
# VectroReranker tests
# ---------------------------------------------------------------------------


class TestVectroReranker(unittest.TestCase):
    def _make_store_and_candidates(self, n=8, dim=32):
        rng = np.random.default_rng(3)
        mat = rng.standard_normal((n, dim)).astype(np.float32)
        ids = [f"doc-{i}" for i in range(n)]
        store = _FakeStore(mat, ids)
        docs = [_Doc(f"text {i}", f"doc-{i}") for i in range(n)]
        candidates = [(docs[i].id_, docs[i], float(i)) for i in range(n)]
        q = rng.standard_normal(dim).astype(np.float32)
        return store, candidates, q

    def test_cosine_strategy_returns_top_k(self):
        store, cands, q = self._make_store_and_candidates()
        reranker = VectroReranker(store, strategy="cosine")
        result = reranker.rerank(q, cands, top_k=3)
        self.assertEqual(len(result), 3)

    def test_rrf_strategy_returns_top_k(self):
        store, cands, q = self._make_store_and_candidates()
        reranker = VectroReranker(store, strategy="rrf")
        result = reranker.rerank(q, cands, top_k=4)
        self.assertEqual(len(result), 4)

    def test_cosine_strategy_scores_descending(self):
        store, cands, q = self._make_store_and_candidates()
        reranker = VectroReranker(store, strategy="cosine")
        result = reranker.rerank(q, cands, top_k=5)
        scores = [s for _, _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rrf_strategy_scores_descending(self):
        store, cands, q = self._make_store_and_candidates()
        reranker = VectroReranker(store, strategy="rrf")
        result = reranker.rerank(q, cands, top_k=5)
        scores = [s for _, _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_candidates_returns_empty(self):
        store, _, q = self._make_store_and_candidates()
        reranker = VectroReranker(store, strategy="cosine")
        result = reranker.rerank(q, [], top_k=5)
        self.assertEqual(result, [])

    def test_invalid_strategy_raises(self):
        store, _, _ = self._make_store_and_candidates()
        with self.assertRaises(ValueError):
            VectroReranker(store, strategy="unknown")

    def test_repr_contains_strategy(self):
        store, _, _ = self._make_store_and_candidates()
        r = VectroReranker(store, strategy="cosine")
        self.assertIn("cosine", repr(r))

    def test_top_k_capped_by_candidates_count(self):
        store, cands, q = self._make_store_and_candidates(n=3)
        reranker = VectroReranker(store, strategy="cosine")
        result = reranker.rerank(q, cands[:3], top_k=100)
        self.assertLessEqual(len(result), 3)

    def test_documents_preserved_in_output(self):
        store, cands, q = self._make_store_and_candidates()
        reranker = VectroReranker(store, strategy="cosine")
        result = reranker.rerank(q, cands, top_k=3)
        for _did, doc, _score in result:
            self.assertIsInstance(doc, _Doc)

    def test_async_rerank_returns_same_count(self):
        async def _run():
            store, cands, q = self._make_store_and_candidates()
            reranker = VectroReranker(store, strategy="cosine")
            return await reranker.arerank(q, cands, top_k=4)

        result = asyncio.run(_run())
        self.assertEqual(len(result), 4)


# ---------------------------------------------------------------------------
# LangChainReranker tests
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Stub embedding model returning a fixed normalised vector."""

    def __init__(self, dim=32):
        self._dim = dim
        self._rng = np.random.default_rng(99)

    def embed_query(self, text: str):
        v = self._rng.standard_normal(self._dim).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


class TestLangChainReranker(unittest.TestCase):
    def _setup(self, n=8, dim=32):
        rng = np.random.default_rng(5)
        mat = rng.standard_normal((n, dim)).astype(np.float32)
        ids = [f"lc-{i}" for i in range(n)]
        store = _FakeStore(mat, ids)
        docs = [_Doc(f"page content {i}", f"lc-{i}", {"idx": i}) for i in range(n)]
        embedder = _FakeEmbedder(dim)
        return store, docs, embedder

    def test_compress_documents_returns_top_k(self):
        store, docs, emb = self._setup()
        reranker = LangChainReranker(store, emb, top_k=3)
        result = reranker.compress_documents(docs, "test query")
        self.assertEqual(len(result), 3)

    def test_compress_documents_returns_doc_objects(self):
        store, docs, emb = self._setup()
        reranker = LangChainReranker(store, emb, top_k=4)
        result = reranker.compress_documents(docs, "test query")
        for doc in result:
            self.assertIsInstance(doc, _Doc)

    def test_compress_empty_documents_returns_empty(self):
        store, _, emb = self._setup()
        reranker = LangChainReranker(store, emb, top_k=5)
        result = reranker.compress_documents([], "test query")
        self.assertEqual(result, [])

    def test_compress_documents_rrf_strategy(self):
        store, docs, emb = self._setup()
        reranker = LangChainReranker(store, emb, top_k=3, strategy="rrf")
        result = reranker.compress_documents(docs, "rrf query")
        self.assertEqual(len(result), 3)

    def test_async_compress_documents(self):
        async def _run():
            store, docs, emb = self._setup()
            reranker = LangChainReranker(store, emb, top_k=4)
            return await reranker.acompress_documents(docs, "async query")

        result = asyncio.run(_run())
        self.assertEqual(len(result), 4)

    def test_invoke_shim(self):
        store, docs, emb = self._setup()
        reranker = LangChainReranker(store, emb, top_k=3)
        result = reranker.invoke({"documents": docs, "query": "invoke test"})
        self.assertEqual(len(result), 3)

    def test_ainvoke_shim(self):
        async def _run():
            store, docs, emb = self._setup()
            reranker = LangChainReranker(store, emb, top_k=2)
            return await reranker.ainvoke({"documents": docs, "query": "ainvoke"})

        result = asyncio.run(_run())
        self.assertEqual(len(result), 2)

    def test_repr_contains_top_k(self):
        store, _, emb = self._setup()
        reranker = LangChainReranker(store, emb, top_k=7)
        self.assertIn("7", repr(reranker))

    def test_top_k_capped_by_input_size(self):
        store, docs, emb = self._setup(n=3)
        reranker = LangChainReranker(store, emb, top_k=100)
        result = reranker.compress_documents(docs[:3], "query")
        self.assertLessEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
