"""Tests for the RRF hybrid retriever (v4.15.0)."""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from typing import List, Tuple

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.retrieval.rrf_retriever import (  # noqa: E402
    reciprocal_rank_fusion,
    rrf_top_k,
    RRFRetriever,
    LangChainRRFRetriever,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ranking_fn(results: List[Tuple[str, str, float]]):
    """Returns a retriever function that always returns the given results."""

    def _fn(query: str, fetch_k: int) -> List[Tuple[str, str, float]]:
        return results[:fetch_k]

    return _fn


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------


class TestRRFAlgorithm(unittest.TestCase):
    def test_single_ranking_gives_decreasing_scores(self):
        ranking = ["doc-a", "doc-b", "doc-c"]
        scores = reciprocal_rank_fusion([ranking])
        self.assertGreater(scores["doc-a"], scores["doc-b"])
        self.assertGreater(scores["doc-b"], scores["doc-c"])

    def test_consensus_boosts_score(self):
        r1 = ["doc-a", "doc-b"]
        r2 = ["doc-a", "doc-c"]
        scores = reciprocal_rank_fusion([r1, r2])
        # doc-a appears in both rankings — must outscore doc-b and doc-c
        self.assertGreater(scores["doc-a"], scores["doc-b"])
        self.assertGreater(scores["doc-a"], scores["doc-c"])

    def test_all_docs_present_in_output(self):
        r1 = ["a", "b"]
        r2 = ["c", "d"]
        scores = reciprocal_rank_fusion([r1, r2])
        self.assertSetEqual(set(scores.keys()), {"a", "b", "c", "d"})

    def test_empty_ranking_ignored(self):
        scores = reciprocal_rank_fusion([[], ["doc-x"]])
        self.assertIn("doc-x", scores)

    def test_duplicate_within_ranking_counted_once(self):
        scores1 = reciprocal_rank_fusion([["a", "a", "a"]])
        scores2 = reciprocal_rank_fusion([["a"]])
        self.assertAlmostEqual(scores1["a"], scores2["a"], places=10)

    def test_rrf_k_affects_score_magnitude(self):
        scores_low_k = reciprocal_rank_fusion([["doc"]], k=1)
        scores_high_k = reciprocal_rank_fusion([["doc"]], k=1000)
        # Lower k → higher score at same rank
        self.assertGreater(scores_low_k["doc"], scores_high_k["doc"])


class TestRRFTopK(unittest.TestCase):
    def test_returns_k_results(self):
        r = ["a", "b", "c", "d", "e"]
        top = rrf_top_k([r], k=3)
        self.assertEqual(len(top), 3)

    def test_sorted_descending(self):
        r1 = ["a", "b", "c"]
        r2 = ["b", "a", "c"]
        top = rrf_top_k([r1, r2], k=3)
        scores = [s for _, s in top]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_k_larger_than_unique_docs(self):
        top = rrf_top_k([["a", "b"]], k=10)
        self.assertEqual(len(top), 2)  # capped at unique doc count


# ---------------------------------------------------------------------------
# RRFRetriever
# ---------------------------------------------------------------------------


class TestRRFRetriever(unittest.TestCase):
    def _make_results(self, prefix: str, n: int) -> List[Tuple[str, str, float]]:
        return [(f"{prefix}-{i}", f"text for {prefix}-{i}", 1.0 / (i + 1)) for i in range(n)]

    def test_empty_stores_raises(self):
        with self.assertRaises(ValueError):
            RRFRetriever([], k=3)

    def test_retrieve_returns_k_results(self):
        fn = _ranking_fn(self._make_results("doc", 10))
        r = RRFRetriever([fn], k=4)
        results = r.retrieve("query")
        self.assertEqual(len(results), 4)

    def test_retrieve_result_structure(self):
        fn = _ranking_fn(self._make_results("doc", 5))
        r = RRFRetriever([fn], k=3)
        results = r.retrieve("query")
        for item in results:
            self.assertIn("id", item)
            self.assertIn("text", item)
            self.assertIn("score", item)

    def test_consensus_boosts_rank(self):
        # doc-0 appears first in both sources → should rank 1st
        r1 = [("doc-0", "text0", 1.0), ("doc-1", "text1", 0.8)]
        r2 = [("doc-0", "text0", 0.9), ("doc-2", "text2", 0.7)]
        retriever = RRFRetriever([_ranking_fn(r1), _ranking_fn(r2)], k=3)
        results = retriever.retrieve("query")
        self.assertEqual(results[0]["id"], "doc-0")

    def test_source_failure_is_non_fatal(self):
        def _bad_fn(q, fk):
            raise RuntimeError("source down")

        good_results = self._make_results("doc", 5)
        r = RRFRetriever([_bad_fn, _ranking_fn(good_results)], k=3)
        results = r.retrieve("query")
        self.assertEqual(len(results), 3)

    def test_async_retrieve(self):
        fn = _ranking_fn(self._make_results("doc", 8))
        r = RRFRetriever([fn], k=4)

        async def _run():
            return await r.aretrieve("query")

        results = asyncio.run(_run())
        self.assertEqual(len(results), 4)


# ---------------------------------------------------------------------------
# LangChainRRFRetriever
# ---------------------------------------------------------------------------


def _inject_langchain_stub():
    lc = sys.modules.get("langchain_core") or types.ModuleType("langchain_core")
    docs_mod = sys.modules.get("langchain_core.documents") or types.ModuleType(
        "langchain_core.documents"
    )

    class _Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    docs_mod.Document = _Document
    lc.documents = docs_mod
    sys.modules.setdefault("langchain_core", lc)
    sys.modules.setdefault("langchain_core.documents", docs_mod)
    return _Document


_LCDoc = _inject_langchain_stub()

# Also inject langchain_core.vectorstores for the LangChainVectorStore import
vs_mod = sys.modules.get("langchain_core.vectorstores") or types.ModuleType(
    "langchain_core.vectorstores"
)
vs_mod.VectorStore = object
sys.modules.setdefault("langchain_core.vectorstores", vs_mod)

from python.integrations.langchain_integration import VectroVectorStore  # noqa: E402

RNG = np.random.default_rng(13)


class _FakeEmbeddings:
    def __init__(self, dim: int = 32):
        self._dim = dim

    def embed_documents(self, texts):
        return [RNG.standard_normal(self._dim).astype(np.float32).tolist() for _ in texts]

    def embed_query(self, text):
        return RNG.standard_normal(self._dim).astype(np.float32).tolist()


def _build_lc_store(n: int = 10, dim: int = 32) -> VectroVectorStore:
    texts = [f"doc-{i}" for i in range(n)]
    return VectroVectorStore.from_texts(texts, embedding=_FakeEmbeddings(dim))


class TestLangChainRRFRetriever(unittest.TestCase):
    def test_empty_stores_raises(self):
        with self.assertRaises(ValueError):
            LangChainRRFRetriever([], k=3)

    def test_get_relevant_documents_returns_k(self):
        store = _build_lc_store()
        r = LangChainRRFRetriever([store], k=3)
        docs = r.get_relevant_documents("query")
        self.assertEqual(len(docs), 3)

    def test_returns_document_objects(self):
        store = _build_lc_store()
        r = LangChainRRFRetriever([store], k=4)
        docs = r.get_relevant_documents("query")
        for doc in docs:
            self.assertTrue(hasattr(doc, "page_content"))
            self.assertIn("_rrf_score", doc.metadata)

    def test_no_duplicate_results(self):
        store = _build_lc_store(n=8)
        r = LangChainRRFRetriever([store, store], k=5)
        docs = r.get_relevant_documents("query")
        ids = [d.metadata["_vectro_id"] for d in docs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_multiple_stores_fused(self):
        store_a = _build_lc_store(n=5)
        store_b = _build_lc_store(n=5)
        r = LangChainRRFRetriever([store_a, store_b], k=4)
        docs = r.get_relevant_documents("query")
        self.assertEqual(len(docs), 4)

    def test_async_get_relevant_documents(self):
        store = _build_lc_store()
        r = LangChainRRFRetriever([store], k=3)

        async def _run():
            return await r.aget_relevant_documents("query")

        docs = asyncio.run(_run())
        self.assertEqual(len(docs), 3)

    def test_invoke_interface(self):
        store = _build_lc_store()
        r = LangChainRRFRetriever([store], k=3)
        docs = r.invoke("query")
        self.assertEqual(len(docs), 3)

    def test_n_stores_property(self):
        r = LangChainRRFRetriever([_build_lc_store(), _build_lc_store()], k=3)
        self.assertEqual(r.n_stores, 2)

    def test_repr(self):
        r = LangChainRRFRetriever([_build_lc_store()], k=4)
        self.assertIn("LangChainRRFRetriever", repr(r))


if __name__ == "__main__":
    unittest.main()
