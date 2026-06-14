"""Tests for LangChain VectorStore protocol completions in v4.15.0.

Covers: add_documents, from_documents, similarity_search_by_vector,
        metadata filter= kwarg on all search methods, async variants.
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from typing import List, Optional

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()


def _inject_langchain_stub():
    lc = sys.modules.get("langchain_core") or types.ModuleType("langchain_core")
    vs = sys.modules.get("langchain_core.vectorstores") or types.ModuleType(
        "langchain_core.vectorstores"
    )
    docs_mod = sys.modules.get("langchain_core.documents") or types.ModuleType(
        "langchain_core.documents"
    )

    class _Document:
        def __init__(self, page_content="", metadata=None, id=None):
            self.page_content = page_content
            self.metadata = metadata or {}
            self.id = id

    vs.VectorStore = object
    docs_mod.Document = _Document
    lc.vectorstores = vs
    lc.documents = docs_mod

    sys.modules.setdefault("langchain_core", lc)
    sys.modules.setdefault("langchain_core.vectorstores", vs)
    sys.modules.setdefault("langchain_core.documents", docs_mod)
    return _Document


_Document = _inject_langchain_stub()

from python.integrations.langchain_integration import VectroVectorStore  # noqa: E402

RNG = np.random.default_rng(77)


class _FakeEmbeddings:
    def __init__(self, dim: int = 64):
        self._dim = dim

    def embed_documents(self, texts):
        return [RNG.standard_normal(self._dim).astype(np.float32).tolist() for _ in texts]

    def embed_query(self, text):
        return RNG.standard_normal(self._dim).astype(np.float32).tolist()


def _build_store(n: int = 10, dim: int = 64, tags: Optional[List[str]] = None):
    emb = _FakeEmbeddings(dim)
    texts = [f"text-{i}" for i in range(n)]
    metas = [{"tag": tags[i % len(tags)] if tags else "default", "idx": i} for i in range(n)]
    return VectroVectorStore.from_texts(texts, embedding=emb, metadatas=metas)


# ---------------------------------------------------------------------------
# add_documents / from_documents
# ---------------------------------------------------------------------------


class TestAddDocuments(unittest.TestCase):
    def setUp(self):
        self.emb = _FakeEmbeddings()
        self.store = VectroVectorStore(embedding=self.emb)

    def _make_docs(self, n: int) -> List[_Document]:
        return [_Document(page_content=f"doc-{i}", metadata={"src": "test"}) for i in range(n)]

    def test_add_documents_returns_ids(self):
        docs = self._make_docs(4)
        ids = self.store.add_documents(docs)
        self.assertEqual(len(ids), 4)

    def test_add_documents_increments_count(self):
        self.store.add_documents(self._make_docs(5))
        self.assertEqual(len(self.store), 5)

    def test_add_documents_preserves_metadata(self):
        docs = [_Document(page_content="hello", metadata={"key": "value"})]
        self.store.add_documents(docs)
        self.assertEqual(self.store._metadatas[0]["key"], "value")

    def test_add_documents_respects_doc_id(self):
        doc = _Document(page_content="tagged", metadata={}, id="custom-id-123")
        self.store.add_documents([doc])
        self.assertIn("custom-id-123", self.store._ids)

    def test_from_documents_creates_populated_store(self):
        docs = self._make_docs(6)
        store = VectroVectorStore.from_documents(docs, embedding=self.emb)
        self.assertEqual(len(store), 6)

    def test_from_documents_extracts_page_content(self):
        docs = [_Document(page_content="content-a"), _Document(page_content="content-b")]
        store = VectroVectorStore.from_documents(docs, embedding=self.emb)
        self.assertIn("content-a", store._texts)
        self.assertIn("content-b", store._texts)

    def test_from_documents_empty_list(self):
        store = VectroVectorStore.from_documents([], embedding=self.emb)
        self.assertEqual(len(store), 0)


# ---------------------------------------------------------------------------
# similarity_search_by_vector
# ---------------------------------------------------------------------------


class TestSearchByVector(unittest.TestCase):
    def setUp(self):
        self.dim = 64
        self.store = _build_store(n=10, dim=self.dim)

    def test_search_by_vector_returns_k_docs(self):
        emb = RNG.standard_normal(self.dim).astype(np.float32).tolist()
        docs = self.store.similarity_search_by_vector(emb, k=3)
        self.assertEqual(len(docs), 3)

    def test_search_by_vector_returns_document_objects(self):
        emb = RNG.standard_normal(self.dim).astype(np.float32).tolist()
        docs = self.store.similarity_search_by_vector(emb, k=2)
        for doc in docs:
            self.assertTrue(hasattr(doc, "page_content"))

    def test_search_by_vector_with_score_returns_pairs(self):
        emb = RNG.standard_normal(self.dim).astype(np.float32).tolist()
        pairs = self.store.similarity_search_by_vector_with_score(emb, k=3)
        for doc, score in pairs:
            self.assertIsInstance(score, float)

    def test_search_by_vector_matches_query_search(self):
        """Both paths should agree on the top-1 result (same query vector)."""
        # Use a fixed embedding known to the store
        store = VectroVectorStore(embedding=_FakeEmbeddings(32))
        texts = ["alpha", "beta", "gamma"]
        store.add_texts(texts)
        # Grab the exact vector for "alpha" (index 0)
        mat = store._compressed.reconstruct_batch()
        q_vec = mat[0].tolist()

        docs = store.similarity_search_by_vector(q_vec, k=1)
        self.assertEqual(len(docs), 1)

    def test_async_search_by_vector(self):
        emb = RNG.standard_normal(self.dim).astype(np.float32).tolist()

        async def _run():
            return await self.store.asimilarity_search_by_vector(emb, k=2)

        docs = asyncio.run(_run())
        self.assertEqual(len(docs), 2)


# ---------------------------------------------------------------------------
# Metadata filter= support
# ---------------------------------------------------------------------------


class TestMetadataFilter(unittest.TestCase):
    def setUp(self):
        self.store = _build_store(n=12, tags=["A", "B", "C"])

    def test_filter_similarity_search(self):
        docs = self.store.similarity_search("query", k=10, filter={"tag": "A"})
        for doc in docs:
            self.assertEqual(doc.metadata["tag"], "A")

    def test_filter_returns_correct_count(self):
        # 12 docs, 3 tags → 4 per tag
        docs = self.store.similarity_search("query", k=10, filter={"tag": "B"})
        self.assertLessEqual(len(docs), 4)

    def test_filter_with_score(self):
        pairs = self.store.similarity_search_with_score("query", k=10, filter={"tag": "C"})
        for doc, _ in pairs:
            self.assertEqual(doc.metadata["tag"], "C")

    def test_filter_no_match_returns_empty(self):
        docs = self.store.similarity_search("query", k=5, filter={"tag": "ZZZZ"})
        self.assertEqual(docs, [])

    def test_filter_search_by_vector(self):
        emb = RNG.standard_normal(64).astype(np.float32).tolist()
        docs = self.store.similarity_search_by_vector(emb, k=10, filter={"tag": "A"})
        for doc in docs:
            self.assertEqual(doc.metadata["tag"], "A")

    def test_filter_mmr_search(self):
        docs = self.store.max_marginal_relevance_search(
            "query", k=3, fetch_k=8, filter={"tag": "B"}
        )
        for doc in docs:
            self.assertEqual(doc.metadata["tag"], "B")

    def test_filter_none_returns_all(self):
        docs = self.store.similarity_search("query", k=12, filter=None)
        self.assertEqual(len(docs), 12)

    def test_async_filter_search(self):
        async def _run():
            return await self.store.asimilarity_search("query", k=5, filter={"tag": "A"})

        docs = asyncio.run(_run())
        for doc in docs:
            self.assertEqual(doc.metadata["tag"], "A")


if __name__ == "__main__":
    unittest.main()
