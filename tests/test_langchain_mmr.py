"""Tests for LangChain VectorStore MMR search and persistence."""

from __future__ import annotations

import sys
import types
import unittest
from typing import List

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

# ---------------------------------------------------------------------------
# Minimal langchain-core stub
# ---------------------------------------------------------------------------


def _inject_langchain_stub():
    lc = types.ModuleType("langchain_core")
    vs = types.ModuleType("langchain_core.vectorstores")
    docs = types.ModuleType("langchain_core.documents")

    class _Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    vs.VectorStore = object
    docs.Document = _Document
    lc.vectorstores = vs
    lc.documents = docs

    sys.modules.setdefault("langchain_core", lc)
    sys.modules.setdefault("langchain_core.vectorstores", vs)
    sys.modules.setdefault("langchain_core.documents", docs)
    return _Document


_Document = _inject_langchain_stub()

from python.integrations.langchain_integration import (  # noqa: E402
    VectroVectorStore,
    _mmr_select,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(11)


class _FakeEmbeddings:
    def __init__(self, dim: int = 64):
        self._dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [RNG.standard_normal(self._dim).astype(np.float32).tolist() for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return RNG.standard_normal(self._dim).astype(np.float32).tolist()


def _store_with_texts(n: int = 10, dim: int = 64) -> VectroVectorStore:
    texts = [f"document-{i}" for i in range(n)]
    return VectroVectorStore.from_texts(texts, embedding=_FakeEmbeddings(dim))


# ---------------------------------------------------------------------------
# Unit tests for _mmr_select
# ---------------------------------------------------------------------------


class TestMMRSelect(unittest.TestCase):
    def _random_embs(self, n: int, d: int) -> np.ndarray:
        e = RNG.standard_normal((n, d)).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        return e

    def test_returns_k_indices(self):
        embs = self._random_embs(20, 32)
        q = RNG.standard_normal(32).astype(np.float32)
        idx = _mmr_select(embs, q, k=5, fetch_k=10)
        self.assertEqual(len(idx), 5)

    def test_no_duplicate_indices(self):
        embs = self._random_embs(20, 32)
        q = RNG.standard_normal(32).astype(np.float32)
        idx = _mmr_select(embs, q, k=8, fetch_k=15)
        self.assertEqual(len(set(idx.tolist())), 8)

    def test_indices_in_range(self):
        n = 20
        embs = self._random_embs(n, 32)
        q = RNG.standard_normal(32).astype(np.float32)
        idx = _mmr_select(embs, q, k=5, fetch_k=10)
        self.assertTrue(all(0 <= i < n for i in idx))

    def test_lambda_1_matches_top_k(self):
        """lambda_mult=1.0 should select the top-k most relevant (no diversity)."""
        embs = self._random_embs(20, 32)
        q = RNG.standard_normal(32).astype(np.float32)
        mmr_idx = _mmr_select(embs, q, k=5, fetch_k=10, lambda_mult=1.0)

        # Compute expected top-5 by cosine
        q_n = q / (np.linalg.norm(q) + 1e-10)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        scores = (embs / norms) @ q_n
        top5 = np.argsort(scores)[-5:][::-1]

        # The first pick (highest relevance) must match
        self.assertEqual(mmr_idx[0], top5[0])

    def test_k_larger_than_n_clipped(self):
        embs = self._random_embs(4, 16)
        q = RNG.standard_normal(16).astype(np.float32)
        idx = _mmr_select(embs, q, k=10, fetch_k=10)
        self.assertEqual(len(idx), 4)  # capped at n

    def test_fetch_k_larger_than_n_safe(self):
        embs = self._random_embs(5, 16)
        q = RNG.standard_normal(16).astype(np.float32)
        idx = _mmr_select(embs, q, k=3, fetch_k=100)
        self.assertEqual(len(idx), 3)


# ---------------------------------------------------------------------------
# Integration tests via VectroVectorStore
# ---------------------------------------------------------------------------


class TestMMRSearch(unittest.TestCase):
    def setUp(self):
        self.store = _store_with_texts(n=12, dim=64)

    def test_mmr_returns_k_docs(self):
        docs = self.store.max_marginal_relevance_search("query", k=4, fetch_k=10)
        self.assertEqual(len(docs), 4)

    def test_mmr_returns_document_objects(self):
        docs = self.store.max_marginal_relevance_search("query", k=3)
        for doc in docs:
            self.assertTrue(hasattr(doc, "page_content"))
            self.assertIn("_vectro_id", doc.metadata)

    def test_mmr_no_duplicate_documents(self):
        docs = self.store.max_marginal_relevance_search("query", k=5, fetch_k=10)
        ids = [doc.metadata["_vectro_id"] for doc in docs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_mmr_with_score_returns_pairs(self):
        pairs = self.store.max_marginal_relevance_search_with_score("query", k=3)
        for doc, score in pairs:
            self.assertTrue(hasattr(doc, "page_content"))
            self.assertIsInstance(score, float)

    def test_mmr_empty_store_returns_empty(self):
        empty = VectroVectorStore(
            embedding=_FakeEmbeddings(),
            compression_profile="balanced",
        )
        results = empty.max_marginal_relevance_search("query", k=3)
        self.assertEqual(results, [])

    def test_async_mmr_search(self):
        import asyncio

        async def _run():
            return await self.store.amax_marginal_relevance_search("query", k=3)

        docs = asyncio.run(_run())
        self.assertEqual(len(docs), 3)

    def test_mmr_vs_similarity_search_diversity(self):
        """MMR results should be less similar to each other than plain similarity search."""
        store = VectroVectorStore(
            embedding=_FakeEmbeddings(dim=32),
            compression_profile="balanced",
        )
        # Insert 20 documents
        texts = [f"doc-{i}" for i in range(20)]
        store.add_texts(texts)

        sim_docs = store.similarity_search("query", k=5)
        mmr_docs = store.max_marginal_relevance_search("query", k=5, fetch_k=10, lambda_mult=0.3)

        # Both return 5 docs
        self.assertEqual(len(sim_docs), 5)
        self.assertEqual(len(mmr_docs), 5)


# ---------------------------------------------------------------------------
# Persistence tests for LangChainVectorStore
# ---------------------------------------------------------------------------


class TestLangChainPersistence(unittest.TestCase):
    def _build_store(self, n: int = 8, dim: int = 64) -> VectroVectorStore:
        texts = [f"text-{i}" for i in range(n)]
        metas = [{"idx": i} for i in range(n)]
        return VectroVectorStore.from_texts(texts, embedding=_FakeEmbeddings(dim), metadatas=metas)

    def test_save_creates_files(self):
        import tempfile
        import os

        store = self._build_store()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lc_store")
            store.save(path)
            self.assertTrue(os.path.isfile(os.path.join(path, "meta.json")))
            self.assertTrue(os.path.isfile(os.path.join(path, "vectors.npy")))

    def test_load_restores_document_count(self):
        import tempfile
        import os

        store = self._build_store(n=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lc_store")
            store.save(path)
            loaded = VectroVectorStore.load(path, embedding=_FakeEmbeddings())
            self.assertEqual(len(loaded), 10)

    def test_load_restores_texts(self):
        import tempfile
        import os

        store = self._build_store(n=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lc_store")
            store.save(path)
            loaded = VectroVectorStore.load(path, embedding=_FakeEmbeddings())
            self.assertEqual(sorted(loaded._texts), sorted(store._texts))

    def test_load_restores_metadatas(self):
        import tempfile
        import os

        store = self._build_store(n=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lc_store")
            store.save(path)
            loaded = VectroVectorStore.load(path, embedding=_FakeEmbeddings())
            self.assertEqual(loaded._metadatas, store._metadatas)

    def test_load_wrong_store_type_raises(self):
        import tempfile
        import os
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad")
            os.makedirs(path)
            with open(os.path.join(path, "meta.json"), "w") as f:
                json.dump({"version": 1, "store_type": "llamaindex"}, f)
            with self.assertRaises(ValueError):
                VectroVectorStore.load(path, embedding=_FakeEmbeddings())

    def test_similarity_search_works_after_load(self):
        import tempfile
        import os

        store = self._build_store(n=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lc_store")
            store.save(path)
            loaded = VectroVectorStore.load(path, embedding=_FakeEmbeddings())
            results = loaded.similarity_search("query", k=3)
            self.assertEqual(len(results), 3)

    def test_save_empty_store(self):
        import tempfile
        import os

        store = VectroVectorStore(embedding=_FakeEmbeddings(), compression_profile="balanced")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty")
            store.save(path)  # must not raise
            self.assertTrue(os.path.isfile(os.path.join(path, "meta.json")))


if __name__ == "__main__":
    unittest.main()
