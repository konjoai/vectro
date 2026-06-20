"""Tests for VectroDocumentStore.max_marginal_relevance_search (Haystack 2.x)."""

from __future__ import annotations

import sys
import types
import unittest
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

# ---------------------------------------------------------------------------
# Minimal haystack stub
# ---------------------------------------------------------------------------


@dataclass
class _Document:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Optional[str] = None
    embedding: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    __dataclass_fields__ = {
        "id": None,
        "content": None,
        "embedding": None,
        "meta": None,
        "score": None,
    }


def _make_haystack_stub():
    root = types.ModuleType("haystack")
    dc = types.ModuleType("haystack.dataclasses")
    dc.Document = _Document
    root.dataclasses = dc
    sys.modules.setdefault("haystack", root)
    sys.modules.setdefault("haystack.dataclasses", dc)


_make_haystack_stub()

from python.integrations.haystack_integration import VectroDocumentStore  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
DIM = 32


def _emb(d: int = DIM) -> List[float]:
    return RNG.standard_normal(d).astype(np.float32).tolist()


def _doc(content: str = "text", d: int = DIM, meta: Optional[dict] = None) -> _Document:
    return _Document(
        id=str(uuid.uuid4()),
        content=content,
        embedding=_emb(d),
        meta=meta or {},
    )


def _docs(n: int, d: int = DIM) -> List[_Document]:
    return [_doc(content=f"doc-{i}", d=d) for i in range(n)]


def _store_with_docs(n: int = 10, d: int = DIM) -> VectroDocumentStore:
    store = VectroDocumentStore(compression_profile="balanced")
    store.write_documents(_docs(n, d))
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMMRBasic(unittest.TestCase):
    def setUp(self):
        self.store = _store_with_docs(10)

    def test_mmr_returns_k_documents(self):
        results = self.store.max_marginal_relevance_search(_emb(), k=4)
        self.assertEqual(len(results), 4)

    def test_mmr_returns_fewer_than_k_when_store_small(self):
        store = VectroDocumentStore()
        store.write_documents(_docs(3))
        results = store.max_marginal_relevance_search(_emb(), k=5)
        self.assertEqual(len(results), 3)

    def test_mmr_returns_documents(self):
        results = self.store.max_marginal_relevance_search(_emb(), k=3)
        for doc in results:
            self.assertIsNotNone(getattr(doc, "content", None))

    def test_mmr_no_duplicates(self):
        results = self.store.max_marginal_relevance_search(_emb(), k=5)
        ids = [doc.id for doc in results]
        self.assertEqual(len(ids), len(set(ids)))

    def test_mmr_empty_store_returns_empty(self):
        store = VectroDocumentStore()
        results = store.max_marginal_relevance_search(_emb(), k=3)
        self.assertEqual(results, [])


class TestMMRDiversity(unittest.TestCase):
    """Verify that lambda_mult controls the relevance/diversity trade-off."""

    def test_lambda_1_same_as_top_k_order(self):
        """lambda_mult=1.0 → pure relevance, first result should match top-1."""
        store = _store_with_docs(20)
        q = np.asarray(_emb(), dtype=np.float32)

        # Top-1 from embedding_retrieval
        top1 = store.embedding_retrieval(q.tolist(), top_k=1)
        # MMR with lambda=1.0, fetch_k=5
        mmr = store.max_marginal_relevance_search(q.tolist(), k=1, fetch_k=5, lambda_mult=1.0)
        self.assertEqual(len(mmr), 1)
        self.assertEqual(mmr[0].id, top1[0].id)

    def test_lambda_0_avoids_most_similar_after_first(self):
        """lambda_mult=0.0 → pure diversity; subsequent docs should be dissimilar."""
        store = _store_with_docs(20)
        q = np.asarray(_emb(), dtype=np.float32)
        mmr_diverse = store.max_marginal_relevance_search(
            q.tolist(), k=5, fetch_k=20, lambda_mult=0.0
        )
        mmr_relevant = store.max_marginal_relevance_search(
            q.tolist(), k=5, fetch_k=20, lambda_mult=1.0
        )
        # The two should differ (diversity vs relevance selects different sets)
        diverse_ids = {d.id for d in mmr_diverse}
        relevant_ids = {d.id for d in mmr_relevant}
        # They must not be identical sets (with 20 docs and k=5 this is essentially guaranteed)
        self.assertNotEqual(diverse_ids, relevant_ids)

    def test_default_lambda_returns_results(self):
        store = _store_with_docs(15)
        results = store.max_marginal_relevance_search(_emb(), k=5, lambda_mult=0.5)
        self.assertEqual(len(results), 5)


class TestMMRFetchK(unittest.TestCase):
    def test_fetch_k_clamped_to_store_size(self):
        store = _store_with_docs(4)
        # fetch_k > store size — should not error
        results = store.max_marginal_relevance_search(_emb(), k=3, fetch_k=100)
        self.assertEqual(len(results), 3)

    def test_fetch_k_equals_k(self):
        store = _store_with_docs(8)
        results = store.max_marginal_relevance_search(_emb(), k=3, fetch_k=3)
        self.assertEqual(len(results), 3)

    def test_k_equals_1(self):
        store = _store_with_docs(6)
        results = store.max_marginal_relevance_search(_emb(), k=1)
        self.assertEqual(len(results), 1)


class TestMMRFilters(unittest.TestCase):
    def setUp(self):
        self.store = VectroDocumentStore()
        docs = [_doc(content="cat-A", meta={"category": "A"}) for _ in range(5)] + [
            _doc(content="cat-B", meta={"category": "B"}) for _ in range(5)
        ]
        self.store.write_documents(docs)

    def test_filter_reduces_candidate_set(self):
        results = self.store.max_marginal_relevance_search(_emb(), k=3, filters={"category": "A"})
        self.assertEqual(len(results), 3)
        for doc in results:
            self.assertEqual(doc.meta.get("category"), "A")

    def test_filter_no_match_returns_empty(self):
        results = self.store.max_marginal_relevance_search(_emb(), k=3, filters={"category": "Z"})
        self.assertEqual(results, [])

    def test_filter_none_returns_from_all(self):
        results = self.store.max_marginal_relevance_search(_emb(), k=5, filters=None)
        self.assertEqual(len(results), 5)


class TestMMRAsync(unittest.IsolatedAsyncioTestCase):
    async def test_async_mmr_returns_results(self):
        store = _store_with_docs(10)
        results = await store.async_max_marginal_relevance_search(_emb(), k=4)
        self.assertEqual(len(results), 4)

    async def test_async_mmr_empty_store(self):
        store = VectroDocumentStore()
        results = await store.async_max_marginal_relevance_search(_emb(), k=3)
        self.assertEqual(results, [])

    async def test_async_mmr_with_filter(self):
        store = VectroDocumentStore()
        store.write_documents(
            [_doc(meta={"tag": "x"}) for _ in range(4)]
            + [_doc(meta={"tag": "y"}) for _ in range(4)]
        )
        results = await store.async_max_marginal_relevance_search(_emb(), k=2, filters={"tag": "x"})
        self.assertEqual(len(results), 2)
        for doc in results:
            self.assertEqual(doc.meta.get("tag"), "x")

    async def test_async_mmr_no_duplicates(self):
        store = _store_with_docs(8)
        results = await store.async_max_marginal_relevance_search(_emb(), k=5)
        ids = [doc.id for doc in results]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
