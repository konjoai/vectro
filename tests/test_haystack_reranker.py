"""Tests for HaystackReranker — Haystack 2.x pipeline component."""

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
from python.retrieval.reranker import HaystackReranker  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(99)
DIM = 32


def _emb(d: int = DIM) -> np.ndarray:
    return RNG.standard_normal(d).astype(np.float32)


def _doc(content: str = "text", d: int = DIM) -> _Document:
    return _Document(
        id=str(uuid.uuid4()),
        content=content,
        embedding=_emb(d).tolist(),
    )


def _store_with_docs(n: int = 8, d: int = DIM) -> VectroDocumentStore:
    store = VectroDocumentStore(compression_profile="balanced")
    docs = [_doc(f"doc-{i}", d) for i in range(n)]
    store.write_documents(docs)
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHaystackRerankerInit(unittest.TestCase):
    def test_default_init(self):
        store = _store_with_docs()
        rr = HaystackReranker(store)
        self.assertEqual(rr._top_k, 5)

    def test_custom_top_k(self):
        store = _store_with_docs()
        rr = HaystackReranker(store, top_k=3)
        self.assertEqual(rr._top_k, 3)

    def test_invalid_strategy_raises(self):
        store = _store_with_docs()
        with self.assertRaises(ValueError):
            HaystackReranker(store, strategy="bad")

    def test_repr(self):
        store = _store_with_docs()
        rr = HaystackReranker(store, top_k=3)
        self.assertIn("HaystackReranker", repr(rr))
        self.assertIn("top_k=3", repr(rr))


class TestHaystackRerankerRun(unittest.TestCase):
    def setUp(self):
        self.store = _store_with_docs(10)
        self.reranker = HaystackReranker(self.store, top_k=5)
        # Pull some documents as candidates from the store
        q = _emb()
        raw = self.store.embedding_retrieval(q.tolist(), top_k=8)
        self.candidates = raw
        self.query_emb = q

    def test_run_returns_dict(self):
        result = self.reranker.run(self.query_emb, self.candidates)
        self.assertIsInstance(result, dict)
        self.assertIn("documents", result)

    def test_run_returns_top_k(self):
        result = self.reranker.run(self.query_emb, self.candidates)
        self.assertEqual(len(result["documents"]), min(5, len(self.candidates)))

    def test_run_empty_candidates(self):
        result = self.reranker.run(self.query_emb, [])
        self.assertEqual(result, {"documents": []})

    def test_run_top_k_override(self):
        result = self.reranker.run(self.query_emb, self.candidates, top_k=2)
        self.assertEqual(len(result["documents"]), 2)

    def test_run_top_k_override_larger_than_candidates(self):
        result = self.reranker.run(self.query_emb, self.candidates[:3], top_k=10)
        self.assertLessEqual(len(result["documents"]), 3)

    def test_run_returns_document_objects(self):
        result = self.reranker.run(self.query_emb, self.candidates)
        for doc in result["documents"]:
            self.assertIsNotNone(getattr(doc, "content", None))

    def test_run_no_duplicates(self):
        result = self.reranker.run(self.query_emb, self.candidates)
        ids = [doc.id for doc in result["documents"]]
        self.assertEqual(len(ids), len(set(ids)))


class TestHaystackRerankerStrategies(unittest.TestCase):
    def setUp(self):
        self.store = _store_with_docs(12)
        q = _emb()
        self.candidates = self.store.embedding_retrieval(q.tolist(), top_k=10)
        self.query_emb = q

    def test_cosine_strategy(self):
        rr = HaystackReranker(self.store, top_k=5, strategy="cosine")
        result = rr.run(self.query_emb, self.candidates)
        self.assertEqual(len(result["documents"]), 5)

    def test_rrf_strategy(self):
        rr = HaystackReranker(self.store, top_k=5, strategy="rrf")
        result = rr.run(self.query_emb, self.candidates)
        self.assertEqual(len(result["documents"]), 5)

    def test_cosine_and_rrf_may_differ(self):
        rr_cos = HaystackReranker(self.store, top_k=5, strategy="cosine")
        rr_rrf = HaystackReranker(self.store, top_k=5, strategy="rrf")
        res_cos = {d.id for d in rr_cos.run(self.query_emb, self.candidates)["documents"]}
        res_rrf = {d.id for d in rr_rrf.run(self.query_emb, self.candidates)["documents"]}
        # Not necessarily different, but both must be non-empty
        self.assertGreater(len(res_cos), 0)
        self.assertGreater(len(res_rrf), 0)


class TestHaystackRerankerAsync(unittest.IsolatedAsyncioTestCase):
    async def test_async_run_returns_documents(self):
        store = _store_with_docs(8)
        rr = HaystackReranker(store, top_k=4)
        q = _emb()
        candidates = store.embedding_retrieval(q.tolist(), top_k=6)
        result = await rr.async_run(q, candidates)
        self.assertIn("documents", result)
        self.assertEqual(len(result["documents"]), 4)

    async def test_async_run_empty_candidates(self):
        store = _store_with_docs(5)
        rr = HaystackReranker(store, top_k=3)
        result = await rr.async_run(_emb(), [])
        self.assertEqual(result, {"documents": []})

    async def test_async_run_top_k_override(self):
        store = _store_with_docs(8)
        rr = HaystackReranker(store, top_k=5)
        q = _emb()
        candidates = store.embedding_retrieval(q.tolist(), top_k=8)
        result = await rr.async_run(q, candidates, top_k=2)
        self.assertEqual(len(result["documents"]), 2)


if __name__ == "__main__":
    unittest.main()
