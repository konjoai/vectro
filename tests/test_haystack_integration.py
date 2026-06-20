"""Tests for VectroDocumentStore (Haystack 2.x adapter).

All tests use stub classes injected via sys.modules so that haystack-ai
is not required to run the suite.
"""

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
# Minimal haystack stub injected before importing the integration
# ---------------------------------------------------------------------------


@dataclass
class _Document:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Optional[str] = None
    embedding: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    # Expose __dataclass_fields__ for the integration helper functions
    __dataclass_fields__ = {
        "id": None,
        "content": None,
        "embedding": None,
        "meta": None,
        "score": None,
    }

    def __post_init__(self):
        # Make __dataclass_fields__ an instance-visible attribute too
        pass


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

RNG = np.random.default_rng(7)


def _doc(content: str = "text", dim: int = 64, meta: Optional[dict] = None) -> _Document:
    emb = RNG.standard_normal(dim).astype(np.float32).tolist()
    return _Document(
        id=str(uuid.uuid4()),
        content=content,
        embedding=emb,
        meta=meta or {},
    )


def _docs(n: int, dim: int = 64) -> List[_Document]:
    return [_doc(content=f"doc-{i}", dim=dim) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVectroDocumentStoreBasic(unittest.TestCase):
    def setUp(self):
        self.store = VectroDocumentStore(compression_profile="balanced")

    def test_empty_store_count(self):
        self.assertEqual(self.store.count_documents(), 0)

    def test_write_returns_count(self):
        n = self.store.write_documents(_docs(5))
        self.assertEqual(n, 5)
        self.assertEqual(self.store.count_documents(), 5)

    def test_write_empty_list(self):
        n = self.store.write_documents([])
        self.assertEqual(n, 0)

    def test_write_no_embedding_skipped(self):
        doc = _Document(id="x", content="no emb")
        n = self.store.write_documents([doc])
        self.assertEqual(n, 0)
        self.assertEqual(self.store.count_documents(), 0)

    def test_filter_documents_all(self):
        self.store.write_documents(_docs(4))
        results = self.store.filter_documents()
        self.assertEqual(len(results), 4)

    def test_filter_documents_metadata(self):
        doc_a = _doc(content="tagged", meta={"src": "wiki"})
        doc_b = _doc(content="other", meta={"src": "news"})
        self.store.write_documents([doc_a, doc_b])
        results = self.store.filter_documents({"src": "wiki"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "tagged")

    def test_get_documents_by_id(self):
        docs = _docs(3)
        self.store.write_documents(docs)
        ids = [docs[0].id, docs[2].id]
        results = self.store.get_documents_by_id(ids)
        self.assertEqual(len(results), 2)

    def test_delete_by_ids(self):
        docs = _docs(4)
        self.store.write_documents(docs)
        self.store.delete_documents([docs[0].id, docs[1].id])
        self.assertEqual(self.store.count_documents(), 2)

    def test_delete_all(self):
        self.store.write_documents(_docs(4))
        self.store.delete_documents(None)
        self.assertEqual(self.store.count_documents(), 0)

    def test_delete_nonexistent_noop(self):
        self.store.write_documents(_docs(2))
        self.store.delete_documents(["no-such-id"])
        self.assertEqual(self.store.count_documents(), 2)

    def test_len(self):
        self.store.write_documents(_docs(6))
        self.assertEqual(len(self.store), 6)

    def test_repr(self):
        r = repr(self.store)
        self.assertIn("VectroDocumentStore", r)


class TestVectroDocumentStorePolicies(unittest.TestCase):
    def setUp(self):
        self.store = VectroDocumentStore()

    def test_policy_none_skips_duplicate(self):
        doc = _doc()
        self.store.write_documents([doc])
        n = self.store.write_documents([doc], policy="none")
        self.assertEqual(n, 0)
        self.assertEqual(self.store.count_documents(), 1)

    def test_policy_overwrite_replaces(self):
        doc = _doc(content="original")
        self.store.write_documents([doc])
        doc2 = _Document(
            id=doc.id,
            content="updated",
            embedding=RNG.standard_normal(64).astype(np.float32).tolist(),
        )
        n = self.store.write_documents([doc2], policy="overwrite")
        self.assertEqual(n, 1)
        self.assertEqual(self.store.count_documents(), 1)

    def test_policy_fail_raises(self):
        doc = _doc()
        self.store.write_documents([doc])
        with self.assertRaises(ValueError):
            self.store.write_documents([doc], policy="fail")


class TestVectroDocumentStoreRetrieval(unittest.TestCase):
    def setUp(self):
        self.store = VectroDocumentStore(compression_profile="balanced")
        self.dim = 64

    def test_embedding_retrieval_returns_top_k(self):
        self.store.write_documents(_docs(10, self.dim))
        q = RNG.standard_normal(self.dim).astype(np.float32).tolist()
        results = self.store.embedding_retrieval(q, top_k=3)
        self.assertEqual(len(results), 3)

    def test_embedding_retrieval_scores_set(self):
        self.store.write_documents(_docs(5, self.dim))
        q = RNG.standard_normal(self.dim).astype(np.float32).tolist()
        results = self.store.embedding_retrieval(q, top_k=2)
        for doc in results:
            self.assertIsNotNone(doc.score)
            self.assertGreater(doc.score, -1.0)

    def test_embedding_retrieval_ordered_by_score(self):
        self.store.write_documents(_docs(8, self.dim))
        q = RNG.standard_normal(self.dim).astype(np.float32).tolist()
        results = self.store.embedding_retrieval(q, top_k=5)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_embedding_retrieval_empty_store(self):
        q = RNG.standard_normal(self.dim).astype(np.float32).tolist()
        results = self.store.embedding_retrieval(q, top_k=3)
        self.assertEqual(results, [])

    def test_embedding_retrieval_with_filter(self):
        docs = [
            _doc(content="tagged", meta={"cat": "A"}, dim=self.dim),
            _doc(content="other", meta={"cat": "B"}, dim=self.dim),
            _doc(content="another tagged", meta={"cat": "A"}, dim=self.dim),
        ]
        self.store.write_documents(docs)
        q = RNG.standard_normal(self.dim).astype(np.float32).tolist()
        results = self.store.embedding_retrieval(q, top_k=5, filters={"cat": "A"})
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r.meta.get("cat"), "A")

    def test_embedding_retrieval_return_embedding(self):
        self.store.write_documents(_docs(3, self.dim))
        q = RNG.standard_normal(self.dim).astype(np.float32).tolist()
        results = self.store.embedding_retrieval(q, top_k=2, return_embedding=True)
        for doc in results:
            self.assertIsNotNone(doc.embedding)
            self.assertEqual(len(doc.embedding), self.dim)

    def test_cosine_floor_int8(self):
        """Retrieval must respect INT8 cosine similarity floor ≥ 0.9999."""
        rng = np.random.default_rng(42)
        dim = 128
        n = 32
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        docs = [
            _Document(
                id=str(i),
                content=f"doc-{i}",
                embedding=vecs[i].tolist(),
                meta={},
            )
            for i in range(n)
        ]
        store = VectroDocumentStore(compression_profile="balanced")
        store.write_documents(docs)

        results = store.embedding_retrieval(vecs[0].tolist(), top_k=n)
        result_ids = [r.id for r in results]
        # The query vector's own document (id="0") must be in top results
        self.assertIn("0", result_ids[:5])


class TestVectroDocumentStoreCompressionStats(unittest.TestCase):
    def test_empty_stats(self):
        store = VectroDocumentStore()
        stats = store.compression_stats
        self.assertEqual(stats["n_documents"], 0)

    def test_populated_stats(self):
        store = VectroDocumentStore()
        store.write_documents(_docs(20, dim=128))
        stats = store.compression_stats
        self.assertEqual(stats["n_documents"], 20)
        self.assertGreater(stats["compression_ratio"], 1.0)
        self.assertIn("original_mb", stats)
        self.assertIn("compressed_mb", stats)


class TestVectroDocumentStorePersistence(unittest.TestCase):
    def test_save_and_load(self, tmp_path=None):
        import tempfile
        import os

        store = VectroDocumentStore(compression_profile="balanced")
        store.write_documents(_docs(8, dim=64))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store")
            store.save(path)
            self.assertTrue(os.path.isfile(os.path.join(path, "meta.json")))
            self.assertTrue(os.path.isfile(os.path.join(path, "vectors.npy")))

            loaded = VectroDocumentStore.load(path)
            self.assertEqual(loaded.count_documents(), 8)

    def test_load_wrong_type_raises(self):
        import tempfile
        import os
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad")
            os.makedirs(path)
            with open(os.path.join(path, "meta.json"), "w") as f:
                json.dump({"version": 1, "store_type": "langchain"}, f)
            with self.assertRaises(ValueError):
                VectroDocumentStore.load(path)

    def test_persistence_retrieval_quality(self):
        import tempfile
        import os

        rng = np.random.default_rng(99)
        dim = 64
        n = 16
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        docs = [
            _Document(
                id=str(i),
                content=f"doc-{i}",
                embedding=vecs[i].tolist(),
                meta={},
            )
            for i in range(n)
        ]
        store = VectroDocumentStore()
        store.write_documents(docs)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s")
            store.save(path)
            loaded = VectroDocumentStore.load(path)

        q = vecs[0].tolist()
        orig_results = store.embedding_retrieval(q, top_k=3)
        loaded_results = loaded.embedding_retrieval(q, top_k=3)

        # Top result should be the same document in both
        self.assertEqual(orig_results[0].id, loaded_results[0].id)


if __name__ == "__main__":
    unittest.main()
