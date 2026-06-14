"""Tests for Haystack async methods.

Covers:
- async_embedding_retrieval — top-k ANN with optional metadata filter
- async_write_documents — write with duplicate policies
- Concurrent async operations
- Empty store safety
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import types
import unittest
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Stub Haystack document type (dataclass so _doc_to_dict works)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Document:
    id: str = ""
    content: str = ""
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: Optional[float] = None

    @property
    def __dataclass_fields__(self):  # accessed by haystack helpers
        return {f.name: f for f in dataclasses.fields(self)}


_haystack_mod = types.ModuleType("haystack")
_haystack_dc = types.ModuleType("haystack.dataclasses")
_haystack_dc.Document = _Document
_haystack_mod.dataclasses = _haystack_dc

for mod_name, mod in [
    ("haystack", _haystack_mod),
    ("haystack.dataclasses", _haystack_dc),
]:
    sys.modules.setdefault(mod_name, mod)

from python.integrations.haystack_integration import VectroDocumentStore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(n: int = 6, dim: int = 64) -> VectroDocumentStore:
    rng = np.random.default_rng(11)
    store = VectroDocumentStore(compression_profile="fast")
    docs = []
    for i in range(n):
        emb = rng.standard_normal(dim).astype(np.float32).tolist()
        docs.append(
            _Document(
                id=f"doc-{i}",
                content=f"content {i}",
                meta={"cat": f"cat-{i % 3}", "lang": "en" if i < 4 else "fr"},
                embedding=emb,
            )
        )
    store.write_documents(docs)
    return store


# ---------------------------------------------------------------------------
# async_embedding_retrieval
# ---------------------------------------------------------------------------


class TestHaystackAsyncEmbeddingRetrieval(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()

    def test_async_retrieval_returns_top_k(self):
        async def _run():
            q = np.ones(64, dtype=np.float32).tolist()
            return await self.store.async_embedding_retrieval(q, top_k=3)

        results = asyncio.run(_run())
        self.assertEqual(len(results), 3)

    def test_async_retrieval_scores_ordered(self):
        async def _run():
            q = np.ones(64, dtype=np.float32).tolist()
            return await self.store.async_embedding_retrieval(q, top_k=5)

        results = asyncio.run(_run())
        scores = [r.score for r in results if r.score is not None]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_async_retrieval_with_metadata_filter(self):
        async def _run():
            q = np.ones(64, dtype=np.float32).tolist()
            return await self.store.async_embedding_retrieval(q, top_k=10, filters={"cat": "cat-0"})

        results = asyncio.run(_run())
        for doc in results:
            self.assertEqual(doc.meta.get("cat"), "cat-0")

    def test_async_retrieval_return_embedding_flag(self):
        async def _run():
            q = np.ones(64, dtype=np.float32).tolist()
            return await self.store.async_embedding_retrieval(q, top_k=2, return_embedding=True)

        results = asyncio.run(_run())
        for doc in results:
            self.assertIsNotNone(doc.embedding)
            self.assertEqual(len(doc.embedding), 64)

    def test_async_retrieval_empty_store(self):
        async def _run():
            empty = VectroDocumentStore(compression_profile="fast")
            q = np.ones(64, dtype=np.float32).tolist()
            return await empty.async_embedding_retrieval(q, top_k=5)

        results = asyncio.run(_run())
        self.assertEqual(results, [])

    def test_async_retrieval_filter_no_match(self):
        async def _run():
            q = np.ones(64, dtype=np.float32).tolist()
            return await self.store.async_embedding_retrieval(
                q, top_k=10, filters={"cat": "nonexistent"}
            )

        results = asyncio.run(_run())
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# async_write_documents
# ---------------------------------------------------------------------------


class TestHaystackAsyncWriteDocuments(unittest.TestCase):
    def test_async_write_returns_count(self):
        async def _run():
            store = VectroDocumentStore(compression_profile="fast")
            rng = np.random.default_rng(77)
            docs = [
                _Document(
                    id=f"wd-{i}",
                    content=f"text {i}",
                    meta={},
                    embedding=rng.standard_normal(32).astype(np.float32).tolist(),
                )
                for i in range(4)
            ]
            return await store.async_write_documents(docs)

        count = asyncio.run(_run())
        self.assertEqual(count, 4)

    def test_async_write_documents_visible_after_write(self):
        async def _run():
            store = VectroDocumentStore(compression_profile="fast")
            rng = np.random.default_rng(88)
            docs = [
                _Document(
                    id=f"aw-{i}",
                    content=f"text {i}",
                    meta={},
                    embedding=rng.standard_normal(32).astype(np.float32).tolist(),
                )
                for i in range(3)
            ]
            await store.async_write_documents(docs)
            return store.count_documents()

        count = asyncio.run(_run())
        self.assertEqual(count, 3)

    def test_async_write_policy_overwrite(self):
        async def _run():
            store = VectroDocumentStore(compression_profile="fast")
            rng = np.random.default_rng(55)
            doc = _Document(
                id="overlap",
                content="original",
                meta={},
                embedding=rng.standard_normal(32).astype(np.float32).tolist(),
            )
            await store.async_write_documents([doc])
            doc2 = _Document(
                id="overlap",
                content="updated",
                meta={},
                embedding=rng.standard_normal(32).astype(np.float32).tolist(),
            )
            await store.async_write_documents([doc2], policy="overwrite")
            return store.count_documents()

        count = asyncio.run(_run())
        self.assertEqual(count, 1)


# ---------------------------------------------------------------------------
# Concurrent operations
# ---------------------------------------------------------------------------


class TestHaystackConcurrentAsync(unittest.TestCase):
    def test_concurrent_retrieval_and_write(self):
        async def _run():
            store = VectroDocumentStore(compression_profile="fast")
            rng = np.random.default_rng(44)

            # Seed some data first
            seed_docs = [
                _Document(
                    id=f"seed-{i}",
                    content=f"seed {i}",
                    meta={},
                    embedding=rng.standard_normal(32).astype(np.float32).tolist(),
                )
                for i in range(5)
            ]
            store.write_documents(seed_docs)

            q = np.ones(32, dtype=np.float32).tolist()
            results = await asyncio.gather(
                store.async_embedding_retrieval(q, top_k=3),
                store.async_embedding_retrieval(q, top_k=2),
            )
            return results

        res1, res2 = asyncio.run(_run())
        self.assertEqual(len(res1), 3)
        self.assertEqual(len(res2), 2)


if __name__ == "__main__":
    unittest.main()
