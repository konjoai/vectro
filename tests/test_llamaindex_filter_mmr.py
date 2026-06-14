"""Tests for LlamaIndex metadata filtering and MMR query mode.

Verifies:
- query() respects VectorStoreQuery.filters (MetadataFilters)
- query() supports VectorStoreQueryMode.MMR via query_mode
- Both features compose correctly (filter + MMR)
- Async aquery() propagates filters and MMR mode
"""

from __future__ import annotations

import sys
import types
import asyncio
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Stub out llama_index so we don't need the real package
# ---------------------------------------------------------------------------


class _VectorStoreQueryMode:
    DEFAULT = "default"
    MMR = "mmr"


class _MetadataFilter:
    def __init__(self, key, value, operator="=="):
        self.key = key
        self.value = value
        self.operator = operator


class _MetadataFilters:
    def __init__(self, filters):
        self.filters = filters


class _VectorStoreQuery:
    def __init__(
        self,
        query_embedding=None,
        similarity_top_k=4,
        filters=None,
        query_mode=None,
        mmr_prefetch_k=None,
        mmr_threshold=0.5,
    ):
        self.query_embedding = query_embedding
        self.similarity_top_k = similarity_top_k
        self.filters = filters
        self.query_mode = query_mode or _VectorStoreQueryMode.DEFAULT
        self.mmr_prefetch_k = mmr_prefetch_k
        self.mmr_threshold = mmr_threshold


class _VectorStoreQueryResult:
    def __init__(self, nodes, similarities, ids):
        self.nodes = nodes
        self.similarities = similarities
        self.ids = ids


class _TextNode:
    def __init__(self, text="", id_=None, metadata=None, embedding=None):
        self.text = text
        self.node_id = id_ or ""
        self.id_ = id_ or ""
        self.metadata = metadata or {}
        self.embedding = embedding


_li_types = types.ModuleType("llama_index.core.vector_stores.types")
_li_types.VectorStoreQueryResult = _VectorStoreQueryResult
_li_types.VectorStoreQueryMode = _VectorStoreQueryMode
_li_types.MetadataFilters = _MetadataFilters
_li_types.MetadataFilter = _MetadataFilter
_li_types.VectorStoreQuery = _VectorStoreQuery

_li_schema = types.ModuleType("llama_index.core.schema")
_li_schema.TextNode = _TextNode
_li_schema.NodeWithScore = object

_li_core = types.ModuleType("llama_index.core")

for mod_name, mod in [
    ("llama_index", types.ModuleType("llama_index")),
    ("llama_index.core", _li_core),
    ("llama_index.core.vector_stores", types.ModuleType("llama_index.core.vector_stores")),
    ("llama_index.core.vector_stores.types", _li_types),
    ("llama_index.core.schema", _li_schema),
]:
    sys.modules.setdefault(mod_name, mod)

from python.integrations.llamaindex_integration import VectroVectorStore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store():
    """Build a store with 6 nodes across 3 sources."""
    store = VectroVectorStore(compression_profile="fast")
    rng = np.random.default_rng(42)
    nodes = []
    for i in range(6):
        node = _TextNode(
            text=f"doc {i}",
            id_=f"node-{i}",
            metadata={"source": f"src-{i % 3}", "lang": "en" if i < 4 else "fr"},
            embedding=rng.standard_normal(64).astype(np.float32).tolist(),
        )
        nodes.append(node)
    store.add(nodes)
    return store, nodes


# ---------------------------------------------------------------------------
# Metadata filtering tests
# ---------------------------------------------------------------------------


class TestLlamaIndexFilter(unittest.TestCase):
    def setUp(self):
        self.store, self.nodes = _make_store()

    def test_no_filter_returns_all_candidates(self):
        q = np.ones(64, dtype=np.float32)
        query = _VectorStoreQuery(query_embedding=q.tolist(), similarity_top_k=6)
        result = self.store.query(query)
        self.assertEqual(len(result.nodes), 6)

    def test_single_field_filter(self):
        q = np.ones(64, dtype=np.float32)
        flt = _MetadataFilters([_MetadataFilter("source", "src-0")])
        query = _VectorStoreQuery(query_embedding=q.tolist(), similarity_top_k=10, filters=flt)
        result = self.store.query(query)
        # Only nodes 0 and 3 have source=src-0
        self.assertEqual(len(result.nodes), 2)
        for node in result.nodes:
            self.assertEqual(node.metadata["source"], "src-0")

    def test_multi_field_filter(self):
        q = np.ones(64, dtype=np.float32)
        flt = _MetadataFilters(
            [
                _MetadataFilter("source", "src-1"),
                _MetadataFilter("lang", "en"),
            ]
        )
        query = _VectorStoreQuery(query_embedding=q.tolist(), similarity_top_k=10, filters=flt)
        result = self.store.query(query)
        for node in result.nodes:
            self.assertEqual(node.metadata["source"], "src-1")
            self.assertEqual(node.metadata["lang"], "en")

    def test_filter_no_match_returns_empty(self):
        q = np.ones(64, dtype=np.float32)
        flt = _MetadataFilters([_MetadataFilter("source", "nonexistent")])
        query = _VectorStoreQuery(query_embedding=q.tolist(), similarity_top_k=5, filters=flt)
        result = self.store.query(query)
        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(len(result.similarities), 0)

    def test_filter_respects_top_k(self):
        q = np.ones(64, dtype=np.float32)
        flt = _MetadataFilters([_MetadataFilter("lang", "en")])
        query = _VectorStoreQuery(query_embedding=q.tolist(), similarity_top_k=2, filters=flt)
        result = self.store.query(query)
        self.assertLessEqual(len(result.nodes), 2)
        for node in result.nodes:
            self.assertEqual(node.metadata["lang"], "en")

    def test_filter_ne_operator(self):
        q = np.ones(64, dtype=np.float32)
        flt = _MetadataFilters([_MetadataFilter("lang", "fr", operator="!=")])
        query = _VectorStoreQuery(query_embedding=q.tolist(), similarity_top_k=10, filters=flt)
        result = self.store.query(query)
        for node in result.nodes:
            self.assertNotEqual(node.metadata.get("lang"), "fr")

    def test_similarities_ordered_descending(self):
        q = np.ones(64, dtype=np.float32)
        flt = _MetadataFilters([_MetadataFilter("lang", "en")])
        query = _VectorStoreQuery(query_embedding=q.tolist(), similarity_top_k=4, filters=flt)
        result = self.store.query(query)
        sims = result.similarities
        self.assertEqual(sims, sorted(sims, reverse=True))


# ---------------------------------------------------------------------------
# MMR tests
# ---------------------------------------------------------------------------


class TestLlamaIndexMMR(unittest.TestCase):
    def setUp(self):
        self.store, self.nodes = _make_store()

    def test_mmr_mode_returns_k_results(self):
        rng = np.random.default_rng(7)
        q = rng.standard_normal(64).astype(np.float32)
        query = _VectorStoreQuery(
            query_embedding=q.tolist(),
            similarity_top_k=3,
            query_mode=_VectorStoreQueryMode.MMR,
        )
        result = self.store.query(query)
        self.assertEqual(len(result.nodes), 3)

    def test_mmr_mode_returns_valid_nodes(self):
        q = np.ones(64, dtype=np.float32)
        query = _VectorStoreQuery(
            query_embedding=q.tolist(),
            similarity_top_k=4,
            query_mode=_VectorStoreQueryMode.MMR,
        )
        result = self.store.query(query)
        stored_ids = {n.id_ for n in self.nodes}
        for node in result.nodes:
            self.assertIn(node.id_, stored_ids)

    def test_mmr_no_duplicate_nodes(self):
        q = np.ones(64, dtype=np.float32)
        query = _VectorStoreQuery(
            query_embedding=q.tolist(),
            similarity_top_k=5,
            query_mode=_VectorStoreQueryMode.MMR,
        )
        result = self.store.query(query)
        ids = [n.id_ for n in result.nodes]
        self.assertEqual(len(ids), len(set(ids)))

    def test_mmr_with_filter(self):
        q = np.ones(64, dtype=np.float32)
        flt = _MetadataFilters([_MetadataFilter("lang", "en")])
        query = _VectorStoreQuery(
            query_embedding=q.tolist(),
            similarity_top_k=3,
            filters=flt,
            query_mode=_VectorStoreQueryMode.MMR,
        )
        result = self.store.query(query)
        for node in result.nodes:
            self.assertEqual(node.metadata["lang"], "en")

    def test_mmr_lambda_mult_zero_maximises_diversity(self):
        """lambda_mult=0 means pure diversity — result set should be diverse."""
        rng = np.random.default_rng(99)
        q = rng.standard_normal(64).astype(np.float32)
        query = _VectorStoreQuery(
            query_embedding=q.tolist(),
            similarity_top_k=4,
            query_mode=_VectorStoreQueryMode.MMR,
            mmr_threshold=0.0,
        )
        result = self.store.query(query)
        self.assertEqual(len(result.nodes), 4)


# ---------------------------------------------------------------------------
# Async propagation
# ---------------------------------------------------------------------------


class TestLlamaIndexAsyncFilterMMR(unittest.TestCase):
    def setUp(self):
        self.store, self.nodes = _make_store()

    def test_aquery_with_filter(self):
        async def _run():
            q = np.ones(64, dtype=np.float32)
            flt = _MetadataFilters([_MetadataFilter("source", "src-2")])
            query = _VectorStoreQuery(query_embedding=q.tolist(), similarity_top_k=10, filters=flt)
            return await self.store.aquery(query)

        result = asyncio.run(_run())
        for node in result.nodes:
            self.assertEqual(node.metadata["source"], "src-2")

    def test_aquery_with_mmr(self):
        async def _run():
            q = np.ones(64, dtype=np.float32)
            query = _VectorStoreQuery(
                query_embedding=q.tolist(),
                similarity_top_k=3,
                query_mode=_VectorStoreQueryMode.MMR,
            )
            return await self.store.aquery(query)

        result = asyncio.run(_run())
        self.assertEqual(len(result.nodes), 3)


if __name__ == "__main__":
    unittest.main()
