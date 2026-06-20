"""Tests for LlamaIndex VectorStore async methods (v4.15.0)."""

from __future__ import annotations

import asyncio
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


def _inject_llamaindex_stub():
    li = sys.modules.get("llama_index") or types.ModuleType("llama_index")
    core = sys.modules.get("llama_index.core") or types.ModuleType("llama_index.core")
    schema = sys.modules.get("llama_index.core.schema") or types.ModuleType(
        "llama_index.core.schema"
    )
    vs_types = sys.modules.get("llama_index.core.vector_stores.types") or types.ModuleType(
        "llama_index.core.vector_stores.types"
    )

    @dataclass
    class _TextNode:
        text: str = ""
        id_: str = field(default_factory=lambda: str(uuid.uuid4()))
        metadata: Dict[str, Any] = field(default_factory=dict)
        embedding: Optional[List[float]] = None

        @property
        def node_id(self):
            return self.id_

    @dataclass
    class _VectorStoreQuery:
        query_embedding: Optional[List[float]] = None
        similarity_top_k: int = 4

    @dataclass
    class _VectorStoreQueryResult:
        nodes: List[Any] = field(default_factory=list)
        similarities: List[float] = field(default_factory=list)
        ids: List[str] = field(default_factory=list)

    schema.TextNode = _TextNode
    schema.NodeWithScore = object
    vs_types.VectorStoreQuery = _VectorStoreQuery
    vs_types.VectorStoreQueryResult = _VectorStoreQueryResult
    core.schema = schema
    li.core = core

    vs_parent = types.ModuleType("llama_index.core.vector_stores")
    for mod_name, mod in [
        ("llama_index", li),
        ("llama_index.core", core),
        ("llama_index.core.schema", schema),
        ("llama_index.core.vector_stores", vs_parent),
        ("llama_index.core.vector_stores.types", vs_types),
    ]:
        sys.modules.setdefault(mod_name, mod)

    return _TextNode, _VectorStoreQuery, _VectorStoreQueryResult


_TextNode, _VStoreQuery, _VStoreResult = _inject_llamaindex_stub()

from python.integrations.llamaindex_integration import VectroVectorStore  # noqa: E402

RNG = np.random.default_rng(99)


def _node(text: str = "text", dim: int = 64) -> _TextNode:
    emb = RNG.standard_normal(dim).astype(np.float32).tolist()
    return _TextNode(text=text, id_=str(uuid.uuid4()), embedding=emb)


def _nodes(n: int, dim: int = 64) -> List[_TextNode]:
    return [_node(f"node-{i}", dim=dim) for i in range(n)]


class TestLlamaIndexAsync(unittest.TestCase):
    def _build_store(self, n: int = 8, dim: int = 64) -> VectroVectorStore:
        store = VectroVectorStore(compression_profile="balanced")
        store.add(_nodes(n, dim))
        return store

    def test_async_add_returns_ids(self):
        store = VectroVectorStore()
        nodes = _nodes(4)

        async def _run():
            return await store.async_add(nodes)

        ids = asyncio.run(_run())
        self.assertEqual(len(ids), 4)

    def test_async_add_increments_count(self):
        store = VectroVectorStore()
        nodes = _nodes(6)

        async def _run():
            await store.async_add(nodes)

        asyncio.run(_run())
        self.assertEqual(len(store), 6)

    def test_aquery_returns_result(self):
        store = self._build_store()
        q_emb = RNG.standard_normal(64).astype(np.float32).tolist()
        query = _VStoreQuery(query_embedding=q_emb, similarity_top_k=3)

        async def _run():
            return await store.aquery(query)

        result = asyncio.run(_run())
        self.assertEqual(len(result.nodes), 3)

    def test_aquery_empty_store(self):
        store = VectroVectorStore()
        q_emb = RNG.standard_normal(64).astype(np.float32).tolist()
        query = _VStoreQuery(query_embedding=q_emb, similarity_top_k=3)

        async def _run():
            return await store.aquery(query)

        result = asyncio.run(_run())
        self.assertEqual(result.nodes, [])

    def test_aquery_scores_present(self):
        store = self._build_store()
        q_emb = RNG.standard_normal(64).astype(np.float32).tolist()
        query = _VStoreQuery(query_embedding=q_emb, similarity_top_k=3)

        async def _run():
            return await store.aquery(query)

        result = asyncio.run(_run())
        for score in result.similarities:
            self.assertIsInstance(score, float)

    def test_async_add_then_aquery(self):
        store = VectroVectorStore()
        nodes = _nodes(8)

        async def _run():
            await store.async_add(nodes)
            q_emb = RNG.standard_normal(64).astype(np.float32).tolist()
            return await store.aquery(_VStoreQuery(query_embedding=q_emb, similarity_top_k=4))

        result = asyncio.run(_run())
        self.assertEqual(len(result.nodes), 4)

    def test_concurrent_async_add(self):
        store = VectroVectorStore()
        batches = [_nodes(4) for _ in range(3)]

        async def _run():
            import asyncio as _asyncio

            await _asyncio.gather(*[store.async_add(b) for b in batches])

        asyncio.run(_run())
        # Due to sequential lock, should have 12 total nodes
        self.assertEqual(len(store), 12)


if __name__ == "__main__":
    unittest.main()
