"""Tests for LlamaIndex VectorStore persistence (save / load)."""

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
# Minimal llama-index stubs
# ---------------------------------------------------------------------------


def _inject_llamaindex_stub():
    li = types.ModuleType("llama_index")
    core = types.ModuleType("llama_index.core")
    schema = types.ModuleType("llama_index.core.schema")
    vs_types = types.ModuleType("llama_index.core.vector_stores.types")

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

    for mod_name, mod in [
        ("llama_index", li),
        ("llama_index.core", core),
        ("llama_index.core.schema", schema),
        ("llama_index.core.vector_stores", types.ModuleType("llama_index.core.vector_stores")),
        ("llama_index.core.vector_stores.types", vs_types),
    ]:
        sys.modules.setdefault(mod_name, mod)

    return _TextNode, _VectorStoreQuery, _VectorStoreQueryResult


_TextNode, _VStoreQuery, _VStoreResult = _inject_llamaindex_stub()

from python.integrations.llamaindex_integration import VectroVectorStore  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(55)


def _node(text: str = "text", dim: int = 64, meta: Optional[dict] = None) -> _TextNode:
    emb = RNG.standard_normal(dim).astype(np.float32).tolist()
    return _TextNode(text=text, id_=str(uuid.uuid4()), embedding=emb, metadata=meta or {})


def _nodes(n: int, dim: int = 64) -> List[_TextNode]:
    return [_node(f"node-{i}", dim=dim) for i in range(n)]


class TestLlamaIndexPersistence(unittest.TestCase):
    def _build_store(self, n: int = 8, dim: int = 64) -> VectroVectorStore:
        store = VectroVectorStore(compression_profile="balanced")
        store.add(_nodes(n, dim))
        return store

    def test_save_creates_files(self):
        import tempfile
        import os

        store = self._build_store()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "li_store")
            store.save(path)
            self.assertTrue(os.path.isfile(os.path.join(path, "meta.json")))
            self.assertTrue(os.path.isfile(os.path.join(path, "vectors.npy")))

    def test_load_restores_node_count(self):
        import tempfile
        import os

        store = self._build_store(n=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "li_store")
            store.save(path)
            loaded = VectroVectorStore.load(path)
            self.assertEqual(len(loaded), 10)

    def test_load_restores_node_ids(self):
        import tempfile
        import os

        store = self._build_store(n=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "li_store")
            store.save(path)
            loaded = VectroVectorStore.load(path)
            self.assertEqual(sorted(loaded._node_ids), sorted(store._node_ids))

    def test_load_wrong_store_type_raises(self):
        import tempfile
        import os
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad")
            os.makedirs(path)
            with open(os.path.join(path, "meta.json"), "w") as f:
                json.dump({"version": 1, "store_type": "haystack"}, f)
            with self.assertRaises(ValueError):
                VectroVectorStore.load(path)

    def test_get_nodes_after_load(self):
        import tempfile
        import os

        original_nodes = _nodes(4, dim=64)
        store = VectroVectorStore()
        store.add(original_nodes)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "li_store")
            store.save(path)
            loaded = VectroVectorStore.load(path)

        ids = [n.id_ for n in original_nodes]
        retrieved = loaded.get_nodes(ids)
        self.assertEqual(len(retrieved), 4)

    def test_query_after_load(self):
        import tempfile
        import os

        dim = 64
        original_nodes = _nodes(8, dim=dim)
        store = VectroVectorStore()
        store.add(original_nodes)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "li_store")
            store.save(path)
            loaded = VectroVectorStore.load(path)

        q_emb = RNG.standard_normal(dim).astype(np.float32).tolist()
        query = _VStoreQuery(query_embedding=q_emb, similarity_top_k=3)
        result = loaded.query(query)
        self.assertEqual(len(result.nodes), 3)

    def test_save_empty_store(self):
        import tempfile
        import os

        store = VectroVectorStore()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty")
            store.save(path)  # must not raise

    def test_text_preserved_after_load(self):
        import tempfile
        import os

        nodes = [_node(text=f"content-{i}", dim=32) for i in range(4)]
        store = VectroVectorStore()
        store.add(nodes)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "li_store")
            store.save(path)
            loaded = VectroVectorStore.load(path)

        orig_texts = sorted(t for t, _ in store._node_store.values())
        loaded_texts = sorted(t for t, _ in loaded._node_store.values())
        self.assertEqual(orig_texts, loaded_texts)

    def test_metadata_preserved_after_load(self):
        import tempfile
        import os

        nodes = [_node(text=f"n-{i}", dim=32, meta={"idx": i}) for i in range(4)]
        store = VectroVectorStore()
        store.add(nodes)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "li_store")
            store.save(path)
            loaded = VectroVectorStore.load(path)

        orig_metas = sorted(m["idx"] for _, m in store._node_store.values())
        loaded_metas = sorted(m["idx"] for _, m in loaded._node_store.values())
        self.assertEqual(orig_metas, loaded_metas)

    def test_compression_profile_preserved(self):
        import tempfile
        import os

        store = VectroVectorStore(compression_profile="quality")
        store.add(_nodes(4))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "li_store")
            store.save(path)
            loaded = VectroVectorStore.load(path)
        self.assertEqual(loaded._profile, "quality")


if __name__ == "__main__":
    unittest.main()
