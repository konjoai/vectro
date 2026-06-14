"""Tests for python.integrations.llamaindex_integration.VectroVectorStore.

All tests use minimal stubs so llama-index-core is NOT required to run
the suite.  The adapter logic is exercised against the real Vectro stack.
"""

from __future__ import annotations

import sys
import types as _types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pytest

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

# ---------------------------------------------------------------------------
# Minimal llama-index stubs — injected before the module-under-test imports
# ---------------------------------------------------------------------------


@dataclass
class _TextNode:
    text: str = ""
    id_: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def node_id(self) -> str:
        return self.id_


@dataclass
class _VectorStoreQuery:
    query_embedding: Optional[List[float]] = None
    similarity_top_k: int = 4
    filters: Any = None


@dataclass
class _VectorStoreQueryResult:
    nodes: List[Any] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)


# Register stubs so any `from llama_index...` inside the module works.
_lli_core = _types.ModuleType("llama_index")
_lli_core_sub = _types.ModuleType("llama_index.core")
_lli_schema = _types.ModuleType("llama_index.core.schema")
_lli_schema.TextNode = _TextNode
_lli_schema.BaseNode = _TextNode
_lli_schema.NodeWithScore = object
_lli_vs = _types.ModuleType("llama_index.core.vector_stores")
_lli_vs_types = _types.ModuleType("llama_index.core.vector_stores.types")
_lli_vs_types.VectorStoreQuery = _VectorStoreQuery
_lli_vs_types.VectorStoreQueryResult = _VectorStoreQueryResult
_lli_vs_types.BasePydanticVectorStore = object
for mod_name, mod_obj in [
    ("llama_index", _lli_core),
    ("llama_index.core", _lli_core_sub),
    ("llama_index.core.schema", _lli_schema),
    ("llama_index.core.vector_stores", _lli_vs),
    ("llama_index.core.vector_stores.types", _lli_vs_types),
]:
    sys.modules.setdefault(mod_name, mod_obj)

from python.integrations.llamaindex_integration import VectroVectorStore  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(7)
DIM = 64
FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _make_nodes(n: int = 5, dim: int = DIM) -> List[_TextNode]:
    nodes = []
    for i in range(n):
        v = RNG.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-10
        nodes.append(
            _TextNode(
                text=f"Document text {i}",
                id_=f"node-{i}",
                metadata={"idx": i, "doc_id": f"doc-{i // 2}"},
                embedding=v.tolist(),
            )
        )
    return nodes


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty(self):
        store = VectroVectorStore()
        assert len(store) == 0

    def test_repr(self):
        store = VectroVectorStore()
        assert "LlamaIndex" in repr(store)

    def test_compression_profile_quality(self):
        store = VectroVectorStore(compression_profile="quality")
        nodes = _make_nodes(3)
        store.add(nodes)
        assert len(store) == 3

    def test_model_dir_forwarded(self):
        store = VectroVectorStore(model_dir=str(FIXTURE_DIR / "gte"))
        store.add(_make_nodes(2))
        assert len(store) == 2


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_returns_ids(self):
        store = VectroVectorStore()
        nodes = _make_nodes(3)
        ids = store.add(nodes)
        assert len(ids) == 3
        assert ids == [n.node_id for n in nodes]

    def test_empty_add_noop(self):
        store = VectroVectorStore()
        ids = store.add([])
        assert ids == []

    def test_node_without_embedding_raises(self):
        store = VectroVectorStore()
        bad_node = _TextNode(text="no embedding", id_="bad")
        with pytest.raises(ValueError, match="no embedding"):
            store.add([bad_node])

    def test_incremental_add(self):
        store = VectroVectorStore()
        store.add(_make_nodes(2))
        store.add(_make_nodes(3))
        assert len(store) == 5


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


class TestQuery:
    def setup_method(self):
        self.nodes = _make_nodes(6)
        self.store = VectroVectorStore()
        self.store.add(self.nodes)

    def test_returns_result_object(self):
        q_emb = RNG.standard_normal(DIM).astype(np.float32).tolist()
        result = self.store.query(_VectorStoreQuery(query_embedding=q_emb, similarity_top_k=3))
        assert hasattr(result, "nodes")
        assert hasattr(result, "similarities")
        assert hasattr(result, "ids")

    def test_k_results_returned(self):
        q_emb = RNG.standard_normal(DIM).astype(np.float32).tolist()
        result = self.store.query(_VectorStoreQuery(query_embedding=q_emb, similarity_top_k=3))
        assert len(result.nodes) == 3
        assert len(result.similarities) == 3

    def test_similarities_descending(self):
        q_emb = RNG.standard_normal(DIM).astype(np.float32).tolist()
        result = self.store.query(_VectorStoreQuery(query_embedding=q_emb, similarity_top_k=4))
        sims = result.similarities
        assert sims == sorted(sims, reverse=True)

    def test_similarity_scores_in_range(self):
        q_emb = RNG.standard_normal(DIM).astype(np.float32).tolist()
        result = self.store.query(_VectorStoreQuery(query_embedding=q_emb, similarity_top_k=4))
        for s in result.similarities:
            assert -1.0 <= s <= 1.0 + 1e-5

    def test_result_nodes_have_text(self):
        q_emb = RNG.standard_normal(DIM).astype(np.float32).tolist()
        result = self.store.query(_VectorStoreQuery(query_embedding=q_emb, similarity_top_k=2))
        for node in result.nodes:
            assert node.text.startswith("Document text")

    def test_empty_store_returns_empty(self):
        store = VectroVectorStore()
        q_emb = RNG.standard_normal(DIM).astype(np.float32).tolist()
        result = store.query(_VectorStoreQuery(query_embedding=q_emb))
        assert result.nodes == []

    def test_missing_query_embedding_raises(self):
        with pytest.raises(ValueError, match="query_embedding"):
            self.store.query(_VectorStoreQuery(query_embedding=None))

    def test_k_capped_at_store_size(self):
        q_emb = RNG.standard_normal(DIM).astype(np.float32).tolist()
        result = self.store.query(_VectorStoreQuery(query_embedding=q_emb, similarity_top_k=100))
        assert len(result.nodes) == len(self.nodes)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_by_doc_id(self):
        store = VectroVectorStore()
        nodes = _make_nodes(4)  # doc_id: doc-0 (nodes 0,1), doc-1 (nodes 2,3)
        store.add(nodes)
        store.delete("doc-0")
        assert len(store) == 2

    def test_delete_by_node_id(self):
        store = VectroVectorStore()
        nodes = _make_nodes(3)
        store.add(nodes)
        store.delete("node-0")
        assert len(store) == 2

    def test_delete_nonexistent_noop(self):
        store = VectroVectorStore()
        store.add(_make_nodes(2))
        store.delete("nonexistent")
        assert len(store) == 2


# ---------------------------------------------------------------------------
# get_nodes
# ---------------------------------------------------------------------------


class TestGetNodes:
    def test_get_all_nodes(self):
        store = VectroVectorStore()
        nodes = _make_nodes(3)
        store.add(nodes)
        retrieved = store.get_nodes()
        assert len(retrieved) == 3

    def test_get_by_ids(self):
        store = VectroVectorStore()
        nodes = _make_nodes(4)
        store.add(nodes)
        retrieved = store.get_nodes(node_ids=["node-0", "node-2"])
        assert len(retrieved) == 2

    def test_get_unknown_id_skipped(self):
        store = VectroVectorStore()
        store.add(_make_nodes(2))
        retrieved = store.get_nodes(node_ids=["node-0", "ghost"])
        assert len(retrieved) == 1


# ---------------------------------------------------------------------------
# compression_stats
# ---------------------------------------------------------------------------


class TestCompressionStats:
    def test_empty(self):
        store = VectroVectorStore()
        stats = store.compression_stats
        assert stats["n_nodes"] == 0

    def test_stats_after_add(self):
        store = VectroVectorStore()
        store.add(_make_nodes(5))
        stats = store.compression_stats
        assert stats["n_nodes"] == 5
        assert stats["dimensions"] == DIM
        assert stats["compression_ratio"] >= 1.0


# ---------------------------------------------------------------------------
# Import from top-level package
# ---------------------------------------------------------------------------


def test_importable_from_package():
    from python import LlamaIndexVectorStore

    assert LlamaIndexVectorStore is VectroVectorStore
