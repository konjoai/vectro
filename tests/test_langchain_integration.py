"""Tests for python.integrations.langchain_integration.VectroVectorStore.

All tests use a mock Embeddings object so langchain-core is NOT required
to run the suite.  The adapter logic (add, search, delete, async) is
exercised against the real Vectro compression stack.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.integrations.langchain_integration import VectroVectorStore  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
DIM = 128


def _random_emb(n: int = 1) -> np.ndarray:
    v = RNG.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class _FakeDocument:
    """Minimal Document stub — no langchain-core required."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class _FakeEmbeddings:
    """Deterministic embeddings based on text hash — no API calls."""

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-10
            result.append(v.tolist())
        return result

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# Patch langchain Document at import time so tests work without langchain-core
import types as _types

_fake_lc_docs = _types.ModuleType("langchain_core.documents")
_fake_lc_docs.Document = _FakeDocument
sys.modules.setdefault("langchain_core", _types.ModuleType("langchain_core"))
sys.modules.setdefault("langchain_core.documents", _fake_lc_docs)


TEXTS = [
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
    "Tokyo is the capital of Japan",
    "Madrid is the capital of Spain",
    "Rome is the capital of Italy",
]

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_store(self):
        store = VectroVectorStore(_FakeEmbeddings())
        assert len(store) == 0

    def test_from_texts(self):
        store = VectroVectorStore.from_texts(TEXTS, _FakeEmbeddings())
        assert len(store) == len(TEXTS)

    def test_repr(self):
        store = VectroVectorStore(_FakeEmbeddings())
        store.add_texts(["hello"])
        r = repr(store)
        assert "VectroVectorStore" in r
        assert "1" in r

    def test_compression_profile_quality(self):
        store = VectroVectorStore.from_texts(
            TEXTS, _FakeEmbeddings(), compression_profile="quality"
        )
        assert len(store) == len(TEXTS)

    def test_model_dir_gte_selects_int8(self):
        """model_dir should be forwarded to Vectro.compress (no error expected)."""
        store = VectroVectorStore.from_texts(
            TEXTS,
            _FakeEmbeddings(),
            model_dir=str(FIXTURE_DIR / "gte"),
        )
        assert len(store) == len(TEXTS)


# ---------------------------------------------------------------------------
# add_texts
# ---------------------------------------------------------------------------


class TestAddTexts:
    def test_returns_ids(self):
        store = VectroVectorStore(_FakeEmbeddings())
        ids = store.add_texts(TEXTS[:2])
        assert len(ids) == 2
        assert all(isinstance(i, str) for i in ids)

    def test_custom_ids(self):
        store = VectroVectorStore(_FakeEmbeddings())
        custom = ["doc-1", "doc-2"]
        ids = store.add_texts(TEXTS[:2], ids=custom)
        assert ids == custom

    def test_metadatas_stored(self):
        store = VectroVectorStore(_FakeEmbeddings())
        meta = [{"source": "wiki"}, {"source": "news"}]
        store.add_texts(TEXTS[:2], metadatas=meta)
        assert len(store) == 2

    def test_incremental_add(self):
        store = VectroVectorStore(_FakeEmbeddings())
        store.add_texts(TEXTS[:2])
        store.add_texts(TEXTS[2:4])
        assert len(store) == 4

    def test_empty_add_noop(self):
        store = VectroVectorStore(_FakeEmbeddings())
        ids = store.add_texts([])
        assert ids == []
        assert len(store) == 0


# ---------------------------------------------------------------------------
# similarity_search
# ---------------------------------------------------------------------------


class TestSimilaritySearch:
    def setup_method(self):
        self.store = VectroVectorStore.from_texts(TEXTS, _FakeEmbeddings())

    def test_returns_k_results(self):
        docs = self.store.similarity_search("European capitals", k=3)
        assert len(docs) == 3

    def test_returns_documents(self):
        docs = self.store.similarity_search("European capitals", k=2)
        for doc in docs:
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")

    def test_document_text_is_original_text(self):
        docs = self.store.similarity_search("Paris France", k=1)
        # The best match should be the Paris document
        assert docs[0].page_content in TEXTS

    def test_k_capped_at_store_size(self):
        docs = self.store.similarity_search("capital", k=100)
        assert len(docs) == len(TEXTS)

    def test_empty_store_returns_empty(self):
        empty = VectroVectorStore(_FakeEmbeddings())
        docs = empty.similarity_search("anything", k=3)
        assert docs == []

    def test_metadata_vectro_id_present(self):
        docs = self.store.similarity_search("capital", k=1)
        assert "_vectro_id" in docs[0].metadata


class TestSimilaritySearchWithScore:
    def setup_method(self):
        self.store = VectroVectorStore.from_texts(TEXTS, _FakeEmbeddings())

    def test_returns_tuples(self):
        results = self.store.similarity_search_with_score("capital city", k=2)
        assert len(results) == 2
        for doc, score in results:
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.0 + 1e-5

    def test_scores_descending(self):
        results = self.store.similarity_search_with_score("capital", k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevance_scores_alias(self):
        r1 = self.store.similarity_search_with_score("capital", k=2)
        r2 = self.store._similarity_search_with_relevance_scores("capital", k=2)
        assert len(r1) == len(r2)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_by_id(self):
        store = VectroVectorStore(_FakeEmbeddings())
        ids = store.add_texts(TEXTS)
        store.delete(ids=[ids[0]])
        assert len(store) == len(TEXTS) - 1

    def test_delete_all_none(self):
        store = VectroVectorStore.from_texts(TEXTS, _FakeEmbeddings())
        store.delete(ids=None)
        assert len(store) == 0

    def test_delete_nonexistent_id_noop(self):
        store = VectroVectorStore.from_texts(TEXTS[:2], _FakeEmbeddings())
        store.delete(ids=["nonexistent-id"])
        assert len(store) == 2

    def test_search_after_delete(self):
        store = VectroVectorStore(_FakeEmbeddings())
        ids = store.add_texts(TEXTS)
        store.delete(ids=[ids[0]])
        docs = store.similarity_search("capital", k=len(TEXTS))
        texts_in_results = {d.page_content for d in docs}
        assert TEXTS[0] not in texts_in_results


# ---------------------------------------------------------------------------
# compression_stats
# ---------------------------------------------------------------------------


class TestCompressionStats:
    def test_empty_stats(self):
        store = VectroVectorStore(_FakeEmbeddings())
        stats = store.compression_stats
        assert stats["n_vectors"] == 0

    def test_stats_after_add(self):
        store = VectroVectorStore.from_texts(TEXTS, _FakeEmbeddings())
        stats = store.compression_stats
        assert stats["n_vectors"] == len(TEXTS)
        assert stats["dimensions"] == DIM
        assert stats["compression_ratio"] >= 1.0
        assert stats["original_mb"] > stats["compressed_mb"]


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------


class TestAsyncAPI:
    def test_aadd_texts(self):
        store = VectroVectorStore(_FakeEmbeddings())

        async def _run():
            ids = await store.aadd_texts(TEXTS[:3])
            return ids

        ids = asyncio.run(_run())
        assert len(ids) == 3
        assert len(store) == 3

    def test_asimilarity_search(self):
        store = VectroVectorStore.from_texts(TEXTS, _FakeEmbeddings())

        async def _run():
            return await store.asimilarity_search("European capitals", k=2)

        docs = asyncio.run(_run())
        assert len(docs) == 2

    def test_asimilarity_search_with_score(self):
        store = VectroVectorStore.from_texts(TEXTS, _FakeEmbeddings())

        async def _run():
            return await store.asimilarity_search_with_score("capital", k=2)

        results = asyncio.run(_run())
        assert len(results) == 2
        for _, score in results:
            assert isinstance(score, float)

    def test_concurrent_async_adds(self):
        """Concurrent adds must not corrupt the store."""
        store = VectroVectorStore(_FakeEmbeddings())

        async def _run():
            await asyncio.gather(
                store.aadd_texts(TEXTS[:2]),
                store.aadd_texts(TEXTS[2:4]),
            )

        asyncio.run(_run())
        assert len(store) == 4


# ---------------------------------------------------------------------------
# Import from top-level package
# ---------------------------------------------------------------------------


def test_importable_from_package():
    from python import LangChainVectorStore  # noqa: F401

    assert LangChainVectorStore is VectroVectorStore
