"""Tests for python.retriever — VectroRetriever, RetrievalResult, RetrieverProtocol.

All tests are pure-unit / integration (no GPU / no real model) and should run
offline with a `maturin develop`-built `vectro_py` wheel.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
try:
    import vectro_py  # noqa: F401
    from vectro_py import EmbeddingDataset, PyEmbedding

    _SKIP_BINDINGS = False
except ImportError:
    _SKIP_BINDINGS = True

try:
    from python.retriever import RetrievalResult, RetrieverProtocol, VectroRetriever

    _SKIP_RETRIEVER = False
except ImportError:
    _SKIP_RETRIEVER = True

pytestmark = pytest.mark.skipif(
    _SKIP_BINDINGS or _SKIP_RETRIEVER,
    reason="vectro_py bindings or python.retriever not importable",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return v if norm == 0.0 else [x / norm for x in v]


def _make_dataset(ids: list[str], vecs: list[list[float]]) -> "EmbeddingDataset":
    ds = EmbeddingDataset()
    for doc_id, vec in zip(ids, vecs):
        ds.add_embedding(PyEmbedding(doc_id, np.array(vec, dtype=np.float32)))
    return ds


# Small corpus: 4 docs
IDS = ["cat_doc", "dog_doc", "ml_doc", "other_doc"]
TEXTS = [
    "cats are fluffy soft domestic animals",
    "dogs are loyal friendly playful pets",
    "machine learning algorithms optimize neural networks",
    "completely unrelated topic here",
]
VECS = [
    _unit([1.0, 0.1, 0.0]),
    _unit([0.9, 0.2, 0.0]),
    _unit([0.0, 0.0, 1.0]),
    _unit([0.0, 1.0, 0.0]),
]

# Simple mock embed function that returns fixed vector regardless of query
_MOCK_EMBED_FN = lambda q: np.array(_unit([1.0, 0.1, 0.0]), dtype=np.float32)  # noqa: E731


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def retriever():
    ds = _make_dataset(IDS, VECS)
    return VectroRetriever(
        dataset=ds,
        texts=TEXTS,
        ids=IDS,
        embed_fn=_MOCK_EMBED_FN,
        alpha=0.7,
    )


@pytest.fixture()
def retriever_no_embed():
    ds = _make_dataset(IDS, VECS)
    return VectroRetriever(
        dataset=ds,
        texts=TEXTS,
        ids=IDS,
        embed_fn=None,
        alpha=0.7,
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_implements_protocol(retriever):
    assert isinstance(retriever, RetrieverProtocol)


def test_retriever_without_embed_implements_protocol(retriever_no_embed):
    assert isinstance(retriever_no_embed, RetrieverProtocol)


# ---------------------------------------------------------------------------
# Return-type and structure
# ---------------------------------------------------------------------------


def test_returns_list(retriever):
    results = retriever.retrieve("cats fluffy", k=3)
    assert isinstance(results, list)


def test_returns_retrieval_result_instances(retriever):
    results = retriever.retrieve("cats", k=3)
    for r in results:
        assert isinstance(r, RetrievalResult), f"Expected RetrievalResult, got {type(r)}"


def test_result_fields_populated(retriever):
    results = retriever.retrieve("cats", k=3)
    for r in results:
        assert isinstance(r.id, str) and r.id
        assert isinstance(r.combined_score, float)
        assert isinstance(r.dense_score, float)
        assert isinstance(r.bm25_score, float)


def test_text_field_populated(retriever):
    results = retriever.retrieve("cats", k=3)
    for r in results:
        assert isinstance(r.text, str) and r.text, f"text empty for id={r.id}"


def test_ids_from_corpus(retriever):
    results = retriever.retrieve("cats", k=4)
    assert all(r.id in IDS for r in results)


# ---------------------------------------------------------------------------
# Ordering and score range
# ---------------------------------------------------------------------------


def test_sorted_descending_by_combined_score(retriever):
    results = retriever.retrieve("cats", k=4)
    scores = [r.combined_score for r in results]
    assert scores == sorted(scores, reverse=True), f"Results not sorted descending: {scores}"


def test_combined_scores_in_unit_interval(retriever):
    results = retriever.retrieve("cats", k=4)
    for r in results:
        assert 0.0 <= r.combined_score <= 1.0 + 1e-6, (
            f"combined_score out of [0,1]: {r.combined_score}"
        )


# ---------------------------------------------------------------------------
# k parameter
# ---------------------------------------------------------------------------


def test_k_respected(retriever):
    for k in [1, 2, 3]:
        results = retriever.retrieve("dogs loyal", k=k)
        assert len(results) <= k


def test_k_zero_returns_empty_list(retriever):
    assert retriever.retrieve("cats", k=0) == []


# ---------------------------------------------------------------------------
# BM25-only mode (embed_fn=None)
# ---------------------------------------------------------------------------


def test_no_embed_fn_still_returns_results(retriever_no_embed):
    results = retriever_no_embed.retrieve("machine learning", k=3)
    assert isinstance(results, list)
    assert len(results) > 0


def test_no_embed_fn_bm25_wins_for_keyword(retriever_no_embed):
    """Without embedding, BM25 selects the keyword-matching doc."""
    results = retriever_no_embed.retrieve("machine learning algorithms", k=2)
    assert results[0].id == "ml_doc", f"Expected ml_doc first (keyword match), got {results[0].id}"


# ---------------------------------------------------------------------------
# Property accessors
# ---------------------------------------------------------------------------


def test_alpha_property(retriever):
    assert retriever.alpha == pytest.approx(0.7)


def test_n_docs_property(retriever):
    assert retriever.n_docs == len(IDS)


def test_bm25_property_not_none(retriever):
    assert retriever.bm25 is not None


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_mismatched_ids_texts_raises_value_error():
    ds = _make_dataset(IDS, VECS)
    with pytest.raises(ValueError, match="ids.*texts|texts.*ids|length"):
        VectroRetriever(ds, texts=["only one"], ids=["a", "b"])


def test_empty_corpus():
    ds = EmbeddingDataset()
    r = VectroRetriever(ds, texts=[], ids=[], embed_fn=_MOCK_EMBED_FN)
    results = r.retrieve("anything", k=5)
    assert results == []


def test_single_element_corpus():
    ds = _make_dataset(["solo"], [[1.0, 0.0]])
    r = VectroRetriever(
        ds,
        texts=["the only document in the world"],
        ids=["solo"],
        embed_fn=lambda q: [1.0, 0.0],
        alpha=0.7,
    )
    results = r.retrieve("only document", k=1)
    assert len(results) == 1
    assert results[0].id == "solo"


# ---------------------------------------------------------------------------
# RetrievalResult dataclass
# ---------------------------------------------------------------------------


def test_retrieval_result_construction():
    r = RetrievalResult(
        id="x1",
        dense_score=0.9,
        bm25_score=0.6,
        combined_score=0.78,
        text="some text",
    )
    assert r.id == "x1"
    assert r.combined_score == pytest.approx(0.78)
    assert r.text == "some text"


def test_retrieval_result_default_text():
    r = RetrievalResult(id="x2", dense_score=0.5, bm25_score=0.3, combined_score=0.4)
    assert r.text == ""
