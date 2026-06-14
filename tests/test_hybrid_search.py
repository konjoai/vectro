"""Tests for hybrid_search_py — BM25 + dense cosine fusion (v6.0.0).

All tests use small synthetic datasets so they run without any model or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import guard — skip entire module if Rust bindings are not built yet
# ---------------------------------------------------------------------------
try:
    import vectro_py  # noqa: F401
    from vectro_py import (
        BM25Index,
        EmbeddingDataset,
        PyEmbedding,
        hybrid_search_py,
    )

    _SKIP = False
except ImportError:
    _SKIP = True

pytestmark = pytest.mark.skipif(
    _SKIP,
    reason="vectro_py bindings not built — run `maturin develop` first",
)


# ---------------------------------------------------------------------------
# Tiny corpus helpers
# ---------------------------------------------------------------------------


def _make_dataset(ids: list[str], vectors: list[list[float]]) -> "EmbeddingDataset":
    ds = EmbeddingDataset()
    for doc_id, vec in zip(ids, vectors):
        emb = PyEmbedding(doc_id, np.array(vec, dtype=np.float32))
        ds.add_embedding(emb)
    return ds


def _unit(v: list[float]) -> np.ndarray:
    """Return L2-normalised copy of *v* as a float32 numpy array."""
    arr = np.array(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return (arr / norm).astype(np.float32)


# Corpus: 4 docs, 3-dimensional embeddings
DOCS = [
    ("doc_a", "the quick brown fox jumps", [1.0, 0.0, 0.0]),
    ("doc_b", "machine learning neural networks", [0.0, 1.0, 0.0]),
    ("doc_c", "quick fox machine learning", [0.5, 0.5, 0.0]),
    ("doc_d", "unrelated completely different text", [0.0, 0.0, 1.0]),
]

IDS = [d[0] for d in DOCS]
TEXTS = [d[1] for d in DOCS]
VECS = [d[2] for d in DOCS]


@pytest.fixture()
def corpus():
    dataset = _make_dataset(IDS, VECS)
    bm25 = BM25Index.build(IDS, TEXTS)
    return dataset, bm25


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------


def test_returns_list(corpus):
    dataset, bm25 = corpus
    results = hybrid_search_py(dataset, bm25, _unit([1, 0, 0]), "fox", k=2, alpha=0.5)
    assert isinstance(results, list)


def test_returns_at_most_k(corpus):
    dataset, bm25 = corpus
    for k in [1, 2, 3, 4]:
        results = hybrid_search_py(dataset, bm25, _unit([1, 0, 0]), "fox", k=k, alpha=0.5)
        assert len(results) <= k


def test_k_zero_returns_empty(corpus):
    dataset, bm25 = corpus
    results = hybrid_search_py(dataset, bm25, _unit([1, 0, 0]), "fox", k=0, alpha=0.5)
    assert results == []


def test_result_type(corpus):
    """Each element must be a (str, float) pair."""
    dataset, bm25 = corpus
    results = hybrid_search_py(dataset, bm25, _unit([1, 0, 0]), "fox", k=2, alpha=0.5)
    for item in results:
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], float)


def test_scores_in_unit_interval(corpus):
    dataset, bm25 = corpus
    results = hybrid_search_py(dataset, bm25, _unit([1, 0, 0]), "fox", k=4, alpha=0.5)
    for _, score in results:
        assert 0.0 <= score <= 1.0 + 1e-6, f"score out of range: {score}"


def test_sorted_descending(corpus):
    dataset, bm25 = corpus
    results = hybrid_search_py(dataset, bm25, _unit([1, 0, 0]), "fox", k=4, alpha=0.5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "results not sorted descending"


# ---------------------------------------------------------------------------
# Alpha=1.0 (pure dense) — ordering should follow cosine similarity
# ---------------------------------------------------------------------------


def test_alpha_1_matches_dense_winner(corpus):
    """With alpha=1.0 the result should rank doc_a highest (query=[1,0,0])."""
    dataset, bm25 = corpus
    results = hybrid_search_py(dataset, bm25, _unit([1, 0, 0]), "", k=4, alpha=1.0)
    assert results[0][0] == "doc_a", f"Expected doc_a first, got {results}"


def test_alpha_1_ranks_d_last(corpus):
    """doc_d ([0,0,1]) should be ranked last for a [1,0,0] query."""
    dataset, bm25 = corpus
    results = hybrid_search_py(dataset, bm25, _unit([1, 0, 0]), "", k=4, alpha=1.0)
    assert results[-1][0] == "doc_d"


# ---------------------------------------------------------------------------
# Alpha=0.0 (pure BM25) — ordering follows BM25 keyword ranking
# ---------------------------------------------------------------------------


def test_alpha_0_bm25_dominant(corpus):
    """'machine learning' query → doc_b or doc_c should appear in top-2."""
    dataset, bm25 = corpus
    results = hybrid_search_py(
        dataset, bm25, np.array([0, 0, 0], dtype=np.float32), "machine learning", k=2, alpha=0.0
    )
    top_ids = {r[0] for r in results}
    assert len(top_ids & {"doc_b", "doc_c"}) > 0, f"Expected doc_b or doc_c in top-2, got {top_ids}"


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


def test_empty_query_text_with_pure_dense(corpus):
    """Empty query text + alpha=1.0 must still return dense results."""
    dataset, bm25 = corpus
    results = hybrid_search_py(dataset, bm25, _unit([1, 0, 0]), "", k=3, alpha=1.0)
    assert len(results) == 3


def test_oov_query_text(corpus):
    """Query text with no vocabulary overlap must not crash."""
    dataset, bm25 = corpus
    results = hybrid_search_py(
        dataset, bm25, _unit([0, 1, 0]), "xyzzy quux zorkian", k=4, alpha=0.5
    )
    assert isinstance(results, list)


def test_empty_query_vector_bm25_only(corpus):
    """Empty query vector with alpha=0.0 must work (BM25-only path)."""
    dataset, bm25 = corpus
    results = hybrid_search_py(
        dataset, bm25, np.array([], dtype=np.float32), "machine learning", k=4, alpha=0.0
    )
    assert len(results) > 0


def test_alpha_clamp_above_1(corpus):
    """alpha > 1.0 must be clamped to 1.0 (no crash, valid output)."""
    dataset, bm25 = corpus
    results = hybrid_search_py(dataset, bm25, _unit([1, 0, 0]), "fox", k=3, alpha=5.0)
    scores = [s for _, s in results]
    assert all(0.0 <= s <= 1.0 + 1e-6 for s in scores)


def test_alpha_clamp_below_0(corpus):
    dataset, bm25 = corpus
    results = hybrid_search_py(dataset, bm25, _unit([1, 0, 0]), "fox", k=3, alpha=-2.0)
    scores = [s for _, s in results]
    assert all(0.0 <= s <= 1.0 + 1e-6 for s in scores)


# ---------------------------------------------------------------------------
# BM25Index Python binding tests (piggy-backed here to avoid extra file)
# ---------------------------------------------------------------------------


def test_bm25_build():
    idx = BM25Index.build(IDS, TEXTS)
    assert len(idx) == 4


def test_bm25_top_k_type():
    idx = BM25Index.build(IDS, TEXTS)
    results = idx.top_k("fox", 2)
    assert isinstance(results, list)
    for item in results:
        assert isinstance(item[0], str)
        assert isinstance(item[1], float)


def test_bm25_top_k_len():
    idx = BM25Index.build(IDS, TEXTS)
    results = idx.top_k("machine", 2)
    assert len(results) <= 2


def test_bm25_contains_expected_result():
    idx = BM25Index.build(IDS, TEXTS)
    top = idx.top_k("machine learning", 2)
    top_ids = {r[0] for r in top}
    assert len(top_ids & {"doc_b", "doc_c"}) > 0


def test_bm25_unknown_term_idf_zero():
    idx = BM25Index.build(IDS, TEXTS)
    assert idx.idf("doesnotexist") == 0.0


def test_bm25_known_term_idf_positive():
    idx = BM25Index.build(IDS, TEXTS)
    assert idx.idf("machine") > 0.0


def test_bm25_mismatched_ids_texts_raises():
    with pytest.raises(Exception):
        BM25Index.build(["a"], ["text1", "text2"])


def test_bm25_build_with_params():
    idx = BM25Index.build_with_params(IDS, TEXTS, k1=1.2, b=0.5)
    assert len(idx) == 4
    # Different params → may yield different score order but must not crash
    results = idx.top_k("fox", 2)
    assert isinstance(results, list)
