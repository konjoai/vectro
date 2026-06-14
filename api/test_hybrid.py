"""Hybrid (dense ⊕ BM25) search coverage for ``POST /index/{name}/search``.

Two layers:
  * Endpoint tests via FastAPI's ``TestClient`` — dense-only (backward
    compatible), BM25-only, and ``alpha``-fused modes.
  * Unit tests for the pure-numpy helpers in :mod:`api.store`
    (``tokenize``, ``bm25_scores``, ``hybrid_topk``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app import STORE, app  # noqa: E402
from api.store import bm25_scores, hybrid_topk, tokenize  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_store():
    STORE.reset()
    yield
    STORE.reset()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# Three documents: orthonormal vectors so a basis-vector query has an
# unambiguous nearest neighbour, plus text for the BM25 leg.  "fox" appears
# in doc0 and doc2 only; doc1 shares no query terms.
_VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_TEXTS = ["quick brown fox", "lazy dog sleeps", "quick fox jumps high"]
_IDS = ["d0", "d1", "d2"]


def _seed(client: TestClient, name: str = "docs") -> None:
    payload = {
        "vectors": _VECTORS,
        "ids": _IDS,
        "metadata": [{"text": t} for t in _TEXTS],
    }
    r = client.post(f"/index/{name}/add", json=payload)
    assert r.status_code == 200, r.text


# ── Endpoint: dense-only (backward compatible) ───────────────────────────


def test_dense_only_search_is_backward_compatible(client: TestClient) -> None:
    _seed(client)
    r = client.post("/index/docs/search", json={"query": [1.0, 0.0, 0.0], "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "dense"
    assert body["results"][0]["id"] == "d0"
    # Each hit exposes both component scores for transparency.
    assert "dense_score" in body["results"][0]
    assert "bm25_score" in body["results"][0]
    assert body["results"][0]["bm25_score"] == 0.0


def test_search_requires_query_or_text(client: TestClient) -> None:
    _seed(client)
    r = client.post("/index/docs/search", json={"k": 3})
    assert r.status_code == 400


def test_search_empty_index_returns_no_results(client: TestClient) -> None:
    client.post("/index/empty", json={"dim": 3})
    r = client.post("/index/empty/search", json={"text": "fox", "k": 3})
    assert r.status_code == 200
    assert r.json()["results"] == []


# ── Endpoint: BM25-only ──────────────────────────────────────────────────


def test_bm25_only_search(client: TestClient) -> None:
    _seed(client)
    r = client.post("/index/docs/search", json={"text": "fox", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "bm25"
    returned = {hit["id"] for hit in body["results"] if hit["bm25_score"] > 0.0}
    # "fox" matches d0 and d2; never d1.
    assert returned == {"d0", "d2"}
    assert all(hit["dense_score"] == 0.0 for hit in body["results"])


def test_bm25_ranking_is_descending(client: TestClient) -> None:
    _seed(client)
    r = client.post("/index/docs/search", json={"text": "quick fox", "k": 3})
    scores = [hit["score"] for hit in r.json()["results"]]
    assert scores == sorted(scores, reverse=True)


# ── Endpoint: hybrid fusion ──────────────────────────────────────────────


def test_hybrid_alpha_one_matches_dense(client: TestClient) -> None:
    _seed(client)
    # Dense query nearest d2; alpha=1.0 must rank exactly like dense-only.
    hybrid = client.post(
        "/index/docs/search",
        json={"query": [0.0, 0.0, 1.0], "text": "fox", "k": 3, "alpha": 1.0},
    ).json()
    dense = client.post("/index/docs/search", json={"query": [0.0, 0.0, 1.0], "k": 3}).json()
    assert hybrid["mode"] == "hybrid"
    assert [h["id"] for h in hybrid["results"]] == [h["id"] for h in dense["results"]]
    assert hybrid["results"][0]["id"] == "d2"


def test_hybrid_alpha_zero_matches_bm25(client: TestClient) -> None:
    _seed(client)
    # alpha=0.0 → BM25-only ordering even though a dense query is supplied.
    hybrid = client.post(
        "/index/docs/search",
        json={"query": [1.0, 0.0, 0.0], "text": "fox", "k": 3, "alpha": 0.0},
    ).json()
    # d1 has no query term, so it must rank last.
    assert hybrid["results"][-1]["id"] == "d1"
    assert hybrid["results"][-1]["score"] == 0.0


def test_hybrid_blends_both_signals(client: TestClient) -> None:
    _seed(client)
    # Dense favours d0; text "jumps" favours d2.  A balanced fusion keeps
    # both in contention and never elevates the unrelated d1.
    body = client.post(
        "/index/docs/search",
        json={"query": [1.0, 0.0, 0.0], "text": "jumps", "k": 3, "alpha": 0.5},
    ).json()
    top_two = {h["id"] for h in body["results"][:2]}
    assert "d1" not in top_two


def test_alpha_out_of_range_is_rejected(client: TestClient) -> None:
    _seed(client)
    r = client.post(
        "/index/docs/search",
        json={"query": [1.0, 0.0, 0.0], "text": "fox", "alpha": 1.5},
    )
    assert r.status_code == 422  # pydantic validation (alpha ∈ [0, 1])


# ── Unit: store helpers ──────────────────────────────────────────────────


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("The Quick, BROWN fox!") == ["the", "quick", "brown", "fox"]


def test_bm25_scores_zero_for_unmatched_and_empty_query() -> None:
    scores = bm25_scores(_TEXTS, "fox")
    assert scores.shape == (3,)
    assert scores[1] == 0.0  # "lazy dog sleeps" has no "fox"
    assert scores[0] > 0.0 and scores[2] > 0.0
    # Empty query → all-zero, never NaN.
    assert np.all(bm25_scores(_TEXTS, "") == 0.0)


def test_bm25_scores_empty_corpus() -> None:
    assert bm25_scores([], "fox").shape == (0,)


def test_hybrid_topk_requires_a_signal() -> None:
    M = np.asarray(_VECTORS, dtype=np.float32)
    with pytest.raises(ValueError):
        hybrid_topk(M, _TEXTS, query=None, text=None, k=3)


def test_hybrid_topk_dense_only_orders_by_cosine() -> None:
    M = np.asarray(_VECTORS, dtype=np.float32)
    hits = hybrid_topk(M, _TEXTS, query=np.asarray([0.0, 1.0, 0.0]), k=3)
    assert hits[0]["index"] == 1
    assert hits[0]["bm25_score"] == 0.0
