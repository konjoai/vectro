"""Happy-path coverage for the vectro REST API.

Uses FastAPI's ``TestClient`` (built on httpx) — drives the ASGI app
in-process, no server boot, no socket binding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))

from main import _REGISTRY, app  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_registry():
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "vectro"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["indices"] == 0


def test_create_index(client):
    r = client.post("/index", json={"name": "demo", "dim": 8, "metric": "cosine"})
    assert r.status_code == 201
    assert r.json() == {"name": "demo", "dim": 8, "metric": "cosine"}


def test_create_index_duplicate(client):
    payload = {"name": "demo", "dim": 8, "metric": "cosine"}
    assert client.post("/index", json=payload).status_code == 201
    assert client.post("/index", json=payload).status_code == 409


def test_add_vectors_with_ids(client):
    client.post("/index", json={"name": "demo", "dim": 4, "metric": "cosine"})
    r = client.post(
        "/index/demo/add",
        json={
            "vectors": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            "ids": ["a", "b"],
        },
    )
    assert r.status_code == 200
    assert r.json() == {"added": 2, "total": 2}


def test_add_vectors_auto_ids(client):
    client.post("/index", json={"name": "demo", "dim": 4, "metric": "cosine"})
    r = client.post(
        "/index/demo/add",
        json={"vectors": [[1.0, 0.0, 0.0, 0.0]]},
    )
    assert r.status_code == 200
    assert r.json() == {"added": 1, "total": 1}


def test_add_dim_mismatch(client):
    client.post("/index", json={"name": "demo", "dim": 4, "metric": "cosine"})
    r = client.post("/index/demo/add", json={"vectors": [[1.0, 2.0, 3.0]]})
    assert r.status_code == 400


def test_add_id_count_mismatch(client):
    client.post("/index", json={"name": "demo", "dim": 3, "metric": "cosine"})
    r = client.post(
        "/index/demo/add",
        json={"vectors": [[1.0, 0.0, 0.0]], "ids": ["a", "b"]},
    )
    assert r.status_code == 400


def test_search_returns_nearest(client):
    client.post("/index", json={"name": "demo", "dim": 4, "metric": "cosine"})
    client.post(
        "/index/demo/add",
        json={
            "vectors": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            "ids": ["x", "y", "z"],
        },
    )
    r = client.post(
        "/index/demo/search",
        json={"query": [0.99, 0.01, 0.0, 0.0], "k": 2},
    )
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) == 2
    assert hits[0]["id"] == "x"
    assert hits[0]["distance"] < hits[1]["distance"]


def test_search_empty_index(client):
    client.post("/index", json={"name": "demo", "dim": 4, "metric": "cosine"})
    r = client.post(
        "/index/demo/search",
        json={"query": [1.0, 0.0, 0.0, 0.0], "k": 5},
    )
    assert r.status_code == 200
    assert r.json() == {"hits": []}


def test_search_dim_mismatch(client):
    client.post("/index", json={"name": "demo", "dim": 4, "metric": "cosine"})
    r = client.post("/index/demo/search", json={"query": [1.0, 2.0], "k": 1})
    assert r.status_code == 400


def test_stats(client):
    client.post("/index", json={"name": "demo", "dim": 4, "metric": "l2"})
    client.post(
        "/index/demo/add",
        json={"vectors": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]},
    )
    r = client.get("/index/demo/stats")
    assert r.status_code == 200
    assert r.json() == {"name": "demo", "dim": 4, "metric": "l2", "count": 2}


def test_delete_index(client):
    client.post("/index", json={"name": "demo", "dim": 4, "metric": "cosine"})
    r = client.delete("/index/demo")
    assert r.status_code == 204
    assert client.get("/index/demo/stats").status_code == 404


def test_delete_unknown(client):
    assert client.delete("/index/missing").status_code == 404


def test_unknown_index_404(client):
    assert client.get("/index/missing/stats").status_code == 404


def test_nan_rejected(client):
    # httpx blocks NaN at JSON encode; send a raw body with the NaN literal
    # (Python's json.loads accepts it) to exercise the server-side guard.
    client.post("/index", json={"name": "demo", "dim": 3, "metric": "cosine"})
    r = client.post(
        "/index/demo/add",
        content=b'{"vectors": [[1.0, NaN, 0.0]]}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400


def test_benchmark_returns_timings(client):
    client.post("/index", json={"name": "b", "dim": 32, "metric": "cosine"})
    # Tiny load — the test asserts shape and ordering, not raw speed.
    r = client.get(
        "/index/b/benchmark",
        params={"insert_count": 50, "search_count": 20, "k": 5, "ef": 32, "seed": 7},
    )
    assert r.status_code == 200
    body = r.json()

    assert body["insert_count"] == 50
    assert body["search_count"] == 20
    assert body["k"] == 5
    assert body["ef"] == 32
    assert body["seed"] == 7
    assert body["insert_ms_total"] > 0
    assert body["search_ms_total"] > 0
    assert body["insert_throughput_vps"] > 0
    assert body["search_throughput_qps"] > 0
    # Percentiles must be ordered.
    assert body["search_p50_ms"] <= body["search_p95_ms"] <= body["search_p99_ms"]
    # The index should now contain those 50 vectors.
    assert client.get("/index/b/stats").json()["count"] == 50


def test_benchmark_rejects_out_of_range(client):
    client.post("/index", json={"name": "b", "dim": 8, "metric": "cosine"})
    assert client.get("/index/b/benchmark", params={"insert_count": 0}).status_code == 400
    assert client.get("/index/b/benchmark", params={"search_count": 999_999}).status_code == 400


def test_benchmark_unknown_index_404(client):
    assert client.get("/index/nope/benchmark").status_code == 404


def test_full_roundtrip(client):
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((50, 16)).astype(np.float32)

    client.post("/index", json={"name": "rt", "dim": 16, "metric": "cosine"})
    client.post(
        "/index/rt/add",
        json={"vectors": vecs.tolist(), "ids": [f"v{i}" for i in range(50)]},
    )

    stats = client.get("/index/rt/stats").json()
    assert stats["count"] == 50

    r = client.post(
        "/index/rt/search",
        json={"query": vecs[7].tolist(), "k": 1},
    )
    hits = r.json()["hits"]
    assert hits[0]["id"] == "v7"
    assert hits[0]["distance"] < 1e-4
