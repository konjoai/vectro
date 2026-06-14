"""api/test_viz.py — V7 visualization endpoints (project + cluster).

Coverage:
    1. project returns 2-D coords matching index size
    2. project 404s on a missing index
    3. project handles a single vector (zero coords, no NaN)
    4. project centres the data (top-PC sums to zero on a balanced pair)
    5. project handles an empty index without raising
    6. cluster recovers well-separated synthetic clusters
    7. cluster clamps k > N to N
    8. cluster 404s on a missing index
    9. cluster default k=3 yields one label per vector
   10. project + cluster agree on ids ordering (so the UI can join them)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Make the repo root importable so ``api.app`` resolves regardless of
# pytest's rootdir choice.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app import STORE, app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_store():
    STORE.reset()
    yield
    STORE.reset()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _add_clustered(
    client: TestClient,
    name: str,
    n_per_cluster: int = 10,
    dim: int = 4,
    seed: int = 0,
) -> int:
    """Append three well-separated Gaussian blobs to ``name``.  Returns
    total points added."""
    rng = np.random.default_rng(seed)
    centres = np.zeros((3, dim), dtype=np.float32)
    centres[0, 0] = 5.0
    centres[1, 1] = 5.0
    centres[2, 2] = 5.0
    rows = []
    for c in centres:
        noise = rng.standard_normal((n_per_cluster, dim)).astype(np.float32) * 0.1
        rows.append(c[None, :] + noise)
    vectors = np.concatenate(rows, axis=0)
    r = client.post(f"/index/{name}/add", json={"vectors": vectors.tolist()})
    assert r.status_code == 200, r.text
    return int(vectors.shape[0])


# ─────────────────────────────────────────────────────────────────────────
# /project
# ─────────────────────────────────────────────────────────────────────────


def test_project_returns_2d_coords(client: TestClient) -> None:
    n = _add_clustered(client, "demo")
    r = client.post("/index/demo/project")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "demo"
    assert body["n"] == n
    coords = body["coords"]
    assert len(coords) == n
    assert all(len(row) == 2 for row in coords)
    arr = np.asarray(coords, dtype=np.float32)
    assert np.isfinite(arr).all()


def test_project_404_for_missing_index(client: TestClient) -> None:
    r = client.post("/index/missing/project")
    assert r.status_code == 404
    assert "missing" in r.json()["detail"]


def test_project_handles_single_vector(client: TestClient) -> None:
    r = client.post("/index/single/add", json={"vectors": [[1.0, 2.0, 3.0]]})
    assert r.status_code == 200
    r = client.post("/index/single/project")
    assert r.status_code == 200
    assert r.json()["coords"] == [[0.0, 0.0]]


def test_project_centres_data(client: TestClient) -> None:
    payload = {"vectors": [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]}
    r = client.post("/index/two/add", json=payload)
    assert r.status_code == 200
    coords = np.asarray(client.post("/index/two/project").json()["coords"])
    assert coords.shape == (2, 2)
    # Centred → opposite sign on the principal axis, sum near zero.
    assert abs(float(coords[:, 0].sum())) < 1e-4
    # Pair is colinear in 1-D, so the second column is exactly zero.
    assert float(np.abs(coords[:, 1]).sum()) < 1e-6


def test_project_empty_index(client: TestClient) -> None:
    r = client.post("/index/empty", json={"dim": 4})
    assert r.status_code == 200
    r = client.post("/index/empty/project")
    assert r.status_code == 200
    body = r.json()
    assert body["coords"] == []
    assert body["ids"] == []
    assert body["n"] == 0


# ─────────────────────────────────────────────────────────────────────────
# /cluster
# ─────────────────────────────────────────────────────────────────────────


def test_cluster_recovers_synthetic_clusters(client: TestClient) -> None:
    _add_clustered(client, "demo", n_per_cluster=10)
    r = client.post("/index/demo/cluster", json={"k": 3, "seed": 42})
    assert r.status_code == 200, r.text
    labels = np.asarray(r.json()["labels"], dtype=np.int32)
    assert labels.shape == (30,)
    assert set(labels.tolist()) <= {0, 1, 2}
    # Each block of 10 was generated from one Gaussian — the dominant
    # label inside each block should appear at least 8/10 times.
    for blk in range(3):
        block = labels[blk * 10 : (blk + 1) * 10]
        dominant = Counter(block.tolist()).most_common(1)[0][1]
        assert dominant >= 8, f"block {blk} dominant label only {dominant}/10"
    # And the three blocks must each map to a distinct dominant label —
    # otherwise k-means collapsed two blobs together.
    dominants = {
        Counter(labels[blk * 10 : (blk + 1) * 10].tolist()).most_common(1)[0][0] for blk in range(3)
    }
    assert len(dominants) == 3


def test_cluster_clamps_k_to_n(client: TestClient) -> None:
    r = client.post("/index/small/add", json={"vectors": [[1.0, 0.0], [0.0, 1.0]]})
    assert r.status_code == 200
    r = client.post("/index/small/cluster", json={"k": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["labels"]) == 2
    assert body["k"] == 2
    assert all(0 <= label <= 1 for label in body["labels"])


def test_cluster_404_for_missing_index(client: TestClient) -> None:
    r = client.post("/index/missing/cluster", json={"k": 3})
    assert r.status_code == 404


def test_cluster_default_k(client: TestClient) -> None:
    n = _add_clustered(client, "demo")
    r = client.post("/index/demo/cluster", json={})
    assert r.status_code == 200
    body = r.json()
    assert len(body["labels"]) == n
    assert body["k"] == 3


# ─────────────────────────────────────────────────────────────────────────
# Joint behaviour
# ─────────────────────────────────────────────────────────────────────────


def test_project_and_cluster_share_id_order(client: TestClient) -> None:
    n = _add_clustered(client, "demo")
    proj = client.post("/index/demo/project").json()
    clus = client.post("/index/demo/cluster", json={"k": 3, "seed": 0}).json()
    assert proj["ids"] == clus["ids"], (
        "viz frontend joins coords and labels by position — "
        "id ordering must be identical across endpoints"
    )
    assert len(proj["coords"]) == len(clus["labels"]) == n
