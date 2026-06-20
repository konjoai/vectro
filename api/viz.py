"""api/viz.py — V7 visualization endpoints, mounted under ``/viz`` by main.py.

Three endpoints back the demo/viz.html scatter plot:

* ``POST /viz/index/{name}/add``     — seed an in-memory index with vectors
* ``POST /viz/index/{name}/project`` — PCA(2) projection via numpy SVD
* ``POST /viz/index/{name}/cluster`` — k-means assignments (k param, default 3)

Plus ``DELETE /viz/index/{name}`` to drop the in-memory state.

In-memory store is intentional: the visualization is exploratory and the
demo server has no persistence story.  All numerics use float64 internally
and round-trip to plain Python floats at the JSON boundary so clients
never see numpy scalars.

Originally `api/main.py` in V7 (commit 3e5d4d6).  Relocated to its own
router during the V6 rebase so the V6 HNSW REST endpoints could occupy
``/index/{name}/*`` without colliding with V7's flat-numpy ``INDICES``.
"""

from __future__ import annotations

from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/viz", tags=["viz"])

INDICES: dict[str, np.ndarray] = {}


class AddRequest(BaseModel):
    vectors: List[List[float]] = Field(..., description="Row-major 2D float array")


class ClusterRequest(BaseModel):
    k: int = Field(default=3, ge=1, le=64)
    max_iter: int = Field(default=50, ge=1, le=500)
    seed: int = Field(default=42)


class ProjectedPoint(BaseModel):
    id: int
    x: float
    y: float
    label: str


class ClusterAssignment(BaseModel):
    id: int
    cluster: int


@router.post("/index/{name}/add")
def add_vectors(name: str, req: AddRequest) -> dict:
    if not req.vectors:
        raise HTTPException(400, "vectors must be non-empty")
    arr = np.asarray(req.vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(400, "vectors must be 2D")
    if name in INDICES:
        existing = INDICES[name]
        if existing.shape[1] != arr.shape[1]:
            raise HTTPException(
                400, f"dim mismatch: existing={existing.shape[1]} new={arr.shape[1]}"
            )
        INDICES[name] = np.vstack([existing, arr])
    else:
        INDICES[name] = arr
    return {"name": name, "n": int(INDICES[name].shape[0]), "dim": int(arr.shape[1])}


@router.delete("/index/{name}")
def drop_index(name: str) -> dict:
    INDICES.pop(name, None)
    return {"name": name, "dropped": True}


@router.post("/index/{name}/project", response_model=List[ProjectedPoint])
def project(name: str) -> List[ProjectedPoint]:
    if name not in INDICES:
        raise HTTPException(404, f"index not found: {name}")
    X = INDICES[name].astype(np.float64)
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    if n == 1 or d == 0:
        coords = np.zeros((n, 2), dtype=np.float64)
    elif d == 1:
        coords = np.column_stack([Xc[:, 0], np.zeros(n)])
    else:
        # SVD-based PCA: Xc = U S Vt; principal axes are rows of Vt.
        # Project onto the first two axes => U[:, :2] * S[:2].
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(2, Vt.shape[0])
        proj = Xc @ Vt[:k].T
        if k == 1:
            proj = np.column_stack([proj[:, 0], np.zeros(n)])
        coords = proj
    return [
        ProjectedPoint(id=i, x=float(coords[i, 0]), y=float(coords[i, 1]), label=str(i))
        for i in range(n)
    ]


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: pick the first center uniformly, then each
    subsequent center with probability proportional to squared distance
    from the nearest already-chosen center.  This dramatically reduces
    the chance of degenerate inits where two centers land in one blob.
    """
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    centers[0] = X[rng.integers(n)]
    closest_sq = ((X - centers[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = float(closest_sq.sum())
        if total <= 0.0:
            centers[i] = X[rng.integers(n)]
        else:
            probs = closest_sq / total
            idx = int(rng.choice(n, p=probs))
            centers[i] = X[idx]
        new_sq = ((X - centers[i]) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centers


def _kmeans(X: np.ndarray, k: int, max_iter: int, seed: int) -> np.ndarray:
    n = X.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(X, k, rng)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # squared euclidean: ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2
        d = (
            (X * X).sum(axis=1, keepdims=True)
            - 2.0 * (X @ centers.T)
            + (centers * centers).sum(axis=1)[None, :]
        )
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        new_centers = centers.copy()
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centers[c] = X[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


@router.post("/index/{name}/cluster", response_model=List[ClusterAssignment])
def cluster(name: str, req: ClusterRequest | None = None) -> List[ClusterAssignment]:
    if name not in INDICES:
        raise HTTPException(404, f"index not found: {name}")
    req = req or ClusterRequest()
    X = INDICES[name].astype(np.float64)
    labels = _kmeans(X, k=req.k, max_iter=req.max_iter, seed=req.seed)
    return [ClusterAssignment(id=i, cluster=int(labels[i])) for i in range(X.shape[0])]
