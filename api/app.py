"""api/app.py — FastAPI surface for the live vector index.

Endpoints
---------
POST   /index/{name}                — create an empty index (body: {dim})
GET    /index/{name}                — index info
DELETE /index/{name}                — drop an index
POST   /index/{name}/add            — append vectors (auto-creates index)
POST   /index/{name}/search         — dense / BM25 / hybrid top-k
POST   /index/{name}/project        — PCA 2-D coords for every vector
POST   /index/{name}/cluster        — k-means labels for every vector
GET    /indexes                     — list all index names
GET    /healthz                     — liveness probe

The two visualization endpoints (``project``, ``cluster``) are the V7
focus.  Everything else is the minimum required to drive them from a
browser.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .store import IndexStore, hybrid_topk, kmeans, pca_2d

app = FastAPI(title="vectro-viz", version="0.8.0")
STORE = IndexStore()


# ─────────────────────────────────────────────────────────────────────────
# Request bodies
# ─────────────────────────────────────────────────────────────────────────


class CreateBody(BaseModel):
    dim: int = Field(..., ge=1, le=8192)


class AddBody(BaseModel):
    vectors: List[List[float]]
    ids: Optional[List[str]] = None
    metadata: Optional[List[Dict[str, Any]]] = None


class SearchBody(BaseModel):
    query: Optional[List[float]] = None
    text: Optional[str] = None
    k: int = 5
    alpha: float = Field(0.5, ge=0.0, le=1.0)


class ClusterBody(BaseModel):
    k: int = 3
    seed: int = 0


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _require(name: str):
    try:
        return STORE.get(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such index: {name!r}")


def _summary(idx) -> Dict[str, Any]:
    return {"name": idx.name, "dim": idx.dim, "n": len(idx)}


# ─────────────────────────────────────────────────────────────────────────
# Index CRUD
# ─────────────────────────────────────────────────────────────────────────


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "indexes": STORE.names()}


@app.get("/indexes")
def list_indexes() -> Dict[str, Any]:
    return {"indexes": STORE.names()}


@app.post("/index/{name}")
def create_index(name: str, body: CreateBody) -> Dict[str, Any]:
    try:
        idx = STORE.create(name, body.dim)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _summary(idx)


@app.get("/index/{name}")
def get_index(name: str) -> Dict[str, Any]:
    return _summary(_require(name))


@app.delete("/index/{name}")
def delete_index(name: str) -> Dict[str, Any]:
    return {"deleted": STORE.delete(name), "name": name}


# ─────────────────────────────────────────────────────────────────────────
# Vector ops
# ─────────────────────────────────────────────────────────────────────────


@app.post("/index/{name}/add")
def add_vectors(name: str, body: AddBody) -> Dict[str, Any]:
    if not body.vectors:
        raise HTTPException(status_code=400, detail="vectors must be non-empty")
    arr = np.asarray(body.vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail="vectors must be 2-D")
    n_new, dim_new = arr.shape

    try:
        idx = STORE.get(name)
    except KeyError:
        idx = STORE.create(name, dim_new)
    if dim_new != idx.dim:
        raise HTTPException(
            status_code=400,
            detail=f"dim mismatch: payload {dim_new} vs index {idx.dim}",
        )

    base = len(idx)
    for i in range(n_new):
        idx.vectors.append(arr[i].copy())
        if body.ids and i < len(body.ids):
            idx.ids.append(str(body.ids[i]))
        else:
            idx.ids.append(f"{name}:{base + i}")
        if body.metadata and i < len(body.metadata):
            idx.metadata.append(dict(body.metadata[i]))
        else:
            idx.metadata.append({})

    return {"name": name, "added": n_new, "total": len(idx)}


@app.post("/index/{name}/search")
def search(name: str, body: SearchBody) -> Dict[str, Any]:
    """Dense, BM25, or hybrid top-k.

    Supply ``query`` (a vector) for dense cosine search, ``text`` for BM25
    over each vector's ``metadata["text"]``, or both for an ``alpha``-weighted
    fusion (``alpha=1.0`` dense-only, ``alpha=0.0`` BM25-only).
    """
    has_query = body.query is not None
    has_text = body.text is not None
    if not has_query and not has_text:
        raise HTTPException(
            status_code=400,
            detail="search requires 'query' (vector) and/or 'text'",
        )
    mode = "hybrid" if (has_query and has_text) else ("dense" if has_query else "bm25")

    idx = _require(name)
    if len(idx) == 0:
        return {"name": name, "k": body.k, "alpha": body.alpha, "mode": mode, "results": []}

    query_vec = np.asarray(body.query, dtype=np.float32) if has_query else None
    docs = [str(m.get("text", "")) for m in idx.metadata]
    try:
        top = hybrid_topk(
            idx.matrix(),
            docs,
            query=query_vec,
            text=body.text,
            k=body.k,
            alpha=body.alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "name": name,
        "k": body.k,
        "alpha": body.alpha,
        "mode": mode,
        "results": [
            {
                "id": idx.ids[hit["index"]],
                "score": hit["score"],
                "index": hit["index"],
                "dense_score": hit["dense_score"],
                "bm25_score": hit["bm25_score"],
            }
            for hit in top
        ],
    }


# ─────────────────────────────────────────────────────────────────────────
# Visualization endpoints — V7
# ─────────────────────────────────────────────────────────────────────────


@app.post("/index/{name}/project")
def project(name: str) -> Dict[str, Any]:
    """PCA 2-D projection of every vector in the index."""
    idx = _require(name)
    coords = pca_2d(idx.matrix())
    return {
        "name": name,
        "n": len(idx),
        "coords": coords.tolist(),
        "ids": list(idx.ids),
    }


@app.post("/index/{name}/cluster")
def cluster(name: str, body: Optional[ClusterBody] = None) -> Dict[str, Any]:
    """K-means labels (one per vector).  ``k`` is clamped to ``[1, N]``."""
    idx = _require(name)
    body = body or ClusterBody()
    labels = kmeans(idx.matrix(), body.k, seed=body.seed)
    return {
        "name": name,
        "k": int(min(max(body.k, 1), max(len(idx), 1))),
        "labels": labels.tolist(),
        "ids": list(idx.ids),
    }
