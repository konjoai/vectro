"""Vectro REST API — FastAPI wrapper around vectro.HNSWIndex.

Exposes a small, language-agnostic HTTP surface for creating named
indices, adding vectors, running kNN search, and inspecting stats.

The index store is in-process and not persisted across restarts —
intended for embedding-as-a-service deployments behind a stateful
worker (Render, Fly.io, ECS task with EBS, etc.) where the lifetime
of the indices matches the lifetime of the process.

Threading model: FastAPI runs handlers in a threadpool by default.
A single ``asyncio.Lock``-equivalent ``threading.RLock`` per index
serialises mutating operations; reads (search, stats) take the same
lock to keep the underlying ``HNSWIndex`` state consistent.

V7 visualization endpoints (``/viz/index/{name}/{add,project,cluster}``)
live in :mod:`api.viz` and are mounted via :func:`include_router` so
both feature sets share one ASGI app without path collisions.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

# vectro ships as a top-level package named ``python`` (see project pyproject.toml,
# ``packages = ["python"]``).  Support both that import path and a future
# ``vectro`` rename without forcing the API to know which one is live.
try:
    from vectro import HNSWIndex  # type: ignore[import-not-found]
except ImportError:
    from python import HNSWIndex  # type: ignore[no-redef]

from api.viz import router as viz_router

__version__ = "0.1.0"

Metric = Literal["cosine", "l2"]


class _IndexEntry:
    """Per-index state: the HNSW backend plus a counter for auto-ids."""

    __slots__ = ("name", "dim", "metric", "index", "count", "lock")

    def __init__(self, name: str, dim: int, metric: Metric) -> None:
        self.name = name
        self.dim = dim
        self.metric = metric
        self.index = HNSWIndex(dim=dim, space=metric)
        self.count = 0
        self.lock = threading.RLock()


# ---------------------------------------------------------------------------
# Process-wide registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, _IndexEntry] = {}
_REGISTRY_LOCK = threading.RLock()


def _get(name: str) -> _IndexEntry:
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(name)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"index '{name}' not found",
        )
    return entry


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CreateIndexRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    dim: int = Field(..., gt=0, le=65536)
    metric: Metric = "cosine"


class CreateIndexResponse(BaseModel):
    name: str
    dim: int
    metric: Metric


class AddRequest(BaseModel):
    vectors: List[List[float]] = Field(..., min_length=1)
    ids: Optional[List[str]] = None

    @field_validator("vectors")
    @classmethod
    def _vectors_nonempty_rows(cls, v: List[List[float]]) -> List[List[float]]:
        if any(len(row) == 0 for row in v):
            raise ValueError("each vector must be non-empty")
        return v


class AddResponse(BaseModel):
    added: int
    total: int


class SearchRequest(BaseModel):
    query: List[float] = Field(..., min_length=1)
    k: int = Field(10, gt=0, le=1000)
    ef: Optional[int] = Field(None, gt=0, le=4096)


class SearchHit(BaseModel):
    id: str
    distance: float


class SearchResponse(BaseModel):
    hits: List[SearchHit]


class StatsResponse(BaseModel):
    name: str
    dim: int
    metric: Metric
    count: int


class BenchmarkResponse(BaseModel):
    """Timing report for a self-contained insert + search workload.

    All latencies are wall-clock milliseconds.  Throughput is whole-batch
    operations divided by total wall time, so insert throughput accounts
    for graph-construction overhead and search throughput accounts for
    the inner-loop cost of beam search at the configured ``ef``.
    """

    name: str
    dim: int
    metric: Metric
    insert_count: int
    search_count: int
    k: int
    ef: int
    seed: int
    insert_ms_total: float
    insert_throughput_vps: float
    search_ms_total: float
    search_throughput_qps: float
    search_p50_ms: float
    search_p95_ms: float
    search_p99_ms: float


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vectro REST API",
    version=__version__,
    description="HTTP wrapper around the vectro embedding-compression library.",
)
app.include_router(viz_router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "vectro", "version": __version__}


@app.get("/health")
def health() -> dict:
    with _REGISTRY_LOCK:
        n = len(_REGISTRY)
    return {"status": "ok", "indices": n}


@app.post(
    "/index",
    response_model=CreateIndexResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_index(req: CreateIndexRequest) -> CreateIndexResponse:
    with _REGISTRY_LOCK:
        if req.name in _REGISTRY:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"index '{req.name}' already exists",
            )
        _REGISTRY[req.name] = _IndexEntry(req.name, req.dim, req.metric)
    return CreateIndexResponse(name=req.name, dim=req.dim, metric=req.metric)


@app.post("/index/{name}/add", response_model=AddResponse)
def add_vectors(name: str, req: AddRequest) -> AddResponse:
    entry = _get(name)

    if req.ids is not None and len(req.ids) != len(req.vectors):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="len(ids) must equal len(vectors) when ids are provided",
        )

    arr = np.asarray(req.vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != entry.dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"vectors must be 2-D with dim={entry.dim}, got shape={list(arr.shape)}"),
        )
    if not np.isfinite(arr).all():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="vectors contain NaN or Inf",
        )

    with entry.lock:
        if req.ids is None:
            ids = [str(entry.count + i) for i in range(arr.shape[0])]
        else:
            ids = list(req.ids)
        entry.index.add_batch(arr, ids=ids)
        entry.count += arr.shape[0]
        total = entry.count

    return AddResponse(added=arr.shape[0], total=total)


@app.post("/index/{name}/search", response_model=SearchResponse)
def search(name: str, req: SearchRequest) -> SearchResponse:
    entry = _get(name)

    q = np.asarray(req.query, dtype=np.float32)
    if q.ndim != 1 or q.shape[0] != entry.dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"query must be 1-D with dim={entry.dim}, got shape={list(q.shape)}",
        )
    if not np.isfinite(q).all():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query contains NaN or Inf",
        )

    ef = req.ef if req.ef is not None else max(req.k * 4, 64)

    with entry.lock:
        if entry.count == 0:
            return SearchResponse(hits=[])
        user_ids, distances = entry.index.search(q, top_k=req.k, ef=ef)
        hits = [
            SearchHit(id=str(uid), distance=float(d))
            for uid, d in zip(user_ids, distances.tolist())
        ]

    return SearchResponse(hits=hits)


@app.get("/index/{name}/stats", response_model=StatsResponse)
def stats(name: str) -> StatsResponse:
    entry = _get(name)
    with entry.lock:
        count = entry.count
    return StatsResponse(name=entry.name, dim=entry.dim, metric=entry.metric, count=count)


@app.get("/index/{name}/benchmark", response_model=BenchmarkResponse)
def benchmark(
    name: str,
    insert_count: int = 1000,
    search_count: int = 100,
    k: int = 10,
    ef: int = 64,
    seed: int = 42,
) -> BenchmarkResponse:
    """Run an in-process throughput + latency benchmark against the index.

    Inserts ``insert_count`` random unit-norm vectors (drawn from a
    deterministic ``np.random.default_rng(seed)``), then runs
    ``search_count`` kNN queries from independent random vectors.

    Bounded for safety: ``insert_count`` ≤ 100_000, ``search_count`` ≤ 10_000.
    The work is appended to the existing index — call against a fresh
    index for clean numbers.
    """
    if not 1 <= insert_count <= 100_000:
        raise HTTPException(400, "insert_count must be in [1, 100000]")
    if not 1 <= search_count <= 10_000:
        raise HTTPException(400, "search_count must be in [1, 10000]")
    if not 1 <= k <= 1000:
        raise HTTPException(400, "k must be in [1, 1000]")
    if not 1 <= ef <= 4096:
        raise HTTPException(400, "ef must be in [1, 4096]")

    entry = _get(name)
    rng = np.random.default_rng(seed)
    insert_vecs = rng.standard_normal((insert_count, entry.dim)).astype(np.float32)
    query_vecs = rng.standard_normal((search_count, entry.dim)).astype(np.float32)

    with entry.lock:
        ids = [f"bench-{entry.count + i}" for i in range(insert_count)]
        t_insert_start = time.perf_counter()
        entry.index.add_batch(insert_vecs, ids=ids)
        insert_ms_total = (time.perf_counter() - t_insert_start) * 1000.0
        entry.count += insert_count

        per_query_ms: list[float] = []
        t_search_start = time.perf_counter()
        for q in query_vecs:
            t0 = time.perf_counter()
            entry.index.search(q, top_k=k, ef=ef)
            per_query_ms.append((time.perf_counter() - t0) * 1000.0)
        search_ms_total = (time.perf_counter() - t_search_start) * 1000.0

    sorted_lat = sorted(per_query_ms)

    def _percentile(p: float) -> float:
        # Nearest-rank percentile: index = ceil(p/100 * N) - 1, clamped.
        idx = max(0, min(len(sorted_lat) - 1, int(np.ceil(p / 100.0 * len(sorted_lat))) - 1))
        return float(sorted_lat[idx])

    return BenchmarkResponse(
        name=entry.name,
        dim=entry.dim,
        metric=entry.metric,
        insert_count=insert_count,
        search_count=search_count,
        k=k,
        ef=ef,
        seed=seed,
        insert_ms_total=round(insert_ms_total, 3),
        insert_throughput_vps=round(insert_count / (insert_ms_total / 1000.0), 1),
        search_ms_total=round(search_ms_total, 3),
        search_throughput_qps=round(search_count / (search_ms_total / 1000.0), 1),
        search_p50_ms=round(_percentile(50), 3),
        search_p95_ms=round(_percentile(95), 3),
        search_p99_ms=round(_percentile(99), 3),
    )


@app.delete("/index/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_index(name: str) -> None:
    with _REGISTRY_LOCK:
        if name not in _REGISTRY:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"index '{name}' not found",
            )
        del _REGISTRY[name]
    return None
