#!/usr/bin/env python3
"""demo/server.py — live HTTP backend for the Vectro demo.

Runs **real Vectro code** behind the demo page so every number you see
in the browser is measured on this machine, right now.  No simulations,
no precomputed numbers.

    python3 demo/server.py [--port 8765]

Then open the URL it prints (default http://127.0.0.1:8765/).

Endpoints
---------
GET  /                  → serves demo/index.html
POST /api/compress      → real Vectro.compress on a synthetic batch
POST /api/search        → real VectroDSPyRetriever search over a built-in corpus
POST /api/benchmark     → real INT8 encode throughput on this CPU
GET  /api/index-stats   → current corpus / compression stats
GET  /api/health        → liveness probe (used by the page banner)

Standard library only: ``http.server`` + ``socketserver``.  No Flask,
no FastAPI, no extra installs.  Vectro itself is the only third-party
dependency, and it's resolved from the repo root via ``sys.path``.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import socketserver
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="[vectro-demo] %(message)s")
_LOG = logging.getLogger("vectro.demo")


def tracing_warn(msg: str) -> None:
    """Surface a swallowed fallback — never fail silently (CLAUDE.md)."""
    _LOG.warning(msg)

# DSPy isn't required to be installed — the retriever falls back to a
# plain object when it's missing.  Install a tiny stub so the demo
# server doesn't depend on the real SDK.
if "dspy" not in sys.modules:
    _stub = types.ModuleType("dspy")

    class _Prediction:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    _stub.Prediction = _Prediction
    sys.modules["dspy"] = _stub

import python as vectro  # noqa: E402  (sys.path manipulation above)
from python.integrations.dspy_integration import VectroDSPyRetriever  # noqa: E402
from python.hnsw_api import HNSWIndex  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Demo corpus — small, hand-curated, identical to the static map in
# index.html so the search panel agrees with the in-page visualisation.
# ─────────────────────────────────────────────────────────────────────────

CORPUS: List[Dict[str, Any]] = [
    # Geography
    {"text": "Paris is the capital of France",         "tags": ["paris", "france", "capital"]},
    {"text": "Berlin is the capital of Germany",       "tags": ["berlin", "germany", "capital"]},
    {"text": "Tokyo is the capital of Japan",          "tags": ["tokyo", "japan", "capital"]},
    {"text": "Athens is the capital of Greece",        "tags": ["athens", "greece", "capital"]},
    {"text": "Rome is the capital of Italy",           "tags": ["rome", "italy", "capital"]},
    {"text": "Madrid is the capital of Spain",         "tags": ["madrid", "spain", "capital"]},
    {"text": "London is the capital of the UK",        "tags": ["london", "uk", "capital"]},
    # Climate
    {"text": "Berlin gets cold and wet in winter",     "tags": ["berlin", "weather", "cold"]},
    {"text": "Tokyo summers are humid and rainy",      "tags": ["tokyo", "weather", "summer"]},
    {"text": "Athens has a mild Mediterranean climate", "tags": ["athens", "weather"]},
    {"text": "Cairo is hot and arid year round",       "tags": ["cairo", "weather", "hot"]},
    {"text": "Moscow has long snowy winters",          "tags": ["moscow", "weather", "snow"]},
    # Food
    {"text": "French cuisine emphasises butter and wine", "tags": ["france", "food"]},
    {"text": "Italian pasta carbonara comes from Rome",   "tags": ["rome", "italy", "food"]},
    {"text": "Japanese ramen has many regional styles",   "tags": ["japan", "food"]},
    {"text": "Greek salad is a Mediterranean staple",     "tags": ["greece", "food"]},
    {"text": "Sushi originated in Japan",                  "tags": ["japan", "food", "history"]},
    # Transit
    {"text": "Trains in France connect Paris to Marseille at 320 km/h", "tags": ["france", "trains"]},
    {"text": "The Tokyo subway carries 8 million riders per day",       "tags": ["tokyo", "transit"]},
    {"text": "Italy's high-speed rail spans Milan to Naples",           "tags": ["italy", "trains"]},
    {"text": "Berlin's U-Bahn is one of Europe's oldest",               "tags": ["berlin", "transit"]},
    # Landmarks
    {"text": "The Eiffel Tower is in Paris",            "tags": ["paris", "landmark"]},
    {"text": "Mount Fuji is a volcano on Honshu",       "tags": ["japan", "landmark"]},
    {"text": "Acropolis sits above Athens",             "tags": ["athens", "landmark"]},
    {"text": "Colosseum is an amphitheatre in Rome",    "tags": ["rome", "landmark"]},
    {"text": "Brandenburg Gate stands in central Berlin", "tags": ["berlin", "landmark"]},
]

EMBED_DIM = 128


# Deterministic embed_fn — concatenates per-token random projections so
# that "France" + "capital" land near "Paris is the capital of France".
# Fully real numpy arithmetic; no models downloaded.
_VOCAB_RNG = np.random.default_rng(seed=0xC0DEC0DE)
_VOCAB_BASIS: Dict[str, np.ndarray] = {}


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "they",
    "this", "to", "was", "were", "where", "which", "with", "what", "who", "how",
    "many", "much", "do", "does", "i", "you", "we", "us", "our", "your", "their",
    "would", "should", "could", "can", "will", "may", "one",
}


def _token_vec(tok: str) -> np.ndarray:
    if tok not in _VOCAB_BASIS:
        # Stable per-token random projection — same token always lands in
        # the same direction.  L2-normalised so dot products correspond
        # to cosine similarity.
        rng = np.random.default_rng(abs(hash(("vectro-demo-v1", tok))) & 0xFFFFFFFF)
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        _VOCAB_BASIS[tok] = v
    return _VOCAB_BASIS[tok]


def _content_tokens(text: str) -> List[str]:
    raw = [t for t in text.lower().replace("'", " ").replace("-", " ").split() if t]
    return [t for t in raw if t not in _STOPWORDS]


def _embed_text(text: str) -> np.ndarray:
    """Deterministic 'bag of content tokens' embedder.

    Stopwords are dropped so two-token queries like "trains europe" don't
    dilute against a 9-token sentence.  Each surviving token contributes
    its random-projection direction.  Because random projections of
    unrelated tokens are near-orthogonal in 128-D, the dot product
    between two embeddings approximates token-overlap divided by the
    geometric mean of token counts — the same intuition as TF-IDF
    cosine, just simpler and dependency-free.
    """
    toks = _content_tokens(text)
    if not toks:
        v = np.zeros(EMBED_DIM, dtype=np.float32)
        v[0] = 1.0
        return v
    acc = np.zeros(EMBED_DIM, dtype=np.float32)
    for t in toks:
        acc += _token_vec(t)
    n = float(np.linalg.norm(acc))
    if n > 0:
        acc = acc / n
    return acc


def embed_fn(x):
    if isinstance(x, str):
        return _embed_text(x)
    return np.stack([_embed_text(t) for t in x], axis=0)


# ─────────────────────────────────────────────────────────────────────────
# Build the live retriever once at startup
# ─────────────────────────────────────────────────────────────────────────

print("[vectro-demo] building live retriever ...", flush=True)
_t0 = time.perf_counter()
_corpus_texts = [item["text"] for item in CORPUS]
_corpus_embs = np.stack([_embed_text(t) for t in _corpus_texts], axis=0)

RETRIEVER = VectroDSPyRetriever(
    embed_fn=embed_fn, k=5, compression_profile="balanced",
)
RETRIEVER.add_texts(_corpus_texts, embeddings=_corpus_embs,
                    metadatas=[{"i": i, **item} for i, item in enumerate(CORPUS)])
_t1 = time.perf_counter()
print(f"[vectro-demo] retriever ready: {len(CORPUS)} passages, "
      f"dim={EMBED_DIM}, profile=balanced, {(_t1 - _t0) * 1000:.1f} ms",
      flush=True)


# ─────────────────────────────────────────────────────────────────────────
# Endpoint implementations — all hit real Vectro code paths
# ─────────────────────────────────────────────────────────────────────────

VECTRO = vectro.Vectro()


def _api_compress(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Allocate a fresh batch of synthetic vectors, compress with the
    requested mode, return *measured* memory + timing.
    """
    n_vecs = max(1, int(payload.get("n_vecs", 10_000)))
    dim    = max(1, int(payload.get("dim",    128)))
    mode   = str(payload.get("mode", "balanced"))

    # Cap to keep request latency reasonable on tiny machines.
    n_vecs = min(n_vecs, 200_000)
    dim    = min(dim, 4096)

    profile_map = {
        "int8":     "balanced",
        "fast":     "fast",
        "balanced": "balanced",
        "quality":  "quality",
        "nf4":      "quality",     # NF4 ships under the "quality" profile
        "binary":   "binary",
    }
    profile = profile_map.get(mode, "balanced")

    rng = np.random.default_rng(seed=42)
    data = rng.standard_normal((n_vecs, dim)).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    data = (data / norms).astype(np.float32)

    original_bytes = int(data.nbytes)

    t0 = time.perf_counter()
    result = VECTRO.compress(data, profile=profile)
    elapsed = time.perf_counter() - t0

    compressed_bytes = int(result.total_compressed_bytes)
    ratio = float(result.compression_ratio)
    throughput = (n_vecs / elapsed) if elapsed > 0 else float("inf")

    return {
        "n_vecs":              n_vecs,
        "dim":                 dim,
        "mode":                mode,
        "profile":             profile,
        "original_bytes":      original_bytes,
        "compressed_bytes":    compressed_bytes,
        "original_mb":         round(original_bytes  / (1024 ** 2), 4),
        "compressed_mb":       round(compressed_bytes / (1024 ** 2), 4),
        "ratio":               round(ratio, 3),
        "saved_bytes":         original_bytes - compressed_bytes,
        "timing_ms":           round(elapsed * 1000, 3),
        "throughput_vec_per_s": round(throughput, 1),
        "throughput_M_vec_s":   round(throughput / 1_000_000, 3),
    }


def _api_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Real VectroDSPyRetriever.forward against the built-in corpus."""
    query = str(payload.get("query_text", "")).strip() or "capital of France"
    k = max(1, min(int(payload.get("k", 5)), len(CORPUS)))

    t0 = time.perf_counter()
    out = RETRIEVER.forward(query, k=k)
    elapsed = time.perf_counter() - t0

    passages = list(getattr(out, "passages", []) or [])
    scores   = list(getattr(out, "scores",   []) or [])
    indices  = list(getattr(out, "indices",  []) or [])

    return {
        "query":     query,
        "k":         k,
        "passages":  passages,
        "scores":    [round(float(s), 4) for s in scores],
        "indices":   [int(i) for i in indices],
        "tags":      [list(CORPUS[i]["tags"]) for i in indices] if indices else [],
        "timing_ms": round(elapsed * 1000, 3),
    }


def _api_benchmark(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Measure real INT8 encode throughput at the requested shape."""
    n_vecs = max(1000, min(int(payload.get("n_vecs", 50_000)), 200_000))
    dim    = max(64,   min(int(payload.get("dim",    768)),   4096))
    reps   = max(1,    min(int(payload.get("reps",   3)),     8))

    rng = np.random.default_rng(seed=0xBEEF)
    data = rng.standard_normal((n_vecs, dim)).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    data = (data / norms).astype(np.float32)

    # Warmup
    _ = VECTRO.compress(data[:1000], profile="balanced")

    samples_ms: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = VECTRO.compress(data, profile="balanced")
        elapsed = time.perf_counter() - t0
        samples_ms.append(elapsed * 1000)

    best_ms = min(samples_ms)
    best_throughput = n_vecs / (best_ms / 1000.0)
    median_ms = sorted(samples_ms)[len(samples_ms) // 2]
    median_throughput = n_vecs / (median_ms / 1000.0)

    return {
        "n_vecs":               n_vecs,
        "dim":                  dim,
        "reps":                 reps,
        "samples_ms":           [round(s, 3) for s in samples_ms],
        "best_ms":              round(best_ms, 3),
        "median_ms":            round(median_ms, 3),
        "best_throughput_vec_s":   round(best_throughput, 1),
        "median_throughput_vec_s": round(median_throughput, 1),
        "best_M_vec_s":         round(best_throughput / 1_000_000, 3),
        "median_M_vec_s":       round(median_throughput / 1_000_000, 3),
        "compressed_mb":        round(result.total_compressed_bytes / (1024 ** 2), 4),
        "ratio":                round(float(result.compression_ratio), 3),
        "platform":             f"{platform.system()} / {platform.machine()}",
        "python":               platform.python_version(),
    }


def _api_index_stats(_payload: Dict[str, Any]) -> Dict[str, Any]:
    s = RETRIEVER.compression_stats
    return {
        "version":              vectro.__version__,
        "platform":             f"{platform.system()} / {platform.machine()}",
        "python":               platform.python_version(),
        "n_passages":           int(s.get("n_passages", 0)),
        "dimensions":           int(s.get("dimensions", EMBED_DIM)),
        "profile":              str(s.get("compression_profile", "balanced")),
        "original_mb":          float(s.get("original_mb", 0.0)),
        "compressed_mb":        float(s.get("compressed_mb", 0.0)),
        "memory_saved_mb":      float(s.get("memory_saved_mb", 0.0)),
        "compression_ratio":    float(s.get("compression_ratio", 1.0)),
    }


def _api_health(_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok":       True,
        "version":  vectro.__version__,
        "platform": f"{platform.system()} / {platform.machine()}",
        "python":   platform.python_version(),
        "uptime_s": round(time.monotonic(), 2),
    }


# ─────────────────────────────────────────────────────────────────────────
# v5.1.0 — P1 endpoints: recall estimator, compaction, metadata filter
# ─────────────────────────────────────────────────────────────────────────
#
# A small in-process HNSWIndex is seeded once at startup from the CORPUS
# embeddings so all three endpoints have a realistic index to operate on.
# The index is intentionally separate from RETRIEVER so the user can
# delete, compact, and re-estimate without affecting the search demo.

print("[vectro-demo] building demo HNSW index for P1 endpoints ...", flush=True)
_HNSW_LOCK = threading.Lock()
_DEMO_HNSW: HNSWIndex = HNSWIndex(M=8, ef_construction=80)
_t_hnsw0 = time.perf_counter()
_demo_vecs = np.stack([_embed_text(item["text"]) for item in CORPUS], axis=0)
_demo_meta = [{"category": item["tags"][0] if item["tags"] else "other",
               "tags": item["tags"], "text": item["text"]}
              for item in CORPUS]
_DEMO_HNSW.add(_demo_vecs, metadata=_demo_meta)
print(f"[vectro-demo] HNSW ready: {len(_DEMO_HNSW)} vectors, "
      f"{(time.perf_counter() - _t_hnsw0)*1000:.1f} ms", flush=True)


def _api_recall_estimate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate Recall@k against brute-force ground truth.

    Body: ``{sample_size?: int, k?: int, ef?: int}``
    Returns the full result dict from ``HNSWIndex.estimate_recall()``,
    plus a plain-English ``label`` for the demo gauge.
    """
    sample_size = max(1, min(int(payload.get("sample_size", 50)), len(_DEMO_HNSW)))
    k           = max(1, min(int(payload.get("k", 5)), len(_DEMO_HNSW) - 1))
    ef          = max(k, int(payload.get("ef", 32)))
    with _HNSW_LOCK:
        result = _DEMO_HNSW.estimate_recall(sample_size=sample_size, k=k, ef=ef)
    r = result["recall"]
    label = "Excellent" if r >= 0.95 else "Good" if r >= 0.85 else "Fair" if r >= 0.70 else "Poor"
    return {**result, "label": label}


def _api_compact(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run graph compaction — tombstone removal + orphan reconnection.

    Optionally delete a random subset first (for demo effect).
    Body: ``{delete_n?: int}`` — soft-delete up to *delete_n* random vectors
    before compacting (default 3).
    """
    delete_n = max(0, min(int(payload.get("delete_n", 3)), len(_DEMO_HNSW) - 2))
    with _HNSW_LOCK:
        # Soft-delete a few random live vectors
        alive = [i for i in range(len(_DEMO_HNSW._vectors))
                 if i not in _DEMO_HNSW._deleted]
        rng = np.random.default_rng()
        chosen = rng.choice(alive, size=min(delete_n, len(alive)), replace=False)
        for nid in chosen:
            _DEMO_HNSW.delete(int(nid))
        stats_before = _DEMO_HNSW.stats()
        t0 = time.perf_counter()
        result = _DEMO_HNSW.compact()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        stats_after  = _DEMO_HNSW.stats()
    return {
        "deleted_count_before": stats_before["n_deleted"],
        "orphan_count_before":  stats_before["orphan_count"],
        "removed":              result["removed"],
        "repaired":             result["repaired"],
        "orphan_count_after":   stats_after["orphan_count"],
        "timing_ms":            elapsed_ms,
        "n_alive":              stats_after["n_alive"],
    }


def _api_hnsw_stats(_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return health stats for the demo HNSW index."""
    with _HNSW_LOCK:
        s = _DEMO_HNSW.stats()
    return s


def _api_filtered_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    """HNSW pre-filtered nearest-neighbour search.

    Body: ``{query_text: str, k?: int, filter?: {field: value}}``
    Runs search on the demo HNSW index with optional metadata pre-filter.
    """
    query  = str(payload.get("query_text", "")).strip() or "capital"
    k      = max(1, min(int(payload.get("k", 5)), len(_DEMO_HNSW)))
    filt   = payload.get("filter") or None

    # Validate filter dict
    if filt is not None:
        if not isinstance(filt, dict):
            filt = None
        else:
            # Only allow simple string/number equality filters
            filt = {str(fk): fv for fk, fv in filt.items()
                    if isinstance(fv, (str, int, float, bool))}

    q_vec = _embed_text(query)
    t0 = time.perf_counter()
    with _HNSW_LOCK:
        indices, distances = _DEMO_HNSW.search(q_vec, k=k, ef=max(k, 32), filter=filt)
    elapsed = time.perf_counter() - t0

    results = []
    for nid, dist in zip(indices.tolist(), distances.tolist()):
        meta = _DEMO_HNSW._metadata[nid] if nid < len(_DEMO_HNSW._metadata) else {}
        results.append({
            "id":       int(nid),
            "distance": round(float(dist), 4),
            "similarity": round(max(0.0, 1.0 - float(dist)), 4),
            "text":     (meta or {}).get("text", ""),
            "category": (meta or {}).get("category", ""),
            "tags":     (meta or {}).get("tags", []),
        })

    return {
        "query":      query,
        "filter":     filt,
        "k":          k,
        "results":    results,
        "timing_ms":  round(elapsed * 1000, 3),
    }


# ─────────────────────────────────────────────────────────────────────────
# Multi-index registry  (named HNSWIndex instances, created on demand)
# ─────────────────────────────────────────────────────────────────────────

_INDEXES: Dict[str, HNSWIndex]          = {}
_INDEXES_META: Dict[str, Dict[str, Any]] = {}   # creation params + timestamps
_INDEXES_LOCK = threading.Lock()

_INDEX_ROUTE_RE = re.compile(
    r"^/api/indexes(?:/(?P<name>[A-Za-z0-9_.\-]{1,64})(?:/(?P<action>[a-z_]+))?)?$"
)

# ── PCA / k-means (numpy-only, no sklearn) ──────────────────────────────

def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Project (n, d) float32 matrix to 2-D via SVD.  Returns (n, 2)."""
    mu = vectors.mean(axis=0)
    c  = vectors - mu
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return (c @ Vt[:2].T).astype(np.float32)


def _kmeans(coords: np.ndarray, k: int, n_iter: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """Simple Lloyd's k-means on (n, 2) coords.  Returns (labels, centers)."""
    n = len(coords)
    k = min(k, n)
    rng = np.random.default_rng(seed=42)
    centers = coords[rng.choice(n, k, replace=False)].copy().astype(np.float64)
    labels  = np.zeros(n, dtype=np.int32)
    for _ in range(n_iter):
        dists  = np.linalg.norm(coords[:, None].astype(np.float64) - centers[None], axis=2)
        labels = dists.argmin(axis=1).astype(np.int32)
        for ki in range(k):
            mask = labels == ki
            if mask.any():
                centers[ki] = coords[mask].astype(np.float64).mean(axis=0)
    return labels, centers.astype(np.float32)


# ── Index CRUD handlers ──────────────────────────────────────────────────

def _idx_list() -> Dict[str, Any]:
    with _INDEXES_LOCK:
        result = []
        for name, idx in _INDEXES.items():
            s = idx.stats()
            meta = _INDEXES_META.get(name, {})
            result.append({
                "name":           name,
                "n_alive":        s["n_alive"],
                "n_total":        s["n_total"],
                "n_deleted":      s["n_deleted"],
                "M":              idx.M,
                "ef_construction": idx.ef_construction,
                "space":          idx.space,
                "dim":            meta.get("dim", 0),
                "created_at":     meta.get("created_at", ""),
            })
    return {"indexes": result}


def _idx_create(name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    if not re.match(r"^[A-Za-z0-9_.\-]{1,64}$", name):
        raise ValueError(f"invalid index name: {name!r}")
    M                = max(4, min(int(body.get("M", 16)), 64))
    ef_construction  = max(M, min(int(body.get("ef_construction", 200)), 500))
    space            = str(body.get("space", "cosine"))
    if space not in ("cosine", "l2"):
        raise ValueError(f"space must be 'cosine' or 'l2', got {space!r}")
    with _INDEXES_LOCK:
        if name in _INDEXES:
            raise ValueError(f"index {name!r} already exists; DELETE it first")
        idx = HNSWIndex(M=M, ef_construction=ef_construction, space=space)
        _INDEXES[name] = idx
        _INDEXES_META[name] = {
            "dim": int(body.get("dim", 0)),
            "ef_construction": ef_construction,
            "M": M,
            "space": space,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    return {"name": name, "M": M, "ef_construction": ef_construction, "space": space}


def _idx_delete(name: str) -> Dict[str, Any]:
    with _INDEXES_LOCK:
        if name not in _INDEXES:
            raise KeyError(f"index {name!r} not found")
        _INDEXES.pop(name)
        _INDEXES_META.pop(name, None)
    return {"deleted": name}


def _idx_stats(name: str) -> Dict[str, Any]:
    with _INDEXES_LOCK:
        if name not in _INDEXES:
            raise KeyError(f"index {name!r} not found")
        s    = _INDEXES[name].stats()
        meta = _INDEXES_META.get(name, {})
    return {**s, **meta, "name": name}


def _idx_add(name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Add vectors to a named index.
    Body: {vectors: [[…], …], ids?: [str, …], metadata?: [{…}, …]}
    """
    with _INDEXES_LOCK:
        if name not in _INDEXES:
            raise KeyError(f"index {name!r} not found")
        idx = _INDEXES[name]

    raw_vecs = body.get("vectors") or []
    if not raw_vecs:
        raise ValueError("vectors must be a non-empty list")
    vecs = np.asarray(raw_vecs, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs[np.newaxis, :]
    if vecs.ndim != 2:
        raise ValueError("vectors must be 1-D or 2-D")

    ids_raw  = body.get("ids")
    meta_raw = body.get("metadata")
    ids_list: Optional[List[str]] = [str(x) for x in ids_raw] if ids_raw else None
    meta_list: Optional[List[Optional[Dict[str, Any]]]] = (
        [m if isinstance(m, dict) else None for m in meta_raw]
        if meta_raw else None
    )

    t0 = time.perf_counter()
    with _INDEXES_LOCK:
        # Update stored dim on first add
        if _INDEXES_META.get(name, {}).get("dim", 0) == 0:
            _INDEXES_META[name]["dim"] = vecs.shape[1]
        r = idx.add_batch(vecs, ids=ids_list, metadata=meta_list)
    elapsed = time.perf_counter() - t0

    return {**r, "timing_ms": round(elapsed * 1000, 2)}


def _idx_search(name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """kNN search in a named index.
    Body: {vector: [f32, …], k?: int, ef?: int, filter?: {…}, trace?: bool}
    """
    with _INDEXES_LOCK:
        if name not in _INDEXES:
            raise KeyError(f"index {name!r} not found")
        idx = _INDEXES[name]

    if not idx._vectors:
        return {"results": [], "timing_ms": 0.0, "k": 0}

    vec_raw = body.get("vector")
    if vec_raw is None:
        # Random vector matching the stored dimension
        d   = idx._vectors[0].shape[0]
        rng = np.random.default_rng()
        q   = rng.standard_normal(d).astype(np.float32)
    else:
        q = np.asarray(vec_raw, dtype=np.float32)

    k     = max(1, min(int(body.get("k", 10)), max(1, len(idx) - len(idx._deleted))))
    ef    = max(k, int(body.get("ef", 64)))
    filt  = body.get("filter") or None
    if filt and not isinstance(filt, dict):
        filt = None

    t0 = time.perf_counter()
    with _INDEXES_LOCK:
        result = idx.search(q, k=k, ef=ef, filter=filt, trace=bool(body.get("trace")))
    elapsed = time.perf_counter() - t0

    if len(result) == 3:
        indices, distances, tr = result
        trace_data = {
            "entry_point":          tr.entry_point,
            "layer_descents":       tr.layer_descents,
            "l0_visited":           tr.l0_visited,
            "l0_candidates_final":  [(float(d), int(n)) for d, n in tr.l0_candidates_final],
        }
    else:
        indices, distances = result
        trace_data = None

    hits = []
    with _INDEXES_LOCK:
        for nid, dist in zip(indices.tolist(), distances.tolist()):
            meta = idx._metadata[nid] if nid < len(idx._metadata) else {}
            hits.append({
                "id":         int(nid),
                "distance":   round(float(dist), 5),
                "similarity": round(max(0.0, 1.0 - float(dist)), 5),
                "metadata":   meta or {},
            })

    resp = {"results": hits, "k": k, "timing_ms": round(elapsed * 1000, 3)}
    if trace_data is not None:
        resp["trace"] = trace_data
    return resp


def _idx_benchmark(name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Measure kNN search latency distribution.
    Body: {k?: int, ef?: int, n_queries?: int, warmup?: int}
    """
    with _INDEXES_LOCK:
        if name not in _INDEXES:
            raise KeyError(f"index {name!r} not found")
        idx = _INDEXES[name]
        n_vecs = max(1, len(idx._vectors) - len(idx._deleted))

    if n_vecs == 0:
        return {"error": "index is empty"}

    d         = idx._vectors[0].shape[0]
    k         = max(1, min(int(body.get("k", 10)), n_vecs))
    ef        = max(k, int(body.get("ef", 64)))
    n_queries = max(10, min(int(body.get("n_queries", 200)), 2000))
    warmup    = max(0,  min(int(body.get("warmup", 5)), 20))

    rng = np.random.default_rng(seed=0)
    queries = rng.standard_normal((n_queries + warmup, d)).astype(np.float32)

    # Warmup
    for i in range(warmup):
        with _INDEXES_LOCK:
            idx.search(queries[i], k=k, ef=ef)

    # Timed runs
    latencies_ms: List[float] = []
    for i in range(warmup, n_queries + warmup):
        t0 = time.perf_counter()
        with _INDEXES_LOCK:
            idx.search(queries[i], k=k, ef=ef)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    lats = sorted(latencies_ms)
    n = len(lats)

    def pct(p: float) -> float:
        idx_f = (p / 100) * (n - 1)
        lo, hi = int(idx_f), min(int(idx_f) + 1, n - 1)
        return round(lats[lo] + (lats[hi] - lats[lo]) * (idx_f - lo), 4)

    mean_ms = round(sum(lats) / n, 4)
    qps     = round(1000 / mean_ms if mean_ms > 0 else float("inf"), 1)

    # Recall estimate on small sample
    recall_r = None
    try:
        sample = min(20, n_vecs - 1)
        if sample > 1:
            with _INDEXES_LOCK:
                rr = idx.estimate_recall(sample_size=sample, k=k, ef=ef)
            recall_r = round(rr["recall"], 4)
    except Exception:
        pass

    return {
        "n_queries":   n_queries,
        "k":           k,
        "ef":          ef,
        "n_vectors":   n_vecs,
        "dim":         d,
        "p50_ms":      pct(50),
        "p95_ms":      pct(95),
        "p99_ms":      pct(99),
        "mean_ms":     mean_ms,
        "min_ms":      round(lats[0], 4),
        "max_ms":      round(lats[-1], 4),
        "qps":         qps,
        "recall_k":    recall_r,
    }


def _idx_project(name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """PCA-project stored vectors to 2D.  Returns {points: [{id, x, y, meta}]}."""
    with _INDEXES_LOCK:
        if name not in _INDEXES:
            raise KeyError(f"index {name!r} not found")
        idx = _INDEXES[name]
        alive = [i for i in range(len(idx._vectors)) if i not in idx._deleted]
        if len(alive) < 2:
            return {"points": [], "n": 0}
        vecs = np.stack([idx._vectors[i] for i in alive], axis=0)
        metas = [idx._metadata[i] if i < len(idx._metadata) else {} for i in alive]

    t0     = time.perf_counter()
    coords = _pca_2d(vecs)
    elapsed = time.perf_counter() - t0

    # Normalise to [-1, 1] for consistent canvas rendering
    mx = np.abs(coords).max()
    if mx > 0:
        coords = coords / mx

    points = [
        {
            "id":    int(alive[i]),
            "x":     round(float(coords[i, 0]), 5),
            "y":     round(float(coords[i, 1]), 5),
            "meta":  metas[i] or {},
        }
        for i in range(len(alive))
    ]
    return {"points": points, "n": len(points), "timing_ms": round(elapsed * 1000, 2)}


def _idx_cluster(name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """k-means cluster the projected 2D coords.
    Requires a prior call to /project to get the coords, or does it inline.
    Returns {labels: [int, …], centers: [[x,y], …]}.
    """
    k = max(2, min(int(body.get("k", 4)), 12))
    proj = _idx_project(name, body)
    if not proj["points"]:
        return {"labels": [], "centers": [], "k": k}

    coords = np.array([[p["x"], p["y"]] for p in proj["points"]], dtype=np.float32)
    labels, centers = _kmeans(coords, k=k)
    return {
        "points": [
            {**pt, "cluster": int(labels[i])}
            for i, pt in enumerate(proj["points"])
        ],
        "centers": [[round(float(c[0]), 5), round(float(c[1]), 5)] for c in centers],
        "k":       k,
        "n":       len(proj["points"]),
        "timing_ms": proj.get("timing_ms", 0.0),
    }


# ─────────────────────────────────────────────────────────────────────────
# Quantization Lab — every Vectro codec, measured side-by-side on one batch
# ─────────────────────────────────────────────────────────────────────────
#
# This is the flagship "show every feature live" endpoint.  A single batch
# of vectors is encoded with INT8, NF4, Product Quantization, Residual
# Quantization, and Binary.  For each codec we report *measured* numbers:
# real compressed byte count, reconstruction cosine fidelity (encode →
# decode → compare), and wall-clock build+encode time.  Nothing is
# precomputed — every value is produced on this machine, on request.

_LAB_LOCK = threading.Lock()

# Theme colour per codec — kept identical to index.html so the Pareto
# plot, the cards, and the geometry panel all agree.
_LAB_COLORS: Dict[str, str] = {
    "int8":   "#22d3ee",
    "nf4":    "#7c3aed",
    "rq":     "#10b981",
    "pq":     "#f59e0b",
    "binary": "#ef4444",
}


def _lab_batch(n: int, dim: int, dist: str, seed: int = 42) -> np.ndarray:
    """Allocate a fresh, L2-normalised batch for the lab.

    ``dist='clustered'`` plants 8 Gaussian blobs so codecs that exploit
    structure (PQ, RQ) visibly pull ahead of the data-agnostic ones.
    """
    rng = np.random.default_rng(seed)
    if dist == "clustered":
        n_clusters = 8
        centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
        labels = rng.integers(0, n_clusters, size=n)
        data = centers[labels] + 0.22 * rng.standard_normal((n, dim)).astype(np.float32)
    else:
        data = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (data / norms).astype(np.float32)


def _mean_cosine(orig: np.ndarray, rec: np.ndarray) -> float:
    """Mean cosine similarity between originals and reconstructions."""
    a = orig / (np.linalg.norm(orig, axis=1, keepdims=True) + 1e-9)
    b = rec / (np.linalg.norm(rec, axis=1, keepdims=True) + 1e-9)
    return float(np.clip((a * b).sum(axis=1), -1.0, 1.0).mean())


def _pick_divisor(dim: int, target: int) -> int:
    """Largest divisor of ``dim`` not exceeding ``target`` (≥ 1)."""
    for m in range(min(target, dim), 0, -1):
        if dim % m == 0:
            return m
    return 1


def _enc_int8(data: np.ndarray) -> Tuple[np.ndarray, int]:
    result = VECTRO.compress(data, precision_mode="int8")
    return result.reconstruct_batch(), int(result.total_compressed_bytes)


def _enc_nf4(data: np.ndarray) -> Tuple[np.ndarray, int]:
    from python.nf4_api import dequantize_nf4, quantize_nf4
    packed, scales = quantize_nf4(data)
    rec = dequantize_nf4(packed, scales, data.shape[1])
    return rec, int(packed.nbytes + scales.nbytes)


def _enc_binary(data: np.ndarray) -> Tuple[np.ndarray, int]:
    from python.binary_api import dequantize_binary, quantize_binary
    packed = quantize_binary(data)
    rec = dequantize_binary(packed, data.shape[1])
    return rec, int(packed.nbytes)


# Codebook codecs (PQ/RQ) train on a capped subsample so the lab stays
# interactive — codebooks generalise, so a few thousand vectors fit them
# well, and we still *encode the whole batch* for honest ratio/fidelity.
_LAB_TRAIN_CAP = 1500


def _enc_pq(data: np.ndarray) -> Tuple[np.ndarray, int]:
    import python.pq_api as pq
    m = _pick_divisor(data.shape[1], 16)
    train = data[:_LAB_TRAIN_CAP]
    cb = pq.train_pq_codebook(train, n_subspaces=m, n_centroids=256, random_state=0)
    codes = pq.pq_encode(data, cb)
    return pq.pq_decode(codes, cb), int(codes.nbytes)


def _enc_rq(data: np.ndarray) -> Tuple[np.ndarray, int]:
    from python.rq_api import ResidualQuantizer
    m = _pick_divisor(data.shape[1], 8)
    rq = ResidualQuantizer(n_passes=4, n_subspaces=m, n_centroids=64, seed=0)
    rq.train(data[:_LAB_TRAIN_CAP])
    codes = rq.encode(data)
    rec = rq.decode(codes)
    total = sum(int(c.nbytes) for c in codes) if isinstance(codes, list) else int(codes.nbytes)
    return rec, total


# Ordered so the UI renders dense→sparse left to right.
_LAB_CODECS: List[Tuple[str, str, Callable[[np.ndarray], Tuple[np.ndarray, int]]]] = [
    ("int8",   "INT8",   _enc_int8),
    ("nf4",    "NF4",    _enc_nf4),
    ("rq",     "RQ",     _enc_rq),
    ("pq",     "PQ",     _enc_pq),
    ("binary", "Binary", _enc_binary),
]


def _pca_basis(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, top-2 components)`` for projecting onto a shared 2-D plane."""
    mu = vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(vectors - mu, full_matrices=False)
    return mu, vt[:2]


def _run_one_codec(
    name: str, label: str, fn: Callable[[np.ndarray], Tuple[np.ndarray, int]],
    data: np.ndarray, original_bytes: int,
) -> Dict[str, Any]:
    """Encode/decode with a single codec and collect measured metrics."""
    try:
        t0 = time.perf_counter()
        rec, comp_bytes = fn(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    except ImportError as exc:
        tracing_warn(f"lab codec {name!r} unavailable: {exc}")
        return {"name": name, "label": label, "available": False,
                "reason": "optional dependency missing", "color": _LAB_COLORS[name]}
    except Exception as exc:  # noqa: BLE001 — surface, never mask (CLAUDE.md)
        tracing_warn(f"lab codec {name!r} failed: {exc}")
        return {"name": name, "label": label, "available": False,
                "reason": str(exc), "color": _LAB_COLORS[name]}

    fidelity = _mean_cosine(data, rec)
    ratio = original_bytes / comp_bytes if comp_bytes else 0.0
    n = data.shape[0]
    return {
        "name":          name,
        "label":         label,
        "available":     True,
        "color":         _LAB_COLORS[name],
        "ratio":         round(ratio, 3),
        "fidelity":      round(fidelity, 5),
        "compressed_bytes": comp_bytes,
        "compressed_mb": round(comp_bytes / (1024 ** 2), 5),
        "timing_ms":     round(elapsed_ms, 3),
        "throughput_M_vec_s": round((n / (elapsed_ms / 1000.0)) / 1e6, 4) if elapsed_ms > 0 else None,
        "_rec":          rec,   # consumed by the geometry projection, then dropped
    }


def _lab_geometry(
    data: np.ndarray, codecs: List[Dict[str, Any]], max_points: int = 140,
) -> Dict[str, Any]:
    """Project originals and every reconstruction onto one shared PCA plane.

    Sharing the basis makes the distortion comparable across codecs — you
    can literally watch Binary collapse the cloud while INT8 leaves it
    untouched.
    """
    n = data.shape[0]
    take = min(max_points, n)
    idx = np.linspace(0, n - 1, take).astype(int)
    sample = data[idx]
    mu, comps = _pca_basis(sample)

    def project(mat: np.ndarray) -> np.ndarray:
        return ((mat[idx] - mu) @ comps.T).astype(np.float32)

    orig_xy = project(data)
    layers: Dict[str, List[List[float]]] = {}
    scale = float(np.abs(orig_xy).max()) or 1.0
    for c in codecs:
        if c.get("available") and c.get("_rec") is not None:
            xy = project(c["_rec"]) / scale
            layers[c["name"]] = [[round(float(x), 4), round(float(y), 4)] for x, y in xy]
    orig_norm = [[round(float(x / scale), 4), round(float(y / scale), 4)] for x, y in orig_xy]
    return {"original": orig_norm, "reconstructions": layers, "n": take}


def _api_quantize_lab(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Encode one batch with every codec; return measured comparison + geometry."""
    n_vecs = max(256, min(int(payload.get("n_vecs", 2500)), 12_000))
    dim    = max(16,  min(int(payload.get("dim",    256)),  1024))
    dist   = "clustered" if str(payload.get("dist", "gaussian")) == "clustered" else "gaussian"

    with _LAB_LOCK:
        data = _lab_batch(n_vecs, dim, dist)
        original_bytes = int(data.nbytes)
        codecs = [_run_one_codec(name, label, fn, data, original_bytes)
                  for name, label, fn in _LAB_CODECS]
        geometry = _lab_geometry(data, codecs)

    for c in codecs:
        c.pop("_rec", None)  # never serialise raw arrays

    return {
        "n_vecs":         n_vecs,
        "dim":            dim,
        "dist":           dist,
        "original_bytes": original_bytes,
        "original_mb":    round(original_bytes / (1024 ** 2), 4),
        "codecs":         codecs,
        "geometry":       geometry,
        "platform":       f"{platform.system()} / {platform.machine()}",
    }


# ─────────────────────────────────────────────────────────────────────────
# LoRA adapter compression — NF4-quantise the low-rank fine-tuning factors
# ─────────────────────────────────────────────────────────────────────────
#
# A LoRA adapter stores, per attention projection, two low-rank factors
# A (rank × in) and B (out × rank) whose product B·A is the weight delta.
# compress_lora_adapter() NF4-quantises every factor; we report the real
# per-factor cosine fidelity, the reconstructed-delta cosine, and the byte
# savings vs both the float32 factors and the equivalent dense weight.

_LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _nf4_payload_bytes(payload: Dict[str, Any]) -> int:
    """Sum the byte footprint of an NF4 payload dict ({packed, scales, ...})."""
    return int(sum(v.nbytes for v in payload.values() if hasattr(v, "nbytes")))


def _matrix_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Flattened cosine similarity between two matrices."""
    av, bv = a.ravel(), b.ravel()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-9
    return float(np.clip((av @ bv) / denom, -1.0, 1.0))


def _api_lora(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compress a synthetic LoRA adapter and return measured fidelity + size."""
    from python.lora_api import compress_lora_adapter, decompress_lora

    rank = max(2,  min(int(payload.get("rank", 16)),          128))
    din  = max(32, min(int(payload.get("in_features", 768)),  4096))
    dout = max(32, min(int(payload.get("out_features", 768)), 4096))

    rng = np.random.default_rng(7)
    adapter = {
        m: (
            rng.standard_normal((rank, din)).astype(np.float32),
            rng.standard_normal((dout, rank)).astype(np.float32),
        )
        for m in _LORA_MODULES
    }

    t0 = time.perf_counter()
    results = compress_lora_adapter(adapter, profile="lora-nf4")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    modules: List[Dict[str, Any]] = []
    tot_orig = tot_comp = 0
    for name, (a_mat, b_mat) in adapter.items():
        res = results[name]
        orig = int(a_mat.nbytes + b_mat.nbytes)
        comp = _nf4_payload_bytes(res.A_data) + _nf4_payload_bytes(res.B_data)
        a2, b2 = decompress_lora(res)
        delta_cos = _matrix_cosine(b_mat @ a_mat, b2 @ a2)
        tot_orig += orig
        tot_comp += comp
        modules.append({
            "module":      name,
            "rank":        int(res.rank),
            "cos_A":       round(float(res.cosine_sim_A), 5),
            "cos_B":       round(float(res.cosine_sim_B), 5),
            "delta_cos":   round(delta_cos, 5),
            "orig_bytes":  orig,
            "comp_bytes":  comp,
            "ratio":       round(orig / comp, 3) if comp else 0.0,
        })

    dense_bytes = len(_LORA_MODULES) * dout * din * 4
    return {
        "rank":             rank,
        "in_features":      din,
        "out_features":     dout,
        "modules":          modules,
        "total_orig_bytes": tot_orig,
        "total_comp_bytes": tot_comp,
        "ratio":            round(tot_orig / tot_comp, 3) if tot_comp else 0.0,
        "dense_bytes":      dense_bytes,
        "vs_dense_ratio":   round(dense_bytes / tot_comp, 1) if tot_comp else 0.0,
        "timing_ms":        round(elapsed_ms, 3),
        "platform":         f"{platform.system()} / {platform.machine()}",
    }


ROUTES_POST: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "/api/compress":         _api_compress,
    "/api/search":           _api_search,
    "/api/benchmark":        _api_benchmark,
    "/api/compact":          _api_compact,
    "/api/filtered-search":  _api_filtered_search,
    "/api/recall_estimate":  _api_recall_estimate,
    "/api/quantize-lab":     _api_quantize_lab,
    "/api/lora":             _api_lora,
}
ROUTES_GET: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "/api/index-stats":  _api_index_stats,
    "/api/hnsw-stats":   _api_hnsw_stats,
    "/api/recall_estimate": lambda _: _api_recall_estimate({}),
    "/api/health":       _api_health,
}


# ─────────────────────────────────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────────────────────────────────

INDEX_HTML = (Path(__file__).resolve().parent / "index.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version = f"VectroDemo/{vectro.__version__}"

    # quiet down the default access log a bit
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(
            f"  {self.address_string()}  {fmt % args}\n"
        )

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Don't cache so the live page picks up edits during dev.
        self.send_header("Cache-Control", "no-store")
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(n) if n > 0 else b""
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _dispatch_index_route(self, method: str, path: str) -> bool:
        """Try to handle a /api/indexes/... route.  Return True if handled."""
        m = _INDEX_ROUTE_RE.match(path)
        if not m:
            return False
        name   = m.group("name")
        action = m.group("action")

        try:
            if method == "GET":
                if name is None:
                    self._send_json(_idx_list())
                else:
                    self._send_json(_idx_stats(name))

            elif method == "POST":
                body = self._read_body()
                if name is None:
                    # POST /api/indexes  — create, name comes from body
                    iname = str(body.get("name", "")).strip()
                    if not iname:
                        raise ValueError("body must contain 'name'")
                    self._send_json(_idx_create(iname, body))
                elif action == "add":
                    self._send_json(_idx_add(name, body))
                elif action == "search":
                    self._send_json(_idx_search(name, body))
                elif action == "benchmark":
                    self._send_json(_idx_benchmark(name, body))
                elif action == "project":
                    self._send_json(_idx_project(name, body))
                elif action == "cluster":
                    self._send_json(_idx_cluster(name, body))
                elif action is None:
                    # POST /api/indexes/{name}  — also valid for create
                    self._send_json(_idx_create(name, body))
                else:
                    self._send_json({"error": f"unknown action: {action}"}, status=404)

            elif method == "DELETE":
                if name is None:
                    self._send_json({"error": "DELETE requires index name"}, status=400)
                else:
                    self._send_json(_idx_delete(name))

            else:
                return False

        except (KeyError, ValueError) as exc:
            code = 404 if isinstance(exc, KeyError) else 400
            self._send_json({"error": str(exc)}, status=code)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

        return True

    def do_OPTIONS(self) -> None:        # noqa: N802
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_DELETE(self) -> None:         # noqa: N802
        path = self.path.split("?", 1)[0]
        if not self._dispatch_index_route("DELETE", path):
            self._send_json({"error": f"not found: {path}"}, status=404)

    def do_GET(self) -> None:            # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._send_html(INDEX_HTML)
            return
        if self._dispatch_index_route("GET", path):
            return
        if path in ROUTES_GET:
            try:
                self._send_json(ROUTES_GET[path]({}))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        self._send_json({"error": f"not found: {path}"}, status=404)

    def do_POST(self) -> None:           # noqa: N802
        path = self.path.split("?", 1)[0]
        if self._dispatch_index_route("POST", path):
            return
        handler = ROUTES_POST.get(path)
        if handler is None:
            self._send_json({"error": f"not found: {path}"}, status=404)
            return
        try:
            body = self._read_body()
        except Exception as exc:
            self._send_json({"error": f"bad json: {exc}"}, status=400)
            return

        try:
            self._send_json(handler(body))
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    httpd = ThreadingServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print()
    print("  ╔════════════════════════════════════════════════════╗")
    print(f"  ║  Vectro live demo — v{vectro.__version__:<8}                      ║")
    print("  ║                                                    ║")
    print(f"  ║  open: {url:<43}║")
    print(f"  ║  api:  {url + 'api/health':<43}║")
    print("  ║                                                    ║")
    print("  ║  every number on the page is measured here, now    ║")
    print("  ╚════════════════════════════════════════════════════╝")
    print("  Ctrl-C to stop", flush=True)
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[vectro-demo] shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
