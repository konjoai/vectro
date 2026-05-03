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
import os
import platform
import socketserver
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


ROUTES_POST: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "/api/compress":  _api_compress,
    "/api/search":    _api_search,
    "/api/benchmark": _api_benchmark,
}
ROUTES_GET: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "/api/index-stats": _api_index_stats,
    "/api/health":      _api_health,
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

    def do_OPTIONS(self) -> None:        # noqa: N802
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:            # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._send_html(INDEX_HTML)
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
        handler = ROUTES_POST.get(path)
        if handler is None:
            self._send_json({"error": f"not found: {path}"}, status=404)
            return
        try:
            n = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(n) if n > 0 else b""
            body = json.loads(raw.decode("utf-8")) if raw else {}
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
    print(f"  ╔════════════════════════════════════════════════════╗")
    print(f"  ║  Vectro live demo — v{vectro.__version__:<8}                      ║")
    print(f"  ║                                                    ║")
    print(f"  ║  open: {url:<43}║")
    print(f"  ║  api:  {url + 'api/health':<43}║")
    print(f"  ║                                                    ║")
    print(f"  ║  every number on the page is measured here, now    ║")
    print(f"  ╚════════════════════════════════════════════════════╝")
    print(f"  Ctrl-C to stop", flush=True)
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[vectro-demo] shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
