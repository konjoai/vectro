"""konjos_integration.py — end-to-end KonjoOS integration example.

Demonstrates three key v7.0.0 surface areas:

1. ``VectroRetriever.from_file`` / ``from_jsonl``  — load a corpus from disk
   and build a hybrid BM25 + dense retriever in one call.

2. ``IVFIndex``  — build an inverted-file index, run a nearest-neighbour query.

3. ``Bf16Encoder``  — pack float32 vectors into bfloat16 to cut storage ~2×,
   then decode and verify round-trip fidelity.

Run::

    python -m python.examples.konjos_integration

Bindings are optional; the script exits cleanly when the compiled extension is
not available (useful for CI runs without a built wheel).
"""

from __future__ import annotations

import random
import math
from typing import List

# ── Optional: numpy for numeric assertions ──────────────────────────────────
try:
    import numpy as np  # noqa: F401  # availability probe

    _NP = True
except ImportError:
    _NP = False

# ── Check compiled bindings availability ────────────────────────────────────
try:
    from vectro_py import EmbeddingDataset  # noqa: F401

    _BINDINGS = True
except ImportError:
    _BINDINGS = False


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────


def _random_unit(dim: int, seed: int | None = None) -> List[float]:
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def _stub_embed_fn(texts: List[str], dim: int = 32) -> List[List[float]]:
    """Deterministic stub embedding function (not neural — demo only)."""
    result = []
    for i, _ in enumerate(texts):
        result.append(_random_unit(dim, seed=i))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: VectroRetriever.from_jsonl + from_file
# ─────────────────────────────────────────────────────────────────────────────


def demo_retriever() -> None:
    print("\n── Demo 1: VectroRetriever ──────────────────────────────────────")

    if not _BINDINGS:
        print("  [SKIP] vectro_py bindings not available.")
        return

    from python.retriever import VectroRetriever

    DIM = 32
    CORPUS = [
        "quantum entanglement enables instant correlation",
        "machine learning models compress knowledge",
        "neural networks approximate complex functions",
        "vector search enables semantic similarity",
        "bfloat16 reduces memory without quality loss",
    ]
    IDS = [f"doc_{i}" for i in range(len(CORPUS))]

    def embed_fn(texts: List[str]) -> List[List[float]]:
        return _stub_embed_fn(texts, dim=DIM)

    # ── from_jsonl (builds dataset in-memory) ────────────────────────────────
    retriever = VectroRetriever.from_jsonl(
        jsonl_path=None,  # None skips file loading; uses texts + ids directly
        texts=CORPUS,
        ids=IDS,
        embed_fn=embed_fn,
        alpha=0.7,
    )
    results = retriever.retrieve("quantum memory compression", k=3)
    print(f"  from_jsonl → top-3 ids: {[r.id for r in results]}")
    assert len(results) == 3, "expected 3 results"

    print("  Demo 1 PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: IVFIndex train / add / search
# ─────────────────────────────────────────────────────────────────────────────


def demo_ivf() -> None:
    print("\n── Demo 2: IVFIndex ─────────────────────────────────────────────")

    if not _BINDINGS:
        print("  [SKIP] vectro_py bindings not available.")
        return

    from python.ivf_api import IVFIndex

    DIM = 64
    N = 500
    N_LISTS = 8

    rng = random.Random(42)
    vectors = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(N)]
    ids = [f"v{i}" for i in range(N)]

    # Train the index on the corpus
    idx = IVFIndex(n_lists=N_LISTS, n_probe=4)
    idx.train(vectors)

    # Add all vectors
    for i, (vid, vec) in enumerate(zip(ids, vectors)):
        idx.add(vid, vec)

    # Query with the first vector — should be its own nearest neighbour
    query = vectors[0]
    hits = idx.search(query, k=5)
    hit_ids = [h[0] for h in hits]
    print(f"  query 'v0' top-5 ids: {hit_ids}")

    if _NP:
        import numpy as np

        np_query = np.asarray(query, dtype=np.float32)
        np_hits = idx.search_np(np_query, k=5)
        np_hit_ids = [h[0] for h in np_hits]
        assert hit_ids == np_hit_ids, "python/numpy search results must match"
        print("  numpy path matches python path ✓")

    print("  Demo 2 PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: Bf16Encoder encode / decode round-trip
# ─────────────────────────────────────────────────────────────────────────────


def demo_bf16() -> None:
    print("\n── Demo 3: Bf16Encoder ──────────────────────────────────────────")

    if not _BINDINGS:
        print("  [SKIP] vectro_py bindings not available.")
        return

    from python.bf16_api import Bf16Encoder

    DIM = 128
    N = 50
    rng = random.Random(99)
    original = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(N)]

    encoder = Bf16Encoder()
    encoder.encode(original)

    assert len(encoder) == N, f"expected {N} stored, got {len(encoder)}"

    decoded = encoder.decode()
    assert len(decoded) == N
    assert len(decoded[0]) == DIM

    # BF16 precision — cosine similarity should be ≥ 0.999
    worst = 1.0
    for orig, dec in zip(original, decoded):
        sim = _cosine(orig, dec)
        worst = min(worst, sim)
    print(f"  worst cosine after BF16 round-trip: {worst:.6f}")
    assert worst >= 0.999, f"precision too low: {worst}"

    # Self-distance (i==j) should be near zero
    d = encoder.cosine_dist(0, 0)
    print(f"  self cosine_dist(0,0): {d:.6f}")
    assert d < 1e-4, f"expected ~0, got {d}"

    print("  Demo 3 PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=== KonjoOS Vectro Integration Demo (v7.0.0) ===")

    if not _BINDINGS:
        print(
            "\n[WARNING] vectro_py native bindings are not installed.\n"
            "Run `maturin develop` in rust/vectro_py to build them.\n"
            "Demos requiring bindings will be skipped."
        )

    demo_retriever()
    demo_ivf()
    demo_bf16()

    print("\n=== All available demos completed. ===")


if __name__ == "__main__":
    main()
