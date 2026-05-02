#!/usr/bin/env python3
"""Vectro — Demo Day showcase.

A single runnable script that exercises the actual vectro library end-to-end:

    1. Synthetic 50 × 128-dim corpus → compress under fast / balanced / quality
       profiles, print compression ratio, bytes saved, and reconstruction MSE.
    2. k-NN search via VectroDSPyRetriever — top-5 with cosine scores.
    3. MMR search — diversity vs relevance trade-off across lambda values.
    4. LangChainVectorStore + LlamaIndexVectorStore construction (compression
       active, no real LLM needed).
    5. VectroDSPyRetriever construction wired to the same embed_fn.
    6. OpenAIEmbeddings driving the whole pipeline against a mock OpenAI
       client — cache-stat read-out shows the SQLite cache landing rows.

No real API keys are required.  Run with::

    python3 demo/demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path
from typing import List

import numpy as np

# Ensure the repo root is importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Install a tiny stand-in for the dspy SDK so VectroDSPyRetriever returns
# a Prediction-shaped object even when the real package is absent.
if "dspy" not in sys.modules:
    _dspy = types.ModuleType("dspy")

    class _Prediction:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __repr__(self) -> str:
            return f"Prediction({list(self.__dict__.keys())})"

    _dspy.Prediction = _Prediction
    sys.modules["dspy"] = _dspy


from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from python import (
    LangChainVectorStore,
    LlamaIndexVectorStore,
    OpenAIEmbeddings,
    VectroDSPyRetriever,
    Vectro,
    __version__ as VECTRO_VERSION,
)


console = Console(width=100)


# ---------------------------------------------------------------------------
# Mock OpenAI client — deterministic, dependency-free, network-free
# ---------------------------------------------------------------------------

class _OAIData:
    def __init__(self, embedding):
        self.embedding = embedding


class _OAIResp:
    def __init__(self, data):
        self.data = data


class _OAIEmbeddingsAPI:
    """Deterministic 128-dim embedder driven by token overlap with a vocabulary.

    Each text becomes a vector in the same 128-dim space the rest of the
    demo uses, so the OpenAI provider plugs straight into the LangChain /
    LlamaIndex / DSPy adapters without dimensional mismatch.
    """

    def __init__(self, vocabulary: List[str], dim: int = 128, calls_log=None):
        self._vocab = vocabulary
        self._dim = dim
        self.calls = 0
        self._calls_log = calls_log if calls_log is not None else []

    def create(self, *, model, input):
        self.calls += 1
        self._calls_log.append({"model": model, "n_inputs": len(input)})
        rows = []
        for text in input:
            v = np.zeros(self._dim, dtype=np.float32)
            tokens = text.lower().split()
            for tok in tokens:
                # Hash token onto two dimensions for richer signal
                h = abs(hash(tok))
                v[h % self._dim] += 1.0
                v[(h // 7) % self._dim] += 0.5
                if tok in self._vocab:
                    v[self._vocab.index(tok) % self._dim] += 1.5
            n = float(np.linalg.norm(v))
            if n > 0:
                v /= n
            else:
                v[0] = 1.0
            rows.append(_OAIData(v.tolist()))
        return _OAIResp(rows)


class _OAIClient:
    def __init__(self, vocabulary, dim=128):
        self._calls_log: List[dict] = []
        self.embeddings = _OAIEmbeddingsAPI(vocabulary, dim=dim,
                                             calls_log=self._calls_log)


# ---------------------------------------------------------------------------
# Demo corpus
# ---------------------------------------------------------------------------

CORPUS: List[str] = [
    # Geography
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
    "Tokyo is the capital of Japan",
    "Rome is the capital of Italy",
    "Madrid is the capital of Spain",
    "London is the capital of the United Kingdom",
    "Athens is the capital of Greece",
    "Cairo is the capital of Egypt",
    "Beijing is the capital of China",
    "Moscow is the capital of Russia",
    # Climate
    "Berlin gets cold and wet in winter",
    "Cairo is hot and arid year round",
    "Tokyo summers are humid and rainy",
    "Moscow has long snowy winters",
    "Athens enjoys a mild Mediterranean climate",
    # AI / ML
    "Machine learning powers modern AI systems",
    "Deep learning is a subfield of machine learning",
    "Transformers revolutionized natural language processing",
    "Large language models generate human-like text",
    "Reinforcement learning trains agents through rewards",
    "Convolutional networks excel at image recognition",
    "Embeddings map text into dense vector spaces",
    "Attention mechanisms let models focus on relevant tokens",
    "Diffusion models generate images from noise",
    "Self-supervised learning unlocks massive unlabeled data",
    # RAG / retrieval
    "Vector search enables retrieval augmented generation",
    "RAG pipelines combine retrieval and generation",
    "Reciprocal rank fusion blends multiple ranked lists",
    "MMR balances relevance and diversity in retrieval",
    "Re-ranking refines initial retrieval results",
    "BM25 is a strong sparse keyword baseline",
    "Hybrid search combines dense and sparse signals",
    "Query expansion broadens recall on rare terms",
    "Approximate nearest neighbor scales to billions of vectors",
    "HNSW graphs deliver fast high-recall search",
    # Quantization / compression
    "INT8 quantization compresses vectors four times",
    "NF4 quantization preserves quality at eight times compression",
    "Binary quantization compresses thirty-two times for speed",
    "Product quantization splits vectors into sub-quantizers",
    "Residual quantization stacks codebooks for accuracy",
    # Programming / systems
    "Mojo combines Python ergonomics with C-level performance",
    "Rust delivers memory safety without garbage collection",
    "SIMD instructions process multiple values per cycle",
    "GPU kernels parallelize matrix multiplications",
    "Mac silicon offers unified memory for large models",
    # Misc
    "Coffee fuels long debugging sessions",
    "Open source software accelerates innovation",
    "Tests catch regressions before they ship",
    "Good documentation is an act of hospitality",
    "Beautiful code is the floor not the ceiling",
]
assert len(CORPUS) == 50, f"expected 50 passages, got {len(CORPUS)}"


VOCAB = sorted(set(
    tok for text in CORPUS for tok in text.lower().split()
))


def banner() -> Panel:
    title = Text()
    title.append("V E C T R O", style="bold magenta")
    title.append("   ", style="")
    title.append(f"v{VECTRO_VERSION}", style="bold cyan")
    sub = Text(
        "Mojo-accelerated embedding compression  ·  4× smaller indexes, same recall",
        style="italic dim",
    )
    body = Group(
        Align.center(title),
        Align.center(sub),
        Text(""),
        Align.center(Text("Demo Day  ·  50 passages  ·  128-dim  ·  no API keys", style="dim")),
    )
    return Panel(body, box=box.DOUBLE, border_style="magenta", padding=(1, 2))


# ---------------------------------------------------------------------------
# Section 1 — compression profiles
# ---------------------------------------------------------------------------

def section_1_compression(embeddings: np.ndarray) -> dict:
    console.print(Rule("[bold]1. Compression profiles[/]", style="cyan"))
    console.print()

    n, d = embeddings.shape
    original_bytes = embeddings.nbytes
    vectro = Vectro()

    table = Table(box=box.SIMPLE_HEAVY, expand=True,
                  border_style="cyan", header_style="bold cyan")
    table.add_column("Profile", style="bold")
    table.add_column("Algorithm", style="dim")
    table.add_column("Bytes", justify="right")
    table.add_column("Saved", justify="right", style="green")
    table.add_column("Ratio", justify="right", style="bold magenta")
    table.add_column("MSE (recon)", justify="right", style="yellow")
    table.add_column("Mem bar", justify="left")

    profiles = ["fast", "balanced", "quality"]
    algo = {
        "fast": "INT8 symmetric per-row",
        "balanced": "INT8 per-row + dtype tuning",
        "quality": "INT8 with optimised scale",
    }

    results = {}
    for prof in profiles:
        r = vectro.compress(embeddings, profile=prof)
        recon = r.reconstruct_batch()
        mse = float(np.mean((recon - embeddings) ** 2))
        saved = original_bytes - r.total_compressed_bytes
        bar_filled = int(round(20 * (1.0 - r.total_compressed_bytes / original_bytes)))
        bar = "█" * bar_filled + "░" * (20 - bar_filled)

        results[prof] = {
            "result": r,
            "recon": recon,
            "mse": mse,
            "bytes": r.total_compressed_bytes,
            "ratio": r.compression_ratio,
            "saved": saved,
        }

        table.add_row(
            prof,
            algo[prof],
            f"{r.total_compressed_bytes:,}",
            f"{saved:,}",
            f"{r.compression_ratio:.2f}×",
            f"{mse:.2e}",
            f"[green]{bar}[/]",
        )

    console.print(table)
    console.print(
        f"[dim]Original float32 size: [bold]{original_bytes:,}[/] bytes "
        f"({n} × {d} × 4)[/]"
    )
    console.print()
    return results


# ---------------------------------------------------------------------------
# Section 2 — k-NN search
# ---------------------------------------------------------------------------

def section_2_knn(rm: VectroDSPyRetriever, query: str) -> None:
    console.print(Rule("[bold]2. k-NN search (cosine)[/]", style="cyan"))
    console.print()
    console.print(f"  query → [italic green]{query!r}[/]")
    console.print()

    out = rm.forward(query, k=5)

    table = Table(box=box.SIMPLE_HEAVY, expand=True,
                  border_style="green", header_style="bold green")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Bar", justify="left")
    table.add_column("Passage")

    max_score = max(out.scores) if out.scores else 1.0
    for rank, (passage, score) in enumerate(zip(out.passages, out.scores), 1):
        norm = max(0.0, min(1.0, score / max_score)) if max_score > 0 else 0.0
        bar_w = int(round(20 * norm))
        bar = "█" * bar_w + "░" * (20 - bar_w)
        table.add_row(str(rank), f"{score:.4f}", f"[magenta]{bar}[/]", passage)

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Section 3 — MMR diversity
# ---------------------------------------------------------------------------

def section_3_mmr(rm: VectroDSPyRetriever, query: str) -> None:
    console.print(Rule("[bold]3. MMR — diversity vs relevance[/]", style="cyan"))
    console.print()
    console.print(f"  query → [italic green]{query!r}[/]   k=4   fetch_k=12")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, expand=True,
                  border_style="yellow", header_style="bold yellow")
    table.add_column("λ", justify="right", style="bold")
    table.add_column("Mode", style="dim")
    table.add_column("Selected passages")

    modes = [
        (1.0, "pure relevance"),
        (0.5, "balanced"),
        (0.0, "pure diversity"),
    ]
    for lam, label in modes:
        out = rm.forward_mmr(query, k=4, fetch_k=12, lambda_mult=lam)
        passages = "\n".join(f"  • {p}" for p in out.passages)
        table.add_row(f"{lam:.1f}", label, passages)

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Section 4 — LangChain + LlamaIndex stores
# ---------------------------------------------------------------------------

def section_4_lc_li(embed_provider) -> None:
    console.print(Rule("[bold]4. LangChain + LlamaIndex stores[/]", style="cyan"))
    console.print()

    lc_store = LangChainVectorStore(
        embedding=embed_provider, compression_profile="balanced"
    )
    seed_texts = CORPUS[:6]
    lc_store.add_texts(seed_texts, metadatas=[{"src": "demo"}] * len(seed_texts))

    li_store = LlamaIndexVectorStore(compression_profile="quality")
    # Seed the LlamaIndex store with raw embeddings (LlamaIndex protocol)
    li_embs = np.stack(
        [np.asarray(embed_provider(t), dtype=np.float32) for t in seed_texts],
        axis=0,
    )

    # The LlamaIndex protocol uses a NodeWithEmbedding-style add().
    # Construct via the internal _rebuild path so the demo works without
    # pulling in real LlamaIndex dataclasses.
    with li_store._lock:
        for i, (text, emb) in enumerate(zip(seed_texts, li_embs)):
            node_id = f"node-{i}"
            li_store._node_store[node_id] = (text, {"src": "demo"})
            li_store._node_ids.append(node_id)
        li_store._n_dims = li_embs.shape[1]
        li_store._compressed = li_store._vectro.compress(
            li_embs, profile=li_store._profile
        )

    table = Table(box=box.SIMPLE_HEAVY, expand=True,
                  border_style="blue", header_style="bold blue")
    table.add_column("Adapter", style="bold")
    table.add_column("Profile", style="cyan")
    table.add_column("Items", justify="right")
    table.add_column("Compression", justify="right", style="magenta")
    table.add_column("Sample passage", style="dim")

    table.add_row(
        "LangChainVectorStore",
        lc_store._profile,
        str(len(lc_store._ids)),
        f"{lc_store._compressed.compression_ratio:.2f}×",
        seed_texts[0],
    )
    table.add_row(
        "LlamaIndexVectorStore",
        li_store._profile,
        str(len(li_store._node_ids)),
        f"{li_store._compressed.compression_ratio:.2f}×",
        seed_texts[0],
    )
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Section 5 — VectroDSPyRetriever (already constructed in main; here we
# print its summary card for visual continuity).
# ---------------------------------------------------------------------------

def section_5_dspy(rm: VectroDSPyRetriever) -> None:
    console.print(Rule("[bold]5. VectroDSPyRetriever[/]", style="cyan"))
    console.print()

    stats = rm.compression_stats
    info = Table.grid(padding=(0, 1))
    info.add_column(justify="right", style="dim")
    info.add_column()
    info.add_row("passages", f"[bold]{stats['n_passages']}[/]")
    info.add_row("dimensions", f"[bold]{stats['dimensions']}[/]")
    info.add_row("profile", f"[bold cyan]{stats['compression_profile']}[/]")
    info.add_row("original",   f"{stats['original_mb']:.3f} MB")
    info.add_row("compressed", f"[bold magenta]{stats['compressed_mb']:.3f} MB[/]")
    info.add_row("saved",      f"[bold green]{stats['memory_saved_mb']:.3f} MB[/]")
    info.add_row("ratio",      f"[bold]{stats['compression_ratio']:.2f}×[/]")
    info.add_row("k (default)", f"[bold]{rm.k}[/]")

    console.print(Panel(info, title="VectroDSPyRetriever",
                        border_style="magenta", box=box.ROUNDED))
    console.print()


# ---------------------------------------------------------------------------
# Section 6 — OpenAIEmbeddings + cache stats
# ---------------------------------------------------------------------------

def section_6_openai(provider: OpenAIEmbeddings, client: _OAIClient) -> None:
    console.print(Rule("[bold]6. OpenAIEmbeddings (mock client) + cache[/]", style="cyan"))
    console.print()

    # Hit the cache hard so stats become interesting:
    #   - a brand-new query (miss)
    #   - same query repeated (hit)
    #   - mixed batch (partial hit)
    fresh_query = "fresh-tokens-nobody-said-before " + str(os.getpid())
    provider(fresh_query)
    provider(fresh_query)                       # hit
    provider([fresh_query, "another novel sentence for the demo"])  # 1 hit + 1 miss

    stats = provider.cache_stats()

    info = Table(box=box.SIMPLE_HEAVY, expand=True,
                 border_style="green", header_style="bold green")
    info.add_column("Metric", style="bold")
    info.add_column("Value", justify="right")

    info.add_row("model",           f"[cyan]{provider.model}[/]")
    info.add_row("batch_size",      str(provider.batch_size))
    info.add_row("cache_dir",       str(provider.cache_dir))
    info.add_row("dimension",       f"{provider.dimension}")
    info.add_row("API calls (mock)", f"[bold]{client.embeddings.calls}[/]")
    info.add_row("cache hits",       f"[bold green]{stats['hits']}[/]")
    info.add_row("cache misses",     f"[yellow]{stats['misses']}[/]")
    info.add_row("rows in SQLite",   f"[bold magenta]{stats['size']}[/]")

    console.print(info)

    if client._calls_log:
        recent = Table(box=box.MINIMAL, border_style="dim")
        recent.add_column("#", style="dim")
        recent.add_column("model", style="cyan")
        recent.add_column("inputs", justify="right")
        for i, call in enumerate(client._calls_log[-6:], 1):
            recent.add_row(str(i), call["model"], str(call["n_inputs"]))
        console.print(Panel(recent, title="last 6 mock-API calls",
                            border_style="dim", padding=(0, 1)))

    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.print()
    console.print(banner())
    console.print()

    # ----- shared state: mock client + cache + provider --------------------
    cache_dir = tempfile.mkdtemp(prefix="vectro-demo-cache-")
    mock_client = _OAIClient(vocabulary=VOCAB, dim=128)
    provider = OpenAIEmbeddings(
        model="text-embedding-3-small-mock",
        client=mock_client,
        batch_size=16,
        cache_dir=cache_dir,
        normalize=True,
    )

    # Embed the corpus once (drives the SQLite cache to ~50 rows)
    with Progress(
        TextColumn("[bold green]embedding corpus"),
        BarColumn(complete_style="green"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("embed", total=len(CORPUS))
        embeddings_list = []
        for text in CORPUS:
            embeddings_list.append(np.asarray(provider(text), dtype=np.float32))
            progress.advance(task)
    embeddings = np.stack(embeddings_list, axis=0)

    # ----- 1. compression profiles ----------------------------------------
    section_1_compression(embeddings)

    # ----- build a retriever once for sections 2, 3, 5 --------------------
    rm = VectroDSPyRetriever(
        embed_fn=provider, k=5, compression_profile="balanced",
    )
    rm.add_texts(CORPUS, embeddings=embeddings,
                 metadatas=[{"id": i} for i in range(len(CORPUS))])

    # ----- 2. k-NN ---------------------------------------------------------
    section_2_knn(rm, "What is the capital of France?")

    # ----- 3. MMR ----------------------------------------------------------
    section_3_mmr(rm, "machine learning models")

    # ----- 4. LangChain + LlamaIndex --------------------------------------
    section_4_lc_li(provider)

    # ----- 5. VectroDSPyRetriever summary ----------------------------------
    section_5_dspy(rm)

    # ----- 6. OpenAIEmbeddings + cache ------------------------------------
    section_6_openai(provider, mock_client)

    # ----- footer ----------------------------------------------------------
    foot = Padding(
        Align.center(Text(
            "Build, ship, repeat.    ቆንጆ · 根性 · 康宙",
            style="bold magenta",
        )),
        (0, 0),
    )
    console.print(Panel(foot, box=box.DOUBLE, border_style="magenta"))
    console.print()


if __name__ == "__main__":
    main()
