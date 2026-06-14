"""VectroRetriever — hybrid BM25 + dense vector retrieval interface.

Provides:
- ``RetrievalResult``     — dataclass carrying per-document retrieval scores.
- ``RetrieverProtocol``   — structural protocol for duck-typing any retriever.
- ``VectroRetriever``     — high-level retriever backed by a ``PyEmbeddingDataset``
                              and an Okapi BM25 index, fused with configurable alpha.

Usage example::

    from vectro_py import EmbeddingDataset
    from python.retriever import VectroRetriever

    dataset = EmbeddingDataset.load("corpus.stream1")
    texts   = [doc.text for doc in corpus]       # your text corpus
    ids     = [doc.id   for doc in corpus]

    # Build a RetrieverProtocol-compliant retriever (alpha=0.7 → dense-dominant)
    retriever = VectroRetriever(dataset, texts=texts, ids=ids, alpha=0.7)

    results = retriever.retrieve("quantum entanglement", k=10)
    for r in results:
        print(r.id, r.combined_score, r.text[:80])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

try:
    from vectro_py import BM25Index, hybrid_search_py, EmbeddingDataset  # type: ignore

    _BINDINGS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BINDINGS_AVAILABLE = False


@dataclass
class RetrievalResult:
    """A single retrieval result combining dense and BM25 scores.

    Attributes
    ----------
    id : str
        Document identifier.
    dense_score : float
        Raw cosine similarity with the query vector (before normalisation).
    bm25_score : float
        Raw BM25 score (before normalisation).
    combined_score : float
        Final fused score after min-max normalisation and alpha-blending.
        Range: ``[0, 1]``.
    text : str
        Original document text (empty string if not stored).
    """

    id: str
    dense_score: float
    bm25_score: float
    combined_score: float
    text: str = ""

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(id={self.id!r}, combined={self.combined_score:.4f}, "
            f"dense={self.dense_score:.4f}, bm25={self.bm25_score:.4f})"
        )


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Structural protocol for any hybrid/dense retriever.

    Any class that implements ``retrieve(query, k)`` returning a list of
    :class:`RetrievalResult` objects satisfies this protocol.
    """

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Retrieve the top-k results for *query*.

        Parameters
        ----------
        query : str
            Natural language query string.  For hybrid retrieval this text is
            both embedded (dense) and used for BM25 keyword matching.
        k : int
            Maximum number of results to return.

        Returns
        -------
        list[RetrievalResult]
            Results sorted by ``combined_score`` descending.
        """
        ...  # pragma: no cover


class VectroRetriever:
    """Hybrid BM25 + dense cosine retriever backed by Vectro.

    Internally:
    1. Builds an Okapi BM25 index from the supplied *texts*.
    2. At query time, embeds the query with *embed_fn* and calls
       :func:`hybrid_search_py` from the Rust bindings for efficient score
       fusion.

    Parameters
    ----------
    dataset : EmbeddingDataset
        Pre-populated Vectro embedding dataset. Document IDs must align with
        *ids*.
    texts : list[str]
        Corpus texts in the same order as *ids*.
    ids : list[str]
        Document identifiers in the same order as *texts*.
    embed_fn : callable, optional
        Function ``(query: str) -> list[float]`` that produces the query
        embedding.  When *None*, the retriever falls back to BM25-only mode
        (``alpha`` is coerced to 0.0).
    alpha : float
        Score fusion weight: ``1.0`` = pure dense, ``0.0`` = pure BM25.
        Default ``0.7`` (dense-dominant, suitable for semantic tasks).
    """

    def __init__(
        self,
        dataset: "EmbeddingDataset",
        texts: list[str],
        ids: list[str],
        embed_fn=None,
        alpha: float = 0.7,
    ) -> None:
        if not _BINDINGS_AVAILABLE:
            raise ImportError(
                "vectro_py is required.  Build it with `maturin develop` or "
                "`pip install vectro` first."
            )
        if len(texts) != len(ids):
            raise ValueError(
                f"`texts` and `ids` must be the same length, got {len(texts)} and {len(ids)}."
            )

        self._dataset = dataset
        self._bm25 = BM25Index.build(ids, texts)
        self._embed_fn = embed_fn
        self._alpha = float(np.clip(alpha, 0.0, 1.0))
        self._text_map: dict[str, str] = dict(zip(ids, texts))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Return the top-*k* results for *query*.

        If no *embed_fn* was supplied at construction, the alpha is coerced to
        0.0 (BM25-only) and the dense component is skipped.
        """
        if k <= 0:
            return []

        effective_alpha = self._alpha
        if self._embed_fn is None:
            effective_alpha = 0.0
            query_vec: list[float] = []
        else:
            query_vec = list(self._embed_fn(query))

        raw: list[tuple[str, float]] = hybrid_search_py(
            self._dataset,
            self._bm25,
            query_vec,
            query,
            k,
            effective_alpha,
        )

        return [
            RetrievalResult(
                id=doc_id,
                dense_score=0.0,  # raw per-doc breakdown not exposed by Rust binding
                bm25_score=0.0,  # (combined_score is the fused normalised value)
                combined_score=score,
                text=self._text_map.get(doc_id, ""),
            )
            for doc_id, score in raw
        ]

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str,
        texts: list[str],
        ids: list[str],
        embed_fn=None,
        alpha: float = 0.7,
    ) -> "VectroRetriever":
        """Construct a :class:`VectroRetriever` from a Vectro dataset file.

        Convenience factory that calls :meth:`EmbeddingDataset.load` for you,
        removing the need to manage the dataset object manually.

        Parameters
        ----------
        path : str
            Path to a ``.stream1`` or ``.qstream1`` file on disk.
        texts : list[str]
            Corpus texts in the same order as *ids*.
        ids : list[str]
            Document identifiers in the same order as *texts*.
        embed_fn : callable, optional
            ``(query: str) -> list[float]`` producing query embeddings.
        alpha : float
            Score fusion weight (1.0 = pure dense, 0.0 = pure BM25).

        Returns
        -------
        VectroRetriever
        """
        if not _BINDINGS_AVAILABLE:
            raise ImportError("vectro_py is required.  Build with `maturin develop` first.")
        dataset = EmbeddingDataset.load(path)
        return cls(dataset, texts=texts, ids=ids, embed_fn=embed_fn, alpha=alpha)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        text_field: str = "text",
        id_field: str = "id",
        embed_fn=None,
        alpha: float = 0.7,
    ) -> "VectroRetriever":
        """Construct a :class:`VectroRetriever` from a JSONL corpus file.

        Each line must be a JSON object with at least *id_field* and
        *text_field* keys.  Embeddings must already be present in the JSONL
        (any ``"vector"`` field is ignored; use :meth:`from_file` for
        pre-embedded datasets).

        This factory is intended for text-only BM25 retrieval scenarios where
        you supply your own *embed_fn* or want BM25-only mode.

        Parameters
        ----------
        path : str
            Path to a ``.jsonl`` file.
        text_field : str
            Key used for document text (default ``"text"``).
        id_field : str
            Key used for document identifier (default ``"id"``).
        embed_fn : callable, optional
            ``(query: str) -> list[float]`` producing query embeddings.
        alpha : float
            Score fusion weight.

        Returns
        -------
        VectroRetriever
        """
        import json

        texts: list[str] = []
        ids: list[str] = []
        vectors: list[list[float]] = []

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                texts.append(str(obj[text_field]))
                ids.append(str(obj[id_field]))
                if "vector" in obj:
                    vectors.append(list(obj["vector"]))

        if not _BINDINGS_AVAILABLE:
            raise ImportError("vectro_py is required.  Build with `maturin develop` first.")

        dataset = (
            EmbeddingDataset.from_embeddings(ids, vectors) if vectors else EmbeddingDataset.empty()
        )
        return cls(dataset, texts=texts, ids=ids, embed_fn=embed_fn, alpha=alpha)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Current dense/BM25 balance (read-only)."""
        return self._alpha

    @property
    def bm25(self) -> "BM25Index":
        """Underlying BM25 index."""
        return self._bm25

    @property
    def n_docs(self) -> int:
        """Number of indexed documents."""
        return len(self._bm25)

    def __repr__(self) -> str:
        return f"VectroRetriever(n_docs={self.n_docs}, alpha={self._alpha:.2f})"
