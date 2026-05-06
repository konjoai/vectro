"""Haystack 2.x DocumentStore backed by Vectro compression.

Provides a drop-in replacement for Haystack's ``InMemoryDocumentStore`` that
transparently compresses stored document embeddings to INT8 or NF4, reducing
in-memory footprint by 4–8×.

Usage::

    from haystack.dataclasses import Document
    from python.integrations.haystack_integration import VectroDocumentStore

    # Create store
    store = VectroDocumentStore(compression_profile="balanced")

    # Write documents (embeddings set by Haystack's DocumentEmbedder)
    docs = [
        Document(content="Paris is the capital of France",
                 embedding=[0.1, 0.2, ...]),
        Document(content="Berlin is cold in winter",
                 embedding=[0.3, 0.4, ...]),
    ]
    store.write_documents(docs)

    # Similarity search (used internally by EmbeddingRetriever)
    results = store.embedding_retrieval(
        query_embedding=[0.15, 0.25, ...], top_k=3
    )

    # Count documents
    print(store.count_documents())   # 2

Memory comparison (768-dim, 1M documents):
    float32 baseline  : 3 072 MB
    INT8  (balanced)  :   784 MB  (3.9× reduction)
    NF4   (quality)   :   416 MB  (7.4× reduction)
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from ..retrieval.mmr import cosine_scores as _cosine_scores_fn, mmr_select as _mmr_select

_HAYSTACK_ERROR = (
    "haystack-ai is required for VectroDocumentStore. "
    "Install with: pip install haystack-ai"
)


def _require_haystack() -> Any:
    try:
        from haystack.dataclasses import Document
        return Document
    except ImportError as exc:
        raise ImportError(_HAYSTACK_ERROR) from exc


class VectroDocumentStore:
    """Haystack 2.x DocumentStore with Vectro embedding compression.

    Implements the Haystack ``DocumentStore`` duck-typing protocol:
    ``write_documents``, ``filter_documents``, ``delete_documents``,
    ``count_documents``, ``get_documents_by_id``, plus
    ``embedding_retrieval`` for ANN search.

    Args:
        compression_profile: ``"fast"``, ``"balanced"`` (default),
            ``"quality"``, or ``"binary"``.
        model_dir: Optional HuggingFace model directory.  When supplied,
            the model-family registry selects the optimal quantization
            method (e.g. GTE → INT8, BGE → NF4).
    """

    # Haystack protocol attribute
    type: str = "VectroDocumentStore"

    def __init__(
        self,
        compression_profile: str = "balanced",
        model_dir: Optional[str] = None,
    ) -> None:
        from ..vectro import Vectro  # relative import — avoids circular at module level

        self._profile = compression_profile
        self._model_dir = model_dir
        self._vectro = Vectro()

        self._lock = threading.Lock()
        # doc_id → Document (without embedding to avoid float32 bloat)
        self._doc_store: Dict[str, Any] = {}
        # Ordered list of doc_ids matching rows in self._compressed
        self._doc_ids: List[str] = []
        self._compressed: Any = None  # BatchQuantizationResult | None
        self._n_dims: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild(self, new_embs: np.ndarray) -> None:
        """Append *new_embs* rows to the compressed store (caller holds lock)."""
        if self._compressed is None:
            self._compressed = self._vectro.compress(
                new_embs, profile=self._profile, model_dir=self._model_dir
            )
            self._n_dims = new_embs.shape[1]
        else:
            existing = self._compressed.reconstruct_batch()
            combined = np.vstack([existing, new_embs])
            self._compressed = self._vectro.compress(
                combined, profile=self._profile, model_dir=self._model_dir
            )

    def _cosine_scores(self, query_emb: np.ndarray) -> np.ndarray:
        return _cosine_scores_fn(query_emb, self._compressed.reconstruct_batch())

    # ------------------------------------------------------------------
    # Haystack DocumentStore protocol
    # ------------------------------------------------------------------

    def count_documents(self) -> int:
        """Return the number of documents in the store."""
        with self._lock:
            return len(self._doc_ids)

    def filter_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Return all documents, optionally filtered by metadata.

        Args:
            filters: Optional dict of ``{meta_field: value}`` pairs.
                Only equality filters are supported.  Pass ``None`` (or
                omit) to retrieve all documents.
        """
        _require_haystack()
        with self._lock:
            docs = list(self._doc_store.values())

        if filters is None:
            return list(docs)

        result = []
        for doc in docs:
            meta = getattr(doc, "meta", {}) or {}
            if all(meta.get(k) == v for k, v in filters.items()):
                result.append(doc)
        return result

    def write_documents(
        self,
        documents: List[Any],
        policy: str = "none",
    ) -> int:
        """Store documents and compress their embeddings.

        Args:
            documents: List of Haystack ``Document`` objects.  Each must
                have its ``.embedding`` field set (done automatically by
                Haystack's ``DocumentEmbedder`` component).
            policy: Duplicate handling — ``"none"`` (skip duplicates),
                ``"overwrite"`` (replace), or ``"fail"`` (raise).

        Returns:
            Number of documents written.
        """
        if not documents:
            return 0

        embeddings = []
        to_add = []
        for doc in documents:
            emb = getattr(doc, "embedding", None)
            if emb is None:
                continue  # skip documents without embeddings
            doc_id = getattr(doc, "id", None) or str(uuid.uuid4())

            with self._lock:
                exists = doc_id in self._doc_store

            if exists:
                if policy == "fail":
                    raise ValueError(
                        f"Document with id={doc_id!r} already exists "
                        "and policy='fail'."
                    )
                if policy == "overwrite":
                    self._remove_by_id(doc_id)
                else:  # "none" — skip
                    continue

            embeddings.append(np.asarray(emb, dtype=np.float32))
            to_add.append((doc_id, doc))

        if not to_add:
            return 0

        emb_matrix = np.stack(embeddings, axis=0)

        with self._lock:
            for doc_id, doc in to_add:
                # Strip embedding from stored doc (saves memory; we hold compressed)
                stripped = _strip_embedding(doc)
                self._doc_store[doc_id] = stripped
                self._doc_ids.append(doc_id)
            self._rebuild(emb_matrix)

        return len(to_add)

    def delete_documents(
        self,
        document_ids: Optional[List[str]] = None,
    ) -> None:
        """Remove documents by id.  Pass ``None`` to clear the entire store."""
        with self._lock:
            if document_ids is None:
                self._doc_ids.clear()
                self._doc_store.clear()
                self._compressed = None
                return

            ids_set = set(document_ids)
            keep_idx = [i for i, did in enumerate(self._doc_ids) if did not in ids_set]

            if not keep_idx:
                self._doc_ids.clear()
                self._doc_store.clear()
                self._compressed = None
                return

            if len(keep_idx) == len(self._doc_ids):
                return  # nothing to remove

            kept_embs = self._compressed.reconstruct_batch()[keep_idx]
            for did in ids_set:
                self._doc_store.pop(did, None)
            self._doc_ids = [self._doc_ids[i] for i in keep_idx]
            self._compressed = self._vectro.compress(
                kept_embs, profile=self._profile, model_dir=self._model_dir
            )

    def get_documents_by_id(self, ids: List[str]) -> List[Any]:
        """Retrieve documents by id."""
        with self._lock:
            return [self._doc_store[did] for did in ids if did in self._doc_store]

    # ------------------------------------------------------------------
    # Embedding retrieval (called by Haystack EmbeddingRetriever)
    # ------------------------------------------------------------------

    def embedding_retrieval(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_embedding: bool = False,
    ) -> List[Any]:
        """Return top-k documents by cosine similarity to *query_embedding*.

        Args:
            query_embedding: Query vector as list or numpy array.
            top_k: Number of results to return.
            filters: Optional metadata equality filters applied after scoring.
            return_embedding: If ``True``, attach the reconstructed float32
                embedding to each result document.

        Returns:
            List of Documents ordered by descending score.  Each document has
            its ``.score`` field set to the cosine similarity.
        """
        _require_haystack()
        q_arr = np.asarray(query_embedding, dtype=np.float32)

        with self._lock:
            if self._compressed is None or not self._doc_ids:
                return []
            scores = self._cosine_scores(q_arr)
            doc_ids = list(self._doc_ids)
            doc_store = dict(self._doc_store)
            mat = self._compressed.reconstruct_batch() if return_embedding else None

        # Apply metadata filters before ranking
        filtered_idx = [
            i for i, did in enumerate(doc_ids)
            if _matches_filters(doc_store.get(did), filters)
        ]
        if not filtered_idx:
            return []

        k = min(top_k, len(filtered_idx))
        filtered_scores = scores[filtered_idx]
        top_local = np.argpartition(filtered_scores, -k)[-k:]
        top_local = top_local[np.argsort(filtered_scores[top_local])[::-1]]

        results = []
        for local_i in top_local:
            global_i = filtered_idx[local_i]
            did = doc_ids[global_i]
            doc = doc_store[did]
            doc = _clone_with_score(doc, float(scores[global_i]))
            if return_embedding and mat is not None:
                doc = _clone_with_embedding(doc, mat[global_i].tolist())
            results.append(doc)

        return results

    # ------------------------------------------------------------------
    # MMR retrieval
    # ------------------------------------------------------------------

    def max_marginal_relevance_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Diversity-promoting retrieval using Maximal Marginal Relevance.

        Selects *k* documents that balance relevance to *query_embedding*
        with dissimilarity to each other.  Useful when a plain top-k returns
        near-duplicate results.

        Args:
            query_embedding: Query vector as list or numpy array.
            k: Number of documents to return.
            fetch_k: Candidate pool size (≥ k).  More candidates → better
                coverage at the cost of extra compute.
            lambda_mult: Trade-off weight.  1.0 = pure relevance (same as
                top-k); 0.0 = pure diversity.  Default 0.5.
            filters: Optional metadata equality filters applied before MMR.

        Returns:
            List of Documents in MMR selection order.
        """
        _require_haystack()
        q_arr = np.asarray(query_embedding, dtype=np.float32)

        with self._lock:
            if self._compressed is None or not self._doc_ids:
                return []
            mat = self._compressed.reconstruct_batch()
            doc_ids = list(self._doc_ids)
            doc_store = dict(self._doc_store)

        filtered_idx = [
            i for i, did in enumerate(doc_ids)
            if _matches_filters(doc_store.get(did), filters)
        ]
        if not filtered_idx:
            return []

        filtered_arr = np.array(filtered_idx)
        filtered_mat = mat[filtered_arr]

        mmr_local = _mmr_select(
            filtered_mat, q_arr, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        mmr_global = filtered_arr[mmr_local]
        return [doc_store[doc_ids[i]] for i in mmr_global]

    async def async_max_marginal_relevance_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Non-blocking variant of :meth:`max_marginal_relevance_search`."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.max_marginal_relevance_search(
                query_embedding, k=k, fetch_k=fetch_k,
                lambda_mult=lambda_mult, filters=filters,
            ),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the store to *path* (a directory).

        Creates two files inside *path*:
        - ``meta.json`` — document metadata, ids, profile settings
        - ``vectors.npy`` — reconstructed float32 embeddings

        The directory is created automatically if it does not exist.
        """
        import json
        import os

        os.makedirs(path, exist_ok=True)

        with self._lock:
            n = len(self._doc_ids)
            if n == 0:
                mat = np.zeros((0, max(self._n_dims, 1)), dtype=np.float32)
            else:
                mat = self._compressed.reconstruct_batch()

            doc_ids = list(self._doc_ids)
            docs_serial = {
                did: _doc_to_dict(self._doc_store[did]) for did in doc_ids
            }

        np.save(os.path.join(path, "vectors.npy"), mat)
        meta = {
            "version": 1,
            "store_type": "haystack",
            "profile": self._profile,
            "model_dir": self._model_dir,
            "n_dims": self._n_dims,
            "doc_ids": doc_ids,
            "documents": docs_serial,
        }
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump(meta, fh)

    @classmethod
    def load(cls, path: str) -> "VectroDocumentStore":
        """Load a previously saved store from *path*.

        Returns:
            A new ``VectroDocumentStore`` with all documents and compressed
            embeddings restored.
        """
        import json
        import os

        with open(os.path.join(path, "meta.json")) as fh:
            meta = json.load(fh)

        if meta.get("store_type") != "haystack":
            raise ValueError(
                f"meta.json store_type={meta.get('store_type')!r} is not 'haystack'."
            )

        store = cls(
            compression_profile=meta["profile"],
            model_dir=meta.get("model_dir"),
        )
        mat = np.load(os.path.join(path, "vectors.npy"))
        doc_ids = meta["doc_ids"]
        docs_serial = meta["documents"]

        if len(mat) > 0 and len(doc_ids) > 0:
            with store._lock:
                for did in doc_ids:
                    store._doc_store[did] = _doc_from_dict(docs_serial[did])
                    store._doc_ids.append(did)
                store._n_dims = meta["n_dims"]
                store._compressed = store._vectro.compress(
                    mat, profile=meta["profile"], model_dir=meta.get("model_dir")
                )

        return store

    # ------------------------------------------------------------------
    # Async variants
    # ------------------------------------------------------------------

    async def async_embedding_retrieval(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_embedding: bool = False,
    ) -> List[Any]:
        """Non-blocking variant of :meth:`embedding_retrieval`.

        Delegates to a thread-pool executor so this method never blocks the
        asyncio event loop — safe for use in FastAPI / AIOHTTP handlers.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.embedding_retrieval(
                query_embedding, top_k, filters, return_embedding
            ),
        )

    async def async_write_documents(
        self,
        documents: List[Any],
        policy: str = "none",
    ) -> int:
        """Non-blocking variant of :meth:`write_documents`.

        Returns:
            Number of documents successfully written.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.write_documents(documents, policy)
        )

    # ------------------------------------------------------------------
    # Vectro-specific helpers
    # ------------------------------------------------------------------

    @property
    def compression_stats(self) -> dict:
        """Return memory usage and compression ratio statistics."""
        with self._lock:
            n = len(self._doc_ids)
            if n == 0 or self._compressed is None:
                return {"n_documents": 0, "compression_ratio": 1.0}
            d = self._n_dims
            original_mb = n * d * 4 / (1024 ** 2)
            compressed_mb = self._compressed.total_compressed_bytes / (1024 ** 2)
            return {
                "n_documents": n,
                "dimensions": d,
                "compression_profile": self._profile,
                "original_mb": round(original_mb, 3),
                "compressed_mb": round(compressed_mb, 3),
                "compression_ratio": round(self._compressed.compression_ratio, 2),
                "memory_saved_mb": round(original_mb - compressed_mb, 3),
            }

    def _remove_by_id(self, doc_id: str) -> None:
        """Remove a single document by id (acquires lock internally)."""
        self.delete_documents([doc_id])

    def __len__(self) -> int:
        return len(self._doc_ids)

    def __repr__(self) -> str:
        return (
            f"VectroDocumentStore(n={len(self)}, profile={self._profile!r}, "
            f"dims={self._n_dims})"
        )


# ---------------------------------------------------------------------------
# Private helpers for Document manipulation without hard importing haystack
# ---------------------------------------------------------------------------

def _strip_embedding(doc: Any) -> Any:
    """Return a copy of *doc* with the embedding set to None."""
    try:
        return doc.__class__(**{
            **{k: getattr(doc, k) for k in doc.__dataclass_fields__},
            "embedding": None,
        })
    except Exception:
        return doc


def _clone_with_score(doc: Any, score: float) -> Any:
    """Return a copy of *doc* with the score set."""
    try:
        return doc.__class__(**{
            **{k: getattr(doc, k) for k in doc.__dataclass_fields__},
            "score": score,
        })
    except Exception:
        return doc


def _clone_with_embedding(doc: Any, embedding: List[float]) -> Any:
    """Return a copy of *doc* with the embedding set."""
    try:
        return doc.__class__(**{
            **{k: getattr(doc, k) for k in doc.__dataclass_fields__},
            "embedding": embedding,
        })
    except Exception:
        return doc


def _matches_filters(doc: Any, filters: Optional[Dict[str, Any]]) -> bool:
    if filters is None:
        return True
    meta = getattr(doc, "meta", {}) or {}
    return all(meta.get(k) == v for k, v in filters.items())


def _doc_to_dict(doc: Any) -> dict:
    """Serialize a Document to a plain dict for JSON storage."""
    try:
        d = {k: getattr(doc, k) for k in doc.__dataclass_fields__}
        d.pop("embedding", None)  # never persist raw embedding
        return d
    except Exception:
        return {"id": getattr(doc, "id", ""), "content": getattr(doc, "content", "")}


def _doc_from_dict(d: dict) -> Any:
    """Deserialize a Document from a plain dict."""
    try:
        from haystack.dataclasses import Document
        return Document(**{k: v for k, v in d.items() if k != "embedding"})
    except ImportError:
        return type("_Doc", (), d)()
