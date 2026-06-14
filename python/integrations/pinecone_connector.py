"""Pinecone connector for storing and retrieving Vectro-compressed vectors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import importlib

import numpy as np

from .vector_db import StoredVectorBatch, VectorDBConnector

# A single float placeholder is stored in Pinecone's required `values` field.
# Actual vector data is carried in the metadata payload.
_PLACEHOLDER_VALUE = [0.0]


class PineconeConnector(VectorDBConnector):
    """Connector that stores compressed vectors in a Pinecone index.

    This connector is payload-centric: the compressed int8/int4 payload is stored
    as Pinecone metadata fields.  The mandatory ``values`` field receives a
    single-element placeholder ``[0.0]``.

    Args:
        index_name: Name of the Pinecone index (used for display and identification).
        index: Optional pre-configured Pinecone ``Index`` object.  If ``None`` and
            both *api_key* + *host* are provided, a client is constructed via the
            ``pinecone`` package.  Passing a fake index is the recommended approach
            for unit tests to avoid a live Pinecone connection.
        api_key: Pinecone API key (used only when *index* is ``None``).
        host: Index host URL (used only when *index* is ``None``).
    """

    def __init__(
        self,
        index_name: str,
        index: Optional[Any] = None,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self.index_name = index_name
        self._index = index

        if self._index is None:
            try:
                pinecone_mod = importlib.import_module("pinecone")
            except ImportError as exc:
                raise RuntimeError(
                    "pinecone-client is required for PineconeConnector. "
                    "Install with: pip install 'pinecone-client>=3.0'"
                ) from exc
            pc = pinecone_mod.Pinecone(api_key=api_key)
            self._index = pc.Index(index_name, host=host)

    # ------------------------------------------------------------------
    # VectorDBConnector interface
    # ------------------------------------------------------------------

    def upsert_compressed(
        self,
        ids: Sequence[str],
        quantized: np.ndarray,
        scales: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert compressed vector rows into the Pinecone index.

        Args:
            ids: Unique string identifiers for each vector row.
            quantized: Shape ``(n, d)`` int8 or uint8 quantized codes.
            scales: Shape ``(n,)`` float32 per-row scale factors.
            metadata: Optional user-level metadata dict applied to every row.
        """
        if len(ids) != len(quantized) or len(ids) != len(scales):
            raise ValueError("ids, quantized rows, and scales rows must have matching lengths")

        payload_meta = metadata or {}
        vectors: List[Dict[str, Any]] = []

        for idx, vector_id in enumerate(ids):
            q_row = np.asarray(quantized[idx])
            s_row = np.asarray(scales[idx], dtype=np.float32)

            meta: Dict[str, Any] = {
                "vectro_quantized": q_row.tolist(),
                "vectro_scales": s_row.tolist(),
                "vectro_vector_dim": int(q_row.shape[0] * (2 if q_row.dtype == np.uint8 else 1)),
                "vectro_quantized_dtype": str(q_row.dtype),
                "vectro_precision_mode": "int4" if q_row.dtype == np.uint8 else "int8",
                "vectro_metadata": payload_meta,
            }

            vectors.append(
                {
                    "id": str(vector_id),
                    "values": _PLACEHOLDER_VALUE,
                    "metadata": meta,
                }
            )

        self._index.upsert(vectors=vectors)

    def fetch_compressed(self, ids: Sequence[str]) -> StoredVectorBatch:
        """Fetch compressed vectors by IDs from the Pinecone index.

        Args:
            ids: Sequence of string IDs to retrieve.

        Returns:
            A :class:`StoredVectorBatch` containing the reassembled arrays.

        Raises:
            KeyError: When none of the requested IDs are found.
        """
        response = self._index.fetch(ids=list(ids))
        vectors_dict: Dict[str, Any] = response.vectors

        if not vectors_dict:
            raise KeyError("No vectors found for the requested ids")

        out_ids: List[str] = []
        quantized_rows: List[np.ndarray] = []
        scales_rows: List[np.ndarray] = []
        vector_dim = 0
        metadata: Dict[str, Any] = {}

        # Preserve insertion order when possible by iterating requested ids.
        for vector_id in ids:
            record = vectors_dict.get(str(vector_id))
            if record is None:
                continue

            meta = record.metadata if hasattr(record, "metadata") else record.get("metadata", {})

            precision_mode = meta.get("vectro_precision_mode", "int8")
            q_arr = np.asarray(
                meta.get("vectro_quantized", []),
                dtype=np.uint8 if precision_mode == "int4" else np.int8,
            )
            s_arr = np.asarray(meta.get("vectro_scales", []), dtype=np.float32)

            out_ids.append(str(vector_id))
            quantized_rows.append(q_arr)
            scales_rows.append(s_arr)
            vector_dim = int(meta.get("vectro_vector_dim", 0))
            metadata.update(meta.get("vectro_metadata", {}))

        if not out_ids:
            raise KeyError("No vectors found for the requested ids")

        quantized_arr = np.vstack(quantized_rows)
        scales_arr = np.asarray(scales_rows, dtype=np.float32)
        if vector_dim == 0:
            vector_dim = int(quantized_arr.shape[1])

        return StoredVectorBatch(
            ids=out_ids,
            quantized=quantized_arr,
            scales=scales_arr,
            vector_dim=vector_dim,
            metadata=metadata,
        )

    def delete(self, ids: Sequence[str]) -> int:
        """Delete vectors by IDs from the Pinecone index.

        Args:
            ids: Sequence of string IDs to delete.

        Returns:
            Number of IDs submitted for deletion.
        """
        self._index.delete(ids=list(ids))
        return len(ids)
