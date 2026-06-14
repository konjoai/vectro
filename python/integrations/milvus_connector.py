"""Milvus connector for storing and retrieving Vectro-compressed vectors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import importlib

import numpy as np

from .vector_db import StoredVectorBatch, VectorDBConnector


class MilvusConnector(VectorDBConnector):
    """Connector that stores compressed vectors in a Milvus collection.

    This connector is payload-centric: the compressed int8/int4 payload is stored
    as Milvus JSON fields alongside the collection schema's primary key. No dense
    float32 embedding is stored, keeping the storage footprint minimal.

    Args:
        collection_name: Name of the Milvus collection to use.
        client: Optional pre-configured ``MilvusClient`` instance. If ``None`` an
            in-memory client is created (useful for unit tests without a running
            Milvus server).
    """

    def __init__(self, collection_name: str, client: Optional[Any] = None):
        self.collection_name = collection_name
        self.client = client

        if self.client is None:
            try:
                milvus_mod = importlib.import_module("pymilvus")
            except ImportError as exc:
                raise RuntimeError(
                    "pymilvus is required for MilvusConnector. Install with: pip install pymilvus"
                ) from exc
            # MilvusClient(":memory:") creates a local, ephemeral Milvus Lite instance.
            self.client = milvus_mod.MilvusClient(":memory:")

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
        """Upsert compressed vector rows into the Milvus collection.

        Args:
            ids: Unique string identifiers for each vector row.
            quantized: Shape ``(n, d)`` int8 or uint8 quantized codes.
            scales: Shape ``(n,)`` float32 per-row scale factors.
            metadata: Optional user-level metadata dict applied to every row.
        """
        if len(ids) != len(quantized) or len(ids) != len(scales):
            raise ValueError("ids, quantized rows, and scales rows must have matching lengths")

        payload_meta = metadata or {}
        data: List[Dict[str, Any]] = []

        for idx, vector_id in enumerate(ids):
            q_row = np.asarray(quantized[idx])
            s_row = np.asarray(scales[idx], dtype=np.float32)

            row: Dict[str, Any] = {
                "id": vector_id,
                "vectro_quantized": q_row.tolist(),
                "vectro_scales": s_row.tolist(),
                "vectro_vector_dim": int(q_row.shape[0] * (2 if q_row.dtype == np.uint8 else 1)),
                "vectro_quantized_dtype": str(q_row.dtype),
                "vectro_precision_mode": "int4" if q_row.dtype == np.uint8 else "int8",
                "vectro_metadata": payload_meta,
            }
            data.append(row)

        self.client.upsert(collection_name=self.collection_name, data=data)

    def fetch_compressed(self, ids: Sequence[str]) -> StoredVectorBatch:
        """Fetch compressed vectors by IDs from the Milvus collection.

        Args:
            ids: Sequence of string IDs to retrieve.

        Returns:
            A :class:`StoredVectorBatch` containing the reassembled arrays.

        Raises:
            KeyError: When none of the requested IDs are found.
        """
        records = self.client.get(
            collection_name=self.collection_name,
            ids=list(ids),
        )

        if not records:
            raise KeyError("No vectors found for the requested ids")

        out_ids: List[str] = []
        quantized_rows: List[np.ndarray] = []
        scales_rows: List[np.ndarray] = []
        vector_dim = 0
        metadata: Dict[str, Any] = {}

        for record in records:
            precision_mode = record.get("vectro_precision_mode", "int8")
            q_arr = np.asarray(
                record.get("vectro_quantized", []),
                dtype=np.uint8 if precision_mode == "int4" else np.int8,
            )
            s_arr = np.asarray(record.get("vectro_scales", []), dtype=np.float32)

            out_ids.append(str(record.get("id", "")))
            quantized_rows.append(q_arr)
            scales_rows.append(s_arr)
            vector_dim = int(record.get("vectro_vector_dim", 0))
            metadata.update(record.get("vectro_metadata", {}))

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
        """Delete vectors by IDs from the Milvus collection.

        Args:
            ids: Sequence of string IDs to delete.

        Returns:
            Number of IDs submitted for deletion.
        """
        self.client.delete(collection_name=self.collection_name, ids=list(ids))
        return len(ids)
