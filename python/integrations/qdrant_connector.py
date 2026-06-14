"""Qdrant connector for storing and retrieving Vectro-compressed vectors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import importlib

import numpy as np

from .vector_db import StoredVectorBatch, VectorDBConnector


class QdrantConnector(VectorDBConnector):
    """Connector that stores compressed vectors in Qdrant payloads.

    This connector is intentionally payload-centric:
    - The dense vector field is set to a lightweight placeholder.
    - Compressed payload carries quantized bytes + scales + metadata.
    """

    def __init__(self, collection_name: str, client: Optional[Any] = None):
        self.collection_name = collection_name
        self.client = client

        if self.client is None:
            try:
                qdrant_mod = importlib.import_module("qdrant_client")
            except ImportError as exc:
                raise RuntimeError(
                    "qdrant-client is required for QdrantConnector. Install with: pip install qdrant-client"
                ) from exc
            self.client = qdrant_mod.QdrantClient(":memory:")

    def upsert_compressed(
        self,
        ids: Sequence[str],
        quantized: np.ndarray,
        scales: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(ids) != len(quantized) or len(ids) != len(scales):
            raise ValueError("ids, quantized rows, and scales rows must have matching lengths")

        payload_meta = metadata or {}
        points: List[Dict[str, Any]] = []

        for idx, vector_id in enumerate(ids):
            q_row = np.asarray(quantized[idx])
            s_row = np.asarray(scales[idx], dtype=np.float32)

            payload = {
                "vectro_quantized": q_row.tolist(),
                "vectro_scales": s_row.tolist(),
                "vectro_vector_dim": int(q_row.shape[0] * (2 if q_row.dtype == np.uint8 else 1)),
                "vectro_quantized_dtype": str(q_row.dtype),
                "vectro_precision_mode": "int4" if q_row.dtype == np.uint8 else "int8",
                "vectro_metadata": payload_meta,
            }

            # Keep a minimal vector so collection schemas expecting vectors remain satisfied.
            points.append({"id": vector_id, "vector": [0.0], "payload": payload})

        self.client.upsert(collection_name=self.collection_name, points=points)

    def fetch_compressed(self, ids: Sequence[str]) -> StoredVectorBatch:
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=list(ids),
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            raise KeyError("No vectors found for the requested ids")

        out_ids: List[str] = []
        quantized_rows: List[np.ndarray] = []
        scales_rows: List[np.ndarray] = []
        vector_dim = 0
        metadata: Dict[str, Any] = {}

        for record in records:
            payload = record.payload or {}
            precision_mode = payload.get("vectro_precision_mode", "int8")
            q_arr = np.asarray(
                payload.get("vectro_quantized", []),
                dtype=np.uint8 if precision_mode == "int4" else np.int8,
            )
            s_arr = np.asarray(payload.get("vectro_scales", []), dtype=np.float32)

            out_ids.append(str(record.id))
            quantized_rows.append(q_arr)
            scales_rows.append(s_arr)
            vector_dim = int(payload.get("vectro_vector_dim", 0))
            metadata.update(payload.get("vectro_metadata", {}))

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
        self.client.delete(
            collection_name=self.collection_name, points_selector={"points": list(ids)}
        )
        return len(ids)
