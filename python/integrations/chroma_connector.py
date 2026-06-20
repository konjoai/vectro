"""Chroma connector for storing and retrieving Vectro-compressed vectors."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional, Sequence
import importlib

import numpy as np

from .vector_db import StoredVectorBatch, VectorDBConnector

# Chroma metadata values must be primitives (str, int, float, bool).
# Quantized bytes are base64-encoded and scales are JSON-serialised.
_KEY_QUANTIZED = "vectro_quantized"
_KEY_SCALES = "vectro_scales"
_KEY_VECTOR_DIM = "vectro_vector_dim"
_KEY_DTYPE = "vectro_quantized_dtype"
_KEY_PRECISION = "vectro_precision_mode"
_KEY_META_PREFIX = "vectro_meta__"

# Placeholder embedding stored alongside every entry so that Chroma's
# schema is satisfied without needing real float32 embeddings.
_PLACEHOLDER_EMBEDDING = [0.0]


class ChromaConnector(VectorDBConnector):
    """Connector that stores compressed vectors as Chroma collection metadata.

    Compressed payloads (quantized bytes + scales + metadata) are serialised
    as Chroma metadata primitives.  A single 1-D placeholder embedding is
    stored so the collection schema is valid without shipping float32 vectors.

    Args:
        collection_name: Name of the Chroma collection to use.
        client: Optional pre-configured ``chromadb.ClientAPI`` instance.  If
            ``None`` an ephemeral (in-memory) client is created automatically.
    """

    def __init__(self, collection_name: str, client: Optional[Any] = None):
        self.collection_name = collection_name

        if client is None:
            try:
                chroma_mod = importlib.import_module("chromadb")
            except ImportError as exc:
                raise RuntimeError(
                    "chromadb is required for ChromaConnector. Install with: pip install chromadb"
                ) from exc
            client = chroma_mod.EphemeralClient()

        # Acquire / create collection with no automatic embedding function so
        # callers control what goes into the vector field.
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )

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
        """Upsert compressed vector rows into the Chroma collection.

        Args:
            ids: Unique string identifiers for each vector row.
            quantized: Shape ``(n, d)`` int8 or uint8 quantized codes.
            scales: Shape ``(n,)`` float32 per-row scale factors.
            metadata: Optional user-level metadata (str/int/float values only;
                other types are silently skipped).
        """
        if len(ids) != len(quantized) or len(ids) != len(scales):
            raise ValueError("ids, quantized rows, and scales rows must have matching lengths")

        payload_meta = metadata or {}
        all_ids: List[str] = []
        all_embeddings: List[List[float]] = []
        all_metadatas: List[Dict[str, Any]] = []

        for idx, vector_id in enumerate(ids):
            q_row = np.asarray(quantized[idx])
            s_row = np.asarray(scales[idx], dtype=np.float32)
            precision_mode = "int4" if q_row.dtype == np.uint8 else "int8"
            vector_dim = int(q_row.shape[0] * (2 if q_row.dtype == np.uint8 else 1))

            row_meta: Dict[str, Any] = {
                _KEY_QUANTIZED: base64.b64encode(np.ascontiguousarray(q_row).tobytes()).decode(),
                _KEY_SCALES: json.dumps(s_row.tolist()),
                _KEY_VECTOR_DIM: vector_dim,
                _KEY_DTYPE: str(q_row.dtype),
                _KEY_PRECISION: precision_mode,
            }
            # Flatten user metadata as prefixed primitives.
            for k, v in payload_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    row_meta[f"{_KEY_META_PREFIX}{k}"] = v

            all_ids.append(str(vector_id))
            all_embeddings.append(_PLACEHOLDER_EMBEDDING)
            all_metadatas.append(row_meta)

        self._collection.upsert(
            ids=all_ids,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
        )

    def fetch_compressed(self, ids: Sequence[str]) -> StoredVectorBatch:
        """Fetch compressed vectors by IDs from the Chroma collection.

        Args:
            ids: Sequence of string IDs to retrieve.

        Returns:
            A :class:`StoredVectorBatch` containing the reassembled arrays.

        Raises:
            KeyError: When none of the requested IDs are found.
        """
        result = self._collection.get(
            ids=list(ids),
            include=["metadatas"],
        )

        fetched_ids: List[str] = result.get("ids", [])
        metadatas: List[Dict[str, Any]] = result.get("metadatas", []) or []

        if not fetched_ids:
            raise KeyError("No vectors found for the requested ids")

        out_ids: List[str] = []
        quantized_rows: List[np.ndarray] = []
        scales_rows: List[np.ndarray] = []
        vector_dim = 0
        user_metadata: Dict[str, Any] = {}

        for fid, meta in zip(fetched_ids, metadatas):
            precision_mode = meta.get(_KEY_PRECISION, "int8")
            dtype = np.uint8 if precision_mode == "int4" else np.int8

            q_bytes = base64.b64decode(meta[_KEY_QUANTIZED])
            q_arr = np.frombuffer(q_bytes, dtype=dtype).copy()

            s_arr = np.asarray(json.loads(meta[_KEY_SCALES]), dtype=np.float32)

            out_ids.append(str(fid))
            quantized_rows.append(q_arr)
            scales_rows.append(s_arr)
            vector_dim = int(meta.get(_KEY_VECTOR_DIM, 0))

            # Restore user metadata (remove prefix).
            for k, v in meta.items():
                if k.startswith(_KEY_META_PREFIX):
                    user_metadata[k[len(_KEY_META_PREFIX) :]] = v

        quantized_arr = np.vstack(quantized_rows)
        scales_arr = np.asarray(scales_rows, dtype=np.float32)
        if vector_dim == 0:
            vector_dim = int(quantized_arr.shape[1])

        return StoredVectorBatch(
            ids=out_ids,
            quantized=quantized_arr,
            scales=scales_arr,
            vector_dim=vector_dim,
            metadata=user_metadata,
        )

    def delete(self, ids: Sequence[str]) -> int:
        """Delete vectors by IDs from the Chroma collection.

        Args:
            ids: Sequence of string IDs to delete.

        Returns:
            Number of IDs submitted for deletion.
        """
        self._collection.delete(ids=list(ids))
        return len(ids)
