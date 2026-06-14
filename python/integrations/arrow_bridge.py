"""Apache Arrow and Parquet integration for Vectro.

Serialises and deserialises compressed vector batches to/from Arrow Tables
and Parquet files, enabling interoperability with data-lake pipelines,
PyArrow-based ML frameworks, and columnar storage systems.

Requirements
------------
* ``pyarrow>=12.0`` for Arrow/Parquet support

Install with::

    pip install "vectro[data]"
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from ..interface import QuantizationResult
from ..batch_api import BatchQuantizationResult

if TYPE_CHECKING:  # pragma: no cover
    import pyarrow as pa


def _pa() -> Any:
    """Lazy-load pyarrow; raises a clear RuntimeError when absent."""
    try:
        return importlib.import_module("pyarrow")
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow>=12.0 is required for Arrow/Parquet support. "
            "Install with: pip install 'pyarrow>=12.0'"
        ) from exc


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_VECTRO_METADATA_KEY = b"vectro"


def _build_schema(vector_dim: int, precision_mode: str = "int8") -> Any:
    """Return a pyarrow Schema for a compressed vector batch.

    Columns
    -------
    id            : string   — optional user-supplied identifier
    quantized     : binary   — raw bytes of the quantized int8/uint8 row
    scales        : binary   — raw bytes of the float32 scale row
    vector_dim    : int32
    precision_mode: string
    """
    pa = _pa()
    schema = pa.schema(
        [
            pa.field("id", pa.string(), nullable=True),
            pa.field("quantized", pa.binary()),
            pa.field("scales", pa.binary()),
            pa.field("vector_dim", pa.int32()),
            pa.field("precision_mode", pa.string()),
        ],
        metadata={
            _VECTRO_METADATA_KEY: f"vectro_npz_v2|dim={vector_dim}|prec={precision_mode}".encode()
        },
    )
    return schema


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def result_to_table(
    result: Union[QuantizationResult, BatchQuantizationResult],
    ids: Optional[list] = None,
) -> "pa.Table":
    """Convert a Vectro quantization result to an Arrow Table.

    Each row in the table represents one compressed vector.

    Args:
        result: Output of :func:`~vectro.quantize_embeddings` or
            :class:`~vectro.BatchQuantizationResult`.
        ids: Optional list of string IDs (one per vector).  When *None*,
            the ``id`` column is filled with ``null``.

    Returns:
        ``pyarrow.Table`` with columns: ``id``, ``quantized``, ``scales``,
        ``vector_dim``, ``precision_mode``.
    """
    pa = _pa()

    # Normalise to arrays -------------------------------------------------
    if isinstance(result, BatchQuantizationResult):
        q_arr = np.vstack(result.quantized_vectors)
        s_arr = np.asarray(result.scales, dtype=np.float32)
        vector_dim = result.vector_dim
        precision_mode = result.precision_mode
    else:
        q_arr = np.asarray(
            result.quantized, dtype=np.int8 if result.precision_mode == "int8" else np.uint8
        )
        s_arr = np.asarray(result.scales, dtype=np.float32)
        vector_dim = result.dims
        precision_mode = result.precision_mode if hasattr(result, "precision_mode") else "int8"

    n = q_arr.shape[0]
    if s_arr.ndim == 1:
        s_arr = s_arr.reshape(n, -1)
    if q_arr.ndim == 1:
        q_arr = q_arr.reshape(1, -1)

    # Build columns -------------------------------------------------------
    q_col = pa.array([row.tobytes() for row in q_arr], type=pa.binary())
    s_col = pa.array([row.tobytes() for row in s_arr], type=pa.binary())
    id_col = pa.array(
        [str(ids[i]) if ids is not None else None for i in range(n)],
        type=pa.string(),
    )
    dim_col = pa.array([vector_dim] * n, type=pa.int32())
    prec_col = pa.array([precision_mode] * n, type=pa.string())

    schema = _build_schema(vector_dim, precision_mode)
    return pa.table(
        {
            "id": id_col,
            "quantized": q_col,
            "scales": s_col,
            "vector_dim": dim_col,
            "precision_mode": prec_col,
        },
        schema=schema,
    )


def table_to_result(table: "pa.Table") -> BatchQuantizationResult:
    """Reconstruct a :class:`~vectro.BatchQuantizationResult` from an Arrow Table.

    Args:
        table: Arrow Table previously produced by :func:`result_to_table`.

    Returns:
        :class:`~vectro.BatchQuantizationResult` ready for
        :meth:`~vectro.BatchQuantizationResult.reconstruct_batch`.
    """
    n = len(table)
    precision_mode = table.column("precision_mode")[0].as_py() or "int8"
    vector_dim = int(table.column("vector_dim")[0].as_py())

    q_dtype = np.uint8 if precision_mode == "int4" else np.int8

    q_rows = [np.frombuffer(table.column("quantized")[i].as_py(), dtype=q_dtype) for i in range(n)]
    s_rows = [np.frombuffer(table.column("scales")[i].as_py(), dtype=np.float32) for i in range(n)]

    q_arr = np.vstack(q_rows)
    s_arr = np.asarray(s_rows, dtype=np.float32)
    if s_arr.ndim == 2 and s_arr.shape[1] == 1:
        s_arr = s_arr.ravel()

    original_bytes = n * vector_dim * 4
    compressed_bytes = q_arr.nbytes + s_arr.nbytes
    compression_ratio = original_bytes / max(compressed_bytes, 1)

    return BatchQuantizationResult(
        quantized_vectors=list(q_arr),
        scales=s_arr,
        batch_size=n,
        vector_dim=vector_dim,
        compression_ratio=compression_ratio,
        total_original_bytes=original_bytes,
        total_compressed_bytes=compressed_bytes,
        precision_mode=precision_mode,
    )


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def write_parquet(
    result: Union[QuantizationResult, BatchQuantizationResult],
    path: Union[str, Path],
    ids: Optional[list] = None,
    compression: str = "snappy",
) -> None:
    """Serialise compressed vectors to a Parquet file.

    Args:
        result: Quantization result to persist.
        path: Destination file path (will be created or overwritten).
        ids: Optional list of per-vector string IDs.
        compression: Parquet column compression codec
            (``"snappy"``, ``"gzip"``, ``"zstd"``, ``"none"``).
    """
    pq = importlib.import_module("pyarrow.parquet")
    table = result_to_table(result, ids=ids)
    pq.write_table(table, str(path), compression=compression)


def read_parquet(path: Union[str, Path]) -> BatchQuantizationResult:
    """Deserialise compressed vectors from a Parquet file.

    Args:
        path: Path to a Parquet file written by :func:`write_parquet`.

    Returns:
        :class:`~vectro.BatchQuantizationResult`.
    """
    pq = importlib.import_module("pyarrow.parquet")
    table = pq.read_table(str(path))
    return table_to_result(table)


# ---------------------------------------------------------------------------
# In-memory byte buffer helpers (stream-friendly)
# ---------------------------------------------------------------------------


def to_arrow_bytes(
    result: Union[QuantizationResult, BatchQuantizationResult],
    ids: Optional[list] = None,
) -> bytes:
    """Serialise to the Arrow IPC (stream format) wire encoding.

    Useful for sending compressed vectors over a network stream without
    hitting the filesystem.

    Args:
        result: Quantization result.
        ids: Optional per-vector string IDs.

    Returns:
        Raw bytes in Arrow IPC stream format.
    """
    pa = _pa()
    table = result_to_table(result, ids=ids)
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return bytes(sink.getvalue())


def from_arrow_bytes(data: bytes) -> BatchQuantizationResult:
    """Deserialise from Arrow IPC stream bytes.

    Args:
        data: Raw bytes as produced by :func:`to_arrow_bytes`.

    Returns:
        :class:`~vectro.BatchQuantizationResult`.
    """
    pa = _pa()
    reader = pa.ipc.open_stream(pa.py_buffer(data))
    table = reader.read_all()
    return table_to_result(table)
