import pyarrow as pa
from ..batch_api import BatchQuantizationResult as BatchQuantizationResult
from ..interface import (
    QuantizationResult as QuantizationResult,
    reconstruct_embeddings as reconstruct_embeddings,
)
from pathlib import Path
from typing import Any

def _pa() -> Any: ...

_VECTRO_METADATA_KEY: bytes

def _build_schema(vector_dim: int, precision_mode: str = "int8") -> Any: ...
def result_to_table(
    result: QuantizationResult | BatchQuantizationResult, ids: list | None = None
) -> pa.Table: ...
def table_to_result(table: pa.Table) -> BatchQuantizationResult: ...
def write_parquet(
    result: QuantizationResult | BatchQuantizationResult,
    path: str | Path,
    ids: list | None = None,
    compression: str = "snappy",
) -> None: ...
def read_parquet(path: str | Path) -> BatchQuantizationResult: ...
def to_arrow_bytes(
    result: QuantizationResult | BatchQuantizationResult, ids: list | None = None
) -> bytes: ...
def from_arrow_bytes(data: bytes) -> BatchQuantizationResult: ...
