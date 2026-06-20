import numpy as np
from .vector_db import (
    StoredVectorBatch as StoredVectorBatch,
    VectorDBConnector as VectorDBConnector,
)
from _typeshed import Incomplete
from typing import Any, Sequence

_KEY_QUANTIZED: str
_KEY_SCALES: str
_KEY_VECTOR_DIM: str
_KEY_DTYPE: str
_KEY_PRECISION: str
_KEY_META_PREFIX: str
_PLACEHOLDER_EMBEDDING: Incomplete

class ChromaConnector(VectorDBConnector):
    collection_name: Incomplete
    _collection: Incomplete
    def __init__(self, collection_name: str, client: Any | None = None) -> None: ...
    def upsert_compressed(
        self,
        ids: Sequence[str],
        quantized: np.ndarray,
        scales: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
    def fetch_compressed(self, ids: Sequence[str]) -> StoredVectorBatch: ...
    def delete(self, ids: Sequence[str]) -> int: ...
