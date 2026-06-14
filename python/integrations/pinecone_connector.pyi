import numpy as np
from .vector_db import (
    StoredVectorBatch as StoredVectorBatch,
    VectorDBConnector as VectorDBConnector,
)
from _typeshed import Incomplete
from typing import Any, Sequence

_PLACEHOLDER_VALUE: Incomplete

class PineconeConnector(VectorDBConnector):
    index_name: Incomplete
    _index: Incomplete
    def __init__(
        self,
        index_name: str,
        index: Any | None = None,
        api_key: str | None = None,
        host: str | None = None,
    ) -> None: ...
    def upsert_compressed(
        self,
        ids: Sequence[str],
        quantized: np.ndarray,
        scales: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
    def fetch_compressed(self, ids: Sequence[str]) -> StoredVectorBatch: ...
    def delete(self, ids: Sequence[str]) -> int: ...
