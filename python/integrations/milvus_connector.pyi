import numpy as np
from .vector_db import (
    StoredVectorBatch as StoredVectorBatch,
    VectorDBConnector as VectorDBConnector,
)
from _typeshed import Incomplete
from typing import Any, Sequence

class MilvusConnector(VectorDBConnector):
    collection_name: Incomplete
    client: Incomplete
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
