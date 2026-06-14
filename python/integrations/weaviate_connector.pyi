import numpy as np
from .vector_db import (
    StoredVectorBatch as StoredVectorBatch,
    VectorDBConnector as VectorDBConnector,
)
from _typeshed import Incomplete
from typing import Any, Sequence

def _make_data_object(uuid: str, properties: dict[str, Any]) -> Any: ...

class WeaviateConnector(VectorDBConnector):
    collection_name: Incomplete
    _host: Incomplete
    _port: Incomplete
    _client: Incomplete
    _collection: Incomplete
    def __init__(
        self,
        collection_name: str,
        client: Any | None = None,
        host: str = "localhost",
        port: int = 8080,
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
