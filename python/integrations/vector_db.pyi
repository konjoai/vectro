import abc
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

@dataclass
class StoredVectorBatch:
    ids: list[str]
    quantized: np.ndarray
    scales: np.ndarray
    vector_dim: int
    metadata: dict[str, Any] = field(default_factory=dict)

class VectorDBConnector(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def upsert_compressed(
        self,
        ids: Sequence[str],
        quantized: np.ndarray,
        scales: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
    @abstractmethod
    def fetch_compressed(self, ids: Sequence[str]) -> StoredVectorBatch: ...
    @abstractmethod
    def delete(self, ids: Sequence[str]) -> int: ...

class InMemoryVectorDBConnector(VectorDBConnector):
    _store: dict[str, dict[str, Any]]
    def __init__(self) -> None: ...
    def upsert_compressed(
        self,
        ids: Sequence[str],
        quantized: np.ndarray,
        scales: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
    def fetch_compressed(self, ids: Sequence[str]) -> StoredVectorBatch: ...
    def delete(self, ids: Sequence[str]) -> int: ...
