"""Type stubs for haystack_integration."""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np

def _require_haystack() -> Any: ...

class VectroDocumentStore:
    def __init__(
        self,
        compression_profile: str = ...,
        model_dir: Optional[str] = ...,
    ) -> None: ...

    # ----- Haystack DocumentStore protocol -----
    def count_documents(self) -> int: ...
    def filter_documents(
        self,
        filters: Optional[Dict[str, Any]] = ...,
    ) -> List[Any]: ...
    def write_documents(
        self,
        documents: List[Any],
        policy: str = ...,
    ) -> int: ...
    def delete_documents(
        self,
        document_ids: Optional[List[str]] = ...,
    ) -> None: ...
    def get_documents_by_id(self, ids: List[str]) -> List[Any]: ...
    def embedding_retrieval(
        self,
        query_embedding: List[float],
        top_k: int = ...,
        filters: Optional[Dict[str, Any]] = ...,
        return_embedding: bool = ...,
    ) -> List[Any]: ...

    # ----- MMR retrieval -----
    def max_marginal_relevance_search(
        self,
        query_embedding: List[float],
        k: int = ...,
        fetch_k: int = ...,
        lambda_mult: float = ...,
        filters: Optional[Dict[str, Any]] = ...,
    ) -> List[Any]: ...
    async def async_max_marginal_relevance_search(
        self,
        query_embedding: List[float],
        k: int = ...,
        fetch_k: int = ...,
        lambda_mult: float = ...,
        filters: Optional[Dict[str, Any]] = ...,
    ) -> List[Any]: ...

    # ----- async -----
    async def async_embedding_retrieval(
        self,
        query_embedding: List[float],
        top_k: int = ...,
        filters: Optional[Dict[str, Any]] = ...,
        return_embedding: bool = ...,
    ) -> List[Any]: ...
    async def async_write_documents(
        self,
        documents: List[Any],
        policy: str = ...,
    ) -> int: ...

    # ----- persistence -----
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "VectroDocumentStore": ...

    # ----- Vectro-specific -----
    @property
    def compression_stats(self) -> dict: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
