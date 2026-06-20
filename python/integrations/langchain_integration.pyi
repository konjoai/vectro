"""Type stubs for langchain_integration."""

from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

def _require_langchain() -> Any: ...
def _mmr_select(
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    k: int,
    fetch_k: int,
    lambda_mult: float = ...,
) -> np.ndarray: ...

class VectroVectorStore:
    def __init__(
        self,
        embedding: Any,
        compression_profile: str = ...,
        model_dir: Optional[str] = ...,
    ) -> None: ...

    # ----- ingestion -----
    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[List[dict]] = ...,
        ids: Optional[List[str]] = ...,
        **kwargs: Any,
    ) -> List[str]: ...
    def add_documents(self, documents: List[Any], **kwargs: Any) -> List[str]: ...
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[dict]] = ...,
        compression_profile: str = ...,
        model_dir: Optional[str] = ...,
        **kwargs: Any,
    ) -> "VectroVectorStore": ...
    @classmethod
    def from_documents(
        cls,
        documents: List[Any],
        embedding: Any,
        compression_profile: str = ...,
        model_dir: Optional[str] = ...,
        **kwargs: Any,
    ) -> "VectroVectorStore": ...
    def delete(self, ids: Optional[List[str]] = ..., **kwargs: Any) -> Optional[bool]: ...

    # ----- search -----
    def similarity_search(
        self,
        query: str,
        k: int = ...,
        filter: Optional[dict] = ...,
        **kwargs: Any,
    ) -> List[Any]: ...
    def similarity_search_with_score(
        self,
        query: str,
        k: int = ...,
        filter: Optional[dict] = ...,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]: ...
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = ...,
        filter: Optional[dict] = ...,
        **kwargs: Any,
    ) -> List[Any]: ...
    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = ...,
        filter: Optional[dict] = ...,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]: ...
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = ...,
        fetch_k: int = ...,
        lambda_mult: float = ...,
        filter: Optional[dict] = ...,
        **kwargs: Any,
    ) -> List[Any]: ...
    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = ...,
        fetch_k: int = ...,
        lambda_mult: float = ...,
        filter: Optional[dict] = ...,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]: ...

    # ----- async -----
    async def aadd_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[List[dict]] = ...,
        ids: Optional[List[str]] = ...,
        **kwargs: Any,
    ) -> List[str]: ...
    async def aadd_documents(self, documents: List[Any], **kwargs: Any) -> List[str]: ...
    async def asimilarity_search(
        self,
        query: str,
        k: int = ...,
        filter: Optional[dict] = ...,
        **kwargs: Any,
    ) -> List[Any]: ...
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = ...,
        filter: Optional[dict] = ...,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]: ...
    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = ...,
        filter: Optional[dict] = ...,
        **kwargs: Any,
    ) -> List[Any]: ...
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = ...,
        fetch_k: int = ...,
        lambda_mult: float = ...,
        filter: Optional[dict] = ...,
        **kwargs: Any,
    ) -> List[Any]: ...

    # ----- persistence -----
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str, embedding: Any) -> "VectroVectorStore": ...

    # ----- Vectro-specific -----
    @property
    def compression_stats(self) -> dict: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
