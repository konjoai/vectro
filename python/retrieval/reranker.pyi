"""Type stubs for reranker."""

from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

def _cosine_rerank(
    query_vec: np.ndarray,
    candidates: Sequence[Tuple[str, Any, float]],
    store_mat: np.ndarray,
    id_to_row: Dict[str, int],
    top_k: int,
) -> List[Tuple[str, Any, float]]: ...
def _rrf_rerank(
    query_vec: np.ndarray,
    candidates: Sequence[Tuple[str, Any, float]],
    store_mat: np.ndarray,
    id_to_row: Dict[str, int],
    top_k: int,
    k: int = ...,
) -> List[Tuple[str, Any, float]]: ...

class VectroReranker:
    def __init__(
        self,
        store: Any,
        strategy: str = ...,
        rrf_k: int = ...,
    ) -> None: ...
    def rerank(
        self,
        query_embedding: Any,
        candidates: Sequence[Tuple[str, Any, float]],
        top_k: int = ...,
    ) -> List[Tuple[str, Any, float]]: ...
    async def arerank(
        self,
        query_embedding: Any,
        candidates: Sequence[Tuple[str, Any, float]],
        top_k: int = ...,
    ) -> List[Tuple[str, Any, float]]: ...
    def __repr__(self) -> str: ...

class HaystackReranker:
    def __init__(
        self,
        store: Any,
        top_k: int = ...,
        strategy: str = ...,
        rrf_k: int = ...,
    ) -> None: ...
    def run(
        self,
        query_embedding: Any,
        documents: List[Any],
        top_k: int | None = ...,
    ) -> Dict[str, List[Any]]: ...
    async def async_run(
        self,
        query_embedding: Any,
        documents: List[Any],
        top_k: int | None = ...,
    ) -> Dict[str, List[Any]]: ...
    def __repr__(self) -> str: ...

class LangChainReranker:
    def __init__(
        self,
        store: Any,
        embedding: Any,
        top_k: int = ...,
        strategy: str = ...,
        rrf_k: int = ...,
    ) -> None: ...
    def compress_documents(
        self,
        documents: List[Any],
        query: str,
        callbacks: Any = ...,
    ) -> List[Any]: ...
    async def acompress_documents(
        self,
        documents: List[Any],
        query: str,
        callbacks: Any = ...,
    ) -> List[Any]: ...
    def invoke(self, input: Any, config: Any = ..., **kwargs: Any) -> List[Any]: ...
    async def ainvoke(self, input: Any, config: Any = ..., **kwargs: Any) -> List[Any]: ...
    def __repr__(self) -> str: ...
