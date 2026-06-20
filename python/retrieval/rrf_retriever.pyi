"""Type stubs for rrf_retriever."""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = ...,
) -> Dict[str, float]: ...
def rrf_top_k(
    rankings: Sequence[Sequence[str]],
    k: int,
    rrf_k: int = ...,
) -> List[Tuple[str, float]]: ...

class RRFRetriever:
    def __init__(
        self,
        retrievers: List[Callable],
        k: int = ...,
        fetch_k: int = ...,
        rrf_k: int = ...,
    ) -> None: ...
    def retrieve(self, query: str, k: Optional[int] = ...) -> List[Dict[str, Any]]: ...
    async def aretrieve(self, query: str, k: Optional[int] = ...) -> List[Dict[str, Any]]: ...

class LangChainRRFRetriever:
    def __init__(
        self,
        stores: List[Any],
        k: int = ...,
        fetch_k: int = ...,
        rrf_k: int = ...,
    ) -> None: ...
    def get_relevant_documents(self, query: str, **kwargs: Any) -> List[Any]: ...
    async def aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Any]: ...
    def invoke(self, input: Any, config: Any = ..., **kwargs: Any) -> List[Any]: ...
    async def ainvoke(self, input: Any, config: Any = ..., **kwargs: Any) -> List[Any]: ...
