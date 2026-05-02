from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

QueryLike = Union[str, Sequence[str]]
EmbedFn = Callable[[Union[str, List[str]]], Any]


def _make_prediction(passages: List[str], **fields: Any) -> Any: ...


class VectroDSPyRetriever:
    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = ...,
        k: int = ...,
        compression_profile: str = ...,
        model_dir: Optional[str] = ...,
    ) -> None: ...

    @property
    def k(self) -> int: ...
    @k.setter
    def k(self, value: int) -> None: ...

    def add_texts(
        self,
        passages: Sequence[str],
        embeddings: Optional[np.ndarray] = ...,
        metadatas: Optional[Sequence[Dict[str, Any]]] = ...,
    ) -> int: ...

    def clear(self) -> None: ...

    def forward(
        self,
        query_or_queries: QueryLike,
        k: Optional[int] = ...,
        *,
        query_embedding: Optional[np.ndarray] = ...,
        filters: Optional[Dict[str, Any]] = ...,
    ) -> Any: ...

    def __call__(
        self,
        query_or_queries: QueryLike,
        k: Optional[int] = ...,
        *,
        query_embedding: Optional[np.ndarray] = ...,
        filters: Optional[Dict[str, Any]] = ...,
    ) -> Any: ...

    async def aforward(
        self,
        query_or_queries: QueryLike,
        k: Optional[int] = ...,
        *,
        query_embedding: Optional[np.ndarray] = ...,
        filters: Optional[Dict[str, Any]] = ...,
    ) -> Any: ...

    def forward_mmr(
        self,
        query: str,
        k: int = ...,
        fetch_k: int = ...,
        lambda_mult: float = ...,
        *,
        query_embedding: Optional[np.ndarray] = ...,
        filters: Optional[Dict[str, Any]] = ...,
    ) -> Any: ...

    async def aforward_mmr(
        self,
        query: str,
        k: int = ...,
        fetch_k: int = ...,
        lambda_mult: float = ...,
        *,
        query_embedding: Optional[np.ndarray] = ...,
        filters: Optional[Dict[str, Any]] = ...,
    ) -> Any: ...

    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str, embed_fn: Optional[EmbedFn] = ...) -> "VectroDSPyRetriever": ...

    @property
    def compression_stats(self) -> dict: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
