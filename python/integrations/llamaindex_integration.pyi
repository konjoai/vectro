"""Type stubs for llamaindex_integration."""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np

def _apply_meta_filters(
    node_ids: List[str],
    node_store: Dict[str, tuple],
    filters: Any,
) -> List[int]: ...

class VectroVectorStore:
    stores_text: bool
    is_embedding_query: bool
    flat_metadata: bool

    def __init__(
        self,
        compression_profile: str = ...,
        model_dir: Optional[str] = ...,
    ) -> None: ...

    # ----- LlamaIndex protocol -----
    def add(self, nodes: List[Any], **add_kwargs: Any) -> List[str]: ...
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None: ...
    def query(self, query: Any, **kwargs: Any) -> Any: ...
    def get_nodes(
        self,
        node_ids: Optional[List[str]] = ...,
        filters: Any = ...,
    ) -> List[Any]: ...

    # ----- async -----
    async def async_add(self, nodes: List[Any], **add_kwargs: Any) -> List[str]: ...
    async def aquery(self, query: Any, **kwargs: Any) -> Any: ...

    # ----- persistence -----
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "VectroVectorStore": ...

    # ----- Vectro-specific -----
    @property
    def compression_stats(self) -> dict: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
