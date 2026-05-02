from typing import Any, List, Optional

import numpy as np

from .base import BaseEmbeddingProvider


class VoyageEmbeddings(BaseEmbeddingProvider):
    provider_name: str
    input_type: str

    def __init__(
        self,
        model: str = ...,
        api_key: Optional[str] = ...,
        client: Optional[Any] = ...,
        input_type: str = ...,
        batch_size: int = ...,
        cache_dir: Optional[str] = ...,
        normalize: bool = ...,
        dimension: Optional[int] = ...,
    ) -> None: ...

    def _ensure_client(self) -> Any: ...
    def _embed_batch(self, texts: List[str]) -> np.ndarray: ...
    def _embed_batch_with_type(self, texts: List[str], input_type: str) -> np.ndarray: ...
    def embed_query(self, text: str) -> List[float]: ...
