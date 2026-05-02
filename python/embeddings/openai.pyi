from typing import Any, List, Optional

import numpy as np

from .base import BaseEmbeddingProvider


class OpenAIEmbeddings(BaseEmbeddingProvider):
    provider_name: str

    def __init__(
        self,
        model: str = ...,
        api_key: Optional[str] = ...,
        client: Optional[Any] = ...,
        batch_size: int = ...,
        cache_dir: Optional[str] = ...,
        normalize: bool = ...,
        dimension: Optional[int] = ...,
    ) -> None: ...

    def _ensure_client(self) -> Any: ...
    def _embed_batch(self, texts: List[str]) -> np.ndarray: ...
