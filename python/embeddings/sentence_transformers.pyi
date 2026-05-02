from typing import Any, List, Optional

import numpy as np

from .base import BaseEmbeddingProvider


class SentenceTransformersEmbeddings(BaseEmbeddingProvider):
    provider_name: str

    def __init__(
        self,
        model: str = ...,
        model_obj: Optional[Any] = ...,
        device: Optional[str] = ...,
        batch_size: int = ...,
        cache_dir: Optional[str] = ...,
        normalize: bool = ...,
        dimension: Optional[int] = ...,
    ) -> None: ...

    def _ensure_model(self) -> Any: ...
    def _embed_batch(self, texts: List[str]) -> np.ndarray: ...
