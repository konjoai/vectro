import torch
from ..interface import (
    QuantizationResult as QuantizationResult,
    quantize_embeddings as quantize_embeddings,
    reconstruct_embeddings as reconstruct_embeddings,
)
from _typeshed import Incomplete
from typing import Any

def compress_tensor(
    tensor: torch.Tensor, precision_mode: str = "int8", group_size: int = 64, backend: str = "auto"
) -> QuantizationResult: ...
def reconstruct_tensor(result: QuantizationResult, backend: str = "auto") -> torch.Tensor: ...

class HuggingFaceCompressor:
    _model: Incomplete
    _tokenizer: Incomplete
    _device: Incomplete
    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu") -> None: ...
    @classmethod
    def from_model(cls, model_name: str, device: str | None = None) -> HuggingFaceCompressor: ...
    def encode_and_compress(
        self, texts: list[str], precision_mode: str = "int8", max_length: int = 128
    ) -> QuantizationResult: ...
