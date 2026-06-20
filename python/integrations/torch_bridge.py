"""PyTorch and HuggingFace Transformers integration helpers for Vectro.

This module provides thin wrappers that accept and return ``torch.Tensor``
objects so that Vectro fits naturally into PyTorch-based ML pipelines.

Requirements
------------
* ``torch`` for :func:`compress_tensor` / :func:`reconstruct_tensor`
* ``torch`` and ``transformers`` for :class:`HuggingFaceCompressor`

Install with::

    pip install torch                         # PyTorch only
    pip install transformers                  # HuggingFace Transformers
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from ..interface import QuantizationResult, quantize_embeddings, reconstruct_embeddings

if TYPE_CHECKING:  # pragma: no cover
    import torch


def compress_tensor(
    tensor: "torch.Tensor",
    precision_mode: str = "int8",
    group_size: int = 64,
    backend: str = "auto",
) -> QuantizationResult:
    """Compress a PyTorch tensor of embeddings.

    Args:
        tensor: Float tensor of shape ``(n_vectors, dim)`` or ``(dim,)`` for a
            single vector (auto-promoted to 2-D).
        precision_mode: ``"int8"`` (default) or ``"int4"`` (requires the
            ``squish_quant`` backend).
        group_size: Group size for INT4 grouped quantization.
        backend: Force a specific quantization backend (``None`` = auto).

    Returns:
        :class:`~vectro.interface.QuantizationResult` with the compressed
        representation.
    """
    _ = importlib.import_module("torch")  # presence check — raises ImportError clearly

    arr: np.ndarray = tensor.detach().float().cpu().numpy()  # type: ignore[union-attr]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    return quantize_embeddings(
        arr, backend=backend, precision_mode=precision_mode, group_size=group_size
    )


def reconstruct_tensor(
    result: QuantizationResult,
    backend: str = "auto",
) -> "torch.Tensor":
    """Reconstruct a float32 PyTorch tensor from a :class:`~vectro.interface.QuantizationResult`.

    Args:
        result: Result produced by :func:`compress_tensor`.
        backend: Force a specific reconstruction backend (``None`` = auto).

    Returns:
        Float32 PyTorch tensor of reconstructed embeddings.
    """
    torch = importlib.import_module("torch")
    arr = reconstruct_embeddings(result, backend=backend)
    return torch.from_numpy(arr.astype(np.float32))


class HuggingFaceCompressor:
    """Convenience class that tokenizes, encodes, and compresses text in one call.

    Uses mean-pooling over the last hidden state to produce sentence embeddings,
    then quantizes them with Vectro.

    Example::

        compressor = HuggingFaceCompressor.from_model(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        result = compressor.encode_and_compress(["Hello world", "Vectro rocks"])
        reconstructed = reconstruct_tensor(result)
    """

    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu"):  # type: ignore[name-defined]
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    @classmethod
    def from_model(cls, model_name: str, device: Optional[str] = None) -> "HuggingFaceCompressor":
        """Load model and tokenizer from a HuggingFace Hub identifier.

        Args:
            model_name: HuggingFace model identifier, e.g.
                ``"sentence-transformers/all-MiniLM-L6-v2"``.
            device: ``"cuda"``, ``"mps"``, or ``"cpu"`` (default: auto-detect).

        Raises:
            RuntimeError: If ``transformers`` or ``torch`` is not installed.
        """
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for HuggingFaceCompressor. "
                "Install with: pip install transformers"
            ) from exc

        torch = importlib.import_module("torch")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = transformers.AutoModel.from_pretrained(model_name)
        model = model.to(device).eval()
        return cls(model, tokenizer, device)

    def encode_and_compress(
        self,
        texts: List[str],
        precision_mode: str = "int8",
        max_length: int = 128,
    ) -> QuantizationResult:
        """Tokenize, encode, and compress a list of text strings.

        Args:
            texts: Strings to encode.
            precision_mode: ``"int8"`` (default) or ``"int4"``.
            max_length: Maximum token length.

        Returns:
            :class:`~vectro.interface.QuantizationResult` containing the
            compressed embeddings.
        """
        torch = importlib.import_module("torch")

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean-pool over the token (sequence) dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return compress_tensor(embeddings, precision_mode=precision_mode)
