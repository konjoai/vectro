"""Type stubs for lora_api — LoRA adapter matrix compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

_VALID_LORA_PROFILES: frozenset[str]
_RQ_MIN_ROWS: int

@dataclass
class LoRAResult:
    """Compressed LoRA adapter matrices for a single target module.

    Attributes
    ----------
    profile       : one of "lora-nf4", "lora-int8", "lora-rq"
    rank          : LoRA rank r (A.shape[0] == B.shape[1])
    target_module : name of the target module, e.g. "q_proj"
    A_data        : compressed A matrix payload (profile-dependent keys)
    B_data        : compressed B matrix payload (profile-dependent keys)
    A_shape       : original shape of A, (rank, in_features)
    B_shape       : original shape of B, (out_features, rank)
    cosine_sim_A  : per-row mean cosine similarity of A reconstruction
    cosine_sim_B  : per-row mean cosine similarity of B reconstruction
    """

    profile: str
    rank: int
    target_module: str
    A_data: Dict[str, Any]
    B_data: Dict[str, Any]
    A_shape: Tuple[int, int]
    B_shape: Tuple[int, int]
    cosine_sim_A: float = ...
    cosine_sim_B: float = ...

    def __repr__(self) -> str: ...

def compress_lora(
    A: np.ndarray,
    B: np.ndarray,
    profile: str = "lora-nf4",
    target_module: str = "",
) -> LoRAResult:
    """Compress a LoRA adapter (A, B) matrix pair.

    Parameters
    ----------
    A             : float32 array of shape (rank, in_features)
    B             : float32 array of shape (out_features, rank)
    profile       : "lora-nf4" | "lora-int8" | "lora-rq"
    target_module : human-readable name for the target layer, e.g. "q_proj"

    Returns
    -------
    LoRAResult with compressed payloads and per-matrix reconstruction quality.
    """
    ...

def decompress_lora(result: LoRAResult) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct LoRA (A, B) float32 matrices from a LoRAResult.

    Returns
    -------
    A_recon : float32 array of shape result.A_shape
    B_recon : float32 array of shape result.B_shape
    """
    ...

def compress_lora_adapter(
    adapter: Dict[str, Tuple[np.ndarray, np.ndarray]],
    profile: str = "lora-nf4",
) -> Dict[str, LoRAResult]:
    """Compress a full LoRA adapter (all target modules).

    Parameters
    ----------
    adapter : dict mapping module name -> (A, B) float32 matrix pair
    profile : compression profile, applied uniformly to all modules

    Returns
    -------
    dict mapping module name -> LoRAResult
    """
    ...
