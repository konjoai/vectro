"""Type stubs for mmr."""

from __future__ import annotations
import numpy as np

def cosine_scores(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray: ...
def mmr_select(
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    k: int,
    fetch_k: int,
    lambda_mult: float = ...,
) -> np.ndarray: ...
