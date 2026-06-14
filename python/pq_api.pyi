import numpy as np
from dataclasses import dataclass, field as field

@dataclass
class PQCodebook:
    n_subspaces: int
    n_centroids: int
    sub_dim: int
    centroids: np.ndarray
    rotation: np.ndarray | None = ...

def train_pq_codebook(
    training_data: np.ndarray,
    n_subspaces: int = 96,
    n_centroids: int = 256,
    max_iter: int = 25,
    random_state: int = 0,
) -> PQCodebook: ...
def pq_encode(vectors: np.ndarray, codebook: PQCodebook) -> np.ndarray: ...
def pq_decode(codes: np.ndarray, codebook: PQCodebook) -> np.ndarray: ...
def pq_distance_table(query: np.ndarray, codebook: PQCodebook) -> np.ndarray: ...
def pq_search(
    query: np.ndarray, codes: np.ndarray, codebook: PQCodebook, top_k: int = 10
) -> tuple[np.ndarray, np.ndarray]: ...
def pq_compression_ratio(d: int, M: int) -> float: ...
def opq_rotation(
    training_data: np.ndarray, n_subspaces: int, n_iter: int = 10, random_state: int = 0
) -> tuple[np.ndarray, PQCodebook]: ...
