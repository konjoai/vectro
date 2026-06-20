import numpy as np
from .auto_quantize_api import auto_quantize as auto_quantize
from .binary_api import dequantize_binary as dequantize_binary, quantize_binary as quantize_binary
from .interface import (
    quantize_embeddings as quantize_embeddings,
    reconstruct_embeddings as reconstruct_embeddings,
)
from .nf4_api import (
    dequantize_mixed as dequantize_mixed,
    dequantize_nf4 as dequantize_nf4,
    quantize_mixed as quantize_mixed,
    quantize_nf4 as quantize_nf4,
    select_outlier_dims as select_outlier_dims,
)
from .pq_api import (
    PQCodebook as _PQCodebook,
    pq_decode as pq_decode,
    pq_encode as pq_encode,
    train_pq_codebook as train_pq_codebook,
)
from .rq_api import ResidualQuantizer as ResidualQuantizer
from .storage_v3 import load_vqz as load_vqz, save_vqz as save_vqz
from _typeshed import Incomplete
from dataclasses import dataclass, field
from pathlib import Path as Path
from typing import Any, Sequence

_SAVE_VERSION: int

@dataclass
class V3Result:
    profile: str
    n_vectors: int
    dims: int
    data: dict[str, Any] = field(default_factory=dict)

class PQCodebook:
    _cb: Incomplete
    def __init__(self, _internal: _PQCodebook) -> None: ...
    @classmethod
    def train(
        cls,
        training_data: np.ndarray,
        n_subspaces: int = 96,
        n_centroids: int = 256,
        max_iter: int = 25,
        random_state: int = 0,
    ) -> PQCodebook: ...
    def encode(self, vectors: np.ndarray) -> np.ndarray: ...
    def decode(self, codes: np.ndarray) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> PQCodebook: ...
    @property
    def n_subspaces(self) -> int: ...
    @property
    def n_centroids(self) -> int: ...
    @property
    def sub_dim(self) -> int: ...
    def compression_ratio(self, d: int) -> float: ...
    def __repr__(self) -> str: ...

class HNSWIndex:
    _dim: Incomplete
    _quantization: Incomplete
    _index: Incomplete
    _user_ids: list[Any]
    def __init__(
        self,
        dim: int,
        quantization: str = "int8",
        M: int = 16,
        ef_build: int = 200,
        space: str = "cosine",
    ) -> None: ...
    def add_batch(self, vectors: np.ndarray, ids: Sequence | None = None) -> None: ...
    def search(
        self, query: np.ndarray, top_k: int = 10, ef: int = 64
    ) -> tuple[list[Any], np.ndarray]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> HNSWIndex: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

_VALID_PROFILES: Incomplete

class VectroV3:
    _profile: Incomplete
    _codebook: Incomplete
    _rq: Incomplete
    def __init__(
        self,
        profile: str = "int8",
        codebook: PQCodebook | None = None,
        rq: ResidualQuantizer | None = None,
    ) -> None: ...
    def train_rq(
        self, training_data: np.ndarray, n_subspaces: int = 96, n_passes: int = 3
    ) -> None: ...
    def compress(self, vectors: np.ndarray) -> V3Result: ...
    def decompress(self, result: V3Result) -> np.ndarray: ...
    @classmethod
    def auto_compress(
        cls, vectors: np.ndarray, target_cosine: float = 0.97, target_compression: float = 8.0
    ) -> dict: ...
    def save_compressed(
        self, result: V3Result, path: str, compression: str = "zstd", level: int = 3
    ) -> None: ...
    def load_compressed(self, path: str) -> V3Result: ...
    def _cloud_save(
        self, result: V3Result, uri: str, compression: str = "zstd", level: int = 3
    ) -> None: ...
    def _cloud_load(self, uri: str) -> V3Result: ...
    @property
    def profile(self) -> str: ...
    def __repr__(self) -> str: ...

def _is_cloud_uri(path: str) -> bool: ...
def _is_vqz(path: str) -> bool: ...
def _uri_key(uri: str) -> str: ...
def _make_cloud_backend(uri: str, S3Backend, GCSBackend, AzureBlobBackend): ...
