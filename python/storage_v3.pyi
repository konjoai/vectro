import numpy as np
from _typeshed import Incomplete
from typing import NamedTuple

_HAVE_ZSTD: bool
_HAVE_FSSPEC: bool
MAGIC: bytes
HEADER_SIZE: int
_VERSION: int
_FLAG_NONE: int
_FLAG_ZSTD: int
_FLAG_ZLIB: int

class VQZResult(NamedTuple):
    quantized: Incomplete
    scales: Incomplete
    dims: Incomplete
    n: Incomplete
    precision_mode: Incomplete
    group_size: Incomplete

def _compress(data: bytes, compression: str, level: int) -> tuple[bytes, int]: ...
def _decompress(data: bytes, comp_flag: int) -> bytes: ...
def _build_header(
    comp_flag: int, n_vectors: int, dims: int, n_subspaces: int, metadata_len: int, checksum: bytes
) -> bytes: ...
def _parse_header(raw: bytes) -> dict: ...
def _body_checksum(compressed_body: bytes) -> bytes: ...
def save_vqz(
    quantized: np.ndarray,
    scales: np.ndarray,
    dims: int,
    path: str,
    compression: str = "zstd",
    metadata: bytes = b"",
    level: int = 3,
    n_subspaces: int = 0,
) -> None: ...
def load_vqz(path: str) -> dict: ...

class _CloudBackendBase:
    bucket: Incomplete
    prefix: Incomplete
    _fs: Incomplete
    def __init__(self, bucket: str, prefix: str = "") -> None: ...
    def _open_fs(self) -> None: ...
    def _full_path(self, name: str) -> str: ...
    def upload(self, local_path: str, remote_name: str) -> None: ...
    def download(self, remote_name: str, local_path: str) -> None: ...
    def save_vqz(
        self, quantized: np.ndarray, scales: np.ndarray, dims: int, remote_name: str, **kwargs
    ) -> None: ...
    def load_vqz(self, remote_name: str) -> dict: ...

class S3Backend(_CloudBackendBase):
    def _open_fs(self): ...

class GCSBackend(_CloudBackendBase):
    def _open_fs(self): ...

class AzureBlobBackend(_CloudBackendBase):
    def _open_fs(self): ...

def save_compressed(result: object, filepath: str, codec: str = "zstd", level: int = 3) -> None: ...
def load_compressed(filepath: str) -> object: ...
