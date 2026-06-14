"""storage_v3.py — VQZ container: 64-byte header + ZSTD/zlib-compressed body.

Phase 8: adds a custom binary format (.vqz) on top of the existing NPZ
storage so that large embedding databases can be transported as a single
opaque file with fast decompression and optional cloud backend support.

Binary layout
-------------
Offset  Length  Field
------  ------  -----
0       8       magic: b"VECTRO\\x03\\x00"
8       2       version: uint16, currently 1
10      2       comp_flags: uint16  — 0=none, 1=zstd, 2=zlib
12      8       n_vectors: uint64
20      4       dims: uint32
24      2       n_subspaces: uint16 (used by PQ; 0 otherwise)
26      4       metadata_len: uint32
30      8       checksum: first 8 bytes of blake2b over the compressed body
38      26      reserved: zero-filled
64      ??      metadata_len bytes of raw metadata (UTF-8 or arbitrary bytes)
64+meta ??      compressed (or raw) body

Body layout (uncompressed view)
--------------------------------
quantized   : int8   array  shape (n_vectors, dims)
scales      : float32 array shape (n_vectors,)

Both arrays are concatenated flat in C order; scales immediately follow
quantized.  Total body bytes = n_vectors * dims + n_vectors * 4.
"""

import hashlib
import struct
import zlib
from collections import namedtuple

import numpy as np

# Try zstandard (preferred); fall back to zlib which is stdlib.
try:
    import zstandard as zstd

    _HAVE_ZSTD = True
except ImportError:  # pragma: no cover
    _HAVE_ZSTD = False

# Optional fsspec for cloud backends.
try:
    import fsspec

    _HAVE_FSSPEC = True
except ImportError:  # pragma: no cover
    _HAVE_FSSPEC = False


MAGIC = b"VECTRO\x03\x00"
HEADER_SIZE = 64
_VERSION = 1
_FLAG_NONE = 0
_FLAG_ZSTD = 1
_FLAG_ZLIB = 2

# Lightweight result type returned by load_compressed.  It intentionally
# mirrors python.interface.QuantizationResult so callers that previously used
# that type can switch without code changes.  Defined here so storage_v3 stays
# self-contained with no cross-package relative imports.
VQZResult = namedtuple(
    "VQZResult",
    ["quantized", "scales", "dims", "n", "precision_mode", "group_size"],
    defaults=["int8", 0],
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compress(data: bytes, compression: str, level: int) -> tuple[bytes, int]:
    """Return (compressed_bytes, comp_flag)."""
    if compression == "zstd":
        if _HAVE_ZSTD:
            cctx = zstd.ZstdCompressor(level=min(level, 22))
            return cctx.compress(data), _FLAG_ZSTD
        # Silently fall back to zlib when zstandard not installed.
        return zlib.compress(data, level=min(level, 9)), _FLAG_ZLIB
    if compression == "zlib":
        return zlib.compress(data, level=min(level, 9)), _FLAG_ZLIB
    # compression == "none" or unknown
    return data, _FLAG_NONE


def _decompress(data: bytes, comp_flag: int) -> bytes:
    if comp_flag == _FLAG_ZSTD:
        if not _HAVE_ZSTD:
            raise RuntimeError(
                "File was compressed with zstd but the 'zstandard' package "
                "is not installed.  Install it with: pip install zstandard"
            )
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    if comp_flag == _FLAG_ZLIB:
        return zlib.decompress(data)
    return data  # _FLAG_NONE


def _build_header(
    comp_flag: int,
    n_vectors: int,
    dims: int,
    n_subspaces: int,
    metadata_len: int,
    checksum: bytes,
) -> bytes:
    header = bytearray(HEADER_SIZE)
    header[0:8] = MAGIC
    struct.pack_into("<H", header, 8, _VERSION)
    struct.pack_into("<H", header, 10, comp_flag)
    struct.pack_into("<Q", header, 12, n_vectors)
    struct.pack_into("<I", header, 20, dims)
    struct.pack_into("<H", header, 24, n_subspaces)
    struct.pack_into("<I", header, 26, metadata_len)
    # Checksum: 8 bytes at offset 30
    header[30:38] = checksum[:8]
    # reserved [38:64] stays zero
    return bytes(header)


def _parse_header(raw: bytes) -> dict:
    if raw[:8] != MAGIC:
        raise ValueError(f"Not a valid VQZ file (magic mismatch: {raw[:8]!r})")
    version = struct.unpack_from("<H", raw, 8)[0]
    comp_flag = struct.unpack_from("<H", raw, 10)[0]
    n_vectors = struct.unpack_from("<Q", raw, 12)[0]
    dims = struct.unpack_from("<I", raw, 20)[0]
    n_subspaces = struct.unpack_from("<H", raw, 24)[0]
    metadata_len = struct.unpack_from("<I", raw, 26)[0]
    checksum = raw[30:38]
    return dict(
        version=version,
        comp_flag=comp_flag,
        n_vectors=n_vectors,
        dims=dims,
        n_subspaces=n_subspaces,
        metadata_len=metadata_len,
        checksum=checksum,
    )


def _body_checksum(compressed_body: bytes) -> bytes:
    return hashlib.blake2b(compressed_body, digest_size=8).digest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_vqz(
    quantized: np.ndarray,
    scales: np.ndarray,
    dims: int,
    path: str,
    compression: str = "zstd",
    metadata: bytes = b"",
    level: int = 3,
    n_subspaces: int = 0,
) -> None:
    """Write a VQZ file.

    Parameters
    ----------
    quantized   : int8 ndarray of shape (n_vectors, dims)
    scales      : float32 ndarray of shape (n_vectors,)
    dims        : embedding dimensionality (stored in header for validation)
    path        : output file path
    compression : "zstd" | "zlib" | "none"
    metadata    : arbitrary bytes for application use (max ~4 GB)
    level       : compression level (1–22 for zstd, 1–9 for zlib)
    n_subspaces : PQ sub-space count, 0 for non-PQ data
    """
    quantized = np.ascontiguousarray(quantized, dtype=np.int8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    n_vectors = quantized.shape[0]

    # Flat body: int8 quantized || float32 scales
    body_raw = quantized.flatten().tobytes() + scales.tobytes()
    body_compressed, comp_flag = _compress(body_raw, compression, level)

    checksum = _body_checksum(body_compressed)
    if isinstance(metadata, str):
        metadata = metadata.encode("utf-8")
    header = _build_header(comp_flag, n_vectors, dims, n_subspaces, len(metadata), checksum)

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(metadata)
        fh.write(body_compressed)


def load_vqz(path: str) -> dict:
    """Read a VQZ file.

    Returns
    -------
    dict with keys:
        quantized  : int8 ndarray (n_vectors, dims)
        scales     : float32 ndarray (n_vectors,)
        dims       : int
        n_vectors  : int
        metadata   : bytes
        version    : int
        n_subspaces: int
    """
    with open(path, "rb") as fh:
        header_raw = fh.read(HEADER_SIZE)
        if len(header_raw) < HEADER_SIZE:
            raise ValueError("File too short to be a valid VQZ file.")
        hdr = _parse_header(header_raw)

        metadata_raw = fh.read(hdr["metadata_len"]) if hdr["metadata_len"] else b""
        body_compressed = fh.read()

    # Verify checksum
    computed = _body_checksum(body_compressed)
    if computed != hdr["checksum"]:
        raise ValueError("VQZ checksum mismatch — file may be corrupt or truncated.")

    body_raw = _decompress(body_compressed, hdr["comp_flag"])
    n_vectors = hdr["n_vectors"]
    dims = hdr["dims"]

    q_bytes = n_vectors * dims  # int8 = 1 byte each
    s_bytes = n_vectors * 4  # float32 = 4 bytes each

    if len(body_raw) < q_bytes + s_bytes:
        raise ValueError(
            f"VQZ body too short: expected {q_bytes + s_bytes} bytes, got {len(body_raw)}."
        )

    quantized = np.frombuffer(body_raw[:q_bytes], dtype=np.int8).reshape(n_vectors, dims).copy()
    scales = np.frombuffer(body_raw[q_bytes : q_bytes + s_bytes], dtype=np.float32).copy()

    return dict(
        quantized=quantized,
        scales=scales,
        dims=dims,
        n_vectors=n_vectors,
        metadata=metadata_raw,
        version=hdr["version"],
        n_subspaces=hdr["n_subspaces"],
    )


# ---------------------------------------------------------------------------
# Cloud backend stubs
# ---------------------------------------------------------------------------


class _CloudBackendBase:
    """Abstract cloud storage backend (requires fsspec)."""

    def __init__(self, bucket: str, prefix: str = ""):
        if not _HAVE_FSSPEC:
            raise ImportError(
                "Cloud backends require the 'fsspec' package.  Install it with: pip install fsspec"
            )
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._fs = self._open_fs()

    def _open_fs(self):  # pragma: no cover
        raise NotImplementedError

    def _full_path(self, name: str) -> str:
        if self.prefix:
            return f"{self.bucket}/{self.prefix}/{name}"
        return f"{self.bucket}/{name}"

    def upload(self, local_path: str, remote_name: str) -> None:
        with open(local_path, "rb") as src:
            with self._fs.open(self._full_path(remote_name), "wb") as dst:
                dst.write(src.read())

    def download(self, remote_name: str, local_path: str) -> None:
        with self._fs.open(self._full_path(remote_name), "rb") as src:
            with open(local_path, "wb") as dst:
                dst.write(src.read())

    def save_vqz(
        self,
        quantized: np.ndarray,
        scales: np.ndarray,
        dims: int,
        remote_name: str,
        **kwargs,
    ) -> None:
        # Write to a temp file then upload.
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".vqz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            save_vqz(quantized, scales, dims, tmp_path, **kwargs)
            self.upload(tmp_path, remote_name)
        finally:
            os.unlink(tmp_path)

    def load_vqz(self, remote_name: str) -> dict:
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".vqz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.download(remote_name, tmp_path)
            return load_vqz(tmp_path)
        finally:
            os.unlink(tmp_path)


class S3Backend(_CloudBackendBase):
    """AWS S3 backend (requires fsspec[s3])."""

    def _open_fs(self):  # pragma: no cover
        return fsspec.filesystem("s3")


class GCSBackend(_CloudBackendBase):
    """Google Cloud Storage backend (requires fsspec[gcs])."""

    def _open_fs(self):  # pragma: no cover
        return fsspec.filesystem("gcs")


class AzureBlobBackend(_CloudBackendBase):
    """Azure Blob Storage backend (requires fsspec[abfs])."""

    def _open_fs(self):  # pragma: no cover
        return fsspec.filesystem("abfs")


# ---------------------------------------------------------------------------
# QuantizationResult convenience wrappers
# ---------------------------------------------------------------------------


def save_compressed(
    result: object,
    filepath: str,
    codec: str = "zstd",
    level: int = 3,
) -> None:
    """Persist a ``QuantizationResult`` to a VQZ file with lossless compression.

    This is a thin convenience wrapper around :func:`save_vqz` for callers who
    already hold a ``QuantizationResult`` from ``compress_vectors`` /
    ``quantize_embeddings``.

    Args:
        result:   A ``QuantizationResult`` instance (must have ``.quantized``,
                  ``.scales``, and ``.dims`` attributes).
        filepath: Destination file path (conventionally ending in ``.vqz``).
        codec:    Compression codec — ``"zstd"`` (default), ``"zlib"``, or
                  ``"none"``.
        level:    Compression level (1–22 for zstd, 1–9 for zlib).
    """
    save_vqz(
        result.quantized,
        result.scales,
        result.dims,
        filepath,
        compression=codec,
        level=level,
    )


def load_compressed(filepath: str) -> object:
    """Load a ``VQZResult`` from a VQZ file.

    This is a thin convenience wrapper around :func:`load_vqz` that
    reconstructs a :class:`VQZResult` namedtuple rather than returning
    a plain dict.  The returned type intentionally mirrors
    ``python.interface.QuantizationResult`` so callers can use it
    interchangeably.

    Args:
        filepath: Path to a ``.vqz`` file previously written by
                  :func:`save_compressed` or :func:`save_vqz`.

    Returns:
        A :class:`VQZResult` instance with the restored arrays.
    """
    d = load_vqz(filepath)
    return VQZResult(
        quantized=d["quantized"],
        scales=d["scales"],
        dims=d["dims"],
        n=d["n_vectors"],
    )
