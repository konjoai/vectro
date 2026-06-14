"""tests/test_storage_v3.py — Phase 8: VQZ container and cloud backend stubs."""

import io
import os

import numpy as np
import pytest

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.storage_v3 import (  # noqa: E402
    MAGIC,
    HEADER_SIZE,
    save_vqz,
    load_vqz,
    S3Backend,
    GCSBackend,
    AzureBlobBackend,
    _compress,
    _decompress,
    _parse_header,
    _body_checksum,
    _build_header,
    _FLAG_NONE,
    _FLAG_ZSTD,
    _FLAG_ZLIB,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_data():
    rng = np.random.default_rng(42)
    n, d = 32, 64
    quantized = rng.integers(-128, 128, size=(n, d), dtype=np.int8)
    scales = rng.random(n).astype(np.float32)
    return quantized, scales, d


@pytest.fixture()
def tmp_vqz(tmp_path, small_data):
    quantized, scales, d = small_data
    path = str(tmp_path / "test.vqz")
    save_vqz(quantized, scales, d, path)
    return path, quantized, scales, d


# ---------------------------------------------------------------------------
# 1. Header construction and parsing
# ---------------------------------------------------------------------------


class TestHeader:
    def test_magic_length(self):
        assert len(MAGIC) == 8

    def test_header_size_constant(self):
        assert HEADER_SIZE == 64

    def test_build_and_parse_round_trip(self):
        checksum = b"\xab\xcd\xef\x01\x23\x45\x67\x89"
        raw = _build_header(
            comp_flag=_FLAG_ZSTD,
            n_vectors=1000,
            dims=768,
            n_subspaces=96,
            metadata_len=42,
            checksum=checksum,
        )
        assert len(raw) == HEADER_SIZE
        hdr = _parse_header(raw)
        assert hdr["version"] == 1
        assert hdr["comp_flag"] == _FLAG_ZSTD
        assert hdr["n_vectors"] == 1000
        assert hdr["dims"] == 768
        assert hdr["n_subspaces"] == 96
        assert hdr["metadata_len"] == 42
        assert hdr["checksum"] == checksum[:8]

    def test_parse_rejects_bad_magic(self):
        bad = b"INVALID\x00" + b"\x00" * (HEADER_SIZE - 8)
        with pytest.raises(ValueError, match="magic"):
            _parse_header(bad)

    def test_reserved_bytes_are_zero(self):
        checksum = b"\x00" * 8
        raw = _build_header(_FLAG_NONE, 1, 4, 0, 0, checksum)
        assert raw[38:64] == b"\x00" * 26


# ---------------------------------------------------------------------------
# 2. Compression helpers
# ---------------------------------------------------------------------------


class TestCompression:
    def test_zlib_round_trip(self):
        data = b"hello world " * 1000
        compressed, flag = _compress(data, "zlib", 6)
        assert flag == _FLAG_ZLIB
        assert len(compressed) < len(data)
        assert _decompress(compressed, flag) == data

    def test_none_round_trip(self):
        data = b"\x00\x01\x02\x03" * 100
        out, flag = _compress(data, "none", 0)
        assert flag == _FLAG_NONE
        assert out == data
        assert _decompress(out, _FLAG_NONE) == data

    def test_zstd_round_trip(self):
        pytest.importorskip("zstandard")
        data = b"compress me " * 500
        compressed, flag = _compress(data, "zstd", 3)
        assert flag == _FLAG_ZSTD
        assert _decompress(compressed, flag) == data

    def test_zstd_fallback_to_zlib_when_missing(self, monkeypatch):
        import python.storage_v3 as storage_v3

        monkeypatch.setattr(storage_v3, "_HAVE_ZSTD", False)
        data = b"fallback test " * 200
        compressed, flag = storage_v3._compress(data, "zstd", 3)
        assert flag == _FLAG_ZLIB
        assert storage_v3._decompress(compressed, flag) == data

    def test_body_checksum_deterministic(self):
        data = b"checksum me"
        c1 = _body_checksum(data)
        c2 = _body_checksum(data)
        assert c1 == c2
        assert len(c1) == 8

    def test_body_checksum_sensitive(self):
        assert _body_checksum(b"abc") != _body_checksum(b"abd")


# ---------------------------------------------------------------------------
# 3. save_vqz / load_vqz round-trips
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_basic_round_trip_zstd(self, small_data, tmp_path):
        pytest.importorskip("zstandard")
        quantized, scales, d = small_data
        path = str(tmp_path / "out.vqz")
        save_vqz(quantized, scales, d, path, compression="zstd")
        result = load_vqz(path)
        np.testing.assert_array_equal(result["quantized"], quantized)
        np.testing.assert_array_almost_equal(result["scales"], scales)

    def test_basic_round_trip_zlib(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "out_zlib.vqz")
        save_vqz(quantized, scales, d, path, compression="zlib")
        result = load_vqz(path)
        np.testing.assert_array_equal(result["quantized"], quantized)
        np.testing.assert_array_almost_equal(result["scales"], scales)

    def test_basic_round_trip_no_compression(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "out_raw.vqz")
        save_vqz(quantized, scales, d, path, compression="none")
        result = load_vqz(path)
        np.testing.assert_array_equal(result["quantized"], quantized)
        np.testing.assert_array_almost_equal(result["scales"], scales)

    def test_quantized_dtype_preserved(self, tmp_vqz):
        path, _, _, _ = tmp_vqz
        result = load_vqz(path)
        assert result["quantized"].dtype == np.int8

    def test_scales_dtype_preserved(self, tmp_vqz):
        path, _, _, _ = tmp_vqz
        result = load_vqz(path)
        assert result["scales"].dtype == np.float32

    def test_shape_preserved(self, small_data, tmp_vqz):
        path, quantized, scales, d = tmp_vqz
        result = load_vqz(path)
        assert result["quantized"].shape == quantized.shape
        assert result["scales"].shape == scales.shape

    def test_dims_in_result(self, tmp_vqz):
        path, _, _, d = tmp_vqz
        result = load_vqz(path)
        assert result["dims"] == d

    def test_n_vectors_in_result(self, small_data, tmp_vqz):
        path, quantized, _, _ = tmp_vqz
        result = load_vqz(path)
        assert result["n_vectors"] == quantized.shape[0]

    def test_version_in_result(self, tmp_vqz):
        path, _, _, _ = tmp_vqz
        result = load_vqz(path)
        assert result["version"] == 1

    def test_metadata_round_trip_bytes(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "meta.vqz")
        meta = b"\xde\xad\xbe\xef app-specific blob"
        save_vqz(quantized, scales, d, path, metadata=meta)
        result = load_vqz(path)
        assert result["metadata"] == meta

    def test_metadata_round_trip_string(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "meta_str.vqz")
        save_vqz(quantized, scales, d, path, metadata="hello vectro")
        result = load_vqz(path)
        assert result["metadata"] == b"hello vectro"

    def test_empty_metadata(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "nometa.vqz")
        save_vqz(quantized, scales, d, path, metadata=b"")
        result = load_vqz(path)
        assert result["metadata"] == b""

    def test_n_subspaces_stored(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "pq.vqz")
        save_vqz(quantized, scales, d, path, n_subspaces=8)
        result = load_vqz(path)
        assert result["n_subspaces"] == 8

    def test_compressed_file_smaller_than_raw(self, tmp_path):
        rng = np.random.default_rng(7)
        n, d = 512, 256
        # Use non-random (compressible) data
        quantized = np.zeros((n, d), dtype=np.int8)
        scales = np.ones(n, dtype=np.float32)
        raw_path = str(tmp_path / "raw.vqz")
        zlib_path = str(tmp_path / "zlib.vqz")
        save_vqz(quantized, scales, d, raw_path, compression="none")
        save_vqz(quantized, scales, d, zlib_path, compression="zlib")
        assert os.path.getsize(zlib_path) < os.path.getsize(raw_path)

    def test_checksum_detects_corruption(self, tmp_vqz):
        path, _, _, _ = tmp_vqz
        with open(path, "r+b") as fh:
            # Flip a byte in the body (after header)
            fh.seek(HEADER_SIZE + 5)
            orig = fh.read(1)
            fh.seek(HEADER_SIZE + 5)
            fh.write(bytes([orig[0] ^ 0xFF]))
        with pytest.raises(ValueError, match="checksum"):
            load_vqz(path)

    def test_rejects_truncated_file(self, tmp_path):
        path = str(tmp_path / "truncated.vqz")
        with open(path, "wb") as fh:
            fh.write(b"\x00" * 10)
        with pytest.raises(ValueError):
            load_vqz(path)

    def test_large_batch(self, tmp_path):
        rng = np.random.default_rng(99)
        n, d = 4096, 384
        quantized = rng.integers(-128, 128, size=(n, d), dtype=np.int8)
        scales = rng.random(n).astype(np.float32)
        path = str(tmp_path / "large.vqz")
        save_vqz(quantized, scales, d, path, compression="zlib")
        result = load_vqz(path)
        np.testing.assert_array_equal(result["quantized"], quantized)
        np.testing.assert_array_almost_equal(result["scales"], scales)


# ---------------------------------------------------------------------------
# 4. Cloud backend stubs (instantiation-only; no real I/O)
# ---------------------------------------------------------------------------


class TestCloudBackendStubs:
    def test_s3_backend_requires_fsspec(self, monkeypatch):
        import python.storage_v3 as storage_v3

        monkeypatch.setattr(storage_v3, "_HAVE_FSSPEC", False)
        with pytest.raises(ImportError, match="fsspec"):
            S3Backend("my-bucket")

    def test_gcs_backend_requires_fsspec(self, monkeypatch):
        import python.storage_v3 as storage_v3

        monkeypatch.setattr(storage_v3, "_HAVE_FSSPEC", False)
        with pytest.raises(ImportError, match="fsspec"):
            GCSBackend("my-bucket")

    def test_azure_backend_requires_fsspec(self, monkeypatch):
        import python.storage_v3 as storage_v3

        monkeypatch.setattr(storage_v3, "_HAVE_FSSPEC", False)
        with pytest.raises(ImportError, match="fsspec"):
            AzureBlobBackend("my-container")

    def test_backends_have_expected_attributes(self, monkeypatch):
        import python.storage_v3 as storage_v3

        # Patch _open_fs so we don't need real cloud credentials.
        monkeypatch.setattr(storage_v3, "_HAVE_FSSPEC", True)

        class _FakeFS:
            pass

        for BackendCls in (S3Backend, GCSBackend, AzureBlobBackend):
            monkeypatch.setattr(BackendCls, "_open_fs", lambda self: _FakeFS())
            backend = BackendCls("test-bucket", "prefix/v1")
            assert backend.bucket == "test-bucket"
            assert backend.prefix == "prefix/v1"
            assert hasattr(backend, "upload")
            assert hasattr(backend, "download")
            assert hasattr(backend, "save_vqz")
            assert hasattr(backend, "load_vqz")


# ── In-memory filesystem mock ─────────────────────────────────────────────────


class _MemFS:
    """Minimal in-memory filesystem mock that satisfies _CloudBackendBase.upload/download."""

    def __init__(self):
        self._store: dict[str, bytes] = {}

    def open(self, path: str, mode: str):
        if "w" in mode:
            return _MemWriteCtx(self._store, path)
        return _MemReadCtx(self._store[path])


class _MemWriteCtx:
    def __init__(self, store, path):
        self._store = store
        self._path = path
        self._buf = io.BytesIO()

    def write(self, data: bytes):
        self._buf.write(data)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._store[self._path] = self._buf.getvalue()


class _MemReadCtx:
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# ── Cloud backend round-trip tests ────────────────────────────────────────────


class TestCloudBackendRoundTrip:
    """Full save_vqz → upload → download → load_vqz round-trips via mock FS."""

    @pytest.fixture
    def small(self):
        rng = np.random.default_rng(99)
        q = rng.integers(-127, 128, size=(8, 16), dtype=np.int8)
        s = rng.random(8).astype(np.float32)
        return q, s, 16

    def _make_backend(self, BackendCls, monkeypatch):
        """Return a backend instance wired to a MemFS, no real fsspec needed."""
        import python.storage_v3 as storage_v3

        monkeypatch.setattr(storage_v3, "_HAVE_FSSPEC", True)
        mem = _MemFS()
        monkeypatch.setattr(BackendCls, "_open_fs", lambda self: mem)
        return BackendCls("test-bucket", "prefix"), mem

    def test_s3_save_and_load_round_trip(self, monkeypatch, small):
        q, s, dims = small
        backend, mem = self._make_backend(S3Backend, monkeypatch)

        backend.save_vqz(q, s, dims, "vectors.vqz")
        assert "test-bucket/prefix/vectors.vqz" in mem._store

        result = backend.load_vqz("vectors.vqz")
        np.testing.assert_array_equal(result["quantized"], q)
        np.testing.assert_array_almost_equal(result["scales"], s, decimal=5)
        assert result["dims"] == dims

    def test_gcs_save_and_load_round_trip(self, monkeypatch, small):
        q, s, dims = small
        backend, mem = self._make_backend(GCSBackend, monkeypatch)

        backend.save_vqz(q, s, dims, "model/vecs.vqz")
        assert "test-bucket/prefix/model/vecs.vqz" in mem._store

        result = backend.load_vqz("model/vecs.vqz")
        np.testing.assert_array_equal(result["quantized"], q)
        assert result["n_vectors"] == q.shape[0]

    def test_azure_save_and_load_round_trip(self, monkeypatch, small):
        q, s, dims = small
        backend, mem = self._make_backend(AzureBlobBackend, monkeypatch)

        backend.save_vqz(q, s, dims, "embeddings.vqz")
        result = backend.load_vqz("embeddings.vqz")
        np.testing.assert_array_equal(result["quantized"], q)

    def test_full_path_with_prefix(self, monkeypatch, small):
        q, s, dims = small
        backend, mem = self._make_backend(S3Backend, monkeypatch)
        assert backend._full_path("file.vqz") == "test-bucket/prefix/file.vqz"

    def test_full_path_without_prefix(self, monkeypatch, small):
        import python.storage_v3 as storage_v3

        monkeypatch.setattr(storage_v3, "_HAVE_FSSPEC", True)
        mem = _MemFS()
        monkeypatch.setattr(S3Backend, "_open_fs", lambda self: mem)
        backend = S3Backend("my-bucket")
        assert backend._full_path("file.vqz") == "my-bucket/file.vqz"

    def test_upload_download_directly(self, monkeypatch, tmp_path, small):
        q, s, dims = small
        backend, mem = self._make_backend(S3Backend, monkeypatch)

        # Write a file and upload it
        local_in = tmp_path / "in.vqz"
        save_vqz(q, s, dims, str(local_in))
        backend.upload(str(local_in), "round_trip.vqz")
        assert "test-bucket/prefix/round_trip.vqz" in mem._store

        # Download it and verify
        local_out = tmp_path / "out.vqz"
        backend.download("round_trip.vqz", str(local_out))
        result = load_vqz(str(local_out))
        np.testing.assert_array_equal(result["quantized"], q)

    def test_save_vqz_passes_compression_kwargs(self, monkeypatch, small):
        """save_vqz should accept **kwargs like compression='none'."""
        q, s, dims = small
        backend, _ = self._make_backend(GCSBackend, monkeypatch)
        backend.save_vqz(q, s, dims, "no_comp.vqz", compression="none")
        result = backend.load_vqz("no_comp.vqz")
        assert result["dims"] == dims


# ---------------------------------------------------------------------------
# 5. save_compressed / load_compressed (QuantizationResult wrappers)
# ---------------------------------------------------------------------------

from python.storage_v3 import save_compressed, load_compressed, VQZResult  # noqa: E402


# Local lightweight stand-in for python.interface.QuantizationResult.
# save_compressed only reads .quantized, .scales, .dims (duck-typing); using
# a local namedtuple avoids triggering relative-import chains in interface.py
# when storage_v3 is imported standalone in this test module.
from typing import NamedTuple as _NT


class _QResult(_NT):
    quantized: np.ndarray
    scales: np.ndarray
    dims: int
    n: int
    precision_mode: str = "int8"
    group_size: int = 0


class TestSaveLoadCompressed:
    """Tests for the QuantizationResult-aware save_compressed / load_compressed."""

    @pytest.fixture()
    def qresult(self):
        rng = np.random.default_rng(55)
        n, d = 16, 32
        q = rng.integers(-128, 128, size=(n, d), dtype=np.int8)
        s = rng.random(n).astype(np.float32)
        return _QResult(quantized=q, scales=s, dims=d, n=n)

    def test_round_trip_default_codec(self, qresult, tmp_path):
        path = str(tmp_path / "out.vqz")
        save_compressed(qresult, path)
        r2 = load_compressed(path)
        assert r2.n == qresult.n
        assert r2.dims == qresult.dims
        np.testing.assert_array_equal(r2.quantized, qresult.quantized)
        np.testing.assert_array_almost_equal(r2.scales, qresult.scales)

    def test_round_trip_zlib(self, qresult, tmp_path):
        path = str(tmp_path / "out_zlib.vqz")
        save_compressed(qresult, path, codec="zlib")
        r2 = load_compressed(path)
        np.testing.assert_array_equal(r2.quantized, qresult.quantized)

    def test_round_trip_none(self, qresult, tmp_path):
        path = str(tmp_path / "out_none.vqz")
        save_compressed(qresult, path, codec="none")
        r2 = load_compressed(path)
        np.testing.assert_array_equal(r2.quantized, qresult.quantized)

    def test_quantized_dtype_preserved(self, qresult, tmp_path):
        path = str(tmp_path / "dtype.vqz")
        save_compressed(qresult, path)
        r2 = load_compressed(path)
        assert r2.quantized.dtype == np.int8

    def test_scales_dtype_preserved(self, qresult, tmp_path):
        path = str(tmp_path / "scales.vqz")
        save_compressed(qresult, path)
        r2 = load_compressed(path)
        assert r2.scales.dtype == np.float32

    def test_returns_quantization_result(self, qresult, tmp_path):
        path = str(tmp_path / "type.vqz")
        save_compressed(qresult, path)
        r2 = load_compressed(path)
        assert isinstance(r2, VQZResult)

    def test_compressed_smaller_than_raw(self, tmp_path):
        """zstd-compressed file must be smaller than the uncompressed file."""
        pytest.importorskip("zstandard")
        n, d = 256, 128
        q = np.zeros((n, d), dtype=np.int8)
        s = np.ones(n, dtype=np.float32)
        result = _QResult(quantized=q, scales=s, dims=d, n=n)
        raw_path = str(tmp_path / "raw.vqz")
        zstd_path = str(tmp_path / "zstd.vqz")
        save_compressed(result, raw_path, codec="none")
        save_compressed(result, zstd_path, codec="zstd")
        assert os.path.getsize(zstd_path) < os.path.getsize(raw_path)

    def test_zstd_round_trip_if_available(self, qresult, tmp_path):
        pytest.importorskip("zstandard")
        path = str(tmp_path / "zstd.vqz")
        save_compressed(qresult, path, codec="zstd", level=5)
        r2 = load_compressed(path)
        np.testing.assert_array_equal(r2.quantized, qresult.quantized)
        np.testing.assert_array_almost_equal(r2.scales, qresult.scales)

    def test_large_result_round_trip(self, tmp_path):
        rng = np.random.default_rng(77)
        n, d = 2048, 512
        q = rng.integers(-128, 128, size=(n, d), dtype=np.int8)
        s = rng.random(n).astype(np.float32)
        result = _QResult(quantized=q, scales=s, dims=d, n=n)
        path = str(tmp_path / "large.vqz")
        save_compressed(result, path, codec="zlib")
        r2 = load_compressed(path)
        assert r2.n == n
        assert r2.dims == d
        np.testing.assert_array_equal(r2.quantized, q)
