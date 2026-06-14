"""Tests for python/integrations/arrow_bridge.py using a pyarrow mock.

All tests run without a real pyarrow installation, making CI fast.
"""

import io
import os
import pickle
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.batch_api import BatchQuantizationResult
from python.interface import QuantizationResult


# ---------------------------------------------------------------------------
# Module-level mock classes (must be top-level so pickle can find them)
# ---------------------------------------------------------------------------


class _MockArray:
    def __init__(self, data, dtype=None):
        self._data = list(data)
        self.type = dtype

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        class _Scalar:
            def __init__(self, v):
                self._v = v

            def as_py(self):
                return self._v

        return _Scalar(self._data[idx])


class _MockTable:
    def __init__(self, data: dict, schema=None):
        self._data = data
        self.schema = schema

    def __len__(self):
        cols = list(self._data.values())
        return len(cols[0]) if cols else 0

    def column(self, name):
        return self._data[name]


class _MockBufferOutputStream:
    def __init__(self):
        self._buf = io.BytesIO()

    def write(self, data):
        self._buf.write(data)

    def getvalue(self):
        b = self._buf.getvalue()

        class _Buf:
            def __bytes__(self):
                return b

        return _Buf()


class _MockStreamWriter:
    def __init__(self, sink: _MockBufferOutputStream, schema):
        self._sink = sink
        self._tables: list = []

    def write_table(self, table: _MockTable):
        self._tables.append(table)

    def close(self):
        # Serialise each table as (data_dict, schema) tuples for portability
        payload = [(t._data, t.schema) for t in self._tables]
        self._sink._buf.write(pickle.dumps(payload))


class _MockStreamReader:
    def __init__(self, tables: list):
        self._tables = tables

    def read_all(self) -> _MockTable:
        return self._tables[0] if self._tables else _MockTable({}, None)


class _MockPyBuffer:
    def __init__(self, data: bytes):
        self._data = data

    def __bytes__(self):
        return self._data


class _MockIPC:
    @staticmethod
    def new_stream(sink, schema):
        return _MockStreamWriter(sink, schema)

    @staticmethod
    def open_stream(buf: _MockPyBuffer):
        payload = pickle.loads(bytes(buf))
        tables = [_MockTable(d, s) for d, s in payload]
        return _MockStreamReader(tables)


class _MockSchema:
    def __init__(self, fields, metadata=None):
        self.fields = fields
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# Minimal pyarrow mock factory
# ---------------------------------------------------------------------------


def _make_pyarrow_mock() -> types.ModuleType:
    pa = types.ModuleType("pyarrow")

    pa.string = lambda: "string"
    pa.binary = lambda: "binary"
    pa.int32 = lambda: "int32"

    def _field(name, dtype, nullable=True):
        return {"name": name, "type": dtype, "nullable": nullable}

    pa.field = _field
    pa.schema = lambda fields, metadata=None: _MockSchema(fields, metadata)
    pa.array = lambda data, type=None: _MockArray(data, type)
    pa.table = lambda data, schema=None: _MockTable(data, schema)
    pa.BufferOutputStream = _MockBufferOutputStream
    pa.ipc = _MockIPC
    pa.py_buffer = lambda data: _MockPyBuffer(data)

    return pa


_pyarrow_mock = _make_pyarrow_mock()


# ---------------------------------------------------------------------------
# Helper to build test fixtures
# ---------------------------------------------------------------------------


def _make_batch_result(n: int = 4, dim: int = 8) -> BatchQuantizationResult:
    rng = np.random.default_rng(99)
    floats = rng.standard_normal((n, dim)).astype(np.float32)
    scales = np.max(np.abs(floats), axis=1) / 127.0
    q = np.clip(np.round(floats / scales[:, None]), -127, 127).astype(np.int8)
    original_bytes = n * dim * 4
    compressed_bytes = q.nbytes + scales.nbytes
    return BatchQuantizationResult(
        quantized_vectors=list(q),
        scales=scales,
        batch_size=n,
        vector_dim=dim,
        compression_ratio=original_bytes / compressed_bytes,
        total_original_bytes=original_bytes,
        total_compressed_bytes=compressed_bytes,
    )


def _make_quant_result(n: int = 4, dim: int = 8) -> QuantizationResult:
    rng = np.random.default_rng(77)
    floats = rng.standard_normal((n, dim)).astype(np.float32)
    scales = np.max(np.abs(floats), axis=1) / 127.0
    q = np.clip(np.round(floats / scales[:, None]), -127, 127).astype(np.int8)
    return QuantizationResult(quantized=q, scales=scales, dims=dim, n=n)


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_python_snippet(snippet: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) if not existing else f"{_REPO_ROOT}{os.pathsep}{existing}"
    return subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        check=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestArrowBridgeImportError(unittest.TestCase):
    """When pyarrow is absent, functions raise a clear RuntimeError."""

    def test_lazy_import_raises_when_absent(self):
        """_pa() raises RuntimeError with a helpful message.

        Run this in a subprocess to avoid mutating import state in the test runner.
        """
        snippet = textwrap.dedent(
            """
            import importlib

            from python.integrations import arrow_bridge

            _real_import_module = importlib.import_module

            def _blocked_import(name, package=None):
                if name == "pyarrow":
                    raise ImportError("blocked by test")
                return _real_import_module(name, package)

            importlib.import_module = _blocked_import
            try:
                try:
                    arrow_bridge._pa()
                except (RuntimeError, ImportError):
                    print("arrow_missing_ok")
                else:
                    raise SystemExit("Expected _pa() to fail when pyarrow is unavailable")
            finally:
                importlib.import_module = _real_import_module
            """
        )

        result = _run_python_snippet(snippet)
        self.assertEqual(
            result.returncode,
            0,
            msg=(
                "Arrow import-error subprocess check failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
        )
        self.assertIn("arrow_missing_ok", result.stdout)


class TestResultToTable(unittest.TestCase):
    """result_to_table builds the correct column structure."""

    def setUp(self):
        self._pyarrow_was_present = "pyarrow" in sys.modules
        self._pyarrow_orig = sys.modules.get("pyarrow")
        sys.modules["pyarrow"] = _pyarrow_mock

    def tearDown(self):
        if self._pyarrow_was_present:
            sys.modules["pyarrow"] = self._pyarrow_orig
        else:
            sys.modules.pop("pyarrow", None)

    def test_batch_result_column_count(self):
        from python.integrations.arrow_bridge import result_to_table

        result = _make_batch_result(4, 8)
        table = result_to_table(result)
        self.assertIsNotNone(table)

    def test_table_length_matches_vectors(self):
        from python.integrations.arrow_bridge import result_to_table

        result = _make_batch_result(6, 16)
        table = result_to_table(result)
        self.assertEqual(len(table), 6)

    def test_ids_stored_correctly(self):
        from python.integrations.arrow_bridge import result_to_table

        result = _make_batch_result(3, 8)
        ids = ["a", "b", "c"]
        table = result_to_table(result, ids=ids)
        for i, expected in enumerate(ids):
            self.assertEqual(table.column("id")[i].as_py(), expected)

    def test_null_ids_when_none(self):
        from python.integrations.arrow_bridge import result_to_table

        result = _make_batch_result(2, 8)
        table = result_to_table(result, ids=None)
        for i in range(2):
            self.assertIsNone(table.column("id")[i].as_py())

    def test_vector_dim_stored(self):
        from python.integrations.arrow_bridge import result_to_table

        result = _make_batch_result(2, 16)
        table = result_to_table(result)
        self.assertEqual(table.column("vector_dim")[0].as_py(), 16)

    def test_precision_mode_stored(self):
        from python.integrations.arrow_bridge import result_to_table

        result = _make_batch_result()
        table = result_to_table(result)
        self.assertEqual(table.column("precision_mode")[0].as_py(), "int8")

    def test_quantized_binary_column_nonempty(self):
        from python.integrations.arrow_bridge import result_to_table

        result = _make_batch_result(3, 8)
        table = result_to_table(result)
        for i in range(3):
            data = table.column("quantized")[i].as_py()
            self.assertIsNotNone(data)
            self.assertGreater(len(data), 0)

    def test_quant_result_table_length(self):
        from python.integrations.arrow_bridge import result_to_table

        result = _make_quant_result(5, 8)
        table = result_to_table(result)
        self.assertEqual(len(table), 5)


class TestTableToResult(unittest.TestCase):
    """table_to_result reconstructs a BatchQuantizationResult."""

    def setUp(self):
        self._pyarrow_was_present = "pyarrow" in sys.modules
        self._pyarrow_orig = sys.modules.get("pyarrow")
        sys.modules["pyarrow"] = _pyarrow_mock

    def tearDown(self):
        if self._pyarrow_was_present:
            sys.modules["pyarrow"] = self._pyarrow_orig
        else:
            sys.modules.pop("pyarrow", None)

    def _roundtrip(self, n: int = 4, dim: int = 8) -> BatchQuantizationResult:
        from python.integrations.arrow_bridge import result_to_table, table_to_result

        original = _make_batch_result(n, dim)
        table = result_to_table(original)
        return table_to_result(table)

    def test_roundtrip_batch_size(self):
        rt = self._roundtrip(4, 8)
        self.assertEqual(rt.batch_size, 4)

    def test_roundtrip_vector_dim(self):
        rt = self._roundtrip(4, 16)
        self.assertEqual(rt.vector_dim, 16)

    def test_roundtrip_precision_mode(self):
        rt = self._roundtrip()
        self.assertEqual(rt.precision_mode, "int8")

    def test_roundtrip_quantized_values_preserved(self):
        """Binary round-trip: quantized ints are unchanged."""
        from python.integrations.arrow_bridge import result_to_table, table_to_result

        original = _make_batch_result(4, 8)
        table = result_to_table(original)
        rt = table_to_result(table)

        orig_arr = np.vstack(original.quantized_vectors)
        rt_arr = np.vstack(rt.quantized_vectors)
        np.testing.assert_array_equal(orig_arr, rt_arr)

    def test_roundtrip_scales_preserved(self):
        from python.integrations.arrow_bridge import result_to_table, table_to_result

        original = _make_batch_result(4, 8)
        table = result_to_table(original)
        rt = table_to_result(table)
        np.testing.assert_allclose(
            np.asarray(original.scales).ravel(),
            np.asarray(rt.scales).ravel(),
            rtol=1e-6,
        )

    def test_compression_ratio_positive(self):
        rt = self._roundtrip(4, 8)
        self.assertGreater(rt.compression_ratio, 0)


class TestArrowBytes(unittest.TestCase):
    """to_arrow_bytes / from_arrow_bytes IPC wire round-trip."""

    def setUp(self):
        self._pyarrow_was_present = "pyarrow" in sys.modules
        self._pyarrow_orig = sys.modules.get("pyarrow")
        sys.modules["pyarrow"] = _pyarrow_mock

    def tearDown(self):
        if self._pyarrow_was_present:
            sys.modules["pyarrow"] = self._pyarrow_orig
        else:
            sys.modules.pop("pyarrow", None)

    def test_bytes_nonempty(self):
        from python.integrations.arrow_bridge import to_arrow_bytes

        result = _make_batch_result(3, 8)
        data = to_arrow_bytes(result)
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

    def test_bytes_roundtrip_batch_size(self):
        from python.integrations.arrow_bridge import to_arrow_bytes, from_arrow_bytes

        result = _make_batch_result(3, 8)
        rt = from_arrow_bytes(to_arrow_bytes(result))
        self.assertEqual(rt.batch_size, 3)

    def test_bytes_roundtrip_quantized(self):
        from python.integrations.arrow_bridge import to_arrow_bytes, from_arrow_bytes

        result = _make_batch_result(3, 8)
        rt = from_arrow_bytes(to_arrow_bytes(result))
        orig = np.vstack(result.quantized_vectors)
        recon = np.vstack(rt.quantized_vectors)
        np.testing.assert_array_equal(orig, recon)


if __name__ == "__main__":
    unittest.main()
