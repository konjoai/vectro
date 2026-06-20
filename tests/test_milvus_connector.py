"""Tests for MilvusConnector using a fake MilvusClient."""

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python import MilvusConnector  # noqa: E402
from python.integrations.vector_db import StoredVectorBatch  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal in-memory MilvusClient mock
# ---------------------------------------------------------------------------


class _FakeMilvusClient:
    """Minimal mock that satisfies MilvusConnector's usage of MilvusClient."""

    def __init__(self):
        self._collections: dict = {}

    def _get_collection(self, collection_name: str) -> dict:
        if collection_name not in self._collections:
            self._collections[collection_name] = {}
        return self._collections[collection_name]

    def upsert(self, collection_name: str, data: list) -> None:
        store = self._get_collection(collection_name)
        for row in data:
            row_id = str(row["id"])
            store[row_id] = dict(row)

    def get(self, collection_name: str, ids: list) -> list:
        store = self._get_collection(collection_name)
        result = []
        for row_id in ids:
            row = store.get(str(row_id))
            if row is not None:
                result.append(dict(row))
        return result

    def delete(self, collection_name: str, ids: list) -> None:
        store = self._get_collection(collection_name)
        for row_id in ids:
            store.pop(str(row_id), None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMilvusConnectorBasic(unittest.TestCase):
    def setUp(self):
        self.client = _FakeMilvusClient()
        self.connector = MilvusConnector(collection_name="test_vecs", client=self.client)

    def test_upsert_and_fetch_int8(self):
        ids = ["v1", "v2", "v3"]
        q = np.array([[1, 2, 4], [5, 6, 7], [8, 9, 10]], dtype=np.int8)
        s = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        self.connector.upsert_compressed(ids, q, s)
        batch = self.connector.fetch_compressed(ids)

        self.assertIsInstance(batch, StoredVectorBatch)
        self.assertEqual(batch.ids, ids)
        np.testing.assert_array_equal(batch.quantized, q)
        np.testing.assert_array_almost_equal(batch.scales.flatten(), s)
        self.assertEqual(batch.vector_dim, 3)

    def test_upsert_preserves_dtype_int8(self):
        ids = ["a"]
        q = np.array([[10, 20, 30]], dtype=np.int8)
        s = np.array([0.5], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)
        batch = self.connector.fetch_compressed(ids)
        self.assertEqual(batch.quantized.dtype, np.int8)

    def test_upsert_preserves_dtype_uint8_int4(self):
        """uint8 packed rows travel through as uint8 and trigger int4 path."""
        ids = ["b"]
        q = np.array([[11, 22, 33, 44]], dtype=np.uint8)
        s = np.array([[0.5, 0.25]], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)
        batch = self.connector.fetch_compressed(ids)
        self.assertEqual(batch.quantized.dtype, np.uint8)
        self.assertEqual(batch.vector_dim, 8)  # 4 uint8 cols × 2 = 8 dims

    def test_upsert_with_metadata(self):
        ids = ["m1"]
        q = np.array([[1, 2]], dtype=np.int8)
        s = np.array([0.9], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s, metadata={"model": "bert", "version": 3})
        batch = self.connector.fetch_compressed(ids)
        self.assertEqual(batch.metadata["model"], "bert")
        self.assertEqual(batch.metadata["version"], 3)

    def test_upsert_updates_existing_id(self):
        ids = ["dup"]
        q1 = np.array([[1, 2]], dtype=np.int8)
        s1 = np.array([0.1], dtype=np.float32)
        q2 = np.array([[99, 98]], dtype=np.int8)
        s2 = np.array([0.9], dtype=np.float32)
        self.connector.upsert_compressed(ids, q1, s1)
        self.connector.upsert_compressed(ids, q2, s2)
        batch = self.connector.fetch_compressed(ids)
        np.testing.assert_array_equal(batch.quantized, q2)

    def test_fetch_missing_ids_raises(self):
        with self.assertRaises(KeyError):
            self.connector.fetch_compressed(["nonexistent"])

    def test_delete_returns_count(self):
        ids = ["d1", "d2"]
        q = np.zeros((2, 4), dtype=np.int8)
        s = np.array([0.1, 0.2], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)
        n = self.connector.delete(["d1", "d2"])
        self.assertEqual(n, 2)

    def test_delete_removes_vectors(self):
        ids = ["x1"]
        q = np.array([[7, 8, 9]], dtype=np.int8)
        s = np.array([0.3], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)
        self.connector.delete(ids)
        with self.assertRaises(KeyError):
            self.connector.fetch_compressed(ids)

    def test_delete_empty_list_returns_zero(self):
        n = self.connector.delete([])
        self.assertEqual(n, 0)

    def test_shape_preserved_multi_row(self):
        rng = np.random.default_rng(42)
        n, d = 16, 32
        q = rng.integers(-128, 128, size=(n, d), dtype=np.int8)
        s = rng.random(n).astype(np.float32)
        ids = [f"vec_{i}" for i in range(n)]
        self.connector.upsert_compressed(ids, q, s)
        batch = self.connector.fetch_compressed(ids)
        self.assertEqual(batch.quantized.shape, (n, d))
        self.assertEqual(batch.scales.shape, (n,))

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            self.connector.upsert_compressed(
                ["a", "b"],
                np.zeros((3, 4), dtype=np.int8),
                np.zeros(3, dtype=np.float32),
            )

    def test_partial_fetch_returns_found_only(self):
        ids = ["p1", "p2"]
        q = np.array([[1, 2], [3, 4]], dtype=np.int8)
        s = np.array([0.1, 0.2], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)
        # "p3" does not exist; mock silently skips it
        batch = self.connector.fetch_compressed(["p1", "p3", "p2"])
        self.assertIn("p1", batch.ids)
        self.assertIn("p2", batch.ids)
        self.assertNotIn("p3", batch.ids)

    def test_no_pymilvus_raises_runtime_error(self):
        """Constructing without a client raises RuntimeError if pymilvus absent."""
        import importlib
        import unittest.mock as mock

        original_import = importlib.import_module

        def _failing_import(name, *args, **kwargs):
            if name == "pymilvus":
                raise ImportError("No module named 'pymilvus'")
            return original_import(name, *args, **kwargs)

        with mock.patch("importlib.import_module", side_effect=_failing_import):
            with self.assertRaises(RuntimeError, msg="pymilvus"):
                MilvusConnector("col")


class TestMilvusConnectorRoundTrip(unittest.TestCase):
    """End-to-end compress → upsert → fetch → reconstruct quality check."""

    def test_round_trip_data_integrity(self):
        rng = np.random.default_rng(7)
        n, d = 8, 16
        q = rng.integers(-127, 128, size=(n, d), dtype=np.int8)
        s = rng.random(n).astype(np.float32)
        ids = [f"rt_{i}" for i in range(n)]

        client = _FakeMilvusClient()
        connector = MilvusConnector("rt_col", client=client)
        connector.upsert_compressed(ids, q, s)
        batch = connector.fetch_compressed(ids)

        np.testing.assert_array_equal(batch.quantized, q)
        np.testing.assert_array_almost_equal(batch.scales.flatten(), s, decimal=5)

    def test_multiple_collections_isolated(self):
        client = _FakeMilvusClient()
        c1 = MilvusConnector("col_a", client=client)
        c2 = MilvusConnector("col_b", client=client)

        q = np.array([[1, 2]], dtype=np.int8)
        s = np.array([0.5], dtype=np.float32)

        c1.upsert_compressed(["id1"], q, s)
        with self.assertRaises(KeyError):
            c2.fetch_compressed(["id1"])


if __name__ == "__main__":
    unittest.main()
