"""Tests for Weaviate connector behavior using a fake v4-style client."""

import unittest
from types import SimpleNamespace

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python import WeaviateConnector  # noqa: E402


# ---------------------------------------------------------------------------
# Fake Weaviate v4 client stubs
# ---------------------------------------------------------------------------


class _FakeData:
    def __init__(self, store: dict):
        self._store = store

    def insert_many(self, objects):
        for obj in objects:
            self._store[str(obj.uuid)] = obj.properties

    def delete_by_id(self, uuid: str):
        self._store.pop(str(uuid), None)


class _FakeQuery:
    def __init__(self, store: dict):
        self._store = store

    def fetch_object_by_id(self, uuid: str):
        props = self._store.get(str(uuid))
        if props is None:
            return None
        return SimpleNamespace(uuid=str(uuid), properties=props)


class _FakeCollection:
    def __init__(self):
        self._store: dict = {}
        self.data = _FakeData(self._store)
        self.query = _FakeQuery(self._store)


class _FakeCollections:
    def __init__(self, collection: _FakeCollection):
        self._collection = collection

    def get(self, name: str) -> _FakeCollection:
        return self._collection


class _FakeWeaviateClient:
    def __init__(self):
        self._collection = _FakeCollection()
        self.collections = _FakeCollections(self._collection)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWeaviateConnector(unittest.TestCase):
    def setUp(self):
        self.fake_client = _FakeWeaviateClient()
        self.connector = WeaviateConnector(collection_name="test_vectors", client=self.fake_client)

    def test_upsert_and_fetch_int8(self):
        ids = ["v1", "v2", "v3"]
        q = np.array([[10, -20, 30], [-40, 50, -60], [70, -80, 90]], dtype=np.int8)
        s = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        self.connector.upsert_compressed(ids, q, s, metadata={"dataset": "test"})
        batch = self.connector.fetch_compressed(["v1", "v2", "v3"])

        self.assertEqual(batch.ids, ["v1", "v2", "v3"])
        self.assertEqual(batch.quantized.dtype, np.int8)
        self.assertEqual(batch.quantized.shape, (3, 3))
        np.testing.assert_array_equal(batch.quantized, q)
        self.assertEqual(batch.metadata["dataset"], "test")

    def test_upsert_and_fetch_int4_payload(self):
        """INT4 (uint8 nibble-packed) payloads preserve dtype and vector_dim."""
        ids = ["packed_0"]
        # 4 bytes = 8 nibbles = 8 int4 elements → vector_dim=8
        packed = np.array([[0xAB, 0xCD, 0xEF, 0x12]], dtype=np.uint8)
        scales = np.array([[0.5, 0.25, 0.5, 0.25]], dtype=np.float32)

        self.connector.upsert_compressed(ids, packed, scales)
        batch = self.connector.fetch_compressed(ids)

        self.assertEqual(batch.quantized.dtype, np.uint8)
        self.assertEqual(batch.quantized.shape, (1, 4))
        self.assertEqual(batch.vector_dim, 8)  # 4 bytes × 2 nibbles

    def test_partial_fetch_skips_missing(self):
        ids = ["a"]
        q = np.array([[1, 2]], dtype=np.int8)
        s = np.array([0.1], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)

        batch = self.connector.fetch_compressed(["a"])
        self.assertEqual(batch.ids, ["a"])

    def test_fetch_missing_raises(self):
        with self.assertRaises(KeyError):
            self.connector.fetch_compressed(["does_not_exist"])

    def test_delete_returns_requested_count(self):
        ids = ["d1", "d2"]
        q = np.array([[1, 2], [3, 4]], dtype=np.int8)
        s = np.array([0.1, 0.2], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)

        count = self.connector.delete(["d1", "d2"])
        self.assertEqual(count, 2)

        # Deleted items no longer fetchable
        with self.assertRaises(KeyError):
            self.connector.fetch_compressed(["d1"])

    def test_upsert_mismatch_raises(self):
        ids = ["x", "y"]
        q = np.array([[1, 2]], dtype=np.int8)  # only 1 row
        s = np.array([0.1, 0.2], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.connector.upsert_compressed(ids, q, s)

    def test_metadata_merged_on_fetch(self):
        """Metadata from multiple vectors is merged (last-write-wins per key)."""
        ids = ["m1", "m2"]
        q = np.zeros((2, 4), dtype=np.int8)
        s = np.array([1.0, 1.0], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s, metadata={"source": "demo"})

        batch = self.connector.fetch_compressed(ids)
        self.assertEqual(batch.metadata["source"], "demo")


if __name__ == "__main__":
    unittest.main()
