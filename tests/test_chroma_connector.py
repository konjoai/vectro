"""Tests for ChromaConnector using a fake chromadb client."""

import unittest
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python import ChromaConnector  # noqa: E402
from python.integrations.vector_db import StoredVectorBatch  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal in-memory Chroma mock
# ---------------------------------------------------------------------------


class _FakeChromaCollection:
    """Minimal mock of a chromadb Collection object."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        for row_id, emb, meta in zip(ids, embeddings, metadatas):
            self._store[row_id] = {"embedding": emb, "metadata": dict(meta)}

    def get(self, ids: List[str], include: Optional[List[str]] = None) -> Dict:
        out_ids: List[str] = []
        out_metas: List[Dict] = []
        for row_id in ids:
            row = self._store.get(row_id)
            if row is not None:
                out_ids.append(row_id)
                out_metas.append(dict(row["metadata"]))
        return {"ids": out_ids, "metadatas": out_metas}

    def delete(self, ids: List[str]) -> None:
        for row_id in ids:
            self._store.pop(row_id, None)


class _FakeChromaClient:
    """Minimal mock of a chromadb.ClientAPI object."""

    def __init__(self):
        self._collections: Dict[str, _FakeChromaCollection] = {}

    def get_or_create_collection(
        self,
        name: str,
        embedding_function: Any = None,
    ) -> _FakeChromaCollection:
        if name not in self._collections:
            self._collections[name] = _FakeChromaCollection()
        return self._collections[name]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChromaConnectorBasic(unittest.TestCase):
    def setUp(self):
        self.client = _FakeChromaClient()
        self.connector = ChromaConnector(collection_name="test_vecs", client=self.client)

    def test_upsert_and_fetch_int8(self):
        ids = ["v1", "v2"]
        q = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int8)
        s = np.array([0.1, 0.2], dtype=np.float32)

        self.connector.upsert_compressed(ids, q, s)
        batch = self.connector.fetch_compressed(ids)

        self.assertIsInstance(batch, StoredVectorBatch)
        self.assertEqual(set(batch.ids), {"v1", "v2"})
        np.testing.assert_array_equal(batch.quantized, q)
        self.assertEqual(batch.scales.dtype, np.float32)

    def test_dtype_int8_preserved(self):
        ids = ["dtype_i8"]
        q = np.array([[-1, 2, -3]], dtype=np.int8)
        s = np.array([0.5], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)
        batch = self.connector.fetch_compressed(ids)
        self.assertEqual(batch.quantized.dtype, np.int8)

    def test_dtype_uint8_preserved(self):
        """uint8 rows use the int4 precision path and are restored as uint8."""
        ids = ["dtype_u8"]
        q = np.array([[200, 100, 50, 25]], dtype=np.uint8)
        s = np.array([[0.8, 0.4]], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)
        batch = self.connector.fetch_compressed(ids)
        self.assertEqual(batch.quantized.dtype, np.uint8)
        self.assertEqual(batch.vector_dim, 8)  # 4 × 2

    def test_metadata_round_trip(self):
        ids = ["meta_test"]
        q = np.array([[5, 6]], dtype=np.int8)
        s = np.array([0.7], dtype=np.float32)
        user_meta = {"env": "test", "version": 2, "score": 0.99}
        self.connector.upsert_compressed(ids, q, s, metadata=user_meta)
        batch = self.connector.fetch_compressed(ids)
        self.assertEqual(batch.metadata["env"], "test")
        self.assertEqual(batch.metadata["version"], 2)

    def test_non_primitive_metadata_skipped(self):
        """Non-primitive metadata values (lists, dicts) are silently excluded."""
        ids = ["skip_meta"]
        q = np.array([[1, 2]], dtype=np.int8)
        s = np.array([0.3], dtype=np.float32)
        meta = {"ok_key": "ok_value", "bad_key": [1, 2, 3]}
        self.connector.upsert_compressed(ids, q, s, metadata=meta)
        batch = self.connector.fetch_compressed(ids)
        self.assertEqual(batch.metadata["ok_key"], "ok_value")
        self.assertNotIn("bad_key", batch.metadata)

    def test_vector_dim_stored(self):
        ids = ["dim_check"]
        q = np.zeros((1, 12), dtype=np.int8)
        s = np.array([1.0], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)
        batch = self.connector.fetch_compressed(ids)
        self.assertEqual(batch.vector_dim, 12)

    def test_upsert_updates_existing(self):
        ids = ["dup"]
        q1 = np.array([[1, 2, 3]], dtype=np.int8)
        q2 = np.array([[7, 8, 9]], dtype=np.int8)
        s = np.array([0.5], dtype=np.float32)
        self.connector.upsert_compressed(ids, q1, s)
        self.connector.upsert_compressed(ids, q2, s)
        batch = self.connector.fetch_compressed(ids)
        np.testing.assert_array_equal(batch.quantized, q2)

    def test_fetch_missing_ids_raises(self):
        with self.assertRaises(KeyError):
            self.connector.fetch_compressed(["does_not_exist"])

    def test_delete_returns_count(self):
        ids = ["del1", "del2"]
        q = np.zeros((2, 4), dtype=np.int8)
        s = np.ones(2, dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)
        n = self.connector.delete(ids)
        self.assertEqual(n, 2)

    def test_delete_removes_entries(self):
        ids = ["gone"]
        q = np.array([[1, 2]], dtype=np.int8)
        s = np.array([0.1], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)
        self.connector.delete(ids)
        with self.assertRaises(KeyError):
            self.connector.fetch_compressed(ids)

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            self.connector.upsert_compressed(
                ["a", "b"],
                np.zeros((3, 4), dtype=np.int8),
                np.zeros(3, dtype=np.float32),
            )

    def test_no_chromadb_raises_runtime_error(self):
        """Constructing without a client raises RuntimeError if chromadb absent."""
        import importlib
        import unittest.mock as mock

        original_import = importlib.import_module

        def _failing_import(name, *args, **kwargs):
            if name == "chromadb":
                raise ImportError("No module named 'chromadb'")
            return original_import(name, *args, **kwargs)

        with mock.patch("importlib.import_module", side_effect=_failing_import):
            with self.assertRaises(RuntimeError, msg="chromadb"):
                ChromaConnector("col")

    def test_shape_preserved_multi_row(self):
        rng = np.random.default_rng(13)
        n, d = 12, 8
        q = rng.integers(-128, 128, size=(n, d), dtype=np.int8)
        s = rng.random(n).astype(np.float32)
        ids = [f"s_{i}" for i in range(n)]
        self.connector.upsert_compressed(ids, q, s)
        batch = self.connector.fetch_compressed(ids)
        self.assertEqual(batch.quantized.shape, (n, d))

    def test_scales_values_accurate(self):
        ids = ["sc1"]
        q = np.array([[5, 6, 7]], dtype=np.int8)
        s = np.array([3.14159], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)
        batch = self.connector.fetch_compressed(ids)
        np.testing.assert_allclose(batch.scales.flatten(), s, rtol=1e-5)


class TestChromaConnectorRoundTrip(unittest.TestCase):
    """End-to-end data integrity tests."""

    def test_full_round_trip(self):
        rng = np.random.default_rng(17)
        n, d = 10, 20
        q = rng.integers(-127, 128, size=(n, d), dtype=np.int8)
        s = rng.random(n).astype(np.float32)
        ids = [f"rt_{i}" for i in range(n)]

        client = _FakeChromaClient()
        connector = ChromaConnector("rt_col", client=client)
        connector.upsert_compressed(ids, q, s)
        batch = connector.fetch_compressed(ids)

        np.testing.assert_array_equal(batch.quantized, q)
        np.testing.assert_allclose(batch.scales.flatten(), s, rtol=1e-5)

    def test_multiple_collections_isolated(self):
        client = _FakeChromaClient()
        c1 = ChromaConnector("col_alpha", client=client)
        c2 = ChromaConnector("col_beta", client=client)

        q = np.array([[1, 2, 3]], dtype=np.int8)
        s = np.array([0.5], dtype=np.float32)
        c1.upsert_compressed(["id1"], q, s)

        with self.assertRaises(KeyError):
            c2.fetch_compressed(["id1"])


if __name__ == "__main__":
    unittest.main()
