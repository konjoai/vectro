"""Tests for integration adapter primitives."""

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python import InMemoryVectorDBConnector  # noqa: E402


class TestInMemoryVectorConnector(unittest.TestCase):
    def setUp(self):
        self.connector = InMemoryVectorDBConnector()

    def test_upsert_fetch_delete_cycle(self):
        ids = ["a", "b", "c"]
        quantized = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
            ],
            dtype=np.int8,
        )
        scales = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        self.connector.upsert_compressed(ids, quantized, scales, metadata={"source": "test"})
        batch = self.connector.fetch_compressed(["a", "c"])

        self.assertEqual(batch.ids, ["a", "c"])
        self.assertEqual(batch.vector_dim, 4)
        self.assertEqual(batch.quantized.shape, (2, 4))
        self.assertEqual(batch.scales.shape, (2,))
        self.assertEqual(batch.metadata["source"], "test")

        deleted = self.connector.delete(["a", "x"])  # x does not exist
        self.assertEqual(deleted, 1)
        with self.assertRaises(KeyError):
            self.connector.fetch_compressed(["a"])

    def test_upsert_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.connector.upsert_compressed(
                ["a", "b"],
                np.array([[1, 2, 3, 4]], dtype=np.int8),
                np.array([0.1, 0.2], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
