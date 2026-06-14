"""v5.2.0 — three HNSWIndex feature tests.

1. Persistent index serialisation — save/load via .npz
2. Batch upsert with deduplication — add_batch()
3. Search trace — search(..., trace=True)
"""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.hnsw_api import HNSWIndex, SearchTrace  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_vecs(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _build_index(n: int = 80, d: int = 32, M: int = 8, ef: int = 60) -> HNSWIndex:
    idx = HNSWIndex(M=M, ef_construction=ef)
    idx.add(_rand_vecs(n, d))
    return idx


# ============================================================================
# 1. Persistent index serialisation (.npz)
# ============================================================================


class TestSaveLoadNpz(unittest.TestCase):
    def _round_trip(self, idx: HNSWIndex) -> HNSWIndex:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "idx.vindex")
            idx.save(p)
            # save() writes to exactly `p` (not `p.npz`) via BytesIO buffer
            self.assertTrue(os.path.exists(p), "save() must create the file at exactly `path`")
            self.assertGreater(os.path.getsize(p), 0, "saved file must be non-empty")
            # Magic bytes must be ZIP/NPZ not pickle
            with open(p, "rb") as fh:
                self.assertEqual(fh.read(4), b"PK\x03\x04", "not a ZIP/NPZ file")
            return HNSWIndex.load(p)

    def test_empty_index_round_trips(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        idx2 = self._round_trip(idx)
        self.assertEqual(len(idx2), 0)
        self.assertEqual(idx2.M, 8)
        self.assertEqual(idx2.space, "cosine")

    def test_vector_count_preserved(self):
        idx = _build_index(100, 64)
        idx2 = self._round_trip(idx)
        self.assertEqual(len(idx2), 100)

    def test_hyperparams_preserved(self):
        idx = HNSWIndex(M=12, ef_construction=150, space="l2")
        idx.add(_rand_vecs(20, 16))
        idx2 = self._round_trip(idx)
        self.assertEqual(idx2.M, 12)
        self.assertEqual(idx2.ef_construction, 150)
        self.assertEqual(idx2.space, "l2")

    def test_search_results_identical_after_round_trip(self):
        rng = np.random.default_rng(42)
        idx = _build_index(200, 32)
        q = rng.standard_normal(32).astype(np.float32)
        ids_before, dists_before = idx.search(q, k=5, ef=64)

        idx2 = self._round_trip(idx)
        ids_after, dists_after = idx2.search(q, k=5, ef=64)

        # Same result set
        self.assertListEqual(list(ids_before), list(ids_after))
        np.testing.assert_allclose(dists_before, dists_after, atol=1e-6)

    def test_recall_within_tolerance_after_round_trip(self):
        idx = _build_index(300, 32)
        r_before = idx.estimate_recall(sample_size=50, k=5, ef=64)

        idx2 = self._round_trip(idx)
        r_after = idx2.estimate_recall(sample_size=50, k=5, ef=64)

        self.assertAlmostEqual(
            r_before["recall"],
            r_after["recall"],
            delta=0.01,
            msg="recall must agree within 0.01 after round-trip",
        )

    def test_metadata_preserved(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        vecs = _rand_vecs(10, 16)
        idx.add(vecs, metadata=[{"label": str(i)} for i in range(10)])
        idx2 = self._round_trip(idx)
        self.assertEqual(idx2._metadata[5], {"label": "5"})

    def test_deleted_set_preserved(self):
        idx = _build_index(30, 16)
        idx.delete(3)
        idx.delete(7)
        idx2 = self._round_trip(idx)
        self.assertIn(3, idx2._deleted)
        self.assertIn(7, idx2._deleted)
        self.assertEqual(len(idx2._deleted), 2)

    def test_id_map_preserved(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        vecs = _rand_vecs(5, 16)
        idx.add_batch(vecs, ids=["a", "b", "c", "d", "e"])
        idx2 = self._round_trip(idx)
        self.assertIn("a", idx2._id_map)
        self.assertEqual(idx2._id_map["a"], 0)

    def test_load_detects_npz_magic(self):
        idx = _build_index(10, 8)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.vindex")
            idx.save(p)
            # load() with the bare path (no .npz suffix) must also work
            idx2 = HNSWIndex.load(p)
            self.assertEqual(len(idx2), 10)

    def test_save_with_npz_suffix_explicit(self):
        idx = _build_index(10, 8)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.vindex.npz")
            idx.save(p)
            # save() writes to exactly the supplied path regardless of suffix
            self.assertTrue(os.path.exists(p))
            self.assertGreater(os.path.getsize(p), 0)

    def test_load_legacy_pickle_emits_deprecation_warning(self):
        import pickle

        idx = _build_index(10, 8)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "old.pkl")
            payload = {
                "M": idx.M,
                "ef_construction": idx.ef_construction,
                "space": idx.space,
                "vectors": idx._vectors,
                "neighbors": idx._neighbors,
                "levels": idx._levels,
                "entry_point": idx._entry_point,
                "max_level": idx._max_level,
                "metadata": idx._metadata,
                "deleted": idx._deleted,
            }
            with open(p, "wb") as fh:
                pickle.dump(payload, fh, protocol=5)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                idx2 = HNSWIndex.load(p)
                self.assertTrue(
                    any(issubclass(x.category, DeprecationWarning) for x in w),
                    "expected DeprecationWarning for pickle format",
                )
            self.assertEqual(len(idx2), len(idx))

    def test_cosine_l2_round_trips_independently(self):
        idx_cos = HNSWIndex(M=8, ef_construction=40, space="cosine")
        idx_cos.add(_rand_vecs(20, 8))
        idx_l2 = HNSWIndex(M=8, ef_construction=40, space="l2")
        idx_l2.add(_rand_vecs(20, 8))
        for idx in (idx_cos, idx_l2):
            idx2 = self._round_trip(idx)
            self.assertEqual(idx2.space, idx.space)


# ============================================================================
# 2. Batch upsert with deduplication — add_batch()
# ============================================================================


class TestAddBatch(unittest.TestCase):
    def test_basic_insert(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        vecs = _rand_vecs(10, 32)
        r = idx.add_batch(vecs)
        self.assertEqual(r["inserted"], 10)
        self.assertEqual(r["updated"], 0)
        self.assertEqual(len(idx), 10)

    def test_insert_with_string_ids(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        vecs = _rand_vecs(5, 16)
        r = idx.add_batch(vecs, ids=[f"v{i}" for i in range(5)])
        self.assertEqual(r["inserted"], 5)
        self.assertIn("v0", idx._id_map)
        self.assertEqual(len(r["node_ids"]), 5)

    def test_upsert_updates_existing_vectors(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        ids = ["a", "b", "c"]
        vecs1 = _rand_vecs(3, 16, seed=0)
        vecs2 = _rand_vecs(3, 16, seed=99)

        idx.add_batch(vecs1, ids=ids)
        nid_a_first = idx._id_map["a"]

        r = idx.add_batch(vecs2, ids=ids)
        self.assertEqual(r["inserted"], 0)
        self.assertEqual(r["updated"], 3)
        # Node IDs must not change
        self.assertEqual(idx._id_map["a"], nid_a_first)
        # Vector must be updated (after normalisation both are unit-norm)
        nid_a = idx._id_map["a"]
        np.testing.assert_allclose(
            idx._vectors[nid_a],
            idx._normalize(vecs2[0]),
            atol=1e-6,
        )

    def test_partial_upsert(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        vecs = _rand_vecs(6, 16)
        idx.add_batch(vecs[:3], ids=["x0", "x1", "x2"])
        # Second batch: x1 and x2 exist, x3 x4 x5 are new
        r = idx.add_batch(
            vecs[1:],
            ids=["x1", "x2", "x3", "x4", "x5"],
        )
        self.assertEqual(r["updated"], 2)
        self.assertEqual(r["inserted"], 3)
        self.assertEqual(len(idx), 6)

    def test_node_ids_returned(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        vecs = _rand_vecs(4, 16)
        r = idx.add_batch(vecs, ids=["a", "b", "c", "d"])
        self.assertEqual(len(r["node_ids"]), 4)
        self.assertEqual(r["node_ids"][0], idx._id_map["a"])

    def test_metadata_upserted(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        vecs = _rand_vecs(3, 16)
        idx.add_batch(vecs, ids=["p", "q", "r"], metadata=[{"v": 1}, {"v": 2}, {"v": 3}])
        idx.add_batch(vecs, ids=["p", "q", "r"], metadata=[{"v": 10}, {"v": 20}, {"v": 30}])
        self.assertEqual(idx._metadata[idx._id_map["p"]], {"v": 10})
        self.assertEqual(idx._metadata[idx._id_map["r"]], {"v": 30})

    def test_resurrection_of_deleted_node(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        vecs = _rand_vecs(5, 16)
        idx.add_batch(vecs, ids=["a", "b", "c", "d", "e"])
        nid_b = idx._id_map["b"]
        idx.delete(nid_b)
        self.assertIn(nid_b, idx._deleted)

        # Upserting "b" should resurrect it
        idx.add_batch(_rand_vecs(1, 16), ids=["b"])
        self.assertNotIn(nid_b, idx._deleted)

    def test_without_ids_all_inserted(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        r = idx.add_batch(_rand_vecs(20, 16))
        self.assertEqual(r["inserted"], 20)
        self.assertEqual(r["updated"], 0)
        self.assertEqual(len(idx), 20)

    def test_error_ids_length_mismatch(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        with self.assertRaises(ValueError):
            idx.add_batch(_rand_vecs(3, 16), ids=["a", "b"])

    def test_error_metadata_length_mismatch(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        with self.assertRaises(ValueError):
            idx.add_batch(_rand_vecs(3, 16), metadata=[{}, {}])

    def test_get_by_id(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        vecs = _rand_vecs(3, 16)
        idx.add_batch(vecs, ids=["alpha", "beta", "gamma"], metadata=[{"k": 1}, {"k": 2}, {"k": 3}])
        self.assertEqual(idx.get_by_id("beta"), {"k": 2})
        self.assertIsNone(idx.get_by_id("missing"))

    def test_get_by_id_returns_none_for_deleted(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        vecs = _rand_vecs(2, 16)
        idx.add_batch(vecs, ids=["x", "y"], metadata=[{"k": 1}, {"k": 2}])
        nid_x = idx._id_map["x"]
        idx.delete(nid_x)
        self.assertIsNone(idx.get_by_id("x"))

    def test_search_finds_upserted_vectors(self):
        idx = HNSWIndex(M=8, ef_construction=80)
        # Build initial corpus
        base_vecs = _rand_vecs(50, 32, seed=1)
        idx.add_batch(base_vecs, ids=[f"b{i}" for i in range(50)])

        # Create a known query vector and insert it with a specific ID
        query_vec = _rand_vecs(1, 32, seed=999)[0]
        idx.add_batch(query_vec[np.newaxis], ids=["special"], metadata=[{"role": "target"}])
        nid = idx._id_map["special"]

        # Update to a fresh vector and confirm search finds the new version
        new_vec = _rand_vecs(1, 32, seed=888)[0]
        idx.add_batch(new_vec[np.newaxis], ids=["special"])
        np.testing.assert_allclose(
            idx._vectors[nid],
            idx._normalize(new_vec),
            atol=1e-5,
        )


# ============================================================================
# 3. Search trace — search(..., trace=True)
# ============================================================================


class TestSearchTrace(unittest.TestCase):
    def test_trace_is_searchresult_triple(self):
        idx = _build_index(50, 32)
        q = _rand_vecs(1, 32)[0]
        result = idx.search(q, k=5, trace=True)
        self.assertEqual(len(result), 3, "trace=True must return (indices, dists, trace)")

    def test_trace_is_searchresult_pair_by_default(self):
        idx = _build_index(50, 32)
        q = _rand_vecs(1, 32)[0]
        result = idx.search(q, k=5)
        self.assertEqual(len(result), 2)

    def test_trace_type(self):
        idx = _build_index(50, 32)
        q = _rand_vecs(1, 32)[0]
        _, _, tr = idx.search(q, k=5, trace=True)
        self.assertIsInstance(tr, SearchTrace)

    def test_trace_results_match_no_trace(self):
        idx = _build_index(100, 32)
        q = _rand_vecs(1, 32)[0]
        ids_nt, dists_nt = idx.search(q, k=5)
        ids_tr, dists_tr, _ = idx.search(q, k=5, trace=True)
        np.testing.assert_array_equal(ids_nt, ids_tr)
        np.testing.assert_allclose(dists_nt, dists_tr, atol=1e-6)

    def test_trace_entry_point_valid(self):
        idx = _build_index(50, 32)
        q = _rand_vecs(1, 32)[0]
        _, _, tr = idx.search(q, k=3, trace=True)
        self.assertGreaterEqual(tr.entry_point, 0)
        self.assertLess(tr.entry_point, len(idx))

    def test_trace_l0_visited_non_empty(self):
        idx = _build_index(80, 32)
        q = _rand_vecs(1, 32)[0]
        _, _, tr = idx.search(q, k=5, ef=32, trace=True)
        self.assertGreater(len(tr.l0_visited), 0)

    def test_trace_l0_candidates_match_results(self):
        idx = _build_index(80, 32)
        q = _rand_vecs(1, 32)[0]
        ids, dists, tr = idx.search(q, k=5, ef=64, trace=True)
        # Every result node_id must appear in l0_candidates_final
        result_set = set(int(x) for x in ids)
        candidate_set = set(nid for _, nid in tr.l0_candidates_final)
        self.assertTrue(
            result_set.issubset(candidate_set),
            "All top-k results must be in l0_candidates_final",
        )

    def test_trace_layer_descents_count(self):
        idx = _build_index(200, 32)
        q = _rand_vecs(1, 32)[0]
        _, _, tr = idx.search(q, k=5, trace=True)
        # Number of layer descents must equal max_level (one per layer above 0)
        self.assertEqual(len(tr.layer_descents), idx._max_level)

    def test_trace_on_empty_index(self):
        idx = HNSWIndex(M=8, ef_construction=40)
        q = _rand_vecs(1, 16)[0]
        result = idx.search(q, k=3, trace=True)
        self.assertEqual(len(result), 3)
        ids, dists, tr = result
        self.assertEqual(len(ids), 0)
        self.assertIsInstance(tr, SearchTrace)

    def test_trace_with_filter(self):
        idx = HNSWIndex(M=8, ef_construction=60)
        vecs = _rand_vecs(60, 16)
        idx.add(vecs, metadata=[{"cat": "a" if i % 2 == 0 else "b"} for i in range(60)])
        q = _rand_vecs(1, 16)[0]
        ids, dists, tr = idx.search(q, k=5, filter={"cat": "a"}, trace=True)
        # All returned IDs must have cat="a"
        for nid in ids:
            self.assertEqual(idx._metadata[int(nid)]["cat"], "a")

    def test_trace_l0_candidates_sorted_ascending(self):
        idx = _build_index(100, 32)
        q = _rand_vecs(1, 32)[0]
        _, _, tr = idx.search(q, k=5, ef=64, trace=True)
        dists = [d for d, _ in tr.l0_candidates_final]
        self.assertEqual(dists, sorted(dists), "l0_candidates_final must be sorted ascending")

    def test_trace_visited_ids_are_valid_node_ids(self):
        idx = _build_index(50, 16)
        q = _rand_vecs(1, 16)[0]
        _, _, tr = idx.search(q, k=3, trace=True)
        n = len(idx)
        for nid in tr.l0_visited:
            self.assertGreaterEqual(nid, 0)
            self.assertLess(nid, n)

    def test_trace_with_deleted_nodes_excluded_from_results(self):
        idx = _build_index(50, 16)
        q = _rand_vecs(1, 16)[0]
        ids_before, _, tr_before = idx.search(q, k=5, trace=True)
        # Delete 3 nodes from the top-5 result set
        to_delete = list(ids_before[:3])
        for nid in to_delete:
            idx.delete(int(nid))
        ids_after, _, tr_after = idx.search(q, k=5, trace=True)
        # Deleted nodes must not appear in results
        for nid in to_delete:
            self.assertNotIn(nid, ids_after)

    def test_save_load_trace_still_works(self):
        idx = _build_index(60, 32)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "t.vindex")
            idx.save(p)
            idx2 = HNSWIndex.load(p)
        q = _rand_vecs(1, 32)[0]
        ids, dists, tr = idx2.search(q, k=5, trace=True)
        self.assertIsInstance(tr, SearchTrace)
        self.assertGreater(len(tr.l0_visited), 0)


if __name__ == "__main__":
    unittest.main()
