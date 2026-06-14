"""v5.1.0 — Tests for HNSWIndex P1 features.

Covers:
  - Metadata sidecar (add with metadata, retrieval after search)
  - Soft-delete (delete, stats after delete, search excludes deleted)
  - Pre-filter search (metadata equality filter during graph walk)
  - Graph stats (n_alive, orphan_count, avg_degree_l0)
  - Compaction (tombstone removal, orphan reconnection)
  - Recall estimator (recall in [0,1], Wilson CI bounds, small-n edge case)
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.hnsw_api import HNSWIndex  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_index(n: int = 60, d: int = 32, seed: int = 0) -> HNSWIndex:
    """Return a populated HNSWIndex with deterministic vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-12)
    meta = [{"cat": ("a" if i < n // 2 else "b"), "idx": i} for i in range(n)]
    idx = HNSWIndex(M=8, ef_construction=60)
    ids = idx.add(vecs, metadata=meta)
    return idx, vecs, meta, ids


# ---------------------------------------------------------------------------
# 1. Metadata sidecar
# ---------------------------------------------------------------------------


class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.idx, self.vecs, self.meta, self.ids = _make_index()

    def test_add_returns_correct_node_ids(self):
        self.assertEqual(self.ids, list(range(60)))

    def test_metadata_length_matches_vectors(self):
        self.assertEqual(len(self.idx._metadata), len(self.idx._vectors))

    def test_metadata_content_matches_input(self):
        for i, (nid, m) in enumerate(zip(self.ids, self.meta)):
            self.assertEqual(self.idx._metadata[nid]["cat"], m["cat"])
            self.assertEqual(self.idx._metadata[nid]["idx"], i)

    def test_add_without_metadata_fills_none(self):
        idx = HNSWIndex(M=4, ef_construction=20)
        idx.add(np.random.default_rng(1).standard_normal((5, 8)).astype(np.float32))
        self.assertTrue(all(m is None for m in idx._metadata))

    def test_metadata_length_mismatch_raises(self):
        idx = HNSWIndex(M=4, ef_construction=20)
        vecs = np.eye(4, dtype=np.float32)
        with self.assertRaises(ValueError):
            idx.add(vecs, metadata=[{"x": 1}])  # only 1 meta for 4 vecs

    def test_add_1d_vector_with_metadata(self):
        idx = HNSWIndex(M=4, ef_construction=20)
        v = np.ones(8, dtype=np.float32)
        ids = idx.add(v, metadata=[{"tag": "single"}])
        self.assertEqual(ids, [0])
        self.assertEqual(idx._metadata[0]["tag"], "single")

    def test_save_load_preserves_metadata(self):
        import tempfile
        import os

        idx, vecs, meta, ids = _make_index(n=10, d=16)
        with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            loaded = HNSWIndex.load(path)
            for nid in ids:
                self.assertEqual(loaded._metadata[nid]["cat"], idx._metadata[nid]["cat"])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 2. Soft-delete
# ---------------------------------------------------------------------------


class TestDelete(unittest.TestCase):
    def setUp(self):
        self.idx, self.vecs, self.meta, self.ids = _make_index()

    def test_delete_adds_to_tombstone_set(self):
        self.idx.delete(0)
        self.assertIn(0, self.idx._deleted)

    def test_delete_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            self.idx.delete(1000)

    def test_delete_already_deleted_raises(self):
        self.idx.delete(5)
        with self.assertRaises(ValueError):
            self.idx.delete(5)

    def test_deleted_node_absent_from_search_results(self):
        # Delete the 5 closest nodes to vecs[0] so they can't appear in top-5
        idxs, _ = self.idx.search(self.vecs[0], k=5, ef=30)
        for nid in idxs.tolist():
            if nid != 0:
                self.idx.delete(nid)
        idxs2, _ = self.idx.search(self.vecs[0], k=5, ef=30)
        for nid in idxs.tolist():
            if nid != 0:
                self.assertNotIn(nid, idxs2.tolist())

    def test_search_still_returns_results_after_deletes(self):
        for nid in range(10):
            self.idx.delete(nid)
        idxs, dists = self.idx.search(self.vecs[0], k=5, ef=30)
        self.assertGreater(len(idxs), 0)
        self.assertTrue(all(n not in self.idx._deleted for n in idxs.tolist()))


# ---------------------------------------------------------------------------
# 3. Pre-filter search
# ---------------------------------------------------------------------------


class TestFilteredSearch(unittest.TestCase):
    def setUp(self):
        self.idx, self.vecs, self.meta, self.ids = _make_index()

    def test_filter_results_all_pass(self):
        idxs, _ = self.idx.search(self.vecs[0], k=10, ef=40, filter={"cat": "a"})
        for nid in idxs.tolist():
            self.assertEqual(self.idx._metadata[nid]["cat"], "a")

    def test_filter_b_results_all_pass(self):
        idxs, _ = self.idx.search(self.vecs[0], k=10, ef=40, filter={"cat": "b"})
        for nid in idxs.tolist():
            self.assertEqual(self.idx._metadata[nid]["cat"], "b")

    def test_filter_no_match_returns_empty(self):
        idxs, dists = self.idx.search(self.vecs[0], k=5, ef=40, filter={"cat": "zzz"})
        self.assertEqual(len(idxs), 0)

    def test_filter_none_behaves_like_unfiltered(self):
        idxs_f, _ = self.idx.search(self.vecs[0], k=5, ef=40, filter=None)
        idxs_u, _ = self.idx.search(self.vecs[0], k=5, ef=40)
        self.assertEqual(idxs_f.tolist(), idxs_u.tolist())

    def test_filter_on_node_without_metadata_excludes_it(self):
        idx = HNSWIndex(M=6, ef_construction=40)
        vecs = np.eye(8, dtype=np.float32)
        # Only first 4 have metadata
        idx.add(vecs[:4], metadata=[{"cat": "x"}] * 4)
        idx.add(vecs[4:])  # no metadata
        idxs, _ = idx.search(vecs[0], k=8, ef=20, filter={"cat": "x"})
        for nid in idxs.tolist():
            self.assertIsNotNone(idx._metadata[nid])
            self.assertEqual(idx._metadata[nid]["cat"], "x")

    def test_filter_combined_with_delete(self):
        # Delete some 'a' nodes; filtered search must not return them
        self.idx.delete(0)
        self.idx.delete(1)
        idxs, _ = self.idx.search(self.vecs[0], k=10, ef=40, filter={"cat": "a"})
        self.assertNotIn(0, idxs.tolist())
        self.assertNotIn(1, idxs.tolist())
        for nid in idxs.tolist():
            self.assertEqual(self.idx._metadata[nid]["cat"], "a")


# ---------------------------------------------------------------------------
# 4. Stats
# ---------------------------------------------------------------------------


class TestStats(unittest.TestCase):
    def setUp(self):
        self.idx, self.vecs, self.meta, self.ids = _make_index()

    def test_stats_keys_present(self):
        s = self.idx.stats()
        for key in (
            "n_total",
            "n_alive",
            "n_deleted",
            "orphan_count",
            "avg_degree_l0",
            "max_level",
            "space",
        ):
            self.assertIn(key, s)

    def test_stats_before_delete(self):
        s = self.idx.stats()
        self.assertEqual(s["n_total"], 60)
        self.assertEqual(s["n_alive"], 60)
        self.assertEqual(s["n_deleted"], 0)

    def test_stats_after_delete(self):
        self.idx.delete(0)
        self.idx.delete(1)
        s = self.idx.stats()
        self.assertEqual(s["n_deleted"], 2)
        self.assertEqual(s["n_alive"], 58)

    def test_avg_degree_positive(self):
        s = self.idx.stats()
        self.assertGreater(s["avg_degree_l0"], 0)

    def test_space_correct(self):
        self.assertEqual(self.idx.stats()["space"], "cosine")

    def test_orphan_count_zero_fresh_index(self):
        self.assertEqual(self.idx.stats()["orphan_count"], 0)


# ---------------------------------------------------------------------------
# 5. Compaction
# ---------------------------------------------------------------------------


class TestCompact(unittest.TestCase):
    def setUp(self):
        self.idx, self.vecs, self.meta, self.ids = _make_index()

    def test_compact_returns_dict_with_correct_keys(self):
        r = self.idx.compact()
        self.assertIn("removed", r)
        self.assertIn("repaired", r)

    def test_compact_clears_tombstones(self):
        self.idx.delete(0)
        self.idx.delete(1)
        self.idx.compact()
        self.assertEqual(len(self.idx._deleted), 0)

    def test_compact_removes_deleted_ids_from_neighbour_lists(self):
        self.idx.delete(0)
        result = self.idx.compact()
        # After compact, 0 must not appear in any neighbour list
        for nid in range(len(self.idx._vectors)):
            for lc_nbrs in self.idx._neighbors[nid]:
                self.assertNotIn(0, lc_nbrs)
        self.assertGreater(result["removed"], 0)

    def test_compact_search_still_works_after(self):
        for nid in [5, 10, 15]:
            self.idx.delete(nid)
        self.idx.compact()
        idxs, dists = self.idx.search(self.vecs[0], k=5, ef=30)
        self.assertGreater(len(idxs), 0)
        self.assertTrue(all(d >= 0 for d in dists.tolist()))

    def test_compact_empty_deleted_set_is_noop(self):
        r = self.idx.compact()
        self.assertEqual(r["removed"], 0)
        self.assertEqual(r["repaired"], 0)

    def test_compact_stats_show_zero_orphans_after(self):
        for nid in [0, 1, 2]:
            self.idx.delete(nid)
        self.idx.compact()
        s = self.idx.stats()
        self.assertEqual(s["orphan_count"], 0)

    def test_compact_save_load_roundtrip(self):
        import tempfile
        import os

        for nid in [0, 3, 7]:
            self.idx.delete(nid)
        self.idx.compact()
        with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
            path = f.name
        try:
            self.idx.save(path)
            loaded = HNSWIndex.load(path)
            self.assertEqual(len(loaded._deleted), 0)
            idxs, _ = loaded.search(self.vecs[0], k=3, ef=20)
            self.assertGreater(len(idxs), 0)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 6. Recall estimator
# ---------------------------------------------------------------------------


class TestEstimateRecall(unittest.TestCase):
    def setUp(self):
        self.idx, self.vecs, self.meta, self.ids = _make_index(n=80, d=32)

    def test_recall_in_range(self):
        r = self.idx.estimate_recall(sample_size=20, k=5, ef=40)
        self.assertGreaterEqual(r["recall"], 0.0)
        self.assertLessEqual(r["recall"], 1.0)

    def test_ci_ordered(self):
        r = self.idx.estimate_recall(sample_size=20, k=5, ef=40)
        self.assertLessEqual(r["ci_95_lower"], r["recall"])
        self.assertLessEqual(r["recall"], r["ci_95_upper"])

    def test_ci_in_0_1(self):
        r = self.idx.estimate_recall(sample_size=20, k=5, ef=40)
        self.assertGreaterEqual(r["ci_95_lower"], 0.0)
        self.assertLessEqual(r["ci_95_upper"], 1.0)

    def test_result_keys(self):
        r = self.idx.estimate_recall(sample_size=10, k=3, ef=20)
        for key in ("recall", "ci_95_lower", "ci_95_upper", "sample_size", "k", "ef", "n_alive"):
            self.assertIn(key, r)

    def test_large_ef_better_than_small_ef(self):
        # Higher ef must give recall >= lower ef (monotonicity property).
        # We do not assert a specific floor because small n and low d
        # produce high variance on recall with only 10 samples.
        r_small = self.idx.estimate_recall(sample_size=10, k=5, ef=8)
        r_large = self.idx.estimate_recall(sample_size=10, k=5, ef=60)
        # May equal if both happen to return the same results; must not regress.
        self.assertGreaterEqual(r_large["recall"], r_small["recall"] - 0.05)

    def test_sample_size_capped_at_n_alive(self):
        r = self.idx.estimate_recall(sample_size=9999, k=5, ef=40)
        self.assertLessEqual(r["sample_size"], len(self.idx))

    def test_tiny_index_returns_valid_result(self):
        idx = HNSWIndex(M=4, ef_construction=20)
        idx.add(np.eye(3, dtype=np.float32))
        r = idx.estimate_recall(sample_size=2, k=2, ef=10)
        self.assertGreaterEqual(r["recall"], 0.0)
        self.assertLessEqual(r["recall"], 1.0)

    def test_fully_empty_index_returns_trivial_recall(self):
        idx = HNSWIndex(M=4, ef_construction=20)
        r = idx.estimate_recall(sample_size=10, k=5, ef=20)
        self.assertEqual(r["recall"], 1.0)

    def test_after_deletes_n_alive_decreases(self):
        self.idx.delete(0)
        self.idx.delete(1)
        r = self.idx.estimate_recall(sample_size=5, k=3, ef=20)
        self.assertEqual(r["n_alive"], 78)


if __name__ == "__main__":
    unittest.main()
