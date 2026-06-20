"""Tests for IVFIndex and IVFPQIndex Python wrappers (v7.0.0).

Coverage:
- shape / dtype contracts
- numerical correctness (recall@1 with exact query)
- IVFPQIndex train extra args (n_subspaces, n_centroids)
- save / load round-trip
- delete / vacuum
- search_filtered_np (IVFIndex only)
- search_for_recall
- failure cases: untrained search, wrong path
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

try:
    from python.ivf_api import IVFIndex, IVFPQIndex, _BINDINGS_AVAILABLE
except ImportError:
    from vectro.ivf_api import IVFIndex, IVFPQIndex, _BINDINGS_AVAILABLE  # type: ignore


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_vectors(n: int = 256, d: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    # Normalise to unit sphere for cosine-based retrieval
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


# ---------------------------------------------------------------------------
# IVFIndex — unit tests (mocked bindings)
# ---------------------------------------------------------------------------


class TestIVFIndexUnit:
    """Pure-unit tests — mock _PyIvfIndex to avoid native-build requirement."""

    def _make_ivf(self):
        with (
            patch("python.ivf_api._BINDINGS_AVAILABLE", True),
            patch("python.ivf_api._PyIvfIndex") as MockClass,
        ):
            mock_instance = MagicMock()
            MockClass.return_value = mock_instance
            idx = IVFIndex.__new__(IVFIndex)
            idx._inner = mock_instance
            idx._count = 0
            return idx, mock_instance

    def test_train_delegates(self):
        idx, inner = self._make_ivf()
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        idx.train(vecs, max_iter=50, seed=7)
        inner.train.assert_called_once_with(vecs, 50, 7)

    def test_add_returns_id(self):
        idx, inner = self._make_ivf()
        inner.add.return_value = 42
        assert idx.add([0.1, 0.2]) == 42

    def test_search_delegates(self):
        idx, inner = self._make_ivf()
        inner.search.return_value = [(0, 0.01)]
        result = idx.search([0.1, 0.2], k=1)
        assert result == [(0, 0.01)]
        inner.search.assert_called_once_with([0.1, 0.2], 1)

    def test_delete_and_vacuum(self):
        idx, inner = self._make_ivf()
        inner.vacuum.return_value = 3
        idx.delete(5)
        inner.delete.assert_called_once_with(5)
        count = idx.vacuum()
        assert count == 3

    def test_repr_delegates(self):
        idx, inner = self._make_ivf()
        inner.__repr__ = lambda _: "IVFIndex(n_lists=8)"
        assert repr(idx) == "IVFIndex(n_lists=8)"

    def test_train_np_casts_to_float32(self):
        idx, inner = self._make_ivf()
        arr = np.ones((10, 4), dtype=np.float64)  # float64 — should be cast
        idx.train_np(arr)
        call_arr = inner.train_np.call_args[0][0]
        assert call_arr.dtype == np.float32

    def test_add_np_casts_to_float32(self):
        idx, inner = self._make_ivf()
        arr = np.ones((5, 4), dtype=np.float64)
        idx.add_np(arr)
        call_arr = inner.add_np.call_args[0][0]
        assert call_arr.dtype == np.float32

    def test_search_np_casts_to_float32(self):
        idx, inner = self._make_ivf()
        inner.search_np.return_value = []
        q = np.ones(4, dtype=np.float64)
        idx.search_np(q, k=1)
        call_arr = inner.search_np.call_args[0][0]
        assert call_arr.dtype == np.float32

    def test_search_filtered_np_casts_to_float32(self):
        idx, inner = self._make_ivf()
        inner.search_filtered_np.return_value = []
        q = np.ones(4, dtype=np.float64)
        idx.search_filtered_np(q, k=2, allowed_ids=[0, 1])
        call_arr = inner.search_filtered_np.call_args[0][0]
        assert call_arr.dtype == np.float32

    def test_search_for_recall_delegates(self):
        idx, inner = self._make_ivf()
        inner.search_for_recall.return_value = ([(0, 0.0)], 8)
        result, n_probe_used = idx.search_for_recall([0.1, 0.2], k=1, target_recall=0.95)
        assert n_probe_used == 8


# ---------------------------------------------------------------------------
# IVFPQIndex — unit tests (mocked bindings)
# ---------------------------------------------------------------------------


class TestIVFPQIndexUnit:
    def _make_ivfpq(self):
        with (
            patch("python.ivf_api._BINDINGS_AVAILABLE", True),
            patch("python.ivf_api._PyIvfPqIndex") as MockClass,
        ):
            mock_instance = MagicMock()
            MockClass.return_value = mock_instance
            idx = IVFPQIndex.__new__(IVFPQIndex)
            idx._inner = mock_instance
            return idx, mock_instance

    def test_train_passes_extra_args(self):
        idx, inner = self._make_ivfpq()
        vecs = [[0.1] * 16]
        idx.train(vecs, n_subspaces=4, n_centroids=256, max_iter=50, seed=1)
        inner.train.assert_called_once_with(vecs, 4, 256, 50, 1)

    def test_train_np_passes_extra_args(self):
        idx, inner = self._make_ivfpq()
        arr = np.ones((8, 16), dtype=np.float32)
        idx.train_np(arr, n_subspaces=4, n_centroids=256)
        call = inner.train_np.call_args[0]
        assert call[1] == 4
        assert call[2] == 256

    def test_add_delegates(self):
        idx, inner = self._make_ivfpq()
        inner.add.return_value = 0
        assert idx.add([0.1] * 4) == 0

    def test_search_for_recall_delegates(self):
        idx, inner = self._make_ivfpq()
        inner.search_for_recall.return_value = ([(0, 0.1)], 4)
        results, n = idx.search_for_recall([0.1] * 4, k=1, target_recall=0.9)
        assert n == 4

    def test_delete_vacuum(self):
        idx, inner = self._make_ivfpq()
        inner.vacuum.return_value = 1
        idx.delete(0)
        inner.delete.assert_called_once_with(0)
        assert idx.vacuum() == 1


# ---------------------------------------------------------------------------
# Import-guard failure tests (no bindings)
# ---------------------------------------------------------------------------


class TestBindingsGuard:
    def test_ivfindex_raises_without_bindings(self):
        with patch("python.ivf_api._BINDINGS_AVAILABLE", False):
            with pytest.raises(ImportError, match="vectro_py"):
                IVFIndex(8, 4)

    def test_ivfpqindex_raises_without_bindings(self):
        with patch("python.ivf_api._BINDINGS_AVAILABLE", False):
            with pytest.raises(ImportError, match="vectro_py"):
                IVFPQIndex(8, 4)

    def test_ivfindex_load_raises_without_bindings(self):
        with patch("python.ivf_api._BINDINGS_AVAILABLE", False):
            with pytest.raises(ImportError, match="vectro_py"):
                IVFIndex.load("/nonexistent/path")

    def test_ivfpqindex_load_raises_without_bindings(self):
        with patch("python.ivf_api._BINDINGS_AVAILABLE", False):
            with pytest.raises(ImportError, match="vectro_py"):
                IVFPQIndex.load("/nonexistent/path")


# ---------------------------------------------------------------------------
# Integration tests (skipped if native bindings unavailable)
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.skipif(
    not _BINDINGS_AVAILABLE,
    reason="vectro_py native bindings not available; run `maturin develop` first",
)


@pytestmark_integration
class TestIVFIndexIntegration:
    DIM = 32
    N = 512
    N_LISTS = 16
    N_PROBE = 4

    def setup_method(self):
        self.vecs = _make_vectors(self.N, self.DIM)
        self.idx = IVFIndex(self.N_LISTS, self.N_PROBE)
        self.idx.train_np(self.vecs, max_iter=20, seed=42)
        self.idx.add_np(self.vecs)

    def test_search_returns_correct_shape(self):
        results = self.idx.search_np(self.vecs[0], k=5)
        assert len(results) == 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_recall_at_1_with_exact_query(self):
        """Exact query of an indexed vector should retrieve itself as top-1."""
        for i in range(0, self.N, 64):
            results = self.idx.search_np(self.vecs[i], k=1)
            top_id, _ = results[0]
            assert top_id == i, f"Expected id {i}, got {top_id}"

    def test_delete_and_vacuum_reduce_size(self):
        count_before = len(self.idx) if hasattr(self.idx, "__len__") else None
        self.idx.delete(0)
        removed = self.idx.vacuum()
        assert removed >= 1
        if count_before is not None:
            assert len(self.idx) == count_before - 1

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ivf.bin")
            self.idx.save(path)
            loaded = IVFIndex.load(path)
            results_orig = self.idx.search_np(self.vecs[10], k=3)
            results_loaded = loaded.search_np(self.vecs[10], k=3)
            ids_orig = [r[0] for r in results_orig]
            ids_loaded = [r[0] for r in results_loaded]
            assert ids_orig == ids_loaded, "Loaded index returns different results"

    def test_search_filtered(self):
        allowed = list(range(0, self.N, 2))  # even-indexed only
        results = self.idx.search_filtered_np(self.vecs[4], k=5, allowed_ids=allowed)
        returned_ids = [r[0] for r in results]
        assert all(rid in allowed for rid in returned_ids)


@pytestmark_integration
class TestIVFPQIndexIntegration:
    DIM = 32
    N = 512
    N_LISTS = 8
    N_PROBE = 2
    N_SUBSPACES = 4
    N_CENTROIDS = 64

    def setup_method(self):
        self.vecs = _make_vectors(self.N, self.DIM)
        self.idx = IVFPQIndex(self.N_LISTS, self.N_PROBE)
        self.idx.train_np(
            self.vecs,
            n_subspaces=self.N_SUBSPACES,
            n_centroids=self.N_CENTROIDS,
            max_iter=20,
            seed=42,
        )
        self.idx.add_np(self.vecs)

    def test_search_returns_k_results(self):
        results = self.idx.search_np(self.vecs[0], k=10)
        assert 1 <= len(results) <= 10

    def test_recall_at_5_above_threshold(self):
        """PQ approximation: top-5 should contain true neighbour ≥60% of time."""
        hits = 0
        n_probe = 50
        for i in range(0, min(self.N, 100)):
            results = self.idx.search_with_probe(list(self.vecs[i]), k=5, n_probe=n_probe)
            ids = [r[0] for r in results]
            if i in ids:
                hits += 1
        recall = hits / 100
        assert recall >= 0.6, f"Recall@5 too low: {recall:.2f}"

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ivfpq.bin")
            self.idx.save(path)
            loaded = IVFPQIndex.load(path)
            r1 = self.idx.search_np(self.vecs[0], k=3)
            r2 = loaded.search_np(self.vecs[0], k=3)
            assert [x[0] for x in r1] == [x[0] for x in r2]

    def test_delete_vacuum(self):
        self.idx.delete(0)
        removed = self.idx.vacuum()
        assert removed >= 1
