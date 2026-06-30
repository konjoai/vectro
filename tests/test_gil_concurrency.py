"""GIL-release regression tests (WS-B, perf sprint).

``train``/``train_np``/``add``/``add_batch``/``add_np`` on the IVF, IVF-PQ,
HNSW and PQ-codebook PyO3 paths wrap their heavy compute in
``py.allow_threads`` (see ``rust/vectro_py/src/lib.rs``), so several Python
threads building indexes at once run their Rust work truly in parallel
instead of serialising on the GIL.

These tests assert:
  1. concurrent builds stay numerically correct, and
  2. they overlap in wall-clock — a held GIL would make the concurrent run
     no faster than the serial sum.

The speedup test deliberately uses the HNSW *serial* build path (a batch
below ``PARALLEL_BUILD_THRESHOLD`` so the Rust core does not itself spin up
rayon) — otherwise the index's internal parallelism would already saturate
the cores and mask the GIL-release effect.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

try:
    import vectro_py
except ImportError:  # pragma: no cover - extension not built
    vectro_py = None

pytestmark = pytest.mark.skipif(vectro_py is None, reason="vectro_py extension not built")

# Keep batches under the Rust PARALLEL_BUILD_THRESHOLD (512) so each individual
# build stays single-threaded and the only parallelism is across Python threads.
_SERIAL_N = 500


def _unit_data(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def _build_hnsw(data: np.ndarray):
    h = vectro_py.PyHnswIndex(32, 400, "cosine")
    h.add_np(data)
    return h


def test_concurrent_hnsw_builds_are_correct():
    """Four graphs built in parallel threads each self-retrieve correctly."""
    datasets = [_unit_data(_SERIAL_N, 128, seed) for seed in range(4)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        indexes = list(ex.map(_build_hnsw, datasets))
    for h, data in zip(indexes, datasets):
        res = h.search_np(data[10], 1, 64)
        assert res[0][0] == 10, f"self-query failed: {res[:2]}"


def test_concurrent_ivf_train_is_correct():
    """IVF train_np/add_np across threads stay correct (GIL released)."""

    def build(seed):
        data = _unit_data(_SERIAL_N, 64, seed)
        idx = vectro_py.PyIvfIndex(16, 16)
        idx.train_np(data, 10, 42)
        idx.add_np(data)
        return idx, data

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(build, range(4)))
    for idx, data in results:
        res = idx.search_np(data[0], 1)
        assert res[0][1] < 1e-4, f"self-query distance too large: {res[:2]}"


@pytest.mark.skipif((os.cpu_count() or 1) < 4, reason="needs >=4 cores to observe overlap")
def test_gil_released_during_build_gives_speedup():
    """Concurrent serial-path builds finish well under the serial sum.

    A held GIL would make ``concurrent`` ~= ``serial`` (often worse, from
    contention); releasing it lets the four Rust builds run in parallel.
    """
    data = _unit_data(_SERIAL_N, 256, 0)
    rounds = 4
    n_threads = 4

    def work(_):
        for _ in range(rounds):
            _build_hnsw(data)

    t0 = time.perf_counter()
    for i in range(n_threads):
        work(i)
    serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(work, range(n_threads)))
    concurrent = time.perf_counter() - t0

    # Very generous bound: 4-way parallelism should approach 4x; even a heavily
    # degraded ~1.3x clears this. Only a non-released GIL fails it.
    assert concurrent < serial * 0.8, (
        f"no GIL-release speedup: serial={serial:.3f}s concurrent={concurrent:.3f}s "
        f"(ratio {concurrent / serial:.2f})"
    )
