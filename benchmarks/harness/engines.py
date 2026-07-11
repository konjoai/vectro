"""Engine adapters for the recall-matched benchmark harness.

Every engine exposes the same tiny interface so the protocol layer is
engine-agnostic:

    build(train, metric)          -> None      (index the corpus)
    search(queries, k, param)     -> (Q, k) ids array
    param_name                    -> "ef" | "n_probe"   (the recall knob to sweep)
    version                       -> str        (pinned into results JSON)

Baselines required by the sprint: VECTRO fp32 HNSW, VECTRO int8 HNSW,
VECTRO IVF-PQ4, plus faiss (HNSWFlat, IVF-PQ) and hnswlib. Optional third-party
engines degrade gracefully: an unavailable engine reports *why* (so a results
table never silently omits a baseline) instead of crashing the run.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class Availability:
    ok: bool
    version: str = "n/a"
    reason: str = ""


class Engine:
    """Base adapter. Subclasses set ``param_name`` and implement build/search."""

    name: str = "engine"
    param_name: str = "ef"

    @classmethod
    def availability(cls) -> Availability:  # pragma: no cover - trivial
        return Availability(ok=True)

    @property
    def version(self) -> str:
        return "unknown"

    def build(self, train: np.ndarray, metric: str) -> None:
        raise NotImplementedError

    def search(self, queries: np.ndarray, k: int, param: int) -> np.ndarray:
        raise NotImplementedError


# ─────────────────────────────── VECTRO ─────────────────────────────────────


def _vectro_hnsw_cls():
    return importlib.import_module("python.hnsw_api").HNSWIndex


def _vectro_version() -> str:
    try:
        return importlib.import_module("python").__version__
    except Exception:  # pragma: no cover
        return "unknown"


class VectroHnswFp32(Engine):
    """VECTRO HNSW, full-precision (fp32) distances.

    Uses the native Rust core when built and ``space='cosine'``; otherwise the
    pure-Python correctness baseline (which is orders of magnitude slower and
    infeasible at 1M — the harness reports the active backend so a slow pure-
    Python run is never mistaken for the shipped kernel path).
    """

    name = "vectro-hnsw-fp32"
    param_name = "ef"

    def __init__(self, m: int = 16, ef_construction: int = 200) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self._idx = None
        self.backend = "?"

    @classmethod
    def availability(cls) -> Availability:
        try:
            _vectro_hnsw_cls()
            return Availability(ok=True, version=_vectro_version())
        except Exception as exc:  # pragma: no cover
            return Availability(ok=False, reason=f"import python.hnsw_api failed: {exc}")

    @property
    def version(self) -> str:
        return f"vectro {_vectro_version()} (backend={self.backend})"

    def _space(self, metric: str) -> str:
        return "cosine" if metric == "cosine" else "l2"

    def build(self, train: np.ndarray, metric: str) -> None:
        cls = _vectro_hnsw_cls()
        self._idx = cls(M=self.m, ef_construction=self.ef_construction, space=self._space(metric))
        self._idx.add(np.ascontiguousarray(train, dtype=np.float32))
        self.backend = getattr(self._idx, "backend", "?")

    def search(self, queries: np.ndarray, k: int, param: int) -> np.ndarray:
        labels, _dists = self._idx.search_batch(
            np.ascontiguousarray(queries, dtype=np.float32), k=k, ef=int(param)
        )
        return np.asarray(labels, dtype=np.int64)


class VectroHnswInt8(VectroHnswFp32):
    """VECTRO INT8 quant-HNSW — the flagship quantized search path.

    The INT8 quant-HNSW (Rust ``QuantHnswIndex``, the target of the NEON ``sdot``
    kernel) is not yet exposed through the pure-Python API — only the fp32 graph
    is wired through PyO3. This adapter therefore reports itself unavailable with
    that reason until the quant-HNSW search entry is bound, rather than silently
    scoring the fp32 path and mislabelling it INT8.
    """

    name = "vectro-hnsw-int8"

    @classmethod
    def availability(cls) -> Availability:
        try:
            mod = importlib.import_module("python.hnsw_api")
        except Exception as exc:  # pragma: no cover
            return Availability(ok=False, reason=f"import python.hnsw_api failed: {exc}")
        idx_cls = getattr(mod, "HNSWIndex", None)
        if idx_cls is not None and "quant" in (idx_cls.__init__.__doc__ or "").lower():
            return Availability(ok=True, version=_vectro_version())
        return Availability(
            ok=False,
            version=_vectro_version(),
            reason=(
                "INT8 quant-HNSW search not exposed via python API "
                "(Rust QuantHnswIndex awaits PyO3 binding); "
                "run this baseline from the Rust harness on target hardware."
            ),
        )


class VectroIvfPq(Engine):
    """VECTRO IVF-PQ (``python.ivf_api.IVFPQIndex``).

    NB: this is the pure-Python IVF-PQ reference. The audit's headline
    "IVF-PQ4" that beat faiss-IVF-PQ 1.76x is the Rust ``IvfPq4Index`` batched
    path; the harness records ``backend`` so the two are never conflated.
    """

    name = "vectro-ivfpq"
    param_name = "n_probe"

    def __init__(self, n_lists: int = 256, n_subspaces: int = 8, n_centroids: int = 256) -> None:
        self.n_lists = n_lists
        self.n_subspaces = n_subspaces
        self.n_centroids = n_centroids
        self._idx = None

    @classmethod
    def availability(cls) -> Availability:
        try:
            mod = importlib.import_module("python.ivf_api")
        except Exception as exc:  # pragma: no cover
            return Availability(ok=False, reason=f"import python.ivf_api failed: {exc}")
        # IVFPQIndex is a thin wrapper over the Rust bindings — it raises at
        # construction if vectro_py is not built. Probe that here so an
        # unbuilt-extension host is reported (not crashed) at suite time.
        try:
            mod.IVFPQIndex(n_lists=2, n_probe=1)
            return Availability(ok=True, version=_vectro_version())
        except Exception as exc:
            return Availability(
                ok=False,
                version=_vectro_version(),
                reason=f"vectro_py extension not built ({exc}); run `maturin develop`",
            )

    @property
    def version(self) -> str:
        return f"vectro {_vectro_version()} (ivfpq/python)"

    def build(self, train: np.ndarray, metric: str) -> None:
        cls = importlib.import_module("python.ivf_api").IVFPQIndex
        self._idx = cls(n_lists=self.n_lists, n_probe=1)
        rows = np.ascontiguousarray(train, dtype=np.float32).tolist()
        self._idx.train(rows, n_subspaces=self.n_subspaces, n_centroids=self.n_centroids)
        for row in rows:
            self._idx.add(row)

    def search(self, queries: np.ndarray, k: int, param: int) -> np.ndarray:
        self._idx.n_probe = int(param)
        out = np.empty((queries.shape[0], k), dtype=np.int64)
        for i, q in enumerate(np.ascontiguousarray(queries, dtype=np.float32)):
            res = self._idx.search(q.tolist(), k=k)
            ids = [int(rid) for rid, _ in res][:k]
            ids += [-1] * (k - len(ids))
            out[i] = ids
        return out


# ─────────────────────────────── faiss ──────────────────────────────────────


class _FaissBase(Engine):
    @classmethod
    def _faiss(cls):
        return importlib.import_module("faiss")

    @classmethod
    def availability(cls) -> Availability:
        try:
            faiss = cls._faiss()
            return Availability(ok=True, version=getattr(faiss, "__version__", "unknown"))
        except Exception as exc:
            return Availability(ok=False, reason=f"faiss not installed: {exc}")

    @property
    def version(self) -> str:
        return f"faiss {getattr(self._faiss(), '__version__', 'unknown')}"

    @staticmethod
    def _metric(faiss, metric: str):
        return faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2


class FaissHnswFlat(_FaissBase):
    name = "faiss-hnsw-flat"
    param_name = "ef"

    def __init__(self, m: int = 16, ef_construction: int = 200) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self._idx = None

    def build(self, train: np.ndarray, metric: str) -> None:
        faiss = self._faiss()
        x = np.ascontiguousarray(train, dtype=np.float32)
        if metric == "cosine":
            faiss.normalize_L2(x)
        self._idx = faiss.IndexHNSWFlat(x.shape[1], self.m, self._metric(faiss, metric))
        self._idx.hnsw.efConstruction = self.ef_construction
        self._idx.add(x)
        self._metric_name = metric

    def search(self, queries: np.ndarray, k: int, param: int) -> np.ndarray:
        faiss = self._faiss()
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if self._metric_name == "cosine":
            faiss.normalize_L2(q)
        self._idx.hnsw.efSearch = int(param)
        _d, ids = self._idx.search(q, k)
        return ids.astype(np.int64)


class FaissIvfPq(_FaissBase):
    name = "faiss-ivfpq"
    param_name = "n_probe"

    def __init__(self, n_lists: int = 1024, m_pq: int = 8, nbits: int = 8) -> None:
        self.n_lists = n_lists
        self.m_pq = m_pq
        self.nbits = nbits
        self._idx = None

    def build(self, train: np.ndarray, metric: str) -> None:
        faiss = self._faiss()
        x = np.ascontiguousarray(train, dtype=np.float32)
        if metric == "cosine":
            faiss.normalize_L2(x)
        quant = faiss.IndexFlatL2(x.shape[1])
        self._idx = faiss.IndexIVFPQ(quant, x.shape[1], self.n_lists, self.m_pq, self.nbits)
        self._idx.train(x)
        self._idx.add(x)
        self._metric_name = metric

    def search(self, queries: np.ndarray, k: int, param: int) -> np.ndarray:
        faiss = self._faiss()
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if self._metric_name == "cosine":
            faiss.normalize_L2(q)
        self._idx.nprobe = int(param)
        _d, ids = self._idx.search(q, k)
        return ids.astype(np.int64)


# ────────────────────────────── hnswlib ─────────────────────────────────────


class HnswlibHnsw(Engine):
    name = "hnswlib"
    param_name = "ef"

    def __init__(self, m: int = 16, ef_construction: int = 200) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self._idx = None

    @classmethod
    def _lib(cls):
        return importlib.import_module("hnswlib")

    @classmethod
    def availability(cls) -> Availability:
        try:
            cls._lib()
            # hnswlib exposes no __version__; read the installed dist metadata.
            try:
                from importlib.metadata import version as _v

                ver = _v("hnswlib")
            except Exception:  # pragma: no cover
                ver = "unknown"
            return Availability(ok=True, version=ver)
        except Exception as exc:
            return Availability(ok=False, reason=f"hnswlib not installed: {exc}")

    @property
    def version(self) -> str:
        return f"hnswlib {self.availability().version}"

    def build(self, train: np.ndarray, metric: str) -> None:
        hnswlib = self._lib()
        x = np.ascontiguousarray(train, dtype=np.float32)
        space = "cosine" if metric == "cosine" else "l2"
        self._idx = hnswlib.Index(space=space, dim=x.shape[1])
        self._idx.init_index(
            max_elements=x.shape[0], ef_construction=self.ef_construction, M=self.m
        )
        self._idx.add_items(x, np.arange(x.shape[0]))

    def search(self, queries: np.ndarray, k: int, param: int) -> np.ndarray:
        self._idx.set_ef(int(param))
        ids, _d = self._idx.knn_query(np.ascontiguousarray(queries, dtype=np.float32), k=k)
        return ids.astype(np.int64)


# name → factory. The suite picks from here; availability() gates each at runtime.
REGISTRY: dict[str, type[Engine]] = {
    VectroHnswFp32.name: VectroHnswFp32,
    VectroHnswInt8.name: VectroHnswInt8,
    VectroIvfPq.name: VectroIvfPq,
    FaissHnswFlat.name: FaissHnswFlat,
    FaissIvfPq.name: FaissIvfPq,
    HnswlibHnsw.name: HnswlibHnsw,
}

# The default "core" suite — the sprint's before-snapshot baseline set.
CORE_SUITE = [
    VectroHnswFp32.name,
    VectroHnswInt8.name,
    VectroIvfPq.name,
    FaissHnswFlat.name,
    FaissIvfPq.name,
    HnswlibHnsw.name,
]
