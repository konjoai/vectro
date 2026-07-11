"""Dataset loaders for the recall-matched benchmark harness.

Standard ANN benchmark datasets ship as ``.fvecs`` / ``.ivecs`` (little-endian
``int32`` dimension prefix per row, then that many ``float32`` / ``int32``
values). This module reads them, exposes a small ``Dataset`` value type, and a
``LOADERS`` registry so an embedding dataset (Cohere / MS MARCO, …) can be added
later by registering one function — no harness changes elsewhere, per the
sprint's loader-interface requirement.

Data is cached under ``benchmarks/data/`` (gitignored) and never committed.
Downloads are checksum-verified; a mismatch is a hard error, not a warning.
"""

from __future__ import annotations

import hashlib
import logging
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Repo-root/benchmarks/data — the gitignored cache. Resolved relative to this
# file so the single entry point works from any CWD.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Dataset:
    """A loaded ANN benchmark dataset in the harness's canonical form."""

    name: str
    train: np.ndarray  # (N, d) float32 — the corpus to index
    queries: np.ndarray  # (Q, d) float32 — the query set
    ground_truth: np.ndarray  # (Q, K) int32 — true nearest-neighbour ids per query
    metric: str  # "l2" or "cosine" (angular)

    @property
    def dim(self) -> int:
        return int(self.train.shape[1])

    @property
    def n(self) -> int:
        return int(self.train.shape[0])

    def scope(self) -> dict:
        return {
            "dataset": self.name,
            "n": self.n,
            "dim": self.dim,
            "queries": int(self.queries.shape[0]),
            "metric": self.metric,
            "gt_k": int(self.ground_truth.shape[1]),
        }


# ─────────────────────────── .fvecs / .ivecs ────────────────────────────────


def read_fvecs(path: Path) -> np.ndarray:
    """Read a little-endian ``.fvecs`` file into an ``(N, d) float32`` array."""
    raw = np.fromfile(str(path), dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    dim = int(raw[0])
    row = dim + 1  # each row is [dim, v0..v_{dim-1}]
    if raw.size % row != 0:
        raise ValueError(f"{path}: size {raw.size} not a multiple of row width {row}")
    data = raw.reshape(-1, row)
    if not np.all(data[:, 0] == dim):
        raise ValueError(f"{path}: inconsistent per-row dimension prefix")
    return np.ascontiguousarray(data[:, 1:].view(np.float32))


def read_ivecs(path: Path) -> np.ndarray:
    """Read a little-endian ``.ivecs`` file into an ``(N, d) int32`` array."""
    raw = np.fromfile(str(path), dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dim = int(raw[0])
    row = dim + 1
    if raw.size % row != 0:
        raise ValueError(f"{path}: size {raw.size} not a multiple of row width {row}")
    return np.ascontiguousarray(raw.reshape(-1, row)[:, 1:])


def write_fvecs(path: Path, arr: np.ndarray) -> None:
    """Write an ``(N, d) float32`` array as ``.fvecs`` (used by the synthetic fixture)."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    with open(path, "wb") as fh:
        for row in arr:
            fh.write(struct.pack("<i", d))
            fh.write(row.tobytes())


def write_ivecs(path: Path, arr: np.ndarray) -> None:
    """Write an ``(N, d) int32`` array as ``.ivecs``."""
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    n, d = arr.shape
    with open(path, "wb") as fh:
        for row in arr:
            fh.write(struct.pack("<i", d))
            fh.write(row.tobytes())


# ─────────────────────────── download + checksum ────────────────────────────


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, sha256: Optional[str]) -> Path:
    """Download ``url`` → ``dest`` (skipping if present + verified), verify checksum.

    A checksum mismatch is a hard error: silent acceptance of corrupt benchmark
    data would poison every downstream claim.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256 and sha256_of(dest) == sha256:
        logger.info("cached + verified: %s", dest.name)
        return dest
    if dest.exists() and not sha256:
        logger.warning("using cached %s (no checksum registered to verify)", dest.name)
        return dest
    logger.info("downloading %s → %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - known static benchmark URLs
    if sha256:
        got = sha256_of(tmp)
        if got != sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch for {url}: expected {sha256}, got {got}")
    tmp.replace(dest)
    return dest


# ───────────────────────────── ground truth ─────────────────────────────────


def compute_ground_truth(train: np.ndarray, queries: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Exact brute-force top-k ids per query — the recall reference.

    Chunked over queries to bound memory. Used to build ground truth for the
    synthetic fixture and for any dataset shipped without a ``groundtruth`` file.
    """
    train = np.ascontiguousarray(train, dtype=np.float32)
    q = np.ascontiguousarray(queries, dtype=np.float32)
    if metric == "cosine":
        train = _l2_normalize(train)
        q = _l2_normalize(q)
    out = np.empty((q.shape[0], k), dtype=np.int32)
    step = 256
    for i in range(0, q.shape[0], step):
        block = q[i : i + step]
        if metric == "l2":
            # ||a-b||^2 = ||a||^2 - 2 a·b + ||b||^2; argmin over train.
            d = (
                (block * block).sum(1, keepdims=True)
                - 2.0 * block @ train.T
                + (train * train).sum(1)[None, :]
            )
            idx = np.argpartition(d, k, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(d, idx, axis=1), axis=1)
        else:  # cosine/angular: larger dot = closer
            sim = block @ train.T
            idx = np.argpartition(-sim, k, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1)
        out[i : i + block.shape[0]] = np.take_along_axis(idx, order, axis=1).astype(np.int32)
    return out


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Mean recall@k: fraction of the true top-k found, averaged over queries."""
    found = np.asarray(found)[:, :k]
    truth = np.asarray(truth)[:, :k]
    hits = 0
    for f_row, t_row in zip(found, truth):
        hits += len(set(int(x) for x in f_row) & set(int(x) for x in t_row))
    return hits / (found.shape[0] * k)


# ───────────────────────────── loader registry ──────────────────────────────

# Checksums intentionally left None until the first verified fetch on the target
# host pins them (mismatch is fatal, so a wrong pin is worse than an unset one).
# The URLs are the canonical TEXMEX / ANN-benchmarks mirrors.
_SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
_GIST_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"


def _load_texmex(name: str, subdir: str, metric: str) -> Dataset:
    base = DATA_DIR / subdir
    needed = {
        "train": base / f"{subdir}_base.fvecs",
        "queries": base / f"{subdir}_query.fvecs",
        "gt": base / f"{subdir}_groundtruth.ivecs",
    }
    missing = [p for p in needed.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{name}: missing {[str(p) for p in missing]}. Run "
            f"`python benchmarks/harness/download.py --dataset {name}` first "
            f"(URL: {_SIFT_URL if name == 'sift1m' else _GIST_URL})."
        )
    train = read_fvecs(needed["train"])
    queries = read_fvecs(needed["queries"])
    gt = read_ivecs(needed["gt"])
    return Dataset(name=name, train=train, queries=queries, ground_truth=gt, metric=metric)


def load_sift1m() -> Dataset:
    """SIFT1M: 1M x 128 SIFT descriptors, L2 metric (TEXMEX)."""
    return _load_texmex("sift1m", "sift", "l2")


def load_gist1m() -> Dataset:
    """GIST1M: 1M x 960 GIST descriptors, L2 metric (TEXMEX)."""
    return _load_texmex("gist1m", "gist", "l2")


def load_synthetic(
    n: int = 5000, q: int = 200, dim: int = 64, k: int = 100, seed: int = 20260711
) -> Dataset:
    """A small clustered synthetic dataset for harness self-test / CI.

    Deliberately tiny so the full recall-matched suite runs end-to-end in
    seconds (proving the harness + its two-run stability check work) without the
    multi-GB SIFT/GIST download or the target hardware. Queries are lightly
    perturbed copies of random corpus points, so each query has a well-separated,
    unambiguous true neighbourhood and recall@k can actually climb to 0.90/0.95
    as the knob rises — the regime the recall-matched sweep needs. Heavily
    clustered data instead produces many near-equidistant ties, where exact-match
    recall@10 is unstable for *any* approximate method and never reaches target.
    """
    rng = np.random.default_rng(seed)
    # Moderately-clustered corpus with queries = perturbed corpus points. This is
    # the realistic middle ground: too much clustering saturates recall@10 on
    # near-equidistant ties; extreme cluster separation breaks HNSW graph
    # navigability (isolated islands). The harness self-test asserts *mechanics*
    # (sweep → measure → stability gate), not that a specific recall target is
    # reachable on a toy corpus with a pure-Python graph — an out-of-band recall
    # is flagged ⚠ and reported honestly, which is the intended behaviour. The
    # 0.90/0.95 operating points and the full baseline table are produced on real
    # SIFT1M/GIST1M with the built extension on target hardware.
    n_centers = max(8, n // 200)
    centers = rng.standard_normal((n_centers, dim)).astype(np.float32) * 6.0
    assign = rng.integers(0, n_centers, size=n)
    train = (centers[assign] + rng.standard_normal((n, dim)).astype(np.float32)).astype(np.float32)
    pick = rng.choice(n, size=q, replace=False)
    queries = (train[pick] + rng.standard_normal((q, dim)).astype(np.float32) * 0.15).astype(
        np.float32
    )
    gt = compute_ground_truth(train, queries, k=k, metric="l2")
    return Dataset(name="synthetic", train=train, queries=queries, ground_truth=gt, metric="l2")


# name → zero-arg loader. Register an embedding dataset here to add it.
LOADERS: dict[str, Callable[[], Dataset]] = {
    "sift1m": load_sift1m,
    "gist1m": load_gist1m,
    "synthetic": load_synthetic,
}


def load(name: str) -> Dataset:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset '{name}'; known: {sorted(LOADERS)}")
    return LOADERS[name]()
