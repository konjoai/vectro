"""Recall-matched measurement protocol.

The cardinal rule (sprint prompt Phase 1.2): never compare raw QPS at unmatched
recall. For each engine we sweep its recall knob (ef / n_probe) to the smallest
value that reaches a target recall@k, then measure QPS *at that operating point*.
Two engines are only ever compared at the same recall target.

Timing discipline (house rules + Phase 1.3/1.5): warm up, then take 30 timed
runs per operating point; when two engines are compared, interleave their runs
A/B/A/B so neither systematically runs hot; record p50/p95/p99 QPS and per-query
latency percentiles; log start/end timestamps.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import datasets
from .engines import Engine
from .stats import Percentiles, summarize

logger = logging.getLogger(__name__)

# Monotone knob grids: recall is non-decreasing in ef / n_probe, so a smallest-
# param-reaching-target scan is well defined.
DEFAULT_EF_GRID = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
DEFAULT_NPROBE_GRID = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]

RECALL_TARGETS = (0.90, 0.95)
RECALL_TOL = 0.005


@dataclass
class OperatingPoint:
    """The (param, achieved recall) an engine was tuned to for a recall target."""

    recall_target: float
    param_name: str
    param: Optional[int]
    achieved_recall: float
    within_tolerance: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "recall_target": self.recall_target,
            "param_name": self.param_name,
            "param": self.param,
            "achieved_recall": self.achieved_recall,
            "within_tolerance": self.within_tolerance,
            "note": self.note,
        }


@dataclass
class Measurement:
    """QPS + latency at one operating point over N interleaved timed runs."""

    engine: str
    version: str
    operating_point: OperatingPoint
    qps_runs: list[float]
    qps: Percentiles
    latency_ms: Percentiles
    start_ts: str
    end_ts: str

    def as_dict(self) -> dict:
        return {
            "engine": self.engine,
            "version": self.version,
            "operating_point": self.operating_point.as_dict(),
            "qps_runs": self.qps_runs,
            "qps": self.qps.as_dict(),
            "latency_ms": self.latency_ms.as_dict(),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
        }


def _grid_for(engine: Engine) -> list[int]:
    return list(DEFAULT_NPROBE_GRID if engine.param_name == "n_probe" else DEFAULT_EF_GRID)


def find_operating_point(
    engine: Engine,
    ds: datasets.Dataset,
    k: int,
    recall_target: float,
    tol: float = RECALL_TOL,
    grid: Optional[Sequence[int]] = None,
) -> OperatingPoint:
    """Smallest knob value reaching ``recall_target`` (± ``tol``) for this engine.

    Assumes the index is already built. Scans the monotone grid upward, returns
    the first param whose recall ≥ target − tol, flagging whether the achieved
    recall landed inside the ± tol band (an out-of-band result is reported, not
    silently treated as a match).
    """
    grid = list(grid) if grid is not None else _grid_for(engine)
    best_below: Optional[tuple[int, float]] = None
    for param in grid:
        found = engine.search(ds.queries, k=k, param=param)
        recall = datasets.recall_at_k(found, ds.ground_truth, k=k)
        logger.info(
            "%s @ %s=%d -> recall@%d=%.4f (target %.2f)",
            engine.name,
            engine.param_name,
            param,
            k,
            recall,
            recall_target,
        )
        if recall >= recall_target - tol:
            within = abs(recall - recall_target) <= tol or recall >= recall_target
            note = "" if within else f"closest achievable recall {recall:.4f}"
            return OperatingPoint(
                recall_target=recall_target,
                param_name=engine.param_name,
                param=param,
                achieved_recall=recall,
                within_tolerance=within,
                note=note,
            )
        best_below = (param, recall)
    # Never reached the target across the whole grid.
    param, recall = best_below if best_below else (None, float("nan"))
    return OperatingPoint(
        recall_target=recall_target,
        param_name=engine.param_name,
        param=param,
        achieved_recall=recall,
        within_tolerance=False,
        note="grid exhausted without reaching target; increase grid ceiling",
    )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _one_qps_run(engine: Engine, queries: np.ndarray, k: int, param: int) -> float:
    t0 = time.perf_counter()
    engine.search(queries, k=k, param=param)
    dt = time.perf_counter() - t0
    return queries.shape[0] / dt if dt > 0 else float("inf")


def _latency_percentiles(engine: Engine, queries: np.ndarray, k: int, param: int) -> Percentiles:
    """Per-query latency (ms) percentiles from one pass of single-query searches."""
    samples: list[float] = []
    for q in queries:
        qq = q.reshape(1, -1)
        t0 = time.perf_counter()
        engine.search(qq, k=k, param=param)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return summarize(samples)


def measure(
    engine: Engine,
    ds: datasets.Dataset,
    op: OperatingPoint,
    k: int,
    n_runs: int,
    warmup: int,
    run_hook=None,
) -> Measurement:
    """Take ``n_runs`` timed QPS runs at ``op`` plus a latency pass.

    ``run_hook`` (optional) is called between runs; the interleaver uses it to
    alternate engines. Warmup runs are discarded.
    """
    if op.param is None:
        raise ValueError(f"{engine.name}: no operating point for target {op.recall_target}")
    for _ in range(warmup):
        _one_qps_run(engine, ds.queries, k, op.param)
    start = _now_iso()
    qps_runs: list[float] = []
    for _ in range(n_runs):
        qps_runs.append(_one_qps_run(engine, ds.queries, k, op.param))
        if run_hook is not None:
            run_hook()
    end = _now_iso()
    latency = _latency_percentiles(engine, ds.queries, k, op.param)
    return Measurement(
        engine=engine.name,
        version=engine.version,
        operating_point=op,
        qps_runs=qps_runs,
        qps=summarize(qps_runs),
        latency_ms=latency,
        start_ts=start,
        end_ts=end,
    )


def measure_interleaved(
    engine_a: Engine,
    op_a: OperatingPoint,
    engine_b: Engine,
    op_b: OperatingPoint,
    ds: datasets.Dataset,
    k: int,
    n_runs: int,
    warmup: int,
) -> tuple[Measurement, Measurement]:
    """A/B/A/B interleaved timing so neither engine systematically runs hot.

    Both engines' warmups run first, then runs strictly alternate. Returns the
    two measurements with paired-comparable ``qps_runs`` (run i of A is adjacent
    in time to run i of B), which is what the paired Wilcoxon assumes.
    """
    if op_a.param is None or op_b.param is None:
        raise ValueError("interleaved measure needs an operating point for both engines")
    for _ in range(warmup):
        _one_qps_run(engine_a, ds.queries, k, op_a.param)
        _one_qps_run(engine_b, ds.queries, k, op_b.param)
    start = _now_iso()
    a_runs: list[float] = []
    b_runs: list[float] = []
    for _ in range(n_runs):
        a_runs.append(_one_qps_run(engine_a, ds.queries, k, op_a.param))
        b_runs.append(_one_qps_run(engine_b, ds.queries, k, op_b.param))
    end = _now_iso()
    lat_a = _latency_percentiles(engine_a, ds.queries, k, op_a.param)
    lat_b = _latency_percentiles(engine_b, ds.queries, k, op_b.param)
    m_a = Measurement(
        engine_a.name, engine_a.version, op_a, a_runs, summarize(a_runs), lat_a, start, end
    )
    m_b = Measurement(
        engine_b.name, engine_b.version, op_b, b_runs, summarize(b_runs), lat_b, start, end
    )
    return m_a, m_b
