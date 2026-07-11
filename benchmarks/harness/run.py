"""Single entry point for the recall-matched benchmark harness.

    python benchmarks/harness/run.py --suite core --dataset sift1m

No manual steps: loads the dataset, probes each engine's availability (skipping
unavailable baselines with a recorded reason), builds indexes, tunes each engine
to the recall@k targets, takes interleaved timed runs, runs paired Wilcoxon
comparisons, and writes JSON + a Markdown table to benchmarks/results/.

Harness kill-test (``--stability``): runs the full suite twice back-to-back;
the two runs' p50 QPS per config must agree within the CoV gate (10 %). If they
don't, the harness — not the engine — needs fixing before any optimization is
measured through it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow both `python benchmarks/harness/run.py` and `python -m`.
_HERE = Path(__file__).resolve()
if __package__ in (None, ""):
    sys.path.insert(0, str(_HERE.parent.parent.parent))
    from benchmarks.harness import datasets, engines, protocol, report, stats
else:
    from . import datasets, engines, protocol, report, stats

logger = logging.getLogger("vectro.harness")


def _build_engine(name: str):
    cls = engines.REGISTRY[name]
    avail = cls.availability()
    return cls, avail


def _run_once(ds, engine_names, k, recall_targets, n_runs, warmup, rep):
    """Build + tune + measure every available engine once. Returns (measurements, skipped)."""
    measurements = []  # (engine_instance, recall_target, Measurement)
    skipped = []
    for name in engine_names:
        cls, avail = _build_engine(name)
        if not avail.ok:
            logger.warning("skip %s: %s", name, avail.reason)
            skipped.append((name, avail.reason))
            continue
        eng = cls()
        logger.info("[rep %d] building %s ...", rep, name)
        try:
            eng.build(ds.train, ds.metric)
        except Exception as exc:  # a single engine failing must not kill the suite
            logger.warning("skip %s: build failed: %s", name, exc)
            skipped.append((name, f"build failed: {exc}"))
            continue
        for target in recall_targets:
            op = protocol.find_operating_point(eng, ds, k=k, recall_target=target)
            if op.param is None:
                skipped.append((f"{name}@{target}", op.note))
                continue
            m = protocol.measure(eng, ds, op, k=k, n_runs=n_runs, warmup=warmup)
            measurements.append((eng, target, m))
    return measurements, skipped


def _stability(run1, run2, gate=stats.COV_GATE):
    """Compare per-(engine, recall_target) p50 QPS between two full-suite runs."""
    idx = {(e.name, t): m for (e, t, m) in run2}
    per_config = []
    max_drift = 0.0
    for eng, target, m1 in run1:
        m2 = idx.get((eng.name, target))
        if m2 is None:
            continue
        p1, p2 = m1.qps.p50, m2.qps.p50
        drift = abs(p2 - p1) / p1 if p1 else float("inf")
        max_drift = max(max_drift, drift)
        per_config.append(
            {
                "engine": eng.name,
                "recall_target": target,
                "run1_p50": p1,
                "run2_p50": p2,
                "drift": drift,
            }
        )
    return {
        "passes": max_drift <= gate,
        "max_drift": max_drift,
        "gate": gate,
        "per_config": per_config,
    }


def _pairwise_comparisons(measurements):
    """Interleave-free paired Wilcoxon over already-collected runs, per recall target.

    Compares every non-VECTRO engine against the VECTRO fp32 baseline at the same
    recall target (the sprint's baseline diff). Uses the collected per-run QPS,
    which are equal-length by construction.
    """
    verdicts = []
    by_target: dict[float, dict[str, list]] = {}
    for eng, target, m in measurements:
        by_target.setdefault(target, {})[eng.name] = m
    baseline = engines.VectroHnswFp32.name
    for target, engs in by_target.items():
        if baseline not in engs:
            continue
        base_m = engs[baseline]
        for name, m in engs.items():
            if name == baseline:
                continue
            if len(m.qps_runs) != len(base_m.qps_runs):
                continue
            v = stats.build_verdict(baseline, name, base_m.qps_runs, m.qps_runs)
            d = v.as_dict()
            d["recall_target"] = target
            verdicts.append(d)
    return verdicts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="VECTRO recall-matched benchmark harness")
    p.add_argument("--suite", default="core", choices=["core"], help="engine suite")
    p.add_argument("--dataset", default="synthetic", help=f"one of {sorted(datasets.LOADERS)}")
    p.add_argument("--k", type=int, default=10, help="recall@k / top-k")
    p.add_argument("--runs", type=int, default=30, help="timed runs per operating point")
    p.add_argument("--warmup", type=int, default=5, help="discarded warmup runs")
    p.add_argument(
        "--recall",
        type=float,
        nargs="+",
        default=list(protocol.RECALL_TARGETS),
        help="recall@k targets to match",
    )
    p.add_argument(
        "--engines",
        nargs="+",
        default=None,
        help="subset of engine names (default: the full core suite)",
    )
    p.add_argument(
        "--stability",
        action="store_true",
        help="run the full suite twice and check two-run p50 QPS drift (kill-test)",
    )
    p.add_argument("--tag", default=None, help="results filename tag")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ds = datasets.load(args.dataset)
    logger.info("loaded %s: %s", args.dataset, ds.scope())
    engine_names = args.engines or engines.CORE_SUITE

    rep = report.SuiteReport(dataset_scope=ds.scope())
    rep.config = {
        "suite": args.suite,
        "k": args.k,
        "n_runs": args.runs,
        "warmup": args.warmup,
        "recall_targets": args.recall,
        "engines": engine_names,
    }

    run1, skipped = _run_once(ds, engine_names, args.k, args.recall, args.runs, args.warmup, rep=1)
    for eng, _t, m in run1:
        rep.add_measurement(m)
    for name, reason in skipped:
        rep.add_skipped(name, reason)
    for v in _pairwise_comparisons(run1):
        rep.add_comparison(v)

    if args.stability:
        logger.info("stability: second full-suite run ...")
        run2, _ = _run_once(ds, engine_names, args.k, args.recall, args.runs, args.warmup, rep=2)
        rep.stability = _stability(run1, run2)

    tag = args.tag or f"{args.suite}_{args.dataset}"
    json_path, md_path = report.write(rep, tag=tag)
    print(report.to_markdown(rep))
    print(f"\nWrote: {json_path}\n       {md_path}")
    if rep.stability is not None and not rep.stability["passes"]:
        logger.error("HARNESS KILL-TEST FAILED: two-run p50 drift exceeds the CoV gate")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
