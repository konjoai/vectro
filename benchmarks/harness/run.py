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
    from benchmarks.harness import _resume, datasets, engines, protocol, report, stats
else:
    from . import _resume, datasets, engines, protocol, report, stats

logger = logging.getLogger("vectro.harness")


def _build_engine(name: str):
    cls = engines.REGISTRY[name]
    avail = cls.availability()
    return cls, avail


def _run_once(ds, engine_names, k, recall_targets, n_runs, warmup, rep, ckpt=None):
    """Build + tune + measure every available engine once.

    Returns ``(measurements, skipped, resumed)`` where ``measurements`` is a list
    of ``(engine, target, Measurement)`` with live engine instances (for the
    interleaved comparison), and ``resumed`` is a list of measurement dicts
    reloaded from the checkpoint for engines already completed in a prior run
    (``--resume``). A single engine failing never kills the suite.
    """
    measurements = []  # (engine_instance, recall_target, Measurement)
    skipped = []
    resumed: list[dict] = []
    for name in engine_names:
        unit = f"rep{rep}:{name}"
        if ckpt is not None and ckpt.done(unit):
            stored = ckpt.results().get(unit) if hasattr(ckpt, "results") else None
            n = len(stored) if stored else 0
            logger.info("resume: %s already measured (%d op point(s)); skipping", name, n)
            resumed.extend(stored or [])
            continue
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
        engine_dicts = []
        for target in recall_targets:
            op = protocol.find_operating_point(eng, ds, k=k, recall_target=target)
            if op.param is None:
                skipped.append((f"{name}@{target}", op.note))
                continue
            m = protocol.measure(eng, ds, op, k=k, n_runs=n_runs, warmup=warmup)
            measurements.append((eng, target, m))
            engine_dicts.append(m.as_dict())
        if ckpt is not None:  # checkpoint this engine so --resume skips it next time
            ckpt.mark(unit, engine_dicts)
    return measurements, skipped, resumed


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


def _interleaved_comparisons(measurements, ds, k, n_runs, warmup):
    """Paired Wilcoxon over freshly **interleaved** A/B/A/B runs, per recall target.

    Compares every other engine against the VECTRO fp32 baseline at the same
    recall target (the sprint's baseline diff). Re-measures each pair with
    ``protocol.measure_interleaved`` so run i of A is adjacent in time to run i of
    B — the fairness control the paired test assumes, and the sprint's explicit
    "interleave A/B/A/B so neither runs systematically hot" requirement. Operates
    on live engine instances from this run (resumed engines were compared in the
    run that first measured them).
    """
    verdicts = []
    by_target: dict[float, dict[str, tuple]] = {}
    for eng, target, m in measurements:
        by_target.setdefault(target, {})[eng.name] = (eng, m.operating_point)
    baseline = engines.VectroHnswFp32.name
    for target, engs in by_target.items():
        if baseline not in engs:
            continue
        base_eng, base_op = engs[baseline]
        for name, (cand_eng, cand_op) in engs.items():
            if name == baseline:
                continue
            m_base, m_cand = protocol.measure_interleaved(
                base_eng, base_op, cand_eng, cand_op, ds, k=k, n_runs=n_runs, warmup=warmup
            )
            v = stats.build_verdict(baseline, name, m_base.qps_runs, m_cand.qps_runs)
            d = v.as_dict()
            d["recall_target"] = target
            d["interleaved"] = True
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
    _resume.add_resume_args(p, default_fresh=True)  # a fresh full run is the default
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ds = datasets.load(args.dataset)
    logger.info("loaded %s: %s", args.dataset, ds.scope())
    engine_names = args.engines or engines.CORE_SUITE
    tag = args.tag or f"{args.suite}_{args.dataset}"

    # Resumable checkpoint: a multi-hour real-data suite that dies mid-engine
    # resumes instead of restarting. Each (rep, engine) is one unit.
    ckpt = _resume.make_checkpoint(
        _resume.checkpoint_path(f"run_{tag}"), fresh=_resume.is_fresh(args)
    )

    rep = report.SuiteReport(dataset_scope=ds.scope())
    rep.config = {
        "suite": args.suite,
        "k": args.k,
        "n_runs": args.runs,
        "warmup": args.warmup,
        "recall_targets": args.recall,
        "engines": engine_names,
    }

    run1, skipped, resumed = _run_once(
        ds, engine_names, args.k, args.recall, args.runs, args.warmup, rep=1, ckpt=ckpt
    )
    for eng, _t, m in run1:
        rep.add_measurement(m)
    for m_dict in resumed:  # engines reloaded from a prior --resume run
        rep.measurements.append(m_dict)
    for name, reason in skipped:
        rep.add_skipped(name, reason)
    for v in _interleaved_comparisons(run1, ds, args.k, args.runs, args.warmup):
        rep.add_comparison(v)

    if args.stability:
        logger.info("stability: second full-suite run ...")
        run2, _, _ = _run_once(
            ds, engine_names, args.k, args.recall, args.runs, args.warmup, rep=2, ckpt=ckpt
        )
        rep.stability = _stability(run1, run2)

    json_path, md_path = report.write(rep, tag=tag)
    print(report.to_markdown(rep))
    print(f"\nWrote: {json_path}\n       {md_path}")
    if rep.stability is not None and not rep.stability["passes"]:
        logger.error("HARNESS KILL-TEST FAILED: two-run p50 drift exceeds the CoV gate")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
