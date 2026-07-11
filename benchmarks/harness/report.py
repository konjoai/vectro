"""Result emission: JSON + a generated Markdown table with the full scope line.

Every table carries a scope line (chip, dataset, metric, recall target, config)
so no number can be read out of context — the sprint's "scoped claims only"
rule. Results are written to ``benchmarks/results/`` with a timestamp and the
git commit hash, and never overwrite an existing file.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .protocol import Measurement

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).resolve().parent),
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def hardware_scope() -> dict:
    """Best-effort hardware/OS metadata for the scope line."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    # Apple Silicon chip name, when present, is the load-bearing scope token.
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        if out.stdout.strip():
            info["cpu_brand"] = out.stdout.strip()
    except Exception:
        pass
    return info


@dataclass
class SuiteReport:
    """A full harness run: config, environment, and every measurement + verdict."""

    dataset_scope: dict
    hardware: dict = field(default_factory=hardware_scope)
    commit: str = field(default_factory=git_commit)
    config: dict = field(default_factory=dict)
    measurements: list[dict] = field(default_factory=list)
    comparisons: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    stability: Optional[dict] = None

    def add_measurement(self, m: Measurement) -> None:
        self.measurements.append(m.as_dict())

    def add_comparison(self, verdict_dict: dict) -> None:
        self.comparisons.append(verdict_dict)

    def add_skipped(self, engine: str, reason: str) -> None:
        self.skipped.append({"engine": engine, "reason": reason})

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset_scope,
            "hardware": self.hardware,
            "commit": self.commit,
            "config": self.config,
            "measurements": self.measurements,
            "comparisons": self.comparisons,
            "skipped": self.skipped,
            "stability": self.stability,
        }


def _scope_line(report: SuiteReport, recall_target: Optional[float] = None) -> str:
    hw = report.hardware
    chip = hw.get("cpu_brand", hw.get("processor", hw.get("machine", "?")))
    ds = report.dataset_scope
    parts = [
        f"chip={chip}",
        f"dataset={ds.get('dataset')} (n={ds.get('n')}, d={ds.get('dim')}, "
        f"queries={ds.get('queries')})",
        f"metric={ds.get('metric')}",
        f"k={report.config.get('k')}",
        f"runs={report.config.get('n_runs')}",
        f"commit={report.commit}",
    ]
    if recall_target is not None:
        parts.insert(3, f"recall_target@{report.config.get('k')}={recall_target}")
    return "Scope: " + " · ".join(str(p) for p in parts)


def to_markdown(report: SuiteReport) -> str:
    """Render the report as Markdown: scope line + per-measurement QPS table."""
    lines: list[str] = []
    lines.append(f"# VECTRO recall-matched benchmark — {report.dataset_scope.get('dataset')}")
    lines.append("")
    lines.append(_scope_line(report))
    lines.append("")
    lines.append(
        "| Engine | Version | Recall target | Param | Achieved recall | "
        "p50 QPS | p95 QPS | p99 QPS | CoV | p50 lat (ms) | p99 lat (ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in report.measurements:
        op = m["operating_point"]
        q = m["qps"]
        lat = m["latency_ms"]
        param = f"{op['param_name']}={op['param']}"
        recall = f"{op['achieved_recall']:.4f}"
        if not op["within_tolerance"]:
            recall += " ⚠"
        lines.append(
            f"| {m['engine']} | {m['version']} | {op['recall_target']:.2f} | {param} | "
            f"{recall} | {q['p50']:.1f} | {q['p95']:.1f} | {q['p99']:.1f} | "
            f"{q['cov']:.3f} | {lat['p50']:.3f} | {lat['p99']:.3f} |"
        )
    lines.append("")
    if report.comparisons:
        lines.append("## Paired comparisons (Wilcoxon signed-rank, interleaved A/B/A/B)")
        lines.append("")
        lines.append(
            "| A (baseline) | B (candidate) | Median Δ% | p-value | effect r | CoV gate | verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for c in report.comparisons:
            w = c["wilcoxon"]
            verdict = _verdict_label(c)
            lines.append(
                f"| {c['baseline']} | {c['candidate']} | "
                f"{w['median_improvement_pct']:+.2f} | {w['p_value']:.2e} | "
                f"{w['effect_r']:.3f} | {'pass' if c['cov_gate_passes'] else 'FAIL'} | "
                f"{verdict} |"
            )
        lines.append("")
    if report.skipped:
        lines.append("## Skipped engines")
        lines.append("")
        for s in report.skipped:
            lines.append(f"- **{s['engine']}**: {s['reason']}")
        lines.append("")
    if report.stability:
        lines.append("## Harness two-run stability (kill-test)")
        lines.append("")
        st = report.stability
        lines.append(
            f"- Verdict: **{'PASS' if st['passes'] else 'FAIL'}** "
            f"(max per-config p50 QPS drift {st['max_drift']:.3%}, gate {st['gate']:.0%})"
        )
        for row in st.get("per_config", []):
            lines.append(
                f"  - {row['engine']} @ recall {row['recall_target']}: "
                f"run1 p50={row['run1_p50']:.1f}, run2 p50={row['run2_p50']:.1f}, "
                f"drift={row['drift']:.3%}"
            )
        lines.append("")
    return "\n".join(lines)


def _verdict_label(comparison: dict) -> str:
    if not comparison["cov_gate_passes"]:
        return "inconclusive (too noisy)"
    w = comparison["wilcoxon"]
    if not w["significant_at_0.05"]:
        return "no difference"
    return "B faster" if w["median_improvement_pct"] > 0 else "A faster"


def write(report: SuiteReport, tag: str) -> tuple[Path, Path]:
    """Write ``<ts>_<tag>.json`` and ``.md`` to results/ (never overwriting)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import time

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stem = f"{ts}_{tag}"
    json_path = RESULTS_DIR / f"{stem}.json"
    md_path = RESULTS_DIR / f"{stem}.md"
    if json_path.exists() or md_path.exists():  # pragma: no cover
        raise FileExistsError(f"refusing to overwrite {stem}.*")
    json_path.write_text(json.dumps(report.as_dict(), indent=2))
    md_path.write_text(to_markdown(report))
    logger.info("wrote %s and %s", json_path.name, md_path.name)
    return json_path, md_path
