"""Statistical primitives for the recall-matched benchmark harness.

House-rules bar (see ``VECTRO_OPTIMIZATION_AUDIT_2026-07.md`` / sprint prompt):
30 paired runs, Wilcoxon signed-rank at p < 0.05, a reported effect size, and a
coefficient-of-variation gate — if CoV across runs exceeds 10 % the environment
is too noisy to claim anything, so the caller must say so rather than claim.

Everything here is pure (no I/O, no global state) so it is unit-testable and
identical whether it runs on Apple Silicon or an x86 CI host.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# The CoV gate from the house rules: above this, runs are too noisy to claim.
COV_GATE = 0.10


@dataclass(frozen=True)
class Percentiles:
    """p50 / p95 / p99 of a sample, plus mean / stddev / coefficient-of-variation."""

    p50: float
    p95: float
    p99: float
    mean: float
    stddev: float
    cov: float
    n: int

    def as_dict(self) -> dict:
        return {
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "mean": self.mean,
            "stddev": self.stddev,
            "cov": self.cov,
            "n": self.n,
        }


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 100]); matches numpy 'linear'."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (q / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_vals[int(rank)])
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def summarize(values: Sequence[float]) -> Percentiles:
    """Percentile + dispersion summary of a run sample (e.g. per-run QPS)."""
    vals = [float(v) for v in values]
    if not vals:
        return Percentiles(*(float("nan"),) * 6, n=0)
    ordered = sorted(vals)
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.0
    stddev = math.sqrt(var)
    cov = stddev / mean if mean != 0 else float("inf")
    return Percentiles(
        p50=_percentile(ordered, 50.0),
        p95=_percentile(ordered, 95.0),
        p99=_percentile(ordered, 99.0),
        mean=mean,
        stddev=stddev,
        cov=cov,
        n=len(vals),
    )


def cov_gate_passes(*summaries: Percentiles, gate: float = COV_GATE) -> bool:
    """True iff every summary's CoV is within the gate — else too noisy to claim."""
    return all(math.isfinite(s.cov) and s.cov <= gate for s in summaries)


@dataclass(frozen=True)
class WilcoxonResult:
    """Paired Wilcoxon signed-rank outcome with a reported effect size.

    ``effect_r`` is the matched-pairs rank-biserial correlation |Z| / sqrt(N),
    a standard effect size for the signed-rank test (0 = none, 1 = maximal).
    ``median_improvement_pct`` is the paired median relative change of B vs A
    (positive = B larger), reported alongside because significance without an
    effect size is exactly the headline this harness exists to prevent.
    """

    statistic: float
    p_value: float
    effect_r: float
    n_pairs: int
    n_nonzero: int
    median_improvement_pct: float
    method: str

    @property
    def significant(self) -> bool:
        return math.isfinite(self.p_value) and self.p_value < 0.05

    def as_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_r": self.effect_r,
            "n_pairs": self.n_pairs,
            "n_nonzero": self.n_nonzero,
            "median_improvement_pct": self.median_improvement_pct,
            "method": self.method,
            "significant_at_0.05": self.significant,
        }


def _median(vals: Sequence[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    m = len(s) // 2
    return float(s[m]) if len(s) % 2 else float((s[m - 1] + s[m]) / 2.0)


def paired_wilcoxon(baseline: Sequence[float], candidate: Sequence[float]) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test of ``candidate`` vs ``baseline``.

    Uses ``scipy.stats.wilcoxon`` (two-sided, Pratt zero-handling) with a
    self-contained normal-approximation fallback if scipy is unavailable, so the
    harness has no hard scipy dependency. Returns the statistic, p-value, an
    effect size, and the paired median improvement percentage.
    """
    a = [float(x) for x in baseline]
    b = [float(x) for x in candidate]
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal-length samples: {len(a)} vs {len(b)}")
    n = len(a)
    diffs = [bi - ai for ai, bi in zip(a, b)]
    nonzero = [d for d in diffs if d != 0.0]
    # Paired median relative improvement of B over A (guard zero baselines).
    rel = [(bi - ai) / ai for ai, bi in zip(a, b) if ai != 0.0]
    median_improvement_pct = 100.0 * _median(rel) if rel else float("nan")

    if len(nonzero) < 1:
        return WilcoxonResult(
            statistic=float("nan"),
            p_value=1.0,
            effect_r=0.0,
            n_pairs=n,
            n_nonzero=0,
            median_improvement_pct=median_improvement_pct,
            method="degenerate-all-ties",
        )

    stat, p_value, method = _wilcoxon_backend(nonzero)
    # Effect size: |Z| / sqrt(N) from the normal approximation of the signed-rank.
    effect_r = _effect_r(nonzero)
    return WilcoxonResult(
        statistic=float(stat),
        p_value=float(p_value),
        effect_r=effect_r,
        n_pairs=n,
        n_nonzero=len(nonzero),
        median_improvement_pct=median_improvement_pct,
        method=method,
    )


def _wilcoxon_backend(nonzero: Sequence[float]) -> tuple[float, float, str]:
    try:
        from scipy.stats import wilcoxon  # type: ignore

        res = wilcoxon(list(nonzero), zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue), "scipy.stats.wilcoxon"
    except Exception as exc:  # pragma: no cover - exercised only without scipy
        logger.warning("scipy.wilcoxon unavailable (%s); using normal approximation", exc)
        return _wilcoxon_normal_approx(nonzero)


def _signed_ranks(nonzero: Sequence[float]) -> tuple[float, float]:
    """Return (W+ statistic, sum of squared tie-corrected ranks) for the sample."""
    order = sorted(range(len(nonzero)), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * len(nonzero)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank across the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    sum_sq = sum(r * r for r in ranks)
    return w_plus, sum_sq


def _wilcoxon_normal_approx(nonzero: Sequence[float]) -> tuple[float, float, str]:
    n = len(nonzero)
    w_plus, sum_sq = _signed_ranks(nonzero)
    mean_w = n * (n + 1) / 4.0
    # Tie-corrected variance via the sum of squared ranks.
    var_w = (sum_sq - (n * (n + 1) / 2.0) ** 2 / n) / 4.0 if n > 1 else 0.0
    if var_w <= 0:
        return w_plus, 1.0, "normal-approx-degenerate"
    z = (w_plus - mean_w) / math.sqrt(var_w)
    p = math.erfc(abs(z) / math.sqrt(2.0))  # two-sided
    return w_plus, min(1.0, p), "normal-approx"


def _effect_r(nonzero: Sequence[float]) -> float:
    n = len(nonzero)
    if n < 1:
        return 0.0
    w_plus, sum_sq = _signed_ranks(nonzero)
    mean_w = n * (n + 1) / 4.0
    var_w = (sum_sq - (n * (n + 1) / 2.0) ** 2 / n) / 4.0 if n > 1 else 0.0
    if var_w <= 0:
        return 0.0
    z = (w_plus - mean_w) / math.sqrt(var_w)
    return min(1.0, abs(z) / math.sqrt(n))


@dataclass
class ComparisonVerdict:
    """The full A-vs-B verdict the harness records for any two compared engines."""

    baseline_name: str
    candidate_name: str
    baseline_qps: Percentiles
    candidate_qps: Percentiles
    wilcoxon: WilcoxonResult
    cov_ok: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "baseline": self.baseline_name,
            "candidate": self.candidate_name,
            "baseline_qps": self.baseline_qps.as_dict(),
            "candidate_qps": self.candidate_qps.as_dict(),
            "wilcoxon": self.wilcoxon.as_dict(),
            "cov_gate_passes": self.cov_ok,
            "notes": self.notes,
        }


def build_verdict(
    baseline_name: str,
    candidate_name: str,
    baseline_runs: Sequence[float],
    candidate_runs: Sequence[float],
) -> ComparisonVerdict:
    """Assemble percentiles + Wilcoxon + CoV gate into a single recorded verdict."""
    b_sum = summarize(baseline_runs)
    c_sum = summarize(candidate_runs)
    wil = paired_wilcoxon(baseline_runs, candidate_runs)
    cov_ok = cov_gate_passes(b_sum, c_sum)
    notes: list[str] = []
    if not cov_ok:
        notes.append(
            f"CoV gate FAILED (baseline={b_sum.cov:.3f}, candidate={c_sum.cov:.3f}, "
            f"gate={COV_GATE}): environment too noisy to claim a difference."
        )
    if cov_ok and not wil.significant:
        notes.append("No significant difference at p < 0.05.")
    return ComparisonVerdict(
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        baseline_qps=b_sum,
        candidate_qps=c_sum,
        wilcoxon=wil,
        cov_ok=cov_ok,
        notes=notes,
    )
