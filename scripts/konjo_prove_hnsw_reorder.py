#!/usr/bin/env python3
"""Feed a `hnsw_reorder_prove` artifact through the real Konjo "prove" gate math.

Usage:
    cargo run --release --example hnsw_reorder_prove > \
        benchmarks/results/prove_hnsw_reorder_<timestamp>.json
    python3 scripts/konjo_prove_hnsw_reorder.py \
        benchmarks/results/prove_hnsw_reorder_<timestamp>.json

Requires the pinned `kiban` package (see .konjo/kiban.ref) installed — it ships
the paired Wilcoxon signed-rank test (`lib.prove`) that the CI "prove" gate
trusts a `Konjo-Prove-Merge` trailer to mean was actually run. This script
exists so that trust is checkable: rerun it against the artifact and get the
same verdict.
"""

from __future__ import annotations

import json
import sys

# `lib` is the pinned `kiban` package's internal module (see .konjo/kiban.ref) —
# not a declared project dependency, so it has no type stubs mypy can resolve.
from lib import oneway, prove  # type: ignore[import-not-found]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <artifact.json>", file=sys.stderr)
        return 2

    with open(sys.argv[1]) as f:
        artifact = json.load(f)

    baseline = artifact["baseline_qps"]
    candidate = artifact["candidate_qps"]
    lower_is_better = bool(artifact.get("lower_is_better", False))

    min_effect_pct = 10.0  # .konjo/profile.yml's prove.min_effect_pct
    min_effect = prove.min_effect_from_percent(min_effect_pct, baseline)

    result = prove.paired_wilcoxon(
        baseline, candidate, run_floor=30, lower_is_better=lower_is_better
    )
    v = prove.verdict(result, min_effect=min_effect, lower_is_better=lower_is_better, alpha=0.05)

    print(f"artifact:            {artifact.get('artifact')}")
    print(f"change:              {artifact.get('change')}")
    print(f"n pairs:             {result.n} ({result.n_nonzero} nonzero)")
    print(f"median baseline:     {result.median_baseline:.4g} {artifact.get('unit', '')}")
    print(f"median candidate:    {result.median_candidate:.4g} {artifact.get('unit', '')}")
    print(f"median improvement:  {result.percent_change:+.4g}%")
    print(f"p-value:             {result.p_value:.4g}")
    print(f"min_effect_pct used: {min_effect_pct}")
    print(f"recall overlap:      {artifact.get('recall_overlap_after_vs_before')}")
    print(f"VERDICT:             {v.label} — {v.reason}")

    if v.is_merge:
        # The fingerprint changed files must match the fingerprint the CI
        # gate computed for the same PR (printed in its gate log). This
        # script only reports the verdict; adding the trailer is a separate,
        # deliberate step (not automated here) once the fingerprint is
        # cross-checked against the gate's own log line.
        print(f"\nOn confirmation, the CI-checked trailer label is: {oneway.PROVE_MERGE_TRAILER}")
    return 0 if v.is_merge else 1


if __name__ == "__main__":
    raise SystemExit(main())
