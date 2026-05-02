#!/usr/bin/env python3
"""aggregate_paper_tables.py — consolidate `wave*_*.json` bench records.

Reads the records produced by `reproduce_paper.sh` / `reproduce_paper.ps1`
and writes two artifacts in the same directory as the inputs:

    aggregate.csv  — one row per (platform, wave) with mean / stddev /
                     cold flag / CoV / thermal state / git rev.
    aggregate.md   — markdown table version, suitable for arXiv appendix.

Flags any (platform, wave) entries whose CoV exceeds 5 %.

Usage:
    python scripts/aggregate_paper_tables.py results/paper/*.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List


COV_GATE_PCT = 5.0


def _load(paths: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for pat in paths:
        for fp in sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []):
            try:
                with open(fp) as fh:
                    rec = json.load(fh)
            except Exception as exc:
                print(f"WARN: skipping {fp}: {exc}", file=sys.stderr)
                continue
            rec["__path"] = fp
            records.append(rec)
    return records


def _bucket_key(rec: Dict[str, Any]) -> tuple:
    return (rec.get("platform", "?"), int(rec.get("wave", 0)),
            "cold" if rec.get("cold") else "warm")


def _aggregate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for rec in records:
        buckets.setdefault(_bucket_key(rec), []).append(rec)

    rows: List[Dict[str, Any]] = []
    for (platform, wave, mode), recs in sorted(buckets.items()):
        all_throughputs: List[float] = []
        for r in recs:
            all_throughputs.extend(
                float(x) for x in (r.get("throughputs") or [])
                if isinstance(x, (int, float))
            )
        n = len(all_throughputs)
        if n == 0:
            mean = stdev = cov = float("nan")
        else:
            mean = statistics.mean(all_throughputs)
            stdev = statistics.pstdev(all_throughputs) if n >= 2 else 0.0
            cov = 100.0 * (stdev / mean) if mean else 0.0

        rows.append({
            "platform":  platform,
            "wave":      wave,
            "mode":      mode,
            "samples":   n,
            "mean":      round(mean, 3) if n else None,
            "stdev":     round(stdev, 3) if n else None,
            "cov_pct":   round(cov, 3)   if n else None,
            "git_rev":   recs[-1].get("git_rev", ""),
            "simd":      recs[-1].get("simd", ""),
            "thermal":   f"{recs[-1].get('thermal_before','?')}→{recs[-1].get('thermal_after','?')}",
            "warn":      "⚠️ CoV>5%" if (n and cov > COV_GATE_PCT) else "",
        })
    return rows


def _write_csv(rows: List[Dict[str, Any]], out: Path) -> None:
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_md(rows: List[Dict[str, Any]], out: Path) -> None:
    if not rows:
        out.write_text("# (no rows)\n", encoding="utf-8")
        return
    keys = ["platform", "wave", "mode", "samples", "mean", "stdev",
            "cov_pct", "thermal", "simd", "warn"]
    lines = []
    lines.append("# Vectro paper — cross-platform aggregate")
    lines.append("")
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join("---" for _ in keys) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    lines.append("")
    lines.append(f"_(throughput in vec/s · CoV gate: {COV_GATE_PCT:.1f}%)_")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+",
                    help="JSON files (or globs) emitted by reproduce_paper.{sh,ps1}")
    args = ap.parse_args()

    records = _load(args.paths)
    rows = _aggregate(records)

    # Decide output dir from the first input file's parent.
    first = next(
        (r["__path"] for r in records if "__path" in r),
        args.paths[0] if args.paths else "results/paper/_",
    )
    out_dir = Path(first).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "aggregate.csv"
    md_path  = out_dir / "aggregate.md"
    _write_csv(rows, csv_path)
    _write_md(rows,  md_path)

    print(f"✓ wrote {csv_path}")
    print(f"✓ wrote {md_path}")
    flagged = sum(1 for r in rows if r["warn"])
    if flagged:
        print(f"⚠️  {flagged} bucket(s) exceed CoV gate of {COV_GATE_PCT:.1f}%",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
