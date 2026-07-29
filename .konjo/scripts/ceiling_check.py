#!/usr/bin/env python3
"""Konjo generic ceiling-ratchet gate.

Compares a measured integer count (the output of `--measure-cmd`, run through
the shell, last line parsed as an int) against a ceiling recorded in a
`.konjo/*-ceiling.txt` file. Fails if the measured count exceeds the ceiling
(a regression); passes -- with a hint to ratchet down -- if it's below.

Generic on purpose: several of vectro's soft-to-blocking gate promotions
(clippy::pedantic, vulture, dry_check, rustdoc missing_docs) share this exact
shape (a real, non-zero standing violation count that can't be driven to zero
in one sprint, but must never grow) and don't need N near-identical scripts —
one general-purpose ratchet, one ceiling file per gate. Mirrors
`coverage_floor_check.py`'s floor/ratchet convention (this repo's own
`.konjo/coverage-floor.txt` precedent, and lopi's
`.konjo/scripts/coverage_floor_check.py` / `indexing_floor_check.py`), just
generalized to "any shell command that prints a count" instead of parsing
lcov.info specifically.

Exit codes:
  0 — measured <= ceiling
  1 — measured > ceiling (regression), or the measure command / ceiling file
      could not be read
  2 — ceiling file missing or malformed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def read_ceiling(ceiling_path: Path) -> int:
    """Read the locked ceiling: the first non-comment, non-blank line."""
    for raw_line in ceiling_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        return int(line)
    raise ValueError(f"{ceiling_path} has no ceiling value (only comments/blank lines)")


def measure(cmd: str) -> int:
    """Run `cmd` through the shell and parse its last non-empty stdout line as an int."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"measure command produced no output: {cmd!r}")
    return int(lines[-1])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ceiling-file", type=Path, required=True)
    parser.add_argument(
        "--measure-cmd",
        required=True,
        help="shell command whose last stdout line is the measured integer count",
    )
    parser.add_argument(
        "--name",
        default="",
        help="label for this gate in output (defaults to the ceiling filename)",
    )
    args = parser.parse_args(argv)
    label = args.name or args.ceiling_file.stem

    try:
        ceiling = read_ceiling(args.ceiling_file)
    except (OSError, ValueError) as exc:
        print(f"::error::Cannot read {label} ceiling from {args.ceiling_file}: {exc}")
        return 2

    try:
        measured = measure(args.measure_cmd)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"::error::Cannot measure {label}: {exc}")
        return 1

    print(f"{label}: measured {measured}, ceiling {ceiling}")

    if measured > ceiling:
        print(
            f"::error::{label} rose from {ceiling} to {measured} — a net-new "
            f"standing violation. Fix it, or if it's a deliberate accepted "
            f"tradeoff, ratchet {args.ceiling_file} up in the same PR and say "
            "why in the commit message and LEDGER.md — never silently."
        )
        return 1

    if measured < ceiling:
        print(
            f"{label} dropped {ceiling - measured} below the ceiling. Consider "
            f"ratcheting {args.ceiling_file} down to {measured} in this PR."
        )
    print(f"{label} ceiling gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
