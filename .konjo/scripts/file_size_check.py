#!/usr/bin/env python3
"""Konjo file-size gate — blocking for every file not on the grandfather list.

Any `.rs`/`.py` file over 500 lines fails, unless its repo-relative path is
listed in `.konjo/oversized-allowlist.txt` (squish's own convention, reused
verbatim here — see that file's header). The allowlist exists so the gate can
go blocking today without requiring every legacy oversized file to be split
first; it is not a standing exemption for new growth — a *new* file over 500
lines fails immediately, and an allowlisted file is expected to shrink, not
grow further (not mechanically enforced by this script; see the allowlist's
own "Do NOT add new entries" rule).

Exit codes:
  0 — every non-allowlisted file is <= 500 lines
  1 — at least one non-allowlisted file exceeds 500 lines
"""

from __future__ import annotations

import sys
from pathlib import Path

LIMIT = 500
EXCLUDE_DIRS = {"target", "__pycache__", "node_modules", ".git"}


def read_allowlist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def main(argv: list[str]) -> int:
    repo_root = Path(argv[0]) if argv else Path(".")
    allowlist = read_allowlist(repo_root / ".konjo" / "oversized-allowlist.txt")

    violations = []
    for pattern in ("*.rs", "*.py"):
        for f in repo_root.rglob(pattern):
            if any(part in EXCLUDE_DIRS for part in f.parts):
                continue
            rel = str(f.relative_to(repo_root))
            if rel in allowlist:
                continue
            line_count = sum(1 for _ in f.open(encoding="utf-8", errors="replace"))
            if line_count > LIMIT:
                violations.append((rel, line_count))

    if violations:
        print(f"::error::{len(violations)} file(s) over {LIMIT} lines, not on the allowlist:")
        for rel, line_count in sorted(violations, key=lambda x: -x[1]):
            print(f"  {rel}: {line_count}")
        print(
            "Split the file, or if it's a real legacy exception, add it to "
            ".konjo/oversized-allowlist.txt with maintainer sign-off."
        )
        return 1

    print(f"File size gate: OK (0 non-allowlisted files over {LIMIT} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
