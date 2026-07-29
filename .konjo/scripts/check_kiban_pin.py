#!/usr/bin/env python3
"""Konjo twin-pin drift check.

vectro pins the kiban ref in two places that must always agree:
`.konjo/kiban.ref` (the session-plane pin, read by kiban's own
`lib/self_update.sh`) and `KIBAN_REF` in `.github/workflows/konjo-gates.yml`
(the CI-plane pin, read by the workflow's `pip install kiban @ ... @${KIBAN_REF}`
step). They drifted silently once already -- `.konjo/kiban.ref` read `v1.1.0`
while `KIBAN_REF` read `v1.1.5`, eight minor versions behind kiban's real
`v1.9.0` -- which is why this script exists: the comment-only convention lopi
and squish use ("bump all three/two together") did not, in fact, keep this
repo's two pins in sync. A mechanical check that fails loudly is cheaper than
another silent drift.

Exit codes:
  0 -- pins agree
  1 -- pins disagree, or either file's ref could not be parsed
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KIBAN_REF_FILE = REPO_ROOT / ".konjo" / "kiban.ref"
WORKFLOW_FILE = REPO_ROOT / ".github" / "workflows" / "konjo-gates.yml"

_REF_RE = re.compile(r"^v\d+\.\d+\.\d+$")
_WORKFLOW_REF_RE = re.compile(r'KIBAN_REF:\s*"([^"]+)"')


def read_kiban_ref(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not _REF_RE.match(text):
        raise ValueError(f"{path}: {text!r} is not a plain vX.Y.Z ref")
    return text


def read_workflow_ref(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = _WORKFLOW_REF_RE.search(text)
    if not match:
        raise ValueError(f'{path}: no KIBAN_REF: "..." line found')
    return match.group(1)


def main() -> int:
    try:
        kiban_ref = read_kiban_ref(KIBAN_REF_FILE)
        workflow_ref = read_workflow_ref(WORKFLOW_FILE)
    except (OSError, ValueError) as exc:
        print(f"::error::twin-pin check could not read a pin: {exc}")
        return 1

    if kiban_ref != workflow_ref:
        print(
            f"::error::kiban pin drift: {KIBAN_REF_FILE} reads {kiban_ref!r} but "
            f"{WORKFLOW_FILE} reads KIBAN_REF={workflow_ref!r}. Bump both to the "
            "same ref in the same commit -- see .konjo/kiban.ref's header comment."
        )
        return 1

    print(f"kiban pin check: OK ({kiban_ref} in both files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
