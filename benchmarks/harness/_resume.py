"""Checkpoint/resume contract for the harness's long-running entry points.

Wraps konjo's ``lib.packs.longrun.konjo_longrun`` (the repo's resume protocol:
``--resume`` / ``--fresh`` plus a per-unit ``Checkpoint``) when kiban is
installed, and falls back to a self-contained JSONL checkpoint otherwise — so a
multi-hour SIFT1M/GIST1M suite can die and resume without re-running finished
operating points, while the harness keeps **no hard runtime dependency** on the
CI tooling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

try:  # konjo_longrun ships with the kiban CI tool; optional at harness runtime.
    from lib.packs.longrun import konjo_longrun as _konjo
except Exception:  # pragma: no cover - exercised only where kiban is absent
    _konjo = None  # type: ignore[assignment]


def add_resume_args(
    parser: argparse.ArgumentParser, *, default_fresh: bool = False
) -> argparse.ArgumentParser:
    """Add the mutually-exclusive ``--resume`` / ``--fresh`` pair."""
    if _konjo is not None:
        _konjo.add_resume_args(parser, default_fresh=default_fresh)
        return parser
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="resume from the latest checkpoint")
    group.add_argument("--fresh", action="store_true", help="ignore checkpoints; start clean")
    parser.set_defaults(_longrun_default_fresh=default_fresh)
    return parser


def is_fresh(args: argparse.Namespace) -> bool:
    """Whether this run should ignore prior checkpoints and start clean."""
    if _konjo is not None:
        return bool(_konjo.is_fresh(args))
    if getattr(args, "fresh", False):
        return True
    if getattr(args, "resume", False):
        return False
    return bool(getattr(args, "_longrun_default_fresh", False))


class _JsonlCheckpoint:
    """Self-contained stand-in for ``konjo_longrun.Checkpoint`` (append-per-unit,
    latest-line-wins fold) used when kiban is not importable."""

    def __init__(self, path: Any, *, fresh: bool = False) -> None:
        self.path = Path(path)
        if fresh and self.path.exists():
            self.path.unlink()
        self._done: dict[str, Any] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    key = rec.get("unit")
                    if isinstance(key, str):
                        self._done[key] = rec.get("result")

    def done(self, unit_key: str) -> bool:
        return unit_key in self._done

    def mark(self, unit_key: str, result: Any = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"unit": unit_key, "result": result}) + "\n")
        self._done[unit_key] = result


def make_checkpoint(path: Any, *, fresh: bool = False) -> Any:
    """A resumable ``Checkpoint`` — konjo's when available, else the JSONL fallback."""
    if _konjo is not None:
        return _konjo.Checkpoint(path, fresh=fresh)
    return _JsonlCheckpoint(path, fresh=fresh)


def checkpoint_path(tag: str) -> Optional[str]:
    """Progress file under ``benchmarks/results/.checkpoints/`` for this run tag."""
    base = Path(__file__).resolve().parent.parent / "results" / ".checkpoints"
    return str(base / f"{tag}.jsonl")
