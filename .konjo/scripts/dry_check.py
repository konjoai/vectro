#!/usr/bin/env python3
"""Konjo DRY Checker — cross-language duplicate block detector."""

from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterator


def _strip_rust_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _strip_c_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _strip_python_comments(text: str) -> str:
    text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)
    return re.sub(r"#[^\n]*", "", text)


_STRIPPERS = {
    ".rs": _strip_rust_comments,
    ".py": _strip_python_comments,
    ".mojo": _strip_python_comments,
    ".ts": _strip_c_comments,
    ".tsx": _strip_c_comments,
    ".js": _strip_c_comments,
    ".jsx": _strip_c_comments,
}


def _normalize_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    text = _STRIPPERS.get(path.suffix, lambda t: t)(text)
    return [" ".join(l.split()) for l in text.splitlines() if " ".join(l.split())]


def _window_fingerprints(lines: list[str], window: int) -> Iterator[tuple[str, int]]:
    for i in range(max(0, len(lines) - window + 1)):
        yield hashlib.sha256("\n".join(lines[i : i + window]).encode()).hexdigest(), i


SUPPORTED_EXTENSIONS = {".rs", ".py", ".mojo", ".ts", ".tsx", ".js", ".jsx"}
SKIP_DIRS = {"target", ".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}


def _iter_sources(root: Path, extensions: set[str]) -> Iterator[Path]:
    for path in root.rglob("*"):
        if any(p in SKIP_DIRS for p in path.parts):
            continue
        if path.suffix in extensions and path.is_file():
            yield path


def _staged_files(root: Path, extensions: set[str]) -> list[Path]:
    r = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        root / l
        for l in r.stdout.splitlines()
        if (root / l).suffix in extensions and (root / l).is_file()
    ]


def _changed_files(root: Path, extensions: set[str]) -> list[Path]:
    r = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        root / l
        for l in r.stdout.splitlines()
        if (root / l).suffix in extensions and (root / l).is_file()
    ]


def find_duplicates(
    files: list[Path], all_files: list[Path], threshold: float, min_lines: int
) -> list[dict]:
    index: dict[str, list] = defaultdict(list)
    file_lines: dict[Path, list[str]] = {}
    for path in all_files:
        lines = _normalize_lines(path)
        file_lines[path] = lines
        for digest, start in _window_fingerprints(lines, min_lines):
            index[digest].append((path, start, lines))
    violations, seen_pairs = [], set()
    for target_path in files:
        lines = file_lines.get(target_path) or _normalize_lines(target_path)
        file_lines[target_path] = lines
        for digest, start in _window_fingerprints(lines, min_lines):
            for other_path, other_start, other_lines in index.get(digest, []):
                if other_path == target_path and other_start == start:
                    continue
                pair_key = frozenset([f"{target_path}:{start}", f"{other_path}:{other_start}"])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                sim = SequenceMatcher(
                    None,
                    lines[start : start + min_lines],
                    other_lines[other_start : other_start + min_lines],
                ).ratio()
                if sim >= threshold:
                    violations.append(
                        {
                            "file_a": str(target_path),
                            "line_a": start + 1,
                            "file_b": str(other_path),
                            "line_b": other_start + 1,
                            "similarity": round(sim, 3),
                            "lines": min_lines,
                            "sample": "\n".join(lines[start : start + 5]),
                        }
                    )
    return violations


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--min-lines", type=int, default=10)
    m = p.add_mutually_exclusive_group()
    m.add_argument("--staged-only", action="store_true")
    m.add_argument("--changed-only", action="store_true")
    p.add_argument("--json", action="store_true", dest="json_out")
    p.add_argument("--report")
    p.add_argument("--warn-only", action="store_true")
    p.add_argument("--extensions", default=",".join(sorted(SUPPORTED_EXTENSIONS)))
    args = p.parse_args()
    if args.root:
        root = Path(args.root)
    else:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False
        )
        root = Path(r.stdout.strip()) if r.returncode == 0 else Path(".")
    extensions = {e.strip() for e in args.extensions.split(",") if e.strip()}
    all_files = list(_iter_sources(root, extensions))
    if args.staged_only:
        scan_targets = _staged_files(root, extensions)
    elif args.changed_only:
        scan_targets = _changed_files(root, extensions)
    else:
        scan_targets = all_files
    if not scan_targets:
        if not args.json_out:
            print("[dry-check] No files to check.")
        return 0
    violations = find_duplicates(all_files, scan_targets, args.threshold, args.min_lines)
    report = {
        "duplicates": violations,
        "count": len(violations),
        "threshold": args.threshold,
        "min_lines": args.min_lines,
        "scanned": len(scan_targets),
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    if args.json_out:
        print(json.dumps(report, indent=2))
    else:
        if violations:
            print(f"[dry-check] ❌ {len(violations)} DRY violation(s) found")
            for v in violations:
                print(
                    f"  {v['file_a']}:{v['line_a']} ↔ {v['file_b']}:{v['line_b']} ({v['similarity'] * 100:.0f}%)"
                )
        else:
            print(f"[dry-check] ✅ No DRY violations ({len(scan_targets)} files scanned).")
    return 1 if violations and not args.warn_only else 0


if __name__ == "__main__":
    sys.exit(main())
