#!/usr/bin/env python3
"""Konjo Adversarial Review Agent — Wall 3. Critic: claude-opus-4-6"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

CRITIC_MODEL = "claude-opus-4-6"
MAX_DIFF_CHARS = 80_000
MAX_TOKENS = 4096
MAX_RETRIES = 3

SYSTEM_PROMPT = """You are the Konjo Adversarial Reviewer. Find flaws the builder missed.
Answer Q1-Q10: CORRECTNESS, COVERAGE BLIND SPOTS, DEAD CODE, DOCUMENTATION, ERROR HANDLING, DRY, COMPLEXITY, SECURITY, PERFORMANCE, KONJO STANDARD.
BLOCKER for Q1/Q3/Q5/Q8. WARNING for Q2/Q6/Q7/Q9.
Respond ONLY with valid JSON: {"verdict": "APPROVED"|"WARNING"|"BLOCKER", "summary": "...", "questions": {...}, "blockers": [], "warnings": [], "approved_aspects": []}"""


def _load_anthropic():
    try:
        import anthropic

        return anthropic
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        raise


def _call_api(diff_text: str, mod) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    client = mod.Anthropic(api_key=key)
    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = diff_text[:MAX_DIFF_CHARS] + "\n[TRUNCATED]"
    for attempt in range(MAX_RETRIES):
        try:
            r = client.messages.create(
                model=CRITIC_MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
                ],
                messages=[
                    {
                        "role": "user",
                        "content": f"Review this diff:\n\n<diff>\n{diff_text}\n</diff>",
                    }
                ],
            )
            print(
                f"[konjo-review] tokens: input={r.usage.input_tokens} output={r.usage.output_tokens}",
                file=sys.stderr,
            )
            return json.loads(r.content[0].text.strip())
        except (mod.RateLimitError, mod.APIStatusError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt * 2)
            else:
                raise
        except json.JSONDecodeError as e:
            raise ValueError("Non-JSON response") from e
    raise RuntimeError("Exhausted retries")


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--diff-file")
    g.add_argument("--diff")
    p.add_argument("--output")
    p.add_argument("--json", action="store_true", dest="json_out")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--soft-fail", action="store_true")
    args = p.parse_args()
    if args.diff:
        diff_text = args.diff
    elif args.diff_file:
        diff_text = Path(args.diff_file).read_text()
    else:
        if sys.stdin.isatty():
            print("ERROR: pipe a diff to stdin", file=sys.stderr)
            return 2
        diff_text = sys.stdin.read()
    if not diff_text.strip():
        print("[konjo-review] Empty diff. Approved.", file=sys.stderr)
        return 0
    if args.dry_run:
        print(f"[konjo-review] DRY RUN model={CRITIC_MODEL}", file=sys.stderr)
        return 0
    try:
        mod = _load_anthropic()
        result = _call_api(diff_text, mod)
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"[konjo-review] ERROR: {e}", file=sys.stderr)
        return 0
    verdict = result.get("verdict", "UNKNOWN")
    has_blockers = verdict == "BLOCKER" or bool(result.get("blockers"))
    out = (
        json.dumps(result, indent=2)
        if args.json_out
        else f"# Konjo Review\nVerdict: {verdict}\n{result.get('summary', '')}\n"
    )
    if args.output:
        Path(args.output).write_text(out)
    else:
        print(out)
    if has_blockers and not args.soft_fail:
        print(f"[konjo-review] VERDICT: {verdict} — merge blocked.", file=sys.stderr)
        return 1
    print(f"[konjo-review] VERDICT: {verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
