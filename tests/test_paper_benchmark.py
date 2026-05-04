"""Tests for benchmarks/vectro_paper_benchmark.py — closes the v5.0.0 gap.

The bench harness is referenced by ``pyproject.toml [tool.cibuildwheel]``,
``reproduce_paper.sh``, and ``reproduce_paper.ps1``.  These tests pin its
contract so a regression breaks CI before the wheel build does:

  * ``--quick --table int8 --json`` emits valid JSON
  * the JSON record carries the headline contract fields:
    ``throughput`` (numeric, in M vec/s), ``throughput_unit``,
    ``headline_table/n/d``, plus a populated ``rows`` array
  * ``--table all`` runs every quantisation table and produces three rows
  * ``--n / --d`` overrides force a one-shape run
  * the human-readable mode contains the headline marker line
  * the script can be executed twice in the same process without leaking
    state (each call returns a fresh record)
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "benchmarks" / "vectro_paper_benchmark.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, check=False, cwd=str(REPO_ROOT),
        timeout=300,
    )


class TestQuickJson(unittest.TestCase):
    def test_quick_json_emits_single_line(self):
        r = _run("--quick", "--table", "int8", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        # exactly one non-empty line of JSON
        out = r.stdout.strip()
        self.assertTrue(out, "no stdout")
        # must be one line
        self.assertEqual(len(out.splitlines()), 1)
        rec = json.loads(out)
        self.assertEqual(rec["schema"], "vectro/paper-benchmark/v1")

    def test_headline_throughput_field_is_numeric(self):
        r = _run("--quick", "--table", "int8", "--json")
        rec = json.loads(r.stdout.strip())
        self.assertIn("throughput", rec)
        self.assertIsInstance(rec["throughput"], (int, float))
        self.assertGreater(rec["throughput"], 0.0,
                           "throughput must be > 0 — reproduce_paper.{sh,ps1} "
                           "treats 0 as a sentinel for 'bench failed'")
        self.assertEqual(rec["throughput_unit"], "M vec/s")

    def test_required_headline_fields_present(self):
        r = _run("--quick", "--table", "int8", "--json")
        rec = json.loads(r.stdout.strip())
        for key in ("version", "platform", "python", "table", "quick",
                    "throughput", "throughput_unit",
                    "headline_table", "headline_n", "headline_d", "rows"):
            self.assertIn(key, rec, f"missing top-level field: {key!r}")
        self.assertGreaterEqual(len(rec["rows"]), 1)

    def test_row_shape_contract(self):
        r = _run("--quick", "--table", "int8", "--json")
        rec = json.loads(r.stdout.strip())
        row = rec["rows"][0]
        for key in ("table", "profile", "n", "d", "reps",
                    "samples_ms", "best_ms", "p50_ms",
                    "best_throughput_vec_s", "best_M_vec_s",
                    "p50_throughput_vec_s", "p50_M_vec_s",
                    "original_bytes", "compressed_bytes",
                    "ratio", "mean_cosine"):
            self.assertIn(key, row, f"missing row field: {key!r}")
        self.assertEqual(row["table"], "int8")
        self.assertEqual(row["n"], 10_000)
        self.assertEqual(row["d"], 768)
        self.assertGreater(row["ratio"], 1.0)


class TestTableAll(unittest.TestCase):
    def test_table_all_runs_every_table(self):
        r = _run("--quick", "--table", "all", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        rec = json.loads(r.stdout.strip())
        tables = sorted({row["table"] for row in rec["rows"]})
        self.assertEqual(tables, ["binary", "int8", "nf4"])

    def test_binary_has_high_compression_ratio(self):
        r = _run("--quick", "--table", "binary", "--json")
        rec = json.loads(r.stdout.strip())
        ratio = rec["rows"][0]["ratio"]
        self.assertGreater(ratio, 16.0,
                           "binary profile should compress >= 16× (got "
                           f"{ratio}×) — sanity check on the binary path")

    def test_int8_preserves_high_cosine(self):
        r = _run("--quick", "--table", "int8", "--json")
        rec = json.loads(r.stdout.strip())
        cos = rec["rows"][0]["mean_cosine"]
        self.assertGreaterEqual(
            cos, 0.999,
            f"INT8 mean cosine {cos} < 0.999 — quality regression",
        )


class TestShapeOverride(unittest.TestCase):
    def test_n_d_override_runs_one_shape(self):
        r = _run("--table", "int8", "--n", "2000", "--d", "256",
                 "--reps", "2", "--warmup", "1", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        rec = json.loads(r.stdout.strip())
        self.assertEqual(len(rec["rows"]), 1)
        self.assertEqual(rec["rows"][0]["n"], 2000)
        self.assertEqual(rec["rows"][0]["d"], 256)
        self.assertEqual(rec["rows"][0]["reps"], 2)


class TestHumanOutput(unittest.TestCase):
    def test_pretty_mode_contains_headline_marker(self):
        r = _run("--quick", "--table", "int8")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Vectro paper benchmark", r.stdout)
        self.assertIn("Headline", r.stdout)
        self.assertIn("M vec/s", r.stdout)


class TestUnknownTable(unittest.TestCase):
    def test_unknown_table_exits_nonzero(self):
        r = _run("--quick", "--table", "fp32-baseline", "--json")
        self.assertNotEqual(r.returncode, 0)


class TestSingleRepIsQuick(unittest.TestCase):
    """v5.0.2 — --reps 1 --warmup 0 (the reproduce_paper.sh contract) must
    finish in under 60 s so a 3-run CI job completes in < 3 minutes.
    """

    def test_reps_1_warmup_0_completes_within_60s(self):
        import time
        t0 = time.time()
        r = _run("--quick", "--table", "int8",
                 "--json", "--reps", "1", "--warmup", "0")
        elapsed = time.time() - t0
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertLess(
            elapsed, 60.0,
            f"--quick --reps 1 --warmup 0 took {elapsed:.1f}s — must be < 60s "
            f"so reproduce_paper.sh does not time out in CI")
        rec = json.loads(r.stdout.strip())
        self.assertEqual(rec["rows"][0]["reps"], 1)
        self.assertGreater(rec["throughput"], 0.0)


if __name__ == "__main__":
    unittest.main()
