"""Tests for python/benchmark.py — BenchmarkSuite, BenchmarkReport, BenchmarkEntry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.benchmark import BenchmarkReport, BenchmarkSuite  # noqa: E402


def _small_suite(profiles=None) -> BenchmarkSuite:
    """Return a fast suite with a tiny dataset."""
    return BenchmarkSuite(
        n=40,
        dim=16,
        profiles=profiles or ["fast", "balanced"],
        trials=2,
        seed=0,
    )


# ---------------------------------------------------------------------------
# BenchmarkSuite — run() integration
# ---------------------------------------------------------------------------


class TestBenchmarkSuiteRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = _small_suite()
        cls.report = cls.suite.run()

    def test_run_returns_benchmark_report(self):
        self.assertIsInstance(self.report, BenchmarkReport)

    def test_entries_count_matches_profiles(self):
        self.assertEqual(len(self.report.entries), 2)  # "fast" + "balanced"

    def test_entry_throughput_positive(self):
        for entry in self.report.entries:
            self.assertGreater(entry.throughput_vps, 0.0)

    def test_entry_compression_ratio_gt_one(self):
        for entry in self.report.entries:
            self.assertGreater(entry.compression_ratio, 1.0)

    def test_entry_cosine_sim_in_range(self):
        for entry in self.report.entries:
            self.assertGreaterEqual(entry.mean_cosine_sim, 0.9)
            self.assertLessEqual(entry.mean_cosine_sim, 1.0 + 1e-6)

    def test_benchmark_entry_env_fields_populated(self):
        entry = self.report.entries[0]
        self.assertTrue(entry.python_version)
        self.assertTrue(entry.platform)
        self.assertTrue(entry.numpy_version)


# ---------------------------------------------------------------------------
# BenchmarkReport — serialisation
# ---------------------------------------------------------------------------


class TestBenchmarkReportSerialisation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _small_suite().run()

    def test_report_to_dict_has_entries_key(self):
        d = self.report.to_dict()
        self.assertIn("entries", d)
        self.assertIn("vectro_version", d)
        self.assertIn("generated_at", d)

    def test_report_save_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.report.save(tmp_path)
            import json

            data = json.loads(Path(tmp_path).read_text())
            self.assertIn("entries", data)
            self.assertEqual(len(data["entries"]), len(self.report.entries))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_report_save_csv(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.report.save(tmp_path)
            content = Path(tmp_path).read_text()
            self.assertIn("profile", content)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_report_save_unknown_format_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                self.report.save(tmp_path, fmt="yaml")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_report_to_csv_string_has_header(self):
        csv_str = self.report.to_csv_string()
        self.assertIsInstance(csv_str, str)
        self.assertIn("profile", csv_str.lower())

    def test_report_print_summary_no_raise(self):
        # print_summary writes to stdout; just ensure it doesn't raise.
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.report.print_summary()


if __name__ == "__main__":
    unittest.main()
