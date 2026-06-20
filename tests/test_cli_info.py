"""Tests for the vectro info --benchmark CLI flag."""

import unittest
from io import StringIO
from unittest.mock import patch


try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.cli import _build_parser, _cmd_info  # noqa: E402


def _parse_info(*extra_args):
    """Parse 'vectro info [extra_args]' and return the namespace."""
    parser = _build_parser()
    return parser.parse_args(["info"] + list(extra_args))


class TestInfoCommand(unittest.TestCase):
    """Tests for _cmd_info and its --benchmark flag."""

    def test_info_no_benchmark_exits_zero(self):
        """vectro info (no flags) should return 0."""
        args = _parse_info()
        with patch("sys.stdout", new_callable=StringIO):
            code = _cmd_info(args)
        self.assertEqual(code, 0)

    def test_info_no_benchmark_flag_false(self):
        args = _parse_info()
        self.assertFalse(args.benchmark)

    def test_info_benchmark_flag_true(self):
        args = _parse_info("--benchmark")
        self.assertTrue(args.benchmark)

    def test_benchmark_prints_throughput_line(self):
        """--benchmark should print a throughput figure and MAE lines."""
        args = _parse_info("--benchmark")

        # Patch time.monotonic to expire immediately so the 5-s loop runs once.
        call_count = [0]
        original_monotonic = __import__("time").monotonic()

        def _fast_clock():
            # First call returns t0, second call returns t0 + 6 (deadline passed).
            call_count[0] += 1
            if call_count[0] == 1:
                return original_monotonic
            return original_monotonic + 6.0  # > deadline (t0 + 5.0)

        buf = StringIO()
        with patch("time.monotonic", side_effect=_fast_clock), patch("sys.stdout", buf):
            code = _cmd_info(args)

        self.assertEqual(code, 0)
        output = buf.getvalue()
        self.assertIn("INT8 throughput", output)
        self.assertIn("INT8 MAE", output)

    def test_benchmark_nf4_mae_present_or_unavailable(self):
        """NF4 MAE line is always printed (either a value or 'unavailable')."""
        args = _parse_info("--benchmark")

        call_count = [0]
        t0 = __import__("time").monotonic()

        def _fast_clock():
            call_count[0] += 1
            return t0 if call_count[0] == 1 else t0 + 6.0

        buf = StringIO()
        with patch("time.monotonic", side_effect=_fast_clock), patch("sys.stdout", buf):
            _cmd_info(args)

        output = buf.getvalue()
        self.assertIn("NF4", output)

    def test_no_benchmark_skips_benchmark_output(self):
        """Without --benchmark no throughput lines should appear."""
        args = _parse_info()
        buf = StringIO()
        with patch("sys.stdout", buf):
            _cmd_info(args)
        output = buf.getvalue()
        self.assertNotIn("INT8 throughput", output)
        self.assertNotIn("Benchmark", output)

    def test_benchmark_output_contains_separator(self):
        """The benchmark block should be clearly separated from env info."""
        args = _parse_info("--benchmark")
        call_count = [0]
        t0 = __import__("time").monotonic()

        def _fast_clock():
            call_count[0] += 1
            return t0 if call_count[0] == 1 else t0 + 6.0

        buf = StringIO()
        with patch("time.monotonic", side_effect=_fast_clock), patch("sys.stdout", buf):
            _cmd_info(args)

        output = buf.getvalue()
        self.assertIn("──", output)


if __name__ == "__main__":
    unittest.main()
