"""Tests for python/onnx_export.py — ONNX INT8 dequantization graph."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()


# ---------------------------------------------------------------------------
# Minimal result stub
# ---------------------------------------------------------------------------


class _QResult(NamedTuple):
    quantized: object
    scales: object
    dims: int
    n: int
    precision_mode: str = "int8"
    group_size: int = 0


def _make_result(n: int = 4, d: int = 8, mode: str = "int8") -> _QResult:
    import numpy as np

    rng = np.random.default_rng(0)
    q = rng.integers(-127, 127, size=(n, d), dtype=np.int8)
    s = rng.uniform(0.01, 1.0, size=(n,)).astype(np.float32)
    return _QResult(quantized=q, scales=s, dims=d, n=n, precision_mode=mode)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None


# ---------------------------------------------------------------------------
# Tests 1–6: always run (mock-based; no onnx install required)
# ---------------------------------------------------------------------------


class TestOnnxExportNoInstall(unittest.TestCase):
    """Tests that do not require onnx to be installed."""

    def test_raises_runtime_error_when_onnx_missing(self):
        """to_onnx_model raises RuntimeError when _HAVE_ONNX is False."""
        import python.onnx_export as mod

        result = _make_result()
        with patch.object(mod, "_HAVE_ONNX", False):
            with self.assertRaises(RuntimeError) as ctx:
                mod.to_onnx_model(result)
        self.assertIn("onnx", str(ctx.exception).lower())

    def test_int4_raises_value_error(self):
        """to_onnx_model raises ValueError for int4 results."""
        import python.onnx_export as mod

        result = _make_result(mode="int4")
        # Patch _HAVE_ONNX to True so we get past the first guard.
        with patch.object(mod, "_HAVE_ONNX", True):
            with self.assertRaises(ValueError) as ctx:
                mod.to_onnx_model(result)
        self.assertIn("INT4", str(ctx.exception))

    def test_export_onnx_writes_file(self):
        """export_onnx calls to_onnx_model and writes serialised bytes to disk."""
        import python.onnx_export as mod

        fake_bytes = b"fake_onnx_model"
        fake_model = MagicMock()
        fake_model.SerializeToString.return_value = fake_bytes

        result = _make_result()
        with patch.object(mod, "to_onnx_model", return_value=fake_model) as mock_build:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                tmp_path = tmp.name
            mod.export_onnx(result, tmp_path)

        mock_build.assert_called_once_with(result)
        written = Path(tmp_path).read_bytes()
        self.assertEqual(written, fake_bytes)
        Path(tmp_path).unlink(missing_ok=True)

    def test_export_onnx_propagates_runtime_error_when_onnx_missing(self):
        """export_onnx propagates RuntimeError when onnx is not installed."""
        import python.onnx_export as mod

        result = _make_result()
        with patch.object(mod, "_HAVE_ONNX", False):
            with self.assertRaises(RuntimeError):
                mod.export_onnx(result, "dummy.onnx")

    def test_cli_export_onnx_subcommand_registered(self):
        """The 'export-onnx' subcommand is registered in the parser."""
        from python.cli import _build_parser

        parser = _build_parser()
        # Parse known args to ensure the subcommand exists without error.
        subparsers_actions = [
            a for a in parser._actions if isinstance(a, __import__("argparse")._SubParsersAction)
        ]
        self.assertTrue(
            any("export-onnx" in subparser._name_parser_map for subparser in subparsers_actions),
            msg="'export-onnx' subcommand not found in parser",
        )

    def test_cli_missing_onnx_returns_nonzero(self):
        """_cmd_export_onnx returns non-zero when onnx is unavailable."""
        import python.onnx_export as mod
        from python.cli import _build_parser, _cmd_export_onnx

        parser = _build_parser()
        args = parser.parse_args(["export-onnx", "in.npz", "out.onnx"])

        with (
            patch.object(mod, "_HAVE_ONNX", False),
            patch(
                "python.cli._load_result_for_export", side_effect=RuntimeError("onnx not installed")
            ),
        ):
            code = _cmd_export_onnx(args)

        self.assertNotEqual(code, 0)


# ---------------------------------------------------------------------------
# Tests 7–10: require onnx
# ---------------------------------------------------------------------------


@unittest.skipUnless(_ONNX_AVAILABLE, "onnx package not installed")
class TestOnnxExportWithInstall(unittest.TestCase):
    """Tests that require the onnx package to be installed."""

    def _build(self, n: int = 4, d: int = 8):
        from python.onnx_export import to_onnx_model

        return to_onnx_model(_make_result(n=n, d=d))

    def test_graph_has_three_nodes(self):
        """The exported graph must contain exactly 3 nodes: Cast, Unsqueeze, Mul."""
        model = self._build()
        node_ops = [n.op_type for n in model.graph.node]
        self.assertEqual(node_ops, ["Cast", "Unsqueeze", "Mul"])

    def test_opset_version_is_17(self):
        """The model must declare opset version 17."""
        model = self._build()
        opsets = {oi.domain: oi.version for oi in model.opset_import}
        self.assertEqual(opsets.get(""), 17)

    def test_input_names_are_correct(self):
        """Graph inputs must be named 'quantized' and 'scales'."""
        model = self._build()
        input_names = [inp.name for inp in model.graph.input]
        self.assertIn("quantized", input_names)
        self.assertIn("scales", input_names)

    def test_output_name_is_reconstructed(self):
        """Graph output must be named 'reconstructed'."""
        model = self._build()
        output_names = [out.name for out in model.graph.output]
        self.assertIn("reconstructed", output_names)


if __name__ == "__main__":
    unittest.main()
