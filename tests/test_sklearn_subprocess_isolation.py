"""Subprocess isolation tests for sklearn-backed quantization paths.

These tests verify that sklearn-dependent code paths run correctly in a fresh
interpreter process, which avoids in-process C-extension reload hazards.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent.parent


def _run_python_snippet(snippet: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_ROOT}" if not existing else f"{_ROOT}{os.pathsep}{existing}"
    return subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_ROOT),
        check=False,
    )


def _require_sklearn() -> None:
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn not installed")


def test_rq_api_train_encode_decode_in_subprocess() -> None:
    _require_sklearn()
    snippet = textwrap.dedent(
        """
        import numpy as np
        from python.rq_api import ResidualQuantizer

        rng = np.random.default_rng(7)
        train = rng.standard_normal((32, 16)).astype(np.float32)
        rq = ResidualQuantizer(n_passes=1, n_subspaces=4, n_centroids=8, seed=0)
        rq.train(train)

        codes = rq.encode(train[:6])
        recon = rq.decode(codes)
        assert recon.shape == (6, 16)
        assert recon.dtype == np.float32
        print("rq_subprocess_ok")
        """
    )
    result = _run_python_snippet(snippet)
    assert result.returncode == 0, (
        f"RQ subprocess execution failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "rq_subprocess_ok" in result.stdout


def test_v3_rq_profile_in_subprocess() -> None:
    _require_sklearn()
    snippet = textwrap.dedent(
        """
        import numpy as np
        from python.v3_api import VectroV3

        rng = np.random.default_rng(19)
        train = rng.standard_normal((80, 16)).astype(np.float32)
        db = rng.standard_normal((10, 16)).astype(np.float32)

        v3 = VectroV3(profile="rq-3pass")
        v3.train_rq(train, n_subspaces=4, n_passes=1)

        packed = v3.compress(db)
        recon = v3.decompress(packed)
        assert recon.shape == db.shape
        assert recon.dtype == np.float32
        print("v3_rq_subprocess_ok")
        """
    )
    result = _run_python_snippet(snippet)
    assert result.returncode == 0, (
        f"V3 RQ subprocess execution failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "v3_rq_subprocess_ok" in result.stdout


def test_repeated_subprocess_runs_are_stable() -> None:
    _require_sklearn()
    snippet = textwrap.dedent(
        """
        import numpy as np
        from python.rq_api import ResidualQuantizer

        rng = np.random.default_rng(123)
        train = rng.standard_normal((24, 8)).astype(np.float32)
        rq = ResidualQuantizer(n_passes=1, n_subspaces=2, n_centroids=4, seed=1)
        rq.train(train)
        print("ok")
        """
    )

    first = _run_python_snippet(snippet)
    second = _run_python_snippet(snippet)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "ok" in first.stdout
    assert "ok" in second.stdout
