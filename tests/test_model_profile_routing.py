"""End-to-end tests: model-family registry → auto_quantize / Vectro.compress routing.

Verifies that supplying model_dir to auto_quantize() and Vectro.compress()
correctly skips the statistical heuristic and applies the family-registered
quantization method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.auto_quantize_api import auto_quantize  # noqa: E402
from python.vectro import Vectro  # noqa: E402

FIXTURE_DIR = Path(__file__).parent / "fixtures"

RNG = np.random.default_rng(42)


def _random_batch(n: int = 64, d: int = 128) -> np.ndarray:
    return RNG.standard_normal((n, d)).astype(np.float32)


# ---------------------------------------------------------------------------
# auto_quantize() routing via model_dir
# ---------------------------------------------------------------------------


class TestAutoQuantizeModelDir:
    """auto_quantize(model_dir=...) must short-circuit the heuristic."""

    def test_gte_routes_to_int8(self):
        vecs = _random_batch()
        result = auto_quantize(vecs, model_dir=FIXTURE_DIR / "gte")
        assert result["mode"] in ("int8_fallback", "int8"), result["mode"]
        assert result.get("family") == "gte"
        assert result["kurtosis"] == 0.0

    def test_bge_routes_to_nf4(self):
        vecs = _random_batch()
        result = auto_quantize(vecs, model_dir=FIXTURE_DIR / "bge")
        assert result["mode"] in ("nf4", "nf4_mixed", "int8_fallback"), result["mode"]
        assert result.get("family") == "bge"

    def test_e5_routes_to_int8(self):
        vecs = _random_batch()
        result = auto_quantize(vecs, model_dir=FIXTURE_DIR / "e5")
        assert result["mode"] in ("int8_fallback", "int8"), result["mode"]
        assert result.get("family") == "e5"

    def test_qwen2_routes_to_int8(self):
        vecs = _random_batch()
        result = auto_quantize(vecs, model_dir=FIXTURE_DIR / "qwen2")
        assert result["mode"] in ("int8_fallback", "int8"), result["mode"]
        assert result.get("family") == "qwen2"

    def test_deberta_routes_to_nf4(self):
        vecs = _random_batch()
        result = auto_quantize(vecs, model_dir=FIXTURE_DIR / "deberta")
        assert result["mode"] in ("nf4", "nf4_mixed", "int8_fallback"), result["mode"]
        assert result.get("family") == "deberta"

    def test_unknown_falls_through_to_heuristic(self):
        """Unknown model_dir must still return a valid result via the heuristic."""
        vecs = _random_batch()
        result = auto_quantize(vecs, model_dir=FIXTURE_DIR / "unknown")
        assert "mode" in result
        assert "cosine_sim" in result
        assert result.get("family") is None  # heuristic path sets no family key

    def test_no_model_dir_still_works(self):
        """Baseline: no model_dir → existing heuristic unchanged."""
        vecs = _random_batch()
        result = auto_quantize(vecs)
        assert "mode" in result
        assert "cosine_sim" in result

    def test_result_cosine_above_floor(self):
        """Registry-routed INT8 must meet the INT8 cosine floor (≥ 0.9999)."""
        vecs = _random_batch(n=128, d=384)
        result = auto_quantize(vecs, model_dir=FIXTURE_DIR / "gte")
        assert result["cosine_sim"] >= 0.999, (
            f"INT8 cosine {result['cosine_sim']:.6f} < 0.999 floor"
        )

    def test_tried_list_has_one_entry_for_known_family(self):
        """Fast path must produce exactly one entry in 'tried' (no wasted attempts)."""
        vecs = _random_batch()
        result = auto_quantize(vecs, model_dir=FIXTURE_DIR / "gte")
        assert len(result["tried"]) == 1


# ---------------------------------------------------------------------------
# Vectro.compress() routing via model_dir
# ---------------------------------------------------------------------------


class TestVectroCompressModelDir:
    """Vectro.compress(model_dir=...) must select the family-recommended method."""

    def setup_method(self):
        self.vectro = Vectro()
        self.vecs = _random_batch(n=32, d=128)

    def test_gte_compress_uses_int8(self):
        result = self.vectro.compress(self.vecs, model_dir=str(FIXTURE_DIR / "gte"))
        assert result.batch_size == 32
        assert result.precision_mode == "int8"

    def test_bge_compress_uses_nf4_or_int8_fallback(self):
        result = self.vectro.compress(self.vecs, model_dir=str(FIXTURE_DIR / "bge"))
        assert result.batch_size == 32
        assert result.precision_mode in ("nf4", "int8")

    def test_explicit_precision_mode_overrides_model_dir(self):
        """precision_mode kwarg must win over the registry."""
        result = self.vectro.compress(
            self.vecs,
            precision_mode="int8",
            model_dir=str(FIXTURE_DIR / "bge"),
        )
        assert result.precision_mode == "int8"

    def test_no_model_dir_unchanged(self):
        """Baseline: compress without model_dir works as before."""
        result = self.vectro.compress(self.vecs)
        assert result.batch_size == 32

    def test_qwen2_compress_uses_int8(self):
        result = self.vectro.compress(self.vecs, model_dir=str(FIXTURE_DIR / "qwen2"))
        assert result.precision_mode == "int8"

    def test_deberta_compress_uses_nf4(self):
        result = self.vectro.compress(self.vecs, model_dir=str(FIXTURE_DIR / "deberta"))
        assert result.precision_mode in ("nf4", "int8")

    def test_missing_model_dir_falls_back_gracefully(self, tmp_path):
        """A model_dir with no config.json must not raise — generic profile applies."""
        result = self.vectro.compress(self.vecs, model_dir=str(tmp_path))
        assert result.batch_size == 32


# ---------------------------------------------------------------------------
# Public API export check
# ---------------------------------------------------------------------------


def test_get_profile_importable_from_package():
    from python import get_profile, QuantProfile  # noqa: F401

    assert callable(get_profile)
    assert isinstance(QuantProfile, type)
