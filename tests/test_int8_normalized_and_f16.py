"""Wave 1.3 + Wave 4 — Python-side coverage for the new INT8 entry points.

* ``CompressionProfile`` carries an ``assume_normalized`` flag (default False).
* ``_rust_bridge.quantize_int8_batch(..., assume_normalized=True)`` routes to
  the single-pass normalised kernel when the Rust extension is built; gracefully
  falls back to the regular path when it is not.
* ``_rust_bridge.quantize_int8_batch_from_f16`` accepts an f16 array and
  produces the same shape contract as the f32 entry point; falls back to a
  Python-side widen + standard encode when the extension predates Wave 4.

The tests skip cleanly if the Rust extension is unavailable on the current
host so this file is safe to keep in the always-on test suite.
"""
from __future__ import annotations

import unittest
from typing import Any

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python import _rust_bridge  # noqa: E402
from python.profiles_api import CompressionProfile, CompressionStrategy  # noqa: E402


# ---------------------------------------------------------------------------
# CompressionProfile.assume_normalized field
# ---------------------------------------------------------------------------

class TestProfileNormalizedField(unittest.TestCase):
    def _make_profile(self, **overrides: Any) -> CompressionProfile:
        defaults = dict(
            name="t",
            strategy=CompressionStrategy.FAST,
            quantization_bits=8,
            range_factor=1.0,
            clipping_percentile=100.0,
            adaptive_scaling=True,
            batch_optimization=True,
            precision_mode="int8",
            error_correction=False,
            simd_enabled=True,
            parallel_processing=True,
            memory_efficient=False,
            preserve_norms=False,
            preserve_angles=False,
            min_similarity_threshold=0.99,
        )
        defaults.update(overrides)
        return CompressionProfile(**defaults)

    def test_default_is_false(self):
        p = self._make_profile()
        self.assertFalse(p.assume_normalized)

    def test_can_set_true(self):
        p = self._make_profile(assume_normalized=True)
        self.assertTrue(p.assume_normalized)

    def test_round_trips_via_to_dict_from_dict(self):
        p = self._make_profile(assume_normalized=True)
        d = p.to_dict()
        self.assertTrue(d["assume_normalized"])
        p2 = CompressionProfile.from_dict(d)
        self.assertTrue(p2.assume_normalized)

    def test_from_dict_legacy_payload_defaults_false(self):
        # Serialised payloads from <v5.0.0 don't carry the flag.  Loading
        # them back must default the field to False rather than KeyError.
        p = self._make_profile()
        d = p.to_dict()
        d.pop("assume_normalized", None)
        p2 = CompressionProfile.from_dict(d)
        self.assertFalse(p2.assume_normalized)


# ---------------------------------------------------------------------------
# _rust_bridge — guarded by extension availability
# ---------------------------------------------------------------------------

@unittest.skipUnless(_rust_bridge.is_available(),
                     "vectro_py Rust extension not built")
class TestRustBridgeNormalized(unittest.TestCase):
    def test_assume_normalized_roundtrip(self):
        rng = np.random.default_rng(seed=0xBEEF)
        n, d = 200, 256
        raw = rng.standard_normal((n, d)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        v = (raw / norms).astype(np.float32)

        codes, scales = _rust_bridge.quantize_int8_batch(v, assume_normalized=True)
        self.assertEqual(codes.shape, (n, d))
        self.assertEqual(scales.shape, (n,))
        # Every row's scale must be the canonical 1/127 constant.
        np.testing.assert_allclose(scales, 1.0 / 127.0, atol=1e-9)

        # Decode + per-row cosine vs original
        decoded = _rust_bridge.dequantize_int8_batch(codes, scales)
        for i in range(n):
            a = v[i]
            b = decoded[i]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            cos = float(np.dot(a, b) / denom)
            self.assertGreaterEqual(
                cos, 0.99,
                f"row {i}: assume_normalized cosine {cos:.5f} < 0.99",
            )

    def test_assume_normalized_does_not_clip(self):
        # Caller contract: row L2-norm ≤ 1 ⇒ no element is clipped.
        v = np.array([[0.6, 0.8]], dtype=np.float32)  # L2 = 1.0, max = 0.8
        codes, scales = _rust_bridge.quantize_int8_batch(v, assume_normalized=True)
        # 0.8 * 127 = 101.6 → 102, well within [-127, 127]
        self.assertTrue(np.all(np.abs(codes) <= 127))

    def test_default_path_unchanged(self):
        v = np.random.default_rng(seed=1).standard_normal((10, 32)).astype(np.float32)
        codes_def, scales_def = _rust_bridge.quantize_int8_batch(v)
        # Scales are abs_max(row)/127 — not the constant 1/127.
        self.assertFalse(np.allclose(scales_def, 1.0 / 127.0))


@unittest.skipUnless(_rust_bridge.is_available(),
                     "vectro_py Rust extension not built")
class TestRustBridgeF16(unittest.TestCase):
    def test_f16_input_roundtrip(self):
        rng = np.random.default_rng(seed=42)
        n, d = 100, 128
        v_f32 = rng.standard_normal((n, d)).astype(np.float32)
        v_f16 = v_f32.astype(np.float16)

        codes, scales = _rust_bridge.quantize_int8_batch_from_f16(v_f16)
        self.assertEqual(codes.shape, (n, d))
        self.assertEqual(scales.shape, (n,))

        # Decoded f16 path should agree with f32 path within an f16 ULP.
        decoded = _rust_bridge.dequantize_int8_batch(codes, scales)
        for i in range(n):
            a = v_f16[i].astype(np.float32)
            b = decoded[i]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                continue
            cos = float(np.dot(a, b) / denom)
            self.assertGreaterEqual(cos, 0.99,
                                    f"f16 row {i}: cos {cos:.5f} < 0.99")


if __name__ == "__main__":
    unittest.main()
