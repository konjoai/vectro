"""Tests for SQ2 / SQ3 scalar quantization (NumPy reference + Vectro integration).

The NumPy reference must reproduce the `vectro_lib::quant::sq2` / `sq3` bit
layout and reconstruction quality, and be selectable end-to-end through
``Vectro.compress(precision_mode="sq2"|"sq3")``.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.scalar_quant import (  # noqa: E402
    dequantize_sq2,
    dequantize_sq3,
    quantize_sq2,
    quantize_sq3,
    scalar_quant_compression_ratio,
    scalar_quant_stats,
)
from python.vectro import Vectro  # noqa: E402


def _unit_vec(d: int, seed: float) -> np.ndarray:
    """Mirror the Rust `unit_vec(d, seed)` test generator (sin then L2-normalise)."""
    v = np.sin(np.arange(d) * seed + 0.1).astype(np.float32)
    return v / np.linalg.norm(v)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 1.0


class TestScalarQuantCore(unittest.TestCase):
    def test_packed_shapes(self):
        v = np.stack([_unit_vec(768, 0.01), _unit_vec(768, 0.02)])
        p2, s2 = quantize_sq2(v)
        p3, s3 = quantize_sq3(v)
        self.assertEqual(p2.shape, (2, 192))  # ceil(768/4)
        self.assertEqual(p3.shape, (2, 288))  # ceil(768*3/8)
        self.assertEqual(s2.shape, (2,))
        self.assertEqual(s3.shape, (2,))
        self.assertEqual(p2.dtype, np.uint8)
        self.assertEqual(s2.dtype, np.float32)

    def test_roundtrip_quality_matches_rust_thresholds(self):
        # Same generator + thresholds as the Rust sq2/sq3 cosine-quality tests.
        v = _unit_vec(768, 0.01)[np.newaxis]
        cos2 = _cosine(v[0], dequantize_sq2(*quantize_sq2(v), 768)[0])
        cos3 = _cosine(v[0], dequantize_sq3(*quantize_sq3(v), 768)[0])
        self.assertGreaterEqual(cos2, 0.95, f"SQ2 cosine {cos2:.4f}")
        self.assertGreaterEqual(cos3, 0.99, f"SQ3 cosine {cos3:.4f}")

    def test_sq3_better_than_sq2(self):
        v = _unit_vec(512, 0.013)[np.newaxis]
        c2 = _cosine(v[0], dequantize_sq2(*quantize_sq2(v), 512)[0])
        c3 = _cosine(v[0], dequantize_sq3(*quantize_sq3(v), 512)[0])
        self.assertGreater(c3, c2)

    def test_odd_and_boundary_dims(self):
        # Exercise partial trailing byte (SQ2) and cross-byte codes (SQ3).
        for d in (1, 3, 7, 13, 101, 257):
            v = _unit_vec(d, 0.07)[np.newaxis]
            r2 = dequantize_sq2(*quantize_sq2(v), d)
            r3 = dequantize_sq3(*quantize_sq3(v), d)
            self.assertEqual(r2.shape, (1, d))
            self.assertEqual(r3.shape, (1, d))
            self.assertTrue(np.isfinite(r2).all() and np.isfinite(r3).all())

    def test_reconstruction_levels_are_exact(self):
        # Decoded values must lie on the 4 (SQ2) / 8 (SQ3) levels times scale.
        v = _unit_vec(256, 0.02)[np.newaxis]
        p, s = quantize_sq2(v)
        r = dequantize_sq2(p, s, 256)[0]
        levels = s[0] * (np.array([-3, -1, 1, 3], dtype=np.float32) / 4.0)
        self.assertTrue(np.all(np.isin(np.round(r, 6), np.round(levels, 6))))

    def test_zero_vector_is_safe(self):
        v = np.zeros((1, 16), dtype=np.float32)
        r = dequantize_sq2(*quantize_sq2(v), 16)
        self.assertEqual(r.shape, (1, 16))
        self.assertTrue(np.isfinite(r).all())

    def test_stats_and_compression_ratio(self):
        v = np.stack([_unit_vec(768, 0.01 * (i + 1)) for i in range(8)])
        for mode, floor in (("sq2", 0.95), ("sq3", 0.99)):
            stats = scalar_quant_stats(v, mode)
            self.assertEqual(stats["mode"], mode)
            self.assertGreaterEqual(stats["mean_cosine"], floor)
            self.assertGreater(stats["compression_ratio"], 1.0)
            # Reported ratio should track the closed-form ratio.
            self.assertAlmostEqual(
                stats["compression_ratio"],
                scalar_quant_compression_ratio(768, mode),
                delta=0.5,
            )

    def test_stats_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            scalar_quant_stats(np.zeros((1, 8), dtype=np.float32), "sq4")


class TestScalarQuantViaVectro(unittest.TestCase):
    def setUp(self):
        self.vectro = Vectro()
        self.batch = np.stack([_unit_vec(768, 0.01 * (i + 1)) for i in range(6)])
        self.single = _unit_vec(768, 0.05)

    def test_compress_decompress_batch(self):
        for mode, floor in (("sq2", 0.95), ("sq3", 0.99)):
            result = self.vectro.compress(self.batch, precision_mode=mode)
            self.assertEqual(result.precision_mode, mode)
            self.assertGreater(result.compression_ratio, 1.0)
            recon = self.vectro.decompress(result)
            self.assertEqual(recon.shape, self.batch.shape)
            cos = [_cosine(self.batch[i], recon[i]) for i in range(len(self.batch))]
            self.assertGreaterEqual(float(np.mean(cos)), floor)

    def test_compress_decompress_single(self):
        for mode, floor in (("sq2", 0.95), ("sq3", 0.99)):
            result = self.vectro.compress(self.single, precision_mode=mode)
            self.assertEqual(result.precision_mode, mode)
            recon = self.vectro.decompress(result)
            self.assertEqual(recon.shape, (768,))
            self.assertGreaterEqual(_cosine(self.single, recon), floor)

    def test_quality_metrics_roundtrip(self):
        result, quality = self.vectro.compress(
            self.batch, precision_mode="sq3", return_quality_metrics=True
        )
        self.assertEqual(result.precision_mode, "sq3")
        self.assertIsNotNone(quality)


if __name__ == "__main__":
    unittest.main()
