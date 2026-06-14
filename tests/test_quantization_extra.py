"""Tests for python/quantization_extra.py — INT2 and adaptive INT8 quantization."""

import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.interface import QuantizationResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embeddings(n: int = 20, dim: int = 32, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-row cosine similarity between two (n, d) arrays."""
    dot = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean(dot / (norms + 1e-10)))


# ---------------------------------------------------------------------------
# INT2 quantization tests
# ---------------------------------------------------------------------------


class TestQuantizeInt2(unittest.TestCase):
    def test_returns_three_arrays(self):
        from python.quantization_extra import quantize_int2

        emb = _make_embeddings(10, 32)
        out = quantize_int2(emb)
        self.assertEqual(len(out), 3)

    def test_packed_dtype_uint8(self):
        from python.quantization_extra import quantize_int2

        packed, _, _ = quantize_int2(_make_embeddings(10, 32))
        self.assertEqual(packed.dtype, np.uint8)

    def test_scales_dtype_float32(self):
        from python.quantization_extra import quantize_int2

        _, scales, _ = quantize_int2(_make_embeddings(10, 32))
        self.assertEqual(scales.dtype, np.float32)

    def test_zeroes_dtype_float32(self):
        from python.quantization_extra import quantize_int2

        _, _, zeroes = quantize_int2(_make_embeddings(10, 32))
        self.assertEqual(zeroes.dtype, np.float32)

    def test_packed_shape_n_rows(self):
        from python.quantization_extra import quantize_int2

        n, dim = 8, 20
        packed, _, _ = quantize_int2(_make_embeddings(n, dim))
        self.assertEqual(packed.shape[0], n)

    def test_packed_cols_ceil_div_4(self):
        """packed has ceil(dim/4) columns."""
        import math
        from python.quantization_extra import quantize_int2

        n, dim = 8, 20
        packed, _, _ = quantize_int2(_make_embeddings(n, dim))
        self.assertEqual(packed.shape[1], math.ceil(dim / 4))

    def test_scales_shape(self):
        from python.quantization_extra import quantize_int2

        n, dim, group_size = 8, 32, 16
        _, scales, _ = quantize_int2(_make_embeddings(n, dim), group_size=group_size)
        n_groups = int(np.ceil(dim / group_size))
        self.assertEqual(scales.shape, (n, n_groups))

    def test_raises_on_1d_input(self):
        from python.quantization_extra import quantize_int2

        with self.assertRaises(ValueError):
            quantize_int2(np.ones(32, dtype=np.float32))

    def test_compression_factor(self):
        """Packed representation is ≤ 1/3 of the original bit-width (float32→int2)."""
        from python.quantization_extra import quantize_int2

        n, dim = 100, 64
        emb = _make_embeddings(n, dim)
        packed, _, _ = quantize_int2(emb, group_size=32)
        original_bytes = emb.nbytes
        packed_bytes = packed.nbytes
        ratio = original_bytes / packed_bytes
        self.assertGreater(ratio, 7.0)  # floor for float32→int2


class TestPackUnpackInt2(unittest.TestCase):
    def test_lossless_roundtrip(self):
        """_pack_int2 → _unpack_int2 preserves every ternary value."""
        from python.quantization_extra import _pack_int2, _unpack_int2

        rng = np.random.default_rng(55)
        n, dim = 12, 36
        # Ternary values {0, 1, 2}
        q = rng.integers(0, 3, size=(n, dim), dtype=np.uint8)
        packed = _pack_int2(q, dim)
        recovered = _unpack_int2(packed, dim)
        np.testing.assert_array_equal(recovered, q)

    def test_pack_shape(self):
        import math
        from python.quantization_extra import _pack_int2

        n, d = 5, 17
        q = np.ones((n, d), dtype=np.uint8)
        packed = _pack_int2(q, d)
        self.assertEqual(packed.shape, (n, math.ceil(d / 4)))

    def test_unpack_shape(self):
        from python.quantization_extra import _pack_int2, _unpack_int2

        n, d = 5, 20
        q = np.zeros((n, d), dtype=np.uint8)
        packed = _pack_int2(q, d)
        out = _unpack_int2(packed, d)
        self.assertEqual(out.shape, (n, d))


class TestDequantizeInt2(unittest.TestCase):
    def test_output_shape(self):
        from python.quantization_extra import quantize_int2, dequantize_int2

        n, dim = 10, 32
        emb = _make_embeddings(n, dim)
        packed, scales, zeroes = quantize_int2(emb)
        out = dequantize_int2(packed, scales, zeroes, vector_dim=dim)
        self.assertEqual(out.shape, (n, dim))

    def test_output_dtype_float32(self):
        from python.quantization_extra import quantize_int2, dequantize_int2

        emb = _make_embeddings(8, 32)
        packed, scales, zeroes = quantize_int2(emb)
        out = dequantize_int2(packed, scales, zeroes, vector_dim=32)
        self.assertEqual(out.dtype, np.float32)

    def test_reasonable_cosine_similarity(self):
        """INT2 reconstruction should yield cosine similarity > 0.75."""
        from python.quantization_extra import quantize_int2, dequantize_int2

        n, dim = 50, 64
        emb = _make_embeddings(n, dim)
        packed, scales, zeroes = quantize_int2(emb, group_size=16)
        out = dequantize_int2(packed, scales, zeroes, group_size=16, vector_dim=dim)
        cos_sim = _cosine_similarity(emb, out)
        self.assertGreater(cos_sim, 0.75, msg=f"INT2 cosine similarity {cos_sim:.4f} below 0.75")

    def test_vector_dim_inferred_from_packed(self):
        """vector_dim=None should still produce correct output shape."""
        from python.quantization_extra import quantize_int2, dequantize_int2

        n, dim = 6, 32
        emb = _make_embeddings(n, dim)
        packed, scales, zeroes = quantize_int2(emb)
        # dim inferred as packed.shape[1]*4 which matches dim=32
        out = dequantize_int2(packed, scales, zeroes)
        self.assertEqual(out.shape[1], packed.shape[1] * 4)

    def test_zero_embeddings_reconstruct_near_zero(self):
        """Quantizing zeros should give near-zero reconstruction."""
        from python.quantization_extra import quantize_int2, dequantize_int2

        n, dim = 4, 16
        emb = np.zeros((n, dim), dtype=np.float32)
        packed, scales, zeroes = quantize_int2(emb)
        out = dequantize_int2(packed, scales, zeroes, vector_dim=dim)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Adaptive quantization tests
# ---------------------------------------------------------------------------


class TestQuantizeAdaptive(unittest.TestCase):
    def test_returns_quantization_result(self):
        from python.quantization_extra import quantize_adaptive

        emb = _make_embeddings(10, 32)
        result = quantize_adaptive(emb)
        self.assertIsInstance(result, QuantizationResult)

    def test_precision_mode_int8(self):
        from python.quantization_extra import quantize_adaptive

        result = quantize_adaptive(_make_embeddings(10, 32))
        self.assertEqual(result.precision_mode, "int8")

    def test_output_shape(self):
        from python.quantization_extra import quantize_adaptive

        n, dim = 15, 48
        result = quantize_adaptive(_make_embeddings(n, dim))
        self.assertEqual(result.quantized.shape, (n, dim))
        self.assertEqual(result.dims, dim)
        self.assertEqual(result.n, n)

    def test_scales_nonempty(self):
        from python.quantization_extra import quantize_adaptive

        result = quantize_adaptive(_make_embeddings(10, 32))
        self.assertGreater(result.scales.size, 0)

    def test_group_size_zero_uses_full_dim(self):
        """Default group_size=0 → full-vector groups → per-row scale (n,)."""
        from python.quantization_extra import quantize_adaptive

        n, dim = 8, 32
        result = quantize_adaptive(_make_embeddings(n, dim), group_size=0)
        self.assertEqual(result.group_size, dim)

    def test_group_size_applied(self):
        """Explicit group_size is stored on the result."""
        from python.quantization_extra import quantize_adaptive

        n, dim, gs = 8, 32, 8
        result = quantize_adaptive(_make_embeddings(n, dim), group_size=gs)
        self.assertEqual(result.group_size, gs)

    def test_raises_for_unsupported_bits(self):
        from python.quantization_extra import quantize_adaptive

        with self.assertRaises(ValueError):
            quantize_adaptive(_make_embeddings(4, 16), bits=4)

    def test_raises_for_1d_input(self):
        from python.quantization_extra import quantize_adaptive

        with self.assertRaises(ValueError):
            quantize_adaptive(np.ones(32, dtype=np.float32))

    def test_adaptive_limits_outlier_distortion(self):
        """Adaptive quantization maintains high quality on data with within-row outliers.

        Constructs embeddings where the first element of every vector is a dominant
        outlier (~50×), so standard per-row max scaling maps all other elements to
        near-zero quantized values.  Adaptive clipping keeps normal values in a
        useful range, so reconstruction fidelity is preserved.
        """
        from python.quantization_extra import quantize_adaptive

        rng = np.random.default_rng(99)
        n, dim = 40, 64
        emb = rng.standard_normal((n, dim)).astype(np.float32)
        # Make the first element of every row a massive outlier
        emb[:, 0] *= 50.0

        result = quantize_adaptive(emb, clip_ratio=3.0)
        q = result.quantized
        s = result.scales
        if s.ndim == 1:
            recon = q.astype(np.float32) * s[:, np.newaxis]
        else:
            gs = result.group_size
            recon = q.astype(np.float32) * np.repeat(s, gs, axis=1)[:, :dim]

        # Compare only non-outlier columns
        cos = _cosine_similarity(emb[:, 1:], recon[:, 1:])
        self.assertGreater(
            cos, 0.95, msg=f"Adaptive cos-sim on non-outlier cols = {cos:.4f}, expected > 0.95"
        )

    def test_cosine_similarity_high_on_normal_data(self):
        """Adaptive should maintain high quality on normal distribution data."""
        from python.quantization_extra import quantize_adaptive

        n, dim = 50, 64
        emb = _make_embeddings(n, dim)
        result = quantize_adaptive(emb)
        q = result.quantized
        s = result.scales
        if s.ndim == 1:
            recon = q.astype(np.float32) * s[:, np.newaxis]
        else:
            recon = q.astype(np.float32) * np.repeat(s, dim // s.shape[1], axis=1)[:, :dim]
        cos_sim = _cosine_similarity(emb, recon)
        self.assertGreater(cos_sim, 0.99)


if __name__ == "__main__":
    unittest.main()
