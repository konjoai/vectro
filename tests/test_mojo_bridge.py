"""
Tests for python/_mojo_bridge.py — verifies that all quantization hot paths
actually dispatch to the compiled Mojo binary rather than falling through to
NumPy.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python import _mojo_bridge as mb  # noqa: E402

# Skip the entire module if the binary isn't built yet.
pytestmark = pytest.mark.skipif(
    not mb.is_available(),
    reason="vectro_quantizer binary not found — run: pixi run build-mojo",
)

RNG = np.random.default_rng(0xDEADBEEF)


# ── helpers ────────────────────────────────────────────────────────────────────


def _rand_vecs(n: int, d: int) -> np.ndarray:
    return RNG.standard_normal((n, d)).astype(np.float32)


def _supports_pq_pipe() -> bool:
    """Return True when the current binary exposes pipe pq encode/decode."""
    if not mb.is_available():
        return False
    try:
        vecs = _rand_vecs(1, 4)
        centroids = _rand_vecs(2, 4).reshape(1, 2, 4)
        _ = mb.pq_encode(vecs, centroids)
        return True
    except RuntimeError as exc:
        msg = str(exc)
        if "Unknown op/cmd: pq/encode" in msg or "Unknown op/cmd: pq/decode" in msg:
            return False
        raise


# ── is_available / binary_path ────────────────────────────────────────────────


def test_is_available():
    assert mb.is_available() is True


def test_binary_path_is_executable():
    p = pathlib.Path(mb.binary_path())
    assert p.exists()
    assert p.is_file()


# ── INT8 ──────────────────────────────────────────────────────────────────────


class TestInt8:
    def test_shapes(self):
        vecs = _rand_vecs(10, 64)
        q, scales = mb.int8_quantize(vecs)
        assert q.shape == (10, 64)
        assert q.dtype == np.int8
        assert scales.shape == (10,)
        assert scales.dtype == np.float32

    def test_quantize_range(self):
        vecs = _rand_vecs(32, 128)
        q, _ = mb.int8_quantize(vecs)
        assert int(q.min()) >= -127
        assert int(q.max()) <= 127

    def test_round_trip_accuracy(self):
        vecs = _rand_vecs(64, 128)
        q, scales = mb.int8_quantize(vecs)
        recon = mb.int8_reconstruct(q, scales)
        assert recon.shape == vecs.shape
        mae = float(np.abs(vecs - recon).mean())
        assert mae < 0.02, f"INT8 round-trip MAE too high: {mae:.6f}"

    def test_1d_input(self):
        vec = _rand_vecs(1, 32).ravel()
        q, scales = mb.int8_quantize(vec.reshape(1, -1))
        recon = mb.int8_reconstruct(q, scales, d=32)
        assert recon.shape == (1, 32)

    def test_reconstruct_infers_d(self):
        vecs = _rand_vecs(4, 16)
        q, scales = mb.int8_quantize(vecs)
        recon = mb.int8_reconstruct(q, scales)
        assert recon.shape == vecs.shape

    def test_zero_vector(self):
        vecs = np.zeros((3, 32), dtype=np.float32)
        q, scales = mb.int8_quantize(vecs)
        recon = mb.int8_reconstruct(q, scales)
        np.testing.assert_allclose(recon, vecs, atol=1e-6)

    def test_large_batch(self):
        vecs = _rand_vecs(1000, 768)
        q, scales = mb.int8_quantize(vecs)
        recon = mb.int8_reconstruct(q, scales)
        assert float(np.abs(vecs - recon).mean()) < 0.02


# ── NF4 ───────────────────────────────────────────────────────────────────────


class TestNf4:
    def test_shapes(self):
        vecs = _rand_vecs(10, 64)
        packed, scales = mb.nf4_encode(vecs)
        half_d = (64 + 1) // 2
        assert packed.shape == (10, half_d)
        assert packed.dtype == np.uint8
        assert scales.shape == (10,)
        assert scales.dtype == np.float32

    def test_odd_dimension_shapes(self):
        vecs = _rand_vecs(4, 7)
        packed, scales = mb.nf4_encode(vecs)
        half_d = (7 + 1) // 2
        assert packed.shape == (4, half_d)
        recon = mb.nf4_decode(packed, scales, 7)
        assert recon.shape == (4, 7)

    def test_round_trip_accuracy(self):
        vecs = _rand_vecs(32, 64)
        packed, scales = mb.nf4_encode(vecs)
        recon = mb.nf4_decode(packed, scales, 64)
        assert recon.shape == vecs.shape
        mae = float(np.abs(vecs - recon).mean())
        assert mae < 0.12, f"NF4 round-trip MAE too high: {mae:.6f}"

    def test_zero_vectors(self):
        vecs = np.zeros((2, 16), dtype=np.float32)
        packed, scales = mb.nf4_encode(vecs)
        recon = mb.nf4_decode(packed, scales, 16)
        assert recon.shape == (2, 16)

    def test_large_batch(self):
        vecs = _rand_vecs(500, 128)
        packed, scales = mb.nf4_encode(vecs)
        recon = mb.nf4_decode(packed, scales, 128)
        assert float(np.abs(vecs - recon).mean()) < 0.12


# ── Binary ────────────────────────────────────────────────────────────────────


class TestBinary:
    def test_shapes(self):
        vecs = _rand_vecs(10, 64)
        packed = mb.bin_encode(vecs)
        bpv = (64 + 7) // 8
        assert packed.shape == (10, bpv)
        assert packed.dtype == np.uint8

    def test_odd_dimension_shapes(self):
        vecs = _rand_vecs(4, 9)
        packed = mb.bin_encode(vecs)
        bpv = (9 + 7) // 8
        assert packed.shape == (4, bpv)
        recon = mb.bin_decode(packed, 9)
        assert recon.shape == (4, 9)

    def test_decode_values_are_plus_minus_one(self):
        vecs = _rand_vecs(16, 32)
        packed = mb.bin_encode(vecs)
        recon = mb.bin_decode(packed, 32)
        unique = np.unique(recon)
        assert set(unique.tolist()).issubset({-1.0, 1.0})

    def test_sign_preservation(self):
        """Positive inputs should decode to +1; negative to -1."""
        vecs = np.array([[1.0, -1.0, 2.0, -0.5]], dtype=np.float32)
        packed = mb.bin_encode(vecs)
        recon = mb.bin_decode(packed, 4)
        expected = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
        np.testing.assert_array_equal(recon, expected)

    def test_large_batch(self):
        vecs = _rand_vecs(1000, 768)
        packed = mb.bin_encode(vecs)
        recon = mb.bin_decode(packed, 768)
        assert set(np.unique(recon).tolist()).issubset({-1.0, 1.0})

    def test_all_positive_encodes_all_ones(self):
        vecs = np.ones((2, 8), dtype=np.float32)
        packed = mb.bin_encode(vecs)
        recon = mb.bin_decode(packed, 8)
        np.testing.assert_array_equal(recon, np.ones_like(recon))

    def test_all_negative_encodes_all_minus_one(self):
        vecs = -np.ones((2, 8), dtype=np.float32)
        packed = mb.bin_encode(vecs)
        recon = mb.bin_decode(packed, 8)
        np.testing.assert_array_equal(recon, -np.ones_like(recon))


# ── Product Quantization (PQ) ───────────────────────────────────────────────


class TestPQ:
    @pytest.mark.skipif(
        not _supports_pq_pipe(), reason="vectro_quantizer binary does not expose pq pipe commands"
    )
    def test_pq_encode_shape(self):
        vecs = _rand_vecs(16, 32)
        M, K = 4, 16
        centroids = _rand_vecs(M * K, 8).reshape(M, K, 8)

        codes = mb.pq_encode(vecs, centroids)
        assert codes.shape == (16, M)
        assert codes.dtype == np.uint8

    @pytest.mark.skipif(
        not _supports_pq_pipe(), reason="vectro_quantizer binary does not expose pq pipe commands"
    )
    def test_pq_decode_shape(self):
        n, M, K, sub_dim = 12, 4, 8, 8
        codes = RNG.integers(0, K, size=(n, M), dtype=np.uint8)
        centroids = _rand_vecs(M * K, sub_dim).reshape(M, K, sub_dim)

        recon = mb.pq_decode(codes, centroids)
        assert recon.shape == (n, M * sub_dim)
        assert recon.dtype == np.float32

    @pytest.mark.skipif(
        not _supports_pq_pipe(), reason="vectro_quantizer binary does not expose pq pipe commands"
    )
    def test_decode_then_encode_identity(self):
        """Vectors decoded from centroid codes should re-encode to same codes."""
        n, M, K, sub_dim = 24, 6, 12, 4
        codes = RNG.integers(0, K, size=(n, M), dtype=np.uint8)

        # Spread centroids to reduce accidental ties on nearest-centroid argmin.
        centroids = (3.0 * _rand_vecs(M * K, sub_dim)).reshape(M, K, sub_dim)

        vecs = mb.pq_decode(codes, centroids)
        recoded = mb.pq_encode(vecs, centroids)
        np.testing.assert_array_equal(recoded, codes)


# ── end-to-end via high-level API ─────────────────────────────────────────────


class TestHighLevelDispatch:
    """Verify that high-level Python APIs route through the Mojo binary."""

    def test_interface_quantize_uses_mojo(self):
        from python.interface import quantize_embeddings, reconstruct_embeddings

        vecs = _rand_vecs(8, 32)
        result = quantize_embeddings(vecs, backend="mojo")
        assert result.quantized.dtype == np.int8
        recon = reconstruct_embeddings(result, backend="mojo")
        assert float(np.abs(vecs - recon).mean()) < 0.02

    def test_interface_auto_selects_mojo_without_squish(self):
        """When squish_quant is absent, 'auto' should fall to Mojo."""
        from python import interface

        if interface._squish_quant is not None:
            pytest.skip("squish_quant is installed; Mojo is not the auto choice")
        vecs = _rand_vecs(8, 32)
        result = interface.quantize_embeddings(vecs, backend="auto")
        assert result.quantized.dtype == np.int8

    def test_nf4_api_uses_mojo(self):
        from python.nf4_api import quantize_nf4, dequantize_nf4

        vecs = _rand_vecs(8, 32)
        packed, scales = quantize_nf4(vecs)
        recon = dequantize_nf4(packed, scales, 32)
        assert float(np.abs(vecs - recon).mean()) < 0.12

    def test_binary_api_uses_mojo(self):
        from python.binary_api import quantize_binary, dequantize_binary

        vecs = _rand_vecs(8, 32)
        packed = quantize_binary(vecs, normalize=False)
        recon = dequantize_binary(packed, 32)
        assert set(np.unique(recon).tolist()).issubset({-1.0, 1.0})

    def test_batch_processor_mojo_backend(self):
        from python.batch_api import VectroBatchProcessor

        proc = VectroBatchProcessor(backend="mojo")
        vecs = _rand_vecs(16, 64)
        result = proc.quantize_batch(vecs)
        assert result.batch_size == 16
        assert result.vector_dim == 64
        assert len(result.quantized_vectors) == 16
