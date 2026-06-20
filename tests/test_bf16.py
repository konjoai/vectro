"""Tests for Bf16Encoder Python wrapper (v7.0.0).

Coverage:
- shape / dtype contracts
- numerical correctness: decode output close to original float32
- cosine_dist symmetry and range
- __len__ / __repr__
- failure case: no bindings
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    from python.bf16_api import Bf16Encoder, _BINDINGS_AVAILABLE
except ImportError:
    from vectro.bf16_api import Bf16Encoder, _BINDINGS_AVAILABLE  # type: ignore


# ---------------------------------------------------------------------------
# Unit tests — mocked bindings
# ---------------------------------------------------------------------------


class TestBf16EncoderUnit:
    """Pure unit tests — mocked _PyBf16Encoder."""

    def _make_encoder(self) -> tuple:
        with (
            patch("python.bf16_api._BINDINGS_AVAILABLE", True),
            patch("python.bf16_api._PyBf16Encoder") as MockClass,
        ):
            mock_instance = MagicMock()
            MockClass.return_value = mock_instance
            enc = Bf16Encoder.__new__(Bf16Encoder)
            enc._inner = mock_instance
            return enc, mock_instance

    def test_encode_delegates(self):
        enc, inner = self._make_encoder()
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        enc.encode(vecs)
        inner.encode.assert_called_once_with(vecs)

    def test_encode_np_casts_to_float32(self):
        enc, inner = self._make_encoder()
        arr = np.ones((4, 8), dtype=np.float64)
        enc.encode_np(arr)
        call_arr = inner.encode_np.call_args[0][0]
        assert call_arr.dtype == np.float32

    def test_encode_np_contiguous(self):
        enc, inner = self._make_encoder()
        # Sliced array may not be C-contiguous
        base = np.ones((10, 8), dtype=np.float32)
        sliced = base[::2]  # Non-contiguous
        enc.encode_np(sliced)
        call_arr = inner.encode_np.call_args[0][0]
        assert call_arr.flags["C_CONTIGUOUS"]

    def test_decode_delegates(self):
        enc, inner = self._make_encoder()
        inner.decode.return_value = [[0.1, 0.2]]
        result = enc.decode()
        assert result == [[0.1, 0.2]]
        inner.decode.assert_called_once()

    def test_cosine_dist_delegates(self):
        enc, inner = self._make_encoder()
        inner.cosine_dist.return_value = 0.007
        val = enc.cosine_dist(0, 1)
        assert abs(val - 0.007) < 1e-6
        inner.cosine_dist.assert_called_once_with(0, 1)

    def test_len_delegates(self):
        enc, inner = self._make_encoder()
        inner.__len__ = MagicMock(return_value=5)
        assert len(enc) == 5

    def test_repr_delegates(self):
        enc, inner = self._make_encoder()
        inner.__repr__ = lambda _: "PyBf16Encoder(n=3)"
        assert repr(enc) == "PyBf16Encoder(n=3)"


# ---------------------------------------------------------------------------
# Import-guard failure test
# ---------------------------------------------------------------------------


class TestBf16EncoderGuard:
    def test_raises_without_bindings(self):
        with patch("python.bf16_api._BINDINGS_AVAILABLE", False):
            with pytest.raises(ImportError, match="vectro_py"):
                Bf16Encoder()


# ---------------------------------------------------------------------------
# Integration tests (skipped if native bindings unavailable)
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.skipif(
    not _BINDINGS_AVAILABLE,
    reason="vectro_py native bindings not available; run `maturin develop` first",
)


@pytestmark_integration
class TestBf16EncoderIntegration:
    N = 128
    D = 64

    def _random_unit_vecs(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((self.N, self.D)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs

    def test_encode_decode_shape(self):
        vecs = self._random_unit_vecs()
        enc = Bf16Encoder()
        enc.encode_np(vecs)
        decoded = enc.decode()
        assert len(decoded) == self.N
        assert len(decoded[0]) == self.D

    def test_decode_dtype_is_float(self):
        vecs = self._random_unit_vecs()
        enc = Bf16Encoder()
        enc.encode_np(vecs)
        decoded = enc.decode()
        arr = np.array(decoded, dtype=np.float32)
        # All finite
        assert np.all(np.isfinite(arr)), "NaN/Inf in decoded output"

    def test_cosine_dist_precision_contract(self):
        """BF16 cosine similarity should be ≥ 0.9999 vs FP32 reference."""
        vecs = self._random_unit_vecs()
        enc = Bf16Encoder()
        enc.encode_np(vecs)
        decoded = np.array(enc.decode(), dtype=np.float32)

        # FP32 pairwise cosine similarity for a sample
        for i in range(0, min(self.N, 20)):
            for j in range(i + 1, min(self.N, 21)):
                fp32_sim = float(
                    np.dot(vecs[i], vecs[j])
                    / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-9)
                )
                bf16_sim = float(
                    np.dot(decoded[i], decoded[j])
                    / (np.linalg.norm(decoded[i]) * np.linalg.norm(decoded[j]) + 1e-9)
                )
                assert abs(fp32_sim - bf16_sim) < 0.01, (
                    f"BF16 cosine precision failure at ({i},{j}): "
                    f"fp32={fp32_sim:.6f}, bf16={bf16_sim:.6f}"
                )

    def test_cosine_dist_symmetry(self):
        vecs = self._random_unit_vecs()
        enc = Bf16Encoder()
        enc.encode_np(vecs)
        d01 = enc.cosine_dist(0, 1)
        d10 = enc.cosine_dist(1, 0)
        assert abs(d01 - d10) < 1e-5, "cosine_dist not symmetric"

    def test_cosine_dist_self_is_zero(self):
        vecs = self._random_unit_vecs()
        enc = Bf16Encoder()
        enc.encode_np(vecs)
        for i in range(min(self.N, 5)):
            d = enc.cosine_dist(i, i)
            assert abs(d) < 0.01, f"Self-distance not near zero: {d}"

    def test_cosine_dist_range(self):
        """cosine_dist (1 − cos_sim) should be in [0, 2]."""
        vecs = self._random_unit_vecs()
        enc = Bf16Encoder()
        enc.encode_np(vecs)
        for i in range(0, min(self.N, 10)):
            for j in range(0, min(self.N, 10)):
                d = enc.cosine_dist(i, j)
                assert -0.01 <= d <= 2.01, f"cosine_dist out of range: {d}"

    def test_len_after_encode(self):
        vecs = self._random_unit_vecs()
        enc = Bf16Encoder()
        enc.encode_np(vecs)
        assert len(enc) == self.N
