"""Single-shot latency gate for encode_int8_fast and encode_nf4_fast.

Contract (from ADR-002 / PLAN.md v4.1.0):
  p99 latency < 1 ms for d=768 on Apple Silicon (NEON) and x86-64 (AVX2).

Why 10 000 iterations:
  At ~0.05–0.2 ms per call the full run completes in ≤ 2 s.
  10 000 samples give a stable p99 estimate (100 tail samples).

Skip condition:
  If vectro_py is not installed (e.g. fresh CI before maturin develop) the
  tests are skipped rather than failing with an ImportError.
"""

import time
import statistics

import numpy as np
import pytest

vectro_py = pytest.importorskip(
    "vectro_py", reason="vectro_py not installed — run `maturin develop` first"
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_DIM = 768  # standard embedding dimension
_WARMUP = 1_000  # calls discarded before measurement
_SAMPLES = 10_000  # calls used for statistics
_P99_LIMIT_S = 1e-3  # 1 ms hard ceiling


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _random_vec() -> np.ndarray:
    return np.random.default_rng(42).standard_normal(_DIM).astype(np.float32)


def _warmup_int8(vec: np.ndarray, n: int = _WARMUP) -> None:
    for _ in range(n):
        vectro_py.encode_int8_fast(vec.tolist())


def _warmup_nf4(vec: np.ndarray, n: int = _WARMUP) -> None:
    for _ in range(n):
        vectro_py.encode_nf4_fast(vec.tolist())


def _measure(fn, arg, n: int = _SAMPLES) -> list[float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    return times


def _p99(times: list[float]) -> float:
    idx = int(len(times) * 0.99)
    return sorted(times)[idx]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestEncodeInt8FastLatency:
    """INT8 encode_fast single-shot latency — contract: p99 < 1 ms at d=768."""

    @pytest.fixture(scope="class", autouse=True)
    def warmup(self):
        _warmup_int8(_random_vec())

    def test_p99_sub_1ms(self):
        vec = _random_vec().tolist()
        times = _measure(vectro_py.encode_int8_fast, vec)
        p99 = _p99(times)
        p50 = statistics.median(times)
        mean = statistics.mean(times)
        print(
            f"\nINT8 encode_fast d={_DIM}: "
            f"mean={mean * 1e3:.3f} ms  p50={p50 * 1e3:.3f} ms  p99={p99 * 1e3:.3f} ms"
        )
        assert p99 < _P99_LIMIT_S, (
            f"INT8 encode_fast p99 = {p99 * 1000:.3f} ms exceeds {_P99_LIMIT_S * 1000:.0f} ms limit"
        )

    def test_output_shape_and_dtype(self):
        vec = _random_vec().tolist()
        codes, scale = vectro_py.encode_int8_fast(vec)
        assert len(codes) == _DIM, f"expected {_DIM} codes, got {len(codes)}"
        assert isinstance(scale, float), "scale must be a Python float"
        assert all(-127 <= c <= 127 for c in codes), "codes must be in [-127, 127]"
        assert scale > 0.0, "scale must be positive"

    def test_deterministic(self):
        """Same input must produce identical output on repeated calls."""
        vec = _random_vec().tolist()
        codes_a, scale_a = vectro_py.encode_int8_fast(vec)
        codes_b, scale_b = vectro_py.encode_int8_fast(vec)
        assert codes_a == codes_b
        assert scale_a == scale_b

    def test_zero_vector(self):
        """Zero vector should not panic; scale defaults to 1.0."""
        codes, scale = vectro_py.encode_int8_fast([0.0] * _DIM)
        assert all(c == 0 for c in codes)
        assert scale == 1.0

    def test_roundtrip_cosine_similarity(self):
        """Decoded INT8 vector must have cosine similarity ≥ 0.9999 vs original."""
        import math

        vec = _random_vec().tolist()
        codes, scale = vectro_py.encode_int8_fast(vec)
        reconstructed = [c / 127.0 * scale for c in codes]
        dot = sum(a * b for a, b in zip(vec, reconstructed))
        norm_orig = math.sqrt(sum(x * x for x in vec))
        norm_recon = math.sqrt(sum(x * x for x in reconstructed))
        cosine = dot / (norm_orig * norm_recon)
        assert cosine >= 0.9999, f"INT8 cosine similarity {cosine:.6f} < 0.9999"


class TestEncodeNf4FastLatency:
    """NF4 encode_fast single-shot latency — contract: p99 < 1 ms at d=768."""

    @pytest.fixture(scope="class", autouse=True)
    def warmup(self):
        _warmup_nf4(_random_vec())

    def test_p99_sub_1ms(self):
        vec = _random_vec().tolist()
        times = _measure(vectro_py.encode_nf4_fast, vec)
        p99 = _p99(times)
        p50 = statistics.median(times)
        mean = statistics.mean(times)
        print(
            f"\nNF4 encode_fast d={_DIM}: "
            f"mean={mean * 1e3:.3f} ms  p50={p50 * 1e3:.3f} ms  p99={p99 * 1e3:.3f} ms"
        )
        assert p99 < _P99_LIMIT_S, (
            f"NF4 encode_fast p99 = {p99 * 1000:.3f} ms exceeds {_P99_LIMIT_S * 1000:.0f} ms limit"
        )

    def test_output_shape(self):
        """packed must be ceil(dim/2) bytes; scale > 0; dim == _DIM."""
        import math

        vec = _random_vec().tolist()
        packed, scale, dim = vectro_py.encode_nf4_fast(vec)
        assert dim == _DIM
        assert len(packed) == math.ceil(_DIM / 2)
        assert isinstance(scale, float)
        assert scale > 0.0

    def test_deterministic(self):
        vec = _random_vec().tolist()
        a = vectro_py.encode_nf4_fast(vec)
        b = vectro_py.encode_nf4_fast(vec)
        assert a == b

    def test_zero_vector(self):
        """Zero vector must not panic."""
        packed, scale, dim = vectro_py.encode_nf4_fast([0.0] * _DIM)
        assert dim == _DIM
        assert scale >= 0.0
