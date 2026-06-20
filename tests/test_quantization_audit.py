"""Tests for quantization_audit — python/quantization_audit.py (v5.5.0).

Covers all required behaviours:
01  test_report_fields_present
02  test_report_frozen_per_vector_metrics
03  test_identical_vectors_cosine_1
04  test_identical_vectors_l2_error_0
05  test_cosine_similarity_range
06  test_compression_ratio_positive
07  test_n_vectors_matches_input
08  test_mean_cosine_le_1
09  test_p5_le_mean_cosine
10  test_worst_k_indices_length
11  test_worst_k_are_worst
12  test_recall_at_1_range
13  test_recall_at_5_range
14  test_recall_at_10_range
15  test_recall_disabled
16  test_to_dict_json_roundtrip
17  test_summary_returns_str
18  test_dtype_strings_recorded
19  test_shape_mismatch_raises
20  test_seeded_recall_deterministic
"""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.quantization_audit import (
    QuantizationAuditor,
    QuantizationReport,
    VectorPairMetrics,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_N, _D = 200, 64
_ORIGINAL: np.ndarray = _RNG.standard_normal((_N, _D)).astype(np.float32)
# Simulate INT8-style quantization: add small noise to mimic reconstruction error
_COMPRESSED: np.ndarray = (
    _ORIGINAL + _RNG.standard_normal((_N, _D)).astype(np.float32) * 0.05
).astype(np.float32)

_AUDITOR = QuantizationAuditor(worst_k=10)
_REPORT: QuantizationReport = _AUDITOR.run(
    _ORIGINAL, _COMPRESSED, run_recall=True, recall_ks=(1, 5, 10), seed=42
)

# Identical-vector report for lossless checks
_REPORT_IDENTICAL: QuantizationReport = QuantizationAuditor(worst_k=3).run(
    _ORIGINAL, _ORIGINAL.copy(), run_recall=False
)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestQuantizationAudit(unittest.TestCase):
    # 01
    def test_report_fields_present(self) -> None:
        """QuantizationReport must expose all required fields."""
        required = {
            "n_vectors",
            "original_dtype",
            "compressed_dtype",
            "compression_ratio",
            "per_vector",
            "mean_cosine_similarity",
            "min_cosine_similarity",
            "p5_cosine_similarity",
            "mean_l2_error",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "worst_k_indices",
        }
        for f in required:
            self.assertTrue(hasattr(_REPORT, f), f"Missing field: {f}")

    # 02
    def test_report_frozen_per_vector_metrics(self) -> None:
        """VectorPairMetrics must be frozen (immutable)."""
        m = _REPORT.per_vector[0]
        self.assertIsInstance(m, VectorPairMetrics)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            m.cosine_similarity = 0.0  # type: ignore[misc]

    # 03
    def test_identical_vectors_cosine_1(self) -> None:
        """Cosine similarity must be ≈ 1.0 when original == compressed."""
        self.assertAlmostEqual(_REPORT_IDENTICAL.mean_cosine_similarity, 1.0, places=5)

    # 04
    def test_identical_vectors_l2_error_0(self) -> None:
        """L2 error must be ≈ 0 when original == compressed."""
        self.assertAlmostEqual(_REPORT_IDENTICAL.mean_l2_error, 0.0, places=5)

    # 05
    def test_cosine_similarity_range(self) -> None:
        """All per-vector cosine similarities must be in [-1, 1]."""
        for m in _REPORT.per_vector:
            self.assertGreaterEqual(m.cosine_similarity, -1.0 - 1e-6)
            self.assertLessEqual(m.cosine_similarity, 1.0 + 1e-6)

    # 06
    def test_compression_ratio_positive(self) -> None:
        """Compression ratio must be > 0."""
        self.assertGreater(_REPORT.compression_ratio, 0.0)

    # 07
    def test_n_vectors_matches_input(self) -> None:
        """n_vectors must equal the number of rows in the original array."""
        self.assertEqual(_REPORT.n_vectors, _N)
        self.assertEqual(len(_REPORT.per_vector), _N)

    # 08
    def test_mean_cosine_le_1(self) -> None:
        """mean_cosine_similarity must be ≤ 1.0."""
        self.assertLessEqual(_REPORT.mean_cosine_similarity, 1.0 + 1e-6)

    # 09
    def test_p5_le_mean_cosine(self) -> None:
        """p5_cosine_similarity ≤ mean_cosine_similarity ≤ 1.0."""
        self.assertLessEqual(_REPORT.p5_cosine_similarity, _REPORT.mean_cosine_similarity + 1e-6)
        self.assertLessEqual(_REPORT.mean_cosine_similarity, 1.0 + 1e-6)

    # 10
    def test_worst_k_indices_length(self) -> None:
        """worst_k_indices must have exactly worst_k entries."""
        self.assertEqual(len(_REPORT.worst_k_indices), 10)

    # 11
    def test_worst_k_are_worst(self) -> None:
        """The worst_k_indices must have lower cosine than the remaining vectors."""
        worst_set = set(_REPORT.worst_k_indices)
        all_sims = {m.index: m.cosine_similarity for m in _REPORT.per_vector}
        worst_max = max(all_sims[i] for i in worst_set)
        rest_min = min(v for i, v in all_sims.items() if i not in worst_set)
        # The worst group's maximum ≤ rest's minimum (with float tolerance)
        self.assertLessEqual(worst_max, rest_min + 1e-5)

    # 12
    def test_recall_at_1_range(self) -> None:
        """recall_at_1 must be in [0, 1]."""
        self.assertIsNotNone(_REPORT.recall_at_1)
        self.assertGreaterEqual(_REPORT.recall_at_1, 0.0)
        self.assertLessEqual(_REPORT.recall_at_1, 1.0)

    # 13
    def test_recall_at_5_range(self) -> None:
        """recall_at_5 must be in [0, 1]."""
        self.assertIsNotNone(_REPORT.recall_at_5)
        self.assertGreaterEqual(_REPORT.recall_at_5, 0.0)
        self.assertLessEqual(_REPORT.recall_at_5, 1.0)

    # 14
    def test_recall_at_10_range(self) -> None:
        """recall_at_10 must be in [0, 1]."""
        self.assertIsNotNone(_REPORT.recall_at_10)
        self.assertGreaterEqual(_REPORT.recall_at_10, 0.0)
        self.assertLessEqual(_REPORT.recall_at_10, 1.0)

    # 15
    def test_recall_disabled(self) -> None:
        """When run_recall=False all recall fields must be None."""
        report = QuantizationAuditor().run(_ORIGINAL, _COMPRESSED, run_recall=False)
        self.assertIsNone(report.recall_at_1)
        self.assertIsNone(report.recall_at_5)
        self.assertIsNone(report.recall_at_10)

    # 16
    def test_to_dict_json_roundtrip(self) -> None:
        """to_json() must produce valid JSON that round-trips to the expected keys."""
        raw = _REPORT.to_json()
        parsed = json.loads(raw)
        self.assertIn("n_vectors", parsed)
        self.assertIn("per_vector", parsed)
        self.assertEqual(parsed["n_vectors"], _N)
        self.assertEqual(len(parsed["per_vector"]), _N)

    # 17
    def test_summary_returns_str(self) -> None:
        """summary() must return a non-empty string."""
        s = _REPORT.summary()
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 0)

    # 18
    def test_dtype_strings_recorded(self) -> None:
        """original_dtype and compressed_dtype must be recorded as strings."""
        self.assertIsInstance(_REPORT.original_dtype, str)
        self.assertIsInstance(_REPORT.compressed_dtype, str)
        self.assertEqual(_REPORT.original_dtype, str(_ORIGINAL.dtype))
        self.assertEqual(_REPORT.compressed_dtype, str(_COMPRESSED.dtype))

    # 19
    def test_shape_mismatch_raises(self) -> None:
        """run() must raise ValueError when original and compressed shapes differ."""
        wrong = np.zeros((_N + 1, _D), dtype=np.float32)
        with self.assertRaises(ValueError):
            QuantizationAuditor().run(_ORIGINAL, wrong)

    # 20
    def test_seeded_recall_deterministic(self) -> None:
        """Two runs with the same seed must produce identical recall scores."""
        auditor = QuantizationAuditor()
        r1 = auditor.run(_ORIGINAL, _COMPRESSED, run_recall=True, seed=7)
        r2 = auditor.run(_ORIGINAL, _COMPRESSED, run_recall=True, seed=7)
        self.assertAlmostEqual(r1.recall_at_1, r2.recall_at_1, places=8)
        self.assertAlmostEqual(r1.recall_at_5, r2.recall_at_5, places=8)
        self.assertAlmostEqual(r1.recall_at_10, r2.recall_at_10, places=8)


if __name__ == "__main__":
    unittest.main()
