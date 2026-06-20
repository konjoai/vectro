"""Quantization Audit — v5.5.0.

Compares original float32 vectors against their quantized/compressed
counterparts and produces a rich diagnostic report.

Public API
----------
VectorPairMetrics  — frozen dataclass with per-vector quality fields.
RecallResult       — frozen dataclass for a single Recall@K result.
QuantizationReport — mutable dataclass aggregating per-vector + recall stats.
QuantizationAuditor.run() — entry point; validates shapes, runs all metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorPairMetrics:
    """Per-vector quality metrics between original and reconstructed vectors.

    Attributes
    ----------
    index:             Position in the batch.
    cosine_similarity: FP32 dot product of L2-normalised vectors. ∈ [-1, 1].
    l2_error:          ‖original − reconstructed‖₂.
    relative_error:    l2_error / (‖original‖₂ + ε),  ε = 1e-12.
    """

    index: int
    cosine_similarity: float
    l2_error: float
    relative_error: float


@dataclass(frozen=True)
class RecallResult:
    """Recall@K result for a single query.

    Attributes
    ----------
    k:           Neighbourhood size.
    recall:      Fraction of true top-K neighbours recovered. ∈ [0, 1].
    query_index: Row index of the query vector in the original array.
    """

    k: int
    recall: float
    query_index: int


@dataclass
class QuantizationReport:
    """Full audit report produced by :meth:`QuantizationAuditor.run`.

    Attributes
    ----------
    n_vectors:               Number of vectors audited.
    original_dtype:          NumPy dtype string for the original array.
    compressed_dtype:        NumPy dtype string for the compressed array.
    compression_ratio:       original_bytes / compressed_bytes.
    per_vector:              Per-vector metrics for all N vectors.
    mean_cosine_similarity:  Mean cosine similarity across all vectors.
    min_cosine_similarity:   Minimum cosine similarity.
    p5_cosine_similarity:    5th-percentile cosine similarity (worst-case tail).
    mean_l2_error:           Mean L2 reconstruction error.
    recall_at_1:             Recall@1 (None when run_recall=False).
    recall_at_5:             Recall@5 (None when run_recall=False).
    recall_at_10:            Recall@10 (None when run_recall=False).
    worst_k_indices:         Indices of k vectors with lowest cosine similarity.
    """

    n_vectors: int
    original_dtype: str
    compressed_dtype: str
    compression_ratio: float

    per_vector: list[VectorPairMetrics]

    mean_cosine_similarity: float
    min_cosine_similarity: float
    p5_cosine_similarity: float
    mean_l2_error: float

    recall_at_1: float | None
    recall_at_5: float | None
    recall_at_10: float | None

    worst_k_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "n_vectors": self.n_vectors,
            "original_dtype": self.original_dtype,
            "compressed_dtype": self.compressed_dtype,
            "compression_ratio": self.compression_ratio,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "min_cosine_similarity": self.min_cosine_similarity,
            "p5_cosine_similarity": self.p5_cosine_similarity,
            "mean_l2_error": self.mean_l2_error,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "worst_k_indices": self.worst_k_indices,
            "per_vector": [
                {
                    "index": m.index,
                    "cosine_similarity": m.cosine_similarity,
                    "l2_error": m.l2_error,
                    "relative_error": m.relative_error,
                }
                for m in self.per_vector
            ],
        }

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        """Human-readable one-paragraph summary of audit results."""
        recall_line = (
            f"Recall@1/5/10: {self.recall_at_1:.4f}/{self.recall_at_5:.4f}/{self.recall_at_10:.4f}."
            if self.recall_at_1 is not None
            else "Recall evaluation was skipped."
        )
        return (
            f"Audit of {self.n_vectors} vectors ({self.original_dtype} → "
            f"{self.compressed_dtype}, {self.compression_ratio:.2f}× compression). "
            f"Mean cosine similarity: {self.mean_cosine_similarity:.6f} "
            f"(min {self.min_cosine_similarity:.6f}, "
            f"p5 {self.p5_cosine_similarity:.6f}). "
            f"Mean L2 error: {self.mean_l2_error:.6f}. "
            f"{recall_line} "
            f"Worst-{len(self.worst_k_indices)} vector indices: {self.worst_k_indices}."
        )


# ---------------------------------------------------------------------------
# QuantizationAuditor
# ---------------------------------------------------------------------------


class QuantizationAuditor:
    """Compares original and compressed vector arrays, computes quality metrics.

    All arithmetic accumulated in FP32. Cosine similarity computed via
    np.einsum for numerical stability. Recall@K uses exact brute-force search
    (no approximation) — suitable for audit sets up to ~100 K vectors.

    Parameters
    ----------
    worst_k: How many worst-case vectors to record in the report.
    """

    def __init__(self, worst_k: int = 10) -> None:
        if worst_k < 1:
            raise ValueError(f"worst_k must be ≥ 1, got {worst_k}")
        self._worst_k = worst_k

    def run(
        self,
        original: np.ndarray,
        compressed: np.ndarray,
        *,
        run_recall: bool = True,
        recall_ks: tuple[int, ...] = (1, 5, 10),
        recall_sample_size: int = 100,
        seed: int = 42,
    ) -> QuantizationReport:
        """Run the full audit and return a :class:`QuantizationReport`.

        Parameters
        ----------
        original:           Shape (N, D), any float dtype.
        compressed:         Shape (N, D), any numeric dtype.
        run_recall:         Whether to evaluate Recall@K metrics.
        recall_ks:          K values for Recall@K evaluation.
        recall_sample_size: Number of query vectors sampled for recall eval.
        seed:               RNG seed (logged in the report summary).
        """
        _validate_shapes(original, compressed)
        logger.info(
            "Audit started — N=%d D=%d seed=%d run_recall=%s",
            original.shape[0],
            original.shape[1],
            seed,
            run_recall,
        )

        orig_f32 = original.astype(np.float32)
        comp_f32 = compressed.astype(np.float32)

        per_vector = _compute_per_vector(orig_f32, comp_f32, self)
        cos_sims = np.array([m.cosine_similarity for m in per_vector], dtype=np.float32)
        l2_errors = np.array([m.l2_error for m in per_vector], dtype=np.float32)

        worst_indices = _worst_k_indices(cos_sims, self._worst_k)
        recall_map = (
            _run_recall_eval(orig_f32, comp_f32, recall_ks, recall_sample_size, seed)
            if run_recall
            else {}
        )

        orig_bytes = orig_f32.nbytes
        comp_bytes = compressed.nbytes
        compression_ratio = float(orig_bytes) / max(float(comp_bytes), 1.0)

        return QuantizationReport(
            n_vectors=orig_f32.shape[0],
            original_dtype=str(original.dtype),
            compressed_dtype=str(compressed.dtype),
            compression_ratio=compression_ratio,
            per_vector=per_vector,
            mean_cosine_similarity=float(np.mean(cos_sims)),
            min_cosine_similarity=float(np.min(cos_sims)),
            p5_cosine_similarity=float(np.percentile(cos_sims, 5)),
            mean_l2_error=float(np.mean(l2_errors)),
            recall_at_1=recall_map.get(1),
            recall_at_5=recall_map.get(5),
            recall_at_10=recall_map.get(10),
            worst_k_indices=worst_indices,
        )

    def _cosine_similarities(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return per-row cosine similarity between arrays *a* and *b*.

        Math
        ----
        cos(a_i, b_i) = (a_i / ‖a_i‖) · (b_i / ‖b_i‖)
        Both inputs are L2-normalised in FP32 before the dot product.
        np.einsum('ij,ij->i', ...) avoids constructing an N×N matrix.
        """
        a_f32 = a.astype(np.float32)
        b_f32 = b.astype(np.float32)
        # L2-normalise; add ε to avoid division by zero
        a_norms = np.linalg.norm(a_f32, axis=1, keepdims=True) + 1e-12
        b_norms = np.linalg.norm(b_f32, axis=1, keepdims=True) + 1e-12
        a_hat = a_f32 / a_norms
        b_hat = b_f32 / b_norms
        return np.einsum("ij,ij->i", a_hat, b_hat).astype(np.float32)

    def _recall_at_k(
        self,
        original: np.ndarray,
        compressed: np.ndarray,
        k: int,
        query_indices: np.ndarray,
    ) -> float:
        """Brute-force Recall@k averaged over query_indices.

        Math
        ----
        For each query q_i:
          true_nbrs = argsort(-original @ q_i)[1 : k+1]   (exclude self)
          comp_nbrs = argsort(-compressed @ q_i)[1 : k+1]
          recall_i  = |true_nbrs ∩ comp_nbrs| / k
        Recall@k = mean_i(recall_i)

        Both arrays must already be FP32 (caller's responsibility).
        """
        recalls: list[float] = []
        for qi in query_indices:
            query = original[qi]  # shape (D,)
            true_scores = original @ query  # (N,)
            comp_scores = compressed @ query  # (N,)
            true_scores[qi] = -np.inf  # exclude self
            comp_scores[qi] = -np.inf
            effective_k = min(k, true_scores.shape[0] - 1)
            true_top = set(np.argpartition(true_scores, -effective_k)[-effective_k:])
            comp_top = set(np.argpartition(comp_scores, -effective_k)[-effective_k:])
            recalls.append(len(true_top & comp_top) / effective_k)
        return float(np.mean(recalls)) if recalls else 0.0


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------


def _validate_shapes(original: np.ndarray, compressed: np.ndarray) -> None:
    """Raise ValueError when arrays have incompatible shapes."""
    if original.shape != compressed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} ≠ compressed {compressed.shape}"
        )
    if original.ndim != 2:
        raise ValueError(f"Expected 2-D arrays, got ndim={original.ndim}")


def _compute_per_vector(
    orig: np.ndarray, comp: np.ndarray, auditor: QuantizationAuditor
) -> list[VectorPairMetrics]:
    """Compute per-row cosine similarity, L2 error, and relative error."""
    cos_sims = auditor._cosine_similarities(orig, comp)
    diff = orig - comp
    l2_errors = np.linalg.norm(diff, axis=1).astype(np.float32)
    orig_norms = np.linalg.norm(orig, axis=1).astype(np.float32)
    rel_errors = l2_errors / (orig_norms + 1e-12)
    return [
        VectorPairMetrics(
            index=i,
            cosine_similarity=float(cos_sims[i]),
            l2_error=float(l2_errors[i]),
            relative_error=float(rel_errors[i]),
        )
        for i in range(orig.shape[0])
    ]


def _worst_k_indices(cos_sims: np.ndarray, k: int) -> list[int]:
    """Return the k indices with the lowest cosine similarity values."""
    effective_k = min(k, cos_sims.shape[0])
    # argpartition gives the k smallest elements; sort them for determinism
    indices = np.argpartition(cos_sims, effective_k)[:effective_k]
    return sorted(indices.tolist(), key=lambda i: cos_sims[i])


def _run_recall_eval(
    orig: np.ndarray,
    comp: np.ndarray,
    ks: tuple[int, ...],
    sample_size: int,
    seed: int,
) -> dict[int, float]:
    """Sample query vectors, compute Recall@K for each k in *ks*."""
    n = orig.shape[0]
    rng = np.random.default_rng(seed)
    logger.info("Recall eval seed=%d sample_size=%d", seed, sample_size)
    effective_sample = min(sample_size, n)
    query_indices = rng.choice(n, size=effective_sample, replace=False)

    auditor = QuantizationAuditor.__new__(QuantizationAuditor)
    auditor._worst_k = 10  # dummy — only _recall_at_k is used here

    return {k: auditor._recall_at_k(orig, comp, k, query_indices) for k in ks}
