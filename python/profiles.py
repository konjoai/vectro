"""Model-family → QuantProfile registry.

Maps a HuggingFace model directory (containing ``config.json``) to a
:class:`QuantProfile` that specifies the recommended quantization method
and the detected embedding-model family.

This module is separate from ``profiles_api.py``, which handles per-device
performance strategy selection (FAST / BALANCED / QUALITY).  This module
handles *accuracy* strategy selection based on model architecture.

Usage::

    from python.profiles import get_profile

    profile = get_profile("/path/to/gte-large")
    # QuantProfile(family='gte', method='int8', architectures=['NewModel'])

    profile = get_profile("/path/to/some-unknown-model")
    # QuantProfile(family='generic', method='auto', architectures=['...'])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class QuantProfile:
    """Recommended quantization profile for a model family.

    Attributes:
        family: Human-readable family label, e.g. ``"gte"``, ``"e5"``,
            ``"bert"``, ``"generic"``.
        method: Encoding method to prefer: ``"int8"``, ``"nf4"``, or
            ``"auto"`` (let AutoQuantize decide at runtime).
        architectures: The ``architectures`` list from ``config.json``.
            Preserved for debugging and downstream logging.
    """

    family: str
    method: str
    architectures: list[str] = field(default_factory=list, compare=False)

    def __post_init__(self) -> None:
        valid_methods = {"int8", "nf4", "auto"}
        if self.method not in valid_methods:
            raise ValueError(
                f"QuantProfile.method must be one of {valid_methods}, got {self.method!r}"
            )


# ---------------------------------------------------------------------------
# Family detection table
#
# Each entry is (architecture_keywords: frozenset[str], family_name, method).
# Matching is performed via set intersection: the model matches the first row
# whose architecture_keywords shares at least one element with the
# ``architectures`` set from config.json.
#
# Ordering matters — more specific entries must appear before more general ones.
# ---------------------------------------------------------------------------

_FAMILY_TABLE: list[tuple[frozenset[str], str, str]] = [
    # GTE / New-style GTE (e.g. Alibaba-NLP/gte-large-en-v1.5)
    (frozenset({"NewModel", "GteModel"}), "gte", "int8"),
    # BGE (BAAI/bge-*) — match on BGEModel only; plain BertModel is matched by the
    # bert entry below.  Real BGE fixtures carry "BGEModel" as a discriminator.
    (frozenset({"BGEModel"}), "bge", "nf4"),
    # E5 (intfloat/e5-*)
    (frozenset({"XLMRobertaModel", "RobertaModel", "E5Model"}), "e5", "int8"),
    # Qwen2 embedding models (L2-normalized output → INT8 is lossless)
    (frozenset({"Qwen2Model", "Qwen2_5Model", "Qwen2ForSequenceClassification"}), "qwen2", "int8"),
    # DeBERTa family (unnormalized contextual embeddings → NF4 preserves outliers)
    (
        frozenset(
            {
                "DebertaModel",
                "DebertaV2Model",
                "DebertaForSequenceClassification",
                "DebertaV2ForSequenceClassification",
            }
        ),
        "deberta",
        "nf4",
    ),
    # Classic BERT (bert-base/large, all-MiniLM-*, etc.)
    (frozenset({"BertModel"}), "bert", "nf4"),
]


def get_profile(model_dir: str | Path) -> QuantProfile:
    """Return the recommended :class:`QuantProfile` for *model_dir*.

    Reads ``<model_dir>/config.json`` and matches the ``architectures`` list
    against :data:`_FAMILY_TABLE`.  Falls back to
    ``QuantProfile(family="generic", method="auto")`` if no match is found or
    the config cannot be read.

    Args:
        model_dir: Path to the HuggingFace model directory (local checkout or
            snapshot cache entry).

    Returns:
        A :class:`QuantProfile` with ``family``, ``method``, and
        ``architectures`` populated.

    Raises:
        json.JSONDecodeError: If ``config.json`` exists but is malformed.
    """
    config_path = Path(model_dir) / "config.json"
    try:
        config: dict = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError):
        return QuantProfile(family="generic", method="auto")
    arch_list: list[str] = config.get("architectures") or []
    arch_set = frozenset(arch_list)

    for keywords, family, method in _FAMILY_TABLE:
        if arch_set & keywords:
            return QuantProfile(family=family, method=method, architectures=arch_list)

    return QuantProfile(family="generic", method="auto", architectures=arch_list)
