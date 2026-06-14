"""Pipeline checkpointing for vectro — v5.4.0.

Serialises and deserialises a CompressionPipeline to/from a JSON file,
enabling reproducible pipeline configurations and experiment tracking.

Public API
----------
PipelineCheckpoint   — frozen dataclass capturing version, timestamp,
                       stage configs, and arbitrary user metadata.
save_pipeline()      — serialises a CompressionPipeline to a JSON file
                       using an atomic write (tmp → replace).
load_pipeline()      — deserialises JSON and reconstructs a
                       CompressionPipeline with the same stage sequence.
checkpoint_info()    — reads checkpoint metadata without constructing
                       any pipeline object.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Bump this when the JSON schema changes in a breaking way.
_SCHEMA_VERSION = "5.5.0"
_REQUIRED_KEYS = {"version", "created_at", "stage_configs", "metadata"}


# ---------------------------------------------------------------------------
# PipelineCheckpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineCheckpoint:
    """Immutable snapshot of a CompressionPipeline's configuration.

    Attributes
    ----------
    version:        Vectro version string at save time (e.g. "5.4.0").
    created_at:     ISO-8601 UTC timestamp string.
    stage_configs:  Ordered list of per-stage config dicts.
    metadata:       Arbitrary user-supplied metadata dict.
    """

    version: str
    created_at: str
    stage_configs: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# save_pipeline
# ---------------------------------------------------------------------------


def save_pipeline(
    pipeline: Any,
    path: str | os.PathLike[str],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Serialise *pipeline* to a JSON file at *path*.

    Parameters
    ----------
    pipeline:   A ``CompressionPipeline`` instance to serialise.
    path:       Destination file path (created with parent dirs if needed).
    metadata:   Optional dict of user metadata embedded in the checkpoint.

    Raises
    ------
    TypeError:  If *pipeline* is not a ``CompressionPipeline``.
    OSError:    If the file cannot be written.
    """
    from .async_pipeline import CompressionPipeline  # local import avoids circularity

    if not isinstance(pipeline, CompressionPipeline):
        raise TypeError(
            f"save_pipeline expects a CompressionPipeline, got {type(pipeline).__name__}"
        )

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    stage_configs = [_stage_to_config(s) for s in pipeline.stages]
    checkpoint = PipelineCheckpoint(
        version=_SCHEMA_VERSION,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        stage_configs=stage_configs,
        metadata=metadata or {},
    )

    payload = asdict(checkpoint)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    logger.info("Checkpoint saved to %s (%d stage(s))", dest, len(stage_configs))


# ---------------------------------------------------------------------------
# load_pipeline
# ---------------------------------------------------------------------------


def load_pipeline(path: str | os.PathLike[str]) -> Any:
    """Deserialise a JSON checkpoint and reconstruct a CompressionPipeline.

    Parameters
    ----------
    path:   Path to an existing checkpoint JSON file.

    Returns
    -------
    CompressionPipeline with the same stage sequence (by name/mode).

    Raises
    ------
    FileNotFoundError:  If *path* does not exist.
    ValueError:         If the JSON does not conform to the checkpoint schema.
    """
    from .async_pipeline import CompressionPipeline  # local import

    dest = Path(path)
    if not dest.exists():
        raise FileNotFoundError(f"Checkpoint not found: {dest}")

    payload = _read_validated_json(dest)
    stages = [_stage_from_config(cfg) for cfg in payload["stage_configs"]]

    if not stages:
        # Empty pipeline — wrap in a minimal placeholder so the type contract holds.
        # CompressionPipeline normally rejects 0 stages; we bypass via a sentinel.
        return _EmptyPipeline()

    return CompressionPipeline(stages)


# ---------------------------------------------------------------------------
# checkpoint_info
# ---------------------------------------------------------------------------


def checkpoint_info(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Read checkpoint metadata without constructing a pipeline.

    Parameters
    ----------
    path:   Path to an existing checkpoint JSON file.

    Returns
    -------
    Raw dict with keys: ``version``, ``created_at``, ``stage_configs``,
    ``metadata``.

    Raises
    ------
    FileNotFoundError:  If *path* does not exist.
    ValueError:         If the JSON does not conform to the checkpoint schema.
    """
    dest = Path(path)
    if not dest.exists():
        raise FileNotFoundError(f"Checkpoint not found: {dest}")

    return _read_validated_json(dest)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_validated_json(dest: Path) -> Dict[str, Any]:
    """Read and schema-validate the checkpoint JSON from *dest*."""
    try:
        raw = dest.read_text(encoding="utf-8")
        payload: Dict[str, Any] = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint JSON at {dest}: {exc}") from exc

    missing = _REQUIRED_KEYS - set(payload.keys())
    if missing:
        raise ValueError(f"Checkpoint schema missing keys {missing} in {dest}")

    if not isinstance(payload.get("stage_configs"), list):
        raise ValueError(f"'stage_configs' must be a list in checkpoint {dest}")

    return payload


def _stage_to_config(stage: Any) -> Dict[str, Any]:
    """Convert a PipelineStage to a serialisable config dict."""
    if hasattr(stage, "to_config") and callable(stage.to_config):
        return stage.to_config()
    # Fallback: introspect dataclass fields directly.
    cfg: Dict[str, Any] = {"name": getattr(stage, "mode", "unknown")}
    for attr in ("mode", "profile", "group_size", "precision_mode", "seed"):
        val = getattr(stage, attr, None)
        if val is not None:
            cfg[attr] = val
    return cfg


def _stage_from_config(config: Dict[str, Any]) -> Any:
    """Reconstruct a PipelineStage from a config dict."""
    from .async_pipeline import PipelineStage  # local import

    if hasattr(PipelineStage, "from_config") and callable(PipelineStage.from_config):
        return PipelineStage.from_config(config)
    mode = config.get("mode") or config.get("name") or "int8"
    profile = config.get("profile")
    group_size = config.get("group_size")
    return PipelineStage(mode=mode, profile=profile, group_size=group_size)


class _EmptyPipeline:
    """Sentinel for a zero-stage checkpoint (valid schema, no stages)."""

    @property
    def stages(self) -> List[Any]:
        """Return empty stage list."""
        return []
