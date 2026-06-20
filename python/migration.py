"""Vectro artifact migration and inspection tooling.

Provides a Python API and CLI for inspecting compressed vector artifacts,
detecting their format version, and upgrading from v1 to v2 format.

Breaking changes from v1.x → v2.0
-----------------------------------
* ``storage_format_version`` field added (defaults to ``1`` when absent)
* ``precision_mode`` field added (defaults to ``"int8"`` for legacy files)
* ``group_size`` field added (defaults to ``0``)
* ``metadata_json`` field added (contains provenance, creation time, etc.)

Python API
----------
::

    from python.migration import inspect_artifact, upgrade_artifact

    info = inspect_artifact("old.npz")
    print(info["format_version"])   # 1 or 2
    print(info["needs_upgrade"])    # True if format_version < 2

    upgrade_artifact("old.npz", "upgraded.npz")

CLI
---
::

    python -m python.migration inspect artifact.npz
    python -m python.migration upgrade old.npz upgraded.npz [--dry-run]
    python -m python.migration validate artifact.npz
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

# Current v2 format constants — kept in sync with vectro.py
_CURRENT_FORMAT_VERSION = 2
_STORAGE_FORMAT_NAME = "vectro_npz"

# V1 field defaults applied during upgrade
_V1_DEFAULTS = {
    "precision_mode": "int8",
    "group_size": 0,
    "storage_format": _STORAGE_FORMAT_NAME,
    "storage_format_version": _CURRENT_FORMAT_VERSION,
}


# ---------------------------------------------------------------------------
# Public Python API
# ---------------------------------------------------------------------------


def inspect_artifact(path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect a Vectro compressed artifact and return metadata.

    Args:
        path: Path to a ``.npz`` artifact written by :meth:`~vectro.Vectro.save_compressed`.

    Returns:
        A dictionary with the following keys:

        * ``path`` (str) — absolute file path
        * ``file_size_bytes`` (int) — file size on disk
        * ``format_version`` (int) — ``storage_format_version`` (1 if absent)
        * ``artifact_type`` (str) — ``"single"`` or ``"batch"``
        * ``n_vectors`` (int) — number of compressed vectors
        * ``vector_dim`` (int) — dimension of each vector
        * ``precision_mode`` (str) — ``"int8"``, ``"int4"``, etc.
        * ``group_size`` (int) — quantization group size
        * ``compression_ratio`` (float) — achieved compression ratio
        * ``needs_upgrade`` (bool) — ``True`` when format_version < current
        * ``metadata`` (dict or None) — parsed ``metadata_json`` field
        * ``fields`` (list[str]) — all array keys present in the file
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    data = np.load(path, allow_pickle=False)
    fields = list(data.files)

    format_version = (
        int(data["storage_format_version"]) if "storage_format_version" in fields else 1
    )
    artifact_type = (
        str(data["artifact_type"])
        if "artifact_type" in fields
        else ("batch" if "batch_size" in fields else "single")
    )

    if artifact_type == "batch":
        n_vectors = int(data["batch_size"])
        vector_dim = int(data["vector_dim"])
        compression_ratio = (
            float(data["compression_ratio"]) if "compression_ratio" in fields else 0.0
        )
    else:
        n_vectors = int(data["n"])
        vector_dim = int(data["dims"])
        q = data["quantized"]
        original_bytes = n_vectors * vector_dim * 4
        compressed_bytes = q.nbytes + data["scales"].nbytes
        compression_ratio = original_bytes / max(compressed_bytes, 1)

    precision_mode = str(data["precision_mode"]) if "precision_mode" in fields else "int8"
    group_size = int(data["group_size"]) if "group_size" in fields else 0

    metadata: Optional[Dict[str, Any]] = None
    if "metadata_json" in fields:
        try:
            metadata = json.loads(str(data["metadata_json"]))
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "format_version": format_version,
        "artifact_type": artifact_type,
        "n_vectors": n_vectors,
        "vector_dim": vector_dim,
        "precision_mode": precision_mode,
        "group_size": group_size,
        "compression_ratio": compression_ratio,
        "needs_upgrade": format_version < _CURRENT_FORMAT_VERSION,
        "metadata": metadata,
        "fields": fields,
    }


def validate_artifact(path: Union[str, Path]) -> Dict[str, Any]:
    """Validate a Vectro artifact for structural integrity.

    Checks that required fields are present, dtype consistency, and that
    the ``quantized`` and ``scales`` arrays have compatible shapes.

    Args:
        path: Path to the ``.npz`` artifact.

    Returns:
        Dict with keys ``"valid"`` (bool) and ``"errors"`` (list[str]).
    """
    errors: list = []
    path = Path(path)

    # Open the file first to check required fields before calling inspect_artifact
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}

    fields = list(data.files)

    if "quantized" not in fields:
        errors.append("Missing required field: 'quantized'")
    if "scales" not in fields:
        errors.append("Missing required field: 'scales'")

    # Return early if critical fields are absent (inspect_artifact would fail)
    if errors:
        return {"valid": False, "errors": errors}

    try:
        info = inspect_artifact(path)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}

    q = data["quantized"]
    s = data["scales"]

    if info["artifact_type"] == "batch":
        if q.ndim != 2:
            errors.append(f"Expected quantized.ndim==2, got {q.ndim}")
        if q.shape[0] != info["n_vectors"]:
            errors.append(f"quantized rows ({q.shape[0]}) != batch_size ({info['n_vectors']})")
    else:
        if q.ndim != 2:
            errors.append(f"Expected quantized.ndim==2, got {q.ndim}")
        if q.shape[0] != info["n_vectors"]:
            errors.append(f"quantized rows ({q.shape[0]}) != n ({info['n_vectors']})")

    if s.ndim not in (1, 2):
        errors.append(f"Unexpected scales.ndim={s.ndim}; expected 1 or 2")
    if s.ndim == 1 and s.shape[0] != info["n_vectors"]:
        errors.append(f"scales length ({s.shape[0]}) != n_vectors ({info['n_vectors']})")

    if info["format_version"] > _CURRENT_FORMAT_VERSION:
        errors.append(
            f"format_version {info['format_version']} exceeds current "
            f"{_CURRENT_FORMAT_VERSION}; upgrade your Vectro installation."
        )

    return {"valid": len(errors) == 0, "errors": errors}


def upgrade_artifact(
    src: Union[str, Path],
    dst: Union[str, Path],
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Upgrade a v1 Vectro artifact to v2 format.

    If the artifact is already v2, it is copied unchanged (unless
    ``dry_run=True``).

    Args:
        src: Path to the source (possibly v1) artifact.
        dst: Destination path for the upgraded artifact.
        dry_run: When ``True``, parse and validate without writing.

    Returns:
        Dict with keys:

        * ``"upgraded"`` (bool) — ``False`` if artifact was already current
        * ``"src_version"`` (int) — original format version
        * ``"dst_version"`` (int) — written format version
        * ``"dry_run"`` (bool)
        * ``"dst"`` (str) — destination path
    """
    src = Path(src)
    dst = Path(dst)
    info = inspect_artifact(src)

    src_version = info["format_version"]

    if not info["needs_upgrade"]:
        if not dry_run:
            import shutil

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return {
            "upgraded": False,
            "src_version": src_version,
            "dst_version": src_version,
            "dry_run": dry_run,
            "dst": str(dst),
        }

    # Load original data
    data = np.load(src, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}

    # Patch missing v2 fields
    if "precision_mode" not in arrays:
        arrays["precision_mode"] = np.array("int8")
    if "group_size" not in arrays:
        arrays["group_size"] = np.array(0)
    if "storage_format" not in arrays:
        arrays["storage_format"] = np.array(_STORAGE_FORMAT_NAME)
    if "artifact_type" not in arrays:
        arrays["artifact_type"] = np.array("batch" if "batch_size" in arrays else "single")

    migration_record = {
        "migrated_from_version": src_version,
        "migrated_to_version": _CURRENT_FORMAT_VERSION,
        "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "src_fields": list(data.files),
    }
    # Merge or create metadata_json
    existing_meta: Dict[str, Any] = {}
    if "metadata_json" in arrays:
        try:
            existing_meta = json.loads(str(arrays["metadata_json"]))
        except (json.JSONDecodeError, ValueError):
            pass
    existing_meta["migration"] = migration_record

    arrays["metadata_json"] = np.array(json.dumps(existing_meta))
    arrays["storage_format_version"] = np.array(_CURRENT_FORMAT_VERSION)

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(dst), **arrays)

    return {
        "upgraded": True,
        "src_version": src_version,
        "dst_version": _CURRENT_FORMAT_VERSION,
        "dry_run": dry_run,
        "dst": str(dst),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_inspect(args: list) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m python.migration inspect",
        description="Inspect a Vectro compressed artifact.",
    )
    parser.add_argument("path", help="Path to .npz artifact")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    ns = parser.parse_args(args)

    try:
        info = inspect_artifact(ns.path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if ns.json:
        # Exclude non-serialisable fields for JSON output
        out = {k: v for k, v in info.items() if k != "metadata"}
        out["metadata"] = info["metadata"]
        print(json.dumps(out, indent=2, default=str))
    else:
        upg = "[NEEDS UPGRADE]" if info["needs_upgrade"] else "[current]"
        print(f"Path           : {info['path']}")
        print(f"File size      : {info['file_size_bytes']:,} bytes")
        print(f"Format version : v{info['format_version']}  {upg}")
        print(f"Artifact type  : {info['artifact_type']}")
        print(f"Vectors        : {info['n_vectors']} × {info['vector_dim']}")
        print(f"Precision mode : {info['precision_mode']}")
        print(f"Group size     : {info['group_size']}")
        print(f"Compression    : {info['compression_ratio']:.2f}×")
        if info["metadata"]:
            print(f"Created at     : {info['metadata'].get('created_at_utc', 'unknown')}")
            print(f"Vectro version : {info['metadata'].get('vectro_version', 'unknown')}")
    return 0


def _cmd_upgrade(args: list) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m python.migration upgrade",
        description="Upgrade a v1 artifact to v2 format.",
    )
    parser.add_argument("src", help="Source artifact path")
    parser.add_argument("dst", help="Destination artifact path")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing output")
    ns = parser.parse_args(args)

    try:
        result = upgrade_artifact(ns.src, ns.dst, dry_run=ns.dry_run)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if result["dry_run"]:
        tag = " (dry run)"
    elif result["upgraded"]:
        tag = ""
    else:
        tag = " (already current)"

    print(
        f"{'Upgraded' if result['upgraded'] else 'Copied'}"
        f" v{result['src_version']} → v{result['dst_version']}"
        f"{tag}: {result['dst']}"
    )
    return 0


def _cmd_validate(args: list) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m python.migration validate",
        description="Validate structural integrity of a Vectro artifact.",
    )
    parser.add_argument("path", help="Path to .npz artifact")
    ns = parser.parse_args(args)

    result = validate_artifact(ns.path)
    if result["valid"]:
        print(f"✓ {ns.path}: valid")
    else:
        print(f"✗ {ns.path}: {len(result['errors'])} error(s)")
        for err in result["errors"]:
            print(f"  - {err}")
    return 0 if result["valid"] else 1


def _main(argv: Optional[list] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("Usage: python -m python.migration <inspect|upgrade|validate> ...", file=sys.stderr)
        sys.exit(1)

    subcommand = argv[0]
    rest = argv[1:]

    dispatch = {
        "inspect": _cmd_inspect,
        "upgrade": _cmd_upgrade,
        "validate": _cmd_validate,
    }
    if subcommand not in dispatch:
        print(
            f"Unknown subcommand: {subcommand!r}. Choose from: inspect, upgrade, validate.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(dispatch[subcommand](rest))


if __name__ == "__main__":
    _main()
