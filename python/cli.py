"""Vectro command-line interface.

Provides a unified CLI for compression, inspection, and benchmarking.

Entry point registered as ``vectro`` by pyproject.toml.

Usage examples::

    vectro --version
    vectro compress embeddings.npy embeddings.npz
    vectro compress embeddings.npy embeddings.npz --profile quality
    vectro decompress embeddings.npz restored.npy
    vectro inspect embeddings.npz
    vectro inspect embeddings.npz --json
    vectro upgrade old.npz new.npz
    vectro upgrade old.npz new.npz --dry-run
    vectro validate embeddings.npz
    vectro benchmark --n 1000 --dim 768
    vectro info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


_VERSION = "3.6.0"


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------


def _cmd_compress(args: argparse.Namespace) -> int:
    import numpy as np
    from .vectro import Vectro
    from .profiles_api import get_compression_profile

    try:
        vectors = np.load(args.input, allow_pickle=False)
    except Exception as exc:
        print(f"Error loading {args.input}: {exc}", file=sys.stderr)
        return 1

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    profile = None
    if args.profile:
        try:
            profile = get_compression_profile(args.profile)
        except Exception as exc:
            print(f"Unknown profile {args.profile!r}: {exc}", file=sys.stderr)
            return 1

    vectro = Vectro()
    result = vectro.compress_batch(vectors, profile=profile)

    output: str = args.output
    n, d = vectors.shape
    lossless_pass: str = getattr(args, "lossless_pass", None) or "zstd"

    # Cloud URI dispatch: s3://, gs://, abfs://
    if output.startswith("s3://"):
        from .storage_v3 import S3Backend
        bucket, _, remote = output[5:].partition("/")
        S3Backend(bucket).save_vqz(result.quantized, result.scales, d, remote,
                                   compression=lossless_pass)
    elif output.startswith("gs://"):
        from .storage_v3 import GCSBackend
        bucket, _, remote = output[5:].partition("/")
        GCSBackend(bucket).save_vqz(result.quantized, result.scales, d, remote,
                                    compression=lossless_pass)
    elif output.startswith("abfs://"):
        from .storage_v3 import AzureBlobBackend
        container, _, remote = output[7:].partition("/")
        AzureBlobBackend(container).save_vqz(result.quantized, result.scales, d, remote,
                                             compression=lossless_pass)
    elif output.endswith(".vqz"):
        from .storage_v3 import save_compressed
        save_compressed(result, output, codec=lossless_pass)
    else:
        vectro.save_compressed(result, output)

    print(f"Compressed {n} vectors ({d}D)")
    print(f"Ratio  : {result.compression_ratio:.2f}×")
    print(f"Output : {output}")
    return 0


def _cmd_decompress(args: argparse.Namespace) -> int:
    import numpy as np
    from .vectro import Vectro

    vectro = Vectro()
    try:
        result = vectro.load_compressed(args.input)
    except Exception as exc:
        print(f"Error loading {args.input}: {exc}", file=sys.stderr)
        return 1

    from . import decompress_vectors
    restored = decompress_vectors(result)
    np.save(args.output, restored)

    print(f"Decompressed {restored.shape[0]} × {restored.shape[1]} → {args.output}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .migration import inspect_artifact

    try:
        info = inspect_artifact(args.path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(info, indent=2, default=str))
        return 0

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
        print(f"Vectro version : {info['metadata'].get('vectro_version', 'unknown')}")
        print(f"Created at     : {info['metadata'].get('created_at_utc', 'unknown')}")
    return 0


def _cmd_upgrade(args: argparse.Namespace) -> int:
    from .migration import upgrade_artifact

    try:
        result = upgrade_artifact(args.src, args.dst, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    tag = " (dry run)" if result["dry_run"] else (" (already current)" if not result["upgraded"] else "")
    verb = "Upgraded" if result["upgraded"] else "Copied"
    print(f"{verb} v{result['src_version']} → v{result['dst_version']}{tag}: {result['dst']}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from .migration import validate_artifact

    result = validate_artifact(args.path)
    if result["valid"]:
        print(f"✓ {args.path}: valid")
        return 0
    else:
        print(f"✗ {args.path}: {len(result['errors'])} error(s)")
        for err in result["errors"]:
            print(f"  - {err}")
        return 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    import numpy as np
    from .benchmark import BenchmarkSuite

    rng = np.random.default_rng(args.seed)
    embeddings = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    print(f"Benchmarking {args.n} × {args.dim} vectors ({args.runs} run(s) per profile)…")
    suite = BenchmarkSuite(embeddings, n_runs=args.runs)
    report = suite.run_all()

    print(f"\nBest profile    : {report.best_profile}")
    print(f"Best compression: {report.best_compression_ratio:.2f}×")
    print(f"Best quality    : {report.best_recall:.4f} cosine sim")

    if args.output:
        ext = Path(args.output).suffix.lower()
        if ext == ".csv":
            report.to_csv(args.output)
        else:
            report.to_json(args.output)
        print(f"Report saved to {args.output}")
    return 0


def _load_result_for_export(path: str):
    """Load a compressed artifact from *path* for ONNX export.

    Supports ``.npz`` artifacts via :class:`~python.vectro.Vectro` and
    ``.vqz`` files via :func:`~python.storage_v3.load_compressed`.

    Raises
    ------
    RuntimeError
        If the ``onnx`` package is not installed (propagated from
        :func:`~python.onnx_export.to_onnx_model`).
    FileNotFoundError
        If *path* does not exist.
    """
    if path.endswith(".vqz"):
        from .storage_v3 import load_compressed
        return load_compressed(path)
    from .vectro import Vectro
    return Vectro().load_compressed(path)


def _cmd_export_onnx(args: argparse.Namespace) -> int:
    """Export a compressed artifact as an ONNX INT8 dequantization graph."""
    from .onnx_export import export_onnx

    try:
        result = _load_result_for_export(args.input)
    except FileNotFoundError as exc:
        print(f"Error loading {args.input}: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        export_onnx(result, args.output)
    except (RuntimeError, ValueError) as exc:
        print(f"ONNX export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Exported ONNX model → {args.output}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from . import get_backend_info, get_version_info

    vi = get_version_info()
    bi = get_backend_info()

    print(f"Vectro {_VERSION}")
    print(f"Backend : {bi.get('backend', 'unknown')}")
    print(f"Available: {', '.join(bi.get('available', []))}")
    print(f"Python  : {vi.get('python_version', 'unknown')}")
    print(f"NumPy   : {vi.get('numpy_version', 'unknown')}")
    print(f"Platform: {vi.get('platform', 'unknown')}")

    if getattr(args, "benchmark", False):
        _run_inline_benchmark()

    return 0


def _run_inline_benchmark() -> None:
    """5-second throughput estimation printed to stdout."""
    import time
    import numpy as np
    from . import quantize_embeddings, reconstruct_embeddings
    from .nf4_api import quantize_nf4, dequantize_nf4

    _DIM   = 768
    _BATCH = 256

    rng  = np.random.default_rng(0)
    vecs = rng.standard_normal((_BATCH, _DIM)).astype(np.float32)

    print()
    print("── Benchmark (5 s) ────────────────────────────────────────")

    # ── INT8 throughput ───────────────────────────────────────────────────────────────────────────────────
    deadline  = time.monotonic() + 5.0
    n_batches = 0
    while time.monotonic() < deadline:
        quantize_embeddings(vecs)
        n_batches += 1

    total_vecs = n_batches * _BATCH
    print(f"INT8 throughput : {total_vecs / 5:,.0f} vec/s  "
          f"({n_batches} × {_BATCH}-batch in 5 s)")

    # ── INT8 MAE ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
    result_int8 = quantize_embeddings(vecs)
    recon_int8  = reconstruct_embeddings(result_int8)
    mae_int8    = float(np.mean(np.abs(vecs - recon_int8)))
    print(f"INT8 MAE        : {mae_int8:.6f}")

    # ── NF4 MAE (best-effort; skipped when unavailable) ──────────────────────────────────────────────────────────────────────────────
    try:
        packed, scales = quantize_nf4(vecs)
        recon_nf4 = dequantize_nf4(packed, scales, _DIM)
        mae_nf4   = float(np.mean(np.abs(vecs - recon_nf4)))
        print(f"NF4  MAE        : {mae_nf4:.6f}")
    except Exception:  # noqa: BLE001 — NF4 backend may not be built
        print("NF4  MAE        : unavailable (NF4 backend not found)")

    print("─────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vectro",
        description="Vectro — ultra-high-performance LLM embedding compressor",
    )
    parser.add_argument("--version", action="version", version=f"vectro {_VERSION}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # compress
    p = sub.add_parser("compress", help="Compress float32 embeddings to a .npz artifact")
    p.add_argument("input", help="Input .npy file (float32 embeddings)")
    p.add_argument("output", help="Output .npz artifact path")
    p.add_argument("--profile", default=None,
                   choices=["speed", "balanced", "quality", "extreme", "adaptive"],
                   help="Compression profile (default: balanced)")
    p.add_argument("--lossless-pass", dest="lossless_pass",
                   choices=["zstd", "zlib", "none"], default="zstd",
                   help="Lossless compression codec for .vqz output (default: zstd)")

    # decompress
    p = sub.add_parser("decompress", help="Reconstruct float32 vectors from a .npz artifact")
    p.add_argument("input", help="Input .npz artifact")
    p.add_argument("output", help="Output .npy file")

    # inspect
    p = sub.add_parser("inspect", help="Inspect a compressed artifact")
    p.add_argument("path", help="Path to .npz artifact")
    p.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # upgrade
    p = sub.add_parser("upgrade", help="Upgrade a v1 artifact to v2 format")
    p.add_argument("src", help="Source artifact path")
    p.add_argument("dst", help="Destination artifact path")
    p.add_argument("--dry-run", action="store_true", help="Validate without writing")

    # validate
    p = sub.add_parser("validate", help="Validate artifact structural integrity")
    p.add_argument("path", help="Path to .npz artifact")

    # benchmark
    p = sub.add_parser("benchmark", help="Run a compression benchmark")
    p.add_argument("--n", type=int, default=1000, help="Number of vectors (default: 1000)")
    p.add_argument("--dim", type=int, default=768, help="Vector dimension (default: 768)")
    p.add_argument("--runs", type=int, default=3, help="Runs per profile (default: 3)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    p.add_argument("--output", default=None, help="Save report to .json or .csv file")

    # info
    p = sub.add_parser("info", help="Show backend and environment information")
    p.add_argument(
        "--benchmark", action="store_true", default=False,
        help="Run a 5-second throughput benchmark and print MAE figures",
    )

    # export-onnx
    p = sub.add_parser(
        "export-onnx",
        help="Export a compressed artifact as an ONNX INT8 dequantization graph (opset 17)",
    )
    p.add_argument("input", help="Input .npz or .vqz compressed artifact")
    p.add_argument("output", help="Output .onnx file path")

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "compress": _cmd_compress,
        "decompress": _cmd_decompress,
        "inspect": _cmd_inspect,
        "upgrade": _cmd_upgrade,
        "validate": _cmd_validate,
        "benchmark": _cmd_benchmark,
        "info": _cmd_info,
        "export-onnx": _cmd_export_onnx,
    }
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
