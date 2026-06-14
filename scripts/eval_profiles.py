#!/usr/bin/env python3
"""Profile accuracy evaluation harness.

Runs the family-detect → encode → decode roundtrip pipeline for every test
fixture and asserts that the reconstructed cosine similarity meets the
per-method minimum from profiles.py.

Cosine gates (from quantization method accuracy contracts):
  int8  >= 0.9999
  nf4   >= 0.9800
  auto  >= 0.9999  (auto defaults to INT8 at runtime)

Usage:
    python scripts/eval_profiles.py [--dim DIM] [--n N] [--quiet]

Exit codes:
    0  All fixtures meet their cosine gate.
    1  One or more fixtures failed their cosine gate.
    2  Runtime error (vectro_py not installed, fixture missing, etc.).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so python/ is importable as a package
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Guard: vectro_py must be installed (maturin develop --release)
# ---------------------------------------------------------------------------
try:
    import vectro_py
except ImportError:
    print(
        "error: vectro_py is not installed.\n"
        "Run:  maturin develop --release -m rust/vectro_py/Cargo.toml",
        file=sys.stderr,
    )
    sys.exit(2)

from python.profiles import QuantProfile, get_profile  # noqa: E402

# Per-method minimum cosine similarity (architecture contracts)
_COSINE_GATES: dict[str, float] = {
    "int8": 0.9999,
    "nf4": 0.9800,
    # "auto" defers to INT8 at runtime — hold to the same gate
    "auto": 0.9999,
}

_FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _random_vectors(n: int, dim: int, seed: int = 42) -> list[list[float]]:
    """Generate reproducible pseudo-random unit vectors."""
    import random

    rng = random.Random(seed)
    vecs: list[list[float]] = []
    for _ in range(n):
        raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        vecs.append([x / norm for x in raw])
    return vecs


def _encode_decode_int8(vecs: list[list[float]]) -> list[list[float]]:
    """INT8 encode + manual decode using the per-vector fast API."""
    out: list[list[float]] = []
    for v in vecs:
        codes, scale = vectro_py.encode_int8_fast(v)
        reconstructed = [c / 127.0 * scale for c in codes]
        out.append(reconstructed)
    return out


def _encode_decode_nf4(vecs: list[list[float]]) -> list[list[float]]:
    """NF4 encode + decode using the batch class API."""
    encoder = vectro_py.PyNf4Encoder()
    encoder.encode(vecs)
    return encoder.decode()


def _encode_decode(method: str, vecs: list[list[float]]) -> list[list[float]]:
    """Dispatch encode+decode by method string."""
    if method in ("int8", "auto"):
        return _encode_decode_int8(vecs)
    if method == "nf4":
        return _encode_decode_nf4(vecs)
    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _eval_fixture(
    fixture_path: Path,
    n: int,
    dim: int,
    quiet: bool,
) -> tuple[str, QuantProfile, float, float, bool]:
    """Evaluate one fixture directory.

    Returns:
        (fixture_name, profile, mean_cosine, gate, passed)
    """
    profile = get_profile(fixture_path)
    gate = _COSINE_GATES[profile.method]
    vecs = _random_vectors(n, dim)
    reconstructed = _encode_decode(profile.method, vecs)

    cosines = [_cosine(orig, rec) for orig, rec in zip(vecs, reconstructed)]
    mean_cos = sum(cosines) / len(cosines)
    passed = mean_cos >= gate

    if not quiet:
        status = "PASS" if passed else "FAIL"
        print(
            f"  {fixture_path.name:<12}  family={profile.family:<8}  "
            f"method={profile.method:<5}  cosine={mean_cos:.6f}  "
            f"gate={gate:.4f}  [{status}]"
        )
    return fixture_path.name, profile, mean_cos, gate, passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate vectro quantization profiles against cosine gates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/eval_profiles.py\n"
            "  python scripts/eval_profiles.py --dim 384 --n 200\n"
            "  python scripts/eval_profiles.py --quiet\n"
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=768,
        help="Vector dimension (default: 768)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of vectors per fixture (default: 100)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-fixture output; only print summary.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=_FIXTURES_DIR,
        help="Path to fixtures directory (default: tests/fixtures/)",
    )
    args = parser.parse_args(argv)

    fixtures_dir: Path = args.fixtures_dir
    if not fixtures_dir.is_dir():
        print(f"error: fixtures directory not found: {fixtures_dir}", file=sys.stderr)
        return 2

    fixtures = sorted(p for p in fixtures_dir.iterdir() if p.is_dir())
    if not fixtures:
        print(f"error: no fixture subdirectories found in {fixtures_dir}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Evaluating {len(fixtures)} fixture(s)  dim={args.dim}  n={args.n}")
        print("-" * 72)

    results: list[tuple[str, QuantProfile, float, float, bool]] = []
    for fixture_path in fixtures:
        try:
            result = _eval_fixture(fixture_path, args.n, args.dim, args.quiet)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            print(f"error: fixture {fixture_path.name!r} raised: {exc}", file=sys.stderr)
            return 2

    passed = sum(1 for *_, ok in results if ok)
    failed = len(results) - passed

    if not args.quiet:
        print("-" * 72)
        print(f"Result: {passed}/{len(results)} passed", end="")
        if failed:
            print(f"  ({failed} FAILED)")
        else:
            print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
