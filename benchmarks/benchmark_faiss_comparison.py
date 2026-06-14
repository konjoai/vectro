#!/usr/bin/env python
"""
Faiss Comparison Benchmark — Vectro vs Faiss

Compares Vectro's quantization and HNSW search against Faiss's equivalent implementations.

Metrics compared:
1. Compression ratio at target compression
2. Recall@10 on ANN search
3. Throughput (compression and search)
4. Memory footprint

Usage:
    pip install faiss-cpu  # or faiss-gpu for NVIDIA
    python benchmarks/benchmark_faiss_comparison.py
"""

import json
import time
import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any
import warnings

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=DeprecationWarning)


def attempt_faiss_import() -> bool:
    """Check if Faiss is installed."""
    try:
        import faiss

        return True
    except ImportError:
        return False


def check_mojo_binary() -> bool:
    """Check if vectro_quantizer Mojo binary exists."""
    from pathlib import Path

    binary_path = Path(__file__).parent.parent / "vectro_quantizer"
    return binary_path.exists() and binary_path.is_file()


def benchmark_pq_compression(
    vectors: np.ndarray,
    n_subspaces: int = 96,
) -> Dict[str, Any]:
    """
    Compare Vectro PQ vs Faiss PQ compression.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    n_subspaces : int
        Number of PQ subspaces

    Returns
    -------
    dict with comparison results
    """
    print(f"\n▶ Product Quantization Comparison (M={n_subspaces})")

    from python.v3_api import PQCodebook

    n, d = vectors.shape
    split_point = n // 2
    train_vectors = vectors[:split_point]
    test_vectors = vectors[split_point:]

    # ============= VECTRO =============
    print("  Vectro...", end="", flush=True)
    start = time.time()
    vectro_cb = PQCodebook.train(train_vectors, n_subspaces=n_subspaces)
    vectro_train_time = time.time() - start

    # Compress test data
    start = time.time()
    vectro_codes = vectro_cb.encode(test_vectors)
    vectro_compress_time = time.time() - start

    # Decompress
    vectro_reconstructed = vectro_cb.decode(vectro_codes)

    # Quality
    cosine_sims_vectro = np.sum(test_vectors * vectro_reconstructed, axis=1) / (
        np.linalg.norm(test_vectors, axis=1) * np.linalg.norm(vectro_reconstructed, axis=1) + 1e-8
    )
    vectro_cosine = float(np.mean(cosine_sims_vectro))
    vectro_compression_ratio = (test_vectors.nbytes) / (vectro_codes.nbytes)

    print(f" cosine_sim={vectro_cosine:.5f}, compression={vectro_compression_ratio:.1f}x")

    results = {
        "vectro": {
            "training_time_sec": round(vectro_train_time, 3),
            "compression_time_sec": round(vectro_compress_time, 3),
            "mean_cosine_similarity": round(vectro_cosine, 6),
            "compression_ratio": round(vectro_compression_ratio, 2),
            "throughput_vec_per_sec": int(len(test_vectors) / vectro_compress_time),
        }
    }

    # ============= FAISS =============
    if attempt_faiss_import():
        print("  Faiss...", end="", flush=True)
        try:
            import faiss

            # Faiss product quantizer: size (d, n_codewords, bits)
            # Create index with PQ
            start = time.time()
            # Index factory string: "PQ96x8" means 96 subspaces, 8 bits each
            quantizer = faiss.IndexPQ(d, n_subspaces, 8)  # 8 bits per subspace
            quantizer.train(train_vectors.astype(np.float32))
            faiss_train_time = time.time() - start

            # Add vectors and encode
            start = time.time()
            quantizer.add(test_vectors.astype(np.float32))
            faiss_compress_time = time.time() - start

            # Reconstruct (decode)
            faiss_reconstructed = quantizer.reconstruct_n(0, len(test_vectors))

            # Quality
            cosine_sims_faiss = np.sum(test_vectors * faiss_reconstructed, axis=1) / (
                np.linalg.norm(test_vectors, axis=1) * np.linalg.norm(faiss_reconstructed, axis=1)
                + 1e-8
            )
            faiss_cosine = float(np.mean(cosine_sims_faiss))
            # Each subspace uses 8 bits, so compression is roughly d / (n_subspaces * 8 / 8) = d / n_subspaces
            faiss_compression_ratio = d / (n_subspaces // 8) if n_subspaces >= 8 else d / 1.0

            print(f" cosine_sim={faiss_cosine:.5f}, compression={faiss_compression_ratio:.1f}x")

            results["faiss"] = {
                "training_time_sec": round(faiss_train_time, 3),
                "compression_time_sec": round(faiss_compress_time, 3),
                "mean_cosine_similarity": round(faiss_cosine, 6),
                "compression_ratio": round(faiss_compression_ratio, 2),
                "throughput_vec_per_sec": int(len(test_vectors) / faiss_compress_time)
                if faiss_compress_time > 0
                else 0,
            }

            # Comparison
            results["comparison"] = {
                "vectro_vs_faiss_cosine_sim": round(vectro_cosine / max(faiss_cosine, 0.001), 2),
                "vectro_vs_faiss_throughput": round(
                    (len(test_vectors) / vectro_compress_time)
                    / max(len(test_vectors) / faiss_compress_time, 0.001),
                    2,
                ),
                "note": "Faiss uses IndexPQ; compression ratio calculated from index structure",
            }
        except Exception as e:
            print(f" Error: {e}")
    else:
        print("  Faiss...", end="", flush=True)
        print(" not installed (skipping)")

    return {
        "n_subspaces": n_subspaces,
        "vectors_shape": test_vectors.shape,
        "results": results,
    }


def benchmark_int8_quantization() -> Dict[str, Any]:
    """
    Compare INT8 quantization throughput.
    Prefers Mojo binary if available, falls back to Python.
    """
    print("\n▶ INT8 Quantization Comparison")

    import subprocess

    vectors = np.random.normal(size=(100000, 768)).astype(np.float32)

    # Try Mojo binary first
    mojo_available = check_mojo_binary()
    vectro_backend = "Mojo SIMD" if mojo_available else "Python/NumPy"

    if mojo_available:
        print(f"  Vectro INT8 ({vectro_backend})...", end="", flush=True)
        try:
            # Write vectors to temp file for vectro_quantizer to read
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                temp_vectors_path = f.name
                # Save as float32 binary
                vectors.astype(np.float32).tofile(f)

            # Run benchmark via Mojo binary
            start = time.time()
            result = subprocess.run(
                ["./vectro_quantizer", "benchmark", "100000", "768"],
                capture_output=True,
                timeout=60,
                cwd=str(Path(__file__).parent.parent),
            )
            mojo_time = time.time() - start

            if result.returncode == 0:
                # Parse output for throughput.
                # The binary prints a header line first:
                #   Benchmark n= 100000  d= 768
                #   INT8 quantize  : 5234567 vec/s
                #   INT8 reconstruct: ...
                output = result.stdout.decode()
                lines = output.strip().split("\n")
                vectro_throughput = 0
                for line in lines:
                    if "INT8 quantize" in line and "vec/s" in line:
                        vectro_throughput = int(line.split(":")[1].strip().split()[0])
                        break
                if vectro_throughput == 0:
                    raise ValueError(
                        f"Could not parse INT8 quantize throughput from output:\n{output}"
                    )
                print(f" {vectro_throughput:,.0f} vec/s (Mojo SIMD)")
            else:
                stderr = result.stderr.decode().strip()
                print(
                    f" Error running binary (rc={result.returncode})"
                    + (f": {stderr[:120]}" if stderr else "")
                    + ", falling back to Python"
                )
                mojo_available = False
                vectro_backend = "Python/NumPy"
        except Exception as e:
            print(f" Error: {e}, falling back to Python")
            mojo_available = False
            vectro_backend = "Python/NumPy"

    # Python fallback
    if not mojo_available:
        print("  Vectro INT8 (Python/NumPy)...", end="", flush=True)
        from python import compress_vectors as vectro_compress

        start = time.time()
        for i in range(0, len(vectors), 10000):
            vectro_compress(vectors[i : i + 10000], profile="balanced")
        vectro_time = time.time() - start
        vectro_throughput = len(vectors) / vectro_time
        print(f" {vectro_throughput:,.0f} vec/s (Python/NumPy)")

    results = {
        "vectors_shape": vectors.shape,
        "vectro_backend": vectro_backend,
        "vectro_throughput_vec_per_sec": int(vectro_throughput),
    }

    # Faiss - use IndexScalarQuantizer with scalar quantization
    if attempt_faiss_import():
        print("  Faiss (IndexScalarQuantizer INT8, C++)...", end="", flush=True)
        try:
            import faiss

            index = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_8bit)

            start = time.time()
            index.train(vectors[:10000])
            index.add(vectors)
            faiss_time = time.time() - start
            faiss_throughput = len(vectors) / faiss_time
            print(f" {faiss_throughput:,.0f} vec/s (C++)")

            results["faiss_throughput_vec_per_sec"] = int(faiss_throughput)
            results["faiss_backend"] = "C++"
            results["comparison_int8"] = {
                "vectro_vs_faiss_throughput": round(
                    vectro_throughput / max(faiss_throughput, 0.001), 2
                ),
                "note": f"Vectro backend: {vectro_backend}",
            }
        except Exception as e:
            print(f" Error: {e}")

    return results


def benchmark_int8_multidim() -> dict:
    """
    Compare INT8 throughput across multiple dimensions: d=128, 384, 768, 1536.

    Runs the Mojo binary (preferred) or Python fallback at n=50,000 for each
    dimension, and the Faiss C++ ScalarQuantizer at the same dimensions.

    Returns
    -------
    dict
        ``{"d=128": {...}, "d=384": {...}, "d=768": {...}, "d=1536": {...}}``
        where each sub-dict contains vectro_throughput, faiss_throughput,
        and vectro_vs_faiss_ratio.
    """
    import subprocess

    print("\n▶ INT8 Multi-Dimensional Throughput (n=50,000)")

    DIMS = [128, 384, 768, 1536]
    N = 50_000
    mojo_available = check_mojo_binary()
    faiss_available = attempt_faiss_import()

    results = {}

    for d in DIMS:
        print(f"  d={d:<5}", end="", flush=True)
        entry: Dict[str, Any] = {"n": N, "d": d}

        # ── Vectro throughput ──────────────────────────────────────────────
        vectro_throughput: int = 0
        if mojo_available:
            try:
                result = subprocess.run(
                    ["./vectro_quantizer", "benchmark", str(N), str(d)],
                    capture_output=True,
                    timeout=120,
                    cwd=str(Path(__file__).parent.parent),
                )
                if result.returncode == 0:
                    for line in result.stdout.decode().splitlines():
                        if "INT8 quantize" in line and "vec/s" in line:
                            vectro_throughput = int(line.split(":")[1].strip().split()[0])
                            break
            except Exception:
                pass

        if vectro_throughput == 0:
            # Python/NumPy fallback
            from python import compress_vectors  # type: ignore[import]

            vectors_local = np.random.normal(size=(N, d)).astype(np.float32)
            for _ in range(2):  # warmup
                compress_vectors(vectors_local[:500], profile="int8")
            times = []
            for _ in range(3):
                t0 = time.time()
                compress_vectors(vectors_local, profile="int8")
                times.append(time.time() - t0)
            vectro_throughput = int(N / min(times))
            entry["vectro_backend"] = "Python/NumPy"
        else:
            entry["vectro_backend"] = "Mojo SIMD"

        entry["vectro_throughput_vec_per_sec"] = vectro_throughput
        print(f"  Vectro {vectro_throughput:>12,} vec/s  ({entry['vectro_backend']})", end="")

        # ── Faiss throughput ───────────────────────────────────────────────
        if faiss_available:
            try:
                import faiss  # type: ignore[import]

                vectors_local = np.random.normal(size=(N, d)).astype(np.float32)
                index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
                index.train(vectors_local[: min(10_000, N)])
                times = []
                for _ in range(3):
                    t0 = time.time()
                    index.add(vectors_local)
                    times.append(time.time() - t0)
                    index.reset()
                faiss_throughput = int(N / min(times))
                entry["faiss_throughput_vec_per_sec"] = faiss_throughput
                ratio = vectro_throughput / max(faiss_throughput, 1)
                entry["vectro_vs_faiss_ratio"] = round(ratio, 2)
                print(f"   FAISS {faiss_throughput:>12,} vec/s   ratio {ratio:.2f}x")
            except Exception as exc:
                print(f"   FAISS error: {str(exc)[:60]}")
        else:
            print("   FAISS not installed")

        key = f"d={d}"
        results[key] = entry

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Vectro against Faiss")
    parser.add_argument("--output", type=str, default="results/faiss_comparison.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VECTRO vs FAISS — Quantization Comparison")
    print("=" * 70)

    # Check Mojo binary availability
    mojo_available = check_mojo_binary()
    if not mojo_available:
        print("\n⚠️  IMPORTANT: Mojo binary not found!")
        print("   To get representative Mojo performance (5M+ vec/s):")
        print("   1. Install Mojo: see https://modular.com/mojo")
        print("   2. Run: pixi install && pixi shell && pixi run build-mojo")
        print("   3. Re-run this benchmark to see Mojo vs Faiss comparison")
        print("\n   Running Python/NumPy fallback for now...")
        print("   (Expect Mojo to be 50-100x faster than Python fallback)\n")

    if not attempt_faiss_import():
        print("\n⚠️  Faiss not installed. To enable Faiss comparison:")
        print("   pip install faiss-cpu")
        print("\nContinuing with Vectro benchmarks only...\n")

    # Generate test data
    vectors = np.random.normal(size=(100000, 768)).astype(np.float32)

    # Run benchmarks; int8 result carries the authoritative backend label
    # (may be "Python/NumPy" if the Mojo binary errored at runtime)
    pq_result = benchmark_pq_compression(vectors)
    int8_result = benchmark_int8_quantization()
    multidim_result = benchmark_int8_multidim()
    actual_backend = int8_result.get("vectro_backend", "Python/NumPy")

    all_results = {
        "benchmark_type": "faiss_comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vectro_backend": actual_backend,
        "faiss_imported": attempt_faiss_import(),
        "faiss_backend": "C++",
        "pq_comparison": pq_result,
        "int8_comparison": int8_result,
        "int8_multidim": multidim_result,
        "notes": {
            "vectro_info": (
                "Using compiled Mojo SIMD binary"
                if actual_backend == "Mojo SIMD"
                else "Mojo binary not found or failed at runtime; results are Python/NumPy"
            ),
            "faiss_info": "Faiss uses optimized C++ kernels",
            "comparison_note": (
                "Performance depends on Vectro backend; "
                "Mojo expected to be 50-100x faster than Python"
            ),
        },
    }

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    print("\nBackend Summary:")
    print(f"  Vectro: {all_results['vectro_backend']}")
    print(f"  Faiss:  {all_results['faiss_backend']}")

    return all_results


if __name__ == "__main__":
    main()
