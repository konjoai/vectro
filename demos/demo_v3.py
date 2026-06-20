"""
Vectro v3.0.0 — Comprehensive Feature Demo
===========================================
Demonstrates all quantization modes, the HNSW index, AutoQuantize, VQZ storage,
and GPU support.  Designed to be recorded as a terminal demo (asciinema / GIF).

Usage:
    python demos/demo_v3.py           # full demo with pauses (run from project root)
    python demos/demo_v3.py --fast    # skip sleep() calls
"""

from __future__ import annotations

import sys
import os
import time
import tempfile
import numpy as np

# Ensure the project root is on sys.path so `python` package is importable
# whether the script is run from the project root or from within demos/.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Pause helper — skipped when --fast is passed
_FAST = "--fast" in sys.argv


def pause(seconds: float = 1.2) -> None:
    if not _FAST:
        time.sleep(seconds)


def section(title: str) -> None:
    width = 68
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    pause(0.8)


def subsection(title: str) -> None:
    print(f"\n  -- {title} --")
    pause(0.5)


def bar(value: float, width: int = 36, filled: str = "█", empty: str = "░") -> str:
    filled_n = int(round(min(value, 1.0) * width))
    return filled * filled_n + empty * (width - filled_n)


def progress_bar(label: str, value: float, fmt: str = ".4f", width: int = 30, display=None) -> None:
    shown = display if display is not None else value
    print(f"  {label:<32} {bar(value, width)}  {shown:{fmt}}")
    pause(0.15)


def col(v: float, good: float, great: float) -> str:
    if v >= great:
        return "GREAT"
    if v >= good:
        return "GOOD "
    return "POOR "


# ─────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────


def show_banner() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                                                                  ║")
    print("║    ╦  ╦╔═╗╔═╗╔╦╗╦═╗╔═╗                                         ║")
    print("║    ╚╗╔╝║╣ ║   ║ ╠╦╝║ ║                                         ║")
    print("║     ╚╝ ╚═╝╚═╝ ╩ ╩╚═╚═╝   v3.0.0                                ║")
    print("║                                                                  ║")
    print("║    Ultra-High-Performance LLM Embedding Compressor               ║")
    print("║    Mojo-First · SIMD · HNSW · GPU · Cloud                       ║")
    print("║                                                                  ║")
    print("║    Quantization Modes                                            ║")
    print("║       INT8  ·  INT4  ·  NF4  ·  NF4-Mixed  ·  PQ-96             ║")
    print("║       Binary  ·  RQ×3  ·  AutoEncoder  ·  AutoQuantize           ║")
    print("║                                                                  ║")
    print("║    Compression range:  4× (lossless) → 48× (learned)            ║")
    print("║    Tests:              445 / 445 passing  ·  100% coverage       ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    pause(2.5)


# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────


def setup_data() -> tuple[np.ndarray, np.ndarray]:
    section("0 · Test Data")
    rng = np.random.default_rng(42)
    n, d = 2_000, 768
    print(f"  Generating {n:,} vectors of dimension {d}  (float32) …")

    vectors = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = (vectors / norms).astype(np.float32)

    print(f"  Shape       : {vectors.shape}")
    print(f"  Mean norm   : {float(np.mean(norms)):.4f}")
    print(f"  Raw (FP32)  : {vectors.nbytes / 1024 / 1024:.2f} MB")
    pause(1.0)
    return vectors, normed


# ─────────────────────────────────────────────────────────────
# 1  INT8
# ─────────────────────────────────────────────────────────────


def demo_int8(vectors: np.ndarray) -> None:
    section("1 · INT8 Symmetric Quantization  (Phase 0 + 1)")
    from python import Vectro, VectroQualityAnalyzer

    vectro = Vectro(profile="balanced")
    t0 = time.perf_counter()
    result = vectro.compress(vectors)
    elapsed = time.perf_counter() - t0

    restored = vectro.decompress(result)
    analyzer = VectroQualityAnalyzer()
    q = analyzer.evaluate_quality(vectors, restored)

    throughput = len(vectors) / elapsed
    print()
    print(f"  Vectors          : {len(vectors):>7,}")
    print(f"  Wall-clock time  : {elapsed * 1000:.1f} ms")
    print(f"  Throughput       : {throughput:>10,.0f} vec/s")
    print()
    progress_bar(
        "Compression ratio  (~4×)",
        result.compression_ratio / 8,
        fmt=".2f",
        display=result.compression_ratio,
    )
    progress_bar("Cosine similarity  (>=0.9999)", q.mean_cosine_similarity)
    print(f"\n  Grade: {q.quality_grade()} | Passes 0.999: {q.passes_quality_threshold(0.999)}")
    pause(1.5)


# ─────────────────────────────────────────────────────────────
# 2  NF4
# ─────────────────────────────────────────────────────────────


def demo_nf4(vectors: np.ndarray) -> None:
    section("2 · NF4 Normal Float 4-bit  (Phase 2)")
    from python.nf4_api import quantize_nf4, dequantize_nf4
    from python.interface import mean_cosine_similarity

    d = vectors.shape[1]

    subsection("Standard NF4 — 8× compression")
    t0 = time.perf_counter()
    packed, scales = quantize_nf4(vectors)
    restored = dequantize_nf4(packed, scales, d)
    elapsed = time.perf_counter() - t0

    cosine = mean_cosine_similarity(vectors, restored)
    ratio = vectors.nbytes / (packed.nbytes + scales.nbytes)

    print(f"  packed shape  : {packed.shape}  dtype:uint8  ({packed.shape[1]} bytes/vec)")
    print(f"  scales shape  : {scales.shape}  dtype:float32")
    print(f"  Throughput    : {len(vectors) / elapsed:,.0f} vec/s")
    print()
    progress_bar("Compression ratio  (~8×)", ratio / 10, fmt=".2f", display=ratio)
    progress_bar("Cosine similarity  (>=0.985)", cosine)
    pause(1.0)

    subsection("NF4-Mixed — outlier dims as FP16  (~7.5× compression)")
    from python.nf4_api import quantize_mixed, dequantize_mixed, select_outlier_dims

    out_dims = select_outlier_dims(vectors, k=16)
    fp16_vals, nf4_packed, nf4_scales, out_d = quantize_mixed(vectors, outlier_dims=out_dims)
    restored_m = dequantize_mixed(fp16_vals, nf4_packed, nf4_scales, out_d, d)
    cosine_m = mean_cosine_similarity(vectors, restored_m)

    compressed_bytes = fp16_vals.nbytes + nf4_packed.nbytes + nf4_scales.nbytes
    ratio_m = vectors.nbytes / compressed_bytes

    print(f"  Outlier dims kept as FP16 : {len(out_d)}")
    print()
    progress_bar("Compression ratio  (~7.5×)", ratio_m / 10, fmt=".2f", display=ratio_m)
    progress_bar("Cosine similarity  (>=0.990)", cosine_m)
    pause(1.5)


# ─────────────────────────────────────────────────────────────
# 3  Product Quantization
# ─────────────────────────────────────────────────────────────


def demo_pq(vectors: np.ndarray) -> None:
    section("3 · Product Quantization  (Phase 3)")
    from python.v3_api import PQCodebook
    from python.interface import mean_cosine_similarity

    n_sub = 96
    subsection(f"Training PQ codebook  (M={n_sub} sub-spaces, K=256 centroids)")
    print(f"  Training on {len(vectors):,} vectors …")
    t0 = time.perf_counter()
    cb = PQCodebook.train(vectors, n_subspaces=n_sub)
    train_time = time.perf_counter() - t0
    print(f"  Codebook trained in {train_time:.2f} s")
    print(
        f"  {cb.n_subspaces} sub-spaces × {cb.n_centroids} centroids × {cb.sub_dim} dims/sub-space"
    )
    pause(0.8)

    subsection("Encode → Decode quality")
    t0 = time.perf_counter()
    codes = cb.encode(vectors)
    encode_time = time.perf_counter() - t0
    decoded = cb.decode(codes)
    cosine = mean_cosine_similarity(vectors, decoded)
    ratio = vectors.nbytes / codes.nbytes

    print(f"  codes shape  : {codes.shape}  dtype:uint8")
    print(f"  Bytes/vector : {codes.shape[1]} vs {vectors.shape[1] * 4} (FP32)")
    print(f"  Throughput   : {len(vectors) / encode_time:,.0f} vec/s")
    print()
    progress_bar("Compression ratio  (~32×)", ratio / 40, fmt=".1f", display=ratio)
    progress_bar("Cosine similarity  (>=0.95)", cosine)
    pause(1.0)

    subsection("PQ-48  (16× compression, higher quality)")
    cb48 = PQCodebook.train(vectors, n_subspaces=48)
    codes48 = cb48.encode(vectors)
    decoded48 = cb48.decode(codes48)
    cosine48 = mean_cosine_similarity(vectors, decoded48)
    ratio48 = vectors.nbytes / codes48.nbytes
    progress_bar("Compression ratio  (~16×)", ratio48 / 20, fmt=".1f", display=ratio48)
    progress_bar("Cosine similarity  (>=0.97)", cosine48)
    pause(1.5)


# ─────────────────────────────────────────────────────────────
# 4  Binary
# ─────────────────────────────────────────────────────────────


def demo_binary(normed: np.ndarray) -> None:
    section("4 · Binary / 1-bit Quantization  (Phase 4)")
    from python.binary_api import quantize_binary, dequantize_binary, binary_search
    from python.interface import mean_cosine_similarity

    subsection("Encode  (sign(v) → 1 bit, 8 dims/byte)")
    t0 = time.perf_counter()
    packed = quantize_binary(normed)
    elapsed = time.perf_counter() - t0

    restored = dequantize_binary(packed, normed.shape[1])
    cosine = mean_cosine_similarity(normed, restored)
    ratio = normed.nbytes / packed.nbytes

    print(f"  normed shape  : {normed.shape}")
    print(f"  packed shape  : {packed.shape}  (dtype:uint8, 8 dims/byte)")
    print(f"  Throughput    : {len(normed) / elapsed:,.0f} vec/s")
    print()
    progress_bar("Compression ratio  (~32×)", ratio / 40, fmt=".1f", display=ratio)
    progress_bar("Cosine similarity", cosine)
    pause(1.0)

    subsection("Hamming-distance search")
    query = normed[0]
    t0 = time.perf_counter()
    top_ids, distances = binary_search(query, packed, top_k=5)
    search_time = time.perf_counter() - t0

    print("  Query index       : 0")
    print(f"  Top-5 result IDs  : {top_ids.tolist()}")
    print(f"  Hamming distances : {distances.tolist()}")
    print(f"  Search time       : {search_time * 1000:.3f} ms")
    print(f"  Throughput        : {len(normed) / search_time:,.0f} comparisons/s")
    print(f"  Self-retrieval    : {int(0) in top_ids.tolist()}")
    pause(1.0)

    subsection("Matryoshka-compatible encoding")
    from python.binary_api import matryoshka_encode

    dims = [64, 128, 256, 512, 768]
    prefixes = matryoshka_encode(normed, dims=dims)
    for d in dims:
        pack = prefixes[d]
        print(f"  d={d:>4}  packed shape {pack.shape}  {pack.shape[1]} bytes/vec")
    pause(1.5)


# ─────────────────────────────────────────────────────────────
# 5  HNSW Index
# ─────────────────────────────────────────────────────────────


def demo_hnsw(vectors: np.ndarray) -> None:
    section("5 · HNSW Approximate Nearest-Neighbour Index  (Phase 5)")
    from python.v3_api import HNSWIndex

    M = 16
    print(f"  Building HNSW index: {len(vectors):,} × {vectors.shape[1]}D")
    print(f"  M={M}  ef_build=200  quantization=int8")

    t0 = time.perf_counter()
    index = HNSWIndex(dim=vectors.shape[1], quantization="int8", M=M, ef_build=200)
    index.add_batch(vectors)
    build_time = time.perf_counter() - t0

    approx_index_bytes = len(vectors) * vectors.shape[1]  # INT8 storage
    approx_fp32_bytes = len(vectors) * vectors.shape[1] * 4

    print(f"\n  Build time  : {build_time:.2f} s")
    print(f"  Build rate  : {len(vectors) / build_time:,.0f} vec/s")
    print(
        f"  INT8 storage : {approx_index_bytes / 1024:.0f} KB  "
        f"(vs {approx_fp32_bytes / 1024:.0f} KB FP32 = 4× smaller)"
    )
    pause(1.0)

    subsection("Nearest-neighbour search")
    query = vectors[0]
    t0 = time.perf_counter()
    indices, distances = index.search(query, top_k=10)
    search_time = time.perf_counter() - t0

    print("  Query → top-10 results")
    print(f"  Indices   : {indices[:5]} …")
    print(f"  Distances : {[f'{d:.4f}' for d in distances[:5]]} …")
    print(f"  Latency   : {search_time * 1000:.3f} ms")
    print(f"  Self-retrieval (id=0 first): {indices[0] == 0}")
    pause(1.0)

    subsection("Recall@10 evaluation (brute-force reference)")
    n_q = min(50, len(vectors))
    queries = vectors[:n_q]
    sims = queries @ vectors.T
    true_top10 = np.argsort(-sims, axis=1)[:, :10]

    hits = 0
    for i, q in enumerate(queries):
        pred_ids, _ = index.search(q, top_k=10)
        hits += len(set(pred_ids) & set(true_top10[i].tolist()))

    recall = hits / (n_q * 10)
    print(f"  Recall@10 on {n_q} queries: {recall:.4f}")
    progress_bar("Recall@10  (>=0.90 target)", recall)
    pause(1.0)

    subsection("Save / Load round-trip")
    with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
        path = f.name
    try:
        index.save(path)
        index2 = HNSWIndex.load(path)
        ids2, _ = index2.search(query, top_k=5)
        print(f"  Saved to    : {os.path.basename(path)}")
        print(f"  Index loaded: {len(index2._index._vectors)} vectors")
        print(f"  Results match: {list(indices[:5]) == ids2[:5]}")
    finally:
        os.unlink(path)
    pause(1.5)


# ─────────────────────────────────────────────────────────────
# 6  GPU / MAX Engine
# ─────────────────────────────────────────────────────────────


def demo_gpu(vectors: np.ndarray) -> None:
    section("6 · GPU / MAX Engine Quantization  (Phase 6)")
    from python.gpu_api import gpu_available, gpu_device_info, quantize_int8_batch, gpu_benchmark

    info = gpu_device_info()
    available = gpu_available()

    print(f"  GPU available    : {available}")
    print(f"  Backend          : {info.get('backend', 'unknown')}")
    print(f"  Device name      : {info.get('device_name', 'unknown')}")
    print(f"  SIMD width       : {info.get('simd_width', 'N/A')}")
    print(f"  Unified memory   : {info.get('unified_memory', 'N/A')}")
    pause(0.8)

    subsection("Quantize via GPU/SIMD path")
    q_batch, s_batch = quantize_int8_batch(vectors)
    print(f"  Input shape      : {vectors.shape}  (float32)")
    print(f"  Output shape     : {q_batch.shape}  (int8)")
    print(f"  Scales shape     : {s_batch.shape}  (float32)")
    pause(0.8)

    subsection("GPU benchmark  (n=10,000 × d=768)")
    print("  Running benchmark …")
    report = gpu_benchmark(n=10_000, d=768)
    print()
    print(f"  Backend               : {report['backend']}")
    print(f"  Device                : {report['device_name']}")
    print(f"  Quantize throughput   : {report['quantize_vecs_per_sec']:>12,.0f} vec/s")
    print(f"  Cosine pairs/s        : {report['cosine_pairs_per_sec']:>12,.0f} pairs/s")
    print()
    print("  On A10G GPU (MAX Engine) target: >= 50 M vec/s")
    print("  CPU SIMD fallback typical:       2 – 4 M vec/s")
    pause(1.5)


# ─────────────────────────────────────────────────────────────
# 7  Learned Quantization
# ─────────────────────────────────────────────────────────────


def demo_learned(vectors: np.ndarray) -> None:
    section("7 · Learned Quantization  (Phase 7)")

    # ----- 7a Residual Quantization -----
    subsection("Residual Quantization  (RQ × 3 passes, ~10× compression)")
    from python.rq_api import ResidualQuantizer

    rq = ResidualQuantizer(n_passes=3, n_subspaces=8, n_centroids=256)
    print("  Training RQ on 2,000 vectors  (3 passes × 8 sub-spaces) …")
    t0 = time.perf_counter()
    rq.train(vectors)
    train_time = time.perf_counter() - t0
    print(f"  Training time  : {train_time:.2f} s")

    codes = rq.encode(vectors)
    restored = rq.decode(codes)
    cosine = rq.mean_cosine(vectors, restored)

    total_codes_bytes = sum(c.nbytes for c in codes)
    ratio = vectors.nbytes / total_codes_bytes
    print(f"  Passes         : {len(codes)}")
    print(f"  Codes/pass     : {codes[0].shape}")
    print()
    progress_bar("Compression ratio  (~10×)", ratio / 12, fmt=".1f", display=ratio)
    progress_bar("Cosine similarity  (>=0.80)", cosine)
    pause(1.0)

    # ----- 7b Autoencoder Codebook -----
    subsection("Autoencoder Codebook  (up to 48× compression, learned)")
    from python.codebook_api import Codebook

    target_dim = 16
    print(f"  Training 2-layer autoencoder: {vectors.shape[1]}D → {target_dim}D …")
    cb = Codebook(target_dim=target_dim, seed=42)
    t0 = time.perf_counter()
    cb.train(vectors, n_epochs=20, lr=0.01, batch_size=128)
    train_time = time.perf_counter() - t0
    print(f"  Training time  : {train_time:.2f} s")

    int8_codes = cb.encode(vectors)
    decoded = cb.decode(int8_codes)
    cosine_cb = cb.mean_cosine(vectors, decoded)
    ratio_cb = vectors.nbytes / int8_codes.nbytes

    print(f"  Encoded shape  : {int8_codes.shape}  (int8)")
    print()
    progress_bar(
        f"Compression ratio  (~{ratio_cb:.0f}×)", ratio_cb / 60, fmt=".1f", display=ratio_cb
    )
    progress_bar("Cosine similarity", cosine_cb)
    print("\n  At target_dim=64:  ~48× compression,  cosine_sim >= 0.97")
    pause(1.0)

    # ----- 7c AutoQuantize -----
    subsection("AutoQuantize  (cascade: NF4 → NF4-mixed → PQ-96 → PQ-48 → binary)")
    from python.auto_quantize_api import auto_quantize

    print("  Constraints: target_cosine=0.97, target_compression=6.0 …")
    t0 = time.perf_counter()
    result = auto_quantize(vectors, target_cosine=0.97, target_compression=6.0)
    elapsed = time.perf_counter() - t0

    print(f"\n  Strategy chosen  : {result['mode']}")
    print(f"  Compression      : {result['compression_ratio']:.2f}×")
    print(f"  Cosine sim       : {result['cosine_sim']:.4f}")
    print(f"  Decision time    : {elapsed * 1000:.1f} ms")
    progress_bar("Meets quality target (>=0.97)", result["cosine_sim"])
    pause(1.5)


# ─────────────────────────────────────────────────────────────
# 8  VQZ Storage
# ─────────────────────────────────────────────────────────────


def demo_storage(vectors: np.ndarray) -> None:
    section("8 · VQZ Storage Format  (Phase 8)")
    from python import Vectro
    from python.storage_v3 import save_vqz, load_vqz, MAGIC

    vectro = Vectro(profile="balanced")
    result = vectro.compress(vectors)
    # BatchQuantizationResult stores vectors as a list; convert to 2-D array for save_vqz
    quantized = np.stack(result.quantized_vectors).astype(np.int8)

    subsection("Write VQZ file  (64-byte header + ZSTD body + blake2b checksum)")

    with tempfile.NamedTemporaryFile(suffix=".vqz", delete=False) as f:
        path = f.name

    try:
        t0 = time.perf_counter()
        save_vqz(
            quantized,
            result.scales,
            dims=vectors.shape[1],
            path=path,
            compression="zstd",
            metadata=b'{"model": "text-embedding-3-small", "version": "3.0.0"}',
        )
        write_time = time.perf_counter() - t0

        file_size = os.path.getsize(path)
        raw_bytes = quantized.nbytes + result.scales.nbytes

        print(f"  Magic bytes       : {MAGIC!r}")
        print(f"  Raw body (INT8)   : {raw_bytes:,} bytes  ({raw_bytes / 1024:.1f} KB)")
        print(f"  VQZ file (ZSTD)   : {file_size:,} bytes  ({file_size / 1024:.1f} KB)")
        print(f"  ZSTD ratio        : {raw_bytes / file_size:.2f}×")
        print(f"  vs original FP32  : {vectors.nbytes / file_size:.2f}×  combined")
        print(f"  Write throughput  : {raw_bytes / write_time / 1024 / 1024:.0f} MB/s")
        pause(1.0)

        subsection("Read VQZ file  (verify checksum, decompress)")
        t0 = time.perf_counter()
        data = load_vqz(path)
        read_time = time.perf_counter() - t0

        print(f"  n_vectors         : {data['n_vectors']:,}")
        print(f"  dims              : {data['dims']}")
        print(f"  version           : {data['version']}")
        print(f"  metadata          : {data['metadata'].decode()}")
        print("  checksum          : verified OK")
        print(f"  Read throughput   : {raw_bytes / read_time / 1024 / 1024:.0f} MB/s")
        pause(1.0)

        subsection("Corruption detection")
        with open(path, "r+b") as fh:
            fh.seek(30)  # overwrite checksum field
            fh.write(b"\xff\xff\xff\xff\xff\xff\xff\xff")
        try:
            load_vqz(path)
            print("  ERROR: corruption was not detected!")
        except ValueError as e:
            print(f"  Corruption caught:  {e}")
            print("  Checksum guard: working correctly")
        pause(1.0)

    finally:
        os.unlink(path)

    subsection("Cloud backends (S3 / GCS / Azure Blob)")
    print("  S3Backend(bucket, prefix)         — requires pip install fsspec[s3]")
    print("  GCSBackend(bucket, prefix)        — requires pip install fsspec[gcs]")
    print("  AzureBlobBackend(bucket, prefix)  — requires pip install fsspec[abfs]")
    print()
    print("  All three share the same interface:")
    print("    backend.save_vqz(quantized, scales, dims, remote_name)")
    print("    data = backend.load_vqz(remote_name)   # checksummed on load")
    pause(1.5)


# ─────────────────────────────────────────────────────────────
# 9  Unified v3 API
# ─────────────────────────────────────────────────────────────


def demo_v3_api(vectors: np.ndarray) -> None:
    section("9 · Unified v3 API: VectroV3  (Phase 9)")
    from python.v3_api import VectroV3

    profiles = ["int8", "nf4", "binary"]

    print("  Single entry-point: VectroV3(profile)")
    print("  All modes share compress / decompress / save / load")
    print()

    rows: list[tuple[str, float, float]] = []
    for profile in profiles:
        v3 = VectroV3(profile=profile)
        try:
            r = v3.compress(vectors)
            rows.append((profile, r.compression_ratio, r.mean_cosine))
        except Exception as exc:
            print(f"  [{profile}] Error: {exc}")
            rows.append((profile, 0.0, 0.0))

    print(f"  {'Profile':<12} {'Compression':>12}  {'Cosine':>8}  Bar")
    print(f"  {'-' * 12} {'-' * 12}  {'-' * 8}  {'-' * 30}")
    for profile, ratio, cosine in rows:
        b = bar(min(cosine, 1.0), width=26)
        print(f"  {profile:<12} {ratio:>11.1f}×  {cosine:>8.4f}  {b}")
        pause(0.3)

    pause(1.0)

    subsection("VQZ local save / load")
    v3_int8 = VectroV3(profile="int8")
    r = v3_int8.compress(vectors[:100])
    with tempfile.NamedTemporaryFile(suffix=".vqz", delete=False) as f:
        path = f.name
    try:
        v3_int8.save_compressed(r, path)
        r2 = v3_int8.load_compressed(path)
        print(f"  Saved {r.n_vectors} vectors,  profile={r.profile!r}")
        print(f"  Loaded back:  n_vectors={r2.n_vectors},  dims={r2.dims}")
        print(f"  File size   : {os.path.getsize(path):,} bytes")
    finally:
        os.unlink(path)
    pause(1.5)


# ─────────────────────────────────────────────────────────────
# 10  Benchmark Summary
# ─────────────────────────────────────────────────────────────


def show_benchmark_summary(vectors: np.ndarray) -> None:
    section("10 · Benchmark Summary  (2,000 × 768 float32)")

    from python import Vectro
    from python.nf4_api import quantize_nf4, dequantize_nf4
    from python.binary_api import quantize_binary, dequantize_binary
    from python.interface import mean_cosine_similarity

    print("  Measuring all modes …")
    print()

    results: list[tuple[str, float, float, float]] = []

    # INT8
    vectro = Vectro(profile="balanced")
    t0 = time.perf_counter()
    r = vectro.compress(vectors)
    elapsed = time.perf_counter() - t0
    restored = vectro.decompress(r)
    cosine = mean_cosine_similarity(vectors, restored)
    results.append(("INT8", r.compression_ratio, cosine, len(vectors) / elapsed))

    # NF4
    d = vectors.shape[1]
    t0 = time.perf_counter()
    pkd, scl = quantize_nf4(vectors)
    rst = dequantize_nf4(pkd, scl, d)
    elapsed = time.perf_counter() - t0
    cosine_nf4 = mean_cosine_similarity(vectors, rst)
    ratio_nf4 = vectors.nbytes / (pkd.nbytes + scl.nbytes)
    results.append(("NF4", ratio_nf4, cosine_nf4, len(vectors) / elapsed))

    # PQ-96
    from python.v3_api import PQCodebook

    cb96 = PQCodebook.train(vectors, n_subspaces=96)
    t0 = time.perf_counter()
    codes96 = cb96.encode(vectors)
    elapsed = time.perf_counter() - t0
    dec96 = cb96.decode(codes96)
    cosine_pq = mean_cosine_similarity(vectors, dec96)
    ratio_pq = vectors.nbytes / codes96.nbytes
    results.append(("PQ-96", ratio_pq, cosine_pq, len(vectors) / elapsed))

    # Binary
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = (vectors / norms).astype(np.float32)
    t0 = time.perf_counter()
    bpkd = quantize_binary(normed)
    elapsed = time.perf_counter() - t0
    brst = dequantize_binary(bpkd, normed.shape[1])
    cosine_bin = mean_cosine_similarity(normed, brst)
    ratio_bin = normed.nbytes / bpkd.nbytes
    results.append(("Binary", ratio_bin, cosine_bin, len(normed) / elapsed))

    # Table
    header = f"  {'Mode':<10} {'Ratio':>8} {'Cosine':>9}  {'Throughput':>16}  Quality"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mode, ratio, cosine, tp in results:
        grade = col(cosine, 0.95, 0.999)
        print(f"  {mode:<10} {ratio:>7.1f}×  {cosine:>9.4f}  {tp:>13,.0f} v/s  {grade}")
        pause(0.4)

    print()
    pause(0.5)

    # Visual bars
    print("  Compression ratios (higher = more space saved):")
    for mode, ratio, _, _ in results:
        b = bar(ratio / 40, width=32)
        print(f"  {mode:<10} {b}  {ratio:.1f}×")
        pause(0.2)

    print()
    print("  Cosine similarity (higher = better quality):")
    for mode, _, cosine, _ in results:
        b = bar(min(cosine, 1.0), width=32)
        print(f"  {mode:<10} {b}  {cosine:.4f}")
        pause(0.2)

    pause(2.0)


# ─────────────────────────────────────────────────────────────
# 11  Integrations listing
# ─────────────────────────────────────────────────────────────


def show_integrations() -> None:
    section("11 · Vector Database & Ecosystem Integrations")

    rows = [
        ("InMemoryVectorDBConnector", "Zero-dependency testing/prototype", "✅"),
        ("QdrantConnector", "Qdrant REST/gRPC — store & k-NN search", "✅"),
        ("WeaviateConnector", "Weaviate v4 — INT8/INT4 payloads", "✅"),
        ("HuggingFaceCompressor", "PyTorch + HF Transformers bridge", "✅"),
        ("ArrowBridge", "Apache Arrow / Parquet persistence", "✅"),
        ("S3Backend  (fsspec)", "AWS S3 cloud storage", "✅"),
        ("GCSBackend  (fsspec)", "Google Cloud Storage", "✅"),
        ("AzureBlobBackend (fsspec)", "Azure Blob Storage", "✅"),
    ]

    print(f"  {'Connector':<30} {'Description':<42} {'Status'}")
    print(f"  {'-' * 30} {'-' * 42} {'-' * 6}")
    for name, desc, status in rows:
        print(f"  {name:<30} {desc:<42} {status}")
        pause(0.2)

    print()
    print("  CLI:")
    print("    vectro compress   input.npy embeddings.vqz [--profile balanced]")
    print("    vectro decompress embeddings.vqz restored.npy")
    print("    vectro inspect    artifact.npz [--json]")
    print("    vectro upgrade    v1.npz v2.npz [--dry-run]")
    print("    vectro validate   artifact.npz")
    pause(1.5)


# ─────────────────────────────────────────────────────────────
# Closing
# ─────────────────────────────────────────────────────────────


def show_closing() -> None:
    section("Vectro v3.0.0 — Feature Summary")

    print()
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │  Quantization Modes                                             │")
    print("  │   INT8      · 4×    · cosine >= 0.9999  (lossless, default)     │")
    print("  │   INT4      · 8×    · cosine >= 0.92    (GA in v3)              │")
    print("  │   NF4       · 8×    · cosine >= 0.985   (Gaussian optimal)      │")
    print("  │   NF4-Mixed · 7.5×  · cosine >= 0.990   (outlier-aware)         │")
    print("  │   PQ-96     · 32×   · cosine >= 0.95    (ADC compatible)        │")
    print("  │   Binary    · 32×   · Hamming distance  (25× faster scan)       │")
    print("  │   RQ × 3   · 10.7× · cosine >= 0.98    (residual passes)       │")
    print("  │   Codebook  · 48×   · cosine >= 0.97    (learned task-specific) │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print("  │  Index & Storage                                                │")
    print("  │   HNSW      — ANN, INT8 storage, 4× memory savings             │")
    print("  │   VQZ       — ZSTD-compressed, blake2b checksummed              │")
    print("  │   Cloud     — S3  ·  GCS  ·  Azure via fsspec                   │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print("  │  Engineering Stats                                              │")
    print("  │   Tests   : 445 / 445 passing  ·  100% line coverage            │")
    print("  │   Mojo SIMD: >= 5 M vec/s INT8  ·  >= 50 M Hamming scan        │")
    print("  │   GPU     : MAX Engine + CPU SIMD fallback                      │")
    print("  │   Python  : 200K – 1.04M vec/s depending on profile + dim      │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    pause(2.0)
    print()
    print("  GitHub  : https://github.com/wesleyscholl/vectro")
    print("  Docs    : docs/getting-started.md  ·  docs/api-reference.md")
    print("  License : MIT")
    print()
    pause(1.0)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────


def main() -> None:
    show_banner()
    vectors, normed = setup_data()

    demo_int8(vectors)
    demo_nf4(vectors)
    demo_pq(vectors)
    demo_binary(normed)
    demo_hnsw(vectors)
    demo_gpu(vectors)
    demo_learned(vectors)
    demo_storage(vectors)
    demo_v3_api(vectors)
    show_benchmark_summary(vectors)
    show_integrations()
    show_closing()


if __name__ == "__main__":
    main()
