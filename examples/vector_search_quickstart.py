"""Vector search quickstart with Vectro.

Demonstrates:
  1. Compressing a dataset of embeddings with different profiles
  2. Memory-efficient reconstruction with StreamingDecompressor
  3. Quality comparison across profiles
  4. Approximate nearest-neighbour search on compressed vectors
  5. Running the benchmark harness

Run:
    python examples/vector_search_quickstart.py
"""

from __future__ import annotations

import time

import numpy as np

VECTOR_DIM = 768  # BERT / MPNet output size
N_DOCS = 2_000
N_QUERIES = 10
TOP_K = 5

print("=== Vectro Vector Search Quickstart ===\n")

# ---------------------------------------------------------------------------
# 1. Generate synthetic embeddings
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
dataset = rng.standard_normal((N_DOCS, VECTOR_DIM)).astype(np.float32)
dataset /= np.linalg.norm(dataset, axis=1, keepdims=True)

queries = rng.standard_normal((N_QUERIES, VECTOR_DIM)).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

print(f"Dataset  : {N_DOCS:,} vectors × {VECTOR_DIM} dimensions")
print(f"Queries  : {N_QUERIES}")

# ---------------------------------------------------------------------------
# 2. Ground-truth nearest neighbours (exact, using float32)
# ---------------------------------------------------------------------------
similarity_matrix = dataset @ queries.T  # (N_DOCS, N_QUERIES)
ground_truth = np.argsort(-similarity_matrix, axis=0)[:TOP_K].T  # (N_QUERIES, TOP_K)

# ---------------------------------------------------------------------------
# 3. Compress with multiple profiles and measure recall@K
# ---------------------------------------------------------------------------
from python import Vectro, decompress_vectors, get_compression_profile
from python.interface import mean_cosine_similarity

vectro = Vectro()

print(f"\n{'Profile':<12} {'Ratio':>8} {'CosSim':>8} {'Recall@5':>10} {'Time ms':>9}")
print("-" * 55)

profile_names = ["speed", "balanced", "quality"]

for profile_name in profile_names:
    profile = get_compression_profile(profile_name)

    t0 = time.perf_counter()
    result = vectro.compress_batch(dataset, profile=profile)
    compress_ms = (time.perf_counter() - t0) * 1000

    restored = decompress_vectors(result)
    cos_sim = mean_cosine_similarity(dataset, restored)

    # Compute recall@K against ground truth
    sim_matrix = restored @ queries.T
    approx_nn = np.argsort(-sim_matrix, axis=0)[:TOP_K].T

    recall_total = 0.0
    for q_idx in range(N_QUERIES):
        gt_set = set(ground_truth[q_idx])
        approx_set = set(approx_nn[q_idx])
        recall_total += len(gt_set & approx_set) / TOP_K
    recall = recall_total / N_QUERIES

    print(
        f"{profile_name:<12} {result.compression_ratio:>7.2f}× "
        f"{cos_sim:>8.4f} {recall:>10.4f} {compress_ms:>8.1f}"
    )

# ---------------------------------------------------------------------------
# 4. Streaming decompression
# ---------------------------------------------------------------------------
from python import StreamingDecompressor

print("\n--- Streaming Decompression ---")
result = vectro.compress_batch(dataset)

chunk_sizes = []
t0 = time.perf_counter()
for chunk in StreamingDecompressor(result, chunk_size=256, backend="auto"):
    chunk_sizes.append(len(chunk))
stream_ms = (time.perf_counter() - t0) * 1000

print(f"Decompressed {sum(chunk_sizes):,} vectors in {len(chunk_sizes)} chunks")
print(f"Elapsed: {stream_ms:.1f} ms  (chunk_size=256)")

# ---------------------------------------------------------------------------
# 5. Save, inspect, and validate the best result
# ---------------------------------------------------------------------------
from python.migration import inspect_artifact, validate_artifact

vectro.save_compressed(result, "/tmp/vector_search_demo.npz")

info = inspect_artifact("/tmp/vector_search_demo.npz")
print("\n--- Artifact: /tmp/vector_search_demo.npz ---")
print(f"Format version : v{info['format_version']}")
print(f"Vectors        : {info['n_vectors']:,} × {info['vector_dim']}")
print(f"Precision mode : {info['precision_mode']}")
print(f"Compression    : {info['compression_ratio']:.2f}×")
print(f"File size      : {info['file_size_bytes'] / 1024:.1f} KB")

validation = validate_artifact("/tmp/vector_search_demo.npz")
print(
    f"Validation     : {'✓ valid' if validation['valid'] else '✗ ' + ', '.join(validation['errors'])}"
)

# ---------------------------------------------------------------------------
# 6. Quick benchmark harness
# ---------------------------------------------------------------------------
from python.benchmark import BenchmarkSuite

print("\n--- Benchmark Suite (n_runs=2) ---")
# Use a small slice to keep the demo fast
sample = dataset[:200]
suite = BenchmarkSuite(sample, n_runs=2)
report = suite.run_all()

print(f"Best profile      : {report.best_profile}")
print(f"Best compression  : {report.best_compression_ratio:.2f}×")
print(f"Best mean cos sim : {report.best_recall:.4f}")

report.to_json("/tmp/vector_search_benchmark.json")
print("Report saved to /tmp/vector_search_benchmark.json")

print("\nDone.")
