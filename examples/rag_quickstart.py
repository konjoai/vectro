"""RAG (Retrieval-Augmented Generation) quickstart with Vectro.

Demonstrates the full pipeline:
  1. Encode documents with a HuggingFace sentence-transformer
  2. Compress embeddings with Vectro (4× storage savings)
  3. Store compressed embeddings in an in-memory vector DB
  4. Run a semantic search query

Install dependencies:
    pip install "vectro[integrations]"
    pip install sentence-transformers

Run:
    python examples/rag_quickstart.py
"""

from __future__ import annotations

import time

import numpy as np

# ---------------------------------------------------------------------------
# 1. Simulate document embeddings
#    (Replace this section with a real sentence-transformers encoder in
#     production; see the commented block below.)
# ---------------------------------------------------------------------------

DOCUMENTS = [
    "Vectro compresses LLM embeddings to INT8 with minimal quality loss.",
    "Retrieval-augmented generation grounds LLM responses in external facts.",
    "Vector databases store embeddings for fast approximate nearest neighbor search.",
    "Scalar quantization maps float32 values to integer buckets using a scale factor.",
    "Apache Arrow provides a language-independent columnar memory format.",
    "Sentence transformers convert text into fixed-length semantic embedding vectors.",
    "The cosine similarity between two vectors measures their directional alignment.",
    "INT4 quantization achieves 8× compression at the cost of slight accuracy loss.",
    "Qdrant and Weaviate are purpose-built vector databases for production RAG systems.",
    "Compression ratios of 4× reduce storage costs without meaningful recall degradation.",
]

print("=== Vectro RAG Quickstart ===\n")

# Simulate 384-dim embeddings (e.g., all-MiniLM-L6-v2 output shape)
EMBEDDING_DIM = 384
rng = np.random.default_rng(42)
doc_embeddings = rng.standard_normal((len(DOCUMENTS), EMBEDDING_DIM)).astype(np.float32)
# Normalise to unit sphere (typical for sentence-transformers output)
doc_embeddings /= np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

# Uncomment the block below to use a real encoding model:
# -------------------------------------------------------
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# doc_embeddings = model.encode(DOCUMENTS, convert_to_numpy=True, normalize_embeddings=True)
# -------------------------------------------------------

print(f"Documents    : {len(DOCUMENTS)}")
print(f"Embedding dim: {EMBEDDING_DIM}")

# ---------------------------------------------------------------------------
# 2. Compress embeddings with Vectro
# ---------------------------------------------------------------------------
from python import Vectro, decompress_vectors
from python.interface import mean_cosine_similarity

vectro = Vectro()

t0 = time.perf_counter()
batch_result = vectro.compress_batch(doc_embeddings)
compress_ms = (time.perf_counter() - t0) * 1000

print("\n--- Compression ---")
print(f"Compression ratio: {batch_result.compression_ratio:.2f}×")
print(f"Original size    : {batch_result.total_original_bytes:,} bytes")
print(f"Compressed size  : {batch_result.total_compressed_bytes:,} bytes")
print(f"Elapsed          : {compress_ms:.1f} ms")

# Save to disk
vectro.save_compressed(batch_result, "/tmp/rag_demo_embeddings.npz")
print("Saved to /tmp/rag_demo_embeddings.npz")

# ---------------------------------------------------------------------------
# 3. Store in the in-memory vector DB
# ---------------------------------------------------------------------------
from python.integrations import InMemoryVectorDBConnector

store = InMemoryVectorDBConnector()

# Reconstruct float32 for storage (the DB decompresses internally)
restored_embeddings = decompress_vectors(batch_result)
batch_id = store.store_batch(
    restored_embeddings,
    metadata={"source": "rag_quickstart", "n_docs": len(DOCUMENTS)},
)
print(f"\nStored batch {batch_id!r} in InMemoryVectorDBConnector")

# Measure reconstruction quality
cos_sim = mean_cosine_similarity(doc_embeddings, restored_embeddings)
print(f"Mean cosine similarity (original vs reconstructed): {cos_sim:.4f}")

# ---------------------------------------------------------------------------
# 4. Semantic search
# ---------------------------------------------------------------------------
QUERY = "How does vector quantization reduce memory usage?"
print("\n--- Semantic Search ---")
print(f"Query: {QUERY!r}")

# Simulate query encoding (replace with real model in production)
query_embedding = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
query_embedding /= np.linalg.norm(query_embedding)

# Uncomment to use the real model:
# query_embedding = model.encode([QUERY], normalize_embeddings=True)[0]

# Search: exact cosine similarity over the in-memory store
scores = restored_embeddings @ query_embedding  # dot product = cosine sim (unit vecs)
top_k = min(3, len(DOCUMENTS))
top_indices = np.argsort(scores)[::-1][:top_k]

print(f"\nTop-{top_k} results:")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank}. [{scores[idx]:.3f}] {DOCUMENTS[idx]}")

# ---------------------------------------------------------------------------
# 5. Inspect the saved artifact
# ---------------------------------------------------------------------------
from python.migration import inspect_artifact

print("\n--- Artifact Inspection ---")
info = inspect_artifact("/tmp/rag_demo_embeddings.npz")
print(f"Format version : v{info['format_version']}")
print(f"Artifact type  : {info['artifact_type']}")
print(f"Vectors        : {info['n_vectors']} × {info['vector_dim']}")
print(f"Precision mode : {info['precision_mode']}")
print(f"Needs upgrade  : {info['needs_upgrade']}")

print("\nDone.")
