> **⚠️ This page is a historical record (March 2026, v3.5.0, Apple M3 + Mojo SIMD).**
> For the **current** status, head-to-head benchmarks, wins/ties/losses ledger,
> the 100M+ at-scale test plan, competitor benchmark scales, and the
> publication/licensing analysis, see **[STATUS_AND_BENCHMARKS.md](./STATUS_AND_BENCHMARKS.md)**.
>
> **Current highlights (v5.19.0, Rust kernels, x86 — indicative):**
> INT8 quantize **6.2× faster than FAISS** (was 4.6× on Mojo/M3) · PQ-96 encode
> **~0.88× FAISS** after the AVX2 kernel (was ~5× behind) · HNSW recall@10
> **0.917**, leading hnswlib (0.899) and faiss `IndexHNSWFlat` (0.904) at matched
> params. Largest validated scale remains ~1.18M (GloVe-100); 100M+ is planned.

# Faiss Comparison Results — March 12, 2026

## Backend: Mojo SIMD (v3.5.0)

This benchmark ran with Vectro in **Mojo SIMD mode** (`vectro_quantizer` binary built via
`pixi run build-mojo`). Python/NumPy fallback numbers are included for reference.

To reproduce:
```bash
pixi install && pixi shell
pixi run build-mojo
python benchmarks/benchmark_faiss_comparison.py --output results/faiss_comparison_mojo.json
```

---

## Summary

Vectro has been benchmarked against Faiss 1.13.2 on two key quantization metrics.

### INT8 Quantization Throughput

**Configuration:**
- 100,000 vectors of 768 dimensions
- Best-of-5 timed iterations (2 warmup iterations discarded)
- Hardware: Apple M3 Pro (quiet CPU — background processes terminated)

| Library | Throughput | vs FAISS |
|---------|:----------:|:--------:|
| Python/NumPy (baseline) | 61,589 vec/s | 0.02× |
| **Vectro Mojo SIMD** | **12,121,212 vec/s** | **4.59×** |
| FAISS C++ ScalarQuantizer | 2,639,653 vec/s | 1.00× |

**Key result:** Vectro Mojo SIMD is **4.59× faster than FAISS C++** at INT8 quantization.

The speedup comes from three compounding improvements over the Python fallback:
1. `SIMD_W=16`: tiles 4 NEON loads per `vectorize` call for software pipelining
2. `resize()` init: replaces per-element append loops with `memset`-equivalent init (~6× allocation speedup)
3. Pipe IPC: Python calls the Mojo binary via stdin/stdout, eliminating all disk I/O

### Product Quantization Quality (M=96)

**Configuration:**
- 50,000 vectors of 768 dimensions
- Training set: 25,000 vectors; Test set: 25,000 vectors
- 96 subspaces (PQ-96)

| Library | Training Time | Compression Time | Cosine Sim | Compression Ratio | Throughput |
|---------|:-------------:|:----------------:|:----------:|:----------------:|:----------:|
| **Vectro (Python)** | 12.4s | 3.72s | **0.8185** | 32.0x | 13,402 vec/s |
| **Faiss (C++)** | 7.5s | 0.59s | 0.8207 | 64.0x | 32,799 vec/s |

**Interpretation:**
- ✅ **Quality parity:** Vectro and Faiss achieve equivalent cosine similarity (0.8185 vs 0.8207)
- ⚠️ **PQ throughput gap:** Faiss C++ is ~2.4× faster for PQ training/encoding — expected; Vectro PQ uses Python/sklearn for centroid training
- 🚀 **INT8 wins:** Vectro Mojo SIMD decisively outperforms FAISS C++ for raw INT8 quantization throughput

## Key Findings

1. **INT8 quality is equivalent:** Both achieve cosine sim ≥ 0.9999 for INT8 at d=768
2. **INT8 throughput: Vectro wins 4.59×** due to SIMD vectorization + parallelization across rows
3. **PQ quality parity:** 0.8185 vs 0.8207 at PQ-96 — Vectro's centroid training produces equivalent codebooks
4. **Vectro advantages beyond raw throughput:**
   - Native HNSW index (no separate library required)
   - NF4 quantization (normal-float 4-bit, better quality than INT4)
   - ONNX export
   - VQZ storage format with cloud backends
   - 6 vector DB connectors (Qdrant, Weaviate, Milvus, Chroma, Pinecone + custom)

## Full Benchmark Table

| Metric | Vectro (Python) | Vectro (Mojo SIMD) | FAISS C++ |
|--------|:---------------:|:------------------:|:---------:|
| INT8 throughput (d=768) | 61,589 vec/s | **12,121,212 vec/s** | 2,639,653 vec/s |
| INT8 vs FAISS | 0.02× | **4.59×** | 1.00× |
| PQ-96 cosine_sim | 0.8185 | 0.8185 | 0.8207 |
| PQ-96 throughput | 13,402 vec/s | 13,402 vec/s | 32,799 vec/s |
| Native HNSW | ✅ | ✅ | ❌ (separate) |
| NF4 quantization | ✅ | ✅ | ❌ |
| ONNX export | ✅ | ✅ | ❌ |

## Next Steps

- Add multi-dimensional INT8 throughput analysis (d=128, 384, 768, 1536) — v3.6.0
- Benchmark Vectro HNSW recall@10 vs hnswlib, annoy, usearch — v3.6.0
- Run on real embedding data (GloVe-100, SIFT1M) — v3.6.0

---

**Generated:** 2026-03-12 (updated 2026-03-12 for v3.5.0 Mojo SIMD results)
**Faiss Version:** 1.13.2
**Vectro Version:** 3.5.0 (Mojo SIMD, `pixi run build-mojo`)
**Raw results:** `results/faiss_comparison_mojo.json`
