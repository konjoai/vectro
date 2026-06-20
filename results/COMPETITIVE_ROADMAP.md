# Vectro — Competitive Benchmark Scorecard & Roadmap to #1

**Hardware:** Apple M3, 8 cores, 16 GB · **Dataset:** glove-100-angular (real embeddings) ·
**Methodology:** ann-benchmarks style — exact-cosine ground truth, recall-matched QPS, single-thread search (`faiss.omp_set_num_threads(1)`) unless noted. All numbers measured this session on the post-heuristic branch.

> Provenance note: encode "2.5×" is **real GloVe + Rust SIMD**. The Mojo path hits higher on synthetic L2-normalised vectors (17–118 M vec/s) — not used in the headline below.

---

## 1. ANN search — recall vs QPS (single-thread, n=20k)

| Engine | R@0.90 | R@0.95 | R@0.99 | Max R@10 | Build | Index MB |
|--------|------:|------:|------:|--------:|------:|------:|
| **vectro-hnsw (f32)** | ~85k | **~67k** | **~32k** | 0.9973 | **0.72s** | 8.6 |
| hnswlib | ~62k | ~45k | ~22k | 1.0000 | 2.73s | 10.5 |
| faiss-hnsw | **~135k** | **~100k** | **~40k** | 1.0000 | 1.77s | 10.4 |
| faiss-ivf | ~6.7k | ~6.7k | ~3.4k | 1.0000 | **0.1s** | 7.8 |

**Standing:** vectro **beats hnswlib at every recall level** and has the **fastest build**. faiss-hnsw is still **~1.5× faster** at matched recall (its win is raw per-query search speed). vectro can't quite reach literal 1.000 (caps 0.9973 — parallel-build residual).

## 2. Quantization encode (vectro's core competency)

| Method | Throughput | Compression | Recon. cosine | vs FAISS |
|--------|-----------:|------------:|--------------:|---------:|
| vectro INT8 (Rust SIMD) | **10.9 M vec/s** | 3.9× | 1.0000 | **2.5× faster** |
| faiss ScalarQuantizer INT8 | 4.4 M vec/s | 4.0× | 0.9999 | — |
| vectro PQ (M=25) | 47 K vec/s | 16× | 0.9503 | **18× slower** ❌ |
| faiss IndexPQ (M=25) | 867 K vec/s | 16× | 0.9512 | — |

**Standing:** INT8 encode is a clear win. **PQ encode is the single biggest weakness.**

## 3. Quantized HNSW — recall @ memory (unique to vectro; no competitor ships this)

| Variant | Recall@10 | Memory | Batch QPS (8-core) |
|---------|----------:|-------:|-------------------:|
| f32   | 0.986 | 1× | 34,000 |
| Int8  | **0.993** | **4× smaller** | **50,000** |
| NF4   | 0.925 | 8× smaller | 20,000 |
| Binary| 0.597 | 32× smaller | 17,000 |

**Standing:** A genuine differentiator — searchable graphs over compressed vectors. Binary is at its 1-bit metric ceiling (re-rank needed for higher).

## 4. Throughput wins already banked this session
- Parallel batch search (rayon, GIL released): ~5× → Int8 50k QPS on 8 cores.
- Alloc-free quantized distance: NF4 2.6×, Binary 1.9×.
- SIMD NEON int8 dot_query: 2×.
- Binary encode (dropped Mojo subprocess): 140×.
- HNSW build (parallel): 6.5s → 0.72s.
- Heuristic neighbour selection: max recall 0.984 → 0.9973.

---

# Roadmap to beat ALL competitors

Three remaining gaps to close: **(A) search speed vs faiss-hnsw**, **(B) PQ encode vs faiss**, **(C) literal 1.000 recall**. Plus **(D)** widen the moat where we're already unique.

## Phase 1 — Close the search-speed gap vs faiss-hnsw  *(highest value)*
Goal: match/beat faiss's ~100k QPS@R0.95. faiss wins on memory layout + tight distance loops.
1. **Flat contiguous code/vector buffer** — replace `Vec<Vec<f32>>` / per-node code `Vec`s with one strided buffer; removes a pointer-chase per distance eval. *Est. +10–20% search.*
2. **Software prefetch** the next candidate's vector/codes during beam descent (now possible with the flat buffer). *Est. +5–15%.*
3. **SIMD distance for f32 search** — ensure the hot cosine uses the widest SIMD (it's simsimd today; verify NEON 4-wide FMA, no bounds checks in the inner loop).
4. **`select_heuristic` cost** — it's O(M²) node-to-node distances per insert; cache candidate vectors / use the build f32 buffer contiguously. *Build-time, not search.*

**Beats:** faiss-hnsw on QPS@recall → vectro #1 on the search Pareto frontier.

## Phase 2 — Make PQ competitive  *(fix the one clear loss)*
Goal: PQ encode within 1.5× of faiss (from 18× slower) and better recall.
1. **Vectorize PQ codebook assignment** — current encode is scalar nearest-centroid per subquantizer; SIMD the L2-argmin (or k-means lookup) like int8. *Est. 10–18×.*
2. **OPQ rotation** (learned orthogonal pre-rotation) — closes the 32× vs 64× compression-at-quality gap and lifts recall.
3. **SIMD ADC distance tables** — precompute per-query LUTs and SIMD the table-sum (faiss does this).

**Beats:** faiss IndexPQ on encode speed; matches on compression/quality.

## Phase 3 — Reach literal R@10 = 1.000
Goal: remove the 0.9973 cap (parallel-build intra-chunk loss).
1. **Concurrent-insertion build** with per-node locks (hnswlib-style): each node searches the *live* graph (full visibility) instead of a frozen chunk → serial-quality at parallel speed. Replaces the chunk heuristic.
2. Fallback: expose a `build_quality` knob (serial = 1.000, parallel = 0.997, fast).

**Beats:** faiss/hnswlib on recall ceiling *and* build time simultaneously.

## Phase 4 — Widen the moat (be uncatchable, not just faster)
1. **Binary + re-rank pipeline** — flat binary Hamming prefilter (SIMD popcount) → INT8/f32 re-rank. Targets R@0.95 at 32× memory, a regime faiss/hnswlib can't touch.
2. **Quantized HNSW batch search for all variants** + `search_batch_np` parity (mostly done) → serving throughput crown.
3. **Mojo/Accelerate path** for encode on Apple Silicon (AMX) — push INT8 past 100 M vec/s as a headline.
4. **GPU build/search** (longer term) — the only axis where faiss has a categorical option vectro lacks.

## Suggested order & exit criteria
| Phase | Effort | Exit criterion |
|-------|--------|----------------|
| 1 | Med (Rust) | vectro QPS@R0.95 ≥ faiss-hnsw on glove-100 |
| 2 | Med-High (Rust) | vectro PQ encode ≥ 0.7× faiss; cosine ≥ faiss |
| 3 | High (concurrency) | vectro max R@10 = 1.000 with build ≤ hnswlib |
| 4 | Ongoing | a memory/throughput regime no competitor matches |

**Validation for every phase:** re-run `scripts/benchmark_comprehensive.py` (ann-benchmarks methodology, recall-matched, single-thread) on glove-100 + sift-1m + nytimes-256; never claim a win on synthetic or multi-thread without labeling it.
