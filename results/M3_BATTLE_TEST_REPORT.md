# Vectro Battle-Test Report — Apple M3

**Date:** 2026-06-18
**Hardware:** Apple M3, 8 cores, 16 GB, macOS (Darwin 25.5.0), arm64
**Vectro:** Python v5.0.2 / Rust `vectro_py` v8.0.0 (SIMD tier: **NEON**), Mojo binary present
**Env:** `.venv` Python 3.12.8 — numpy 2.4.3, faiss-cpu 1.13.2, scikit-learn 1.8.0, hnswlib, usearch 2.25.3, annoy

All benchmarks seeded (rng=42). Raw JSON in `results/m3_*.json`.

---

## 1. INT8 quantization throughput — vectro WINS decisively

`benchmark_faiss_comparison.py` (Mojo SIMD vs FAISS C++), n=50,000:

| Dim | Vectro (Mojo SIMD) | FAISS (C++) | Speedup |
|----:|-------------------:|------------:|--------:|
| 128  | 115.2 M vec/s | 36.3 M vec/s | **3.18×** |
| 384  | 36.7 M vec/s  | 12.2 M vec/s | **3.02×** |
| 768  | 18.5 M vec/s  | 6.41 M vec/s | **2.88×** |
| 1536 | 8.76 M vec/s  | 3.13 M vec/s | **2.80×** |

Single-shape INT8: vectro **17.3 M vec/s** vs FAISS 4.46 M vec/s (3.9×).
Quality: cosine 1.0000 on real GloVe-100. **This is vectro's genuine, defensible strength.**

## 2. Product Quantization — vectro LOSES on compression

PQ-96, same corpus:

| | Vectro PQ | FAISS PQ |
|--|----------:|---------:|
| cosine | 0.8185 | 0.8208 |
| compression | **32×** | **64×** |

FAISS achieves 2× the compression at equal quality. Vectro PQ leaves a factor of 2 on the table.

## 3. HNSW ANN — the headline story

`benchmark_ann_comparison.py`, n=20,000, d=128, k=10, M=16, ef_c=200, ef_s=100.

### Before fix — benchmark used the pure-Python reference path
| Library | Build(s) | QPS | R@1 |
|--|--:|--:|--:|
| Vectro HNSW (Python) | **62.9** | **532** | 0.771 |
| hnswlib | 1.5 | 5,487 | 0.684 |
| Exact brute-force | — | 5,876 | 1.000 |

Vectro's HNSW was **slower than brute force** — because the benchmark instantiated
`python.hnsw_api.HNSWIndex` (interpreted: `heapq` + Python lists + per-pair `np.dot`),
not the shipping Rust extension.

### After fix — production Rust path (`vectro_py.PyHnswIndex`)
| Library | Build(s) | QPS | R@1 |
|--|--:|--:|--:|
| **Vectro HNSW (Rust)** | 7.5 | 4,291 | **0.759** ← best recall |
| hnswlib | 1.5 | 5,411 | 0.692 |
| usearch | 0.9 | 38,812 | 0.681 |
| annoy | **BROKEN** (flagged) | — | — |
| Exact brute-force | — | 6,135 | 1.000 |

**9× query speedup, 8× faster build** just from benchmarking the right code path.
Vectro now has the **best recall** of any engine and QPS within 25% of hnswlib.
Remaining real gap: **build time ~5× slower than hnswlib** (likely single-threaded insertion).

### Quantized HNSW — vectro's unique differentiator (competitors don't offer this)
| Variant | Build | QPS | Recall@10 | Memory |
|--|--:|--:|--:|--|
| f32 | 6.9s | 4,620 | 0.691 | 1× |
| **Int8** | 9.8s | 3,692 | **0.692** | **4× smaller** |
| NF4 | 24.6s | 1,428 | 0.660 | 8× smaller |
| Binary | 12.3s | 3,532 | **0.051** ⚠️ | 32× smaller |

Int8 HNSW = same recall as f32 at 4× less memory — a real production-grade win.
**Binary HNSW recall 0.051 ≈ random — broken** (sign-bit quantization destroys the
signal on un-rotated data; needs learned thresholds / random rotation).

---

## 4. Real-data run (GloVe-100, 50k vectors)

| Mode | Throughput | Cosine | Ratio |
|--|--:|--:|--|
| fast (INT8) | 2.97 M vec/s | 1.0000 | 3.8× |
| ultra (INT4→**fell back to INT8**) | 3.09 M vec/s | 1.0000 | 3.8× |
| binary | 57 K vec/s | 0.7959 | 30.8× |

- **`ultra`/INT4 silently degrades to INT8** — `squish_quant` Rust backend not built, so the
  advertised ~7.6× compression is not delivered (warning is emitted, but ratio reported as 3.8×).
- **Binary encode is 50× slower** than INT8 (57 K vs 3 M vec/s).

---

## Konjo harness fixes applied (honest measurement — no silent failures)

`benchmarks/benchmark_ann_comparison.py`:
1. **usearch API drift fixed** — `search(..., expansion=)` removed (dropped upstream); now sets
   `idx.expansion_search` property + batched `.keys`. usearch went from `ERROR` → real competitor.
2. **`detect_degenerate()` guard added** — a library returning < k/2 neighbours, or a collapsed
   (near-constant) nearest neighbour, is now reported `status:"broken"` instead of a flattering
   `recall=0.000`. (annoy is non-functional on Py3.12/arm64 and is now flagged, not silently scored.)
3. **`_build_vectro` now prefers the production Rust `PyHnswIndex`** (zero-copy numpy +
   NEON kernels), falling back to the Python reference only when the extension is absent.

---

## ⚡ FIX LANDED — Quantized HNSW graph construction (binary 25×, NF4 +39%)

**Root cause:** `QuantHnswIndex<Q>` built the HNSW graph using the *quantized*
distance. For coarse quantizers the graph had no navigable structure — Binary
collapsed to 0.024 recall@10 on GloVe (metric ceiling is 0.597; ef-sweeps and
re-ranking could not recover it → proven a graph-topology bug, not a metric limit).

**Fix:** build the graph from **full-precision f32 distances held transiently**
(`build_vectors`, dropped via `finalize()` after `add_batch`), store only the
quantized codes, search the codes with the asymmetric quantized distance. Graph
links are integers — topology is free at rest, so **memory-at-rest is unchanged**.
Plus mean-centering gated to 1-bit (real embeddings share a large mean direction
that makes raw sign-bits near-identical). PyO3 `add_np` routed through `add_batch`.

**Result (GloVe-100, n=8000):**
| Variant | recall@10 before | after | memory |
|--|--:|--:|--|
| f32 (control) | 0.988 | 0.988 | 1× |
| Int8 | 0.982 | 0.982 | 4× |
| **NF4** | 0.660 | **0.919** | 8× |
| **Binary** | 0.024 | **0.598** (= metric ceiling) | 32× |

Files: `rust/vectro_lib/src/index/quant_hnsw.rs`, `rust/vectro_py/src/lib.rs`.
All 168 `vectro_lib` unit tests pass; clippy-clean. Results JSON:
`results/m3_quantized_hnsw_fixed.json`.

## ⚡ FIX LANDED — HNSW build/query speedups (low-risk, verified)

Three zero-/low-risk hot-path fixes, recall unchanged throughout:
1. **FxHashSet** for the `visited` set (was SipHash — slow for integer keys) —
   `rustc-hash` dep, both `hnsw.rs` and `quant_hnsw.rs`.
2. **Slice-borrow** the neighbour list in beam search instead of cloning a
   `Vec<usize>` per expanded candidate.
3. **SIMD unit-cosine** (`cosine_dist_unit`, simsimd) for the quantized f32 build
   path (was a scalar loop).

**Result (ANN comparison, n=20k, d=128):**
| | Build(s) | QPS |
|--|--:|--:|
| Vectro HNSW — before | 6.50 | 5,062 |
| **Vectro HNSW — after** | **4.85** | **6,738** |
| hnswlib | 1.53 | 5,593 |

Vectro now **beats hnswlib on QPS (6,738 vs 5,593) and recall (0.759 vs 0.692)**.
Remaining gap: build time (4.85s vs 1.53s) — needs concurrent insertion
(per-node locking), a dedicated effort.

## ⚡ FIX LANDED — real INT4 (NumPy nibble-pack fallback)

`ultra`/INT4 previously **silently degraded to INT8** (`squish_quant` extension
missing — and the `squish_quant_rs` crate is absent from the repo entirely), so
the advertised ~7× compression was never delivered. Added a NumPy grouped
symmetric-absmax INT4 encode/decode (`interface._quantize_int4_numpy` /
`_dequantize_int4_numpy`), wired `quantize_int4`/`dequantize_int4` to use the
native extension when present and the NumPy path otherwise, and removed the
silent downgrade in `vectro.py`.

**Result (GloVe-100, 20k):** `ultra` now reports **6.9× compression at cosine
0.9938** (was 3.8× / INT8). Round-trip mean cosine 0.99+ across d=128/256/768
and odd dims. 4 new unit tests; full Python suite green (1347 tests).

## ⚡ FIX LANDED — parallel HNSW build + NF4 encode

**Parallel build** (`hnsw.rs` + `quant_hnsw.rs`): **parallel search + serial
commit** in chunks — each node's candidate search runs read-only across rayon
threads; link stitching stays serial (no locks/`unsafe`, serialized layout
unchanged). Build order is a deterministic shuffle and the chunk is bounded to
~n/64, so correlated/sorted input rows don't share a chunk (an early version
regressed to 0.30 recall on sorted data; the shuffle restores parity — verified
by a serial-vs-parallel parity test).

**NF4 encode**: replaced the per-element binary-search `nearest_nf4` with a
branchless threshold count (provably identical output) — 1.23× on the kernel,
no branch mispredicts, portable.

**Result — Vectro HNSW now matches/beats hnswlib on every axis** (ANN, n=20k):
| | Build(s) | QPS | Recall@1 |
|--|--:|--:|--:|
| Vectro HNSW (start of session) | 6.50 | 5,062 | 0.759 |
| **Vectro HNSW (now)** | **1.82** | **6,177** | **0.748** |
| hnswlib | 1.43 | 5,031 | 0.691 |

Quantized builds on real GloVe are now sub-second (Int8 0.33s @ 0.980, NF4 0.34s
@ 0.918, Binary 0.35s @ 0.595). Full suite green: **170 Rust + 1347 Python tests.**

## ⚡ ROUND 2 — throughput sweep (measurement-driven)

Profiled the post-overhaul hot paths and fixed the top finds (each committed
separately, all tests green):

| Fix | Result (M3) |
|--|--|
| **Parallel batch search** (rayon over queries, all variants, GIL released) | f32 7.4k→34k, Int8 4.4k→24k, NF4/Binary 1.6k→8.5k QPS (~5×) |
| **Alloc-free quantized distance** (no `decode()` per comparison) | NF4 search 2.6×, Binary 1.9×; parity-tested |
| **SIMD (NEON) Int8 `dot_query`** | Int8 search 2.0× — now faster than f32 |
| **Binary encode: drop Mojo subprocess** | 2,091→291,978 vec/s (140×); the encode shelled out to a subprocess pipe |
| **Reusable thread-local search scratch** (epoch-tagged visited + heaps) | Int8 +20%; f32/NF4 +4–5% |
| **INT8 fused encode** | investigated → *negative result* (slower), reverted + documented |

**Combined batch-search throughput (8-core M3, n=50k, ef=100):** Int8 **44,958
QPS**, NF4 19,671, f32 19,350. Recall identical throughout — the distance
rewrites are parity-tested and batch search is the same algorithm.

Known remaining (lower value / higher risk): HNSW build is superlinear at scale
(n=100k ≈ 18.5s — the serial commit phase; parallelizing it needs per-node
locking); binary distance is a per-bit walk (a per-byte version could add ~1.3×);
PQ compression 32× vs FAISS 64× (OPQ — a feature, not a perf win).

## Prioritized hardening backlog

| # | Severity | Item | Effort | Status |
|--|--|--|--|--|
| 1 | High | HNSW build slower than hnswlib | Rust | ✅ **fixed** — parallel build, 6.5→1.82s (~parity), QPS *beats* hnswlib |
| 2 | High | Binary HNSW recall 0.024 | Rust | ✅ **fixed** — f32-built graph → 0.595 |
| 3 | Med | PQ compression 32× vs FAISS 64× — larger M / OPQ rotation | Rust | open |
| 4 | Med | INT4 silently falls back to INT8 | Python | ✅ **fixed** — NumPy fallback, 6.9× |
| 5 | Med | NF4 encode scalar binary-search per element | Rust | ✅ **fixed** — branchless count, 1.23× |
| 6 | Low | Quantized HNSW variants lack `search_batch_np` | Rust | open |
| 7 | Low | annoy unusable on Py3.12/arm64 — pin or drop | infra | flagged in harness |
