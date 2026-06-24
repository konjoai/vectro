# vectro Performance Optimization — Results & Roadmap

A full-repo performance audit (every Rust kernel, the PyO3 FFI layer, the Python
hot paths) carried out via six parallel deep-review passes, then implemented and
benchmarked opportunity-by-opportunity. This document records **what shipped**,
**what was rejected with data**, and the **remaining roadmap**.

Benchmark host for this campaign: x86_64, 4-core Intel Xeon @ 2.10 GHz, AVX2 +
AVX-512F + FMA, **260 MiB L3**, glibc malloc, `target-cpu=x86-64-v3`, release
(fat LTO, codegen-units=1). Negative/inconclusive results are detailed in
`PERF_FINDINGS.md`.

---

## Shipped — measured wins

### Campaign 2 (AVX-512 distance kernels — this sweep)

| Change | Win (this host) | PR |
|--------|-----------------|----|
| **AVX-512-VNNI INT8 distance** — `vpdpbusd` integer dot replaces per-call i8→f32 widen; query quantised once/search via new `Quantizer::Prepared`; in-register XOR→u8, no serialization change | INT8 quant-HNSW **1.52× QPS** (kernel 1.6–2.7×), recall-neutral | #86 (merged) |
| **NF4 AVX2 codebook-LUT distance** — the last scalar quant distance; two `permutevar8x32_ps` + blend, 8 dims/iter | NF4 search **2.87× QPS** (kernel 3.9–4.7×), recall-neutral | #87 (merged) |
| **BF16 AVX-512 16-wide widen** — bf16 path is load/widen-bound (not FMA-bound), so AVX-512 wins where the f32 AVX-512 kernels lost | BF16 search **1.07× QPS** (kernel 1.4×), bit-identical | #88 (merged) |

Net effect of campaign 2: the three most-used quantized distance kernels are
materially faster with zero recall impact. NF4 is no longer the slow mode (it was
the last scalar distance). INT8 — the flagship — gained the largest absolute QPS.

### Campaign 1 (foundational kernels)

| Change | Win (this host) | PR |
|--------|-----------------|----|
| AVX2 `l2_sq` kernel (was scalar on x86_64) + IVF SIMD routing + partial-sorts + branchless binary | `l2_sq` **6–12×** kernel | #73 (merged) |
| **Binary** asymmetric distance — AVX2 sign-flip kernel | Binary HNSW **7.0k → 22.5k qps (3.2×)** | #77 |
| **BF16 / SQ2 / SQ3** asymmetric distance — AVX2 kernels (BF16 `<<16` widen; SQ affine + `srlv` unpack) | **BF16 6.2× · SQ2 5.0× · SQ3 5.1×** | #78 |
| **PyO3** GIL release on single-query `search_np` (all 4 index types) | Concurrent serving **3.8–3.9× scaling** on 4 cores | #79 |
| **IVF-PQ ADC scan** — 4-accumulator reduction (breaks the f32 `.sum()` chain) + DRY helper | IVF-PQ search **+10–14%** | #76 |
| **Quant-HNSW prefetch** — `Quantizer::prefetch` hook, PF=2 pipeline | **1.4–1.6×** on >L3 indexes (probe loop) | #75 |
| Small wins: `asarray` over `astype`, float32 SQ decode, O(k) `recall_at_k` | per-call copy + dtype + eval | #80 |

Net effect: the four scalar-distance quant modes (binary, BF16, SQ2, SQ3) went
from **5–10× slower than INT8** to **on par with it**, IVF-PQ search is ~10–14%
faster, and multi-threaded query serving now scales across cores.

---

## Rejected with data (see `PERF_FINDINGS.md`)

- **HNSW thread-local scratch heaps** (#74) — no measurable win single-thread or
  4-core concurrent; the heaps were already pre-sized and tcache absorbs the
  rest. Reverted; may help p99 on 32+ core hosts (untestable here).
- **AVX-512 f32 distance kernels** — implemented and benchmarked at **0.76–0.94×
  of AVX2** on this CPU (double-pumped 512-bit units + costlier reduce). Removed;
  AVX2 is the fastest portable x86 width here.

## Skipped after measurement (Amdahl / already-resolved)

- **IVF-PQ ADC LUT reformulation** — the LUT build is only **2.5%** of query time
  and already SIMD via #73's AVX2 `l2_sq`; the transposed-layout rewrite (needs a
  serialization migration) is not worth ≤1% overall.
- **Coarse k-means across-K SIMD** — already does a SIMD dot per centroid via
  #73; the across-K kernel saves only horizontal-reduction overhead (~1.3×
  build-time) and needs a new u32 argmax-dot kernel (current `assign_nearest`
  returns u8, k≤256, but `n_lists` exceeds 256 on large datasets). Deferred.
- **SQ2/SQ3 stored norms** — microbenched flat (1.00×): the kernel is
  code-unpack-bound, not FMA-bound, so the query-independent norm FMA is free on
  the second port. Not shipped (see `PERF_FINDINGS.md`).
- **Shared f32 `dot_f32`/`l2_sq` micro-tuning** (`hadd`→shuffle, 6/8
  accumulators) — load-port-bound, all variants within noise. Not shipped.

---

## Roadmap — remaining high-value opportunities

Ordered by expected impact. Items 1 and 3 from the original list shipped this
sweep (NF4 SIMD distance) and earlier (#84 IVF-PQ4 fast-scan); the PQ4 PyO3
binding + 2-nibble packing are in flight separately.

### 1. PyO3 — finish the FFI modernization *(highest remaining value)*
- Release the GIL on `train` (k-means) and `add_batch` (parallel build) — the
  heaviest kernels, currently serialized. Mirror the single-query `search_np` fix
  (#79, 3.8–3.9× concurrent). Mechanical: own the rows before `allow_threads`.
- NF4 batch FFI: `_rust_bridge.encode_nf4_batch` does per-row `row.tolist()`
  crossings (N FFI calls + N·D boxed floats); add a batched `quantize_nf4_batch`
  Rust entry mirroring `quantize_int8_batch`. ~10–30× NF4 batch encode.
- Return packed numpy arrays from batch search / `decode` instead of
  list-of-tuples. 2–5× large-batch marshalling.
- `dequantize_int8_batch` writes into a zero-inited `Vec` then copies into a new
  PyArray — write into an uninit `PyArray2` under `allow_threads` (≈2×, like the
  encode path).

### 2. Quantized-HNSW flat code store
Replace `Vec<Q::Encoded>` (array-of-structs, per-node heap allocation) with one
contiguous strided code buffer. Complements the prefetch already shipped (#75):
removes the ~24-bytes/node `Vec`-header overhead and the pointer-chase. Touches
the `Quantizer` trait + serialization (migration needed).

### 3. Binary symmetric Hamming — `avx512_vpopcntdq`
Verify whether simsimd v6 dispatches `vpopcntq` on this host; if not, a ~15-line
`_mm512_popcnt_epi64` kernel for `hamming_search`. Up to 2–4× — but measure
first, may be a no-op. (The asymmetric binary distance is already AVX2, #77.)

### 4. NEON kernels for the new distance paths
The aarch64 (Apple) path is still scalar for INT8-VNNI-equiv / NF4 / BF16 /
SQ2/SQ3 (no regression, but leaves the M-series on the table). Port each to NEON
(`tbl`, `vsubq`, `vfmaq`; INT8 via `sdot`).

### 5. IVF coarse-assign batched GEMM + SIMD argmin
`add` / single-query `top_coarse` score centroids in a serial scalar loop; route
through a batched `B·Cᵀ` GEMM + a u32 across-K argmin (n_lists > 256). Large on
IVF/IVF-PQ build time; also subsample coarse k-means training at very large N.

### 6. AMX-INT8 batch distance
This host exposes `amx_int8` (a separate tile execution unit). For batched /
IVF-coarse / brute-force INT8 scoring (`Q·Cᵀ` matmul) it can far exceed VNNI, but
needs OS tile enablement + a min-batch crossover. High lift, high ceiling.

---

## Method notes (the Konjo discipline applied here)

- Every change was benchmarked before/after on the target hardware; the win (or
  its absence) is recorded in the PR.
- When the in-repo benchmark couldn't exercise the bottleneck, an isolated
  microbenchmark at the right scale (often with a built-in correctness assert)
  was used to confirm or refute the mechanism *before* touching the repo.
- Opportunities that didn't pan out were **documented with numbers**, not shipped
  (SQ stored norms, f32 micro-tuning this sweep).
- All shipped PRs keep the Rust tests + clippy green, with a SIMD-vs-scalar
  parity test and a tracked criterion bench case per kernel.

