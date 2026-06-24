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

---

## Roadmap — remaining high-value opportunities

Ordered by expected impact. Each is scoped for a dedicated effort.

### 1. IVF-PQ PQ4 fast-scan — the 3–5× IVF-PQ lever
FAISS-style 4-bit (K=16) codes in an **interleaved layout** with an in-register
`pshufb`/`tbl` table lookup (16–32 codes/instruction). This is the single biggest
IVF-PQ QPS lever, but a substantial new-format feature: new code layout, uint8
LUT quantization with periodic uint16 normalization, and SIMD accumulation. Can't
reuse the current K=256 format — it's a parallel index variant (`IVFPQFastScan`).

### 2. Quantized-HNSW flat code store
Replace `Vec<Q::Encoded>` (array-of-structs, per-node heap allocation) with one
contiguous strided code buffer. Complements the prefetch already shipped (#75):
removes the ~24-bytes/node `Vec`-header overhead and the pointer-chase. Touches
the `Quantizer` trait + serialization (migration needed).

### 3. NF4 SIMD asymmetric distance
The one remaining scalar quant distance (binary/BF16/SQ2/SQ3 are done). NF4's
dequant is a **non-affine 16-entry normal-float LUT**, so it needs a
`pshufb`/`permute`-based in-register LUT (two `_mm256_permutevar8x32_ps` + blend
for the 16 entries) rather than the affine trick used for SQ. Expected ~3–5× like
its siblings.

### 4. NEON kernels for the new distance paths
Binary/BF16/SQ2/SQ3 distance SIMD currently ships AVX2 + scalar fallback; the
aarch64 (Apple) path is still scalar (no regression, but leaves the M-series on
the table). Port each to NEON (`tbl`, `vsubq`, `vfmaq`).

### 5. PyO3 — finish the FFI modernization
- Release the GIL on `train` (k-means) and `add_batch` (parallel build) — the
  heaviest kernels, currently serialized.
- NF4 batch FFI: `_rust_bridge.encode_nf4_batch` does per-row `row.tolist()`
  crossings (N FFI calls + N·D boxed floats); add a batched `encode_nf4_batch_np`
  Rust entry mirroring `quantize_int8_batch`.
- Return packed numpy arrays from batch search / `decode` instead of
  list-of-tuples.

### 6. Coarse k-means u32 across-K kernel (build-time)
A `assign_argmax_dot` (cosine) kernel returning u32 (n_lists > 256) + parallel
update reduction. ~1.3–2× on IVF/IVF-PQ build time.

### 7. SQ2/SQ3 stored norms (search-time)
`norm_sq` is query-independent; storing it at encode time halves the
per-candidate distance work. Needs a struct field + serialization migration with
a recompute-on-load hook (the generic `QuantHnswIndex` lacks one today).

---

## Method notes (the Konjo discipline applied here)

- Every change was benchmarked before/after on the target hardware; the win (or
  its absence) is recorded in the PR.
- When the in-repo benchmark couldn't exercise the bottleneck (e.g. the 260 MiB
  L3 masking prefetch at n=20k), an isolated microbenchmark at the right scale was
  built to confirm or refute the mechanism.
- Opportunities that didn't pan out were **reverted and documented**, not shipped.
- All shipped PRs keep 216 Rust tests + clippy green and (for Python/PyO3) the
  Python test suite green in a built venv.
