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

### Campaign 3 (AVX-512 encode kernels · allocator · Python hot paths — this sweep)

A fresh full-repo audit (six parallel deep-review passes over the quant kernels,
index code, PyO3 FFI, Python hot paths, and build/dep config) surfaced ~35
ranked opportunities. The harvested wins this sweep — all proved before shipping:

| Change | Win (this host) | PR |
|--------|-----------------|----|
| **PQ AVX-512 nearest-centroid argmin** — 16-wide `assign_argmin_avx512` w/ masked tail; the k-means assignment hot loop was AVX2-only on x86 | kernel **1.42–1.82×**; PQ encode **1.09×** / train **1.06×** end-to-end (mem-bw diluted); recall-neutral | #92 |
| **INT8 AVX-512 normalized encode** — `encode_normalized_avx512`, clone of the bit-identical `encode_avx512_into` pass-2; drives `batch_encode_normalized_into` | kernel **1.34–1.71×** (d=256→1536), bit-identical | #92 |
| **Binary `np.packbits`/`unpackbits`** — replace two 8-iteration Python fancy-index loops | pack **2.1–3.2×**, unpack **3.3–3.9×**, bit-identical | #92 |
| **mimalloc global allocator** (vectro_py + vectro_cli) — sharded per-thread heaps for the GIL-released rayon build/query paths | HNSW build **1.08–1.16×**, concurrent query **1.08–1.10×** | #92 |
| **Reranker batched GEMV** — `_cosine_rerank`/`_rrf_rerank` score all candidates in one `(C,d)@(d,)` matmul instead of a per-candidate norm+dot loop | **1.54–2.69×**, identical ranking | #92 |

The two AVX-512 encode kernels exploit the same host AVX-512F/VNNI the prior
campaign found *slower* for f32 distance — but encode (convert/narrow/store-bound)
and argmin (loop+update-overhead-bound) are different regimes where 512-bit width
and mask registers win. Negative result this sweep: quant-HNSW `apply_center`→`Cow`
(see `PERF_FINDINGS.md`) — unmeasurable against ±25% host variance, reverted.

### Campaign 2 (AVX-512 distance kernels — earlier sweep)

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

## Campaign 3 audit — remaining ranked opportunities (not yet shipped)

The six-pass audit surfaced these beyond the wins above. Ordered by
expected impact × confidence; each needs the usual prove-before-ship pass.

**Index / search path**
1. ✅ **Shipped** (predates this note; re-verified 2026-07-01). **IVF-Flat flat
   vector store** — `IvfIndex.store` is now a flat `Vec<f32>` (`ivf.rs:170`),
   and `search_with_probe` already software-prefetches the posting-list scan
   (`PREFETCH_AHEAD = 2`, `ivf.rs:435`).
2. ✅ **Shipped** (predates this note; re-verified 2026-07-01). **IVF-PQ4
   batched coarse GEMM + `search_batch_flat`** — `IvfPq4Index::search_batch_flat`
   (`ivf_pq4.rs:155`) already tiles the GEMM coarse scan across queries,
   mirroring `IvfPqIndex`. Note: routing the *single-query* path through the
   same batched call (q=1 per call) was tried and measured **~3.4× slower**
   (rayon/GEMM fixed overhead dominates at q=1, not amortized) — the existing
   per-query scalar coarse loop in `search_with_probe` is correctly kept as-is.
3. ✅ **Shipped** (predates this note; re-verified 2026-07-01). **BM25 inverted
   index** — `bm25.rs` already has a `term → postings` map (`postings` field)
   and `top_k` uses `select_nth_unstable_by`.
4. ✅ **Shipped** (predates this note; re-verified 2026-07-01). **GIL release
   on `train`** — `vectro_py/src/lib.rs`'s IVF/IVF-PQ/PQ `train` bindings all
   call `py.allow_threads`. (`add_batch` GIL release not re-checked.)
5. ✅ **Shipped** (predates this note; re-verified 2026-07-01). **NF4 SIMD
   nibble-quantize encode** — `nf4.rs` already has AVX2 (`avx2_abs_max`) and
   NEON (`encode_with_absmax_neon`) encode paths.
6. ❌ **Tried, reverted — see `PERF_FINDINGS.md`** (2026-07-01). Replacing
   `search_layer_locked`'s per-expansion `NeighborList` clone with a reused
   thread-local scratch buffer measured **within noise** (~5.0s both ways,
   ±10–15% run-to-run variance) even in the `M0 > 32` heap-spill regime
   designed to favor it, on this 4-core host.

**FFI / Python marshalling**
7. ✅ **Shipped** (predates this note; re-verified 2026-07-02). **`search_batch`
   → `search_batch_arrays`** — `hnsw_rust.py::search_batch_arrays` exists
   alongside `search_batch`, backed by the native `search_batch_arrays_np`
   (`vectro_py/src/lib.rs:776`).
8. ✅ **Shipped** (predates this note; re-verified 2026-07-02). **Vectorize
   `search_batch` query normalization** — `hnsw_api.py` already batches via
   `normalize_rows` (`hnsw_rust.py:39`) instead of a per-row Python loop.
9. **`get_vectors` row-memcpy** — still open. `PyEmbeddingDataset::get_vectors`
   (`vectro_py/src/lib.rs:95`) fills via N·D per-element 2-D `array[[i,j]] =
   value` writes; replace with N contiguous row copies (`copy_from_slice`).
   3–8×.
10. ✅ **Shipped** (2026-07-02, this session). **`pq_encode_batch` borrows
    contiguous input** — now uses the `quantize_int8_batch` zero-copy idiom
    for `vectors` (`.as_slice()` when C-contiguous) and releases the GIL
    around `pq_encode_into`. Measured (Rust-level copy microbenchmark, no
    `pytest`/`numpy` available in this container to drive an end-to-end
    Python benchmark): the avoided copy was ~220–230ms of a ~935ms call at
    `[200,000, 768]`, M=96, K=256 (~20–25% of total call time). `centroids`
    is left as a direct copy (small, and `PQCodebook` must own it regardless).
    `pq_train_batch`'s per-row `Vec<Vec<f32>>` ownership is structurally
    required by `train_pq_codebook`'s `&[Vec<f32>]` signature — not a
    same-shaped fix; would need a broader `quant/pq.rs` API change to accept
    borrowed rows, out of scope here.
11. **`reconstruct_batch` contiguous store** — still open (not re-verified
    this session). `BatchQuantizationResult` keeps codes as a Python list of
    rows then `np.stack`s them back; store the `[N,D]` matrix.

**Build / methodology**
12. **Opt-in `target-cpu=native` from-source build** — shipped wheels stay v3
    (portable); document a VNNI/AVX-512 native build for the autovectorizer-bound
    shared loops (coarse GEMM, bf16 widen). Exclude f32 distance (proven slower).
13. **Benchmark statistics** — paper harness headlines best-of-3/1-warmup; raise to
    warmup≥5, reps≥20, **p50 headline + p95/p99/stddev** (the regression gate needs
    real percentiles). Add concurrent-query + at-scale-build benches (probes for the
    allocator + HT-oversubscription effects) and a real/anisotropic dataset for
    HNSW recall/QPS claims (current sinusoidal data is a best case).

## Roadmap — remaining high-value opportunities (campaign 2 carry-over)

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

