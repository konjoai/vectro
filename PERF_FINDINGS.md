# vectro Performance Findings — negative & inconclusive results

Companion to `OPTIMIZATION_OPPORTUNITIES.md`. Records optimizations that were
implemented and benchmarked but **did not pan out**, so they are not re-attempted
without new information (e.g. different hardware). Every entry has measured
numbers and a root-cause for why the expected win didn't materialize.

Benchmark host for these results: x86_64, 4 physical cores, AVX2 + AVX-512F + FMA,
glibc malloc, `target-cpu=x86-64-v3`, `--release` (fat LTO, codegen-units=1).

---

## ❌ IVF-PQ4 HNSW coarse quantiser — recall-neutral but no batch QPS gain vs the GEMM coarse (reverted)

**Opportunity:** at fine partitioning (large `n_lists` → small posting lists →
cheap PQ4 scan) the coarse step — finding the `n_probe` nearest cells — was
expected to dominate, since a brute-force scan over every centroid scales with
`n_lists`. Idea: an HNSW over the coarse centroids so the nearest cells are
found in ~`O(log n_lists)` hops, decoupling coarse cost from `n_lists`.

**Implementation:** added `coarse_hnsw: Option<HnswIndex>` to `IvfPq4Index`,
built over the unit-norm centroids; single-query and per-query-batch coarse
probe routed through the graph walk; `coarse_ef = max(2·n_probe, 64)`.

**Benchmark** (200k × d=768, `n_lists=4096`, batch of 2000, **clustered** data so
recall is meaningful — `rand_unit` at d=768 is near-orthogonal and gives ~0
recall for *any* method):

| n_probe | GEMM coarse | HNSW coarse | recall@10 (GEMM / HNSW) |
|---|---|---|---|
| 32 | 20 859 qps | 20 149 qps | 0.0200 / 0.0200 |
| 48 | 20 214 qps | 14 990 qps | 0.0170 / 0.0175 |
| 64 | 15 639 qps | 9 873 qps | 0.0125 / 0.0120 |

Recall is **identical** (HNSW coarse is recall-neutral), but the **batched GEMM
coarse is as-fast-or-faster** at every `n_probe` (tied at 32, 1.3–1.6× ahead at
48/64).

**Why it didn't work:** the batch coarse GEMM tiles 32 queries and **reuses the
centroid matrix across the whole tile** (cache-resident, compute-bound,
matrixmultiply-blocked). The HNSW walk is **per-query** with random access to
centroids and no cross-query reuse, so it can't beat the amortised GEMM for the
*batch* path. (An earlier `rand_unit` run showed HNSW "winning" — an artifact:
on near-orthogonal data the degenerate graph terminates early returning garbage,
so it looked fast.) The coarse HNSW also costs ~`n_lists·d·4` bytes (the
`HnswIndex` duplicates the centroids — 12.6 MB at `n_lists=4096`).

**Resolution:** reverted. The batch QPS goal was already met by the batched GEMM
coarse alone (PR #94): at `n_lists=4096` it reaches **20 859 qps vs faiss-IVF-PQ
11 828 (1.76×)**, recall-neutral. The coarse HNSW added no batch benefit and a
real memory cost. **When to revisit:** single-query *latency* (not batch
throughput) — there the per-query GEMM has no amortisation and a graph walk
would beat the serial O(`n_lists`) scan; worth it only if single-query serving
becomes the bottleneck and the +12.6 MB is acceptable.

## ❌ Quant-HNSW `apply_center` → `Cow` (drop the no-center query copy) — unmeasurable here (reverted)

**Opportunity (campaign 3):** `QuantHnswIndex::search` does
`apply_center(&normalize(query))`. For every non-binary quant mode `center` is
`None`, yet `apply_center` still did `normalized.to_vec()` — so each query
allocated the d-length buffer **twice** (once in `normalize`, once in the
pointless copy). Return `Cow<[f32]>` and borrow when there's no center →
one alloc/query. Also dedup `search_rerank`'s double `normalize(query)` by
splitting out a `search_normalized` core. Estimated 3–8%.

**Benchmark** (Int8 quant-HNSW, n=50k × d=768, m=16, ef=64, k=10, best-of-5):

| | qps |
|---|-----|
| Cow (borrow, 1 alloc) | 11 553 / 14 581 (two runs) |
| always-copy baseline | 11 787 |

The two `Cow` runs span **11.5k–14.6k qps on identical code** — run-to-run
variance on this shared 4-core host is ±25%, dwarfing any sub-5% effect. The
borrow-vs-copy delta sits entirely inside that noise (and the first sample even
landed slightly *below* baseline).

**Why it didn't work:** same root cause as the reverted scratch-heaps (#74) and
`dequantize_int8_batch` findings — a single small per-query allocation is
serviced from the allocator's per-thread cache (now mimalloc's sharded heap)
essentially for free, and is orders of magnitude below the hundreds of AVX2/512
distance evals each query runs. The d-length copy is one streaming memcpy that
the same cache line traffic would touch anyway.

**Resolution:** reverted to keep the diff honest (the change was correct and
idiomatic, just not a *measured* win). The mechanism would matter only where
allocation dominates — many-core hosts with allocator contention, or a tail-
latency (p99) metric, neither demonstrable here. Revisit with a low-variance
host and p99.

---

## ❌ `dequantize_int8_batch` uninit-output + GIL release — single-thread neutral (reverted)

**Opportunity:** the batch INT8 decode allocates `vec![0.0f32; n*d]` (serial
zero-init), fills it, then `into_pyarray` copies it again into Python-owned
memory — two passes + a zero-init, all under the GIL. Rewrite to decode straight
into an **uninitialised** `PyArray2` under `py.allow_threads`, mirroring
`quantize_int8_batch`. Estimated ~2×.

**Benchmark** (built extension, single thread, 50k×768 i8 decode, repeated runs):

| | time |
|---|------|
| current (zero-init Vec + `into_pyarray` copy) | 49 ms |
| uninit PyArray + `allow_threads` | 50–57 ms |

Within run-to-run noise — **no single-thread win** (d=384 likewise flat).

**Why it didn't work:** the decode is memory-bandwidth-bound. Removing the
zero-init memset and the `into_pyarray` copy is offset by **first-touch page
faults** on the freshly-allocated numpy buffer taken during the rayon-parallel
write (the old `vec![0.0; n*d]` pre-faults all pages serially up front, then the
`into_pyarray` copy is a fast sequential memcpy). Net wash.

**Resolution:** reverted. The GIL-release half *would* help concurrent decode
across threads, but that benefit can't be cleanly demonstrated on this 4-core
host and single-thread is neutral, so the change wasn't shipped (no machinery
without a measured payoff). Revisit with a many-thread concurrent-decode metric.

---

## ❌ SQ2 / SQ3 stored norms — no measurable win (not shipped)

**Opportunity:** `sq2_dot_norm` / `sq3_dot_norm` compute the query-independent
`norm_sq = Σ dv²` on every candidate. Store `norm` at encode time and run a
dot-only kernel (drop one FMA chain + one horizontal reduction per candidate).
Estimated ~30–45%.

**Benchmark** (isolated microbench, SQ2 AVX2 kernel, 4096 codes/iter, best of
repeated runs):

| d | dot+norm | dot-only | speedup |
|---|----------|----------|---------|
| 96 | 15.8 ns | 15.9 ns | 1.00× |
| 128 | 21.1 ns | 22.0 ns | 0.98× |
| 256 | 39.6 ns | 39.7 ns | 1.00× |
| 768 | 121 ns | 121 ns | 1.00× |
| 1024 | 161 ns | 160 ns | 1.00× |

Flat — within run-to-run noise at every dim.

**Why it didn't work:** the kernel is bound by the **code-unpack** path
(`_mm256_srlv_epi32` + mask + `cvtepi32_ps` + the affine `(2·code−3)·¼·scale`
sequence), not by FMA throughput. The norm accumulator `fmadd(dv, dv, nrm)` runs
on the second FMA port in parallel with the dot `fmadd(dv, q, dot)` — there is
spare FMA issue width, so the norm is effectively free and removing it saves
nothing. SQ3's unpack is heavier still, so the same conclusion holds. Avoided
the serde-migration cost (an additive `norm` field) for zero gain.

**When to revisit:** only worthwhile if the unpack itself is first made cheaper
(e.g. a `pshufb`-based 2-bit→f32 LUT replacing the `srlv`+affine chain); then the
norm FMA would become a relatively larger share and stored norms might pay off.

---

## ❌ Shared f32 `dot_f32` / `l2_sq` micro-tuning — no measurable win (not shipped)

**Opportunity:** the AVX2 f32 kernels (`index/simd.rs`) use `_mm_hadd_ps` ×2 for
the final horizontal reduction and 4 `f32x8` accumulators. Try the
`movehl`/`shuffle` add ladder instead of `hadd`, and 6 / 8 accumulators (the
NEON sibling uses 8).

**Benchmark** (isolated microbench, 4096 vector pairs, two runs):

| d | 4·hadd | 4·shuf | 6·shuf | 8·shuf |
|---|--------|--------|--------|--------|
| 128 | 3.49 G/s | 3.49 G/s | 3.41 G/s | 3.37 G/s |
| 768 | 3.08–3.41 G/s | 3.09–3.39 G/s | 3.10–3.40 G/s | 3.11–3.38 G/s |

All variants within ±2–3% run-to-run noise; run 2 reversed run 1's ordering.

**Why it didn't work:** the kernel is **load-port-bound** (2 loads per FMA,
2 load ports on this Xeon → ~1 FMA/cycle ceiling), so neither the reduction
style (amortised over `d/8` FMAs) nor more accumulators move the needle. Confirms
the prior campaign's "AVX2 4-accumulator is right here" conclusion.

---

## ❌ HNSW per-query thread-local scratch heaps — no measurable win (reverted)

**Opportunity (was P1 #1):** `search_layer_impl` constructs two `BinaryHeap`s per
call. Move them into the thread-local scratch (alongside the visited epoch array)
and `clear()`+reuse instead of allocating per query — eliminate ~5–8 heap
allocations per query.

**Implementation:** added a `Scratch { visited, cands, window }` thread-local with
a `with_scratch(n, ef, …)` entry + a `parts()` disjoint-borrow accessor; migrated
all four `search_layer_impl` / `*_locked` paths in `hnsw.rs` and `quant_hnsw.rs`;
replaced `into_sorted_vec()` with drain+reverse on the borrowed heap. 217 tests
green, clippy clean.

**Benchmark** (HnswIndex, n=20 000, d=128, m=16, ef=100, k=10, 100 000 queries):

| | main (per-query heaps) | branch (scratch reuse) |
|---|---|---|
| single-thread | 34.7k qps | 34.3k qps |
| concurrent (4 cores) | 132k qps | 132.5k qps |

Within ±1% run-to-run noise — **no improvement**, single-thread or concurrent.
(The criterion `hnsw_search` micro-bench at n=2000/d=64 was likewise flat.)

**Why it didn't work:**
1. The prior optimization had already captured the real cost — the heaps were
   already `BinaryHeap::with_capacity(ef + 1)` (one allocation each, no
   growth-realloc churn), and the visited set was already a reused thread-local
   epoch array. Only 2 small, short-lived alloc/frees per query remained.
2. glibc's per-thread cache (tcache) services those small allocations from a
   thread-local free-list — effectively free, and contention-free even under the
   4-core concurrent run.
3. At d=128 each query does hundreds of AVX2 distance evaluations; heap
   alloc/free is orders of magnitude below that in wall-clock.

**Caveat / when to revisit:** on high-core-count servers (32+) where glibc malloc
arena contention is real, eliminating per-query allocations could measurably cut
tail latency. Not reproducible on a 4-core host, so reverted to keep the code
elegant (no machinery without a measured payoff). Revisit with many-core hardware
and a tail-latency (p99) metric rather than mean throughput.

---

## ❌ AVX-512 f32 distance kernels — slower than AVX2 on this CPU (removed)

**Opportunity:** the shared distance kernels (`index/simd.rs` `dot_f32` / `l2_sq`)
use AVX2 (256-bit). This host has AVX-512F, so a 512-bit kernel "should" be faster
by doubling the lane width.

**Implementation:** added `dot_f32_avx512` / `l2_sq_avx512` (four `f32x16`
accumulators, 64 lanes/iter, `_mm512_reduce_add_ps`) gated behind
`is_x86_feature_detected!("avx512f")`, preferred over the AVX2 path.

**Benchmark** (isolated microbench, best of repeated runs, this 4-core Xeon):

| d | AVX2 | AVX-512 | 512 / 256 |
|---|------|---------|-----------|
| 64 | 220 M/s | 168 M/s | 0.76× |
| 96 | 165 M/s | 177 M/s | 1.07× |
| 128 | 143 M/s | 125 M/s | 0.87× |
| 256 | 80 M/s | 92 M/s | 1.16× |
| 384 | 53 M/s | 45 M/s | 0.85× |
| 768 | 27 M/s | 25 M/s | 0.92× |
| 1024 | 20 M/s | 18 M/s | 0.91× |

AVX-512 is **slower at almost every dimension**, including the d=768 embedding
regime (0.92×). Only d=96 / d=256 saw a marginal win.

**Why it didn't work:** this class of Xeon implements AVX-512 on double-pumped
256-bit execution units, so 512-bit ops carry no throughput advantage, while the
wider `_mm512_reduce_add_ps` horizontal reduction and AVX-512 frequency licensing
cost more. The net is a regression.

**Resolution:** removed both AVX-512 kernels; AVX2 is the fastest portable x86
width here. An in-code comment in `index/simd.rs` (shipped in PR #73) records the
measurement so the 512-bit path is not re-added on spec. This also explained an
earlier end-to-end "IVF regressed" blip that was in fact bench noise, not a real
SimSIMD-AVX512 edge — confirmed by the high run-to-run variance on this shared host.

**When to revisit:** native-512-bit microarchitectures (Intel server P-cores with
a full 512-bit FMA — Skylake-SP / Ice Lake-SP / Sapphire Rapids) may flip the
result. Re-benchmark before re-enabling, and gate on a CPU-family check, not just
the `avx512f` feature bit.
