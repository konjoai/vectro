# vectro Performance Findings — negative & inconclusive results

Companion to `OPTIMIZATION_OPPORTUNITIES.md`. Records optimizations that were
implemented and benchmarked but **did not pan out**, so they are not re-attempted
without new information (e.g. different hardware). Every entry has measured
numbers and a root-cause for why the expected win didn't materialize.

Benchmark host for these results: x86_64, 4 physical cores, AVX2 + AVX-512F + FMA,
glibc malloc, `target-cpu=x86-64-v3`, `--release` (fat LTO, codegen-units=1).

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
