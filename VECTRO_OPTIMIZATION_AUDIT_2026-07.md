# VECTRO Optimization Audit — July 2026

> Full read of the Rust workspace (~21k lines), cross-checked against
> `OPTIMIZATION_OPPORTUNITIES.md`, `PERF_FINDINGS.md`, and current ANN research
> (2024–2026). Everything already shipped, tried-and-reverted, or already on
> the in-repo roadmap is excluded or marked as such. The goal stated for this
> audit: beat FAISS and HNSW across the board, not just on Apple Silicon
> niches.

One structural observation up front. The three completed kernel campaigns
(see `OPTIMIZATION_OPPORTUNITIES.md`) have harvested most of the SIMD-level
wins on the current architecture. The remaining x86 kernel headroom is
single-digit percentages against a ±10–15% host noise floor (per
`PERF_FINDINGS.md`). FAISS will not be beaten across the board with more
kernel tuning on the current index designs. The board gets flipped at the
algorithm layer: the 2024–2026 research generation (RaBitQ,
quantization-graph fusion, distance early-termination, graph reordering)
beats hnswlib by 3–17× at matched recall, and FAISS has not absorbed most of
it yet. That is the opening.

Ranking key:
- **Impact**: expected effect on the recall/QPS Pareto frontier or memory, at
  matched recall. P = Pareto-shifting (changes what is achievable), K = kernel
  (speeds up existing curve), S = systems (build time, load time, memory,
  tails).
- **Difficulty**: S / M / L / XL engineering effort including serialization
  migration and tests.
- **Confidence**: how likely the win survives a kill-test, based on published
  results and what the codebase already has.

---

## Tier 1 — Algorithm layer. This is where FAISS loses.

### 1.1 RaBitQ + extended RaBitQ quantizer  [Impact: P, very high · Difficulty: M · Confidence: high]

The single highest-leverage item in this audit. RaBitQ quantizes each vector
to D bits (1 bit/dim) via a random rotation plus sign codes, with two stored
scalars per vector, and gives an unbiased distance estimator with a sharp
theoretical error bound. PQ with the fast SIMD implementation can incur over
50% average relative error on estimated distances on some real datasets
(MSong), collapsing recall below 60% even with re-ranking — exactly the
unpredictable-failure mode a theoretical error bound prevents. In the
original paper's IVF experiments, RaBitQ consistently outperforms both
OPQ-with-fast-scan and HNSW on all tested datasets, and no single OPQ
re-ranking parameter works well across datasets.

Why it fits VECTRO specifically:
- It deletes the OPQ gap. The known weakness (32× vs FAISS's 64× at PQ, OPQ
  unshipped) stops mattering: RaBitQ at 1 bit/dim is 32× on f32 with better
  accuracy than OPQ at the same budget, and extended RaBitQ (2–8 bits/dim,
  2024 follow-up paper) covers the whole compression sweep with one method.
  Skipping OPQ and leapfrogging to RaBitQ is less work than implementing OPQ.
- The estimator inner loop is a bitwise AND/popcount plus one FMA, which maps
  onto kernels VECTRO already has (binary popcount paths, PQ4 fast-scan
  register-shuffle machinery in `index/pq4.rs`).
- The random rotation is a one-time orthonormal matrix multiply at encode
  time. On the M3 that routes through the existing Accelerate/AMX path.
- Industry has already voted: it is the default quantizer in new engines
  (VectorChord, Progress/Nuclia, and others) precisely because it is more
  accurate and comes with theoretical guarantees, delivering better
  recall-latency trade-offs than the vanilla PQ used in original
  implementations.

Kill-test: IVF-RaBitQ vs shipped IVF-PQ4 and vs faiss IVF-PQ on SIFT1M and
GIST1M at recall@10 = 0.90/0.95. Gate: Pareto-dominant QPS at both recall
points, 30-run paired Wilcoxon.

### 1.2 Quantization-graph fusion (SymphonyQG-style index)  [Impact: P, highest ceiling · Difficulty: L–XL · Confidence: high]

The current quant-HNSW stores one code per node and chases pointers per
expansion. The SIGMOD'25 state of the art inverts the layout: each node
stores its neighbors' quantization codes contiguously, so one beam expansion
becomes one sequential FastScan sweep over 32 packed codes, with no random
access and no explicit rerank stage. The approach replicates and stores the
quantization codes of a vertex's neighbors compactly so they can be accessed
sequentially, and uses the SIMD FastScan implementation to estimate distances
in batch to guide the search. The results are the largest published gains
over the incumbent: at 95% recall, 1.5–4.5× QPS versus the most competitive
baselines and 3.5–17× versus hnswlib across all tested datasets.

VECTRO's head start is real: the AVX2 `_mm256_shuffle_epi8` and NEON
`vqtbl1q_u8` FastScan kernels already exist in `pq4.rs` with the u8 LUT
quantization. What's missing is the graph layout (neighbor-code replication),
the joint graph refinement (degree padded to multiples of 32 so no SIMD
lanes are wasted), and pairing it with RaBitQ codes from 1.1 instead of PQ4
— the paper's own choice, since its memory footprint is smaller than NGT-QG
partly because RaBitQ codes are shorter than PQ codes.

Honest cost: memory footprint is higher than plain HNSW because multiple
quantization codes are stored per vertex, which is likely unavoidable when
the target is eliminating random memory access. At 1 bit/dim that
replication is cheap (d=768 → 96 B/code × degree). This item plus 1.1 is the
credible "beat both FAISS and HNSW across the board" play, on x86 as well as
Apple Silicon, because the win is algorithmic, not ISA-specific.

Kill-test: QPS at recall@10 = 0.95 vs hnswlib, faiss-HNSW, and VECTRO's own
quant-HNSW on SIFT1M/GIST1M/real embedding data. Gate: ≥2× vs hnswlib or no
merge.

### 1.3 Distance early termination (ADSampling / incremental distance)  [Impact: P/K, high · Difficulty: M · Confidence: medium-high]

Most distance computations during graph search are wasted on candidates that
will be rejected. After a one-time random rotation of the dataset, a partial
prefix of the dimensions gives an unbiased distance estimate with known
variance, so each candidate evaluation can stop early with a hypothesis test
once the partial distance already exceeds the current pruning threshold.
Published gains are 1.5–3× QPS at equal recall on HNSW and IVF, and it
composes with everything above (the RaBitQ rotation can be shared). Fits
VECTRO's statistical-rigor ethos unusually well: the early-exit condition is
literally a significance test.

Kill-test: fp32 HNSW and IVF QPS at matched recall, on real data with the
percentile harness. Watch for branch-misprediction overhead at small d.

### 1.4 Vamana / α-pruned graph construction (DiskANN-style)  [Impact: P, medium · Difficulty: M · Confidence: medium-high]

`select_heuristic` in `hnsw.rs` is the classic HNSW diversity heuristic. The
α-RNG pruning rule (keep a candidate only if no already-kept neighbor is α×
closer to it) produces flatter degree distributions and fewer hops at equal
recall, and several results from 2024–2025 show the HNSW hierarchy itself
adds nothing at ≤10M scale on high-dimensional data (a flat single-layer
graph with a good entry point matches it). Two concrete moves:
- Add α as a build parameter to the existing heuristic (small diff, easily
  A/B-able).
- Offer a single-layer build mode: deletes the upper-layer memory, simplifies
  the locked build path, and removes the layer-descent branch from search.

Kill-test: recall/QPS Pareto and build time vs current heuristic, plus
memory.

### 1.5 IVF upgrades: SOAR spilled assignments + anisotropic loss (ScaNN)  [Impact: P, medium · Difficulty: M · Confidence: medium]

Two Google techniques that FAISS also lacks. SOAR assigns each vector to a
second, deliberately orthogonal-residual list, so a missed primary probe is
recovered cheaply, improving recall at fixed n_probe for ~1 extra assignment
of memory. Anisotropic quantization weights the training loss toward the
component of the residual parallel to the vector, which is what matters for
inner-product ranking; it is why ScaNN wins MIPS benchmarks. Both slot into
the existing IVF/IVF-PQ4 machinery without touching search-side kernels
much.

Kill-test: recall@10 at fixed n_probe on an embedding (MIPS) dataset vs
current IVF-PQ4 and faiss-IVF-PQ.

---

## Tier 2 — Apple Silicon hardware layer. The moat items.

### 2.1 NEON `sdot` integer INT8 kernel  [Impact: K, high on M-series · Difficulty: S–M · Confidence: very high]

The single clearest gap found in the source. `quant/int8.rs` has the
AVX-512-VNNI `vpdpbusd` pure-integer path with the `Quantizer::Prepared`
once-per-search query quantization on x86, but the aarch64 path
(`dot_i8_f32_neon`) still widens i8 to f32 and does FMA. Every M-series chip
has FEAT_DotProd: `vdotq_s32` does 16 i8×i8 MACs per instruction with i32
accumulation. Mirror the VNNI design exactly (quantize the query once per
search, integer-accumulate, one final scale multiply, preserving the
FP32-accumulation exactness rule since integer accumulation is exact). The
x86 version measured 1.52× QPS end-to-end with a 1.6–2.7× kernel win; the
NEON gap is larger because the current path pays the widen on every lane, so
2–4× kernel is the reasonable expectation. This is the highest-confidence,
lowest-effort item in the entire audit, and INT8 is the flagship mode.

### 2.2 `i8mm` (`smmla`) batched INT8 scoring  [Impact: K, medium-high · Difficulty: M · Confidence: high]

M2 and later expose ARMv8.6 i8mm: `smmla` computes a 2×2 i32 tile from two
2×8 i8 operands, doubling `sdot` throughput for anything shaped like a small
GEMM. Targets: IVF coarse assignment, batch search, brute-force scoring, PQ
training assignment. Runtime-detect and fall back to 2.1.

### 2.3 `bfdot`/`bfmmla` for the BF16 path  [Impact: K, medium · Difficulty: S–M · Confidence: high]

The BF16 mode currently widens via `<<16` (the AVX2/AVX-512 trick). ARMv8.6
BF16 extensions on M2+ do the dot in hardware. Halves bandwidth on the
load-bound BF16 kernel, which is exactly the regime where the x86 BF16 widen
win (1.4× kernel) came from.

### 2.4 Extend Accelerate/AMX routing to batch GEMM scoring  [Impact: K, medium-high · Difficulty: M · Confidence: medium-high]

`quant/accelerate.rs` routes d≥256 multiplies through `vDSP_vsmsa`. The
batched coarse GEMM (`search_batch_flat`, the thing that beat faiss-IVF-PQ
1.76× on x86) should route through `cblas_sgemm` on macOS so the AMX
coprocessor does the tile work while the CPU cores run the fine scan. The
in-repo roadmap has the x86 AMX-INT8 note; the Apple analogue is nearer-term
since the dispatch plumbing exists.

### 2.5 Metal GPU batch path  [Impact: P for batch throughput · Difficulty: L · Confidence: medium]

A Metal compute kernel (or MPSMatrixMultiplication) for brute-force and IVF
batch scoring on unified memory: no PCIe copy, the index buffer is shared
zero-copy with the CPU. FAISS's GPU story is CUDA-only, so on Apple hardware
this is a category FAISS cannot enter. High lift, and it doubles as a Squish
narrative tie-in (same Metal skill set, same machine). Batch-only at first;
single-query latency stays CPU.

### 2.6 SME2 kernels when M4 hardware lands  [Impact: K · Difficulty: M · Confidence: n/a yet]

Already wired for dispatch per the README (`todo!()` stubs). Just noting it
belongs on this list so it isn't lost; nothing to do until the hardware
exists in CI.

---

## Tier 3 — Memory layout and systems

### 3.1 CSR flat adjacency for the graph  [Impact: K/S, medium-high · Difficulty: M–L · Confidence: high]

`neighbor_store.rs` is `Vec<Vec<SmallVec<[u32; N]>>>`: three levels of
indirection and a heap object per node per layer, serialized through a
legacy `Vec<Vec<Vec<u64>>>` wire format. For the immutable post-build search
phase, freeze into one contiguous `Vec<u32>` plus an offsets array per
layer. Removes the pointer chase per expansion, enables prefetching the
*next* candidate's neighbor list (the shipped PF=2 vector prefetch pattern,
applied to adjacency), shrinks memory by the per-SmallVec overhead, and
makes 3.2 and 1.2 possible. Keep the current structure as the mutable
build-side representation and freeze on save/first-search.

### 3.2 Graph reordering for cache locality  [Impact: K, medium-high · Difficulty: S–M · Confidence: high]

✅ **Shipped 2026-07-05** — `HnswIndex::reorder_for_locality()`
(`rust/vectro_lib/src/index/hnsw.rs`). Measured single-query **1.24–1.30×**,
batch **1.37–1.49×** QPS (3 independent runs, n=200k, d=768, this host),
recall bit-identical before/after every run. See
`OPTIMIZATION_OPPORTUNITIES.md` Campaign 4 and the CHANGELOG `[Unreleased]`
entry for the full numbers. Not yet wired through PyO3 — the description
below is preserved as originally written for context.

Sneaky-large and almost free. After build, renumber node IDs so that graph
neighbors are numerically adjacent (BFS order from the entry point, or
Gorder / recursive graph bisection for more win). Neighbor expansions then
hit vectors and codes that share cache lines and TLB pages. Published
results on HNSW-class indexes: 1.2–2× QPS, zero recall change, purely an
offline pass plus an ID remap table for the API. Requires 3.1's flat layout
to pay fully. Nothing in the repo does this today (verified: no reordering
pass exists).

### 3.3 Quant-HNSW flat code store  [already roadmap item 2 — merge into 3.1]

Listed in-repo. Do it as the same migration as 3.1 so serialization breaks
once, not twice.

### 3.4 Zero-copy index loading (rkyv or aligned custom format)  [Impact: S, high for cold start · Difficulty: M–L · Confidence: high]

bincode deserialization rebuilds every allocation on load. An
archived/aligned format lets `mmap` map the index and search it directly:
near-instant load, OS page cache shared across processes, and lower
resident memory. This is the same "cold start" story that made Squish's
54× headline. Natural follow-on: indexes larger than RAM degrade gracefully
via page cache (a soft DiskANN). Costs a format migration, so bundle with
3.1/3.3.

### 3.5 Huge pages for large indexes  [Impact: S/K, small-medium · Difficulty: S · Confidence: medium]

`madvise(MADV_HUGEPAGE)` on Linux for the vector/code/adjacency buffers;
superpage hints on macOS. Cuts TLB misses on the random-access probe loop.
Typically 5–15% on >L3 indexes. Cheap to test with the existing >L3 bench.

### 3.6 Rerank cascade formalization  [Impact: K/S, medium · Difficulty: S–M · Confidence: high]

`search_rerank` exists, but the winning production pattern is an explicit
three-tier cascade: traverse on 1-bit (RaBitQ/binary), rerank top ~4k on
INT8, final top-k on fp32 only if fp32 is retained at all. With extended
RaBitQ at ~4 bits the fp32 tier can be dropped entirely, which is where the
memory story (vectors never stored at full precision) starts beating FAISS
defaults, not just matching them.

---

## Tier 4 — FFI and API throughput

Items 7–11 of the in-repo Campaign 3 list plus the "finish the FFI
modernization" roadmap already cover the known PyO3 work (GIL release on
train/add_batch, NF4 batch entry, packed arrays from batch search,
uninit-PyArray decode). Not re-claimed here. Two additions:

### 4.1 Stable C ABI + Node/WASM-SIMD parity  [Impact: S/adoption · Difficulty: M]

A `vectro_ffi` C header makes every benchmark comparison "vs the same
binary" and opens Swift (lopi's macOS app), Go, and Java bindings for
near-zero cost. The existing `wasm.rs` should gain `simd128` kernels for the
dot/L2 paths so the browser demo numbers stop being scalar.

### 4.2 True batched graph traversal  [Impact: P for batch · Difficulty: XL · Confidence: low-medium]

Traversing the graph for Q queries simultaneously and tiling their candidate
distance evaluations into small GEMMs. Research-flavored, high variance; the
in-repo finding that per-query HNSW-coarse lost to batched GEMM coarse is
the cautionary precedent (see `PERF_FINDINGS.md`). Park behind 1.2, which
captures most of the same memory-locality win with far less risk.

---

## Tier 5 — Benchmarks are part of "beating FAISS"

### 5.1 ann-benchmarks submission  [Impact: positioning, highest · Difficulty: M]

The board that matters. Wrapping VECTRO for erikbern/ann-benchmarks (Docker +
Python module) makes every claim third-party-reproducible on standard
datasets and puts the QPS-recall curve next to faiss, hnswlib, ScaNN, and
glass on the same axes. This is also the honest forcing function: it makes
cherry-picking structurally impossible, which is the Konjo way of making a
claim.

### 5.2 Real datasets + recall-matched headlines  [already roadmap item 13 — reinforce]

The in-repo audit already flags sinusoidal data as a best case and the
best-of-3 headline as weak. Add SIFT1M, GIST1M, and a real embedding set
(e.g. Cohere/MS MARCO passages), and make every headline "QPS at recall@10 =
0.95" rather than raw throughput. Every Tier 1 kill-test above assumes this
harness exists, so it is sequenced first.

### 5.3 README version drift  [Difficulty: trivial]

README badge and banner still say 5.0.0 against a 5.2x package. One-line
fix, already a known item; noting it since it undermines the benchmark
credibility this tier is about.

---

## Recommended sequence (impact ÷ difficulty, dependencies respected)

| # | Item | Impact | Difficulty | Confidence |
|---|------|--------|------------|------------|
| 1 | 5.2 Real-data percentile harness | gate for everything | M | — |
| 2 | 2.1 NEON `sdot` INT8 | K-high (M-series flagship) | S–M | very high |
| 3 | 1.1 RaBitQ (+extended) | P-very-high, kills OPQ gap | M | high |
| 4 | 3.1+3.3 CSR adjacency + flat codes (one migration) | K/S-med-high | M–L | high |
| 5 | 3.2 Graph reordering | K-med-high, cheap | S–M | high — ✅ shipped 2026-07-05 |
| 6 | 1.2 Quantization-graph fusion (RaBitQ FastScan graph) | P-highest | L–XL | high |
| 7 | 1.3 ADSampling early termination | P/K-high | M | med-high |
| 8 | 5.1 ann-benchmarks submission | positioning | M | — |
| 9 | 1.4 α-pruning / flat graph option | P-medium | M | med-high |
| 10 | 2.2/2.3 `i8mm` + `bfdot` | K-medium | S–M | high |
| 11 | 3.4 Zero-copy mmap loading | S-high (cold start) | M–L | high |
| 12 | 2.4 AMX batch GEMM routing | K-med-high | M | med-high |
| 13 | 1.5 SOAR + anisotropic IVF | P-medium (MIPS) | M | medium |
| 14 | 2.5 Metal batch path | P (batch, Apple-only) | L | medium |
| 15 | 3.5 Huge pages | S-small | S | medium |
| 16 | 3.6 Rerank cascade | K/S-medium | S–M | high |
| 17 | 4.1 C ABI + WASM SIMD | adoption | M | — |
| 18 | 4.2 Batched traversal | P (batch), risky | XL | low-med |

Items 2+3+5 alone are a plausible "Pareto-dominant vs faiss on M3" release.
Items 3+6+7 are the "beat hnswlib 3×+ at 95% recall on x86 too" release,
which is the across-the-board claim.

## Two honest counterweights

First, the physics: campaigns 1–3 prove the remaining kernel wins on the
current architecture are small and hard to measure. The list above is
deliberately weighted toward Pareto-shifting algorithm work because that is
the only class of change that beats FAISS on x86 at high d with PQ, where it
currently wins. More AVX tuning will not get there; RaBitQ-FastScan-graph
can.

Second, the discipline: this is exactly the shape of an optimization-loop
trap, and VECTRO is parked behind the Squish launch by prior decision.
Nothing here is urgent; RaBitQ and SymphonyQG have been public since 2024
and will still be there after launch. Treat this document as the unpark
plan: when VECTRO resumes, item 1 (the harness) is the mandatory pre-flight,
and every item merges only through the standard gate (30-run paired
Wilcoxon, p<0.05, losses documented in `PERF_FINDINGS.md`).
