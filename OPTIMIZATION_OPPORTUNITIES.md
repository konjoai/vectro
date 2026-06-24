# vectro Performance Optimization — Identified Opportunities

A full-repo performance audit (every Rust kernel, the PyO3 FFI layer, and the
Python hot paths) carried out via six parallel deep-review passes. Each item is
located, impact-rated, and given a fix sketch. Status reflects work done on
branch `claude/vectro-performance-optimization-seonf7`.

**Status legend:** ✅ implemented & verified · 🔬 evaluated, rejected with data ·
⬜ identified, not yet done

---

## Measured wins already landed

| Change | Evidence |
|--------|----------|
| AVX2 `l2_sq` kernel (was scalar on x86_64) | **6.3×–11.9× faster** vs scalar in isolated microbench (d=100/128/768) |
| AVX-512 distance kernels | 🔬 **Rejected** — measured 0.76–0.94× vs AVX2 on AVX-512 hardware (double-pumped units + costlier reduce). Kept AVX2. |

---

## P0 — Distance kernels (foundation; every search/train touches these)

- ✅ **`l2_sq` scalar on x86_64** (`index/simd.rs`) — added AVX2+FMA kernel
  (4 accumulators, 32 lanes/iter) mirroring `dot_f32_avx2`. Falls back to
  SimSIMD `sqeuclidean` then scalar. Benefits L2 HNSW search, IVF/PQ k-means,
  PQ ADC-table build. **6–12× on the kernel.**
- ✅ **IVF-Flat `cosine_dist` via SimSIMD per-call dispatch** (`index/ivf.rs:217`)
  — routed through `simd::dot_f32` (AVX2), consistent with IVF-PQ. Removes the
  dispatch indirection the `simd` module exists to avoid.
- ✅ **IVF k-means `l2_sq` scalar duplicate** (`index/ivf.rs:31`) — routed
  through `simd::l2_sq`.
- ✅ **AVX-512 INT8 dot single-accumulator** (`quant/int8.rs`) — rewrote to
  4 independent accumulators (64 elem/iter) breaking the FMA dependency chain.
- 🔬 **AVX-512 f32 kernels** — implemented and benchmarked; **slower** than
  AVX2 on this class of hardware. Removed; documented in-code so nobody re-adds.
- ⬜ **`dot_query` runs feature detection per call** (`quant/int8.rs:76`) — the
  per-candidate INT8 search kernel. `is_x86_feature_detected!` is a cached
  atomic load but still a branch in the hottest loop. Resolve dispatch once into
  a fn-pointer / `OnceLock`. **High** on x86 search QPS.
- ⬜ **`pq.rs` scalar `l2_sq` duplicate** (`quant/pq.rs:240`) — left scalar
  intentionally (sub_dim≈8; dispatch overhead may offset SIMD). DRY-gate flag;
  revisit with the LUT-reformulation below.

## P1 — HNSW (graph search hot path)

- ⬜ **Per-query/per-layer heap allocation** (`index/hnsw.rs:306`,
  `quant_hnsw.rs:290`) — two fresh `BinaryHeap`s per `search_layer` call (~5–8
  allocs/query); `quant_hnsw` doesn't even pre-size them. Move into thread-local
  scratch (like the visited epoch array already is), `clear()` not realloc.
  **Highest-leverage HNSW QPS win, low risk.**
- ⬜ **Greedy-descent allocates a full sorted result Vec to use one element**
  (`hnsw.rs:370`) — add an ef=1 scalar greedy helper (no heap, no result vec).
- ⬜ **`curr_ep = vec![...]` per descent step** (`hnsw.rs:474+`) — use a fixed
  `[usize;1]` / reused buffer.
- ⬜ **`quant_hnsw` has no prefetch + array-of-structs code storage**
  (`quant_hnsw.rs:76,314`) — unlike `hnsw.rs` (PF=2 pipeline + flat buffer),
  every neighbor probe is a cold-miss pointer-chase. Flat `Vec<u8>` code store +
  prefetch. **This is exactly where vectro should beat FAISS** (FAISS HNSW is
  f32-only). High impact, larger lift (touches `Quantizer` trait + serialization).
- ⬜ **Cache `worst` scalar in the beam loop** (`hnsw.rs:356`) — avoid
  re-`peek()` + closure per neighbor.
- ⬜ **`quant_hnsw` upper layers still `Vec<Vec<NeighborList>>`** (`quant_hnsw.rs:80`)
  — migrate to the flat `Graph` store `hnsw.rs` already uses (memory + locality).
- ⬜ **`dist`/`select_heuristic` use scalar L2 on x86** (`hnsw.rs:154`) — now
  fixed at the kernel level by the AVX2 `l2_sq`; could further route through the
  cached-norm dot reformulation.
- ⬜ **`load` slurps whole file** (`quant_hnsw.rs:904`) vs streaming in `hnsw.rs`.

## P1 — IVF / IVF-PQ / PQ

- ⬜ **ADC scan inner loop fully scalar** (`ivf_pq.rs:368`) — *the* IVF-PQ hot
  loop: scalar gather + scalar add, single-threaded, full candidate Vec
  materialized. Adopt FAISS-style **PQ4 (4-bit) interleaved layout + in-register
  `pshufb`/`tbl` lookup** (16–32 codes/instr) and/or rayon over posting lists +
  bounded heap. **Single biggest lever to beat FAISS IVF-PQ (3–5×).** Large lift.
- ⬜ **ADC distance table built with scalar `l2_sq` per (subspace,centroid)**
  (`pq.rs:550`) — M·K scalar reductions per query (24,576 for PQ-96). Reuse the
  transposed-LUT `‖q‖²+‖c‖²−2qc` trick (already used for assignment) + precompute
  `‖c‖²` at train time. **High.**
- ✅ **Single-query coarse scan full sort** (`ivf_pq.rs:620`) — replaced
  O(n_lists log n_lists) sort with `select_nth_unstable` partial selection.
- ⬜ **Coarse k-means uses scalar/per-centroid `cosine_dist`** (`ivf_pq.rs:131`,
  `ivf.rs:89`) — dominant build cost; reuse the SIMD-across-K `assign_nearest`
  kernel that already exists in `pq.rs`. **High** (build time).
- ⬜ **k-means update step serial scatter-add** (`ivf_pq.rs:149`) — parallel
  reduction with per-thread partials (Amdahl bottleneck once assignment is fast).
- ⬜ **Per-query allocations** (`ivf_pq.rs:356,378`) — query-norm Vec, candidate
  Vec per query; thread scratch + bounded heap.
- ⬜ **`pq_encode`/`encode_one` ignore the fast LUT kernel** (`pq.rs:469`) — the
  `Vec<Vec<f32>>` encode path (used by RQ training) is scalar argmin while the
  flat path is SIMD. Route through `pq_encode_into`. **Med-High** (RQ build).
- ⬜ **RQ nested `Vec<Vec<Vec<u8>>>` double-flatten** (`rq.rs:96`) — write
  directly into a flat buffer via `pq_encode_into`.
- ⬜ **`recall_at_k` O(k²)** (`ivf_pq.rs:499`) — use a HashSet like `ivf.rs`.
- ✅ (kernel-level) **IVF-Flat `cosine_dist`/`l2_sq`** — see P0.

## P1 — Other quantizers (NF4 / Binary / BF16 / SQ2 / SQ3)

- ✅ **Binary encode dead per-element f64 division** (`quant/binary.rs:46`) —
  dividing by a strictly-positive norm never changes a sign; replaced with a
  direct `x > 0.0` test. Removes d divides + d f64 casts per vector.
- ✅ **Binary `cosine_dist_to_query` branchy scalar loop** (`binary.rs:70`) —
  rewrote byte-major + branchless (`sign = 2*bit−1`); removes per-element
  `i/8`,`i%8` and the data-dependent branch blocking vectorization.
- ✅ **`hamming_search` full O(n log n) sort** (`binary.rs:117`) — partial
  `select_nth_unstable_by_key` then sort the top-k prefix.
- ⬜ **SQ2/SQ3 distance+decode fully scalar, no stored norm** (`sq2.rs:64`,
  `sq3.rs:77`) — precompute a 4/8-entry per-vector LUT, **store `norm_sq` at
  encode time** (halves per-candidate work), SIMD-unpack SQ2. SQ3 byte-straddle
  branch needs a SIMD-friendly repack. **High** for the aggressive modes.
- ⬜ **NF4 dequant scalar nibble+LUT** (`nf4.rs:160`) — 16-entry codebook is a
  textbook `pshufb`/`vtbl` SIMD-LUT target (decode + cosine_dist).
- ⬜ **BF16 `cosine_dist_to_query` scalar `to_f32`** (`bf16.rs:45`) — widen via
  `(bits as u32)<<16` vectorized (exact, bit-identical).
- ⬜ **`cosine_int8` recomputes encoded norm every call** (`int8.rs:776`) —
  store `enc_norm` at encode time; reuse SIMD `dot_query`.

## P2 — PyO3 FFI boundary

- ⬜ **GIL held during heavy compute** (`vectro_py/src/lib.rs`, train/add_batch/
  single-search across all indices) — wrap in `py.allow_threads`. k-means
  `train` is the heaviest kernel and currently serializes all Python threads.
  **High** for multi-threaded callers.
- ⬜ **`Vec<Vec<f32>>` args → N+1 per-row heap allocs** — add `*_np`
  zero-copy variants for `PyNf4Encoder`, `PyBinaryEncoder`, `PyPQCodebook`
  (pattern proven in the INT8 batch APIs).
- ⬜ **Batch search returns `Vec<Vec<(usize,f32)>>`** — Q·k Python tuple objects
  defeat the `allow_threads` win; return packed `(ids, dists)` numpy arrays.
- ⬜ **`dequantize_int8_batch` / `pq_encode_batch` alloc-then-copy + GIL held** —
  write into uninitialized `PyArray` via `as_slice_mut` under `allow_threads`.
- ⬜ **`__version__ = "4.10.0"`** in lib.rs — stale vs v5.0.2/v8.0.0 (checklist miss).

## P2 — Python hot paths

- ⬜ **NF4 `encode_nf4_batch` per-row FFI with `row.tolist()`** (`_rust_bridge.py:181`)
  — N FFI crossings + N·D boxed Python floats. Add a batch Rust entry. **Biggest
  Python win.**
- ⬜ **`search_batch` per-query normalize list-comp** (`hnsw_api.py:752`) — use
  vectorized `normalize_rows` (one `np.linalg.norm(axis=1)` pass).
- ⬜ **`search_batch` nested Python double-loop unpack** (`hnsw_api.py:755`) —
  add `search_batch_arrays_np` returning packed arrays.
- ⬜ **Unconditional `astype(np.float32)` full copy** (`vectro.py:289`) — use
  `np.asarray` (no-op when already f32). Doubles peak memory on every `compress`.
- ⬜ **List-of-row-arrays storage churn** (`vectro.py:376`, `hnsw_api.py:935`) —
  keep contiguous `(N,D)` matrices instead of exploding into N row objects.
- ⬜ **float64 widening in SQ/INT2 decode** (`scalar_quant.py:71`) — use
  `np.float32` literals to keep intermediates f32.
- ⬜ **`estimate_recall` `alive_ids.index()` O(N) per query** (`hnsw_api.py:1252`)
  — prebuild a position dict. (eval path)
- ⬜ **`simd_tier()` shells out / reads /proc uncached** (`_rust_bridge.py:37`) —
  `lru_cache`. (cold)

---

## Recommended order of attack

1. ✅ Distance-kernel foundation (AVX2 `l2_sq`, IVF SIMD routing) — **done**.
2. HNSW per-query heap → thread-local scratch (P1, highest QPS/risk ratio).
3. Quantized-HNSW flat code store + prefetch (where vectro beats FAISS).
4. IVF-PQ ADC: LUT reformulation + PQ4 fast-scan (beats FAISS IVF-PQ).
5. Coarse k-means via SIMD `assign_nearest` (build time).
6. PyO3 `allow_threads` on train/add/search + NF4 batch FFI.
7. SQ2/SQ3 stored norm + LUT; NF4/BF16 SIMD dequant.
