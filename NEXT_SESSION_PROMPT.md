# Vectro — Next Session Prompt

Read this first. It prevents context drift and tells you exactly where the
optimization-audit sprint sequence stands.

## Current State (as of 2026-07-11)

- Version: Python 5.24.0 / Rust `vectro_lib` 8.17.0 (**deliberately not bumped
  this sprint — see "Version bump deferred" below**).
- Branch: `claude/vectro-audit-sprint-1-lab5pv`.
- Test status: Rust 242 `vectro_lib` tests + Python 1295 passed / 170 skipped /
  0 failed (pure-Python fallback mode; the compiled extension was not built in
  the sprint's x86 CI container).

## What this sprint (audit sprint 1) did

Implemented the first two items of `VECTRO_OPTIMIZATION_AUDIT_2026-07.md` in the
audit's mandated order. **This sprint ran on an x86_64 Linux container, not the
M3 the audit targets** — that shapes what could and could not be measured.

1. **Phase 0** — README version drift fixed (5.0.0 → 5.24.0, badge + banner);
   build verified green (Rust 242 + Python 1295).
2. **Phase 1 — recall-matched benchmark harness (audit 5.2), `benchmarks/harness/`.**
   Shipped and self-tested: datasets/protocol/stats/engines/report/run, 12 unit
   tests, two-run stability kill-test demonstrated on the synthetic self-test
   (PASS at 9.9 % drift on this noisy x86 host). This is the gate every item
   below merges through.
3. **Phase 2 — NEON `sdot` INT8 kernel (audit 2.1), `rust/vectro_lib/src/quant/int8.rs`.**
   Implemented, correctness-proven (bit-identical exactness test vs scalar),
   x86 green + documented clippy gate clean, aarch64 cross-compile assembles the
   inline-asm `sdot`. **Performance NOT measured — no Apple Silicon here.** The
   kill-test is pre-registered PLANNED/PENDING in `PERF_FINDINGS.md`. No speedup
   is claimed.

## FIRST TASK next session (must run on Apple Silicon / FEAT_DotProd hardware)

**Close the Phase 2 kill-test before anything else.** The kernel is committed
but its perf gate is unrun:
1. Run the isolated microbenchmark: ns/vector for `dot_i8_sdot` vs
   `dot_i8_f32_neon` at d = 128 and d = 960.
2. Run the end-to-end gate through the Phase 1 harness on SIFT1M int8-HNSW at
   recall@10 = 0.95: 30-run paired Wilcoxon (p < 0.05) + effect size, recall
   delta < 0.001, CoV ≤ 10 %. Expected 2–4× kernel; if end-to-end < 3 %, treat
   as FAIL, revert, and record the negative result.
3. This requires the INT8 quant-HNSW search path exposed through PyO3 (the
   harness `vectro-hnsw-int8` engine is currently skipped-with-reason because it
   is not bound) — bind it, or run the gate from a Rust-side harness.
4. Fill `PERF_FINDINGS.md`'s pending section with planned-vs-actual numbers.
   Only then bump VERSION (see below) and claim the win.

## Version bump deferred (rationale)

The sprint's post-flight says "bump VERSION once", but the marquee optimization
(Phase 2) has **not passed its kill-test** (unmeasurable on x86). Bumping the
package version to mark a release of an unverified optimization contradicts the
repo's core discipline (`PERF_FINDINGS.md` is a record of unmeasured changes
reverted, not shipped). The bump is deferred to the session that runs the Phase
2 gate on target hardware; CHANGELOG entries stay under `[Unreleased]`. When
bumping, touch all four: `pyproject.toml`, `python/__init__.py`,
`python/vectro.py`, `rust/vectro_lib/Cargo.toml` — and re-sync the README badge.

## Remaining audit sequence (one sprint each, in order — do not reorder)

Every item merges only through the Phase 1 harness on real datasets, with the
standard gate (30-run paired Wilcoxon, p < 0.05, effect size, CoV ≤ 10 %,
losses documented in `PERF_FINDINGS.md`).

1. **RaBitQ + extended RaBitQ quantizer** (audit 1.1) — gate: Pareto-dominant vs
   VECTRO IVF-PQ4 and faiss IVF-PQ at recall 0.90/0.95 on SIFT1M + GIST1M.
2. **CSR flat adjacency + flat code store, one serialization migration**
   (audit 3.1 + 3.3) — gate: no QPS regression, memory reduction measured.
3. **Graph reordering pass** (audit 3.2) — gate: QPS gain at identical recall.
   (Rust core primitive already shipped; wire through PyO3 + measure via harness.)
4. **Quantization-graph fusion index** (audit 1.2) — gate: ≥ 2× hnswlib QPS at
   recall@10 = 0.95 or no merge.
5. **ADSampling early termination** (audit 1.3).
6. **ann-benchmarks wrapper + submission prep** (audit 5.1).

## Load-bearing decisions logged this sprint (Ledger)

- Harness protocol: recall-matched only (never raw QPS at unmatched recall);
  interleaved A/B/A/B; CoV gate as the noise circuit-breaker; scope line on every
  table. Engines unavailable on a host are skipped-with-reason, never dropped.
- NEON `sdot` mirrors the VNNI design (once-per-search prepared query, integer
  accumulate, final scale). ISA asymmetry: VNNI u8×i8 + bias vs NEON signed
  i8×i8 direct. Emitted via inline asm because `vdotq_s32` is unstable on the
  crate's stable toolchain. d ≥ 128 activation floor mirrors the measured VNNI
  crossover; lowering it awaits the on-hardware microbenchmark.
- Version bump deferred (above) — a deliberate deviation from the sprint's
  "bump once" instruction, on Konjo "no unmeasured claims" grounds.

## Commands

- Harness self-test:      `python benchmarks/harness/run.py --dataset synthetic --stability --verbose`
- Harness (real data):    `python benchmarks/harness/download.py --dataset sift1m && python benchmarks/harness/run.py --suite core --dataset sift1m`
- Rust tests:             `cargo test -p vectro_lib`
- Documented clippy gate: `cargo clippy -- -D warnings`
- aarch64 cross-check:    `cargo clippy -p vectro_lib --target aarch64-unknown-linux-gnu -- -D warnings`
- Python tests:           `python3 -m pytest tests/ -q`
