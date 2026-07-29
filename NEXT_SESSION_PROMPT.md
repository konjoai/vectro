# Vectro -- Next Session Prompt

Read this first. It prevents context drift.

## Track A1 -- kiban v1.9.0 reconciliation (as of 2026-07-29, most recent)

CI-reconnection sprint (Konjo cross-repo work order, Track A1): bumped
`.konjo/kiban.ref`/`KIBAN_REF` from the drifted `v1.1.0`/`v1.1.5` pair to
`v1.9.0`, ran `gate_polarity` full-tree, adopted kiban's `profiles/vectro.yml`,
de-decorated all 16 soft (`continue-on-error`) steps in `konjo-gate.yml`, and
converted `CLAUDE.md` to the Phase-13 section contract. See `LEDGER.md` for
every finding and its disposition, and `CHANGELOG.md`'s matching entry.

**Carried forward, not done this sprint (explicit non-goal of Track A1 -- "no
vectro feature/perf work"):**

1. **Rust coverage measurement** -- `cargo llvm-cov nextest --workspace` was
   not run to completion; this sprint's sandbox filled its disk installing
   the tooling. Install `cargo-nextest` + `cargo-llvm-cov` on a host with
   headroom, measure real line coverage, and seed `.konjo/coverage-floor.txt`
   (same ratchet shape as `.konjo/*-ceiling.txt`, see `LEDGER.md`).
2. **Python coverage measurement** -- `pytest --cov` currently fails to
   collect 75 of `tests/`'s files because `vectro_py` isn't built
   (`maturin develop` was not run). Either build it first, or audit which
   test files import `vectro_py` unconditionally at module scope instead of
   behind the `try/except ImportError` + `pytest.mark.skipif` guard
   convention `test_retriever.py`/`test_hybrid_search.py` already use, then
   measure and seed a coverage floor.
3. **Mutation testing kill-test** -- `cargo mutants --in-diff` was never run
   to a real pass/fail this sprint (too slow to fit the session budget).
   Before promoting G3 off `continue-on-error`, run the same kind of
   kill-test this sprint's own KT-A1.1 used for clippy/cargo-deny/cargo-audit:
   introduce a real mutation-catchable defect, confirm cargo-mutants reports
   it as a surviving/caught mutant correctly, then promote.
4. **`clippy::pedantic` + `#[must_use]`/doc-`# Panics`/`# Errors` cleanup** --
   528 standing violations ratcheted in `.konjo/clippy-pedantic-ceiling.txt`,
   almost entirely style/idiom lints (not the unwrap/expect security
   invariant, which is now clean and blocking). Real incremental cleanup
   work, one PR at a time, ratcheting the ceiling down as it goes.
5. **`vulture`/`radon`/DRY/rustdoc debt** -- see
   `.konjo/vulture-ceiling.txt` (11), `.konjo/*-complexity-ceiling.txt`
   (58 Python + 4 Rust), `.konjo/dry-ceiling.txt` (271),
   `.konjo/rustdoc-ceiling.txt` (24). Same pattern: pay down incrementally,
   ratchet down as you go, never up without a stated reason.
6. **pyo3 0.20.3 → >=0.29.0 major upgrade** -- closes RUSTSEC-2025-0020 and
   RUSTSEC-2026-0177 for real instead of via `.cargo/audit.toml` /
   `.konjo/deny.toml` ignore entries. A real API-breaking change across
   `vectro_py`'s whole binding surface -- plan it as its own sprint, not a
   drive-by fix.
7. **bincode → wincode/postcard migration** -- closes RUSTSEC-2025-0141 for
   real (currently ignored: bincode's own former maintainers call 1.3.3
   "complete," not vulnerable, only unmaintained). A real on-disk
   storage-format migration if ever undertaken -- needs a version bump and a
   compatibility/rollback story, not a drive-by fix.
8. **`gate_polarity` false positives worth reporting to kiban upstream** (see
   `LEDGER.md`'s Gate-Polarity-Baseline-1 for detail): (a) the engine's
   Python block-scan can match an unrelated keyword argument inside a
   multi-line function call (`pq_api.py:111`'s `logger.warning(...,
   exc_info=True)` -- matched as if `exc_info=True` were the branch's returned
   value); (b) it currently scans `#[test]`-body Rust code, matching internal
   test-assertion bookkeeping (`main.rs:786`'s `found = true;`) as if it were
   a permissive-pass gate; (c) several "empty container → return True" shapes
   (no-filter-means-match-all, delete-all-when-ids-is-None) are domain
   defaults, not fail-open gates, but the engine doesn't yet recognize them
   as the same class of false negative its own docstring already carves out
   for `until_satisfied`-style domain checks.

## Original audit-sprint carry (below -- a different, still-open track)

The rest of this file is the perf-audit track's own handoff. Track A1 did not
touch perf work (explicit non-goal) -- this is unresolved, not superseded.

## Current State (as of 2026-07-11)

- Version: Python 5.24.0 / Rust `vectro_lib` 8.17.0 (**deliberately not bumped
  this sprint -- see "Version bump deferred" below**).
- Branch: `claude/vectro-audit-sprint-1-lab5pv`.
- Test status: Rust 242 `vectro_lib` tests + Python 1295 passed / 170 skipped /
  0 failed (pure-Python fallback mode; the compiled extension was not built in
  the sprint's x86 CI container).

## What this sprint (audit sprint 1) did

Implemented the first two items of `VECTRO_OPTIMIZATION_AUDIT_2026-07.md` in the
audit's mandated order. **This sprint ran on an x86_64 Linux container, not the
M3 the audit targets** -- that shapes what could and could not be measured.

1. **Phase 0** -- README version drift fixed (5.0.0 → 5.24.0, badge + banner);
   build verified green (Rust 242 + Python 1295).
2. **Phase 1 -- recall-matched benchmark harness (audit 5.2), `benchmarks/harness/`.**
   Shipped and self-tested: datasets/protocol/stats/engines/report/run, 12 unit
   tests, two-run stability kill-test demonstrated on the synthetic self-test
   (PASS at 9.9 % drift on this noisy x86 host). This is the gate every item
   below merges through.
3. **Phase 2 -- NEON `sdot` INT8 kernel (audit 2.1), `rust/vectro_lib/src/quant/int8.rs`.**
   Implemented, correctness-proven (bit-identical exactness test vs scalar),
   x86 green + documented clippy gate clean, aarch64 cross-compile assembles the
   inline-asm `sdot`. **Performance NOT measured -- no Apple Silicon here.** The
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
   is not bound) -- bind it, or run the gate from a Rust-side harness.
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
`python/vectro.py`, `rust/vectro_lib/Cargo.toml` -- and re-sync the README badge.

## Remaining audit sequence (one sprint each, in order -- do not reorder)

Every item merges only through the Phase 1 harness on real datasets, with the
standard gate (30-run paired Wilcoxon, p < 0.05, effect size, CoV ≤ 10 %,
losses documented in `PERF_FINDINGS.md`).

1. **RaBitQ + extended RaBitQ quantizer** (audit 1.1) -- gate: Pareto-dominant vs
   VECTRO IVF-PQ4 and faiss IVF-PQ at recall 0.90/0.95 on SIFT1M + GIST1M.
2. **CSR flat adjacency + flat code store, one serialization migration**
   (audit 3.1 + 3.3) -- gate: no QPS regression, memory reduction measured.
3. **Graph reordering pass** (audit 3.2) -- gate: QPS gain at identical recall.
   (Rust core primitive already shipped; wire through PyO3 + measure via harness.)
4. **Quantization-graph fusion index** (audit 1.2) -- gate: ≥ 2× hnswlib QPS at
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
- Version bump deferred (above) -- a deliberate deviation from the sprint's
  "bump once" instruction, on Konjo "no unmeasured claims" grounds.

## Commands

- Harness self-test:      `python benchmarks/harness/run.py --dataset synthetic --stability --verbose`
- Harness (real data):    `python benchmarks/harness/download.py --dataset sift1m && python benchmarks/harness/run.py --suite core --dataset sift1m`
- Rust tests:             `cargo test -p vectro_lib`
- Documented clippy gate: `cargo clippy -- -D warnings`
- aarch64 cross-check:    `cargo clippy -p vectro_lib --target aarch64-unknown-linux-gnu -- -D warnings`
- Python tests:           `python3 -m pytest tests/ -q`
