# vectro

Ultra-high-performance embedding compression library -- INT8 · NF4 · PQ-96 · Binary · HNSW · RQ · VQZ -- with Rust kernels, optional Mojo SIMD acceleration, and PyO3 Python bindings.

**v5.24.0** (Python) / **v8.17.0** (Rust) -- 1452 Python + 230 Rust tests passing.

## Org rules

@~/.konjo/kiban/plugins/konjo/skills/konjo/SKILL.md

The org ethos applies here: ship over optimize, kill-test first, statistical rigor,
honest negative results, evidence first, token-efficient context.

Editorial rules: no em dashes, no AI-tell vocabulary. The prose lint enforces it; run
`konjo-prose` on docs before pushing.

Log durable decisions with `konjo-decision decide` at `repo:vectro` scope. Search with
`konjo-decision search` before reopening a settled call.

When you catch a mistake worth not repeating, invoke `correct`: it records a learning
with `konjo-learn` and proposes the smallest durable fix. A learning must name where
its rule lives (a CLAUDE.md line, a prose-lint word, a lane, or a gate), or it is
refused.

Build the Konjo way: the `craft` skill carries the four behaviors (think before coding,
simplicity first, surgical changes, goal-driven execution) plus the verify-loop and the
pre-implementation trust-boundary contract. `verify_cmd` is declared in
`.konjo/profile.yml`.

## Stack
Rust 2021 · ndarray · rayon · simsimd · half · PyO3 · anyhow · criterion · Mojo (optional) · Python 3.10+ · NumPy · pixi

## Commands
```bash
cargo build                                  # build workspace
cargo test --workspace                       # run all crate tests
cargo clippy --workspace --lib --bins --examples --all-features -- -D warnings -D clippy::unwrap_used -D clippy::expect_used -D clippy::panic -D clippy::todo -D clippy::dbg_macro
                                              # production-scope strict lint (blocking, see Invariants)
cargo bench --bench encode                   # criterion benchmarks
make bench-darwin-arm64 WAVE=1               # paper benchmark (Darwin arm64)
make bench-arxiv WAVE=1                      # full benchmark + notebook render
python -m pytest tests/ -x                   # Python test suite (needs `maturin develop` first)
pixi install && pixi shell                   # Mojo environment (optional)
pixi run build-mojo                          # compile Mojo kernels (optional)
```

## Invariants
- No `unwrap()`/`expect()` outside tests -- use `anyhow::Result` and `?` (enforced: `repo:clippy` -- `-D clippy::unwrap_used -D clippy::expect_used -D clippy::panic -D clippy::todo -D clippy::dbg_macro`, production scope: `--lib --bins --examples`, blocking in `konjo-gate.yml` G1 and in `.konjo/hooks/pre-commit`; the broader `--all-targets` + `-D clippy::pedantic` sweep is ratcheted, see `.konjo/clippy-pedantic-ceiling.txt`)
- No silent failures -- log via `tracing::warn!` whenever a fallback swallows an error (ADVISORY)
- `cargo build` must stay green -- fix before doing anything else (ADVISORY; the repo's CI build step is the actual check, not a gated diff assertion)
- SIMD kernels require property tests: cosine ≥ 0.9999 on adversarial inputs (enforced: `repo:clippy`'s `cargo test` path runs real `proptest!` cosine-quality tests in `quant/int8.rs`/`bf16.rs`/`binary.rs`, part of `verify_cmd`; ADVISORY on the specific "1e6-magnitude" bound, which is not yet a dedicated fixture -- see `NEXT_SESSION_PROMPT.md`)
- dtype explicit at every Rust/Python array boundary -- never rely on implicit casting (ADVISORY)
- Accumulate in FP32 for all quantized matmuls -- document any exception with a measured benchmark (ADVISORY)
- NaN/Inf assertion checks at module boundaries during development -- never ship masked overflow (ADVISORY)
- Python-only mode is always the correctness baseline -- Rust/Mojo acceleration must match it numerically (ADVISORY)
- `--features vectro_lib_accelerate` is macOS-only -- never gate correctness on it (ADVISORY)
- Benchmark results go to `benchmarks/results/` with timestamp + full hardware metadata -- never overwrite (ADVISORY)
- Experiment outputs in `experiments/runs/<timestamp>_<name>/` -- always new directory, never overwrite (ADVISORY)
- Seed all stochastic ops; log the seed in every benchmark JSON output (ADVISORY)
- Version bumps touch `pyproject.toml` + `python/__init__.py` + `python/vectro.py` + `rust/vectro_lib/Cargo.toml` (ADVISORY)
- No new file over 500 lines outside `.konjo/oversized-allowlist.txt` (enforced: `.konjo/scripts/file_size_check.py`, blocking in `konjo-gate.yml` G4)
- `cargo deny`/`cargo audit` must be clean against `.konjo/deny.toml` / `.cargo/audit.toml` (enforced: `repo:cargo-deny`, `repo:cargo-audit`, blocking in `konjo-gate.yml` G1 and `konjo-gates.yml`)

## Repo map
| Crate | Role |
|-------|------|
| `vectro_lib` | Core quantization kernels: INT8 (NEON 32-wide / AVX2 / AMX), NF4, PQ-96, Binary, HNSW, RQ, VQZ |
| `vectro_cli` | `vectro` CLI binary -- quantize, search, benchmark subcommands |
| `vectro_py` | PyO3 bindings -- `quantize_int8_batch` (zero-copy f32), `quantize_int8_batch_from_f16` |
| `generators` | Vector data generators for benchmarking and property testing |

## Repo-specific rules

### Python Modules
| Module | Role |
|--------|------|
| `python/vectro.py` | Main Python API: `AutoQuantize`, `HNSW`, all quantization modes |
| `python/quantization_extra.py` | INT2/INT4 bit-packing via NumPy (fallback path) |
| `benchmarks/vectro_paper_benchmark.py` | Reproducibility harness: `--quick / --table / --json / --reps / --warmup` |
| `scripts/aggregate_paper_tables.py` | Aggregates `results/paper/*.json` into paper tables |

### Planning Docs
- `PLAN.md` -- current sprint state and version history
- `VECTRO_V3_PLAN.md` -- v3 architecture audit and research landscape (Q1 2026)
- `VECTRO_OPTIMIZATION_AUDIT_2026-07.md` -- algorithm-layer audit (RaBitQ,
  quantization-graph fusion, ANN research 2024–2026); the unpark plan for
  when VECTRO resumes past the current kernel-tuning ceiling
- `CHANGELOG.md` -- all notable changes (Keep a Changelog format)
- `BACKLOG_v2.1.md` -- feature backlog

### Konjo Quality Framework

Three walls against AI slop.

**Wall 1 -- Pre-commit** (`bash .konjo/scripts/install-hooks.sh`):
cargo check, clippy, ruff lint, ruff format, DRY check, TODO scan. Blocks the commit.

**Wall 2 -- CI gate** (`.github/workflows/konjo-gate.yml` + `konjo-gates.yml`):
fmt/clippy (production scope)/dead-code/ruff/cargo-deny/cargo-audit block; file-size
blocks outside the grandfather list; coverage/mutation/DRY/pedantic-clippy/vulture/
rustdoc/complexity are ratcheted (never regress past their recorded ceiling -- see
`.konjo/*-ceiling.txt`) or, where not yet measurable in CI (Rust/Python coverage,
mutation testing), soft with a named owner and target date -- see `LEDGER.md`'s
soft-step triage for the current disposition of every step.

**Wall 3 -- Adversarial review** (local only -- disabled in CI):
`git diff HEAD~1 | python3 .konjo/scripts/konjo_review.py`

See `KONJO_QUALITY_FRAMEWORK.md` for the full specification.

### Skills
See `.claude/skills/` -- auto-loaded when relevant.
Run `/konjo` to boot a full session (Brief + Discovery + Plan).

### Pinning

This repo pins a kiban ref in `.konjo/kiban.ref` and `KIBAN_REF` in
`.github/workflows/konjo-gates.yml` -- bump both together in the same commit.
`.konjo/scripts/check_kiban_pin.py` fails CI if they diverge (see that workflow's
"Check twin kiban pin" step); this repo's `.konjo/kiban.ref`/`KIBAN_REF` pair drifted
silently once already (`v1.1.0` vs `v1.1.5`) before that check existed.
