# vectro

Ultra-high-performance embedding compression library — INT8 · NF4 · PQ-96 · Binary · HNSW · RQ · VQZ — with Rust kernels, optional Mojo SIMD acceleration, and PyO3 Python bindings.

**v5.0.2** (Python) / **v8.0.0** (Rust) — 1020 Python + 109 Rust tests passing.

## Stack
Rust 2021 · ndarray · rayon · simsimd · half · PyO3 · anyhow · criterion · Mojo (optional) · Python 3.10+ · NumPy · pixi

## Commands
```bash
cargo build                                  # build workspace
cargo test                                   # run all crate tests
cargo clippy -- -D warnings                  # lint
cargo bench --bench encode                   # criterion benchmarks
make bench-darwin-arm64 WAVE=1               # paper benchmark (Darwin arm64)
make bench-arxiv WAVE=1                      # full benchmark + notebook render
python -m pytest tests/ -x                   # Python test suite (1020 tests)
pixi install && pixi shell                   # Mojo environment (optional)
pixi run build-mojo                          # compile Mojo kernels (optional)
```

## Critical Constraints
- No `unwrap()`/`expect()` outside tests — use `anyhow::Result` and `?`
- No silent failures — log via `tracing::warn!` whenever a fallback swallows an error
- `cargo build` must stay green — fix before doing anything else
- SIMD kernels require property tests: cosine ≥ 0.9999 on adversarial 1e6-magnitude inputs
- dtype explicit at every Rust/Python array boundary — never rely on implicit casting
- Accumulate in FP32 for all quantized matmuls — document any exception with a measured benchmark
- NaN/Inf assertion checks at module boundaries during development — never ship masked overflow
- Python-only mode is always the correctness baseline — Rust/Mojo acceleration must match it numerically
- `--features vectro_lib_accelerate` is macOS-only — never gate correctness on it
- Benchmark results go to `benchmarks/results/` with timestamp + full hardware metadata — never overwrite
- Experiment outputs in `experiments/runs/<timestamp>_<name>/` — always new directory, never overwrite
- Seed all stochastic ops; log the seed in every benchmark JSON output
- Version bumps touch `pyproject.toml` + `python/__init__.py` + `python/vectro.py` + `rust/vectro_lib/Cargo.toml`

## Crate Map
| Crate | Role |
|-------|------|
| `vectro_lib` | Core quantization kernels: INT8 (NEON 32-wide / AVX2 / AMX), NF4, PQ-96, Binary, HNSW, RQ, VQZ |
| `vectro_cli` | `vectro` CLI binary — quantize, search, benchmark subcommands |
| `vectro_py` | PyO3 bindings — `quantize_int8_batch` (zero-copy f32), `quantize_int8_batch_from_f16` |
| `generators` | Vector data generators for benchmarking and property testing |

## Python Modules
| Module | Role |
|--------|------|
| `python/vectro.py` | Main Python API: `AutoQuantize`, `HNSW`, all quantization modes |
| `python/quantization_extra.py` | INT2/INT4 bit-packing via NumPy (fallback path) |
| `benchmarks/vectro_paper_benchmark.py` | Reproducibility harness: `--quick / --table / --json / --reps / --warmup` |
| `scripts/aggregate_paper_tables.py` | Aggregates `results/paper/*.json` into paper tables |

## Planning Docs
- `PLAN.md` — current sprint state and version history
- `VECTRO_V3_PLAN.md` — v3 architecture audit and research landscape (Q1 2026)
- `CHANGELOG.md` — all notable changes (Keep a Changelog format)
- `BACKLOG_v2.1.md` — feature backlog

## Skills
See `.claude/skills/` — auto-loaded when relevant.
Run `/konjo` to boot a full session (Brief + Discovery + Plan).
