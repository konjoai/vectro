# Vectro — Plan

> Last updated: 2026-05-06
> Current version: **5.2.0** (Python) / **8.0.0** (Rust) — async compression pipeline, CompressionPipeline, PipelineStage, PipelineResult, compress_async.

---

## v5.2.0 — Async Compression Pipeline (2026-05-06)

### Summary

Adds a fully async-capable multi-stage compression pipeline to the Vectro Python API.

**`python/async_pipeline.py` — new module.** `CompressionPipeline` chains multiple
`PipelineStage` objects in sequence, feeding each stage's output as the next stage's
input. `PipelineResult` captures per-stage and total latency, input/output shapes and
dtypes, and overall compression ratio. `compress_async()` is a thin module-level helper
that wraps `Vectro.compress()` in an asyncio thread-pool executor, keeping the event
loop unblocked. `CompressionPipeline.run_async()` does the same for full pipeline runs.

**`tests/test_async_pipeline.py` — 15 new tests.** Covers stage validation,
empty-pipeline guard, 1-D input rejection, compression-ratio positivity, dtype
preservation, async round-trip, and `compress_async` basic smoke test.

**`python/__init__.py`** — exports `CompressionPipeline`, `PipelineStage`,
`PipelineResult`, `compress_async`; version bumped to `5.2.0`.

**Version bump in all 4 version files:** `python/vectro.py`, `python/__init__.py`,
`pyproject.toml`, `pixi.toml`.

### Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `python/async_pipeline.py` — `PipelineStage`, `PipelineResult`, `CompressionPipeline`, `compress_async` | ✅ |
| 2 | `tests/test_async_pipeline.py` — 15 tests | ✅ |
| 3 | `python/__init__.py` — new exports + version `5.2.0` | ✅ |
| 4 | `python/vectro.py` — `__version__ = "5.2.0"` | ✅ |
| 5 | `pyproject.toml` — version `5.2.0` | ✅ |
| 6 | `pixi.toml` — version `5.2.0` | ✅ |

---

## v5.1.0 — QuantizationConfig + Stub Completeness + Test Hardening ✅ COMPLETE (2026-05-05)

### Summary

Four parallel tracks closed in this sprint:

**Track 1 — `QuantizationConfig` dataclass (`python/vectro.py`).** A validated,
structured configuration container for `Vectro.compress()`. All parameters are
validated at construction time — unknown `precision_mode`, unknown `profile`,
non-power-of-2 `group_size`, bad `seed` type all raise `ValueError` immediately
instead of surfacing errors deep in the hot path. `from_profile(name, **overrides)`
class-method constructs a config from a named profile. `to_dict()` returns a
JSON-serialisable snapshot. `Vectro.compress(config=...)` wires it in as a clean
override of the individual kwargs. 36 new tests.

**Track 2 — Stub completeness.** `lora_api.pyi` (previously absent), `vectro.pyi`
rewritten to include `QuantizationConfig`, updated `compress(config=)` signature,
`compress_async`/`decompress_async`. `__init__.pyi` fully synced with `__init__.py`
— previously ~20 symbols behind the runtime (`lora_api`, `retriever`, `retrieval`,
`ivf_api`, `bf16_api`, `profiles`, `embeddings` all absent from the stub).

**Track 3 — Version string consistency.** `test_release_candidate.py`
`EXPECTED_VERSION` was hardcoded to `4.17.1` (3 minor versions stale). All 4
version files bumped: `pyproject.toml`, `pixi.toml`, `python/__init__.py`,
`python/vectro.py`.

**Track 4 — Test correctness gates.** Fixed 4 pre-existing failures in
`test_cross_platform_benchmarks.py`: p999 gate corrected for Python fallback path,
ADR-002 p99 `<1ms` and INT8 throughput floors guarded with `skipif not
_has_rust_ext()` (those floors are calibrated for the Rust SIMD path and should
not be enforced on Python NumPy).

### Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `python/vectro.py` — `QuantizationConfig` dataclass with `__post_init__` validation | ✅ |
| 2 | `python/vectro.py` — `Vectro.compress(config=...)` kwarg | ✅ |
| 3 | `python/lora_api.pyi` — type stubs (new file) | ✅ |
| 4 | `python/vectro.pyi` — full rewrite with `QuantizationConfig`, `compress_async` | ✅ |
| 5 | `python/__init__.pyi` — full sync: +`QuantizationConfig`, +`lora_api`, +`retriever`, +`retrieval`, +`ivf_api`, +`bf16_api`, +`profiles`, +`embeddings` | ✅ |
| 6 | `python/__init__.py` — `QuantizationConfig` exported in imports and `__all__` | ✅ |
| 7 | `tests/test_quantization_config.py` — 36 tests | ✅ |
| 8 | `tests/test_release_candidate.py` — `EXPECTED_VERSION` `4.17.1` → `5.1.0` | ✅ |
| 9 | `tests/test_cross_platform_benchmarks.py` — p999 gate, p99 skip guard, throughput skip guards | ✅ |
| 10 | Version bump `5.0.2` → `5.1.0` in all 4 version files | ✅ |
