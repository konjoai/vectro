# Vectro — Plan

> Last updated: 2026-07-01
> Current version: **5.24.0** (Python) / **8.17.0** (Rust) — GEMM k-means assignment for IVF/IVF-PQ training plus a prefetch fix for the IVF-PQ ADC scan's dominant DRAM-latency cost.

---

## Researched Feature Roadmap

Researched against the vector-search literature and production deployments
(Qdrant, Weaviate, Pinecone, FAISS) as of 2026-05.  Priority assigned by
user impact × implementation cost.

---

### 🔴 P1 — Critical (implement now)

| Feature | Description | Status |
|---------|-------------|--------|
| **Recall estimator** | `index.estimate_recall(sample_size=1000)` — samples random query vectors from the stored corpus, runs both HNSW and brute-force, returns recall@k with Wilson 95% CI. Exposes via `GET /api/recall_estimate`. Demo UI shows a recall gauge. | ✅ v5.1.0 |
| **HNSW graph compaction / tombstone cleanup** | After deletes, nodes become unreachable (silent recall degradation). `compact()` detects orphaned nodes, reconnects dangling edges, removes tombstones. `stats()` includes `orphaned_node_count`, `deleted_count`. Exposes via `POST /api/compact`. | ✅ v5.1.0 |
| **Vector metadata filtering (pre-filter)** | `search(filter={"field": "value"})` alongside the query vector. HNSW traversal skips filtered-out nodes during graph walk (not post-filter). Metadata stored per-vector in a sidecar dict. | ✅ v5.1.0 |

---

### 🟠 P2 — High Impact / Medium Complexity

| Feature | Description | Status |
|---------|-------------|--------|
| **Hybrid BM25 + dense search** | `POST /index/{name}/search` accepts `text` alongside `query` (vector). BM25 scores over each vector's `metadata["text"]`. `alpha`-weighted min-max fusion (0=BM25 only, 1=dense only); response carries `mode` + per-hit `dense_score`/`bm25_score`. | ✅ V8 |
| **Scalar / product quantization** | `quantization: "sq8" \| "pq32"` on collection creation. SQ: scale to int8 per-dim. PQ: 8 sub-quantizers of 4 bits each. 75-97% memory reduction. `GET /collections/{name}/quantization_stats`. | ⬜ Planned |
| **HNSW search trace visualization** | `search(..., trace=True)` returns a `SearchTrace` alongside `(indices, distances)`: entry point, per-layer descent nodes, all layer-0 candidates, final result heap. Powers the animated beam in demo/viz.html. | ✅ v5.2.0 (HNSW) |
| **Batch upsert with deduplication** | `add_batch(vectors, ids, metadata)` — deduplicates by string ID, updates existing vectors in-place (O(1) per update, no graph surgery), returns `{inserted, updated, node_ids}`. Also adds `get_by_id(str_id)`. | ✅ v5.2.0 (HNSW) |

---

### 🟡 P3 — Strategic

| Feature | Description | Status |
|---------|-------------|--------|
| **ACORN-style filtered HNSW** | Filtered search during graph traversal for high-selectivity predicates (solving zero-result post-filter at 1% selectivity). See arXiv:2403.04871. | ⬜ Planned |
| **Persistent HNSW on disk** | `save(path)` / `load(path)` upgraded from pickle to numpy `.npz` format — no arbitrary code execution on load, magic-byte detection, backward-compat DeprecationWarning for old pickle files. | ✅ v5.2.0 (HNSW) |
| **Multi-vector per document** | Multiple embeddings per document ID (title + body), max-pool distances. | ⬜ Planned |
| **Namespace partitioning** | Logical namespaces within a collection, isolated HNSW graphs, unified cross-namespace search. | ⬜ Planned |

---

## IVF-PQ search — ADC scan prefetch ✅ COMPLETE (2026-07-01)

### Summary
The CHANGELOG's IVF-PQ coarse-scan entry flagged the ADC table-lookup loop
as "memory-bound at K=256" and the next lever for search throughput.
Profiling the actual candidate scan in `adc_rank` traced that cost to
`code_row(gid)`, not the ADC table gather: `gid` is global insertion order,
not list-local order, so each candidate's PQ code row sits at an essentially
random offset across the full `[n_vectors * n_subspaces]`-byte buffer — a
DRAM-latency miss per candidate at scale — while the small `[M*K]` ADC table
built once per query stays cache-resident across the whole scan. Added
`simd::prefetch_read` (NEON `prfm` / x86_64 `_mm_prefetch`) and used it to
prefetch each candidate's code row 8 candidates ahead of where it's scored,
overlapping that latency with other candidates' compute instead of stalling
serially on it. Pure latency hiding — no numeric or ranking change.

### Deliverables
| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `simd::prefetch_read` — cfg-gated NEON/x86_64 cache-prefetch hint, no-op elsewhere | ✅ |
| 2 | `IvfPqIndex::adc_rank` posting-list scan issues an 8-ahead prefetch for `code_row` | ✅ |
| 3 | Full `vectro_lib` test suite green; `aarch64-unknown-linux-gnu` cross `cargo check` clean | ✅ |

### Results
Release build, n=500,000, d=128, n_lists=1024, n_probe=32, M=16, K=256,
single-query `search`, 2,000 synthetic unit-norm queries: **458 → ~650 qps
(~1.4×)**, consistent across repeated runs.

---

## v5.24.0 / v8.17.0 — PQ4 fast-scan NEON `vqtbl1q_u8` for aarch64 ✅ COMPLETE (2026-07-01)

### Summary
`IndexPQFastScan`'s PQ4 fast-scan (shared by `Pq4FlatIndex` and
`IvfPq4Index`) had an AVX2 `pshufb` kernel on x86_64 but fell back to the
scalar gather on aarch64, so Apple Silicon — the flagship
`bench-darwin-arm64` target — never got the fast-scan win. Added `scan_neon`:
NEON's `vqtbl1q_u8` is the direct analogue of AVX2 `pshufb`, resolving 16
candidates' per-subspace distances against the 16-byte LUT in one table
lookup. Unlike AVX2's `unpack{lo,hi}_epi8` — which permutes candidate order
within each 128-bit lane and needs a `PERM` table to invert — `vqtbl1q_u8` +
`vget_{low,high}` preserve candidate order, so results store straight to the
output with no permute needed. NEON is mandatory in the aarch64 base ISA, so
the path is unconditional (no runtime detection). Shipped as PR #99; PR #100
fixed the kiban `konjo-gates` CI workflow so its `repo:*` gates actually
provision the Rust toolchain and evaluate the diff.

### Deliverables
| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `scan_neon` — NEON `vqtbl1q_u8` PQ4 fast-scan kernel (`rust/vectro_lib/src/index/pq4.rs`) | ✅ |
| 2 | Dispatcher routes aarch64 unconditionally to `scan_neon` (NEON is baseline, no runtime detection) | ✅ |
| 3 | Scalar gather retained as the reference kernel for non-AVX2 x86_64 and other targets | ✅ |
| 4 | Gate-hygiene fixes: `cargo fmt`, `clippy -D warnings`, `# Safety` docs on the new `unsafe` block | ✅ |
| 5 | CI fix (#100): `konjo-gates` workflow provisions the Rust toolchain so its `repo:*` gates run against the real diff instead of failing as spurious net-new findings | ✅ |

### Results
- Byte-exact vs the scalar reference (`scan_simd_matches_scalar` property test,
  `u16` sums) under `qemu-aarch64` cross-compilation, plus the full
  `ranking_agrees_with_exact_adc` and `ivf_pq4` recall/batch suite.
- Confirmed green on real Apple Silicon via the merged `Rust tests
  (macos-latest)` CI job.
- Throughput number on real Apple Silicon still pending a `bench-darwin-arm64`
  run; the AVX2 twin documents ~22× over the scalar gather on that platform's
  SIMD, which the NEON kernel is expected to approach.

---

## v5.6.0 — INT8 batch path → Rust SIMD kernel ✅ COMPLETE (2026-06-18)

### Summary
`VectroBatchProcessor.quantize_batch` always used the NumPy abs-max path for
INT8 — even when the compiled `vectro_py` SIMD kernel was installed — leaving a
~15-20× speedup unused and dropping the d=1536 end-to-end throughput just below
its 45K vec/s floor on x86 hosts. Routing the batch path through the kernel
(with a NumPy fallback) fixes both. Shipped as PR #36; the throughput-test
de-jitter follow-up shipped on top.

### Deliverables
| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `VectroBatchProcessor` INT8 profiles dispatch to `vectro_py.quantize_int8_batch` when the extension is present; NumPy fallback otherwise | ✅ |
| 2 | `batch_encode_into_with_range(..., range_factor)` in `vectro_lib` — threads rf through the per-row SIMD encode (effective scale `abs_max/rf`); `batch_encode_into` is now a `rf=1.0` wrapper | ✅ |
| 3 | `quantize_int8_batch(vectors, range_factor=1.0)` PyO3 keyword, validated to `(0, 1]`; backward compatible | ✅ |
| 4 | `_rust_bridge` / `batch_api` thread `range_factor`; new Rust + Python parity/fallback/validation tests | ✅ |
| 5 | De-jitter `test_rust_int8_throughput_{1m_floor,cross_dimension}` to best-of-5 with warm-up (floors unchanged) | ✅ |

### Results
- d=1536 end-to-end `VectroBatchProcessor`: ~42K → ~110K vec/s (raw kernel
  ~730K; the `list`/`np.stack` wrapper is the remaining ceiling).
- Numeric parity vs the NumPy baseline (the correctness baseline): scales
  identical, codes differ by ≤1 level only at round-half-to-even vs
  round-half-away ties, cosine ≥ 0.9999 across `fast`/`balanced`/`quality`.

### Rejected: fused single-pass kernel for the batch path
Measured (`int8_fused_bench`, n=100k × d=768): two-pass **7.72 Gelem/s** vs
rayon-fused **5.17 Gelem/s** — fused is **~33% slower**. At d=768 a 3 KB row
already fits L1, so the two-pass second read is a cache hit and the fused
buffer-copy is pure overhead. The two-pass abs-max kernel is optimal here; the
1M-floor flakiness was a measurement-statistic issue (mean-of-3 vs best-of-5),
not a kernel-speed issue (peak on the x86 CI runner is ~1.5-2M vec/s).

---

## v5.2.0 (HNSW) — Persistent .npz index, add_batch upsert, search trace ✅ COMPLETE (2026-05-13)

### Summary
Three P2/P3 items shipped as one sprint, all implemented on `HNSWIndex` in
`python/hnsw_api.py` with zero API breakage.

### Deliverables
| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `HNSWIndex.save(path)` — replaces pickle with `numpy.savez_compressed`; vectors as float32 matrix; graph/metadata/deleted/id_map as JSON byte arrays inside the ZIP archive | ✅ |
| 2 | `HNSWIndex.load(path)` — detects format by magic bytes; `.npz` primary path with `allow_pickle=False`; legacy pickle path emits `DeprecationWarning` | ✅ |
| 3 | `HNSWIndex._load_npz(path)` / `HNSWIndex._load_pickle(path)` — internal helpers, keep `load()` clean | ✅ |
| 4 | `HNSWIndex._id_map: Dict[str, int]` — string-ID registry added to `__init__` and serialised in the new format | ✅ |
| 5 | `HNSWIndex.add_batch(vectors, ids, metadata)` — upsert with deduplication; O(1) in-place update for existing IDs (no graph surgery); returns `{inserted, updated, node_ids}`; resurrects soft-deleted nodes | ✅ |
| 6 | `HNSWIndex.get_by_id(str_id)` — metadata lookup by string ID, `None` for deleted | ✅ |
| 7 | `HNSWIndex.search(..., trace=False)` — optional third return value when `trace=True` | ✅ |
| 8 | `SearchTrace` dataclass — `entry_point`, `layer_descents`, `l0_visited`, `l0_candidates_final` | ✅ |
| 9 | `tests/test_hnsw_v2.py` — 39 tests covering all three features (12 persistence, 15 add_batch, 12 trace) | ✅ |
| 10 | PLAN.md P2/P3 rows updated to ✅ v5.2.0 | ✅ |
| 11 | Version bump 5.1.0 → 5.2.0 | ✅ |

### Design notes
- **Pickle elimination**: numpy `.npz` is a ZIP container — no arbitrary code
  execution, safe to open untrusted files with `allow_pickle=False`. Each
  `.npz` embeds vectors as a proper float32 matrix + JSON blobs for the graph
  and metadata. File sizes are comparable (compressed JSON ≈ compressed pickle).
- **`add_batch` in-place update**: updating an existing vector means overwriting
  `_vectors[nid]` and `_metadata[nid]` and clearing the tombstone. The graph
  links are deliberately unchanged. This is the correct trade-off: an expensive
  graph-reconnect would be needed only if the vector moves drastically (that
  scenario calls for `delete` + re-insert, not upsert).
- **`SearchTrace`**: returned as the third element of a 3-tuple when `trace=True`.
  Caller unpacks naturally via `ids, dists, tr = idx.search(...)`. The
  `l0_candidates_final` list is sorted ascending so the first element is the
  nearest neighbour.

### Validation
- 39 new tests, all pass. No regressions in the 1019-test baseline suite.
- `recall` agree within 0.01 before/after `.npz` round-trip (verified by
  `test_recall_within_tolerance_after_round_trip`).

---

## v5.1.0 (HNSW) — Recall estimator, HNSW compaction, metadata pre-filtering ✅ COMPLETE (2026-05-12)

### Summary
Three P1 items from the Researched Feature Roadmap, shipped as one sprint.

All three are implemented directly on `HNSWIndex` in `python/hnsw_api.py` so
they work with or without the demo server — the server just exposes them as
HTTP endpoints.

### Deliverables
| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `HNSWIndex.add(..., metadata=)` — per-vector metadata sidecar | ✅ |
| 2 | `HNSWIndex.delete(node_id)` — O(1) tombstone mark | ✅ |
| 3 | `HNSWIndex.search(..., filter=)` — pre-filter during graph walk | ✅ |
| 4 | `HNSWIndex.stats()` — node count, deleted count, orphan count, avg degree | ✅ |
| 5 | `HNSWIndex.compact()` — tombstone removal + orphan reconnection | ✅ |
| 6 | `HNSWIndex.estimate_recall(sample_size, k, ef)` — brute-force vs HNSW recall@k with Wilson 95% CI | ✅ |
| 7 | `demo/server.py` — `GET /api/recall_estimate`, `POST /api/compact`, `GET /api/stats`, metadata filter in `POST /api/search` | ✅ |
| 8 | `demo/viz.html` — recall gauge panel (live if server running, static otherwise) | ✅ |
| 9 | `tests/test_hnsw_extended.py` — 27 new tests covering all P1 features | ✅ |
| 10 | Version bump 5.0.2 → 5.1.0 | ✅ |

---

## v5.5.0 — Quantization Audit (2026-05-11)

### Summary

Adds a structured quality-audit layer that compares original float32 vectors
against their quantized/compressed counterparts and produces a rich diagnostic
report.

**`python/quantization_audit.py` — new module.**
`VectorPairMetrics` is a frozen dataclass capturing per-vector `cosine_similarity`,
`l2_error`, and `relative_error`.  `RecallResult` records a single Recall@K
result.  `QuantizationReport` aggregates all per-vector metrics, aggregate
statistics (mean/min/p5 cosine similarity, mean L2 error), optional Recall@K
scores at K=1/5/10, compression ratio, and the k worst-case vector indices.
`QuantizationAuditor.run()` validates shapes, casts to FP32, computes all
metrics, and returns a `QuantizationReport`.  `_cosine_similarities` uses
`np.einsum` for numerical stability; `_recall_at_k` performs brute-force
exact search suitable for audit sets up to ~100 K vectors.

**`python/cli.py` — `audit` subcommand.**
Reads original vectors from a `.npy` file, compresses with the specified
`--precision` mode, runs the audit, and prints `report.summary()` or the full
JSON output with `--json`.

**`tests/test_quantization_audit.py` — 20 tests.**
Covers: all report fields present, frozen VectorPairMetrics, identical-vector
cosine ≈ 1, identical-vector L2 ≈ 0, cosine range [-1,1], positive compression
ratio, n_vectors matches input, mean_cosine ≤ 1, p5 ≤ mean ≤ 1, worst_k
length, worst_k are truly worst, recall_at_{1,5,10} in [0,1], recall disabled
→ None, JSON roundtrip, summary non-empty string, dtype strings recorded,
shape mismatch raises ValueError, seeded recall deterministic.

**`python/__init__.py` / `python/__init__.pyi`** — exports `QuantizationAuditor`,
`QuantizationReport`, `VectorPairMetrics`, `RecallResult`; version bumped to `5.5.0`.

**Version bump in all 4 version files:** `python/vectro.py`,
`python/__init__.py`, `pyproject.toml`, `pixi.toml`.

### Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `python/quantization_audit.py` — `VectorPairMetrics`, `RecallResult`, `QuantizationReport`, `QuantizationAuditor` | ✅ |
| 2 | `python/cli.py` — `audit` subcommand | ✅ |
| 3 | `tests/test_quantization_audit.py` — 20 tests | ✅ |
| 4 | `python/__init__.py` — new exports + version `5.5.0` | ✅ |
| 5 | `python/__init__.pyi` — stubs for 4 new audit symbols | ✅ |
| 6 | `python/vectro.py` — `__version__ = "5.5.0"` | ✅ |
| 7 | `pyproject.toml` — version `5.5.0` | ✅ |
| 8 | `pixi.toml` — version `5.5.0` | ✅ |

---

## v5.4.0 — Pipeline Checkpointing (2026-05-12)

### Summary

Adds save/load checkpointing for `CompressionPipeline`, enabling reproducible
pipeline configurations and experiment tracking without re-specifying stages.

**`python/pipeline_checkpoint.py` — new module.**
`PipelineCheckpoint` is a frozen dataclass capturing `version` (schema version
string), `created_at` (ISO-8601 UTC timestamp), `stage_configs` (ordered list
of per-stage dicts), and `metadata` (arbitrary user dict).
`save_pipeline(pipeline, path, *, metadata)` serialises a `CompressionPipeline`
to a human-readable JSON file using an atomic write (write to `.tmp`, then
`os.replace`) and creates parent directories automatically.
`load_pipeline(path)` deserialises the JSON and reconstructs a
`CompressionPipeline` with the same stage sequence; raises `FileNotFoundError`
on missing file and `ValueError` on invalid schema.
`checkpoint_info(path)` reads only the metadata without constructing a
pipeline, returning the raw dict — useful for inspecting checkpoints in scripts.

**`PipelineStage` introspection** — `to_config()` / `from_config()` added to
`PipelineStage` in `async_pipeline.py`.  `to_config()` returns a serialisable
dict with `name`, `mode`, and optional `profile`/`group_size`; `from_config()`
reconstructs the stage.

**`tests/test_pipeline_checkpoint.py` — 18 tests.** Covers file creation,
valid JSON output, stage-name round-trip, `CompressionPipeline` type,
`checkpoint_info` key set, version correctness, metadata round-trip, atomic
write, parent-dir creation, `TypeError` on wrong arg, `FileNotFoundError`,
`ValueError` on bad schema, zero-stage pipeline, three-stage pipeline,
`None` metadata, nested metadata, `to_config()` and `from_config()`.

**`python/__init__.py`** — exports `PipelineCheckpoint`, `save_pipeline`,
`load_pipeline`, `checkpoint_info`; version bumped to `5.4.0`.

**Version bump in all 4 version files:** `python/vectro.py`,
`python/__init__.py`, `pyproject.toml`, `pixi.toml`.

### Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `python/pipeline_checkpoint.py` — `PipelineCheckpoint`, `save_pipeline`, `load_pipeline`, `checkpoint_info` | ✅ |
| 2 | `python/async_pipeline.py` — `PipelineStage.to_config()` / `from_config()` | ✅ |
| 3 | `tests/test_pipeline_checkpoint.py` — 18 tests | ✅ |
| 4 | `python/__init__.py` — new exports + version `5.4.0` | ✅ |
| 5 | `python/__init__.pyi` — stubs for 4 new checkpoint symbols | ✅ |
| 6 | `python/vectro.py` — `__version__ = "5.4.0"` | ✅ |
| 7 | `pyproject.toml` — version `5.4.0` | ✅ |
| 8 | `pixi.toml` — version `5.4.0` | ✅ |

---

## v5.3.0 — Pipeline Telemetry & Observability (2026-05-07)

### Summary

Adds a structured, pluggable telemetry layer on top of v5.2.0's
`CompressionPipeline`, giving users per-stage metrics (throughput,
cosine fidelity, compression ratio, latency) emitted as
JSON-serialisable `TelemetryEvent` objects through pluggable
`TelemetryHook` callbacks.

**`python/telemetry.py` — new module.**
`TelemetryEvent` is a frozen dataclass capturing `stage_name`,
`stage_index`, `latency_ms`, `input_shape`, `output_shape`,
`input_dtype`, `output_dtype`, `compression_ratio`,
`throughput_vecs_per_sec`, `cosine_fidelity`, and an open-ended `extra`
dict for application metadata.  `TelemetryCollector` manages a list of
`TelemetryHook` callables and fans events out to all of them via
`emit()`.  `InMemoryTelemetryCollector` subclasses the base collector
and stores every event in a list; `export_json()` returns a JSON array
string.  `attach_telemetry()` monkey-patches a `CompressionPipeline`
instance's `run()` method to emit one event per stage, measuring
per-stage latency, throughput, compression ratio, and cosine fidelity
(computed in FP32 via `np.einsum` — accumulation-accurate) — all
transparent to the caller.

**`tests/test_telemetry.py` — 17 new tests.** Covers `TelemetryEvent`
construction and immutability, `to_dict()` key set and type coercions,
`to_json()` validity, `TelemetryCollector` attach/detach/clear/count,
duplicate-attach idempotency, `InMemoryTelemetryCollector` storage and
`export_json()`, and the `attach_telemetry()` pipeline integration:
event count per stage, stage-name/index correctness, throughput
positivity, compression-ratio positivity, the SIMD cosine-fidelity
property test (INT8 cosine similarity ≥ 0.9999 on L2-normalised
unit-vector inputs), multi-stage event ordering, latency non-negativity,
and `run()` return type unchanged.

**`python/__init__.py`** — exports `TelemetryEvent`, `TelemetryCollector`,
`TelemetryHook`, `InMemoryTelemetryCollector`, `attach_telemetry`;
version bumped to `5.3.0`.

**Version bump in all 4 version files:** `python/vectro.py`,
`python/__init__.py`, `pyproject.toml`, `pixi.toml`.

### Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `python/telemetry.py` — `TelemetryEvent`, `TelemetryCollector`, `TelemetryHook`, `InMemoryTelemetryCollector`, `attach_telemetry` | ✅ |
| 2 | `tests/test_telemetry.py` — 17 tests | ✅ |
| 3 | `python/__init__.py` — new exports + version `5.3.0` | ✅ |
| 4 | `python/__init__.pyi` — stubs for 5 new telemetry symbols | ✅ |
| 5 | `python/vectro.py` — `__version__ = "5.3.0"` | ✅ |
| 6 | `pyproject.toml` — version `5.3.0` | ✅ |
| 7 | `pixi.toml` — version `5.3.0` | ✅ |

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
