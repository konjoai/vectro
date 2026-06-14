# Vectro — Plan

> Last updated: 2026-05-19
> Current version: **5.5.0** (Python) / **8.0.0** (Rust) — quantization audit, QuantizationAuditor, QuantizationReport, VectorPairMetrics, RecallResult.

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
