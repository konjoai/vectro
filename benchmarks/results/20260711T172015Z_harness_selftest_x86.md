# VECTRO recall-matched benchmark — synthetic

Scope: chip=x86_64 · dataset=synthetic (n=5000, d=64, queries=200) · metric=l2 · k=10 · runs=30 · commit=121f32b

| Engine | Version | Recall target | Param | Achieved recall | p50 QPS | p95 QPS | p99 QPS | CoV | p50 lat (ms) | p99 lat (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| vectro-hnsw-fp32 | vectro 5.24.0 (backend=python) | 0.90 | ef=1024 | 0.5465 ⚠ | 963.0 | 1008.1 | 1017.5 | 0.033 | 1.225 | 2.154 |
| vectro-hnsw-fp32 | vectro 5.24.0 (backend=python) | 0.95 | ef=1024 | 0.5465 ⚠ | 953.8 | 1011.0 | 1018.7 | 0.050 | 1.107 | 1.399 |

## Skipped engines

- **vectro-hnsw-int8**: INT8 quant-HNSW search not exposed via python API (Rust QuantHnswIndex awaits PyO3 binding); run this baseline from the Rust harness on target hardware.
- **vectro-ivfpq**: vectro_py extension not built (vectro_py is required.  Build it with `maturin develop` or `pip install vectro` first.); run `maturin develop`
- **faiss-hnsw-flat**: faiss not installed: No module named 'faiss'
- **faiss-ivfpq**: faiss not installed: No module named 'faiss'
- **hnswlib**: hnswlib not installed: No module named 'hnswlib'

## Harness two-run stability (kill-test)

- Verdict: **PASS** (max per-config p50 QPS drift 9.923%, gate 10%)
  - vectro-hnsw-fp32 @ recall 0.9: run1 p50=963.0, run2 p50=867.4, drift=9.923%
  - vectro-hnsw-fp32 @ recall 0.95: run1 p50=953.8, run2 p50=863.9, drift=9.428%
