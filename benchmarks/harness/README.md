# VECTRO recall-matched benchmark harness

Audit item **5.2** — the measurement gate every later optimization sprint merges
through. Real datasets, recall-matched QPS comparison, 30-run percentile
statistics with a paired Wilcoxon test and a coefficient-of-variation gate, and
JSON + Markdown results carrying a full scope line.

## One command

```bash
# Fetch a dataset (cached under benchmarks/data/, gitignored, checksum-verified):
python benchmarks/harness/download.py --dataset sift1m

# Run the core suite at the 0.90 / 0.95 recall@10 operating points:
python benchmarks/harness/run.py --suite core --dataset sift1m

# Harness self-test + two-run stability kill-test (no download, seconds):
python benchmarks/harness/run.py --dataset synthetic --stability --verbose
```

Results land in `benchmarks/results/<timestamp>_<tag>.{json,md}` (never
overwritten), each tagged with the git commit and hardware scope.

## What it does

1. **Datasets** (`datasets.py`) — SIFT1M / GIST1M `.fvecs`/`.ivecs` readers, a
   checksummed downloader, exact brute-force ground truth, and a `LOADERS`
   registry so an embedding dataset (Cohere / MS MARCO, …) is added by
   registering one loader — no other harness changes. A tiny `synthetic` loader
   powers the self-test.
2. **Recall-matched protocol** (`protocol.py`) — for each engine, sweep the
   recall knob (`ef` / `n_probe`) to the smallest value reaching recall@10 =
   0.90 / 0.95 (± 0.005), then measure QPS **at that operating point**. QPS is
   never compared at unmatched recall. Runs are interleaved A/B/A/B so neither
   engine runs systematically hot; start/end timestamps are logged.
3. **Statistics** (`stats.py`) — p50/p95/p99 QPS + per-query latency
   percentiles, paired Wilcoxon signed-rank (p < 0.05) with a reported effect
   size and paired-median improvement, and a **CoV gate**: if run-to-run CoV
   exceeds 10 %, the environment is declared too noisy to claim anything.
4. **Engines** (`engines.py`) — VECTRO fp32 HNSW, VECTRO int8 HNSW, VECTRO
   IVF-PQ4, faiss (HNSWFlat, IVF-PQ), hnswlib. A uniform `build`/`search`
   interface; versions pinned into the JSON. An unavailable engine is
   **skipped with a recorded reason**, never silently dropped.
5. **Report** (`report.py`) — JSON + a Markdown table with the full scope line
   (chip · dataset · metric · recall target · config · commit).

## Harness kill-test (before any optimization is measured through it)

`run.py --stability` runs the full suite twice back-to-back; the two runs' p50
QPS per config must agree within the CoV gate (10 %). A failure means the
*harness* (warmup, pinning, interleaving) needs fixing before Phase 2 — not the
engine. The synthetic self-test demonstrates this end-to-end on any host.

## Environment notes

- **faiss / hnswlib** install into a documented venv:
  `pip install faiss-cpu hnswlib`. Absent → those baselines are skipped with a
  reason and their versions are recorded as `n/a`.
- **VECTRO INT8 quant-HNSW** search and the Rust **IVF-PQ4** batched path are
  not exposed through the pure-Python API (they live in the Rust core, awaiting
  PyO3 binding). Until bound, those baselines are skipped with that reason; the
  full 1M-scale baseline table is produced with the built extension
  (`maturin develop`) on the target host — pure-Python HNSW is the correctness
  baseline, not a throughput baseline, and is infeasible at 1M.
- Every headline is **"QPS at recall@10 = 0.95"**, scoped to the chip and
  dataset in the table's scope line — never a raw-throughput or best-of-N
  aggregate.
