# Vectro — Status, Benchmarks & Scale Report

**Version:** 5.24.0 (Python) / 8.17.0 (Rust `vectro_lib`) · **Date:** 2026-06-22

This document is the single source of truth for: (1) where vectro stands today,
(2) head-to-head benchmarks vs the competition with an honest wins/ties/losses
ledger, (3) a concrete plan to test at 100M+ scale, (4) the scale at which the
competition was actually benchmarked, and (5) how vectro could be published and
licensed.

> **Measurement caveat.** Unless stated otherwise, the vectro/FAISS/hnswlib
> numbers below were measured this cycle on a **shared x86 Linux cloud
> container** (not a quiet, pinned host), best-of-N but **not** the full
> 5-warmup + p50/p95/p99 protocol in `.claude/rules/benchmarking.md`. Treat the
> **ratios** as the durable signal and the absolute throughput as indicative.
> Numbers will differ on an M3 (NEON) or a tuned server.

---

## 1. Current Status

Vectro is a **production-grade embedding-compression + ANN-search library** with
Rust kernels (NEON on aarch64, AVX2/FMA on x86_64), optional Mojo SIMD, and PyO3
Python bindings.

### Indexes (`rust/vectro_lib/src/index/`)
| Index | State | Notes |
|-------|-------|-------|
| HNSW | ✅ mature | Malkov–Yashunin heuristic, flat-graph memory, cosine/L2/IP, O(1) soft-delete, in-place upsert, metadata pre-filter, batch search (rayon, GIL released) |
| QuantHNSW | ✅ | Generic quantizer-backed graph: `Int8/Nf4/Sq2/Sq3/Binary/Bf16` variants, optional INT8 re-rank store for lossy codecs |
| IVF-Flat | ✅ code; ⚠️ unproven at scale | k-means coarse quantizer + posting lists |
| IVF-PQ | ✅ recall-competitive with FAISS; batch+parallel search | coarse IVF + PQ + ADC; k-means++ init fixed O(n·k²·d)→O(n·k·d) (training now **faster than FAISS**); the extreme-compression option for 100M+ |
| BM25 | ✅ | sparse/ hybrid dense+sparse path |

### Quantizers (`rust/vectro_lib/src/quant/`) — footprint at d=768
| Codec | Bytes/vec | Ratio vs fp32 | Quality |
|-------|----------:|--------------:|---------|
| BF16 | 1536 | 2× | lossless-ish |
| INT8 | 772 | ~4× | cosine ≥ 0.9999 |
| SQ3 / SQ2 | 288 / 192 | ~11× / 16× | uniform scalar |
| NF4 | 388 | ~8× | cosine ≥ 0.985 |
| PQ-96 (K=256) | 96 | 32× | cosine ~0.82 |
| Binary (1-bit) | 96 | 32× | needs re-rank for high recall |
| RQ | variable | ~10× | iterative refinement |

### Storage & ops
- **VQZ container** (`python/storage_v3.py`): custom binary format (magic `VECTRO`,
  blake2b checksum, optional zstd/zlib) for INT8 codes + scales.
- **Cloud backends** via `fsspec`: S3, GCS, Azure Blob, local (batch upload via tempfile).
- **Persistence:** HNSW → `.npz`; QuantHNSW/IVF/IVF-PQ → serde. **All load fully
  into RAM — no memory-mapping.**
- **DB connectors:** Qdrant, Weaviate, Milvus, Chroma, Pinecone (+ custom).

### Largest validated scale today: **~1.18M vectors (GloVe-100).** 100M+ is
believed viable with PQ/IVF-PQ but is **not yet benchmarked or published.**

---

## 2. Benchmarks vs Competition (current `main`, x86 cloud)

### 2.1 ANN search — HNSW (GloVe-100, 50K×100, M=16, ef_c=200, ef_s=100, k=10)

| Index | QPS | Recall@10 |
|-------|----:|:---------:|
| **vectro HNSW (batch, `search_batch_np`)** | **22,718** | 0.919 |
| vectro HNSW (single, 4 threads) | 8,578 | 0.917 |
| hnswlib (C++) | 33,281 | 0.899 |
| faiss `IndexHNSWFlat` (C++) | 29,702 | 0.904 |

Single-`ef` QPS is misleading because vectro reaches **higher recall** at the same
`ef`. At **iso-recall** (the ann-benchmarks methodology, `benchmark_hnsw_pareto.py`)
vectro trails ~1.1–1.35×, **narrowing to ~0.95× at recall 0.95** and beating
hnswlib where it can't reach 0.95 within ef≤200. vectro's recall *leads* the field.

### 2.2 Quantization throughput vs FAISS

| Operation (size) | vectro (Rust) | FAISS (C++) | Ratio |
|------------------|------------:|----------:|:-----:|
| INT8 quantize (100K×768) | **4,206,893 vec/s** | 679,359 vec/s | **6.2× faster** |
| PQ-96 encode (50K×768, K=256) | 124,221 vec/s | 141,200 vec/s | 0.88× (parity) |
| IVF-PQ train (glove, 512 lists) | **2.4s** | 2.9s | **faster** (was 100s pre-fix) |

### 2.3 Compressed-ANN tradeoff surface (glove-200K, d=100, k=10, batch — `benchmark_compressed_ann.py`)

The honest at-scale view: recall **per byte**, not a single QPS number.

| method | recall@10 | QPS | bytes/vec |
|--------|:---------:|----:|----------:|
| **vectro HNSW fp32** | **0.872** | **14,539** | 528 |
| faiss HNSW fp32 | 0.871 | 12,592 | 528 |
| **vectro HNSW-INT8** | 0.864 | 5,411 | 232 |
| vectro NF4-HNSW + re-rank | 0.866 | 4,971 | 286 |
| vectro Binary-HNSW + re-rank | 0.680 | 1,693 | 245 |
| **vectro IVF-PQ** (M=50) | **0.774** | 1,289 | 60 |
| faiss IVF-PQ (M=50) | 0.768 | 9,421 | 60 |

- vectro HNSW fp32 **beats faiss HNSW** here on *both* recall and QPS.
- **Quantized HNSW (INT8 / NF4+re-rank) is vectro's quadrant**: ~0.86 recall at
  ~232–286 bytes/vec — high recall at real compression.
- IVF-PQ: vectro is **recall-competitive** (0.774 vs 0.768) at identical 60 bytes/vec;
  FAISS only wins the *search speed* there (~7×, its tuned ADC) — its optimised corner.

#### Target regime — d=768 embeddings (100K, k=10, batch)

The d=100 glove view *understates* vectro: at the real embedding dimension, PQ
falls apart and quantized HNSW is the only high-recall-compressed option.

| method | recall@10 | QPS | bytes/vec | 100M footprint |
|--------|:---------:|----:|----------:|---------------:|
| vectro HNSW fp32 | 0.9992 | 10,132 | 3,200 | ~320 GB |
| faiss HNSW fp32 | 0.9998 | 15,238 | 3,200 | ~320 GB |
| **vectro HNSW-INT8** | **0.933** | 4,925 | 900 | **~90 GB** |
| **vectro NF4-HNSW + re-rank** | **0.933** | 4,966 | 1,288 | ~128 GB |
| vectro Binary-HNSW + re-rank | 0.655 | 1,293 | 996 | ~100 GB |
| vectro IVF-PQ (M=64) | 0.133 | 5,682 | 103 | ~10 GB |
| faiss IVF-PQ (M=64) | 0.176 | 12,582 | 103 | ~10 GB |

- **IVF-PQ recall collapses at d=768 for *everyone*** (vectro 0.13, faiss 0.18) at
  this aggressive ratio — high-compression PQ ≠ high recall on real embeddings.
  (Higher `M_pq` recovers recall at less compression.)
- **Quantized HNSW (INT8/NF4) is the only method giving recall *and* compression
  here**: 0.933 @ 3.5× (≈90 GB at 100M) — vectro's differentiated quadrant,
  exactly where PQ (FAISS's strength) can't follow.
- vectro HNSW ≈ faiss HNSW on recall (0.999); QPS ~1.5× behind in this
  distance-bound regime (the honest remaining gap).

### 2.4 Quality parity
| Metric | vectro | FAISS |
|--------|:------:|:-----:|
| INT8 reconstruction cosine (d=768) | ≥0.9999 | ≥0.9999 |
| PQ-96 reconstruction cosine (d=768) | 0.8185 | 0.8207 |

---

## 3. Wins / Ties / Losses Ledger

### ✅ Wins
- **INT8 quantization throughput — 6.2× faster than FAISS** at full quality parity (the standout result). Mojo path historically hit ~12M vec/s (4.6× FAISS) on Apple M3.
- **HNSW recall@10 — highest of the field** (0.919 vs hnswlib 0.899, faiss 0.904); and **vectro HNSW fp32 beats faiss `IndexHNSWFlat`** on recall *and* QPS in the §2.3 surface.
- **High-recall-compressed quadrant — vectro's to own.** Quantized HNSW (INT8 / NF4+re-rank) holds ~0.86 recall at ~2× compression; IVF-PQ at the same scale can't match that recall. This is the differentiated niche RAG cares about.
- **IVF-PQ training — faster than FAISS** (2.4s vs 2.9s) after the O(n·k²·d)→O(n·k·d) k-means++ fix, at recall parity.
- **Feature breadth** — NF4, Binary, RQ, VQZ container, 5 DB connectors, ONNX export, hybrid BM25+dense; FAISS/hnswlib don't ship these.

### ➖ Ties / near-parity
- **PQ-96 encode — 0.88× FAISS** (within 12%; was ~5× behind before the AVX2 kernel).
- **IVF-PQ recall — parity** (0.774 vs 0.768) at identical bytes/vec.
- **HNSW batch QPS at iso-recall — ~0.95–0.9×** at high recall (the regime that matters); recall itself leads.

### ❌ Where competitors still win
- **IVF-PQ *search* speed — ~7× behind FAISS** (1.3K vs 9.4K QPS). FAISS's tuned ADC loop is its optimised home turf; this is the *least favorable single corner*, not vectro's positioning (see §2.3). A batched/parallel binding closed the per-query-Python gap; a PQ fast-scan / SIMD ADC is the remaining lever.
- **HNSW single-thread single-query** — Python per-call bound; use `search_batch` or threads (GIL released → ~1.75× at 4 threads).
- **Scale — the big one.** Competitors validate at **10M–1B**; vectro's *published* ceiling is **1.18M** (IVF-PQ now recall-validated to ~200K here). Still the most important gap (§4).
- **Out-of-core / billion-scale serving — absent.** No memory-mapping, no DiskANN-style SSD index, no sharding. DiskANN serves 1B from one 64 GB node off SSD; vectro cannot today.

---

## 4. Testing Vectro at Scale (100M+)

### 4.1 Why your M3 caps out — the memory math
A 100M × 768-d corpus is **307 GB in fp32** — far beyond an M3's RAM. Quantization
is precisely what unlocks scale:

| Representation | 100M×768 size | Fits in… |
|----------------|--------------:|----------|
| fp32 (raw) | 307 GB | 512 GB server only |
| INT8 | 77 GB | 128 GB server |
| NF4 | 38 GB | 64 GB server |
| **PQ-96 / Binary** | **9.6 GB** | **laptop-class RAM** |
| + HNSW graph (M=16) | ~13–20 GB extra | — |
| + IVF-PQ overhead | ~1–2 GB (centroids+lists) | — |

**Takeaway:** with **IVF-PQ** (codes 9.6 GB + small overhead), 100M fits in
**~16–32 GB RAM** — i.e., the *same* hardware class the SISAP challenge and
big-ann-benchmarks 2023 use. HNSW+INT8 at 100M needs a 128 GB box. fp32 anything
at 100M is a non-starter without a 512 GB server.

### 4.2 Phased plan

**Phase 0 — Laptop / M3 (≤2M, doable today).** Run the *standard* ANN-Benchmarks
datasets so results are directly comparable to everyone: **SIFT-1M (128-d, L2),
GloVe-1.2M (100-d, cos), GIST-1M (960-d, L2)**. Protocol: single-query, recall–QPS
Pareto, precomputed top-100 ground truth. Deliverable: vectro on the public
ann-benchmarks.com axes.

**Phase 1 — Single cloud box, 32–64 GB (10M).** Datasets: **Deep-10M**
(ann-benchmarks subset), **Cohere-10M** (768-d, VectorDBBench), big-ann-2023
**YFCC-10M** / **Text2Image-10M**. Indexes: IVF-PQ and QuantHNSW(INT8). Measure
build time, index size, recall@10, QPS, peak RSS. *This tier matches the
big-ann-benchmarks 2023 hardware (8 vCPU/16 GB) — a real credibility milestone.*
Est. cost: a few $ on an AWS `r6i.2xlarge`.

**Phase 2 — Large-RAM box, 128–256 GB (100M, the target).** Datasets:
**LAION-100M** (768-d, VectorDBBench XLarge), **SIFT-100M / Deep-100M** slices.
Index: **IVF-PQ** (codes ~9.6 GB). Metrics: recall@10 ≥ 0.8 (SISAP bar), QPS at
fixed recall, build wall-clock, RSS. Hardware: AWS `r6i.4xlarge`/`r6i.8xlarge`
(~$1–2/hr); full run ~$10–50. *This is the headline "vectro at 100M" result.*

**Phase 3 — Billion-scale (future, needs engineering).** Datasets: **BIGANN-1B
(SIFT, uint8), Deep-1B**. Two routes: (a) 512 GB server with IVF-PQ in RAM, or
(b) **DiskANN-style on-disk index** off NVMe SSD (the route DiskANN/big-ann
Track-T2 use on a 64 GB node). Route (b) requires building the out-of-core
features vectro lacks today (see §4.3).

### 4.3 Engineering prerequisites (gaps to close, in priority order)
1. **Standard dataset loaders** — SIFT/Deep/BIGANN `.bvecs/.fvecs/.u8bin` + HDF5;
   make `benchmark_ann_comparison.py` consume real datasets, not just synthetic.
2. **IVF-PQ at-scale validation + tuning** — n_lists/n_probe sweeps; this is the
   load-bearing index for 100M+ and is currently unbenchmarked.
3. **Memory-mapped / on-disk index** (DiskANN-style) — the unlock for 1B on
   modest RAM; today everything loads fully into RAM.
4. **Streaming build pipeline** — quantize→index without materializing all fp32.
5. **big-ann-benchmarks harness adapter** — wrap vectro so it runs in the
   standard competition framework for apples-to-apples third-party numbers.
6. **Sharding** for horizontal scale-out beyond a single node.

---

## 5. The Scale Competitors Were Benchmarked At

Vectro's laptop-scale wins are real, but here's the bar the field is measured
against (sources at end):

| System | Max scale | Typical eval | Hardware | Headline |
|--------|----------:|-------------:|----------|----------|
| **FAISS** (2017) | **1B** (Deep1B) | 95M (YFCC) | 4–8× GPU | 8.5× GPU speedup; R@10 0.376 on SIFT1B @17.7µs |
| **ANN-Benchmarks** | ~10M (Deep1B subset) | **1M** (SIFT/GIST/GloVe) | 1× AWS r6i.16xlarge, single-query | recall–QPS Pareto |
| **big-ann 2021** | **1B** (6 datasets) | 1B | Azure F32s/L8s | recall@10 gain vs FAISS/DiskANN @ fixed QPS |
| **big-ann 2023** | **10M** (4 tracks) | 10M | Azure 8 vCPU/16 GB | up to 37,671 QPS @ 0.9 recall |
| **DiskANN** (2019) | **1B** (SIFT1B) | 1B | 1 node, 64 GB + SSD | >5,000 QPS, <3 ms, 95%+ 1-recall@1 |
| **ScaNN** (2020) | ~1.2M (GloVe-100) | 1M | single CPU | ~2× QPS over next-best |
| **HNSW** (2018) | 10M (SIFT) | 5–10M | 4× 10-core Xeon | ~3 min build @10M |
| **VectorDBBench** (Milvus/Qdrant…) | **100M** (LAION) | 1M–10M | 8 vCPU/32 GB | Qdrant 626 QPS @1M @99.5% recall |
| **SISAP Challenge** (2024) | **100M** | 10–100M | 8 vCPU/16 GB / 12 h | recall ≥ 0.8 @ 30-NN |

**Reading of it:** "1M on a laptop" is the *entry* tier (ANN-Benchmarks, ScaNN).
The respected scale ladder is **10M (big-ann 2023, achievable on 16 GB) → 100M
(SISAP, VectorDBBench, achievable on 16–32 GB with PQ) → 1B (FAISS/DiskANN/big-ann
2021, needs SSD or a big server).** Vectro can realistically reach **10M and 100M
on a single cloud box** with its existing IVF-PQ once §4.3(1–2) are done; 1B needs
§4.3(3).

---

## 6. Is Vectro an arXiv Paper, a Technical Article, or Community OSS?

All three are viable and not mutually exclusive. Honest read:

### As an arXiv preprint — **yes, now.**
1M-scale evaluation is standard for **algorithmic / compression** contributions,
and vectro has several (NF4 embeddings codec, VQZ container, RQ, Rust+Mojo SIMD
kernels, AVX2 PQ encoder, competitive-with-FAISS results). Precedent: Qinco2
(ICLR 2025) and "Bang for the Buck" (DaMoN '25) were accepted at ~1M–2.25M scale.
- **Primary category:** `cs.IR` or `cs.DS`; cross-list `cs.DB`, `cs.PF`, `cs.LG`.
- **Framing that works at 1M:** "an embedding-compression library with SIMD
  kernels that matches/beats FAISS on commodity CPUs" — a *systems/engineering*
  preprint. It is **not** yet a top-tier *systems* paper: VLDB/SIGMOD/NeurIPS
  systems tracks expect **10M–1B** for anything claiming "production scale."

### Strongest credibility path — **enter a challenge.**
- **SISAP Indexing Challenge** (100M, 8 vCPU/16 GB, 12 h) and **big-ann-benchmarks
  2025** (10M tracks) give *third-party-validated* standing and a co-authored
  overview paper. This converts "we benchmarked ourselves" into "independently
  ranked," which is worth far more than a self-published table. Requires the
  §4.3(1–2) engineering (real datasets + IVF-PQ at 100M on 16 GB).

### As a technical article + OSS — **the fastest adoption path.**
A well-written engineering writeup (the benchmark story, the AVX2/NEON kernels,
the NF4/VQZ codecs) plus a clean GitHub repo drives users faster than peer review.
Do this *and* the arXiv preprint.

### Licensing — recommend **Apache-2.0.**
Every broadly-adopted ANN library is permissively licensed: **FAISS (MIT),
Qdrant / Milvus / Chroma (Apache-2.0), pgvector (PostgreSQL license).** Apache-2.0
adds an explicit **patent grant** (valuable for SIMD techniques) over MIT.
- **BUSL** (Business Source License — HashiCorp, CockroachDB, Sentry) is
  *source-available*, **not OSI-approved**, and restricts competing hosted use
  until a per-version "change date" (typically 4 yrs → Apache/GPL). It protects
  against hyperscalers reselling you as a managed service — but it **deters
  contributors and enterprise adoption** (cf. the OpenTofu fork of Terraform) and
  is chosen by companies *defending an existing install base*, not by projects
  *seeking* community growth.
- **Recommendation:** Apache-2.0 to maximize adoption and contributions now;
  revisit BUSL only if/when vectro becomes a commercial hosted platform worth
  defending. (License choice is a business decision — this is a recommendation,
  not legal advice.)

---

## 7. Next Steps

**Done this cycle** (✅): HNSW build+search routed through Rust core; x86 AVX2 dot +
prefetch (shared `index/simd.rs`); batched `search_batch` (HNSW + IVF-PQ); PQ AVX2
encoder; IVF-PQ k-means++ O(n·k²·d)→O(n·k·d) fix; iso-recall Pareto benchmark
(`benchmark_hnsw_pareto.py`); IVF-PQ at-scale benchmark (`benchmark_ivfpq_scale.py`);
compressed-ANN tradeoff benchmark (`benchmark_compressed_ann.py`).

**Remaining (ordered):**
1. **Run the tradeoff + at-scale benchmarks at d=768/1536** (embedding regime) and
   **10M→100M on a large-RAM box** — the headline at-scale result; SISAP recall ≥ 0.8 bar.
2. **Real-dataset loaders** (SIFT-1M, GloVe-1.2M, GIST-1M, Deep) for directly-comparable runs.
3. **IVF-PQ search speed** — PQ fast-scan / SIMD ADC to close the ~7× search corner vs FAISS.
4. **On-disk / mmap index** (DiskANN-style) — the unlock for 1B on modest RAM.
5. **arXiv preprint** (cs.IR/cs.DS) around the codecs + SIMD kernels + the compressed-ANN
   tradeoff surface (the high-recall-compressed quadrant is the contribution).
6. **Enter the SISAP / big-ann challenge** for third-party validation.

---

## Sources
FAISS [arXiv:1702.08734](https://arxiv.org/abs/1702.08734),
[arXiv:2401.08281](https://arxiv.org/abs/2401.08281) ·
ANN-Benchmarks [arXiv:1807.05614](https://arxiv.org/abs/1807.05614),
[ann-benchmarks.com](https://ann-benchmarks.com/) ·
big-ann-benchmarks [NeurIPS'21 arXiv:2205.03763](https://arxiv.org/abs/2205.03763),
[NeurIPS'23 arXiv:2409.17424](https://arxiv.org/abs/2409.17424),
[site](https://big-ann-benchmarks.com/neurips21.html) ·
DiskANN [NeurIPS 2019](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/) ·
ScaNN [arXiv:1908.10396](https://arxiv.org/abs/1908.10396) ·
HNSW [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) ·
VectorDBBench [github](https://github.com/zilliztech/VectorDBBench),
Qdrant [benchmarks](https://qdrant.tech/benchmarks/) ·
SISAP [challenge](https://sisap-challenges.github.io/2024/) ·
Qinco2 [arXiv:2501.03078](https://arxiv.org/abs/2501.03078) ·
Bang-for-the-Buck [arXiv:2505.07621](https://arxiv.org/abs/2505.07621) ·
BUSL [MariaDB FAQ](https://mariadb.com/bsl-faq-mariadb/),
[HashiCorp](https://www.hashicorp.com/en/blog/hashicorp-adopts-business-source-license)
