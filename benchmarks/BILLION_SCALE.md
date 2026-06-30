# Billion-scale benchmarking (10M · 100M · 1B)

How to validate vectro past 1M, what the standard datasets are, and the memory
reality that decides which index you can actually run.

## Standard datasets above 1M

The field standardised on a handful of base-1B sets, each shipped with query
sets and **precomputed ground truth for the 1M / 10M / 100M / 1B slices** — so
you benchmark a slice without recomputing GT.

| Dataset | N (base) | dim | dtype | metric | source |
|---|---|---|---|---|---|
| **BIGANN / SIFT1B** | 1 000M | 128 | uint8 | L2 | corpus-texmex.irisa.fr |
| **GIST1M** | 1M | 960 | fp32 | L2 | corpus-texmex.irisa.fr |
| **Deep1B** | 1 000M | 96 | fp32 | L2/angular | Yandex (Babenko) |
| **Microsoft SPACEV1B** | 1 000M | 100 | int8 | L2 | MS (web search) |
| **Microsoft Turing-ANNS 1B** | 1 000M | 100 | fp32 | L2 | MS |
| **Yandex Text-to-Image 1B** | 1 000M | 200 | fp32 | inner-product | Yandex (cross-modal) |
| **Facebook SSNPP 1B** | 1 000M | 256 | uint8 | range/L2 | Meta |

The umbrella framework is **big-ann-benchmarks** (NeurIPS'21 Billion-Scale ANN
Challenge, `github.com/harsha-simhadri/big-ann-benchmarks`). It defines the
10M/100M/1B query sets + GT for six of the above and a common `.u8bin`/`.fbin`
wire format:

```
<uint32 npts><uint32 dim><npts*dim values, row-major, little-endian>
```

GT files: `<uint32 nq><uint32 k><nq*k int32 ids><nq*k f32 dists>`.

`scripts/bench_scale.py --format bigann` reads these directly.

## The memory wall — why the index type matters more than the kernel

At 100M the binding constraint is RAM, and it splits into **vectors** and **graph**:

| Component (N=100M, d=128) | Size |
|---|---|
| fp32 vectors (`N·d·4`) | **51.2 GB** |
| int8 / uint8 vectors (`N·d`) | 12.8 GB |
| **PQ-16 codes** (`N·16`) | **1.6 GB** |
| binary codes (`N·d/8`) | 1.6 GB |
| HNSW layer-0 graph (`N·(m0+1)·4`, m0=32) | **13.2 GB** |
| IVF lists (`N·4` postings) | 0.4 GB |

So:

- **Flat HNSW (fp32)** at 100M ≈ 51 + 13 = **~64 GB** → a 64–128 GB server. The
  *graph* (13 GB) is irreducible for HNSW; the vectors dominate but compress.
- **HNSW + int8 vectors** ≈ 13 + 13 = **~26 GB** → a 32 GB box.
- **IVF-PQ** ≈ 1.6 GB codes + coarse centroids + postings ≈ **<4 GB** → a laptop.
  This is how FAISS runs SIFT1B; it is the only in-RAM option at 1B on commodity
  hardware. vectro exposes it as `PyIvfPqIndex`.
- **DiskANN-style** (graph on SSD) trades RAM for latency; not yet in vectro.

**Takeaway:** the "100M+ on a workstation" claim is a *compression* claim, not a
flat-HNSW claim. It is realised by IVF-PQ (or PQ/int8/binary codes), which is
vectro's core. Benchmark the claim with the compressed index, and report RAM.

## What runs where

| Scale | Flat HNSW (fp32) | HNSW+int8 | IVF-PQ |
|---|---|---|---|
| 1M | any laptop | laptop | laptop |
| 10M (~6.5 GB) | 16 GB laptop ✓ | laptop | laptop |
| 100M | 64 GB server | 32 GB server | **16 GB laptop ✓** |
| 1B | 256 GB+ server | 128 GB server | 32–64 GB server |

## Running it

```bash
# 1M real data (already validated, see headtohead_*.json)
python scripts/bench_l2_headtohead.py data/sift-128-euclidean.hdf5

# 10M slice, flat HNSW, real BIGANN data (needs ~6.5 GB free)
python scripts/bench_scale.py --format bigann --base bigann_base.u8bin \
    --query bigann_query.u8bin --gt gt_10M.ibin --n 10_000_000 --index hnsw

# 100M slice via IVF-PQ on a 16 GB laptop (codes ~1.6 GB)
python scripts/bench_scale.py --format bigann --base bigann_base.u8bin \
    --query bigann_query.u8bin --gt gt_100M.ibin --n 100_000_000 \
    --index ivfpq --nlist 65536 --nprobe 64 --pq-subspaces 16

# No dataset on hand? Stream-generate N synthetic vectors (never holds all
# fp32 in RAM — encodes in chunks) to stress build/memory at chosen scale:
python scripts/bench_scale.py --synthetic --n 10_000_000 --dim 128 --index ivfpq
```

`bench_scale.py` reports build time, **peak RSS** (fresh-process), on-disk size,
and recall-vs-QPS, and writes timestamped JSON to `benchmarks/results/`.

## Honest status

- ✅ 1M validated on real data (SIFT-1M L2, GloVe-1.18M cosine) — vectro wins
  search QPS and builds 3–4× faster than faiss at equal recall + memory.
- ⏳ 10M/100M: harness ready; needs the BIGANN download (~130 GB for the full
  base) or the streaming-synthetic mode. Run on a box sized per the table above.
- 1B is a server-class run; documented for completeness.

### IVF-PQ caveats (the 100M-on-a-laptop path)

`PyIvfPqIndex` is what makes 100M fit in RAM, but two current limitations matter
when reading its numbers:

1. **Cosine/angular only.** Train/add/query all unit-normalise the vectors, so
   the index ranks by cosine. Evaluate it against **cosine** ground truth (the
   synthetic harness does this for `--index ivfpq`). L2 datasets (SIFT/BIGANN,
   Deep, SPACEV) need an L2 mode — not yet implemented; use cosine datasets
   (Text2Image-style, normalised embeddings) for the 100M IVF-PQ run today.
2. **Non-residual PQ.** Codes encode the full normalised vector, not the
   `vector − coarse_centroid` residual. Simpler and fast, but recall trails
   FAISS's residual `IVFx,PQy` at equal bytes — raise `n_probe` to compensate.

Both are tracked as follow-ups; they do not affect the flat-HNSW path (which has
full L2/IP/cosine support, see `Metric`).
