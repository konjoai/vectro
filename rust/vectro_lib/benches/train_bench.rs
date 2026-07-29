//! Training, decode, and index-insertion throughput benchmarks.
//!
//! Run with:
//!   cargo bench -p vectro_lib --bench train_bench
//!
//! These cover paths that the existing benches omit and that the v8 performance
//! sprint optimizes: PQ/RQ training (Lloyd's K-means), RQ/INT8 batch decode, and
//! HNSW insertion. Establishing baselines here lets every optimization be
//! regression-gated (see .claude/rules/benchmarking.md: >5% p95 = hard stop).
//!
//! All datasets are seeded/deterministic so runs are comparable across commits.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use vectro_lib::index::hnsw::{HnswIndex, Metric};
use vectro_lib::index::ivf::IvfIndex;
use vectro_lib::index::ivf_pq::IvfPqIndex;
use vectro_lib::quant::int8;
use vectro_lib::quant::nf4;
use vectro_lib::quant::pq::{train_pq_codebook, PQCodebook};
use vectro_lib::quant::rq::{rq_decode_flat, rq_encode_flat, train_rq_codebook, RQCodebook};

/// Deterministic synthetic vectors (same generator as simd_bench for parity).
fn make_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..d)
                .map(|j| ((i * d + j) as f32 * 0.0013_f32).sin())
                .collect()
        })
        .collect()
}

/// PQ codebook training (Lloyd's K-means) throughput.
fn bench_pq_train(c: &mut Criterion) {
    const N: usize = 5_000;
    const D: usize = 128;
    let vecs = make_vecs(N, D);

    let mut group = c.benchmark_group("pq_train");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("train_n5k_d128_m8_k64_it10", |b| {
        b.iter(|| train_pq_codebook(black_box(&vecs), 8, 64, 10, 42))
    });
    group.finish();
}

/// RQ codebook training (chained PQ over residuals) throughput.
fn bench_rq_train(c: &mut Criterion) {
    const N: usize = 5_000;
    const D: usize = 128;
    let vecs = make_vecs(N, D);

    let mut group = c.benchmark_group("rq_train");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("train_n5k_d128_l2_m8_k64_it10", |b| {
        b.iter(|| train_rq_codebook(black_box(&vecs), 2, 8, 64, 10, 42))
    });
    group.finish();
}

/// RQ flat-code decode throughput (exercises decode_one per-pass path).
fn bench_rq_decode(c: &mut Criterion) {
    const N: usize = 5_000;
    const D: usize = 128;
    let vecs = make_vecs(N, D);
    let cb: RQCodebook = train_rq_codebook(&vecs, 2, 8, 64, 10, 42).expect("rq train failed");
    let codes = rq_encode_flat(&cb, &vecs);

    let mut group = c.benchmark_group("rq_decode");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("decode_flat_n5k_d128_l2", |b| {
        b.iter(|| rq_decode_flat(black_box(&cb), black_box(&codes)))
    });
    group.finish();
}

/// RQ encode throughput — exercises the residual-subtraction path (no k-means),
/// which the fused in-place update targets.
fn bench_rq_encode(c: &mut Criterion) {
    const N: usize = 5_000;
    const D: usize = 128;
    let vecs = make_vecs(N, D);
    let cb: RQCodebook = train_rq_codebook(&vecs, 2, 8, 64, 10, 42).expect("rq train failed");

    let mut group = c.benchmark_group("rq_encode");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("encode_flat_n5k_d128_l2_m8_k64", |b| {
        b.iter(|| rq_encode_flat(black_box(&cb), black_box(&vecs)))
    });
    group.finish();
}

/// PQ encode throughput (assignment-heavy; complements pq_train).
fn bench_pq_encode(c: &mut Criterion) {
    const N: usize = 5_000;
    const D: usize = 128;
    let vecs = make_vecs(N, D);
    let cb: PQCodebook = train_pq_codebook(&vecs, 8, 64, 10, 42).expect("pq train failed");

    let mut group = c.benchmark_group("pq_encode");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("encode_n5k_d128_m8_k64", |b| {
        b.iter(|| vectro_lib::quant::pq::pq_encode(black_box(&vecs), black_box(&cb)))
    });
    group.finish();
}

/// INT8 batch decode: allocating `decode_batch` vs in-place `batch_decode_into`.
fn bench_int8_decode(c: &mut Criterion) {
    const N: usize = 1_000;
    const D: usize = 768;
    let vecs = make_vecs(N, D);
    let encoded = int8::encode_batch(&vecs);

    // Flatten codes/scales for the in-place variant.
    let mut codes: Vec<i8> = Vec::with_capacity(N * D);
    let mut scales: Vec<f32> = Vec::with_capacity(N);
    for e in &encoded {
        codes.extend_from_slice(&e.codes);
        scales.push(e.scale);
    }

    let mut group = c.benchmark_group("int8_decode");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("decode_batch_alloc_n1000_d768", |b| {
        b.iter(|| int8::decode_batch(black_box(&encoded)))
    });
    let mut out = vec![0.0f32; N * D];
    group.bench_function("batch_decode_into_n1000_d768", |b| {
        b.iter(|| int8::batch_decode_into(black_box(&codes), black_box(&scales), D, &mut out))
    });
    group.finish();
}

/// HNSW insertion throughput (graph built inside the timed loop).
fn bench_hnsw_insert(c: &mut Criterion) {
    const N: usize = 2_000;
    const D: usize = 64;
    let vecs = make_vecs(N, D);

    let mut group = c.benchmark_group("hnsw_insert");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("add_batch_n2000_d64_m8_ef40", |b| {
        b.iter(|| {
            let mut idx = HnswIndex::new(8, 40);
            idx.add_batch(black_box(&vecs));
            idx
        })
    });
    group.finish();
}

/// IVF-Flat exact-cosine search throughput (WS-A: flat store + prefetch).
/// The candidate scan over probed posting lists is the hot path; at high `d`
/// it is cold-memory bound, where the contiguous layout + software prefetch of
/// the next vector help most.
///
/// Crucially the bench cycles through **many distinct queries** rather than
/// hammering one — a single repeated query keeps its ~`n_probe·cluster` candidate
/// vectors resident in L2/L3, which hides the cold-memory latency the prefetch
/// exists to attack and reduces the bench to pure prefetch instruction overhead.
fn bench_ivf_search(c: &mut Criterion) {
    const N: usize = 20_000;
    const N_LISTS: usize = 256;
    const N_PROBE: usize = 16;
    const N_QUERIES: usize = 512;
    let mut group = c.benchmark_group("ivf_search");
    for &d in &[128usize, 256, 768] {
        let vecs = make_vecs(N, d);
        let mut idx = IvfIndex::new(N_LISTS, N_PROBE);
        idx.train(&vecs, 10, 42).expect("ivf train failed");
        idx.add_batch(&vecs);
        // Distinct queries (offset generator so they differ from stored vectors).
        let queries: Vec<Vec<f32>> = (0..N_QUERIES)
            .map(|i| {
                (0..d)
                    .map(|j| (((i + 7) * d + j) as f32 * 0.0011_f32).cos())
                    .collect()
            })
            .collect();
        let mut qi = 0usize;
        group.bench_function(format!("search_n20k_d{d}_probe16_k10"), |b| {
            b.iter(|| {
                let q = &queries[qi % N_QUERIES];
                qi += 1;
                idx.search_with_probe(black_box(q), 10, N_PROBE)
            })
        });
    }
    group.finish();
}

/// HNSW concurrent build at higher `d` (WS-D: per-vector `prep` allocation
/// removed from the bulk build path; L2/IP no longer copy the raw vector).
fn bench_hnsw_build_highd(c: &mut Criterion) {
    const N: usize = 3_000;
    const D: usize = 256;
    let vecs = make_vecs(N, D);
    let mut group = c.benchmark_group("hnsw_build_highd");
    group.sample_size(20);
    for (label, metric) in [("cosine", Metric::Cosine), ("l2", Metric::L2)] {
        group.bench_function(format!("build_n3k_d256_{label}"), |b| {
            b.iter(|| {
                let mut idx = HnswIndex::with_metric(16, 200, metric);
                idx.add_batch(black_box(&vecs));
                idx
            })
        });
    }
    group.finish();
}

/// NF4 batch-encode throughput (WS-C: SIMD nibble search). The nearest-level
/// search is a 15-way threshold sum, vectorised across lanes on aarch64.
fn bench_nf4_encode(c: &mut Criterion) {
    const N: usize = 2_000;
    let mut group = c.benchmark_group("nf4_encode");
    for &d in &[256usize, 768] {
        let vecs = make_vecs(N, d);
        group.throughput(Throughput::Elements((N * d) as u64));
        group.bench_function(format!("encode_batch_n2k_d{d}"), |b| {
            b.iter(|| nf4::encode_batch(black_box(&vecs)))
        });
    }
    group.finish();
}

/// IVF-PQ search throughput — the query-time ADC scan path.
///
/// Exercises the flat `pq_codes` layout + bounded top-k selection. The dataset is
/// large enough (per-list ≈ N/n_lists, scanned over n_probe lists) that the ADC
/// gather and the top-k step both dominate, so a regression here flags either the
/// cache-locality or the selection change.
fn bench_ivfpq_search(c: &mut Criterion) {
    const N: usize = 20_000;
    const D: usize = 128;
    const N_LISTS: usize = 256;
    const N_PROBE: usize = 16;
    let vecs = make_vecs(N, D);

    let mut idx = IvfPqIndex::new(N_LISTS, N_PROBE);
    idx.train(&vecs, 8, 256, 10, 42)
        .expect("ivfpq train failed");
    for v in &vecs {
        idx.add(v);
    }
    // A fixed batch of queries drawn from the dataset (deterministic).
    let queries: Vec<&Vec<f32>> = (0..64).map(|i| &vecs[i * 137 % N]).collect();

    let mut group = c.benchmark_group("ivfpq_search");
    group.throughput(Throughput::Elements(queries.len() as u64));
    group.bench_function("search_n20k_d128_lists256_probe16_k10", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(idx.search_with_probe(black_box(q), 10, N_PROBE));
            }
        })
    });
    group.finish();
}

/// IVF-PQ batch search throughput — the `search_batch_flat` path used by the
/// Python/FFI batch API. Exercises the batched coarse scan + parallel ADC. d=768
/// so the coarse-quantizer dots (q × n_lists full-dim) dominate, matching the
/// realistic embedding regime where the GEMM coarse scan pays off.
fn bench_ivfpq_search_batch(c: &mut Criterion) {
    const N: usize = 20_000;
    const D: usize = 768;
    const N_LISTS: usize = 1024;
    const N_PROBE: usize = 32;
    const Q: usize = 256;
    let vecs = make_vecs(N, D);

    let mut idx = IvfPqIndex::new(N_LISTS, N_PROBE);
    idx.train(&vecs, 64, 256, 10, 42)
        .expect("ivfpq train failed");
    for v in &vecs {
        idx.add(v);
    }
    // Flat query buffer drawn deterministically from the dataset.
    let mut flat: Vec<f32> = Vec::with_capacity(Q * D);
    for i in 0..Q {
        flat.extend_from_slice(&vecs[i * 137 % N]);
    }

    let mut group = c.benchmark_group("ivfpq_search_batch");
    group.throughput(Throughput::Elements(Q as u64));
    group.bench_function("batch_q256_n20k_d768_lists1024_probe32_k10", |b| {
        b.iter(|| black_box(idx.search_batch_flat(black_box(&flat), D, 10, N_PROBE)))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_nf4_encode,
    bench_pq_train,
    bench_rq_train,
    bench_rq_decode,
    bench_rq_encode,
    bench_pq_encode,
    bench_int8_decode,
    bench_hnsw_insert,
    bench_ivf_search,
    bench_hnsw_build_highd,
    bench_ivfpq_search,
    bench_ivfpq_search_batch
);
criterion_main!(benches);
