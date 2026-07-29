//! Algorithm throughput benchmarks: INT8, NF4, HNSW.
//!
//! Run with:
//!   cargo bench -p vectro_lib --bench simd_bench
//!
//! Throughput is measured in elements/second.  Divide by D to get vec/s.
//! Phase-17 targets (PLAN.md):
//!   INT8 encode: ≥ 12M vec/s @ n=100K, d=768 (≈ 9.2 Gelem/s)
//!   NF4 encode:  ≥  2M vec/s @ d=768
//!   HNSW recall@10: reported by `recall_at_k_bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use vectro_lib::index::hnsw::HnswIndex;
use vectro_lib::index::ivf::IvfIndex;
use vectro_lib::index::ivf_pq::IvfPqIndex;
use vectro_lib::index::quant_hnsw::{Bf16HnswIndex, Int8HnswIndex, Nf4HnswIndex};
use vectro_lib::quant::sq2::Sq2Vector;
use vectro_lib::quant::sq3::Sq3Vector;
use vectro_lib::quant::{int8, nf4};

fn make_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..d)
                .map(|j| ((i * d + j) as f32 * 0.0013_f32).sin())
                .collect()
        })
        .collect()
}

/// INT8 encode throughput at benchmark scale.
fn bench_int8_throughput(c: &mut Criterion) {
    const N: usize = 1_000;
    const D: usize = 768;
    let vecs = make_vecs(N, D);

    let mut group = c.benchmark_group("int8_throughput");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("encode_batch_n1000_d768", |b| {
        b.iter(|| int8::encode_batch(black_box(&vecs)))
    });
    group.finish();
}

/// NF4 encode throughput.
fn bench_nf4_throughput(c: &mut Criterion) {
    const N: usize = 1_000;
    const D: usize = 768;
    let vecs = make_vecs(N, D);

    let mut group = c.benchmark_group("nf4_throughput");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("encode_batch_n1000_d768", |b| {
        b.iter(|| nf4::encode_batch(black_box(&vecs)))
    });
    group.finish();
}

/// HNSW search throughput (index pre-built outside the timed loop).
fn bench_hnsw_search(c: &mut Criterion) {
    const N: usize = 2_000;
    const D: usize = 64;
    let vecs = make_vecs(N, D);
    let query = vecs[0].clone();

    let mut idx = HnswIndex::new(8, 40);
    idx.add_batch(&vecs);

    let mut group = c.benchmark_group("hnsw_search");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("search_k10_ef50_n2000_d64", |b| {
        b.iter(|| idx.search(black_box(&query), 10, 50))
    });
    group.finish();
}

/// INT8 quant-HNSW asymmetric search throughput at an embedding-scale dimension.
/// Exercises the VNNI prepared-query distance kernel (d=768 ⇒ AVX-512-VNNI path
/// on capable hosts), the flagship INT8 mode's per-candidate hot path.
fn bench_int8_hnsw_search(c: &mut Criterion) {
    const N: usize = 5_000;
    const D: usize = 768;
    let vecs = make_vecs(N, D);
    let query = vecs[1].clone();

    let mut idx = Int8HnswIndex::new(16, 200);
    idx.add_batch(&vecs);
    idx.finalize();

    let mut group = c.benchmark_group("int8_hnsw_search");
    group.throughput(Throughput::Elements(1));
    group.bench_function("search_k10_ef100_n5000_d768", |b| {
        b.iter(|| idx.search(black_box(&query), 10, 100))
    });
    group.finish();
}

/// NF4 quant-HNSW asymmetric search throughput. Exercises the AVX2 in-register
/// codebook-LUT distance kernel (the former scalar hot path).
fn bench_nf4_hnsw_search(c: &mut Criterion) {
    const N: usize = 5_000;
    const D: usize = 768;
    let vecs = make_vecs(N, D);
    let query = vecs[1].clone();

    let mut idx = Nf4HnswIndex::new(16, 200);
    idx.add_batch(&vecs);
    idx.finalize();

    let mut group = c.benchmark_group("nf4_hnsw_search");
    group.throughput(Throughput::Elements(1));
    group.bench_function("search_k10_ef100_n5000_d768", |b| {
        b.iter(|| idx.search(black_box(&query), 10, 100))
    });
    group.finish();
}

/// BF16 quant-HNSW asymmetric search throughput. Exercises the bf16→f32 widen
/// distance kernel (AVX-512 16-wide on capable hosts, AVX2 8-wide otherwise).
fn bench_bf16_hnsw_search(c: &mut Criterion) {
    const N: usize = 5_000;
    const D: usize = 768;
    let vecs = make_vecs(N, D);
    let query = vecs[1].clone();

    let mut idx = Bf16HnswIndex::new(16, 200);
    idx.add_batch(&vecs);
    idx.finalize();

    let mut group = c.benchmark_group("bf16_hnsw_search");
    group.throughput(Throughput::Elements(1));
    group.bench_function("search_k10_ef100_n5000_d768", |b| {
        b.iter(|| idx.search(black_box(&query), 10, 100))
    });
    group.finish();
}

/// IVF-Flat search throughput (index trained and populated outside the timed loop).
fn bench_ivf_search(c: &mut Criterion) {
    const N: usize = 10_000;
    const D: usize = 128;
    let vecs = make_vecs(N, D);
    let query = vecs[0].clone();

    let mut idx = IvfIndex::new(64, 8);
    idx.train(&vecs, 10, 42).expect("IvfIndex train failed");
    idx.add_batch(&vecs);

    let mut group = c.benchmark_group("ivf_search");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("search_k10_n10k_d128", |b| {
        b.iter(|| idx.search(black_box(&query), 10))
    });
    group.finish();
}

/// IVF-PQ ADC search throughput (index trained and populated outside the timed loop).
fn bench_ivfpq_search(c: &mut Criterion) {
    const N: usize = 10_000;
    const D: usize = 128;
    let vecs = make_vecs(N, D);
    let query = vecs[0].clone();

    let mut idx = IvfPqIndex::new(64, 8);
    // D=128, M=8 sub-spaces → sub_dim=16; 64 PQ centroids per sub-space.
    idx.train(&vecs, 8, 64, 10, 42)
        .expect("IvfPqIndex train failed");
    for v in &vecs {
        idx.add(v);
    }

    let mut group = c.benchmark_group("ivfpq_search");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("adc_k10_n10k_d128", |b| {
        b.iter(|| idx.search(black_box(&query), 10))
    });
    group.finish();
}

/// SQ2 batch decode throughput (LUT-based reconstruction).
fn bench_sq2_decode(c: &mut Criterion) {
    const N: usize = 1_000;
    const D: usize = 768;
    let vecs = make_vecs(N, D);
    let encoded: Vec<Sq2Vector> = vecs.iter().map(|v| Sq2Vector::encode(v)).collect();

    let mut group = c.benchmark_group("sq2_decode");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("decode_n1000_d768", |b| {
        b.iter(|| {
            for e in &encoded {
                black_box(e.decode());
            }
        })
    });
    group.finish();
}

/// SQ3 batch decode throughput (LUT-based reconstruction).
fn bench_sq3_decode(c: &mut Criterion) {
    const N: usize = 1_000;
    const D: usize = 768;
    let vecs = make_vecs(N, D);
    let encoded: Vec<Sq3Vector> = vecs.iter().map(|v| Sq3Vector::encode(v)).collect();

    let mut group = c.benchmark_group("sq3_decode");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("decode_n1000_d768", |b| {
        b.iter(|| {
            for e in &encoded {
                black_box(e.decode());
            }
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_int8_throughput,
    bench_nf4_throughput,
    bench_hnsw_search,
    bench_int8_hnsw_search,
    bench_nf4_hnsw_search,
    bench_bf16_hnsw_search,
    bench_ivf_search,
    bench_ivfpq_search,
    bench_sq2_decode,
    bench_sq3_decode
);
criterion_main!(benches);
