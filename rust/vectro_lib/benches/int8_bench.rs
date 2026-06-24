//! INT8 encode benchmarks: scalar path vs SIMD-fast path.
//!
//! Run with:
//!   cargo bench -p vectro_lib --bench int8_bench
//!
//! Throughput is reported in elements/second; derive vec/s by dividing by D.
//! At D=768: 1 Gelem/s ≈ 1.3M vec/s.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use vectro_lib::index::quant_hnsw::Int8HnswIndex;
use vectro_lib::quant::int8::{encode_batch, Int8Vector};

fn make_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| (0..d).map(|j| ((i * d + j) as f32 * 0.0013_f32).sin()).collect())
        .collect()
}

/// Single-vector scalar vs SIMD comparison (d=768 per vector).
fn bench_single_vec(c: &mut Criterion) {
    const D: usize = 768;
    let v: Vec<f32> = (0..D).map(|i| (i as f32 * 0.007).sin()).collect();

    let mut group = c.benchmark_group("int8_single_d768");
    group.throughput(Throughput::Elements(D as u64));

    group.bench_function("encode_scalar", |b| {
        b.iter(|| Int8Vector::encode(black_box(&v)))
    });
    group.bench_function("encode_fast", |b| {
        b.iter(|| Int8Vector::encode_fast(black_box(&v)))
    });

    group.finish();
}

/// Batch encode at various scales (d=768).
fn bench_batch(c: &mut Criterion) {
    const D: usize = 768;

    for &n in &[100usize, 1_000, 10_000] {
        let vecs = make_vecs(n, D);

        let mut group = c.benchmark_group(format!("int8_batch_n{n}_d{D}"));
        group.throughput(Throughput::Elements((n * D) as u64));

        group.bench_function("encode_batch", |b| {
            b.iter(|| encode_batch(black_box(&vecs)))
        });

        group.finish();
    }
}

/// i8×f32 `dot_query` — the per-candidate distance kernel in INT8-HNSW search.
fn bench_dot_query(c: &mut Criterion) {
    const D: usize = 768;
    let v: Vec<f32> = (0..D).map(|i| (i as f32 * 0.007).sin()).collect();
    let q: Vec<f32> = (0..D).map(|i| (i as f32 * 0.011).cos()).collect();
    let enc = Int8Vector::encode_fast(&v);

    let mut group = c.benchmark_group("int8_dot_query_d768");
    group.throughput(Throughput::Elements(D as u64));
    group.bench_function("dot_query", |b| b.iter(|| enc.dot_query(black_box(&q))));
    group.finish();
}

/// INT8-HNSW search — the real consumer of `dot_query`. Exercises the kernel in
/// the graph-traversal hot loop (distance per visited candidate), so the e2e gain
/// reflects the distance-computation fraction of search time, not the kernel alone.
fn bench_int8_hnsw_search(c: &mut Criterion) {
    const N: usize = 20_000;
    const D: usize = 768;
    let vecs: Vec<Vec<f32>> = (0..N)
        .map(|i| (0..D).map(|j| ((i * D + j) as f32 * 0.0013_f32).sin()).collect())
        .collect();
    let mut idx = Int8HnswIndex::new(16, 100);
    idx.add_batch(&vecs);
    let queries: Vec<&Vec<f32>> = (0..64).map(|i| &vecs[i * 137 % N]).collect();

    let mut group = c.benchmark_group("int8_hnsw_search_d768");
    group.throughput(Throughput::Elements(queries.len() as u64));
    group.bench_function("search_n20k_m16_ef100_k10", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(idx.search(black_box(q), 10, 100));
            }
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_single_vec,
    bench_batch,
    bench_dot_query,
    bench_int8_hnsw_search
);
criterion_main!(benches);
