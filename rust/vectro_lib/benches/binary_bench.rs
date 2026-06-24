//! Binary (sign-bit) encode benchmarks: scalar vs SIMD-fast path.
//!
//! Run with:
//!   cargo bench -p vectro_lib --bench binary_bench
//!
//! `encode_fast` dispatches AVX-512F (16 signs → 2-byte mask per `vcmpps`) →
//! AVX2 (`vmovmskps`, 8 signs → 1 byte) → scalar. Throughput in elements/second.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rayon::prelude::*;
use vectro_lib::quant::binary::{encode_batch, BinaryVector};

fn make_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| (0..d).map(|j| ((i * d + j) as f32 * 0.0013_f32).sin()).collect())
        .collect()
}

/// Single-vector scalar vs SIMD-fast sign-pack (d=768).
fn bench_single_vec(c: &mut Criterion) {
    const D: usize = 768;
    let v: Vec<f32> = (0..D).map(|i| (i as f32 * 0.007).sin() - 0.1).collect();

    let mut group = c.benchmark_group("binary_single_d768");
    group.throughput(Throughput::Elements(D as u64));
    group.bench_function("encode_scalar", |b| b.iter(|| BinaryVector::encode(black_box(&v), true)));
    group.bench_function("encode_fast", |b| b.iter(|| BinaryVector::encode_fast(black_box(&v), true)));
    group.finish();
}

/// Batch sign-pack at scale (d=768).
fn bench_batch(c: &mut Criterion) {
    const D: usize = 768;
    for &n in &[1_000usize, 10_000] {
        let vecs = make_vecs(n, D);
        let mut group = c.benchmark_group(format!("binary_batch_n{n}_d{D}"));
        group.throughput(Throughput::Elements((n * D) as u64));
        // Scalar baseline (parallel map of the canonical `encode`) vs the
        // SIMD-dispatching `encode_batch` — same parallelism, isolating the kernel.
        group.bench_function("encode_batch_scalar", |b| {
            b.iter(|| {
                black_box(&vecs)
                    .par_iter()
                    .map(|v| BinaryVector::encode(v, true))
                    .collect::<Vec<_>>()
            })
        });
        group.bench_function("encode_batch", |b| b.iter(|| encode_batch(black_box(&vecs), true)));
        group.finish();
    }
}

criterion_group!(benches, bench_single_vec, bench_batch);
criterion_main!(benches);
