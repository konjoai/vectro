//! Fused single-pass vs two-pass INT8 encode benchmarks (Wave 2).
//!
//! Run with:
//!   cargo bench -p vectro_lib --bench int8_fused_bench
//!
//! Compares `batch_encode_into` (two-pass: abs-max scan + quantise) against
//! `encode_fast_fused_into` row-by-row (single touch per row).  Both call
//! `encode_normalized_into` for an L2-normalised input baseline.
//!
//! If fused is ≥ 20 % faster on n=100k × d=768, promote it to the default
//! path inside `encode_fast_into` (per Wave 2 spec).

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use vectro_lib::quant::int8::{
    batch_encode_into, batch_encode_normalized_into, encode_fast_fused_into,
};

fn make_unit_vectors(n: usize, d: usize) -> Vec<f32> {
    let mut buf = vec![0.0_f32; n * d];
    for i in 0..n {
        // Simple deterministic L2-normalised pattern: sin(j) + small i drift,
        // then divide by L2 norm.
        let mut row = vec![0.0_f32; d];
        for j in 0..d {
            row[j] = ((i + j) as f32 * 0.013_f32).sin();
        }
        let n2: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for (j, x) in row.iter().enumerate() {
            buf[i * d + j] = x / n2;
        }
    }
    buf
}

fn bench_fused_vs_two_pass(c: &mut Criterion) {
    const N: usize = 100_000;
    const D: usize = 768;

    let input = make_unit_vectors(N, D);
    let mut codes = vec![0i8; N * D];
    let mut scales = vec![0.0_f32; N];

    let mut group = c.benchmark_group("int8_n100k_d768");
    group.throughput(Throughput::Elements((N * D) as u64));

    group.bench_function("two_pass_batch", |b| {
        b.iter(|| {
            batch_encode_into(black_box(&input), N, D, &mut codes, &mut scales);
        })
    });

    group.bench_function("fused_per_row", |b| {
        b.iter(|| {
            for i in 0..N {
                let row = &input[i * D..(i + 1) * D];
                let out = &mut codes[i * D..(i + 1) * D];
                let s = encode_fast_fused_into(row, out);
                scales[i] = s / 127.0;
            }
        })
    });

    group.bench_function("normalized_batch", |b| {
        b.iter(|| {
            batch_encode_normalized_into(black_box(&input), N, D, &mut codes, &mut scales);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_fused_vs_two_pass);
criterion_main!(benches);
