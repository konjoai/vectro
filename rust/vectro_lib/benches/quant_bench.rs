use criterion::{criterion_group, criterion_main, Criterion};
use vectro_lib::{
    search::{QuantizedIndex, SearchIndex},
    Embedding,
};

// synthetic dataset generator
fn make_dataset(n: usize, dim: usize) -> Vec<Embedding> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            // simple deterministic values
            v.push(((i + d) % 100) as f32 / 100.0);
        }
        out.push(Embedding::new(format!("id_{}", i), v));
    }
    out
}

fn bench_search(c: &mut Criterion) {
    let ds = make_dataset(1000, 64);
    let query = ds[0].vector.clone();

    let float_idx = SearchIndex::from_dataset(&ds);
    let mut qidx = QuantizedIndex::from_dataset(&ds);

    c.bench_function("float_topk", |b| {
        b.iter(|| {
            let _ = float_idx.top_k(&query, 10);
        })
    });

    c.bench_function("quant_topk_on_the_fly", |b| {
        b.iter(|| {
            let _ = qidx.top_k(&query, 10);
        })
    });

    qidx.precompute_normalized();
    c.bench_function("quant_topk_precomputed", |b| {
        b.iter(|| {
            let _ = qidx.top_k(&query, 10);
        })
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
