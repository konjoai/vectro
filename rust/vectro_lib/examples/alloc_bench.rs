//! Allocator A/B bench for the alloc-heavy HNSW build + concurrent-query paths.
//!
//! Build/run both ways and compare:
//!   cargo run --release --example alloc_bench                 # glibc malloc
//!   cargo run --release --example alloc_bench --features mimalloc
//!
//! The concurrent build allocates per-node neighbor `Vec`s and per-expansion
//! snapshots across rayon workers; the concurrent query phase allocates the
//! per-query beam heaps. Both are where a sharded per-thread allocator
//! (mimalloc) can beat glibc's arena locking on multi-core hosts.

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use rayon::prelude::*;
use std::time::Instant;
use vectro_lib::index::hnsw::HnswIndex;

fn make_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| (0..d).map(|j| ((i * d + j) as f32 * 0.0013).sin()).collect())
        .collect()
}

fn main() {
    let alloc = if cfg!(feature = "mimalloc") { "mimalloc" } else { "system" };
    let (n, d, m, ef_c) = (200_000usize, 128usize, 16usize, 200usize);
    let nq = 50_000usize;
    let (k, ef) = (10usize, 64usize);

    let vecs = make_vecs(n, d);
    let queries = make_vecs(nq, d);

    // ── Build (concurrent) ──────────────────────────────────────────────────
    let mut best_build = f64::INFINITY;
    for _ in 0..3 {
        let mut idx = HnswIndex::new(m, ef_c);
        let t = Instant::now();
        idx.add_batch(&vecs);
        best_build = best_build.min(t.elapsed().as_secs_f64());
        std::hint::black_box(idx.len());
    }

    // ── Concurrent query throughput ─────────────────────────────────────────
    let mut idx = HnswIndex::new(m, ef_c);
    idx.add_batch(&vecs);
    let mut best_qps = 0.0f64;
    for _ in 0..3 {
        let t = Instant::now();
        let sink: usize = queries
            .par_iter()
            .map(|q| idx.search(q, k, ef).len())
            .sum();
        let secs = t.elapsed().as_secs_f64();
        std::hint::black_box(sink);
        best_qps = best_qps.max(nq as f64 / secs);
    }

    println!(
        "[{alloc}] build n={n} d={d} m={m} ef_c={ef_c}: {best_build:.3}s  |  concurrent query nq={nq} k={k} ef={ef}: {:.0} qps",
        best_qps
    );
}
