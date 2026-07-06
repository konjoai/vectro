//! Kill-test for `VECTRO_OPTIMIZATION_AUDIT_2026-07.md` item 3.2 (graph
//! reordering for cache locality): builds an HNSW index at n=200,000/d=768
//! (~614 MiB vector store, past this host's 260 MiB L3 — the same >L3 regime
//! `PERF_FINDINGS.md`'s prefetch results were measured in), verifies
//! `reorder_for_locality` doesn't change recall, then compares single-query
//! and batch QPS before/after.
//!
//!   cargo run --release --example hnsw_reorder_bench

use std::collections::HashSet;
use std::time::Instant;
use vectro_lib::index::hnsw::HnswIndex;

fn make_vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as f32 / (1u64 << 31) as f32 - 1.0
    };
    let n_centers = 64usize;
    let centers: Vec<Vec<f32>> = (0..n_centers)
        .map(|_| (0..d).map(|_| next()).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % n_centers];
            let v: Vec<f32> = c.iter().map(|&x| x + 0.9 * next()).collect();
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            v.iter().map(|x| x / nrm).collect()
        })
        .collect()
}

fn brute_gt(vecs: &[Vec<f32>], q: &[f32], k: usize) -> HashSet<usize> {
    let mut s: Vec<(f32, usize)> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (-q.iter().zip(v).map(|(a, b)| a * b).sum::<f32>(), i))
        .collect();
    s.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
    s.truncate(k);
    s.into_iter().map(|(_, i)| i).collect()
}

/// Best-of-5 wall-clock QPS for single-query `search`, after a warmup pass.
fn qps_single(idx: &HnswIndex, queries: &[Vec<f32>], k: usize, ef: usize) -> f64 {
    for q in queries {
        std::hint::black_box(idx.search(q, k, ef).len());
    }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let mut s = 0usize;
        for q in queries {
            s += idx.search(q, k, ef).len();
        }
        std::hint::black_box(s);
        best = best.min(t.elapsed().as_secs_f64());
    }
    queries.len() as f64 / best
}

/// Best-of-5 wall-clock QPS for `search_batch_flat` (rayon-parallel across queries).
fn qps_batch(idx: &HnswIndex, flat: &[f32], d: usize, q: usize, k: usize, ef: usize) -> f64 {
    std::hint::black_box(idx.search_batch_flat(flat, d, k, ef).len());
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let r = idx.search_batch_flat(flat, d, k, ef);
        std::hint::black_box(r.len());
        best = best.min(t.elapsed().as_secs_f64());
    }
    q as f64 / best
}

fn recall(
    idx: &HnswIndex,
    queries: &[Vec<f32>],
    gt: &[HashSet<usize>],
    k: usize,
    ef: usize,
) -> f64 {
    let mut tot = 0usize;
    for (q, g) in queries.iter().zip(gt) {
        tot += idx
            .search(q, k, ef)
            .iter()
            .filter(|(id, _)| g.contains(id))
            .count();
    }
    tot as f64 / (gt.len() * k) as f64
}

fn main() {
    let (n, d, m, ef_c) = (200_000usize, 768usize, 16usize, 200usize);
    let nq = 3000usize;
    let n_gt = 300usize;
    let k = 10usize;
    let ef = 64usize;

    let vecs = make_vecs(n, d, 1);
    let queries = make_vecs(nq, d, 999);
    let flat: Vec<f32> = queries.iter().flatten().copied().collect();
    let gt: Vec<HashSet<usize>> = queries[..n_gt]
        .iter()
        .map(|q| brute_gt(&vecs, q, k))
        .collect();

    let t = Instant::now();
    let mut idx = HnswIndex::new(m, ef_c);
    idx.add_batch(&vecs);
    println!(
        "build: {:.1}s ({n} vectors, d={d}, ~{:.0} MiB)",
        t.elapsed().as_secs_f64(),
        (n * d * 4) as f64 / (1 << 20) as f64
    );

    let r_before = recall(&idx, &queries[..n_gt], &gt, k, ef);
    let single_before = qps_single(&idx, &queries, k, ef);
    let batch_before = qps_batch(&idx, &flat, d, nq, k, ef);
    println!("before reorder: R@{k}={r_before:.4}  single={single_before:.0}qps  batch={batch_before:.0}qps");

    let t = Instant::now();
    let new_to_old = idx.reorder_for_locality();
    println!("reorder: {:.2}s", t.elapsed().as_secs_f64());

    // Ground truth is in old-id space; translate old ids to new before scoring.
    let mut old_to_new = vec![0usize; n];
    for (new_id, &old_id) in new_to_old.iter().enumerate() {
        old_to_new[old_id] = new_id;
    }
    let gt_new: Vec<HashSet<usize>> = gt
        .iter()
        .map(|g| g.iter().map(|&old_id| old_to_new[old_id]).collect())
        .collect();

    let r_after = recall(&idx, &queries[..n_gt], &gt_new, k, ef);
    let single_after = qps_single(&idx, &queries, k, ef);
    let batch_after = qps_batch(&idx, &flat, d, nq, k, ef);
    println!("after reorder:  R@{k}={r_after:.4}  single={single_after:.0}qps  batch={batch_after:.0}qps");

    println!(
        "\ndR@{k}={:+.4}  single {:.2}x  batch {:.2}x",
        r_after - r_before,
        single_after / single_before,
        batch_after / batch_before
    );
}
