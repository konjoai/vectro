//! bf16-navigation + fp32-rerank HNSW: same graph, measured exact (fp32) vs
//! nav-on. Verifies recall-neutrality and the QPS gain at d=768.
//!
//!   cargo run --release --example hnsw_fp32_bench

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
    s.select_nth_unstable_by(k - 1, |a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    }); // NaN-safe: treat NaN as equal, don't panic a bench run
    s.truncate(k);
    s.into_iter().map(|(_, i)| i).collect()
}

fn main() {
    let (n, d, m, ef_c) = (200_000usize, 768usize, 16usize, 200usize);
    let nq = 3000usize;
    let n_gt = 300usize;
    let k = 10usize;

    let vecs = make_vecs(n, d, 1);
    let queries = make_vecs(nq, d, 999);
    let gt: Vec<HashSet<usize>> = queries[..n_gt]
        .iter()
        .map(|q| brute_gt(&vecs, q, k))
        .collect();

    let t = Instant::now();
    let mut idx = HnswIndex::new(m, ef_c);
    idx.add_batch(&vecs);
    println!("build: {:.1}s", t.elapsed().as_secs_f64());

    let recall = |idx: &HnswIndex, ef: usize| -> f64 {
        let mut tot = 0usize;
        for (q, g) in queries[..n_gt].iter().zip(&gt) {
            tot += idx
                .search(q, k, ef)
                .iter()
                .filter(|(id, _)| g.contains(id))
                .count();
        }
        tot as f64 / (n_gt * k) as f64
    };
    let qps = |idx: &HnswIndex, ef: usize| -> f64 {
        for q in &queries {
            std::hint::black_box(idx.search(q, k, ef).len());
        }
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let mut s = 0usize;
            for q in &queries {
                s += idx.search(q, k, ef).len();
            }
            std::hint::black_box(s);
            best = best.min(t.elapsed().as_secs_f64());
        }
        nq as f64 / best
    };

    let efs = [64usize, 100, 160];
    // Exact fp32 (nav off) across all ef on this graph.
    let fp32: Vec<(f64, f64)> = efs
        .iter()
        .map(|&ef| (recall(&idx, ef), qps(&idx, ef)))
        .collect();
    // Enable bf16 nav on the SAME graph, then measure.
    idx.enable_bf16_nav();
    let nav: Vec<(f64, f64)> = efs
        .iter()
        .map(|&ef| (recall(&idx, ef), qps(&idx, ef)))
        .collect();

    for (i, &ef) in efs.iter().enumerate() {
        let (rf, qf) = fp32[i];
        let (rn, qn) = nav[i];
        println!(
            "ef={ef:>3}: fp32 R@{k}={rf:.4} {qf:.0}qps | bf16-nav R@{k}={rn:.4} {qn:.0}qps | {:.2}x QPS, dR={:+.4}",
            qn / qf,
            rn - rf
        );
    }
}
