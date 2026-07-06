//! Konjo "prove" gate activation artifact for `reorder_for_locality`
//! (VECTRO_OPTIMIZATION_AUDIT_2026-07.md item 3.2, PR #110).
//!
//! Builds ONE HNSW index (n=200,000, d=768 — past this host's 260 MiB L3),
//! clones it, reorders the clone, then takes interleaved raw single-pass
//! QPS measurements on both (never best-of-N — the "prove" gate wants the
//! real run-to-run noise floor, not a variance-minimized number). Emits a
//! paired-samples JSON to stdout, meant to be fed into the pinned `kiban`
//! package's real Wilcoxon test (`lib.prove.paired_wilcoxon` + `verdict`)
//! to get a legitimate MERGE/NOISE/REGRESSION verdict — see
//! `OPTIMIZATION_OPPORTUNITIES.md`'s "Campaign 4" entry for the exact
//! Python invocation used to produce this repo's recorded verdict.
//!
//!   cargo run --release --example hnsw_reorder_prove > \
//!     benchmarks/results/prove_hnsw_reorder_<timestamp>.json

use std::time::{Instant, SystemTime, UNIX_EPOCH};
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

/// One raw (not best-of-N) full pass over `queries`, returning elapsed seconds.
fn raw_pass_secs(idx: &HnswIndex, queries: &[Vec<f32>], k: usize, ef: usize) -> f64 {
    let t = Instant::now();
    let mut s = 0usize;
    for q in queries {
        s += idx.search(q, k, ef).len();
    }
    std::hint::black_box(s);
    t.elapsed().as_secs_f64()
}

fn read_thermal_millideg() -> Option<i64> {
    let mut best: Option<i64> = None;
    for entry in std::fs::read_dir("/sys/class/thermal").ok()?.flatten() {
        let path = entry.path().join("temp");
        if let Ok(s) = std::fs::read_to_string(&path) {
            if let Ok(v) = s.trim().parse::<i64>() {
                best = Some(best.map_or(v, |b: i64| b.max(v)));
            }
        }
    }
    best
}

fn main() {
    let (n, d, m, ef_c) = (200_000usize, 768usize, 16usize, 200usize);
    let nq = 3000usize;
    let k = 10usize;
    let ef = 64usize;
    const N_PAIRS: usize = 35;

    let vecs = make_vecs(n, d, 1);
    let queries = make_vecs(nq, d, 999);

    let thermal_before = read_thermal_millideg();
    let build_start = Instant::now();
    let mut idx_a = HnswIndex::new(m, ef_c);
    idx_a.add_batch(&vecs);
    let build_secs = build_start.elapsed().as_secs_f64();

    let mut idx_b = idx_a.clone();
    let (reorder_secs, new_to_old) = {
        let t = Instant::now();
        let map = idx_b.reorder_for_locality();
        (t.elapsed().as_secs_f64(), map)
    };

    // Warmup both indexes (page faults, branch predictor, allocator) before timing.
    raw_pass_secs(&idx_a, &queries, k, ef);
    raw_pass_secs(&idx_b, &queries, k, ef);

    let mut baseline_qps = Vec::with_capacity(N_PAIRS);
    let mut candidate_qps = Vec::with_capacity(N_PAIRS);
    for i in 0..N_PAIRS {
        // Alternate measurement order each pair to cancel monotonic drift
        // (thermal ramp, frequency scaling) rather than let it bias one side.
        let (b_secs, c_secs) = if i % 2 == 0 {
            let b = raw_pass_secs(&idx_a, &queries, k, ef);
            let c = raw_pass_secs(&idx_b, &queries, k, ef);
            (b, c)
        } else {
            let c = raw_pass_secs(&idx_b, &queries, k, ef);
            let b = raw_pass_secs(&idx_a, &queries, k, ef);
            (b, c)
        };
        baseline_qps.push(nq as f64 / b_secs);
        candidate_qps.push(nq as f64 / c_secs);
    }

    let thermal_after = read_thermal_millideg();
    let unix_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Recall sanity: reorder must be a pure relabel, so confirm zero drift on a
    // ground-truth sample before trusting the QPS-only comparison above.
    // `after`'s ids are in idx_b's (new) numbering; new_to_old[new_id] is the
    // matching idx_a (old) id, which is what `before` is keyed by.
    let n_gt = 300usize;
    let mut hits = 0usize;
    for q in &queries[..n_gt] {
        let before: std::collections::HashSet<usize> =
            idx_a.search(q, k, ef).iter().map(|&(id, _)| id).collect();
        let after = idx_b.search(q, k, ef);
        hits += after
            .iter()
            .filter(|(id, _)| before.contains(&new_to_old[*id]))
            .count();
    }
    let recall_overlap = hits as f64 / (n_gt * k) as f64;

    let to_json_arr = |xs: &[f64]| -> String {
        xs.iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    };

    println!("{{");
    println!("  \"artifact\": \"prove_hnsw_reorder\",");
    println!("  \"change\": \"reorder_for_locality (VECTRO_OPTIMIZATION_AUDIT_2026-07.md item 3.2, PR #110)\",");
    println!("  \"unix_ts\": {unix_ts},");
    println!("  \"n_vectors\": {n},");
    println!("  \"dim\": {d},");
    println!("  \"m\": {m},");
    println!("  \"ef_construction\": {ef_c},");
    println!("  \"k\": {k},");
    println!("  \"ef\": {ef},");
    println!("  \"n_queries\": {nq},");
    println!("  \"n_pairs\": {N_PAIRS},");
    println!("  \"build_secs\": {build_secs:.3},");
    println!("  \"reorder_secs\": {reorder_secs:.3},");
    println!("  \"recall_overlap_after_vs_before\": {recall_overlap:.6},");
    println!(
        "  \"thermal_before_millideg\": {},",
        thermal_before
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".into())
    );
    println!(
        "  \"thermal_after_millideg\": {},",
        thermal_after
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".into())
    );
    println!("  \"metric\": \"qps\",");
    println!("  \"lower_is_better\": false,");
    println!("  \"baseline_qps\": [{}],", to_json_arr(&baseline_qps));
    println!("  \"candidate_qps\": [{}]", to_json_arr(&candidate_qps));
    println!("}}");
}
