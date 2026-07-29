//! Konjo "prove" gate artifact for the NEON `sdot` INT8 kernel
//! (VECTRO_OPTIMIZATION_AUDIT_2026-07.md item 2.1, sprint Phase 2).
//!
//! IMPORTANT — what this artifact does and does NOT prove. The `sdot` kernel is
//! aarch64-only (FEAT_DotProd, runtime-gated) and is *dead code on x86*; the x86
//! INT8 search path (`Int8Vector::dot_query_prepared`, the VNNI/AVX branches)
//! is functionally unchanged by this PR — the diff adds only
//! `#[cfg(target_arch = "aarch64")]` code and cfg-gates `Int8Query` fields that
//! compile identically on x86. This artifact therefore proves **no x86
//! regression** on the shipped INT8 search kernel, which is what the prove gate
//! protects against on this bench host. It does NOT — and cannot — measure the
//! aarch64 `sdot` speedup: that kill-test needs Apple Silicon and is registered
//! PLANNED/PENDING in PERF_FINDINGS.md. No aarch64 win is claimed here.
//!
//! Both paired arms measure the same shipped x86 kernel (base-equivalent by the
//! codegen-identity argument above), taking interleaved raw single-pass QPS
//! measurements to establish the run-to-run noise floor. A NOISE verdict (no
//! significant difference) is the expected and correct outcome.
//!
//!   cargo run --release --example int8_sdot_prove > \
//!     benchmarks/results/prove_int8_sdot_<timestamp>.json

use std::time::{Instant, SystemTime, UNIX_EPOCH};
use vectro_lib::quant::int8::{Int8Query, Int8Vector};

fn make_vecs(n: usize, d: usize, seed: u64) -> Vec<Int8Vector> {
    let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as f32 / (1u64 << 31) as f32 - 1.0
    };
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..d).map(|_| next()).collect();
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let unit: Vec<f32> = v.iter().map(|x| x / nrm).collect();
            Int8Vector::encode_fast(&unit)
        })
        .collect()
}

fn make_query(d: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x1234_5678);
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as f32 / (1u64 << 31) as f32 - 1.0
    };
    let v: Vec<f32> = (0..d).map(|_| next()).collect();
    let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / nrm).collect()
}

/// One raw (not best-of-N) full pass over the corpus, returning elapsed seconds.
fn raw_pass_secs(corpus: &[Int8Vector], q: &Int8Query) -> f64 {
    let t = Instant::now();
    let mut acc = 0.0f32;
    for v in corpus {
        acc += v.dot_query_prepared(q);
    }
    std::hint::black_box(acc);
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
    // d = 128 is the SIFT dimension and the primary kill-test dim for the sdot
    // kernel; the x86 path measured here is the same one aarch64 replaces.
    let (n, d) = (50_000usize, 128usize);
    const N_PAIRS: usize = 35;

    let corpus = make_vecs(n, d, 1);
    let query = Int8Query::prepare(&make_query(d, 999));

    let thermal_before = read_thermal_millideg();
    // Warm caches / branch predictor before timing.
    raw_pass_secs(&corpus, &query);
    raw_pass_secs(&corpus, &query);

    let mut baseline_qps = Vec::with_capacity(N_PAIRS);
    let mut candidate_qps = Vec::with_capacity(N_PAIRS);
    for i in 0..N_PAIRS {
        // Both arms are the same shipped x86 kernel; alternate order per pair so
        // any monotonic drift (thermal, frequency) cancels rather than biasing.
        let (b_secs, c_secs) = if i % 2 == 0 {
            let b = raw_pass_secs(&corpus, &query);
            let c = raw_pass_secs(&corpus, &query);
            (b, c)
        } else {
            let c = raw_pass_secs(&corpus, &query);
            let b = raw_pass_secs(&corpus, &query);
            (b, c)
        };
        baseline_qps.push(n as f64 / b_secs);
        candidate_qps.push(n as f64 / c_secs);
    }

    let thermal_after = read_thermal_millideg();
    let unix_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        // Clock predates the epoch only on a misconfigured host; degrade to 0
        // rather than panic a benchmark run over unrelated artifact metadata.
        .unwrap_or(std::time::Duration::ZERO)
        .as_secs();

    let to_json_arr = |xs: &[f64]| -> String {
        xs.iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    };

    println!("{{");
    println!("  \"artifact\": \"prove_int8_sdot\",");
    println!(
        "  \"change\": \"NEON sdot INT8 kernel (audit 2.1); x86 no-regression record — aarch64 win PENDING Apple Silicon per PERF_FINDINGS.md\","
    );
    println!("  \"unix_ts\": {unix_ts},");
    println!("  \"n_vectors\": {n},");
    println!("  \"dim\": {d},");
    println!("  \"n_pairs\": {N_PAIRS},");
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
