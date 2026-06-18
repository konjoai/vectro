//! Wave 1 benchmark — INT8 encode throughput on the host machine.
//!
//! Run with:
//!     cargo run -p vectro_lib --release --example wave1_bench
//!
//! Or with the AMX path on macOS:
//!     cargo run -p vectro_lib --release --features vectro_lib_accelerate \
//!         --example wave1_bench
//!
//! Reports the best of N timed reps for each kernel.  Throughput is
//! number-of-vectors / wall-clock-seconds, reported in M vec/s.

use std::time::Instant;

use vectro_lib::quant::int8::{
    batch_encode_into, batch_encode_normalized_into, encode_fast_fused_into,
};

const N: usize = 100_000;
const D: usize = 768;
const WARMUP: usize = 3;
const REPS: usize = 7;

fn make_unit_vectors(n: usize, d: usize) -> Vec<f32> {
    let mut buf = vec![0.0_f32; n * d];
    for i in 0..n {
        // deterministic but non-degenerate row
        let mut row = vec![0.0_f32; d];
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = ((i + j) as f32 * 0.013_f32).sin()
                * (((j as f32) * 0.011_f32).cos() + 0.5);
        }
        let n2: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for (j, x) in row.iter().enumerate() {
            buf[i * d + j] = x / n2;
        }
    }
    buf
}

fn time_kernel<F: FnMut()>(label: &str, n: usize, mut f: F) -> f64 {
    // Warmup
    for _ in 0..WARMUP {
        f();
    }
    // Timed
    let mut best_secs = f64::INFINITY;
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = Instant::now();
        f();
        let secs = t0.elapsed().as_secs_f64();
        if secs < best_secs {
            best_secs = secs;
        }
        samples.push(secs);
    }
    let throughput = (n as f64) / best_secs / 1.0e6;
    let p50 = {
        let mut s = samples.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    let p50_tp = (n as f64) / p50 / 1.0e6;
    println!(
        "  {:<28}  best {:>7.2}  p50 {:>7.2}  M vec/s   (best {:.3} ms / p50 {:.3} ms)",
        label,
        throughput,
        p50_tp,
        best_secs * 1000.0,
        p50 * 1000.0,
    );
    throughput
}

fn main() {
    let host = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    println!();
    println!("Vectro Wave 1 benchmark — host {host}/{arch}");
    println!("  N = {N}  D = {D}  (≈ {:.1} MiB f32 input)",
             (N * D * 4) as f64 / 1024.0 / 1024.0);
    println!("  warmup = {WARMUP}  reps = {REPS}");
    let accelerate_on = cfg!(feature = "vectro_lib_accelerate");
    println!("  feature vectro_lib_accelerate = {}", accelerate_on);
    println!("  rayon threads = {}", rayon::current_num_threads());
    println!();

    let input = make_unit_vectors(N, D);
    let mut codes = vec![0i8; N * D];
    let mut scales = vec![0.0_f32; N];

    println!("INT8 encode throughput");
    println!("  {:-<28}  {:->7}        {:->7}", "", "", "");

    let baseline = time_kernel("two-pass (Wave 1.1)", N, || {
        batch_encode_into(&input, N, D, &mut codes, &mut scales);
    });

    let wave1 = time_kernel("normalized (Wave 1.2)", N, || {
        batch_encode_normalized_into(&input, N, D, &mut codes, &mut scales);
    });

    let _fused = time_kernel("fused per-row (Wave 2)", N, || {
        for i in 0..N {
            let row = &input[i * D..(i + 1) * D];
            let out = &mut codes[i * D..(i + 1) * D];
            let s = encode_fast_fused_into(row, out);
            scales[i] = s / 127.0;
        }
    });

    println!();
    println!("Wave 1 summary");
    println!("  baseline (two-pass):    {:>7.2} M vec/s", baseline);
    println!("  Wave 1 (normalized):    {:>7.2} M vec/s", wave1);
    let speedup = wave1 / baseline.max(1e-9);
    println!("  Wave 1 / baseline:      {:>7.2}×", speedup);

    let gate = 19.5_f64;
    if wave1 >= gate {
        println!("  ✓ cleared the ≥{:.1} M vec/s Wave 1 gate", gate);
    } else {
        println!("  ✗ did NOT clear the ≥{:.1} M vec/s Wave 1 gate", gate);
    }
    println!();
}
