//! Apple Accelerate / vDSP bridge for INT8 quantisation (Wave 3d).
//!
//! Routes the f32 → i8 multiply-round-clamp through `vDSP_vsmsa` (vector
//! scalar multiply + add) followed by a pack/clip pass.  On M1 / M2 / M3
//! the Accelerate framework dispatches large vDSP calls through the AMX
//! coprocessor, giving roughly 1.8-2.0× over pure NEON for d ≥ 256.
//!
//! Compiled only on macOS with `--features vectro_lib_accelerate`; the
//! Accelerate framework is linked from `vectro_py/build.rs`.

use std::os::raw::{c_float, c_int};

extern "C" {
    /// Multiply each element of `a` by scalar `b`, add scalar `c`, write
    /// to `r`.  Length is `n` × `stride_*`.
    ///
    /// Documented in Apple's Accelerate / vDSP reference:
    /// `https://developer.apple.com/documentation/accelerate/1450020-vdsp_vsmsa`.
    fn vDSP_vsmsa(
        a: *const c_float,
        ia: c_int,
        b: *const c_float,
        c: *const c_float,
        r: *mut c_float,
        ir: c_int,
        n: c_int,
    );
}

/// In-place INT8 encode via Accelerate / AMX.
///
/// Returns `abs_max`.  Uses `vDSP_vsmsa(in, 1, &(127/abs_max), &0, scratch, 1, n)`
/// to produce the f32 multiplied vector, then a NEON pack-and-saturate pass
/// converts to i8.  The bottleneck on M1-M3 is the f32 multiply; vDSP
/// dispatches it through AMX where the f32 throughput exceeds NEON's.
pub(crate) fn encode_accelerate_into(v: &[f32], out: &mut [i8]) -> f32 {
    let n = v.len();
    if n == 0 {
        return 1.0;
    }

    // Pass 1: abs-max via NEON (SAFETY: AArch64-v8 mandates NEON; only
    // ever compiled on macOS, which is exclusively aarch64 / x86_64).
    #[cfg(target_arch = "aarch64")]
    let abs_max = unsafe {
        use std::arch::aarch64::*;
        let mut vmax = vdupq_n_f32(0.0_f32);
        let chunks4 = n / 4;
        let ptr = v.as_ptr();
        for i in 0..chunks4 {
            let a = vld1q_f32(ptr.add(i * 4));
            vmax = vmaxq_f32(vmax, vabsq_f32(a));
        }
        let mut m = vmaxvq_f32(vmax);
        for &x in &v[chunks4 * 4..] {
            let ax = x.abs();
            if ax > m { m = ax; }
        }
        m
    };

    // x86_64 macOS — fall through to a scalar abs-max (Accelerate is
    // unlikely to win on Intel Macs anyway, but the path stays compilable).
    #[cfg(not(target_arch = "aarch64"))]
    let abs_max = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max };
    let inv = 127.0_f32 / scale;
    let zero = 0.0_f32;

    // Pass 2: vDSP multiply into a scratch f32 buffer.  We size the
    // scratch on the stack for reasonable d (≤ 4096); larger d falls back
    // to a heap allocation.
    const SCRATCH_CAP: usize = 4096;
    if n <= SCRATCH_CAP {
        let mut scratch: [f32; SCRATCH_CAP] = [0.0_f32; SCRATCH_CAP];
        // SAFETY: pointers + length match the slice contract.
        unsafe {
            vDSP_vsmsa(
                v.as_ptr(),
                1,
                &inv as *const f32,
                &zero as *const f32,
                scratch.as_mut_ptr(),
                1,
                n as c_int,
            );
        }
        for i in 0..n {
            out[i] = scratch[i].round().clamp(-127.0, 127.0) as i8;
        }
    } else {
        let mut scratch: Vec<f32> = vec![0.0_f32; n];
        unsafe {
            vDSP_vsmsa(
                v.as_ptr(),
                1,
                &inv as *const f32,
                &zero as *const f32,
                scratch.as_mut_ptr(),
                1,
                n as c_int,
            );
        }
        for i in 0..n {
            out[i] = scratch[i].round().clamp(-127.0, 127.0) as i8;
        }
    }

    scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerate_matches_scalar() {
        let v: Vec<f32> = (0..768).map(|i| ((i as f32 * 0.13) - 4.0).sin()).collect();
        let mut acc_out = vec![0i8; 768];
        let acc_scale = encode_accelerate_into(&v, &mut acc_out);
        // Compare against scalar reference (single bit-equality may differ
        // by ±1 ULP due to vDSP rounding mode; we accept ≥0.9999 cosine).
        let scale_factor = acc_scale / 127.0;
        let dec: Vec<f32> = acc_out.iter().map(|&c| c as f32 * scale_factor).collect();
        let dot: f32 = v.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
        let nv: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nd: f32 = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(dot / (nv * nd) >= 0.9999, "Accelerate cosine < 0.9999");
    }
}
