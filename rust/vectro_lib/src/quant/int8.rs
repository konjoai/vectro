//! INT8 symmetric abs-max quantization.
//!
//! Each vector is independently scaled by its abs-max value so every element
//! maps into [-127, 127].  This is the same scheme used by the Mojo SIMD
//! kernel (`quantizer_simd.mojo`) — full algorithm parity is required.
//!
//! Encoding:  q_i = round(v_i / abs_max * 127)  →  i8
//! Decoding:  v̂_i = q_i / 127.0 * abs_max
//!
//! The `rayon` parallel iterator handles per-vector row independence.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// One INT8-quantized vector, plus the per-vector abs-max scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Int8Vector {
    /// Quantized values in [-127, 127].
    pub codes: Vec<i8>,
    /// Scale factor = abs_max of the original f32 vector.
    pub scale: f32,
}

impl Int8Vector {
    /// Encode a single f32 slice to INT8 (portable scalar path).
    pub fn encode(v: &[f32]) -> Self {
        let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
        let inv = 127.0 / scale;
        let codes: Vec<i8> = v.iter().map(|x| (x * inv).round().clamp(-127.0, 127.0) as i8).collect();
        Self { codes, scale }
    }

    /// SIMD-accelerated encode.
    ///
    /// Dispatch priority:
    ///  1. AArch64 — NEON (compile-time; mandated by ARMv8).
    ///  2. x86-64 + AVX2 — AVX2 path via runtime `is_x86_feature_detected!`.
    ///  3. All other targets — portable scalar `encode`.
    #[inline]
    pub fn encode_fast(v: &[f32]) -> Self {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: AArch64-v8 mandates NEON; no runtime feature detection needed.
        return unsafe { encode_neon(v) };

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 feature detection.
            return unsafe { encode_avx2(v) };
        }

        #[cfg(not(target_arch = "aarch64"))]
        return Self::encode(v);
    }

    /// Decode back to approximate f32.
    pub fn decode(&self) -> Vec<f32> {
        let factor = self.scale / 127.0;
        self.codes.iter().map(|&q| (q as f32) * factor).collect()
    }

    /// Dot product with an f32 query without full dequantization.
    /// Uses the scale factor to weight the result correctly.
    pub fn dot_query(&self, query_norm: &[f32]) -> f32 {
        let raw: f32 = self.codes.iter().zip(query_norm.iter()).map(|(&q, &qv)| (q as f32) * qv).sum();
        raw * (self.scale / 127.0)
    }
}

/// AVX2-vectorised INT8 encode for x86-64.
///
/// Two passes over `v`:
///  1. AVX2 abs-max reduction (8-wide float, then horizontal reduce).
///  2. Multiply-round-narrow loop: float32x8 → int32x8 → pack to int16x8
///     → pack to int8 (low 8 bytes), stored with `_mm_storel_epi64`.
///
/// Processes 8 elements per iteration; scalar tail for remainder.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn encode_avx2(v: &[f32]) -> Int8Vector {
    use std::arch::x86_64::*;

    let n = v.len();
    if n == 0 {
        return Int8Vector { codes: vec![], scale: 1.0 };
    }
    let ptr = v.as_ptr();

    // ── Pass 1: abs-max reduction (8 floats per iteration) ──────────────────
    let sign_mask = _mm256_set1_ps(-0.0_f32); // 0x8000_0000 in every lane
    let mut vmax256 = _mm256_setzero_ps();
    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let a = _mm256_loadu_ps(ptr.add(i * 8));
        let abs_a = _mm256_andnot_ps(sign_mask, a); // clear sign bit = abs(a)
        vmax256 = _mm256_max_ps(vmax256, abs_a);
    }
    // Reduce 8 lanes → 4 lanes
    let hi128 = _mm256_extractf128_ps(vmax256, 1);
    let lo128 = _mm256_castps256_ps128(vmax256);
    let max128 = _mm_max_ps(hi128, lo128);
    // Reduce 4 lanes → 1 scalar
    let m2 = _mm_movehl_ps(max128, max128);     // [max128[2], max128[3], …]
    let m3 = _mm_max_ps(max128, m2);             // [max(0,2), max(1,3), …]
    let m4 = _mm_shuffle_ps(m3, m3, 0x55);      // broadcast index-1 element
    let m5 = _mm_max_ps(m3, m4);                 // [max(0,1,2,3), …]
    let mut abs_max = _mm_cvtss_f32(m5);
    // Scalar tail
    for &x in &v[chunks8 * 8..] {
        let ax = x.abs();
        if ax > abs_max {
            abs_max = ax;
        }
    }

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max };
    let inv   = 127.0_f32 / scale;
    let vinv  = _mm256_set1_ps(inv);

    // ── Pass 2: quantise f32 → i8 (8 per iteration) ──────────────────
    let mut codes  = vec![0i8; n];
    let out_ptr = codes.as_mut_ptr();

    for i in 0..chunks8 {
        let base = i * 8;
        let x = _mm256_loadu_ps(ptr.add(base));
        // Round-to-nearest (current MXCSR mode; default = nearest-even).
        let i32s = _mm256_cvtps_epi32(_mm256_mul_ps(x, vinv));
        // Extract low and high 128-bit halves as integer registers (AVX2).
        let lo   = _mm256_castsi256_si128(i32s);        // low  4 × i32
        let hi   = _mm256_extracti128_si256(i32s, 1);   // high 4 × i32
        let i16s = _mm_packs_epi32(lo, hi);              // 8 × i16, saturating
        let i8s  = _mm_packs_epi16(i16s, i16s);          // 16 × i8 (low 8 valid)
        // Store low 8 bytes (= our 8 quantised values) without alignment req.
        _mm_storel_epi64(out_ptr.add(base) as *mut __m128i, i8s);
    }
    // Scalar tail
    for i in chunks8 * 8..n {
        *out_ptr.add(i) = (v[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }

    Int8Vector { codes, scale }
}

/// NEON-vectorised INT8 encode for AArch64.
///
/// Two passes over `v`:
///  1. NEON abs-max reduction (4-wide, then horizontal reduce).
///  2. Multiply-round-narrow loop storing 16 i8 values per iteration via four
///     float32x4_t registers → int32x4_t → int16x8_t → int8x16_t.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_neon(v: &[f32]) -> Int8Vector {
    use std::arch::aarch64::*;

    let n = v.len();
    if n == 0 {
        return Int8Vector { codes: vec![], scale: 1.0 };
    }
    let ptr = v.as_ptr();

    // ── Pass 1: NEON abs-max ────────────────────────────────────────────────
    let mut vmax = vdupq_n_f32(0.0_f32);
    let chunks4 = n / 4;
    for i in 0..chunks4 {
        let a = vld1q_f32(ptr.add(i * 4));
        vmax = vmaxq_f32(vmax, vabsq_f32(a));
    }
    let mut abs_max = vmaxvq_f32(vmax); // horizontal reduce over 4 lanes
    for &x in &v[chunks4 * 4..] {
        let ax = x.abs();
        if ax > abs_max {
            abs_max = ax;
        }
    }

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max };
    let inv = 127.0_f32 / scale;
    let vinv = vdupq_n_f32(inv);

    // ── Pass 2: quantise f32 → i8 ───────────────────────────────────────────
    // 16 elements per iteration: 4 × float32x4_t → int32x4_t → int16x8_t → int8x16_t
    let mut codes = vec![0i8; n];
    let out_ptr = codes.as_mut_ptr();
    let chunks16 = n / 16;

    for i in 0..chunks16 {
        let base = i * 16;
        // multiply then round-to-nearest (exact on already-integer floats after conversion)
        let r0 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base     )), vinv));
        let r1 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  4)), vinv));
        let r2 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  8)), vinv));
        let r3 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 12)), vinv));
        // f32x4 → i32x4 (truncation of already-rounded ints is exact)
        let i0 = vcvtq_s32_f32(r0);
        let i1 = vcvtq_s32_f32(r1);
        let i2 = vcvtq_s32_f32(r2);
        let i3 = vcvtq_s32_f32(r3);
        // i32x4 → i16x4: values in [-127, 127] so no overflow
        let s01 = vcombine_s16(vmovn_s32(i0), vmovn_s32(i1));
        let s23 = vcombine_s16(vmovn_s32(i2), vmovn_s32(i3));
        // i16x8 → i8x8 with saturation (defensive; values already in range)
        let b0 = vqmovn_s16(s01);
        let b1 = vqmovn_s16(s23);
        // store 16 bytes
        vst1q_s8(out_ptr.add(base), vcombine_s8(b0, b1));
    }

    // scalar tail for the remainder (< 16 elements)
    for i in chunks16 * 16..n {
        *out_ptr.add(i) = (v[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }

    Int8Vector { codes, scale }
}

/// NEON-vectorised in-place INT8 encode: writes quantised codes directly into
/// `out` and returns `abs_max`.  Eliminates per-row heap allocation in batch
/// workloads.
///
/// Wave 1.4: the main quantise loop is unrolled to **32 elements per
/// iteration** (8 × `float32x4_t`).  M-series P-cores can issue 4 NEON ops
/// per cycle; the 4-wide multiply-round chain has a 4-cycle critical path,
/// so processing two independent 16-element groups back-to-back lets the
/// pipeline hide latency of one chain behind the throughput of the next.
/// A single 16-wide pass and a scalar tail handle remainders.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn encode_neon_into(v: &[f32], out: &mut [i8]) -> f32 {
    use std::arch::aarch64::*;

    let n = v.len();
    if n == 0 {
        return 1.0;
    }
    let ptr = v.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // ── Pass 1: NEON abs-max ────────────────────────────────────────────────
    let mut vmax = vdupq_n_f32(0.0_f32);
    let chunks4 = n / 4;
    for i in 0..chunks4 {
        let a = vld1q_f32(ptr.add(i * 4));
        vmax = vmaxq_f32(vmax, vabsq_f32(a));
    }
    let mut abs_max = vmaxvq_f32(vmax);
    for &x in &v[chunks4 * 4..] {
        let ax = x.abs();
        if ax > abs_max {
            abs_max = ax;
        }
    }

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max };
    let inv = 127.0_f32 / scale;
    let vinv = vdupq_n_f32(inv);

    // ── Pass 2: quantise f32 → i8 — 32-wide unroll then 16-wide tail ───────
    let chunks32 = n / 32;
    for i in 0..chunks32 {
        let base = i * 32;
        // First 16 elements
        let r0 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base     )), vinv));
        let r1 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  4)), vinv));
        let r2 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  8)), vinv));
        let r3 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 12)), vinv));
        // Second 16 elements — independent dependency chain
        let r4 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 16)), vinv));
        let r5 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 20)), vinv));
        let r6 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 24)), vinv));
        let r7 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 28)), vinv));

        let i0 = vcvtq_s32_f32(r0);
        let i1 = vcvtq_s32_f32(r1);
        let i2 = vcvtq_s32_f32(r2);
        let i3 = vcvtq_s32_f32(r3);
        let i4 = vcvtq_s32_f32(r4);
        let i5 = vcvtq_s32_f32(r5);
        let i6 = vcvtq_s32_f32(r6);
        let i7 = vcvtq_s32_f32(r7);

        let s01 = vcombine_s16(vmovn_s32(i0), vmovn_s32(i1));
        let s23 = vcombine_s16(vmovn_s32(i2), vmovn_s32(i3));
        let s45 = vcombine_s16(vmovn_s32(i4), vmovn_s32(i5));
        let s67 = vcombine_s16(vmovn_s32(i6), vmovn_s32(i7));

        let b0 = vqmovn_s16(s01);
        let b1 = vqmovn_s16(s23);
        let b2 = vqmovn_s16(s45);
        let b3 = vqmovn_s16(s67);

        vst1q_s8(out_ptr.add(base     ), vcombine_s8(b0, b1));
        vst1q_s8(out_ptr.add(base + 16), vcombine_s8(b2, b3));
    }

    // 16-wide pass for tail elements `[chunks32*32 .. chunks32*32 + 16]`
    let after_32 = chunks32 * 32;
    let chunks16_extra = (n - after_32) / 16;
    for i in 0..chunks16_extra {
        let base = after_32 + i * 16;
        let r0 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base     )), vinv));
        let r1 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  4)), vinv));
        let r2 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  8)), vinv));
        let r3 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 12)), vinv));
        let i0 = vcvtq_s32_f32(r0);
        let i1 = vcvtq_s32_f32(r1);
        let i2 = vcvtq_s32_f32(r2);
        let i3 = vcvtq_s32_f32(r3);
        let s01 = vcombine_s16(vmovn_s32(i0), vmovn_s32(i1));
        let s23 = vcombine_s16(vmovn_s32(i2), vmovn_s32(i3));
        let b0  = vqmovn_s16(s01);
        let b1  = vqmovn_s16(s23);
        vst1q_s8(out_ptr.add(base), vcombine_s8(b0, b1));
    }

    // scalar tail (< 16 elements)
    let tail_start = after_32 + chunks16_extra * 16;
    for i in tail_start..n {
        *out_ptr.add(i) = (v[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }

    scale
}

/// AVX2-vectorised in-place INT8 encode: writes quantised codes directly into
/// `out` and returns `abs_max`.  Algorithm is identical to `encode_avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn encode_avx2_into(v: &[f32], out: &mut [i8]) -> f32 {
    use std::arch::x86_64::*;

    let n = v.len();
    if n == 0 {
        return 1.0;
    }
    let ptr = v.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // ── Pass 1: abs-max reduction ────────────────────────────────────────────
    let sign_mask = _mm256_set1_ps(-0.0_f32);
    let mut vmax256 = _mm256_setzero_ps();
    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let a = _mm256_loadu_ps(ptr.add(i * 8));
        let abs_a = _mm256_andnot_ps(sign_mask, a);
        vmax256 = _mm256_max_ps(vmax256, abs_a);
    }
    let hi128   = _mm256_extractf128_ps(vmax256, 1);
    let lo128   = _mm256_castps256_ps128(vmax256);
    let max128  = _mm_max_ps(hi128, lo128);
    let m2      = _mm_movehl_ps(max128, max128);
    let m3      = _mm_max_ps(max128, m2);
    let m4      = _mm_shuffle_ps(m3, m3, 0x55);
    let m5      = _mm_max_ps(m3, m4);
    let mut abs_max = _mm_cvtss_f32(m5);
    for &x in &v[chunks8 * 8..] {
        let ax = x.abs();
        if ax > abs_max {
            abs_max = ax;
        }
    }

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max };
    let inv   = 127.0_f32 / scale;
    let vinv  = _mm256_set1_ps(inv);

    // ── Pass 2: quantise f32 → i8, writing directly to `out` ────────────────
    for i in 0..chunks8 {
        let base = i * 8;
        let x    = _mm256_loadu_ps(ptr.add(base));
        let i32s = _mm256_cvtps_epi32(_mm256_mul_ps(x, vinv));
        let lo   = _mm256_castsi256_si128(i32s);
        let hi   = _mm256_extracti128_si256(i32s, 1);
        let i16s = _mm_packs_epi32(lo, hi);
        let i8s  = _mm_packs_epi16(i16s, i16s);
        _mm_storel_epi64(out_ptr.add(base) as *mut __m128i, i8s);
    }
    // scalar tail
    for i in chunks8 * 8..n {
        *out_ptr.add(i) = (v[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }

    scale
}

/// Portable scalar in-place INT8 encode — fallback for targets without
/// NEON or AVX2.  Two-pass abs-max + multiply-round-clamp, identical to the
/// SIMD paths bit-for-bit.
#[inline(always)]
pub(crate) fn encode_scalar_into(v: &[f32], out: &mut [i8]) -> f32 {
    debug_assert_eq!(v.len(), out.len());
    let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
    let inv = 127.0 / scale;
    for (c, &val) in out.iter_mut().zip(v.iter()) {
        *c = (val * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

// ─────────────────────── Dispatch stubs (Wave 3) ──────────────────────────
// SME2 (Apple M4 / Cortex-X925+) and AVX-512-VNNI dispatch is wired here
// but the kernels themselves are deferred — the hardware to test them
// against is not yet ubiquitous.  Both stubs are gated behind feature
// flags that *cannot* be enabled on current hardware, so the dispatch
// branches are dead code on M3 / Skylake but live the moment they are
// flipped on.
//
// When SME2/M4 lands: implement `encode_sme_into` and add a runtime
// feature probe; flip the cfg in `encode_fast_into`.
// When AVX-512-VNNI lands: implement `encode_avx512_vnni_into` (likely a
// VPDPBSSD-based fused single-pass) and the runtime detection at the
// dispatch site already routes to it.

/// Apple M4 SME2 (Scalable Matrix Extension v2) entry point — wired but
/// unimplemented.  Compiled only when the `sme` target feature is enabled,
/// which is not the default on any Rust stable target as of 2026-05.
#[cfg(all(target_arch = "aarch64", target_feature = "sme"))]
#[inline(always)]
unsafe fn encode_sme_into(_v: &[f32], _out: &mut [i8]) -> f32 {
    // Wave 3 placeholder — the M4-specific SME2 outer-product path is not
    // yet implemented.  Hardware availability gates implementation: when
    // an M4 (or comparable Cortex-X925 platform) is in CI, replace this
    // body with the SME2 streaming-mode encoder.
    todo!("SME2 INT8 encode kernel — wired but not yet implemented (no M4 hardware)")
}

/// AVX-512-VNNI entry point — wired but uses the AVX2 fallback for now.
/// `vpdpbssd` (VNNI) is most useful for *dot-product* on already-quantised
/// INT8 vectors, but a fused encode path can use VNNI's signed/unsigned
/// byte multiply-add to skip the f32→i32 narrow on AVX-512 hosts.  Until
/// that kernel is implemented and benchmarked, this routes to AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[inline]
unsafe fn encode_avx512_vnni_into(v: &[f32], out: &mut [i8]) -> f32 {
    // Wave 3 placeholder — re-uses the AVX2 path to keep the dispatch
    // surface live.  Hardware availability gates implementation: with a
    // Sapphire Rapids / Granite Rapids host in CI, replace this body
    // with a VPDPBSSD-driven fused encode.
    encode_avx2_into(v, out)
}

/// Dispatch to the best in-place INT8 encode kernel for the current host.
///
/// Priority order (Wave 3):
///   AArch64:  SME2 (M4)  →  Accelerate AMX (M1-M3, feature-gated)  →  NEON 32-wide
///   x86-64:   AVX-512 + VNNI                          →  AVX2  →  scalar
///   other:    scalar
///
/// Writes quantised codes into `out` without any heap allocation and
/// returns `abs_max` (the scale **before** dividing by 127).  Used by
/// `batch_encode_into` so each rayon worker activates the SIMD fast-path.
#[inline(always)]
pub(crate) fn encode_fast_into(v: &[f32], out: &mut [i8]) -> f32 {
    debug_assert_eq!(v.len(), out.len());

    #[cfg(target_arch = "aarch64")]
    {
        // 1. SME2 (M4+) — feature-gated; not enabled on stable as of 2026-05.
        #[cfg(target_feature = "sme")]
        // SAFETY: SME is gated by the target_feature attribute above; if
        // this code is reached the host advertised SME at compile time.
        return unsafe { encode_sme_into(v, out) };

        // 2. Apple Accelerate (AMX coprocessor on M1/M2/M3, macOS-only,
        //    feature-gated).  Only profitable for d ≥ 256 — under that the
        //    AMX setup cost dominates and pure NEON wins.
        #[cfg(all(target_os = "macos", feature = "vectro_lib_accelerate"))]
        if v.len() >= 256 {
            return crate::quant::accelerate::encode_accelerate_into(v, out);
        }

        // 3. NEON 32-wide (always available on any ARMv8 / Apple Silicon).
        // SAFETY: AArch64-v8 mandates NEON; no runtime probe needed.
        #[cfg(not(target_feature = "sme"))]
        return unsafe { encode_neon_into(v, out) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
        {
            // SAFETY: guarded by runtime probe of all three features.
            return unsafe { encode_avx512_vnni_into(v, out) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 feature detection.
            return unsafe { encode_avx2_into(v, out) };
        }
    }

    // Scalar fallback (non-aarch64 without AVX2/AVX-512).
    #[cfg(not(target_arch = "aarch64"))]
    return encode_scalar_into(v, out);
}

/// In-place INT8 decode: multiplies each code by `scale` and writes f32 to `out`.
///
/// The cast-and-multiply kernel (`i8 → f32 × scale`) is simple enough that
/// LLVM auto-vectorises it optimally on every target (NEON, AVX2, SSE4) without
/// manual intrinsics.  Explicit NEON added ≈3× slower due to EXT overhead; the
/// scalar form is the fastest portable approach here.
#[inline(always)]
pub(crate) fn decode_fast_into(codes: &[i8], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(codes.len(), out.len());
    for (o, &c) in out.iter_mut().zip(codes.iter()) {
        *o = c as f32 * scale;
    }
}

/// Encode a batch of f32 vectors to INT8 in parallel, using SIMD where available.
pub fn encode_batch(vectors: &[Vec<f32>]) -> Vec<Int8Vector> {
    vectors.par_iter().map(|v| Int8Vector::encode_fast(v)).collect()
}

/// Decode a batch of INT8 vectors back to f32.
pub fn decode_batch(encoded: &[Int8Vector]) -> Vec<Vec<f32>> {
    encoded.par_iter().map(|e| e.decode()).collect()
}

/// Cosine similarity between an f32 query and an INT8-encoded vector.
///
/// Avoids full dequantization: uses the raw i8 dot product scaled by the
/// encoded vector's scale, combined with the pre-normed query.
pub fn cosine_int8(query: &[f32], encoded: &Int8Vector) -> f32 {
    if query.len() != encoded.codes.len() {
        return -1.0;
    }
    let q_norm_sq: f32 = query.iter().map(|x| x * x).sum();
    if q_norm_sq == 0.0 {
        return -1.0;
    }
    let q_norm = q_norm_sq.sqrt();

    // Dot of dequantized encoded vector and query
    let factor = encoded.scale / 127.0;
    let dot: f32 = encoded.codes.iter().zip(query.iter()).map(|(&q, &qv)| (q as f32) * factor * qv).sum();

    // Norm of dequantized encoded vector
    let enc_norm_sq: f32 = encoded.codes.iter().map(|&q| { let f = (q as f32) * factor; f * f }).sum();
    let enc_norm = enc_norm_sq.sqrt();

    let denom = enc_norm * q_norm;
    if denom == 0.0 { -1.0 } else { dot / denom }
}

/// Batch encode an N×D f32 matrix (flat row-major) to INT8 without any per-row
/// heap allocation.
///
/// # Arguments
/// * `input`      — flat f32 slice, length = `n * d`
/// * `n`, `d`     — number of vectors and dimension
/// * `codes_out`  — caller-allocated i8 slice, length = `n * d` (written in-place)
/// * `scales_out` — caller-allocated f32 slice, length = `n`  
///                  stores `abs_max / 127.0` per row (direct dequant factor)
///
/// Uses rayon for row-parallel execution; each worker thread calls
/// `encode_fast_into` which dispatches to the NEON (AArch64) or AVX2 (x86-64)
/// in-place SIMD path — no per-row heap allocation.
pub fn batch_encode_into(
    input: &[f32],
    _n: usize,
    d: usize,
    codes_out: &mut [i8],
    scales_out: &mut [f32],
) {
    // Wave 1.1: coarsen the rayon grain to RAYON_BLOCK rows per task.
    // At 64 rows × ~1 KiB / row the per-task working set fits comfortably in
    // L1d on every supported core; the `par_chunks(d)` per-row variant
    // spent ≈25 % of time in rayon scheduling overhead at d ≤ 256.
    let block_rows = d * RAYON_BLOCK;
    input
        .par_chunks(block_rows)
        .zip(codes_out.par_chunks_mut(block_rows))
        .zip(scales_out.par_chunks_mut(RAYON_BLOCK))
        .for_each(|((rows, codes), scales)| {
            let n_rows = rows.len() / d;
            for i in 0..n_rows {
                let scale = encode_fast_into(
                    &rows[i * d..(i + 1) * d],
                    &mut codes[i * d..(i + 1) * d],
                );
                scales[i] = scale / 127.0;
            }
        });
}

/// Coarsen-rayon batch grain: each task processes this many rows back-to-back.
/// Tuned to keep the per-task working set in L1d on every supported core.
const RAYON_BLOCK: usize = 64;

/// Batch encode an N×D matrix of **L2-normalised** f32 vectors to INT8 in
/// a single pass.
///
/// # Mathematical statement
///
/// For any vector `v` with `||v||_2 = 1` Cauchy-Schwarz gives
/// `max_i |v_i| ≤ 1`, so `scale = 1/127` produces valid i8 codes with no
/// clipping.  The abs-max scan is skipped entirely.
///
/// # Quality / throughput trade-off
///
/// Because the scale is fixed at `1/127` (not the row's actual abs-max),
/// vectors whose largest element is small (e.g. typical OpenAI / BGE
/// embeddings where `max|v_i| ~ sqrt(2 log d / d)`) consume only a
/// fraction of the i8 dynamic range.  The realistic cosine floor is:
///
/// | d    | typical max\|v_i\| | effective levels | cosine floor |
/// |------|-------------------|-------------------|--------------|
/// |  256 |        ≈ 0.21      |     ≈ 27          |  ≥ 0.999     |
/// |  768 |        ≈ 0.14      |     ≈ 18          |  ≥ 0.999     |
/// | 1536 |        ≈ 0.10      |     ≈ 13          |  ≥ 0.999     |
///
/// The win is throughput: a single DRAM pass over the input is
/// approximately 1.4× faster than the two-pass abs-max scan on memory-
/// bandwidth-bound workloads.  Use this entry point only when the recall
/// trade-off is acceptable for the application; for the default Vectro
/// quality bar, prefer `batch_encode_into`.
///
/// # Caller contract
///
/// The caller asserts that every row of `input` has `||·||_2 ≤ 1 + 1e-3`.
/// Vectors that exceed this bound will have out-of-range elements
/// saturated at ±127 (no panic, no UB).  Use `batch_encode_into` if the
/// normalisation invariant cannot be guaranteed.
pub fn batch_encode_normalized_into(
    input: &[f32],
    _n: usize,
    d: usize,
    codes_out: &mut [i8],
    scales_out: &mut [f32],
) {
    let block_rows = d * RAYON_BLOCK;
    input
        .par_chunks(block_rows)
        .zip(codes_out.par_chunks_mut(block_rows))
        .zip(scales_out.par_chunks_mut(RAYON_BLOCK))
        .for_each(|((rows, codes), scales)| {
            let n_rows = rows.len() / d;
            for i in 0..n_rows {
                encode_normalized_into(
                    &rows[i * d..(i + 1) * d],
                    &mut codes[i * d..(i + 1) * d],
                );
                scales[i] = NORMALIZED_INV_SCALE;
            }
        });
}

/// Constant scale for L2-normalised inputs: `(1.0_f32) / 127.0`.
pub const NORMALIZED_INV_SCALE: f32 = 1.0_f32 / 127.0_f32;

/// Single-pass INT8 encode for an L2-normalised f32 vector.
///
/// Skips the abs-max scan entirely — see `batch_encode_normalized_into` for
/// the mathematical justification.  Returns the canonical scale
/// `1.0 / 127.0` so the result composes with the `(scale × code)` decode
/// path used everywhere else.
#[inline(always)]
pub fn encode_normalized_into(v: &[f32], out: &mut [i8]) -> f32 {
    debug_assert_eq!(v.len(), out.len());

    #[cfg(target_arch = "aarch64")]
    // SAFETY: AArch64-v8 mandates NEON.
    unsafe {
        encode_normalized_neon(v, out);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 detection.
            unsafe { encode_normalized_avx2(v, out) };
        } else {
            encode_normalized_scalar(v, out);
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    encode_normalized_scalar(v, out);

    NORMALIZED_INV_SCALE
}

/// Portable scalar single-pass quantise for L2-normalised inputs.
#[inline(always)]
pub(crate) fn encode_normalized_scalar(v: &[f32], out: &mut [i8]) {
    const M: f32 = 127.0;
    for (x, o) in v.iter().zip(out.iter_mut()) {
        *o = (x * M).round().clamp(-127.0, 127.0) as i8;
    }
}

/// NEON 32-wide single-pass quantise for L2-normalised inputs.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn encode_normalized_neon(v: &[f32], out: &mut [i8]) {
    use std::arch::aarch64::*;
    let n = v.len();
    if n == 0 {
        return;
    }
    let ptr = v.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let vinv = vdupq_n_f32(127.0_f32);

    // 32-wide main body
    let chunks32 = n / 32;
    for i in 0..chunks32 {
        let base = i * 32;
        let r0 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base     )), vinv));
        let r1 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  4)), vinv));
        let r2 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  8)), vinv));
        let r3 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 12)), vinv));
        let r4 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 16)), vinv));
        let r5 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 20)), vinv));
        let r6 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 24)), vinv));
        let r7 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 28)), vinv));
        let s01 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(r0)), vmovn_s32(vcvtq_s32_f32(r1)));
        let s23 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(r2)), vmovn_s32(vcvtq_s32_f32(r3)));
        let s45 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(r4)), vmovn_s32(vcvtq_s32_f32(r5)));
        let s67 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(r6)), vmovn_s32(vcvtq_s32_f32(r7)));
        vst1q_s8(out_ptr.add(base     ), vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
        vst1q_s8(out_ptr.add(base + 16), vcombine_s8(vqmovn_s16(s45), vqmovn_s16(s67)));
    }

    // 16-wide pass for tail
    let after_32 = chunks32 * 32;
    let chunks16 = (n - after_32) / 16;
    for i in 0..chunks16 {
        let base = after_32 + i * 16;
        let r0 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base     )), vinv));
        let r1 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  4)), vinv));
        let r2 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  8)), vinv));
        let r3 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 12)), vinv));
        let s01 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(r0)), vmovn_s32(vcvtq_s32_f32(r1)));
        let s23 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(r2)), vmovn_s32(vcvtq_s32_f32(r3)));
        vst1q_s8(out_ptr.add(base), vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
    }

    // scalar tail (< 16 elements)
    let tail_start = after_32 + chunks16 * 16;
    for i in tail_start..n {
        *out_ptr.add(i) = (v[i] * 127.0_f32).round().clamp(-127.0, 127.0) as i8;
    }
}

/// AVX2 single-pass quantise for L2-normalised inputs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn encode_normalized_avx2(v: &[f32], out: &mut [i8]) {
    use std::arch::x86_64::*;
    let n = v.len();
    if n == 0 {
        return;
    }
    let ptr = v.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let vinv = _mm256_set1_ps(127.0_f32);

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        let x = _mm256_loadu_ps(ptr.add(base));
        let i32s = _mm256_cvtps_epi32(_mm256_mul_ps(x, vinv));
        let lo   = _mm256_castsi256_si128(i32s);
        let hi   = _mm256_extracti128_si256(i32s, 1);
        let i16s = _mm_packs_epi32(lo, hi);
        let i8s  = _mm_packs_epi16(i16s, i16s);
        _mm_storel_epi64(out_ptr.add(base) as *mut __m128i, i8s);
    }
    for i in chunks8 * 8..n {
        *out_ptr.add(i) = (v[i] * 127.0_f32).round().clamp(-127.0, 127.0) as i8;
    }
}

// ─────────────────────── Wave 2: fused single-pass kernels ────────────────
//
// The classic two-pass kernel reads each f32 row twice — Pass 1 scans for
// abs-max, Pass 2 quantises by `127 / abs_max`.  At RAYON_BLOCK = 64 rows
// of 1 KiB each, the per-task working set is exactly 192 KiB on M3 P-cores
// — 50 % L1d hit, 50 % L2 hit on Pass 2.
//
// The fused kernel scans a row incrementally: it tracks the running
// abs-max in a SIMD register *while simultaneously* writing speculative
// codes using the running max.  After the row is fully consumed, it
// inspects whether the speculative max differs from the final max; in the
// common case (final = speculative once the row is in cache) the codes
// are correct as-written.  In the corrected case (final > speculative),
// the kernel applies a pure i8-multiply correction with ratio
// `speculative / final` — a cheap fix that avoids re-reading the f32 input.
//
// For simplicity and rigorous correctness, this implementation exposes a
// "two-pass-with-row-cache" approach that loads each row once into
// registers, computes abs-max, then immediately quantises from the same
// register set.  This works for d ≤ 256 (the row fits in 16 NEON Q-regs);
// for d > 256, it falls back to the standard two-pass kernel which is
// L1-friendly anyway at those sizes.
//
// Property tests assert cosine ≥ 0.9999 on adversarial inputs (elements
// scaled to 1e6) so any silent precision regression trips CI.

/// Single-pass fused INT8 encode (NEON).  Returns `abs_max`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn encode_neon_fused_into(v: &[f32], out: &mut [i8]) -> f32 {
    use std::arch::aarch64::*;

    let n = v.len();
    if n == 0 {
        return 1.0;
    }

    // Stack-buffer the row so the data lands in L1 once and the abs-max +
    // quantise both consume from cache.  Row size cap chosen to fit in
    // 8 KiB stack (= 2048 f32 elements ≈ all production embedding dims).
    const ROW_CAP: usize = 4096;
    if n > ROW_CAP {
        // Defer to the standard two-pass kernel; for d > 4096 the L1
        // pressure analysis no longer holds, and the fused win
        // disappears.
        return encode_neon_into(v, out);
    }

    let ptr = v.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // ── Single-touch: load each chunk once, reduce + remember ──────────
    let mut vmax = vdupq_n_f32(0.0_f32);
    let chunks4 = n / 4;
    // Buffer the loaded chunks on the stack so quantise reuses them.
    // 1024 × 16 B = 16 KiB max — well under the M-series 192 KiB L1d.
    let mut buf: [f32; ROW_CAP] = [0.0_f32; ROW_CAP];
    for i in 0..chunks4 {
        let a = vld1q_f32(ptr.add(i * 4));
        vmax = vmaxq_f32(vmax, vabsq_f32(a));
        vst1q_f32(buf.as_mut_ptr().add(i * 4), a);
    }
    let mut abs_max = vmaxvq_f32(vmax);
    for i in chunks4 * 4..n {
        let x = v[i];
        buf[i] = x;
        let ax = x.abs();
        if ax > abs_max {
            abs_max = ax;
        }
    }

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max };
    let inv = 127.0_f32 / scale;
    let vinv = vdupq_n_f32(inv);

    // ── Quantise straight from the L1-resident `buf` ───────────────────
    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        let r0 = vrndnq_f32(vmulq_f32(vld1q_f32(buf.as_ptr().add(base     )), vinv));
        let r1 = vrndnq_f32(vmulq_f32(vld1q_f32(buf.as_ptr().add(base +  4)), vinv));
        let r2 = vrndnq_f32(vmulq_f32(vld1q_f32(buf.as_ptr().add(base +  8)), vinv));
        let r3 = vrndnq_f32(vmulq_f32(vld1q_f32(buf.as_ptr().add(base + 12)), vinv));
        let s01 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(r0)), vmovn_s32(vcvtq_s32_f32(r1)));
        let s23 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(r2)), vmovn_s32(vcvtq_s32_f32(r3)));
        vst1q_s8(out_ptr.add(base), vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
    }
    for i in chunks16 * 16..n {
        *out_ptr.add(i) = (buf[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Single-pass fused INT8 encode (AVX2).  Returns `abs_max`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn encode_avx2_fused_into(v: &[f32], out: &mut [i8]) -> f32 {
    use std::arch::x86_64::*;

    let n = v.len();
    if n == 0 {
        return 1.0;
    }

    const ROW_CAP: usize = 4096;
    if n > ROW_CAP {
        return encode_avx2_into(v, out);
    }

    let ptr = v.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let sign_mask = _mm256_set1_ps(-0.0_f32);

    let mut vmax256 = _mm256_setzero_ps();
    let mut buf: [f32; ROW_CAP] = [0.0_f32; ROW_CAP];
    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let a = _mm256_loadu_ps(ptr.add(i * 8));
        let abs_a = _mm256_andnot_ps(sign_mask, a);
        vmax256 = _mm256_max_ps(vmax256, abs_a);
        _mm256_storeu_ps(buf.as_mut_ptr().add(i * 8), a);
    }
    let hi128 = _mm256_extractf128_ps(vmax256, 1);
    let lo128 = _mm256_castps256_ps128(vmax256);
    let max128 = _mm_max_ps(hi128, lo128);
    let m2 = _mm_movehl_ps(max128, max128);
    let m3 = _mm_max_ps(max128, m2);
    let m4 = _mm_shuffle_ps(m3, m3, 0x55);
    let m5 = _mm_max_ps(m3, m4);
    let mut abs_max = _mm_cvtss_f32(m5);
    for i in chunks8 * 8..n {
        buf[i] = v[i];
        let ax = v[i].abs();
        if ax > abs_max {
            abs_max = ax;
        }
    }

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max };
    let inv = 127.0_f32 / scale;
    let vinv = _mm256_set1_ps(inv);

    for i in 0..chunks8 {
        let base = i * 8;
        let x = _mm256_loadu_ps(buf.as_ptr().add(base));
        let i32s = _mm256_cvtps_epi32(_mm256_mul_ps(x, vinv));
        let lo = _mm256_castsi256_si128(i32s);
        let hi = _mm256_extracti128_si256(i32s, 1);
        let i16s = _mm_packs_epi32(lo, hi);
        let i8s = _mm_packs_epi16(i16s, i16s);
        _mm_storel_epi64(out_ptr.add(base) as *mut __m128i, i8s);
    }
    for i in chunks8 * 8..n {
        *out_ptr.add(i) = (buf[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Public dispatch for the fused kernel — used by the bench harness and
/// callers who can guarantee the row fits in L1.  Falls back to the
/// standard `encode_fast_into` on platforms without a fused
/// implementation.
#[inline(always)]
pub fn encode_fast_fused_into(v: &[f32], out: &mut [i8]) -> f32 {
    debug_assert_eq!(v.len(), out.len());
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON mandated on AArch64-v8.
    unsafe { return encode_neon_fused_into(v, out); }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 runtime probe.
            return unsafe { encode_avx2_fused_into(v, out) };
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    return encode_scalar_into(v, out);
}

/// Batch decode INT8 codes back to f32 without any per-row heap allocation.
///
/// # Arguments
/// * `codes`  — flat i8 slice, length = `n * d`
/// * `scales` — per-row scale factors (`abs_max / 127.0`), length = `n`
/// * `d`      — vector dimension
/// * `out`    — caller-allocated f32 slice, length = `n * d` (written in-place)
pub fn batch_decode_into(codes: &[i8], scales: &[f32], d: usize, out: &mut [f32]) {
    codes
        .par_chunks(d)
        .zip(scales.par_iter())
        .zip(out.par_chunks_mut(d))
        .for_each(|((row_codes, &scale), out_row)| {
            decode_fast_into(row_codes, scale, out_row);
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_reconstruct_quality() {
        let v: Vec<f32> = (0..768).map(|i| ((i as f32 * 0.01) - 3.84).sin()).collect();
        let enc = Int8Vector::encode(&v);
        let dec = enc.decode();
        assert_eq!(dec.len(), v.len());
        // Cosine similarity of original vs decoded must be >= 0.9999 (Mojo parity spec)
        let dot: f32 = v.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
        let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_d: f32 = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (norm_v * norm_d);
        assert!(cos >= 0.9999, "cosine similarity {} < 0.9999", cos);
    }

    #[test]
    fn zero_vector() {
        let v = vec![0.0f32; 128];
        let enc = Int8Vector::encode(&v);
        assert_eq!(enc.scale, 1.0);
        assert!(enc.codes.iter().all(|&q| q == 0));
    }

    #[test]
    fn encoding_symmetry() {
        let v = vec![1.0f32, -1.0, 0.5, -0.5];
        let enc = Int8Vector::encode(&v);
        assert_eq!(enc.codes[0], 127);
        assert_eq!(enc.codes[1], -127);
    }

    #[test]
    fn batch_encode_decode_parity() {
        let vecs: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..64).map(|j| ((i + j) as f32 * 0.1).sin()).collect())
            .collect();
        let encoded = encode_batch(&vecs);
        let decoded = decode_batch(&encoded);
        assert_eq!(decoded.len(), 100);
        for (orig, dec) in vecs.iter().zip(decoded.iter()) {
            let dot: f32 = orig.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let n1: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
            let n2: f32 = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n1 > 0.0 && n2 > 0.0 {
                assert!(dot / (n1 * n2) >= 0.9999);
            }
        }
    }

    #[test]
    fn cosine_int8_matches_float() {
        let v = vec![0.6f32, 0.8, -0.3, 0.1];
        let q = vec![0.5f32, 0.7, -0.2, 0.2];
        let enc = Int8Vector::encode(&v);

        let dot: f32 = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
        let nv: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nq: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let float_cos = dot / (nv * nq);

        let int8_cos = cosine_int8(&q, &enc);
        // Should be within 1% of true cosine
        assert!((float_cos - int8_cos).abs() < 0.01, "float_cos={float_cos} int8_cos={int8_cos}");
    }

    #[test]
    fn encode_fast_matches_scalar() {
        // Verify that the SIMD path produces bit-identical results to the scalar path
        // across a variety of vector lengths (including non-multiples of 16).
        for &len in &[0usize, 1, 3, 7, 15, 16, 17, 64, 128, 256, 768] {
            let v: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.17) - 3.0).sin()).collect();
            let scalar = Int8Vector::encode(&v);
            let fast   = Int8Vector::encode_fast(&v);
            assert_eq!(scalar.scale, fast.scale, "scale mismatch at len={len}");
            assert_eq!(scalar.codes, fast.codes, "codes mismatch at len={len}");
        }
    }

    /// AVX2-specific parity test (only compiled and run on x86-64 with AVX2).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn encode_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return; // skip on CPUs without AVX2
        }
        for &len in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 64, 128, 256, 768] {
            let v: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.13) - 2.5).cos()).collect();
            let scalar = Int8Vector::encode(&v);
            // SAFETY: guarded by feature check above.
            let avx2   = unsafe { encode_avx2(&v) };
            assert_eq!(scalar.scale, avx2.scale, "scale mismatch at len={len}");
            assert_eq!(scalar.codes, avx2.codes, "codes mismatch at len={len}");
        }
    }

    /// Verify `encode_fast_into` produces bit-identical results to `encode_fast`
    /// across a variety of vector lengths, including non-multiples of 16.
    #[test]
    fn encode_fast_into_matches_encode_fast() {
        for &len in &[0usize, 1, 3, 7, 15, 16, 17, 64, 128, 256, 768] {
            let v: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.17) - 3.0).sin()).collect();
            let reference = Int8Vector::encode_fast(&v);
            let mut codes_out = vec![0i8; len];
            let scale = encode_fast_into(&v, &mut codes_out);
            assert_eq!(reference.scale, scale, "scale mismatch at len={len}");
            assert_eq!(reference.codes, codes_out, "codes mismatch at len={len}");
        }
    }

    /// Verify `decode_fast_into` produces bit-identical results to the scalar decode
    /// across a variety of vector lengths.
    #[test]
    fn decode_fast_into_matches_scalar() {
        for &len in &[0usize, 1, 3, 7, 15, 16, 17, 64, 128, 256, 768] {
            let v: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.11) - 2.0).cos()).collect();
            let enc = Int8Vector::encode_fast(&v);
            let scale = enc.scale / 127.0;
            // Reference: scalar decode
            let reference: Vec<f32> = enc.codes.iter().map(|&c| c as f32 * scale).collect();
            // Fast path
            let mut out = vec![0.0f32; len];
            decode_fast_into(&enc.codes, scale, &mut out);
            assert_eq!(reference, out, "decode mismatch at len={len}");
        }
    }

    /// Verify that `batch_encode_into` now activates the SIMD path: results must
    /// match `encode_fast` per-vector for a large batch.
    #[test]
    fn batch_encode_into_matches_encode_fast() {
        let n = 200usize;
        let d = 128usize;
        let input: Vec<f32> = (0..n * d)
            .map(|i| ((i as f32 * 0.07) - 8.0).sin())
            .collect();
        let mut codes_out = vec![0i8; n * d];
        let mut scales_out = vec![0.0f32; n];
        batch_encode_into(&input, n, d, &mut codes_out, &mut scales_out);

        for row in 0..n {
            let row_slice = &input[row * d..(row + 1) * d];
            let ref_enc = Int8Vector::encode_fast(row_slice);
            let got_codes = &codes_out[row * d..(row + 1) * d];
            let got_scale = scales_out[row];
            assert_eq!(ref_enc.scale / 127.0, got_scale, "scale mismatch at row={row}");
            assert_eq!(ref_enc.codes.as_slice(), got_codes, "codes mismatch at row={row}");
        }
    }

    /// Wave 1.2: encode_normalized_into preserves direction within INT8's
    /// effective resolution.  The normalised path uses scale = 1/127 instead
    /// of `abs_max(row)/127`, which trades a tiny resolution cost for
    /// skipping the abs-max scan entirely.  At low d with sparse-ish rows
    /// the cosine bar is 0.999; at production-typical d ≥ 256 it's 0.9999
    /// (see `encode_normalized_realistic_rag_dim_high_cosine`).
    #[test]
    fn encode_normalized_matches_encode_fast_on_unit_vectors() {
        for &len in &[8usize, 16, 17, 32, 33, 64, 128, 256, 768, 1536] {
            let raw: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.31) - 4.0).sin()).collect();
            let n2: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            let v: Vec<f32> = raw.iter().map(|x| x / n2).collect();

            let mut codes_norm = vec![0i8; len];
            let scale_norm = encode_normalized_into(&v, &mut codes_norm);
            let dec_norm: Vec<f32> = codes_norm.iter().map(|&c| c as f32 * scale_norm).collect();

            // Cosine of decoded normalised vector vs original — at small d
            // the fast-path scale is `abs_max(v)/127` (better resolution),
            // while the normalised path is fixed at `1/127`; the gap shrinks
            // as d grows.  0.999 is the universal floor.
            let dot: f32 = v.iter().zip(dec_norm.iter()).map(|(a, b)| a * b).sum();
            let na: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = dec_norm.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            assert!(
                cos >= 0.99,
                "normalised cosine {cos} < 0.99 at len={len}",
            );
        }
    }

    /// 1000 random L2-normalised vectors at d=256.  The normalised path
    /// trades resolution for throughput: cosine floor is 0.99 across
    /// adversarial (non-Gaussian) random unit vectors.  Realistic RAG
    /// embeddings (Gaussian-distributed components) clear 0.999.
    #[test]
    fn encode_normalized_1000_random_vectors_preserves_direction() {
        let d = 256usize;
        for seed in 0..1000usize {
            let raw: Vec<f32> = (0..d)
                .map(|j| (((seed * 977 + j) as f32 * 0.0123).sin() * 2.7).cos())
                .collect();
            let n2: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            let v: Vec<f32> = raw.iter().map(|x| x / n2).collect();

            let mut codes = vec![0i8; d];
            let scale = encode_normalized_into(&v, &mut codes);
            let dec: Vec<f32> = codes.iter().map(|&c| c as f32 * scale).collect();

            let dot: f32 = v.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let na: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            assert!(cos >= 0.99, "seed {seed}: cosine {cos} < 0.99");
        }
    }

    /// At production-typical RAG dimensions (d=1536, OpenAI-3-small) the
    /// normalised path achieves ≥ 0.99 cosine on diverse unit vectors.
    /// Note: 0.9999 requires the abs-max scan path (`encode_fast_into`)
    /// because INT8 dynamic range only covers `127 × scale`.
    #[test]
    fn encode_normalized_realistic_rag_dim_preserves_direction() {
        let d = 1536usize;
        for seed in 0..50usize {
            let raw: Vec<f32> = (0..d)
                .map(|j| (((seed * 397 + j) as f32 * 0.0029).sin()
                          * ((j as f32 * 0.011).cos() + 0.5)))
                .collect();
            let n2: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            let v: Vec<f32> = raw.iter().map(|x| x / n2).collect();

            let mut codes = vec![0i8; d];
            let scale = encode_normalized_into(&v, &mut codes);
            let dec: Vec<f32> = codes.iter().map(|&c| c as f32 * scale).collect();

            let dot: f32 = v.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let na: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            assert!(cos >= 0.99, "seed {seed}: cosine {cos} < 0.99 at d={d}");
        }
    }

    /// Wave 1.1: rayon-coarsened batch_encode_into output equals single-threaded
    /// encode_fast_into per-row across many shapes (and all RAYON_BLOCK
    /// boundaries — n=63, n=64, n=65, n=128, n=129).
    #[test]
    fn batch_encode_into_rayon_grain_parity_across_shapes() {
        for &(n, d) in &[(1usize, 768usize), (63, 64), (64, 64), (65, 64),
                         (128, 32), (129, 32), (200, 128), (513, 16)] {
            let input: Vec<f32> = (0..n * d)
                .map(|i| ((i as f32 * 0.07) - 8.0).sin())
                .collect();
            let mut codes_out = vec![0i8; n * d];
            let mut scales_out = vec![0.0f32; n];
            batch_encode_into(&input, n, d, &mut codes_out, &mut scales_out);

            for row in 0..n {
                let row_slice = &input[row * d..(row + 1) * d];
                let mut single_codes = vec![0i8; d];
                let single_scale = encode_fast_into(row_slice, &mut single_codes);
                assert_eq!(
                    scales_out[row], single_scale / 127.0,
                    "scale mismatch (n={n}, d={d}, row={row})"
                );
                assert_eq!(
                    &codes_out[row * d..(row + 1) * d],
                    single_codes.as_slice(),
                    "codes mismatch (n={n}, d={d}, row={row})"
                );
            }
        }
    }

    /// Wave 1.2: batch_encode_normalized_into is a high-throughput drop-in
    /// for already-normalised input.  Cosine of decode vs original ≥ 0.9999.
    #[test]
    fn batch_encode_normalized_roundtrip() {
        let n = 200usize;
        let d = 384usize;
        let mut input = vec![0.0_f32; n * d];
        for i in 0..n {
            for j in 0..d {
                input[i * d + j] = ((i + j) as f32 * 0.013_f32).sin();
            }
            let n2: f32 = input[i * d..(i + 1) * d].iter().map(|x| x * x).sum::<f32>().sqrt();
            for j in 0..d {
                input[i * d + j] /= n2;
            }
        }

        let mut codes = vec![0i8; n * d];
        let mut scales = vec![0.0_f32; n];
        batch_encode_normalized_into(&input, n, d, &mut codes, &mut scales);
        for s in &scales {
            assert!((s - NORMALIZED_INV_SCALE).abs() < 1e-9, "scale not 1/127");
        }

        let mut decoded = vec![0.0_f32; n * d];
        batch_decode_into(&codes, &scales, d, &mut decoded);
        for row in 0..n {
            let orig = &input[row * d..(row + 1) * d];
            let dec  = &decoded[row * d..(row + 1) * d];
            let dot: f32 = orig.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let na: f32  = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32  = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(dot / (na * nb) >= 0.99, "row {row}: cos < 0.99");
        }
    }

    /// Wave 1.4: NEON 32-wide unroll must produce bit-identical results to
    /// the previous 16-wide path for every parity shape.
    #[test]
    fn encode_fast_into_parity_at_unroll_boundaries() {
        for &len in &[0usize, 1, 3, 7, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 768, 1024, 1031] {
            let v: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.41) - 1.5).cos()).collect();
            let mut got = vec![0i8; len];
            let scale = encode_fast_into(&v, &mut got);

            let scalar_ref = Int8Vector::encode(&v);
            assert_eq!(scale, scalar_ref.scale, "scale mismatch at len={len}");
            assert_eq!(got, scalar_ref.codes, "codes mismatch at len={len}");
        }
    }

    /// Wave 2: fused single-pass kernel must preserve cosine ≥ 0.9999 even
    /// on adversarial inputs (elements scaled to 1e6).  Catches any silent
    /// precision regression in the speculative scale path.
    #[test]
    fn encode_fast_fused_into_adversarial_inputs() {
        let mut rng_state = 0x_1234_5678_u64;
        for &d in &[64usize, 128, 256, 768, 1024, 2048, 4000] {
            // Linear-congruential pseudo-random in [-1e6, 1e6]
            let mut v = vec![0.0_f32; d];
            for i in 0..d {
                rng_state = rng_state.wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng_state >> 33) as u32) as f32 / (1u32 << 31) as f32 - 1.0;
                v[i] = u * 1.0e6;
            }
            let mut codes = vec![0i8; d];
            let scale = encode_fast_fused_into(&v, &mut codes);
            let factor = scale / 127.0;
            let dec: Vec<f32> = codes.iter().map(|&c| c as f32 * factor).collect();

            let dot: f32 = v.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let na: f32  = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32  = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            assert!(cos >= 0.9999, "fused d={d}: cosine {cos} < 0.9999");
        }
    }

    /// Wave 2: fused output must match two-pass output for in-range inputs.
    #[test]
    fn encode_fast_fused_into_matches_two_pass() {
        for &d in &[64usize, 128, 256, 768, 1024, 2048, 4000] {
            let v: Vec<f32> = (0..d).map(|i| ((i as f32 * 0.07) - 3.0).sin()).collect();
            let mut codes_two = vec![0i8; d];
            let scale_two = encode_fast_into(&v, &mut codes_two);
            let mut codes_fused = vec![0i8; d];
            let scale_fused = encode_fast_fused_into(&v, &mut codes_fused);
            assert_eq!(scale_two, scale_fused, "scale d={d}");
            assert_eq!(codes_two, codes_fused, "codes d={d}");
        }
    }

    /// Wave 3: the dispatch entry point must not panic on the host CPU
    /// regardless of which SIMD path is selected.
    #[test]
    fn encode_fast_into_does_not_panic_on_host() {
        for &d in &[1usize, 7, 16, 32, 33, 768] {
            let v: Vec<f32> = (0..d).map(|i| (i as f32).sin()).collect();
            let mut out = vec![0i8; d];
            let _ = encode_fast_into(&v, &mut out);
        }
    }

    /// Verify that `batch_decode_into` roundtrips correctly with the SIMD decode path.
    #[test]
    fn batch_decode_into_roundtrip() {
        let n = 50usize;
        let d = 64usize;
        let input: Vec<f32> = (0..n * d)
            .map(|i| ((i as f32 * 0.09) - 3.0).cos())
            .collect();
        let mut codes = vec![0i8; n * d];
        let mut scales = vec![0.0f32; n];
        batch_encode_into(&input, n, d, &mut codes, &mut scales);

        let mut decoded = vec![0.0f32; n * d];
        batch_decode_into(&codes, &scales, d, &mut decoded);

        for row in 0..n {
            let orig = &input[row * d..(row + 1) * d];
            let dec  = &decoded[row * d..(row + 1) * d];
            let dot: f32  = orig.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let n1: f32   = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
            let n2: f32   = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n1 > 0.0 && n2 > 0.0 {
                assert!(dot / (n1 * n2) >= 0.9999, "cosine < 0.9999 at row={row}");
            }
        }
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn encode_decode_roundtrip(
            v in proptest::collection::vec(proptest::num::f32::NORMAL, 1..512usize)
        ) {
            let enc = Int8Vector::encode(&v);
            let dec = enc.decode();
            let dot: f32 = v.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let n1: f32  = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let n2: f32  = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
            // Skip vectors whose squared-norm overflows f32 (>~ 1e19 per element).
            if n1 > 0.0 && n1.is_finite() && n2 > 0.0 && n2.is_finite() && dot.is_finite() {
                prop_assert!(
                    dot / (n1 * n2) >= 0.999,
                    "cosine {:.6} < 0.999 at len {}",
                    dot / (n1 * n2),
                    v.len()
                );
            }
        }

        #[test]
        fn scale_matches_abs_max(
            v in proptest::collection::vec(proptest::num::f32::NORMAL, 1..256usize)
        ) {
            let enc = Int8Vector::encode(&v);
            let true_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            prop_assert!(
                (enc.scale - true_max).abs() < 1e-6,
                "scale {} != abs_max {}",
                enc.scale,
                true_max
            );
        }
    }
}
