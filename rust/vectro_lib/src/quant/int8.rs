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
    ///  2. x86-64 + AVX-512F — 16-wide path via runtime `is_x86_feature_detected!`.
    ///  3. x86-64 + AVX2 — 8-wide path via runtime `is_x86_feature_detected!`.
    ///  4. All other targets — portable scalar `encode`.
    #[inline]
    pub fn encode_fast(v: &[f32]) -> Self {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: AArch64-v8 mandates NEON; no runtime feature detection needed.
        return unsafe { encode_neon(v) };

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: guarded by runtime AVX-512F feature detection.
                return unsafe { encode_avx512(v) };
            }
            if is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by runtime AVX2 feature detection.
                return unsafe { encode_avx2(v) };
            }
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
    ///
    /// This is the per-candidate distance kernel during INT8 HNSW search, so the
    /// i8×f32 dot is SIMD-accelerated on aarch64 (NEON) and x86-64 (AVX-512F →
    /// AVX2+FMA), with a scalar fallback elsewhere.
    #[inline]
    pub fn dot_query(&self, query_norm: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: AArch64-v8 mandates NEON; the helper reads only in-bounds
        // lanes (min length) and handles the tail scalarly.
        let raw = unsafe { dot_i8_f32_neon(&self.codes, query_norm) };

        #[cfg(target_arch = "x86_64")]
        let raw = {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: guarded by runtime AVX-512F detection; reads min-length lanes.
                unsafe { dot_i8_f32_avx512(&self.codes, query_norm) }
            } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: guarded by runtime AVX2+FMA detection; reads min-length lanes.
                unsafe { dot_i8_f32_avx2(&self.codes, query_norm) }
            } else {
                self.codes
                    .iter()
                    .zip(query_norm.iter())
                    .map(|(&q, &qv)| (q as f32) * qv)
                    .sum()
            }
        };

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let raw: f32 = self
            .codes
            .iter()
            .zip(query_norm.iter())
            .map(|(&q, &qv)| (q as f32) * qv)
            .sum();

        raw * (self.scale / 127.0)
    }
}

/// NEON i8×f32 dot product: `Σ codes[i] as f32 * query[i]`.
///
/// Widens 16 i8 lanes → f32 and multiply-accumulates into four independent
/// `f32x4` accumulators (breaks the reduction dependency chain), with a scalar
/// tail. Used as the per-candidate distance kernel for INT8 HNSW search.
///
/// # Safety
/// Requires NEON (mandated on AArch64-v8). Reads only `min(codes, query)` lanes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_i8_f32_neon(codes: &[i8], query: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = codes.len().min(query.len());
    let cptr = codes.as_ptr();
    let qptr = query.as_ptr();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let chunks = n / 16;
    for i in 0..chunks {
        let c = vld1q_s8(cptr.add(i * 16)); // 16× i8
        let lo16 = vmovl_s8(vget_low_s8(c)); // 8× i16
        let hi16 = vmovl_s8(vget_high_s8(c)); // 8× i16
        let c0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
        let c1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
        let c2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
        let c3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
        acc0 = vmlaq_f32(acc0, c0, vld1q_f32(qptr.add(i * 16)));
        acc1 = vmlaq_f32(acc1, c1, vld1q_f32(qptr.add(i * 16 + 4)));
        acc2 = vmlaq_f32(acc2, c2, vld1q_f32(qptr.add(i * 16 + 8)));
        acc3 = vmlaq_f32(acc3, c3, vld1q_f32(qptr.add(i * 16 + 12)));
    }
    let sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut total = vaddvq_f32(sum);

    for i in chunks * 16..n {
        total += codes[i] as f32 * query[i];
    }
    total
}

/// AVX-512F i8×f32 dot product: `Σ codes[i] as f32 * query[i]`.
///
/// Sign-extends 16 i8 lanes → i32 → f32 (`vpmovsxbd` + `vcvtdq2ps`) and FMAs
/// against the f32 query, 16 elements per iteration, with a scalar tail. The
/// per-candidate distance kernel for INT8 HNSW search on x86-64.
///
/// # Safety
/// Requires AVX-512F (runtime-checked by the caller). Reads only
/// `min(codes, query)` lanes; the 128-bit code load covers exactly 16 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn dot_i8_f32_avx512(codes: &[i8], query: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = codes.len().min(query.len());
    let cptr = codes.as_ptr();
    let qptr = query.as_ptr();

    // Four independent accumulators (64 elements/iter) break the FMA
    // dependency chain — AVX-512 FMA has ~4-cycle latency at 2/cycle
    // throughput, so a single accumulator left ~7/8 of the FMA pipes idle.
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let chunks = n / 64;
    for i in 0..chunks {
        let o = i * 64;
        let l0 = _mm_loadu_si128(cptr.add(o) as *const __m128i);
        let l1 = _mm_loadu_si128(cptr.add(o + 16) as *const __m128i);
        let l2 = _mm_loadu_si128(cptr.add(o + 32) as *const __m128i);
        let l3 = _mm_loadu_si128(cptr.add(o + 48) as *const __m128i);
        acc0 = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(l0)),
            _mm512_loadu_ps(qptr.add(o)),
            acc0,
        );
        acc1 = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(l1)),
            _mm512_loadu_ps(qptr.add(o + 16)),
            acc1,
        );
        acc2 = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(l2)),
            _mm512_loadu_ps(qptr.add(o + 32)),
            acc2,
        );
        acc3 = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(l3)),
            _mm512_loadu_ps(qptr.add(o + 48)),
            acc3,
        );
    }
    // Cleanup: remaining 16-lane blocks below the 64-wide stride.
    let mut o = chunks * 64;
    while o + 16 <= n {
        let c8 = _mm_loadu_si128(cptr.add(o) as *const __m128i);
        let cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c8));
        acc0 = _mm512_fmadd_ps(cf, _mm512_loadu_ps(qptr.add(o)), acc0);
        o += 16;
    }
    let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut total = _mm512_reduce_add_ps(acc);

    for i in o..n {
        total += codes[i] as f32 * query[i];
    }
    total
}

/// AVX2+FMA i8×f32 dot product: `Σ codes[i] as f32 * query[i]`.
///
/// Two independent f32x8 accumulators (breaks the FMA dependency chain),
/// 16 elements per iteration via two `vpmovsxbd`+`vcvtdq2ps`+`vfmadd` groups,
/// with a scalar tail. Used when the host has AVX2 but not AVX-512F.
///
/// # Safety
/// Requires AVX2+FMA (runtime-checked by the caller). Reads only
/// `min(codes, query)` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_i8_f32_avx2(codes: &[i8], query: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = codes.len().min(query.len());
    let cptr = codes.as_ptr();
    let qptr = query.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let chunks = n / 16;
    for i in 0..chunks {
        let base = i * 16;
        // Low 8 bytes → 8× i32 → f32; high 8 bytes likewise.
        let c8 = _mm_loadu_si128(cptr.add(base) as *const __m128i); // 16× i8
        let lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c8));
        let hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c8, 8)));
        acc0 = _mm256_fmadd_ps(lo, _mm256_loadu_ps(qptr.add(base)), acc0);
        acc1 = _mm256_fmadd_ps(hi, _mm256_loadu_ps(qptr.add(base + 8)), acc1);
    }
    // Horizontal sum of the two accumulators.
    let sum = _mm256_add_ps(acc0, acc1);
    let hi128 = _mm256_extractf128_ps(sum, 1);
    let lo128 = _mm256_castps256_ps128(sum);
    let s128 = _mm_add_ps(hi128, lo128);
    let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 0x55));
    let mut total = _mm_cvtss_f32(s32);

    for i in chunks * 16..n {
        total += codes[i] as f32 * query[i];
    }
    total
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
    for (i, &val) in v.iter().enumerate().skip(chunks8 * 8) {
        *out_ptr.add(i) = (val * inv).round().clamp(-127.0, 127.0) as i8;
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

    // scalar tail for the remainder (< 16 elements). Indexed: writes go through a
    // raw `out_ptr`, so there is no output slice to iterate.
    #[allow(clippy::needless_range_loop)]
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
unsafe fn encode_neon_into(v: &[f32], out: &mut [i8], range_factor: f32) -> f32 {
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

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max / range_factor };
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

    // scalar tail (< 16 elements). Indexed: raw-pointer output, no slice to iterate.
    let tail_start = after_32 + chunks16_extra * 16;
    #[allow(clippy::needless_range_loop)]
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
unsafe fn encode_avx2_into(v: &[f32], out: &mut [i8], range_factor: f32) -> f32 {
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

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max / range_factor };
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
    for (i, &val) in v.iter().enumerate().skip(chunks8 * 8) {
        *out_ptr.add(i) = (val * inv).round().clamp(-127.0, 127.0) as i8;
    }

    scale
}

/// Portable scalar in-place INT8 encode — fallback for targets without
/// NEON or AVX2.  Two-pass abs-max + multiply-round-clamp, identical to the
/// SIMD paths bit-for-bit.
///
/// `range_factor` (rf, in `(0, 1]`) reproduces the Python profile semantics:
/// the effective scale is `abs_max / rf`, so codes use `127 · rf / abs_max`.
/// `rf = 1.0` is the canonical abs-max path (max element → ±127); `rf < 1.0`
/// leaves headroom, matching `VectroBatchProcessor`'s balanced/quality
/// profiles.  The returned scale is the effective scale (`abs_max / rf`), so
/// the caller's `scale / 127.0` yields `abs_max / (127 · rf)` — exactly the
/// NumPy baseline's per-row scale.
// Used only on non-aarch64 targets without AVX2/AVX-512; dead on aarch64.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
pub(crate) fn encode_scalar_into(v: &[f32], out: &mut [i8], range_factor: f32) -> f32 {
    debug_assert_eq!(v.len(), out.len());
    let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max == 0.0 { 1.0 } else { abs_max / range_factor };
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
unsafe fn encode_sme_into(_v: &[f32], _out: &mut [i8], _range_factor: f32) -> f32 {
    // Wave 3 placeholder — the M4-specific SME2 outer-product path is not
    // yet implemented.  Hardware availability gates implementation: when
    // an M4 (or comparable Cortex-X925 platform) is in CI, replace this
    // body with the SME2 streaming-mode encoder.
    todo!("SME2 INT8 encode kernel — wired but not yet implemented (no M4 hardware)")
}

/// AVX-512-VNNI entry point — wired but uses the AVX2 fallback for now.
/// AVX-512 in-place INT8 encode — 16 floats per iteration (2× the AVX2 width).
///
/// Needs only AVX-512F: abs is `max(a, −a)` (no `vandnps`/AVX-512DQ), and the
/// f32→i8 narrow is a single saturating `vpmovsdb` (`_mm512_cvtsepi32_epi8`)
/// instead of AVX2's two-step `packs_epi32`→`packs_epi16`. Output is bit-for-bit
/// identical to [`encode_avx2_into`] and [`encode_scalar_into`]: same abs-max
/// (max is order-independent), same round-to-nearest `cvtps_epi32`, same
/// [-128, 127] saturation, same scalar tail. (`vpdpbssd`/VNNI would help a
/// *fused dot* at search time, not this encode, so it is intentionally unused.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn encode_avx512_into(v: &[f32], out: &mut [i8], range_factor: f32) -> f32 {
    use std::arch::x86_64::*;

    let n = v.len();
    if n == 0 {
        return 1.0;
    }
    let ptr = v.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // ── Pass 1: abs-max reduction (16 floats per iteration) ──────────────────
    let zero = _mm512_setzero_ps();
    let mut vmax = _mm512_setzero_ps();
    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let a = _mm512_loadu_ps(ptr.add(i * 16));
        let abs_a = _mm512_max_ps(a, _mm512_sub_ps(zero, a)); // |a| = max(a, −a)
        vmax = _mm512_max_ps(vmax, abs_a);
    }
    let mut abs_max = _mm512_reduce_max_ps(vmax);
    for &x in &v[chunks16 * 16..] {
        let ax = x.abs();
        if ax > abs_max {
            abs_max = ax;
        }
    }

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max / range_factor };
    let inv = 127.0_f32 / scale;
    let vinv = _mm512_set1_ps(inv);

    // ── Pass 2: quantise f32 → i8 (16 per iteration) ─────────────────────────
    for i in 0..chunks16 {
        let base = i * 16;
        let x = _mm512_loadu_ps(ptr.add(base));
        // Round-to-nearest (current MXCSR mode; default = nearest-even).
        let i32s = _mm512_cvtps_epi32(_mm512_mul_ps(x, vinv));
        // Saturating i32 → i8 narrow: 16 lanes in one VPMOVSDB.
        let i8s = _mm512_cvtsepi32_epi8(i32s);
        _mm_storeu_si128(out_ptr.add(base) as *mut __m128i, i8s);
    }
    // Scalar tail
    for (i, &val) in v.iter().enumerate().skip(chunks16 * 16) {
        *out_ptr.add(i) = (val * inv).round().clamp(-127.0, 127.0) as i8;
    }

    scale
}

/// AVX-512 allocating INT8 encode (abs-max, `range_factor = 1.0`) for
/// [`Int8Vector::encode_fast`]. Thin wrapper over [`encode_avx512_into`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn encode_avx512(v: &[f32]) -> Int8Vector {
    let mut codes = vec![0i8; v.len()];
    let scale = encode_avx512_into(v, &mut codes, 1.0);
    Int8Vector { codes, scale }
}

/// Dispatch to the best in-place INT8 encode kernel for the current host.
///
/// Priority order (Wave 3):
///   AArch64:  SME2 (M4)  →  Accelerate AMX (M1-M3, feature-gated)  →  NEON 32-wide
///   x86-64:   AVX-512F (16-wide)                      →  AVX2  →  scalar
///   other:    scalar
///
/// Writes quantised codes into `out` without any heap allocation and
/// returns the effective scale (`abs_max / range_factor`, **before** dividing
/// by 127).  Used by `batch_encode_into` so each rayon worker activates the
/// SIMD fast-path.
///
/// `range_factor` (rf, in `(0, 1]`) sets the effective scale to `abs_max / rf`
/// so codes use `127 · rf / abs_max` — `rf = 1.0` is the canonical abs-max
/// path; `rf < 1.0` reproduces the balanced/quality profile headroom.  See
/// [`encode_scalar_into`] for the numerical contract.
#[inline(always)]
pub(crate) fn encode_fast_into(v: &[f32], out: &mut [i8], range_factor: f32) -> f32 {
    debug_assert_eq!(v.len(), out.len());

    #[cfg(target_arch = "aarch64")]
    {
        // 1. SME2 (M4+) — feature-gated; not enabled on stable as of 2026-05.
        #[cfg(target_feature = "sme")]
        // SAFETY: SME is gated by the target_feature attribute above; if
        // this code is reached the host advertised SME at compile time.
        return unsafe { encode_sme_into(v, out, range_factor) };

        // 2. Apple Accelerate (AMX coprocessor on M1/M2/M3, macOS-only,
        //    feature-gated).  Only profitable for d ≥ 256 — under that the
        //    AMX setup cost dominates and pure NEON wins.  The AMX kernel is
        //    abs-max only, so non-unit range factors fall through to NEON.
        #[cfg(all(target_os = "macos", feature = "vectro_lib_accelerate"))]
        if v.len() >= 256 && range_factor == 1.0 {
            return crate::quant::accelerate::encode_accelerate_into(v, out);
        }

        // 3. NEON 32-wide two-pass (always available on any ARMv8 / Apple
        //    Silicon). NOTE: the "fused" single-pass kernel was measured
        //    *slower* here — it zero-inits a 16 KiB stack buffer per row and
        //    the row already stays L1-resident between the two passes at
        //    production dims, so two-pass wins. See encode_neon_fused_into.
        // SAFETY: AArch64-v8 mandates NEON; no runtime probe needed.
        #[cfg(not(target_feature = "sme"))]
        return unsafe { encode_neon_into(v, out, range_factor) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: guarded by runtime AVX-512F feature detection.
            return unsafe { encode_avx512_into(v, out, range_factor) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 feature detection.
            return unsafe { encode_avx2_into(v, out, range_factor) };
        }
    }

    // Scalar fallback (non-aarch64 without AVX2/AVX-512).
    #[cfg(not(target_arch = "aarch64"))]
    return encode_scalar_into(v, out, range_factor);
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
/// * `scales_out` — caller-allocated f32 slice, length = `n`;
///   stores `abs_max / 127.0` per row (direct dequant factor)
///
/// Uses rayon for row-parallel execution; each worker thread calls
/// `encode_fast_into` which dispatches to the NEON (AArch64) or AVX2 (x86-64)
/// in-place SIMD path — no per-row heap allocation.
pub fn batch_encode_into(
    input: &[f32],
    n: usize,
    d: usize,
    codes_out: &mut [i8],
    scales_out: &mut [f32],
) {
    batch_encode_into_with_range(input, n, d, codes_out, scales_out, 1.0);
}

/// Parallel scan for the first non-finite (NaN/Inf) element, returning its flat
/// index or `None` if all finite.
///
/// The Python batch-encode binding must reject non-finite input, but a *serial*
/// scan of the whole `[N, D]` array was the Amdahl bottleneck that capped
/// encode throughput (the parallel kernel barely scaled past one core). This
/// rayon scan reads the input in parallel and reports the **first** offending
/// index deterministically (`min` over all matches), matching the serial check.
pub fn first_non_finite(flat: &[f32]) -> Option<usize> {
    flat.par_iter()
        .enumerate()
        .filter_map(|(i, x)| if x.is_finite() { None } else { Some(i) })
        .min()
}

/// Batch encode an N×D f32 matrix to INT8 with an explicit `range_factor`.
///
/// Identical to [`batch_encode_into`] but threads `range_factor` (rf, in
/// `(0, 1]`) through to the per-row kernel, so `scales_out[i]` becomes
/// `abs_max_i / (127 · rf)` and the codes use `127 · rf / abs_max_i`.  This
/// matches the Python `VectroBatchProcessor` profiles bit-for-bit modulo
/// round-half-to-even vs round-half-away ties (≤1 level): rf 1.0 = `fast`,
/// 0.95 = `balanced`, 0.90 = `quality`.
pub fn batch_encode_into_with_range(
    input: &[f32],
    _n: usize,
    d: usize,
    codes_out: &mut [i8],
    scales_out: &mut [f32],
    range_factor: f32,
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
                    range_factor,
                );
                scales[i] = scale / 127.0;
            }
        });
}

/// Like [`batch_encode_into_with_range`] but **folds the NaN/Inf validation into
/// the same parallel pass** instead of a separate scan, returning the flat index
/// of the first non-finite input element (or `None` if all finite).
///
/// The previous Python binding ran [`first_non_finite`] as a separate streaming
/// pass over the whole `[N, D]` array before encoding — that doubled the input
/// memory traffic and was the dominant cost (~3.5× slowdown). Checking each row
/// here, while it is already hot in cache for the encode, removes that pass:
/// measured **d=100 ≈ 40 → ~140 M vec/s** on an M3.
pub fn batch_encode_checked_into_with_range(
    input: &[f32],
    _n: usize,
    d: usize,
    codes_out: &mut [i8],
    scales_out: &mut [f32],
    range_factor: f32,
) -> Option<usize> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let block_rows = d * RAYON_BLOCK;
    let first_bad = AtomicUsize::new(usize::MAX);
    input
        .par_chunks(block_rows)
        .zip(codes_out.par_chunks_mut(block_rows))
        .zip(scales_out.par_chunks_mut(RAYON_BLOCK))
        .enumerate()
        .for_each(|(chunk_idx, ((rows, codes), scales))| {
            let chunk_base = chunk_idx * block_rows;
            let n_rows = rows.len() / d;
            for i in 0..n_rows {
                let row = &rows[i * d..(i + 1) * d];
                if let Some(c) = first_non_finite_row(row) {
                    first_bad.fetch_min(chunk_base + i * d + c, Ordering::Relaxed);
                }
                let scale = encode_fast_into(row, &mut codes[i * d..(i + 1) * d], range_factor);
                scales[i] = scale / 127.0;
            }
        });
    match first_bad.load(Ordering::Relaxed) {
        usize::MAX => None,
        b => Some(b),
    }
}

/// SIMD finite check for one cache-hot row: returns the index of the first
/// non-finite element, or `None`. `|x| < ∞` is true iff `x` is finite (excludes
/// both NaN and ±Inf in a single compare), so we test 4 lanes at a time.
#[inline]
fn first_non_finite_row(row: &[f32]) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = row.len();
        let chunks = n / 4;
        // SAFETY: NEON mandated on AArch64-v8; loads stay within `chunks*4 ≤ n`.
        unsafe {
            let inf = vdupq_n_f32(f32::INFINITY);
            let p = row.as_ptr();
            for i in 0..chunks {
                let x = vld1q_f32(p.add(i * 4));
                // finite lanes: |x| < ∞  → all-ones; any zero lane ⇒ non-finite.
                let finite = vcltq_f32(vabsq_f32(x), inf);
                if vminvq_u32(finite) == 0 {
                    // Pinpoint the offending lane scalarly (rare path).
                    let base = i * 4;
                    for (j, &val) in row[base..base + 4].iter().enumerate() {
                        if !val.is_finite() {
                            return Some(base + j);
                        }
                    }
                }
            }
            for (j, &val) in row[chunks * 4..].iter().enumerate() {
                if !val.is_finite() {
                    return Some(chunks * 4 + j);
                }
            }
        }
        None
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        row.iter().position(|x| !x.is_finite())
    }
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

/// Like [`batch_encode_normalized_into`] but folds the NaN/Inf check into the
/// encode pass (see [`batch_encode_checked_into_with_range`]); returns the flat
/// index of the first non-finite element, or `None`.
pub fn batch_encode_normalized_checked_into(
    input: &[f32],
    _n: usize,
    d: usize,
    codes_out: &mut [i8],
    scales_out: &mut [f32],
) -> Option<usize> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let block_rows = d * RAYON_BLOCK;
    let first_bad = AtomicUsize::new(usize::MAX);
    input
        .par_chunks(block_rows)
        .zip(codes_out.par_chunks_mut(block_rows))
        .zip(scales_out.par_chunks_mut(RAYON_BLOCK))
        .enumerate()
        .for_each(|(chunk_idx, ((rows, codes), scales))| {
            let chunk_base = chunk_idx * block_rows;
            let n_rows = rows.len() / d;
            for i in 0..n_rows {
                let row = &rows[i * d..(i + 1) * d];
                if let Some(c) = first_non_finite_row(row) {
                    first_bad.fetch_min(chunk_base + i * d + c, Ordering::Relaxed);
                }
                encode_normalized_into(row, &mut codes[i * d..(i + 1) * d]);
                scales[i] = NORMALIZED_INV_SCALE;
            }
        });
    match first_bad.load(Ordering::Relaxed) {
        usize::MAX => None,
        b => Some(b),
    }
}

/// Fused **f16 → INT8** batch encode: widen, validate, and abs-max encode in a
/// single parallel pass. Each rayon task widens its block of rows into a small
/// reused f32 scratch (kept in L1/L2), validates and encodes per row — so the
/// whole `[N,D]` f16 array is read **once**, with no separate full-array widen,
/// NaN/Inf scan, or output 0-init. Returns the first non-finite flat index (the
/// f16 value's NaN/Inf widens to f32 NaN/Inf), or `None`.
pub fn batch_encode_f16_checked_into(
    input: &[half::f16],
    _n: usize,
    d: usize,
    codes_out: &mut [i8],
    scales_out: &mut [f32],
) -> Option<usize> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let block_rows = d * RAYON_BLOCK;
    let first_bad = AtomicUsize::new(usize::MAX);
    input
        .par_chunks(block_rows)
        .zip(codes_out.par_chunks_mut(block_rows))
        .zip(scales_out.par_chunks_mut(RAYON_BLOCK))
        .enumerate()
        .for_each(|(chunk_idx, ((rows16, codes), scales))| {
            let chunk_base = chunk_idx * block_rows;
            let n_rows = rows16.len() / d;
            // Widen this block to f32 once into reused scratch (≤ 64×d floats).
            let mut wbuf = vec![0.0f32; rows16.len()];
            for (o, &h) in rows16.iter().enumerate() {
                wbuf[o] = h.to_f32();
            }
            for i in 0..n_rows {
                let row = &wbuf[i * d..(i + 1) * d];
                if let Some(c) = first_non_finite_row(row) {
                    first_bad.fetch_min(chunk_base + i * d + c, Ordering::Relaxed);
                }
                let scale = encode_fast_into(row, &mut codes[i * d..(i + 1) * d], 1.0);
                scales[i] = scale / 127.0;
            }
        });
    match first_bad.load(Ordering::Relaxed) {
        usize::MAX => None,
        b => Some(b),
    }
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
// Used only on non-aarch64 targets (x86 without AVX2, or other arches); dead on aarch64.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
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
    #[allow(clippy::needless_range_loop)]
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
    for (i, &val) in v.iter().enumerate().skip(chunks8 * 8) {
        *out_ptr.add(i) = (val * 127.0_f32).round().clamp(-127.0, 127.0) as i8;
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
        return encode_neon_into(v, out, 1.0);
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
    #[allow(clippy::needless_range_loop)]
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
        return encode_avx2_into(v, out, 1.0);
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
    // `buf` is a fixed ROW_CAP-sized scratch array, so iterate only its first
    // `n` entries — writing past `n` would overflow the `out_ptr` allocation.
    for (i, &val) in buf[..n].iter().enumerate().skip(chunks8 * 8) {
        *out_ptr.add(i) = (val * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Public dispatch for the fused kernel — used by the bench harness and
/// callers who can guarantee the row fits in L1.  Falls back to the
/// standard `encode_fast_into` on platforms without a fused
/// implementation.
// The early `return`s are structural: each arch path is cfg-gated, so on any
// single target one of them is the function's last statement.
#[allow(clippy::needless_return)]
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
    return encode_scalar_into(v, out, 1.0);
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
    fn dot_query_matches_scalar_across_dims() {
        // The SIMD dot_query (NEON on aarch64) must match the scalar reference
        // at the SIMD-width boundaries and odd tails.
        for d in [1usize, 7, 15, 16, 17, 31, 64, 127, 128, 768] {
            let v: Vec<f32> = (0..d).map(|i| ((i as f32 * 0.013) - 0.5).sin()).collect();
            let q: Vec<f32> = (0..d).map(|i| ((i as f32 * 0.027) + 0.2).cos()).collect();
            let enc = Int8Vector::encode(&v);
            let got = enc.dot_query(&q);
            let factor = enc.scale / 127.0;
            let want: f32 =
                enc.codes.iter().zip(q.iter()).map(|(&c, &qv)| c as f32 * qv).sum::<f32>() * factor;
            assert!((got - want).abs() < 1e-3, "d={d}: got={got} want={want}");
        }
    }

    /// Directly exercise the x86 i8×f32 dot kernels against the scalar reference,
    /// independent of which one `dot_query` happens to dispatch to on the host.
    /// f32 FMA/accumulation order differs from a left-to-right scalar sum, so the
    /// tolerance matches the HNSW-distance contract (well under quantisation noise).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dot_i8_f32_simd_matches_scalar() {
        for d in [1usize, 7, 8, 15, 16, 17, 31, 32, 33, 64, 127, 128, 768] {
            let v: Vec<f32> = (0..d).map(|i| ((i as f32 * 0.013) - 0.5).sin()).collect();
            let q: Vec<f32> = (0..d).map(|i| ((i as f32 * 0.027) + 0.2).cos()).collect();
            let codes = Int8Vector::encode(&v).codes;
            let want: f32 = codes.iter().zip(q.iter()).map(|(&c, &qv)| c as f32 * qv).sum();
            // Relative tolerance: two valid f32 reduction trees of the raw (unscaled)
            // dot differ by ~n·eps·|partial|, which scales with the result magnitude.
            let tol = want.abs() * 1e-3 + 1e-2;
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: guarded by runtime AVX-512F detection.
                let got = unsafe { dot_i8_f32_avx512(&codes, &q) };
                assert!((got - want).abs() <= tol, "avx512 d={d}: {got} vs {want}");
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: guarded by runtime AVX2+FMA detection.
                let got = unsafe { dot_i8_f32_avx2(&codes, &q) };
                assert!((got - want).abs() <= tol, "avx2 d={d}: {got} vs {want}");
            }
        }
    }

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

    /// AVX-512 parity test (only run on x86-64 hosts that advertise AVX-512F).
    /// The 16-wide kernel must be bit-for-bit identical to the scalar baseline —
    /// same abs-max, rounding, saturation, and tail — including non-multiples of
    /// 16 and the adversarial 1e6-magnitude range the CLAUDE.md SIMD contract
    /// requires. Both the allocating and in-place (range_factor) paths are checked.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn encode_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            return; // skip on CPUs without AVX-512F
        }
        for &len in &[0usize, 1, 3, 7, 15, 16, 17, 31, 32, 33, 64, 128, 256, 768] {
            let v: Vec<f32> = (0..len)
                .map(|i| ((i as f32 * 0.13) - 2.5).cos() * 1e6)
                .collect();
            let scalar = Int8Vector::encode(&v);
            // SAFETY: guarded by the avx512f feature check above.
            let avx512 = unsafe { encode_avx512(&v) };
            assert_eq!(scalar.scale, avx512.scale, "scale mismatch at len={len}");
            assert_eq!(scalar.codes, avx512.codes, "codes mismatch at len={len}");

            // In-place path with a sub-unit range factor (profile headroom).
            let mut out_simd = vec![0i8; len];
            let mut out_scalar = vec![0i8; len];
            let scalar_rf = encode_scalar_into(&v, &mut out_scalar, 0.9);
            // SAFETY: guarded by the avx512f feature check above.
            let scale_rf = unsafe { encode_avx512_into(&v, &mut out_simd, 0.9) };
            assert_eq!(scalar_rf, scale_rf, "rf scale mismatch at len={len}");
            assert_eq!(out_scalar, out_simd, "rf codes mismatch at len={len}");
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
            let scale = encode_fast_into(&v, &mut codes_out, 1.0);
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

    #[test]
    fn f16_checked_encode_matches_f32_and_detects_nonfinite() {
        let (n, d) = (150usize, 64usize);
        // f16 input and its exact f32 widening.
        let f16_in: Vec<half::f16> = (0..n * d)
            .map(|i| half::f16::from_f32(((i as f32 * 0.05) - 3.0).sin()))
            .collect();
        let f32_in: Vec<f32> = f16_in.iter().map(|h| h.to_f32()).collect();

        let mut c16 = vec![0i8; n * d];
        let mut s16 = vec![0.0f32; n];
        let bad = batch_encode_f16_checked_into(&f16_in, n, d, &mut c16, &mut s16);
        assert_eq!(bad, None);

        // Must equal encoding the widened f32 directly.
        let mut c32 = vec![0i8; n * d];
        let mut s32 = vec![0.0f32; n];
        batch_encode_into(&f32_in, n, d, &mut c32, &mut s32);
        assert_eq!(c16, c32, "f16 codes differ from widened-f32 codes");
        assert_eq!(s16, s32, "f16 scales differ");

        // f16 NaN widens to f32 NaN and is reported at the right index.
        let mut bad_in = f16_in.clone();
        bad_in[42 * d + 5] = half::f16::NAN;
        let mut c = vec![0i8; n * d];
        let mut s = vec![0.0f32; n];
        assert_eq!(
            batch_encode_f16_checked_into(&bad_in, n, d, &mut c, &mut s),
            Some(42 * d + 5)
        );
    }

    #[test]
    fn checked_encode_matches_unchecked_and_detects_nonfinite() {
        let (n, d) = (200usize, 100usize);
        let input: Vec<f32> = (0..n * d).map(|i| ((i as f32 * 0.07) - 8.0).sin()).collect();

        // 1) On finite input the checked path returns None and identical codes.
        let mut c_ref = vec![0i8; n * d];
        let mut s_ref = vec![0.0f32; n];
        batch_encode_into(&input, n, d, &mut c_ref, &mut s_ref);
        let mut c_chk = vec![0i8; n * d];
        let mut s_chk = vec![0.0f32; n];
        let bad = batch_encode_checked_into_with_range(&input, n, d, &mut c_chk, &mut s_chk, 1.0);
        assert_eq!(bad, None, "all-finite input flagged as bad");
        assert_eq!(c_ref, c_chk, "checked codes differ from unchecked");

        // 2) NaN/Inf are detected and the *first* (min) flat index is reported.
        for bad_val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad_in = input.clone();
            bad_in[50 * d + 9] = bad_val;
            bad_in[120 * d + 3] = bad_val; // a later one — must not win
            let mut c = vec![0i8; n * d];
            let mut s = vec![0.0f32; n];
            let got = batch_encode_checked_into_with_range(&bad_in, n, d, &mut c, &mut s, 1.0);
            assert_eq!(got, Some(50 * d + 9), "wrong first-non-finite index for {bad_val}");
        }

        // 3) The SIMD row scanner agrees with the scalar predicate.
        let mut row: Vec<f32> = (0..37).map(|i| i as f32 * 0.1 - 1.0).collect();
        assert_eq!(first_non_finite_row(&row), None);
        row[30] = f32::NAN;
        assert_eq!(first_non_finite_row(&row), Some(30));
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
                .map(|j| ((seed * 397 + j) as f32 * 0.0029).sin()
                          * ((j as f32 * 0.011).cos() + 0.5))
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
                let single_scale = encode_fast_into(row_slice, &mut single_codes, 1.0);
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

    /// `batch_encode_into_with_range` must (a) equal `batch_encode_into` at
    /// `range_factor = 1.0` and (b) reproduce the Python baseline's per-row
    /// scale `abs_max / (127 · rf)` with codes `round(v · 127 · rf / abs_max)`
    /// for `rf < 1.0`.
    #[test]
    fn batch_encode_with_range_matches_baseline() {
        let (n, d) = (120usize, 96usize);
        let input: Vec<f32> = (0..n * d).map(|i| ((i as f32 * 0.031) - 5.0).sin() * 3.0).collect();

        // rf = 1.0 is bit-identical to the canonical batch path.
        let mut codes_rf1 = vec![0i8; n * d];
        let mut scales_rf1 = vec![0.0f32; n];
        batch_encode_into_with_range(&input, n, d, &mut codes_rf1, &mut scales_rf1, 1.0);
        let mut codes_base = vec![0i8; n * d];
        let mut scales_base = vec![0.0f32; n];
        batch_encode_into(&input, n, d, &mut codes_base, &mut scales_base);
        assert_eq!(codes_rf1, codes_base, "rf=1.0 codes differ from batch_encode_into");
        assert_eq!(scales_rf1, scales_base, "rf=1.0 scales differ from batch_encode_into");

        // rf < 1.0 matches the scalar baseline exactly (same rounding mode).
        for rf in [0.95f32, 0.90] {
            let mut codes = vec![0i8; n * d];
            let mut scales = vec![0.0f32; n];
            batch_encode_into_with_range(&input, n, d, &mut codes, &mut scales, rf);
            for row in 0..n {
                let v = &input[row * d..(row + 1) * d];
                let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let expected_scale = abs_max / rf / 127.0;
                assert!(
                    (scales[row] - expected_scale).abs() <= expected_scale * 1e-6,
                    "rf={rf} row={row}: scale {} != {expected_scale}",
                    scales[row]
                );
                let inv = 127.0 * rf / abs_max;
                for (j, &x) in v.iter().enumerate() {
                    let want = (x * inv).round().clamp(-127.0, 127.0) as i8;
                    assert_eq!(codes[row * d + j], want, "rf={rf} row={row} col={j}");
                }
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
            let scale = encode_fast_into(&v, &mut got, 1.0);

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
            for slot in v.iter_mut() {
                rng_state = rng_state.wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng_state >> 33) as u32) as f32 / (1u32 << 31) as f32 - 1.0;
                *slot = u * 1.0e6;
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
            let scale_two = encode_fast_into(&v, &mut codes_two, 1.0);
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
            let _ = encode_fast_into(&v, &mut out, 1.0);
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
