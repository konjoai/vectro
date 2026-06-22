//! Shared SIMD f32 distance kernels for the index hot paths.
//!
//! `dot_f32` / `l2_sq` are the single source of truth for the inner-product and
//! squared-Euclidean reductions used by HNSW search and IVF / IVF-PQ
//! coarse-quantiser scans and k-means. They dispatch to a hand-rolled NEON
//! (aarch64) or AVX2+FMA (x86_64, runtime-detected) kernel, falling back to
//! SimSIMD on other targets / older CPUs.
//!
//! Why hand-rolled instead of SimSIMD everywhere: SimSIMD resolves its kernel
//! through a per-call dispatch indirection that dominates at the low dimensions
//! typical of ANN search (d≈100) and in tight loops that call it thousands of
//! times per query (the IVF coarse scan over `n_lists` centroids). A directly
//! `target_feature`-compiled kernel removes that overhead.

#[cfg(not(target_arch = "aarch64"))]
use simsimd::SpatialSimilarity;

/// `Σ a[i]·b[i]` over `min(a, b)` lanes — NEON on aarch64, AVX2+FMA on x86_64
/// (runtime-detected), SimSIMD fallback otherwise.
#[inline]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandated on AArch64-v8; the helper reads in-bounds lanes.
        unsafe { dot_f32_neon(a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        // `is_x86_feature_detected!` caches its result, so the hot-loop cost is a
        // cached load — far cheaper than SimSIMD's per-call dispatch.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: gated by the runtime detection above; reads in-bounds lanes.
            return unsafe { dot_f32_avx2(a, b) };
        }
        <f32 as SpatialSimilarity>::dot(a, b).unwrap_or(-1.0) as f32
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        <f32 as SpatialSimilarity>::dot(a, b).unwrap_or(-1.0) as f32
    }
}

/// `Σ (a[i] − b[i])²` over `min(a, b)` lanes — NEON on aarch64, scalar otherwise.
#[inline]
pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandated on AArch64-v8; reads in-bounds lanes.
        unsafe { l2_sq_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }
}

/// Inlined NEON f32 dot product. Eight independent `f32x4` accumulators
/// (32 lanes/iter) break the reduction dependency chain — 4 chains stall the FMA
/// pipes at high dim, 8 saturate them.
///
/// # Safety
/// Requires NEON (mandated on AArch64-v8). Reads only `min(a, b)` lanes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut acc4 = vdupq_n_f32(0.0);
    let mut acc5 = vdupq_n_f32(0.0);
    let mut acc6 = vdupq_n_f32(0.0);
    let mut acc7 = vdupq_n_f32(0.0);
    let chunks = n / 32;
    for i in 0..chunks {
        let o = i * 32;
        acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(o)), vld1q_f32(bp.add(o)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(o + 4)), vld1q_f32(bp.add(o + 4)));
        acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(o + 8)), vld1q_f32(bp.add(o + 8)));
        acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(o + 12)), vld1q_f32(bp.add(o + 12)));
        acc4 = vfmaq_f32(acc4, vld1q_f32(ap.add(o + 16)), vld1q_f32(bp.add(o + 16)));
        acc5 = vfmaq_f32(acc5, vld1q_f32(ap.add(o + 20)), vld1q_f32(bp.add(o + 20)));
        acc6 = vfmaq_f32(acc6, vld1q_f32(ap.add(o + 24)), vld1q_f32(bp.add(o + 24)));
        acc7 = vfmaq_f32(acc7, vld1q_f32(ap.add(o + 28)), vld1q_f32(bp.add(o + 28)));
    }
    let mut o = chunks * 32;
    if o + 16 <= n {
        acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(o)), vld1q_f32(bp.add(o)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(o + 4)), vld1q_f32(bp.add(o + 4)));
        acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(o + 8)), vld1q_f32(bp.add(o + 8)));
        acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(o + 12)), vld1q_f32(bp.add(o + 12)));
        o += 16;
    }
    let sum = vaddq_f32(
        vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)),
        vaddq_f32(vaddq_f32(acc4, acc5), vaddq_f32(acc6, acc7)),
    );
    let mut total = vaddvq_f32(sum);
    for i in o..n {
        total += a[i] * b[i];
    }
    total
}

/// Inlined NEON squared-Euclidean distance, mirroring [`dot_f32_neon`].
///
/// # Safety
/// Requires NEON (mandated on AArch64-v8). Reads only `min(a, b)` lanes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn l2_sq_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut acc4 = vdupq_n_f32(0.0);
    let mut acc5 = vdupq_n_f32(0.0);
    let mut acc6 = vdupq_n_f32(0.0);
    let mut acc7 = vdupq_n_f32(0.0);
    let chunks = n / 32;
    for i in 0..chunks {
        let o = i * 32;
        let d0 = vsubq_f32(vld1q_f32(ap.add(o)), vld1q_f32(bp.add(o)));
        let d1 = vsubq_f32(vld1q_f32(ap.add(o + 4)), vld1q_f32(bp.add(o + 4)));
        let d2 = vsubq_f32(vld1q_f32(ap.add(o + 8)), vld1q_f32(bp.add(o + 8)));
        let d3 = vsubq_f32(vld1q_f32(ap.add(o + 12)), vld1q_f32(bp.add(o + 12)));
        let d4 = vsubq_f32(vld1q_f32(ap.add(o + 16)), vld1q_f32(bp.add(o + 16)));
        let d5 = vsubq_f32(vld1q_f32(ap.add(o + 20)), vld1q_f32(bp.add(o + 20)));
        let d6 = vsubq_f32(vld1q_f32(ap.add(o + 24)), vld1q_f32(bp.add(o + 24)));
        let d7 = vsubq_f32(vld1q_f32(ap.add(o + 28)), vld1q_f32(bp.add(o + 28)));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
        acc4 = vfmaq_f32(acc4, d4, d4);
        acc5 = vfmaq_f32(acc5, d5, d5);
        acc6 = vfmaq_f32(acc6, d6, d6);
        acc7 = vfmaq_f32(acc7, d7, d7);
    }
    let mut o = chunks * 32;
    if o + 16 <= n {
        let d0 = vsubq_f32(vld1q_f32(ap.add(o)), vld1q_f32(bp.add(o)));
        let d1 = vsubq_f32(vld1q_f32(ap.add(o + 4)), vld1q_f32(bp.add(o + 4)));
        let d2 = vsubq_f32(vld1q_f32(ap.add(o + 8)), vld1q_f32(bp.add(o + 8)));
        let d3 = vsubq_f32(vld1q_f32(ap.add(o + 12)), vld1q_f32(bp.add(o + 12)));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
        o += 16;
    }
    let sum = vaddq_f32(
        vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)),
        vaddq_f32(vaddq_f32(acc4, acc5), vaddq_f32(acc6, acc7)),
    );
    let mut total = vaddvq_f32(sum);
    for i in o..n {
        let d = a[i] - b[i];
        total += d * d;
    }
    total
}

/// Inlined AVX2+FMA f32 dot product, the x86_64 analogue of [`dot_f32_neon`].
/// Four independent `f32x8` accumulators (32 lanes/iter).
///
/// # Safety
/// Requires AVX2 + FMA (the caller runtime-detects). Reads only `min(a, b)` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let chunks = n / 32;
    for i in 0..chunks {
        let o = i * 32;
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(o)), _mm256_loadu_ps(bp.add(o)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(o + 8)), _mm256_loadu_ps(bp.add(o + 8)), acc1);
        acc2 =
            _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(o + 16)), _mm256_loadu_ps(bp.add(o + 16)), acc2);
        acc3 =
            _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(o + 24)), _mm256_loadu_ps(bp.add(o + 24)), acc3);
    }
    let mut o = chunks * 32;
    while o + 8 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(o)), _mm256_loadu_ps(bp.add(o)), acc0);
        o += 8;
    }
    let sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let lo = _mm256_castps256_ps128(sum);
    let hi = _mm256_extractf128_ps::<1>(sum);
    let mut s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    let mut total = _mm_cvtss_f32(s);
    for i in o..n {
        total += a[i] * b[i];
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_and_l2_match_scalar_reference() {
        let cases: &[usize] = &[1, 7, 8, 31, 32, 100, 128, 257];
        for &d in cases {
            let a: Vec<f32> = (0..d).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
            let b: Vec<f32> = (0..d).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();
            let dot_ref: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let l2_ref: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            assert!((dot_f32(&a, &b) - dot_ref).abs() <= 1e-3, "dot mismatch d={d}");
            assert!((l2_sq(&a, &b) - l2_ref).abs() <= 1e-3, "l2 mismatch d={d}");
        }
    }
}
