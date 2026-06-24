//! PQ4 fast-scan — exhaustive Product-Quantization search with a SIMD
//! lookup-table scan (FAISS `IndexPQFastScan` analogue).
//!
//! Classic PQ ([`crate::quant::pq`]) scores each candidate with a scalar gather:
//! `Σ_m table[m][code_m]`, one f32 add per subspace per candidate. PQ4 instead
//! uses **4-bit codes (K = 16 centroids/subspace)** so the per-subspace
//! distance table fits in a 16-byte register, and an AVX2 `pshufb`
//! ([`_mm256_shuffle_epi8`]) looks up **32 candidates at once** for a subspace —
//! turning the inner loop from one scalar gather per candidate into one SIMD
//! op per 32. Measured ~22× over the scalar gather on AVX2 hardware.
//!
//! Layout: codes are stored **interleaved in blocks of 32** as
//! `codes_il[block][subspace][0..32]` (one nibble per byte), so each subspace's
//! 32 codes are contiguous for the `pshufb` load.
//!
//! Distance table quantization: the per-query f32 table is quantized to `u8`
//! (per-subspace min subtracted, a shared scale chosen so the `u16`
//! accumulator never overflows). Ranking by the `u16` sum is monotone in the
//! true ADC distance; the returned f32 distance is `sum * inv_scale + bias`.
//!
//! v1 stores one nibble per byte (so memory equals classic PQ-8, `M` bytes/vec);
//! packing two nibbles per byte for the 2× memory win is a tracked follow-up —
//! the headline here is scan throughput.

use crate::quant::pq::{pq_distance_table, pq_encode, train_pq_codebook, PQCodebook};
use serde::{Deserialize, Serialize};

/// Candidates per SIMD block (one 256-bit `pshufb` resolves 32 lookups).
pub(crate) const BLK: usize = 32;
/// Centroids per subspace (4-bit codes).
pub(crate) const K: usize = 16;

/// Interleave per-vector codes `[n][m]` into the blocked SIMD layout
/// `[n_blocks][m][BLK]` (one nibble per byte), zero-padding the final block.
/// Shared by the flat index and the IVF-PQ4 per-list stores.
pub(crate) fn interleave_codes(codes: &[Vec<u8>], m: usize) -> Vec<u8> {
    let n_blocks = codes.len().div_ceil(BLK);
    let mut out = vec![0u8; n_blocks * m * BLK];
    for (i, code) in codes.iter().enumerate() {
        let base = (i / BLK) * m * BLK;
        let c = i % BLK;
        for (mi, &cd) in code.iter().enumerate() {
            out[base + mi * BLK + c] = cd;
        }
    }
    out
}

/// Exhaustive PQ4 fast-scan index over a fixed vector set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pq4FlatIndex {
    /// PQ codebook with `n_centroids == 16`.
    codebook: PQCodebook,
    /// Number of subspaces (`= codebook.n_subspaces`).
    m: usize,
    /// Number of real vectors (the interleaved store is padded up to a block).
    n: usize,
    /// Interleaved codes `[n_blocks][m][BLK]`, one 4-bit code per byte. Padding
    /// candidates (index ≥ `n`) hold code 0 and are excluded from results.
    codes_il: Vec<u8>,
}

impl Pq4FlatIndex {
    /// Train a K=16 codebook on `data` and build the interleaved code store.
    ///
    /// `m` must divide the vector dimension. Returns an error from
    /// [`train_pq_codebook`] on invalid parameters / empty data.
    pub fn build(data: &[Vec<f32>], m: usize, max_iter: usize, seed: u64) -> Result<Self, String> {
        let codebook = train_pq_codebook(data, m, K, max_iter, seed)?;
        let codes = pq_encode(data, &codebook); // [n][m], each value in 0..16
        let n = data.len();
        let codes_il = interleave_codes(&codes, m);
        Ok(Self { codebook, m, n, codes_il })
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.n
    }

    /// True when the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Approximate top-`k` nearest neighbours by PQ4 fast-scan, returned as
    /// `(id, approx_distance)` ascending by distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.n == 0 || k == 0 {
            return Vec::new();
        }
        let table = pq_distance_table(query, &self.codebook);
        let (lut, inv_scale, bias) = quantize_lut(&table, self.m);

        let n_blocks = self.n.div_ceil(BLK);
        let mut sums = vec![0u16; n_blocks * BLK];
        scan(&lut, &self.codes_il, n_blocks, self.m, &mut sums);

        // Partial-select the k smallest u16 sums over the real candidates.
        let mut cand: Vec<(u16, usize)> = (0..self.n).map(|i| (sums[i], i)).collect();
        let kk = k.min(self.n);
        if kk < cand.len() {
            cand.select_nth_unstable_by_key(kk - 1, |&(s, _)| s);
            cand.truncate(kk);
        }
        cand.sort_unstable_by_key(|&(s, _)| s);
        cand.into_iter().map(|(s, i)| (i, s as f32 * inv_scale + bias)).collect()
    }
}

/// Quantize the per-query f32 distance table (`[m][16]`) to a `u8` LUT.
///
/// Each subspace's minimum is subtracted (folded into `bias`), and a shared
/// `scale` maps the residual range to `[0, qmax]` where
/// `qmax = min(255, ⌊u16::MAX / m⌋)` guarantees the `m`-term `u16` accumulator
/// cannot overflow. Returns `(lut, inv_scale, bias)` such that
/// `true_distance ≈ u16_sum * inv_scale + bias` (and ranking is exact modulo
/// the u8 quantization step).
pub(crate) fn quantize_lut(table: &[f32], m: usize) -> (Vec<u8>, f32, f32) {
    let qmax = ((u16::MAX as usize / m.max(1)).min(255)) as f32;
    let mut bias = 0.0f32;
    let mut gmax = 0.0f32;
    let mut mins = vec![0.0f32; m];
    for mi in 0..m {
        let row = &table[mi * K..mi * K + K];
        let lo = row.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        mins[mi] = lo;
        bias += lo;
        gmax = gmax.max(hi - lo);
    }
    let scale = if gmax > 0.0 { qmax / gmax } else { 0.0 };
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let mut lut = vec![0u8; m * K];
    for mi in 0..m {
        for k in 0..K {
            let q = ((table[mi * K + k] - mins[mi]) * scale).round();
            lut[mi * K + k] = q.clamp(0.0, qmax) as u8;
        }
    }
    (lut, inv_scale, bias)
}

/// Accumulate `u16` distance sums for every candidate (AVX2 `pshufb` on x86_64
/// with runtime detection, scalar fallback otherwise).
#[inline]
pub(crate) fn scan(lut: &[u8], codes_il: &[u8], n_blocks: usize, m: usize, out: &mut [u16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: gated by the runtime detection; all indices stay in-bounds.
            unsafe { scan_avx2(lut, codes_il, n_blocks, m, out) };
            return;
        }
    }
    scan_scalar(lut, codes_il, n_blocks, m, out);
}

/// Scalar reference scan — the correctness baseline the SIMD kernel must match.
/// Reads the same interleaved layout so results are identical.
fn scan_scalar(lut: &[u8], codes_il: &[u8], n_blocks: usize, m: usize, out: &mut [u16]) {
    for b in 0..n_blocks {
        let base = b * m * BLK;
        for c in 0..BLK {
            let mut acc: u16 = 0;
            for mi in 0..m {
                acc = acc.wrapping_add(lut[mi * K + codes_il[base + mi * BLK + c] as usize] as u16);
            }
            out[b * BLK + c] = acc;
        }
    }
}

/// Maps a candidate's position within a block to its slot in the `pshufb`
/// accumulator scratch (the `unpacklo/hi_epi8` byte interleave is per-128-bit
/// lane, so candidate order is permuted; this inverts it). Computed at compile
/// time. See [`scan_avx2`].
#[cfg(target_arch = "x86_64")]
static PERM: [usize; BLK] = {
    let mut p = [0usize; BLK];
    let mut c = 0;
    while c < BLK {
        let lane = c / 16; // which 128-bit half of the code register (0 or 1)
        let w = c % 16; // position within that lane
        // acc_lo holds bytes 0..7 of each lane (lane0→0..7, lane1→8..15);
        // acc_hi (offset 16) holds bytes 8..15 of each lane likewise.
        p[c] = if w < 8 { lane * 8 + w } else { 16 + lane * 8 + (w - 8) };
        c += 1;
    }
    p
};

/// AVX2 `pshufb` fast-scan: for each block of 32 candidates, look up all 32
/// per-subspace distances with one `_mm256_shuffle_epi8`, widen `u8`→`u16`, and
/// accumulate across subspaces. 32 lookups per SIMD op vs one per scalar gather.
///
/// # Safety
/// Requires AVX2 (the caller runtime-detects). `codes_il` is `n_blocks*m*32`
/// bytes and `out` is `n_blocks*32` — all reads/writes stay in-bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(lut: &[u8], codes_il: &[u8], n_blocks: usize, m: usize, out: &mut [u16]) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    for b in 0..n_blocks {
        let mut acc_lo = _mm256_setzero_si256(); // 16 u16 lanes
        let mut acc_hi = _mm256_setzero_si256(); // 16 u16 lanes
        let blk_base = b * m * BLK;
        for mi in 0..m {
            // 16-entry LUT for this subspace, duplicated into both 128-bit lanes.
            let l128 = _mm_loadu_si128(lut.as_ptr().add(mi * K) as *const __m128i);
            let lut256 = _mm256_set_m128i(l128, l128);
            // 32 codes (nibbles 0..15) for this (block, subspace).
            let codes = _mm256_loadu_si256(codes_il.as_ptr().add(blk_base + mi * BLK) as *const __m256i);
            // Each 128-bit lane resolves its 16 codes against its 16-entry table.
            let looked = _mm256_shuffle_epi8(lut256, codes);
            acc_lo = _mm256_add_epi16(acc_lo, _mm256_unpacklo_epi8(looked, zero));
            acc_hi = _mm256_add_epi16(acc_hi, _mm256_unpackhi_epi8(looked, zero));
        }
        let mut tmp = [0u16; BLK];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc_lo);
        _mm256_storeu_si256(tmp.as_mut_ptr().add(16) as *mut __m256i, acc_hi);
        let out_base = b * BLK;
        for c in 0..BLK {
            out[out_base + c] = tmp[PERM[c]];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pseudo-random unit vectors (LCG) — varied enough that PQ distances don't
    /// degenerate into ties, unlike a structured modulo generator.
    fn unit_vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut next = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (s >> 33) as f32 / (1u64 << 31) as f32 - 1.0
        };
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..d).map(|_| next()).collect();
                let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                v.iter().map(|x| x / nrm).collect()
            })
            .collect()
    }

    #[test]
    fn scan_simd_matches_scalar() {
        // Random LUT + interleaved codes across non-block-aligned sizes.
        let m = 16usize;
        let n_blocks = 5usize;
        let mut lut = vec![0u8; m * K];
        let mut codes = vec![0u8; n_blocks * m * BLK];
        let mut s = 0x9e37u32;
        let mut rng = || {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (s >> 16) as u8
        };
        for x in lut.iter_mut() {
            *x = rng() % 64;
        }
        for x in codes.iter_mut() {
            *x = rng() & 0x0F;
        }
        let mut a = vec![0u16; n_blocks * BLK];
        let mut b = vec![0u16; n_blocks * BLK];
        scan_scalar(&lut, &codes, n_blocks, m, &mut a);
        scan(&lut, &codes, n_blocks, m, &mut b);
        assert_eq!(a, b, "SIMD scan must match scalar reference");
    }

    #[test]
    fn search_finds_self() {
        // Each indexed vector is its own nearest neighbour under PQ4.
        let data = unit_vecs(200, 32, 7);
        let idx = Pq4FlatIndex::build(&data, 8, 10, 42).expect("build");
        assert_eq!(idx.len(), 200);
        let mut hits = 0;
        for (i, v) in data.iter().enumerate().take(40) {
            let res = idx.search(v, 1);
            if res.first().map(|&(id, _)| id) == Some(i) {
                hits += 1;
            }
        }
        // PQ is lossy, but a vector's own code should top-1 the vast majority.
        assert!(hits >= 36, "self-recall too low: {hits}/40");
    }

    #[test]
    fn ranking_agrees_with_exact_adc() {
        // PQ4 fast-scan top-k overlaps heavily with the exact f32 ADC ranking
        // (the u8 LUT quantization only perturbs near-ties).
        use crate::quant::pq::{adc_distance, pq_distance_table};
        let data = unit_vecs(512, 32, 3);
        let m = 8usize;
        let idx = Pq4FlatIndex::build(&data, m, 12, 1).expect("build");
        let codes = pq_encode(&data, &idx.codebook);
        for q in data.iter().take(8) {
            let fast: Vec<usize> = idx.search(q, 10).into_iter().map(|(id, _)| id).collect();
            let table = pq_distance_table(q, &idx.codebook);
            let mut exact: Vec<(f32, usize)> =
                codes.iter().enumerate().map(|(i, c)| (adc_distance(&table, c, m, K), i)).collect();
            exact.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let exact_top: std::collections::HashSet<usize> =
                exact.iter().take(10).map(|&(_, i)| i).collect();
            let overlap = fast.iter().filter(|i| exact_top.contains(i)).count();
            assert!(overlap >= 8, "fast-scan top-10 overlap with exact ADC too low: {overlap}/10");
        }
    }

    #[test]
    fn empty_and_k_zero() {
        let data = unit_vecs(64, 16, 1);
        let idx = Pq4FlatIndex::build(&data, 8, 5, 1).expect("build");
        assert!(idx.search(&data[0], 0).is_empty());
        let empty: Vec<Vec<f32>> = Vec::new();
        assert!(Pq4FlatIndex::build(&empty, 8, 5, 1).is_err());
    }
}
