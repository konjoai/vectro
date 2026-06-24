//! IVF-PQ4 fast-scan — coarse IVF clustering with per-posting-list PQ4
//! `pshufb` fast-scan (FAISS `IndexIVFPQFastScan` analogue).
//!
//! [`super::ivf_pq::IvfPqIndex`] scores probed candidates with a scalar ADC
//! gather. `IvfPq4Index` instead stores each posting list's codes in the PQ4
//! interleaved block layout and scans them with the SIMD lookup-table kernel
//! from [`super::pq4`] — the production-path analogue of [`super::pq4::Pq4FlatIndex`]
//! (exhaustive) restricted to the probed cells.
//!
//! Reuses the coarse k-means / cosine kernel from [`super::ivf_pq`] and the
//! LUT-quantization + `pshufb` scan from [`super::pq4`], so there is a single
//! implementation of each.
//!
//! This is a **build-once** index (train + populate in [`IvfPq4Index::build`]);
//! incremental `add`, nibble-packing, and a PyO3 binding are tracked follow-ups.

use super::ivf_pq::{cosine_dist, kmeans_lloyd};
use super::pq4::{interleave_codes, quantize_lut, scan, BLK, K};
use crate::quant::pq::{pq_distance_table, pq_encode, train_pq_codebook, PQCodebook};
use serde::{Deserialize, Serialize};

/// One coarse cell's PQ4 fast-scan store.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Pq4List {
    /// Global vector id for slot `i` (`len` = real candidate count).
    ids: Vec<usize>,
    /// Nibble-packed interleaved codes `[n_blocks][⌈m/2⌉][BLK]`, padded.
    codes_il: Vec<u8>,
}

/// IVF index with PQ4 SIMD fast-scan over probed posting lists.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfPq4Index {
    n_lists: usize,
    n_probe: usize,
    /// Coarse centroids, `[n_lists * dim]`, unit-norm.
    coarse_centroids: Vec<f32>,
    /// PQ codebook with `n_centroids == 16`.
    codebook: PQCodebook,
    /// Per-cell fast-scan stores, indexed by coarse-cell id.
    lists: Vec<Pq4List>,
    m: usize,
    dim: usize,
    trained: bool,
}

#[inline]
fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

impl IvfPq4Index {
    /// Train (coarse k-means + K=16 PQ codebook) and populate from `data` in one
    /// pass. `m` must divide the vector dimension.
    ///
    /// * `n_lists`  — coarse cells (Voronoi partitions).
    /// * `n_probe`  — default cells scanned per query.
    /// * `m`        — PQ subspaces.
    pub fn build(
        data: &[Vec<f32>],
        n_lists: usize,
        n_probe: usize,
        m: usize,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self, String> {
        if data.is_empty() {
            return Err("data is empty".into());
        }
        if n_lists == 0 || n_probe == 0 {
            return Err("n_lists and n_probe must be > 0".into());
        }
        if data.len() < n_lists {
            return Err(format!("need ≥ n_lists ({n_lists}) vectors, got {}", data.len()));
        }
        let d = data[0].len();
        if d == 0 || !data.iter().all(|v| v.len() == d) {
            return Err("vectors empty or of inconsistent length".into());
        }

        let normed: Vec<Vec<f32>> = data.iter().map(|v| normalize(v)).collect();
        let coarse_centroids = kmeans_lloyd(&normed, n_lists, d, max_iter, seed);
        let codebook = train_pq_codebook(&normed, m, K, max_iter, seed)?;

        // Assign every vector to its nearest coarse cell.
        let mut cell_ids: Vec<Vec<usize>> = vec![Vec::new(); n_lists];
        for (i, v) in normed.iter().enumerate() {
            cell_ids[nearest_coarse(&coarse_centroids, n_lists, d, v)].push(i);
        }

        // Per cell: PQ-encode its members and interleave into the SIMD layout.
        let lists: Vec<Pq4List> = cell_ids
            .into_iter()
            .map(|ids| {
                let cell_vecs: Vec<Vec<f32>> = ids.iter().map(|&i| normed[i].clone()).collect();
                let codes = pq_encode(&cell_vecs, &codebook);
                let codes_il = interleave_codes(&codes, m);
                Pq4List { ids, codes_il }
            })
            .collect();

        Ok(Self { n_lists, n_probe, coarse_centroids, codebook, lists, m, dim: d, trained: true })
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.lists.iter().map(|l| l.ids.len()).sum()
    }

    /// True when the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Approximate top-`k` nearest neighbours, scanning `self.n_probe` cells.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.search_with_probe(query, k, self.n_probe)
    }

    /// Search with an explicit probe width: coarse-scan to pick the `n_probe`
    /// nearest cells, then PQ4 fast-scan their candidates.
    pub fn search_with_probe(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<(usize, f32)> {
        if !self.trained || k == 0 || self.dim == 0 {
            return Vec::new();
        }
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        let q = normalize(query);

        // Coarse scan: pick the n_probe nearest cells (partial select, not sort).
        let mut scored: Vec<(f32, usize)> = (0..self.n_lists)
            .map(|c| (cosine_dist(&q, &self.coarse_centroids[c * self.dim..(c + 1) * self.dim]), c))
            .collect();
        let probe = n_probe.min(self.n_lists);
        if probe < scored.len() {
            scored.select_nth_unstable_by(probe - 1, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(probe);
        }

        // One u8 LUT for the whole query; sums across cells are comparable.
        let table = pq_distance_table(&q, &self.codebook);
        let (lut, inv_scale, bias) = quantize_lut(&table, self.m);

        let mut cand: Vec<(u16, usize)> = Vec::new();
        let mut sums: Vec<u16> = Vec::new();
        for &(_, c) in &scored {
            let list = &self.lists[c];
            let n = list.ids.len();
            if n == 0 {
                continue;
            }
            let n_blocks = n.div_ceil(BLK);
            sums.clear();
            sums.resize(n_blocks * BLK, 0);
            scan(&lut, &list.codes_il, n_blocks, self.m, &mut sums);
            cand.extend(list.ids.iter().enumerate().map(|(slot, &gid)| (sums[slot], gid)));
        }

        let kk = k.min(cand.len());
        if kk == 0 {
            return Vec::new();
        }
        if kk < cand.len() {
            cand.select_nth_unstable_by_key(kk - 1, |&(s, _)| s);
            cand.truncate(kk);
        }
        cand.sort_unstable_by_key(|&(s, _)| s);
        cand.into_iter().map(|(s, gid)| (gid, s as f32 * inv_scale + bias)).collect()
    }
}

/// Index of the nearest coarse centroid to `v` (cosine distance).
fn nearest_coarse(coarse: &[f32], n_lists: usize, d: usize, v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::MAX;
    for c in 0..n_lists {
        let dd = cosine_dist(v, &coarse[c * d..(c + 1) * d]);
        if dd < best_d {
            best_d = dd;
            best = c;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_unit(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut next = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (s >> 33) as f32 / (1u64 << 31) as f32 - 1.0
        };
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..d).map(|_| next()).collect();
                normalize(&v)
            })
            .collect()
    }

    #[test]
    fn build_validates() {
        let data = rand_unit(64, 16, 1);
        assert!(IvfPq4Index::build(&data, 8, 4, 8, 5, 1).is_ok());
        assert!(IvfPq4Index::build(&[], 8, 4, 8, 5, 1).is_err());
        // fewer vectors than cells
        assert!(IvfPq4Index::build(&rand_unit(4, 16, 1), 8, 4, 8, 5, 1).is_err());
    }

    #[test]
    fn len_counts_all_vectors() {
        let data = rand_unit(500, 32, 2);
        let idx = IvfPq4Index::build(&data, 16, 4, 8, 10, 2).expect("build");
        assert_eq!(idx.len(), 500);
    }

    #[test]
    fn self_recall_high_with_full_probe() {
        // With n_probe == n_lists every cell is scanned, so a vector's own code
        // should top-1 it (PQ is lossy but self is the row-min in every subspace).
        let data = rand_unit(800, 32, 3);
        let n_lists = 16;
        let idx = IvfPq4Index::build(&data, n_lists, n_lists, 8, 12, 3).expect("build");
        let mut hits = 0;
        for (i, v) in data.iter().enumerate().take(100) {
            if idx.search(v, 1).first().map(|&(id, _)| id) == Some(i) {
                hits += 1;
            }
        }
        assert!(hits >= 95, "self-recall@1 too low: {hits}/100");
    }

    #[test]
    fn recall_improves_with_probe() {
        // More probed cells → more of the true neighbours are reachable.
        let data = rand_unit(2000, 64, 5);
        let idx = IvfPq4Index::build(&data, 32, 1, 16, 15, 5).expect("build");
        let exact_top = |q: &[f32]| -> std::collections::HashSet<usize> {
            let mut e: Vec<(f32, usize)> = data
                .iter()
                .enumerate()
                .map(|(j, v)| (q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(), j))
                .collect();
            e.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            e.iter().take(10).map(|&(_, i)| i).collect()
        };
        let recall_at = |np: usize| -> f64 {
            let mut ov = 0;
            for q in data.iter().take(50) {
                let et = exact_top(q);
                ov += idx
                    .search_with_probe(q, 10, np)
                    .iter()
                    .filter(|(id, _)| et.contains(id))
                    .count();
            }
            ov as f64 / 500.0
        };
        let r1 = recall_at(1);
        let r8 = recall_at(8);
        // The behavioural invariant is that probing more cells reaches more true
        // neighbours; the absolute floor is modest because m=16/K=16 is a
        // low-bit-budget config (exhaustive recall@10 here is ~0.4).
        assert!(r8 > r1, "recall should rise with probe: r1={r1:.3} r8={r8:.3}");
        assert!(r8 >= 0.25, "recall@10 (probe=8) unexpectedly low: {r8:.3}");
    }
}
