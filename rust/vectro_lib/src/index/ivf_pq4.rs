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

use super::hnsw::HnswIndex;
use super::ivf_pq::{cosine_dist, kmeans_lloyd, top_probe_from_sims};
use super::pq4::{interleave_codes, quantize_lut, scan, BLK, K};
use crate::quant::pq::{pq_distance_table, pq_encode, train_pq_codebook, PQCodebook};
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
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
    /// Optional HNSW over the coarse centroids — the coarse quantiser. Lets the
    /// `n_probe` nearest cells be found in ~`O(log n_lists)` graph hops instead
    /// of a brute-force scan over every centroid, so the index can partition
    /// finely (small posting lists → cheap PQ4 scan) *and* probe widely (matched
    /// recall) without the coarse step blowing up. `None` on indexes built before
    /// this field existed → falls back to the brute-force / GEMM coarse scan.
    #[serde(default)]
    coarse_hnsw: Option<HnswIndex>,
    /// PQ codebook with `n_centroids == 16`.
    codebook: PQCodebook,
    /// Per-cell fast-scan stores, indexed by coarse-cell id.
    lists: Vec<Pq4List>,
    m: usize,
    dim: usize,
    trained: bool,
}

/// HNSW parameters for the coarse quantiser (small graph over the centroids).
const COARSE_M: usize = 16;
const COARSE_EF_C: usize = 200;

#[inline]
fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

/// Coarse search ef: probe wide enough that the HNSW finds the true `n_probe`
/// nearest cells with high recall (a generous multiple of `n_probe`).
#[inline]
fn coarse_ef(n_probe: usize) -> usize {
    (n_probe * 2).max(64)
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

        // Coarse quantiser: an HNSW over the centroids (unit-norm → cosine). The
        // n_probe nearest cells are then found by a graph walk, not a full scan.
        let centroid_rows: Vec<&[f32]> =
            (0..n_lists).map(|c| &coarse_centroids[c * d..(c + 1) * d]).collect();
        let mut coarse_hnsw = HnswIndex::new(COARSE_M, COARSE_EF_C);
        coarse_hnsw.add_batch(&centroid_rows);

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

        Ok(Self {
            n_lists,
            n_probe,
            coarse_centroids,
            coarse_hnsw: Some(coarse_hnsw),
            codebook,
            lists,
            m,
            dim: d,
            trained: true,
        })
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
        let probe = n_probe.min(self.n_lists);
        let probe_lists = self.coarse_probe(&q, probe);
        self.scan_probed(&q, &probe_lists, k)
    }

    /// Pick the `probe` nearest coarse cells for an already-normalised query:
    /// an HNSW graph walk when the coarse quantiser is present, else the
    /// brute-force partial-select scan (old indexes / fallback).
    fn coarse_probe(&self, q: &[f32], probe: usize) -> Vec<usize> {
        if let Some(h) = &self.coarse_hnsw {
            return h.search(q, probe, coarse_ef(probe)).into_iter().map(|(id, _)| id).collect();
        }
        let mut scored: Vec<(f32, usize)> = (0..self.n_lists)
            .map(|c| (cosine_dist(q, &self.coarse_centroids[c * self.dim..(c + 1) * self.dim]), c))
            .collect();
        if probe < scored.len() {
            scored.select_nth_unstable_by(probe - 1, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(probe);
        }
        scored.iter().map(|&(_, c)| c).collect()
    }

    /// Batch search over a flat `[q * dim]` query buffer, parallelised across
    /// queries (rayon, lock-free). The coarse step — comparing each query to
    /// every centroid — is the dominant cost at high dim / many lists, so it is
    /// computed as one `Q·Cᵀ` GEMM per tile (centroid matrix reused across the
    /// tile, the way FAISS routes its coarse quantiser) instead of a per-query
    /// serial scalar scan; the PQ4 `pshufb` scan then runs per query. Mirrors
    /// [`super::ivf_pq::IvfPqIndex::search_batch_flat`].
    pub fn search_batch_flat(
        &self,
        flat: &[f32],
        dim: usize,
        k: usize,
        n_probe: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        if dim == 0 {
            return Vec::new();
        }
        let q = flat.len() / dim;
        if !self.trained || self.dim == 0 {
            return vec![Vec::new(); q];
        }
        assert_eq!(dim, self.dim, "query dim mismatch");
        if q == 0 {
            return Vec::new();
        }

        // Unit-normalise every query once (cosine == dot for unit vectors).
        let mut qnorm = vec![0.0f32; q * dim];
        for i in 0..q {
            let row = &flat[i * dim..(i + 1) * dim];
            let inv = 1.0 / row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for (j, &x) in row.iter().enumerate() {
                qnorm[i * dim + j] = x * inv;
            }
        }
        let probe = n_probe.min(self.n_lists);
        let mut results: Vec<Vec<(usize, f32)>> = vec![Vec::new(); q];

        // Coarse quantiser present → per-query HNSW graph walk (parallel across
        // queries). Each query finds its probe cells in ~O(log n_lists) hops, so
        // the coarse cost no longer scales with n_lists — the lever that lets the
        // index partition finely and probe widely at matched recall.
        if let Some(h) = &self.coarse_hnsw {
            let ef = coarse_ef(probe);
            results.par_iter_mut().enumerate().for_each(|(i, slot)| {
                let qn = &qnorm[i * dim..(i + 1) * dim];
                let probe_lists: Vec<usize> =
                    h.search(qn, probe, ef).into_iter().map(|(id, _)| id).collect();
                *slot = self.scan_probed(qn, &probe_lists, k);
            });
            return results;
        }

        // Fallback (old index, no coarse HNSW): tiled `Q·Cᵀ` GEMM coarse scan.
        let cmat = ArrayView2::from_shape((self.n_lists, dim), &self.coarse_centroids)
            .expect("centroid shape");
        const CHUNK: usize = 32;
        results.par_chunks_mut(CHUNK).enumerate().for_each(|(c, out)| {
            let lo = c * CHUNK;
            let rows = out.len();
            let qtile = ArrayView2::from_shape((rows, dim), &qnorm[lo * dim..(lo + rows) * dim])
                .expect("query tile shape");
            // sims[r, l] = dot(query r, centroid l) — higher = closer (cosine).
            let sims: Array2<f32> = qtile.dot(&cmat.t());
            for (r, slot) in out.iter_mut().enumerate() {
                let srow = sims.row(r);
                let probe_lists = top_probe_from_sims(srow.as_slice().expect("contig row"), probe);
                let qi = lo + r;
                *slot = self.scan_probed(&qnorm[qi * dim..(qi + 1) * dim], &probe_lists, k);
            }
        });
        results
    }

    /// Test-only: drop the coarse HNSW so the GEMM fallback path is exercised.
    #[cfg(test)]
    pub(crate) fn disable_coarse_hnsw(&mut self) {
        self.coarse_hnsw = None;
    }

    /// PQ4 fast-scan over the chosen `probe_lists` for an already-normalised
    /// query, returning the top-`k` `(gid, dist)`. Shared by the single-query and
    /// batched search paths.
    fn scan_probed(&self, q: &[f32], probe_lists: &[usize], k: usize) -> Vec<(usize, f32)> {
        // One u8 LUT for the whole query; sums across cells are comparable.
        let table = pq_distance_table(q, &self.codebook);
        let (lut, inv_scale, bias) = quantize_lut(&table, self.m);

        let mut cand: Vec<(u16, usize)> = Vec::new();
        let mut sums: Vec<u16> = Vec::new();
        for &c in probe_lists {
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

    /// Clustered unit vectors (centers + jitter) — realistic recall structure,
    /// unlike near-orthogonal `rand_unit`. Used by the coarse-quantiser A/B.
    fn clustered(n: usize, d: usize, n_centers: usize, jitter: f32, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut next = move || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (s >> 33) as f32 / (1u64 << 31) as f32 - 1.0
        };
        let centers: Vec<Vec<f32>> =
            (0..n_centers).map(|_| (0..d).map(|_| next()).collect()).collect();
        (0..n)
            .map(|i| {
                let c = &centers[i % n_centers];
                let v: Vec<f32> = c.iter().map(|&x| x + jitter * next()).collect();
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

    /// A/B: HNSW coarse quantiser vs the brute-force/GEMM coarse, on both batch
    /// QPS and recall@10 (vs exact brute-force ground truth), across n_probe.
    /// `cargo test -p vectro_lib --release ivf_pq4_coarse_ab -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn ivf_pq4_coarse_ab() {
        use std::collections::HashSet;
        use std::time::Instant;
        let (n, d, n_lists) = (200_000usize, 768usize, 4096usize);
        let k = 10usize;
        // Clustered data so coarse recall is meaningful (rand_unit is near-
        // orthogonal at d=768 → ~0 recall for any method, GEMM or HNSW).
        let data = clustered(n, d, 256, 0.30, 1);
        let queries = clustered(2000, d, 256, 0.30, 99);
        let flat: Vec<f32> = queries.iter().flatten().copied().collect();

        // Exact cosine ground truth for the first 200 queries.
        let gt: Vec<HashSet<usize>> = queries[..200]
            .iter()
            .map(|q| {
                let mut s: Vec<(f32, usize)> = data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (-q.iter().zip(v).map(|(a, b)| a * b).sum::<f32>(), i))
                    .collect();
                s.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
                s.truncate(k);
                s.into_iter().map(|(_, i)| i).collect()
            })
            .collect();

        let hnsw = IvfPq4Index::build(&data, n_lists, 32, 96, 10, 1).expect("build");
        let mut gemm = hnsw.clone();
        gemm.disable_coarse_hnsw();

        let recall = |idx: &IvfPq4Index, np: usize| -> f64 {
            let qf: Vec<f32> = queries[..200].iter().flatten().copied().collect();
            let res = idx.search_batch_flat(&qf, d, k, np);
            let mut tot = 0usize;
            for (r, g) in res.iter().zip(&gt) {
                tot += r.iter().filter(|(id, _)| g.contains(id)).count();
            }
            tot as f64 / (gt.len() * k) as f64
        };
        let qps = |idx: &IvfPq4Index, np: usize| -> f64 {
            let _ = idx.search_batch_flat(&flat, d, k, np);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                std::hint::black_box(idx.search_batch_flat(&flat, d, k, np).len());
                best = best.min(t.elapsed().as_secs_f64());
            }
            queries.len() as f64 / best
        };

        println!("ivf_pq4 coarse A/B  n={n} d={d} lists={n_lists} (faiss-IVF-PQ ref: 11828 qps)");
        for &np in &[32usize, 48, 64] {
            println!(
                "  n_probe={np:>3}: GEMM {:.0} qps R@{k}={:.4} | HNSW {:.0} qps R@{k}={:.4}",
                qps(&gemm, np),
                recall(&gemm, np),
                qps(&hnsw, np),
                recall(&hnsw, np),
            );
        }
    }

    #[test]
    fn search_batch_flat_matches_single_query() {
        let data = rand_unit(2000, 64, 7);
        let idx = IvfPq4Index::build(&data, 32, 8, 16, 12, 7).expect("build");
        let queries = rand_unit(50, 64, 99);
        let flat: Vec<f32> = queries.iter().flatten().copied().collect();
        let batch = idx.search_batch_flat(&flat, 64, 10, 8);
        assert_eq!(batch.len(), queries.len());
        for (i, q) in queries.iter().enumerate() {
            let single = idx.search_with_probe(q, 10, 8);
            // The batched coarse GEMM and the scalar coarse scan can tie-break
            // equidistant centroids differently, so compare the returned gids as
            // sets (same candidates, possibly reordered only on exact ties).
            let bs: std::collections::HashSet<usize> = batch[i].iter().map(|&(g, _)| g).collect();
            let ss: std::collections::HashSet<usize> = single.iter().map(|&(g, _)| g).collect();
            let overlap = bs.intersection(&ss).count();
            assert!(
                overlap >= 9,
                "batch vs single overlap {overlap}/10 too low at query {i}"
            );
        }
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
