// Shared utilities for embedding generators
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Generate a random normalized embedding vector
pub fn generate_embedding(dim: usize, rng: &mut impl Rng) -> Vec<f64> {
    // SAFETY(clippy::unwrap_used): std=1.0 is a fixed, always-valid parameter --
    // `Normal::new` only errors on a negative or non-finite std, which this
    // literal can never be. Not a fallible runtime input.
    #[allow(clippy::unwrap_used)]
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut vec: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();

    // Normalize to unit length
    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        vec.iter_mut().for_each(|x| *x /= norm);
    }

    vec
}

/// Normalize a vector to unit length
pub fn normalize(vec: &mut [f64]) {
    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        vec.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Generate a cluster of embeddings around a center point
pub fn generate_cluster(
    center: &[f64],
    noise: f64,
    count: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<f64>> {
    // `Normal::new` errors only on a negative or non-finite std; clamp instead of
    // unwrapping so a bad `noise` argument degrades to a point cluster (std=0)
    // rather than panicking a benchmark-data generator run.
    let safe_noise = if noise.is_finite() {
        noise.max(0.0)
    } else {
        0.0
    };
    #[allow(clippy::unwrap_used)]
    let normal = Normal::new(0.0, safe_noise).unwrap();
    let mut embeddings = Vec::with_capacity(count);

    for _ in 0..count {
        let mut vec: Vec<f64> = center
            .iter()
            .map(|&c| (c + normal.sample(rng)).clamp(-1.0, 1.0))
            .collect();
        normalize(&mut vec);
        embeddings.push(vec);
    }

    embeddings
}
