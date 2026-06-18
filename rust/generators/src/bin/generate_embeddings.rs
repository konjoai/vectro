#!/usr/bin/env rust
//! Generate sample embeddings for Vectro+ demos and testing.

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Embedding {
    id: String,
    vector: Vec<f64>,
}

#[derive(Parser)]
#[command(name = "generate_embeddings")]
#[command(about = "Generate sample embeddings for Vectro+ testing")]
struct Args {
    /// Number of embeddings to generate
    #[arg(long, default_value = "1000")]
    count: usize,

    /// Embedding dimension
    #[arg(long, default_value = "128")]
    dim: usize,

    /// ID prefix
    #[arg(long, default_value = "emb")]
    prefix: String,

    /// Random seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,
}

fn main() {
    let args = Args::parse();
    
    let mut rng = match args.seed {
        Some(seed) => ChaCha8Rng::seed_from_u64(seed),
        None => ChaCha8Rng::from_entropy(),
    };

    for i in 0..args.count {
        let vector = generators::generate_embedding(args.dim, &mut rng);
        let embedding = Embedding {
            id: format!("{}_{:06}", args.prefix, i),
            vector,
        };
        println!("{}", serde_json::to_string(&embedding).unwrap());
    }
}
