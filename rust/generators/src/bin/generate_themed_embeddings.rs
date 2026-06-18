#!/usr/bin/env rust
//! Generate themed sample embeddings for compelling Vectro+ demos.
//! Creates semantically meaningful vectors across different categories.

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
struct Embedding {
    id: String,
    vector: Vec<f64>,
}

#[derive(Parser)]
#[command(name = "generate_themed_embeddings")]
#[command(about = "Generate themed semantic embeddings for Vectro+ demos")]
struct Args {
    /// Number of embeddings to generate
    #[arg(long, default_value = "1000")]
    count: usize,

    /// Embedding dimension
    #[arg(long, default_value = "128")]
    dim: usize,

    /// Theme for generated embeddings
    #[arg(long, default_value = "products")]
    #[arg(value_parser = ["products", "movies", "documents", "mixed", "random"])]
    theme: String,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,
}

fn generate_products(dim: usize, count: usize, rng: &mut ChaCha8Rng) -> Vec<Embedding> {
    let mut products = Vec::new();
    
    let categories = vec!["electronics", "clothing", "food", "books", "toys", "sports"];
    let product_names: HashMap<&str, Vec<&str>> = [
        ("electronics", vec!["laptop", "smartphone", "tablet", "headphones", "camera", "monitor"]),
        ("clothing", vec!["shirt", "pants", "dress", "jacket", "shoes", "hat"]),
        ("food", vec!["apple", "bread", "cheese", "pasta", "rice", "coffee"]),
        ("books", vec!["novel", "textbook", "magazine", "comic", "manual", "dictionary"]),
        ("toys", vec!["doll", "puzzle", "lego", "robot", "ball", "game"]),
        ("sports", vec!["basketball", "tennis_racket", "running_shoes", "yoga_mat", "dumbbell", "bike"]),
    ].iter().cloned().collect();
    
    let items_per_category = count / categories.len();
    
    for cat_name in categories {
        let center = generators::generate_embedding(dim, rng);
        let vectors = generators::generate_cluster(&center, 0.15, items_per_category, rng);
        let names = &product_names[cat_name];
        
        for (i, vec) in vectors.into_iter().enumerate() {
            let name = names[i % names.len()];
            let product_id = format!("{}_{}__{:04}", cat_name, name, i);
            products.push(Embedding {
                id: product_id,
                vector: vec,
            });
        }
    }
    
    products
}

fn generate_movies(dim: usize, count: usize, rng: &mut ChaCha8Rng) -> Vec<Embedding> {
    let mut movies = Vec::new();
    
    let genres = vec!["action", "comedy", "drama", "scifi", "horror", "romance"];
    let items_per_genre = count / genres.len();
    
    for genre_name in genres {
        let center = generators::generate_embedding(dim, rng);
        let vectors = generators::generate_cluster(&center, 0.2, items_per_genre, rng);
        
        for (i, vec) in vectors.into_iter().enumerate() {
            let movie_id = format!("movie_{}__{:04}", genre_name, i);
            movies.push(Embedding {
                id: movie_id,
                vector: vec,
            });
        }
    }
    
    movies
}

fn generate_documents(dim: usize, count: usize, rng: &mut ChaCha8Rng) -> Vec<Embedding> {
    let mut documents = Vec::new();
    
    let topics = vec!["tech", "business", "science", "health", "politics", "entertainment"];
    let items_per_topic = count / topics.len();
    
    for topic_name in topics {
        let center = generators::generate_embedding(dim, rng);
        let vectors = generators::generate_cluster(&center, 0.18, items_per_topic, rng);
        
        for (i, vec) in vectors.into_iter().enumerate() {
            let doc_id = format!("doc_{}__{:04}", topic_name, i);
            documents.push(Embedding {
                id: doc_id,
                vector: vec,
            });
        }
    }
    
    documents
}

fn generate_mixed(dim: usize, count: usize, rng: &mut ChaCha8Rng) -> Vec<Embedding> {
    let mut mixed = Vec::new();
    mixed.extend(generate_products(dim, count / 3, rng));
    mixed.extend(generate_movies(dim, count / 3, rng));
    mixed.extend(generate_documents(dim, count / 3, rng));
    mixed
}

fn generate_random(dim: usize, count: usize, rng: &mut ChaCha8Rng) -> Vec<Embedding> {
    let mut embeddings = Vec::new();
    for i in 0..count {
        let vec = generators::generate_embedding(dim, rng);
        embeddings.push(Embedding {
            id: format!("emb_{:06}", i),
            vector: vec,
        });
    }
    embeddings
}

fn main() {
    let args = Args::parse();
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    
    let embeddings = match args.theme.as_str() {
        "products" => generate_products(args.dim, args.count, &mut rng),
        "movies" => generate_movies(args.dim, args.count, &mut rng),
        "documents" => generate_documents(args.dim, args.count, &mut rng),
        "mixed" => generate_mixed(args.dim, args.count, &mut rng),
        "random" => generate_random(args.dim, args.count, &mut rng),
        _ => generate_random(args.dim, args.count, &mut rng),
    };
    
    for emb in embeddings {
        println!("{}", serde_json::to_string(&emb).unwrap());
    }
}
