//! End-to-end pipeline: compress → build HNSW index → optional batch search.
//!
//! Adapted from vectro-plus v2.1.0 into vectro's vectro_lib v4.0.0 API.
use anyhow::{Context, Result};
use std::fs;

/// Run the full pipeline:
///   1. Compress embeddings from `input` (JSONL) into `{out_dir}/compressed.stream1`
///   2. Build an HNSW index over the compressed dataset
///   3. Save the index to `{out_dir}/index.bin`
///   4. If `query_file` is given, run all queries and write JSONL results to stdout.
#[allow(clippy::too_many_arguments)]
pub fn run_pipeline(
    input: &str,
    out_dir: &str,
    format: &str,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    query_file: Option<&str>,
    top_k: usize,
    quiet: bool,
) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create output directory: {out_dir}"))?;

    let compressed_path = format!("{out_dir}/compressed.stream1");

    if !quiet {
        eprintln!("→ compressing [{format}] {input} → {compressed_path}");
    }

    let n = match format {
        "pq" => crate::compress_pq(input, &compressed_path, 8, 256)
            .with_context(|| format!("compress_pq failed for {input}"))?,
        "nf4" => crate::compress_nf4(input, &compressed_path)
            .with_context(|| format!("compress_nf4 failed for {input}"))?,
        "rq" => crate::compress_rq(input, &compressed_path, 2, 8, 256)
            .with_context(|| format!("compress_rq failed for {input}"))?,
        "auto" => crate::compress_auto(input, &compressed_path, 0.97, 8.0)
            .with_context(|| format!("compress_auto failed for {input}"))?,
        _ => crate::compress_stream(input, &compressed_path, false)
            .with_context(|| format!("compress_stream failed for {input}"))?,
    };

    if !quiet {
        eprintln!("  ✓ compressed {n} vectors");
    }

    if !quiet {
        eprintln!("→ loading dataset from {compressed_path}");
    }

    let dataset = vectro_lib::EmbeddingDataset::load(&compressed_path)
        .with_context(|| format!("failed to load dataset from {compressed_path}"))?;

    if !quiet {
        eprintln!("  ✓ loaded {} embeddings", dataset.embeddings.len());
    }

    let index_path = format!("{out_dir}/index.bin");

    if !quiet {
        eprintln!(
            "→ building HNSW index (M={m}, ef_construction={ef_construction}, ef_search={ef_search})"
        );
    }

    let vectors: Vec<Vec<f32>> = dataset
        .embeddings
        .iter()
        .map(|e| e.vector.clone())
        .collect();
    let mut index = vectro_lib::index::hnsw::HnswIndex::new(m, ef_construction);
    index.add_batch(&vectors);

    if !quiet {
        eprintln!("  ✓ built index with {} nodes", index.len());
    }

    index
        .save(std::path::Path::new(&index_path))
        .with_context(|| format!("failed to save index to {index_path}"))?;

    if !quiet {
        eprintln!("  ✓ saved index → {index_path}");
    }

    if let Some(qf) = query_file {
        run_queries(&index, &dataset.embeddings, qf, top_k, ef_search, quiet)?;
    }

    Ok(())
}

/// Run queries from a JSONL file against the built HNSW index.
///
/// Each line must be `{"id": "...", "vector": [f32, ...]}`.
/// Each result line is `{"query_id": "...", "results": [{"id": "...", "score": f32}]}`.
fn run_queries(
    index: &vectro_lib::index::hnsw::HnswIndex,
    embeddings: &[vectro_lib::Embedding],
    query_file: &str,
    top_k: usize,
    ef_search: usize,
    quiet: bool,
) -> Result<()> {
    use std::io::{BufRead, BufReader, Write};

    if !quiet {
        eprintln!("→ running queries from {query_file}");
    }

    let f = fs::File::open(query_file)
        .with_context(|| format!("failed to open query file: {query_file}"))?;
    let reader = BufReader::new(f);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut query_count = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        let line =
            line.with_context(|| format!("I/O error reading {query_file} at line {line_no}"))?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let v: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("JSON parse error at line {line_no}: {line}"))?;

        let query_id = v["id"].as_str().unwrap_or("?").to_owned();
        let vector: Vec<f32> = v["vector"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing 'vector' field at line {line_no}"))?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();

        // HnswIndex::search returns (usize_index, f32_score); map index → embedding ID.
        let results = index.search(&vector, top_k, ef_search);
        let result_json: Vec<serde_json::Value> = results
            .iter()
            .map(|(idx, score)| {
                let id = embeddings.get(*idx).map(|e| e.id.as_str()).unwrap_or("?");
                serde_json::json!({"id": id, "score": score})
            })
            .collect();

        let row = serde_json::json!({ "query_id": query_id, "results": result_json });
        writeln!(out, "{row}")?;
        query_count += 1;
    }

    if !quiet {
        eprintln!("  ✓ processed {query_count} queries");
    }
    Ok(())
}
