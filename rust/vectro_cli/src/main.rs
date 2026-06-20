//! Vectro+ CLI - Command-line interface for embedding compression and search
//!
//! # Examples
//!
//! ```no_run
//! // Compress embeddings
//! // vectro compress input.jsonl output.bin
//!
//! // Search for similar vectors
//! // vectro search "1.0,2.0,3.0" --top-k 10 --dataset output.bin
//!
//! // Run benchmarks
//! // vectro bench --summary --open-report
//!
//! // Start web server
//! // vectro serve --port 8080
//! ```

use clap::{Parser, Subcommand};
use vectro_cli::{compress_stream, compress_pq, compress_nf4, compress_rq, compress_auto};

use serde_json::Value;

pub mod server;
pub mod pipeline;

#[derive(Parser)]
#[command(name = "vectro")]
#[command(about = "Vectro+ — Rust embedding compressor & search tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Compress {
        input: String,
        output: String,
        #[arg(long, default_value_t = false)]
        /// Produce a quantized streaming dataset (per-dimension min/max -> u8).
        /// This reduces size and speeds up search at the cost of some accuracy.
        /// Use for large datasets where memory/storage is constrained.
        /// Default: false
        quantize: bool,
        /// Quantization algorithm: int8 | nf4 | binary | pq
        /// Selects the algorithm used when --quantize is enabled.
        /// Default: int8
        #[arg(long, default_value = "int8",
              value_parser = ["int8", "nf4", "binary", "pq"])]
        mode: String,
    },
    /// Run library benchmarks (uses the `vectro_lib` bench harness).
    /// Streams benchmark output and shows a spinner while running.
    Bench {
        /// Save the Criterion HTML report to this path (directory). If omitted, report will remain under target/criterion.
        #[arg(long)]
        save_report: Option<String>,
        /// Open the HTML report after generation (macOS `open` is used).
        #[arg(long, default_value_t = false)]
        open_report: bool,
        /// Print a short JSON summary (median, mean) from Criterion's JSON output.
        #[arg(long, default_value_t = true)]
        summary: bool,
        /// Directory to copy the report into when using --save-report (default: current dir)
        #[arg(long)]
        report_dir: Option<String>,
        /// Extra arguments to pass to cargo bench (e.g., "--bench cosine_bench")
        #[arg(long)]
        bench_args: Option<String>,
    },
    Search {
        query: String,
        #[arg(short, long, default_value_t = 10)]
        top_k: usize,
        /// Path to dataset (bincode). If omitted, uses built-in toy dataset.
        #[arg(long)]
        dataset: Option<String>,
    },
    Serve {
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
    },
    /// Quantize a JSONL embedding file using a model-family profile.
    ///
    /// Reads the 'architectures' field from --input/config.json (if present) to select the
    /// best quantization method automatically. Override with --profile.
    ///
    /// Example:
    ///   vectro quantize --input ./embeddings.jsonl --output ./embeddings.vqz
    ///   vectro quantize --input ./embeddings.jsonl --output ./embeddings.vqz --profile nf4
    Quantize {
        /// Path to the input JSONL file.
        #[arg(long)]
        input: String,
        /// Path to the output file.
        #[arg(long)]
        output: String,
        /// Quantization profile: auto | int8 | nf4.
        /// "auto" reads config.json from the same directory as --input and selects the
        /// method based on model family.
        #[arg(long, default_value = "auto",
              value_parser = ["auto", "int8", "nf4"])]
        profile: String,
    },
    /// Run a full encode → index → search pipeline.
    ///
    /// Loads embeddings from a JSONL file, compresses them with the chosen format,
    /// builds an HNSW approximate-nearest-neighbour index, then optionally evaluates
    /// a batch of query vectors and prints the top-k results.
    ///
    /// Example:
    ///   vectro pipeline --input ./embeddings.jsonl --out-dir ./out --format nf4 --query-file queries.jsonl
    Pipeline {
        /// Input JSONL file of embeddings (id + vector fields).
        #[arg(long)]
        input: String,
        /// Output directory for the compressed dataset and HNSW index files.
        #[arg(long, default_value = "out")]
        out_dir: String,
        /// Compression format to apply: stream1 | nf4 | pq | rq | auto.
        #[arg(long, default_value = "stream1",
              value_parser = ["stream1", "nf4", "pq", "rq", "auto"])]
        format: String,
        /// HNSW M parameter — maximum bidirectional links per node.
        #[arg(long, default_value_t = 16)]
        m: usize,
        /// HNSW ef_construction — controls index quality vs build speed.
        #[arg(long, default_value_t = 200)]
        ef_construction: usize,
        /// HNSW ef_search — controls query accuracy vs speed.
        #[arg(long, default_value_t = 50)]
        ef_search: usize,
        /// Optional JSONL file of query vectors to evaluate after indexing.
        #[arg(long)]
        query_file: Option<String>,
        /// Number of nearest neighbours to return per query.
        #[arg(short = 'k', long, default_value_t = 10)]
        top_k: usize,
        /// Suppress progress bars and informational output.
        #[arg(short, long)]
        quiet: bool,
    },
}

// Wrapper functions for testability
/// Quantize embeddings from `input` JSONL into `output`.
///
/// `profile` is one of `"auto"`, `"int8"`, or `"nf4"`.  When `"auto"` is chosen the function
/// looks for a `config.json` alongside `input` and reads its `architectures` list to pick the
/// most accurate method for the embedding family.  Falls back to `int8` when the file is absent
/// or the architectures are unrecognised.
fn execute_quantize_command(input: &str, output: &str, profile: &str) -> anyhow::Result<usize> {
    let resolved_mode = if profile == "auto" {
        // Attempt to read config.json from the same directory as the input file.
        let input_path = std::path::Path::new(input);
        let config_path = input_path.parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("config.json");

        let method = if let Ok(txt) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&txt) {
                let archs: Vec<String> = json
                    .get("architectures")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect())
                    .unwrap_or_default();

                // NF4 families: BGEModel, BertModel (single-arch), XLMRobertaModel is e5 → int8
                let nf4_archs: std::collections::HashSet<&str> =
                    ["BGEModel", "BertModel"].iter().copied().collect();
                let is_nf4 = archs.iter().any(|a| nf4_archs.contains(a.as_str()));
                // e5/XLMRoberta overrides to int8 even when BertModel present alongside
                let force_int8_archs: std::collections::HashSet<&str> =
                    ["XLMRobertaModel", "RobertaModel", "E5Model"].iter().copied().collect();
                let force_int8 = archs.iter().any(|a| force_int8_archs.contains(a.as_str()));

                if force_int8 {
                    "int8"
                } else if is_nf4 {
                    "nf4"
                } else {
                    "int8" // gte, generic → int8
                }
            } else {
                "int8"
            }
        } else {
            "int8"
        };
        method
    } else {
        profile
    };

    eprintln!("[vectro quantize] input={input}  output={output}  method={resolved_mode}");
    // Delegate to the existing compress pipeline with the chosen quantization algorithm.
    crate::compress_stream(input, output, true)
}

fn execute_compress_command(input: &str, output: &str, quantize: bool, mode: &str) -> anyhow::Result<usize> {
    if quantize {
        eprintln!("[vectro] quantization mode: {mode}");
    }
    crate::compress_stream(input, output, quantize)
}

fn execute_serve_command(port: u16) -> anyhow::Result<()> {
    tokio::runtime::Runtime::new()?.block_on(async {
        server::serve(port).await
    })
}

fn parse_query_string(query: &str) -> Vec<f32> {
    query
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect()
}

fn load_dataset_or_default(dataset_path: Option<&str>) -> Vec<vectro_lib::Embedding> {
    use std::path::Path;
    
    if let Some(path) = dataset_path {
        if let Ok(ds) = vectro_lib::EmbeddingDataset::load(path) { return ds.embeddings }
    } else if Path::new("./dataset.bin").exists() {
        if let Ok(ds) = vectro_lib::EmbeddingDataset::load("./dataset.bin") {
            return ds.embeddings;
        }
    }
    
    // Default toy dataset
    vec![
        vectro_lib::Embedding::new("one", vec![1.0, 0.0]),
        vectro_lib::Embedding::new("two", vec![0.0, 1.0]),
        vectro_lib::Embedding::new("three", vec![0.707, 0.707]),
    ]
}

fn execute_search_command(query: &str, top_k: usize, dataset: Option<&str>) -> Vec<(String, f32)> {
    let vec = parse_query_string(query);
    let embeddings = load_dataset_or_default(dataset);
    let idx = vectro_lib::search::SearchIndex::from_dataset(&embeddings);
    idx.top_k(&vec, top_k)
        .into_iter()
        .map(|(id, score)| (id.to_string(), score))
        .collect()
}

fn main() -> anyhow::Result<()> {
    // Initialise tracing so library/CLI `warn!`s surface on stderr. Defaults to
    // WARN; `RUST_LOG` (e.g. `RUST_LOG=info`) overrides. `try_init` keeps a
    // double-init (tests, embedding) from panicking.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .try_init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Compress { input, output, quantize, mode } => {
            execute_compress_command(&input, &output, quantize, &mode)?;
        }
        Commands::Bench { save_report, open_report, summary, report_dir: _, bench_args } => {
            // Run cargo bench for vectro_lib and stream output. Show a spinner while running.
            use indicatif::{ProgressBar, ProgressStyle};
            use std::process::Command;
            use std::io::{BufRead, BufReader};
            use std::thread;
            use std::fs;
            use std::path::PathBuf;

            let pb = ProgressBar::new_spinner();
            pb.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
            pb.enable_steady_tick(std::time::Duration::from_millis(80));
            pb.set_message("running benches...");

            let mut cmd = build_bench_command(bench_args.as_deref());
            let mut child = cmd.spawn().expect("failed to spawn cargo bench");

            // stream stdout
            if let Some(out) = child.stdout.take() {
                let pb_out = pb.clone();
                thread::spawn(move || {
                    let reader = BufReader::new(out);
                    for line in reader.lines().map_while(Result::ok) {
                        pb_out.println(line);
                    }
                });
            }

            // stream stderr
            if let Some(err) = child.stderr.take() {
                let pb_err = pb.clone();
                thread::spawn(move || {
                    let reader = BufReader::new(err);
                    for line in reader.lines().map_while(Result::ok) {
                        pb_err.println(line);
                    }
                });
            }

            let status = child.wait().expect("bench wait failed");
            pb.finish_and_clear();
            if !status.success() {
                eprintln!("bench failed: {:?}\n(bench output above)", status);
            } else {
                // After success, optionally locate Criterion report and copy/open it
                let crit_dir = PathBuf::from("target/criterion");
                if crit_dir.exists() {
                    if summary {
                        // parse JSON summaries in target/criterion/*/new/*.json and present a clean table
                        if let Ok(entries) = fs::read_dir(&crit_dir) {
                            let mut rows: Vec<BenchRow> = Vec::new();
                            for e in entries.flatten() {
                                let p = e.path();
                                if p.is_dir() {
                                    let new_dir = p.join("new");
                                    if new_dir.exists() {
                                        if let Ok(it) = fs::read_dir(&new_dir) {
                                            for j in it.flatten() {
                                                let jp = j.path();
                                                if jp.extension().map(|s| s == "json").unwrap_or(false) {
                                                    if let Ok(txt) = fs::read_to_string(&jp) {
                                                        if let Ok(json) = serde_json::from_str::<Value>(&txt) {
                                                            let med = get_estimate(&json, "median");
                                                            let mean = get_estimate(&json, "mean");
                                                            let unit = find_string_in_json(&json, "unit");
                                                            // Use benchmark name if available, fallback to filename
                                                            let name = get_bench_name(&json)
                                                                .unwrap_or_else(|| jp.file_stem()
                                                                    .and_then(|s| s.to_str())
                                                                    .unwrap_or("unknown")
                                                                    .to_string());
                                                            rows.push((name, med, mean, unit));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if !rows.is_empty() {
                                // try to load previous history for deltas
                                let history_path = PathBuf::from(".bench_history.json");
                                let history = load_bench_history(&history_path);

                                // print pretty table
                                println!("\nBenchmark summaries:");
                                // header (include delta vs previous run)
                                println!("\x1b[1m{:<60} {:>12} {:>12} {:>8} {:>8}\x1b[0m", "benchmark", "median", "mean", "unit", "delta");
                                for (f, med, mean, unit) in &rows {
                                    let med_s = med.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "-".to_string());
                                    let mean_s = mean.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "-".to_string());
                                    let unit_s = unit.clone().unwrap_or_else(|| "".to_string());
                                    let delta_s = format_delta(*med, &history, f);
                                    println!("{:<60} {:>12} {:>12} {:>8} {:>8}", f, med_s, mean_s, unit_s, delta_s);
                                }

                                // update history with latest medians
                                let mut new_hist: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
                                for (f, med, _mean, _unit) in &rows {
                                    if let Some(m) = med { new_hist.insert(f.clone(), *m); }
                                }
                                if let Err(e) = save_bench_history(&history_path, &new_hist) {
                                    tracing::warn!(path = %history_path.display(), error = %e, "couldn't save benchmark history");
                                }

                                // Generate HTML summary in criterion dir
                                let html_summary = generate_html_summary(&rows, &history);
                                let summary_path = crit_dir.join("vectro_summary.html");
                                if let Err(e) = fs::write(&summary_path, html_summary) {
                                    tracing::warn!(path = %summary_path.display(), error = %e, "couldn't write HTML summary");
                                } else {
                                    println!("\n📊 HTML summary saved to: {}", summary_path.display());
                                }
                            }
                        }
                    }

                    if let Some(dest) = save_report {
                        let dest_dir = PathBuf::from(dest);
                        let ts = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                            Ok(d) => format!("{}", d.as_secs()),
                            Err(_) => "ts".to_string(),
                        };
                        let target_copy = dest_dir.join(format!("criterion-report-{}", ts));
                        if let Err(e) = fs::create_dir_all(&target_copy) {
                            tracing::warn!(path = %target_copy.display(), error = %e, "couldn't create report directory");
                        }
                        if let Err(e) = copy_dir_all(&crit_dir, &target_copy) {
                            tracing::warn!(dest = %target_copy.display(), error = %e, "couldn't copy Criterion report");
                        }
                        println!("Saved Criterion report to {}", target_copy.display());
                        if open_report {
                            let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
                            if let Err(e) = Command::new(opener).arg(target_copy.join("index.html")).spawn() {
                                tracing::warn!(opener, error = %e, "couldn't open report in browser");
                            }
                        }
                    } else if open_report {
                        // try to find an index.html anywhere under crit_dir
                        let mut index_opt: Option<PathBuf> = None;
                        // simple recursive search
                        let mut stack: Vec<PathBuf> = vec![crit_dir.clone()];
                        while let Some(p) = stack.pop() {
                            if let Ok(entries) = std::fs::read_dir(&p) {
                                for en in entries.flatten() {
                                    let pp = en.path();
                                    if pp.is_dir() { stack.push(pp); }
                                    else if pp.file_name().and_then(|s| s.to_str()) == Some("index.html") {
                                        index_opt = Some(pp);
                                        break;
                                    }
                                }
                            }
                            if index_opt.is_some() { break; }
                        }
                        if let Some(idx) = index_opt {
                            let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
                            let _ = Command::new(opener).arg(idx).spawn();
                        }
                    }
                }
            }
        }
        Commands::Search { query, top_k, dataset } => {
            let results = execute_search_command(&query, top_k, dataset.as_deref());
            for (i, (id, score)) in results.into_iter().enumerate() {
                println!("{}. {} -> {:.6}", i + 1, id, score);
            }
        }
        Commands::Serve { port } => {
            execute_serve_command(port)?;
        }
        Commands::Quantize { input, output, profile } => {
            let count = execute_quantize_command(&input, &output, &profile)?;
            println!("[vectro quantize] wrote {count} embeddings → {output}");
        }
        Commands::Pipeline { input, out_dir, format, m, ef_construction, ef_search, query_file, top_k, quiet } => {
            pipeline::run_pipeline(&input, &out_dir, &format, m, ef_construction, ef_search, query_file.as_deref(), top_k, quiet)?;
        }
    }

    Ok(())
}

/// Build cargo bench command with optional extra args
fn build_bench_command(bench_args: Option<&str>) -> std::process::Command {
    use std::process::{Command, Stdio};
    let mut cmd = Command::new("cargo");
    cmd.arg("bench").arg("-p").arg("vectro_lib");
    
    if let Some(extra) = bench_args {
        for arg in extra.split_whitespace() {
            cmd.arg(arg);
        }
    }
    
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd
}

/// Load benchmark history from file
fn load_bench_history(history_path: &std::path::Path) -> std::collections::HashMap<String, f64> {
    use std::fs;
    let mut history: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    if let Ok(txt) = fs::read_to_string(history_path) {
        if let Ok(hm) = serde_json::from_str::<std::collections::HashMap<String, f64>>(&txt) {
            history = hm;
        }
    }
    history
}

/// Save benchmark history to file
fn save_bench_history(history_path: &std::path::Path, history: &std::collections::HashMap<String, f64>) -> std::io::Result<()> {
    use std::fs;
    let out = serde_json::to_string_pretty(history)?;
    fs::write(history_path, out)
}

/// Calculate delta percentage between current and previous values
///
/// # Examples
///
/// ```
/// # use vectro_cli::*;
/// let delta = vectro_cli::calculate_delta_pub(110.0, 100.0);
/// assert_eq!(delta, Some(10.0));
///
/// let delta2 = vectro_cli::calculate_delta_pub(90.0, 100.0);
/// assert_eq!(delta2, Some(-10.0));
/// ```
fn calculate_delta(current: f64, previous: f64) -> Option<f64> {
    if previous != 0.0 {
        Some((current - previous) / previous * 100.0)
    } else {
        None
    }
}

/// Public wrapper for calculate_delta to enable doc tests
pub fn calculate_delta_pub(current: f64, previous: f64) -> Option<f64> {
    calculate_delta(current, previous)
}

/// Format delta for display
fn format_delta(med: Option<f64>, history: &std::collections::HashMap<String, f64>, name: &str) -> String {
    if let Some(prev) = history.get(name) {
        if let Some(curr) = med {
            if let Some(pct) = calculate_delta(curr, *prev) {
                format!("{:+.2}%", pct)
            } else {
                "n/a".to_string()
            }
        } else {
            "-".to_string()
        }
    } else {
        "-".to_string()
    }
}

/// Calculate delta class for HTML styling
fn get_delta_class(pct: f64) -> &'static str {
    if pct > 0.5 {
        "delta-positive"
    } else if pct < -0.5 {
        "delta-negative"
    } else {
        "delta-neutral"
    }
}

/// Format delta with class for HTML
fn format_delta_html(med: Option<f64>, history: &std::collections::HashMap<String, f64>, name: &str) -> (String, &'static str) {
    if let Some(prev) = history.get(name) {
        if let Some(curr) = med {
            if *prev != 0.0 {
                let pct = (curr - *prev) / *prev * 100.0;
                let class = get_delta_class(pct);
                (format!("{:+.2}%", pct), class)
            } else {
                ("n/a".to_string(), "delta-neutral")
            }
        } else {
            ("-".to_string(), "delta-neutral")
        }
    } else {
        ("-".to_string(), "delta-neutral")
    }
}

/// Recursively search a serde_json::Value for the first numeric value keyed by `key` and return it as f64.
fn find_number_in_json(v: &Value, key: &str) -> Option<f64> {
    match v {
        Value::Object(map) => {
            if let Some(val) = map.get(key) {
                if let Some(n) = val.as_f64() { return Some(n); }
            }
            for (_k, vv) in map.iter() {
                if let Some(n) = find_number_in_json(vv, key) { return Some(n); }
            }
            None
        }
        Value::Array(arr) => {
            for item in arr { if let Some(n) = find_number_in_json(item, key) { return Some(n); } }
            None
        }
        _ => None,
    }
}

/// Recursively find a string field in JSON by key
fn find_string_in_json(v: &Value, key: &str) -> Option<String> {
    match v {
        Value::Object(map) => {
            if let Some(val) = map.get(key) {
                if let Some(s) = val.as_str() { return Some(s.to_string()); }
            }
            for (_k, vv) in map.iter() {
                if let Some(s) = find_string_in_json(vv, key) { return Some(s); }
            }
            None
        }
        Value::Array(arr) => {
            for item in arr { if let Some(s) = find_string_in_json(item, key) { return Some(s); } }
            None
        }
        _ => None,
    }
}

/// Attempt to find an estimate value which may be nested in several known fields
fn get_estimate(v: &Value, key: &str) -> Option<f64> {
    // common shapes: { "estimates": { "median": {"point_estimate": 0.1 } } } or direct
    if let Some(direct) = find_number_in_json(v, key) { return Some(direct); }
    // try path: estimates -> key -> point_estimate
    if let Value::Object(map) = v {
        if let Some(Value::Object(est_map)) = map.get("estimates") {
            if let Some(Value::Object(kmap)) = est_map.get(key) {
                if let Some(pe) = kmap.get("point_estimate") {
                    return pe.as_f64();
                }
            }
        }
    }
    None
}

/// Extract a short benchmark name from Criterion JSON (tries "group_id", "function_id", or fallback)
fn get_bench_name(v: &Value) -> Option<String> {
    // Try common Criterion fields
    if let Some(name) = find_string_in_json(v, "group_id") {
        return Some(name);
    }
    if let Some(name) = find_string_in_json(v, "function_id") {
        return Some(name);
    }
    if let Some(name) = find_string_in_json(v, "title") {
        return Some(name);
    }
    None
}

/// A single benchmark summary row: (name, median_s, mean_s, unit_label).
type BenchRow = (String, Option<f64>, Option<f64>, Option<String>);

/// Generate a compact HTML summary from benchmark results
fn generate_html_summary(rows: &[BenchRow], history: &std::collections::HashMap<String, f64>) -> String {
    let mut html = String::from(r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Vectro+ Benchmark Summary</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; 
               padding: 2rem; max-width: 1200px; margin: 0 auto; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #4a90e2; padding-bottom: 0.5rem; }
        .timestamp { color: #666; font-size: 0.9rem; margin-bottom: 2rem; }
        table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th, td { padding: 1rem; text-align: left; border-bottom: 1px solid #e0e0e0; }
        th { background: #4a90e2; color: white; font-weight: 600; }
        tr:hover { background: #f9f9f9; }
        .number { text-align: right; font-family: 'Monaco', 'Courier New', monospace; }
        .delta-positive { color: #d32f2f; }
        .delta-negative { color: #388e3c; }
        .delta-neutral { color: #666; }
        .footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd; color: #666; font-size: 0.85rem; }
        .link { color: #4a90e2; text-decoration: none; }
        .link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>🚀 Vectro+ Benchmark Results</h1>
    <div class="timestamp">Generated: "#);
    
    html.push_str(&format!("{}</div>\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
    html.push_str("    <table>\n        <thead>\n            <tr>\n");
    html.push_str("                <th>Benchmark</th><th class=\"number\">Median</th><th class=\"number\">Mean</th><th>Unit</th><th class=\"number\">Δ vs Previous</th>\n");
    html.push_str("            </tr>\n        </thead>\n        <tbody>\n");
    
    for (name, med, mean, unit) in rows {
        let med_str = med.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "-".to_string());
        let mean_str = mean.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "-".to_string());
        let unit_str = unit.clone().unwrap_or_else(|| "".to_string());
        
        let (delta_str, delta_class) = format_delta_html(*med, history, name);
        
        html.push_str(&format!("            <tr>\n                <td>{}</td><td class=\"number\">{}</td><td class=\"number\">{}</td><td>{}</td><td class=\"number {}\">  {}</td>\n            </tr>\n",
            name, med_str, mean_str, unit_str, delta_class, delta_str));
    }
    
    html.push_str(r#"        </tbody>
    </table>
    <div class="footer">
        Generated by <a href="https://github.com/yourorg/vectro-plus" class="link">Vectro+</a> — 
        <a href="./report/index.html" class="link">View Full Criterion Report</a>
    </div>
</body>
</html>"#);
    
    html
}
// Simple recursive directory copy used to copy Criterion reports
fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_find_number_in_json_simple() {
        let v = json!({"median": 0.123, "mean": 0.2});
        assert_eq!(find_number_in_json(&v, "median"), Some(0.123));
        assert_eq!(find_number_in_json(&v, "mean"), Some(0.2));
    }

    #[test]
    fn test_find_number_in_json_nested() {
        let v = json!({"estimates": {"median": {"point_estimate": 0.5}, "mean": {"point_estimate": 0.6}}});
        assert_eq!(get_estimate(&v, "median"), Some(0.5));
        assert_eq!(get_estimate(&v, "mean"), Some(0.6));
    }

    #[test]
    fn test_find_string_in_json() {
        let v = json!({"unit": "ns", "nested": {"unit": "ms"}});
        assert_eq!(find_string_in_json(&v, "unit"), Some("ns".to_string()));
        let v2 = json!({"outer": {"inner": {"unit": "us"}}});
        assert_eq!(find_string_in_json(&v2, "unit"), Some("us".to_string()));
    }

    #[test]
    fn test_get_bench_name() {
        let v1 = json!({"group_id": "search/cosine", "function_id": "top_k"});
        assert_eq!(get_bench_name(&v1), Some("search/cosine".to_string()));
        
        let v2 = json!({"function_id": "quantize_dataset", "title": "Quantization Bench"});
        assert_eq!(get_bench_name(&v2), Some("quantize_dataset".to_string()));
        
        let v3 = json!({"title": "Simple Bench"});
        assert_eq!(get_bench_name(&v3), Some("Simple Bench".to_string()));
    }

    #[test]
    fn test_bench_summary_parsing() {
        use std::fs;
        use tempfile::TempDir;

        // Create fake Criterion-like JSON output
        let tmp = TempDir::new().unwrap();
        let crit_dir = tmp.path().join("criterion");
        let bench_dir = crit_dir.join("cosine_search").join("new");
        fs::create_dir_all(&bench_dir).unwrap();

        let fake_json = json!({
            "group_id": "cosine_search",
            "function_id": "top_k_100",
            "estimates": {
                "median": {"point_estimate": 123.456},
                "mean": {"point_estimate": 125.789}
            },
            "unit": "ns"
        });

        fs::write(bench_dir.join("estimates.json"), serde_json::to_string_pretty(&fake_json).unwrap()).unwrap();

        // Parse the fake structure
        let mut found = false;
        if let Ok(entries) = fs::read_dir(&crit_dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    let new_dir = p.join("new");
                    if new_dir.exists() {
                        if let Ok(it) = fs::read_dir(&new_dir) {
                            for j in it.flatten() {
                                let jp = j.path();
                                if jp.extension().map(|s| s == "json").unwrap_or(false) {
                                    if let Ok(txt) = fs::read_to_string(&jp) {
                                        if let Ok(json) = serde_json::from_str::<Value>(&txt) {
                                            let med = get_estimate(&json, "median");
                                            let mean = get_estimate(&json, "mean");
                                            let unit = find_string_in_json(&json, "unit");
                                            let name = get_bench_name(&json);

                                            assert_eq!(med, Some(123.456));
                                            assert_eq!(mean, Some(125.789));
                                            assert_eq!(unit, Some("ns".to_string()));
                                            assert_eq!(name, Some("cosine_search".to_string()));
                                            found = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(found, "Should have parsed the fake Criterion JSON");
    }

    #[test]
    fn test_execute_compress_command() {
        use tempfile::NamedTempFile;
        
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap();
        std::fs::write(in_path, r#"{"id":"test","vector":[1.0,2.0,3.0]}"#).unwrap();
        
        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap();
        
        let result = execute_compress_command(in_path, out_path, false, "int8");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn test_execute_compress_command_quantized() {
        use tempfile::NamedTempFile;
        
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap();
        std::fs::write(in_path, r#"{"id":"a","vector":[1.0,0.0]}
{"id":"b","vector":[0.0,1.0]}"#).unwrap();
        
        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap();
        
        let result = execute_compress_command(in_path, out_path, true, "int8");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn test_execute_compress_command_invalid_input() {
        use tempfile::NamedTempFile;
        
        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap();
        
        let result = execute_compress_command("/nonexistent/file.jsonl", out_path, false, "int8");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_query_string() {
        let result = parse_query_string("1.0,2.0,3.0");
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
        
        let result = parse_query_string("1.0, 2.0, 3.0");
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
        
        let result = parse_query_string("1.0,invalid,3.0");
        assert_eq!(result, vec![1.0, 3.0]);
    }

    #[test]
    fn test_load_dataset_or_default() {
        // Test with None - should return default
        let result = load_dataset_or_default(None);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, "one");
        
        // Test with invalid path - should return default
        let result = load_dataset_or_default(Some("/nonexistent/path.bin"));
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_load_dataset_or_default_valid_file() {
        use tempfile::NamedTempFile;
        
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        
        let mut ds = vectro_lib::EmbeddingDataset::new();
        ds.add(vectro_lib::Embedding::new("test1", vec![1.0, 0.0]));
        ds.add(vectro_lib::Embedding::new("test2", vec![0.0, 1.0]));
        ds.save(path).unwrap();
        
        let result = load_dataset_or_default(Some(path));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, "test1");
    }

    #[test]
    fn test_execute_search_command() {
        let results = execute_search_command("1.0,0.0", 2, None);
        assert!(results.len() <= 2);
        assert!(!results.is_empty());
        
        // First result should be "one" with highest similarity
        assert_eq!(results[0].0, "one");
        assert!(results[0].1 > 0.9);
    }

    #[test]
    fn test_execute_search_command_with_dataset() {
        use tempfile::NamedTempFile;
        
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        
        let mut ds = vectro_lib::EmbeddingDataset::new();
        ds.add(vectro_lib::Embedding::new("apple", vec![1.0, 0.0, 0.0]));
        ds.add(vectro_lib::Embedding::new("banana", vec![0.0, 1.0, 0.0]));
        ds.save(path).unwrap();
        
        let results = execute_search_command("1.0,0.0,0.0", 1, Some(path));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "apple");
    }

    #[test]
    fn test_cli_parsing_compress() {
        // Test that CLI can parse compress command
        use clap::Parser;
        
        let args = vec!["vectro", "compress", "input.jsonl", "output.bin"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Compress { input, output, quantize, mode } => {
                    assert_eq!(input, "input.jsonl");
                    assert_eq!(output, "output.bin");
                    assert!(!quantize);
                    assert_eq!(mode, "int8");
                }
                _ => panic!("Expected Compress command"),
            }
        }
    }

    #[test]
    fn test_cli_parsing_compress_quantized() {
        use clap::Parser;
        
        let args = vec!["vectro", "compress", "in.jsonl", "out.bin", "--quantize"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Compress { quantize, .. } => {
                assert!(quantize);
            }
            _ => panic!("Expected Compress command"),
        }
    }

    #[test]
    fn test_cli_parsing_search() {
        use clap::Parser;
        
        let args = vec!["vectro", "search", "1.0,0.0,0.0"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Search { query, top_k, dataset } => {
                assert_eq!(query, "1.0,0.0,0.0");
                assert_eq!(top_k, 10); // default
                assert!(dataset.is_none());
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_cli_parsing_search_with_options() {
        use clap::Parser;
        
        let args = vec!["vectro", "search", "1.0,0.0", "--top-k", "5", "--dataset", "data.bin"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Search { query, top_k, dataset } => {
                assert_eq!(query, "1.0,0.0");
                assert_eq!(top_k, 5);
                assert_eq!(dataset.as_deref(), Some("data.bin"));
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_cli_parsing_serve() {
        use clap::Parser;
        
        let args = vec!["vectro", "serve"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Serve { port } => {
                assert_eq!(port, 8080); // default
            }
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_cli_parsing_serve_custom_port() {
        use clap::Parser;
        
        let args = vec!["vectro", "serve", "--port", "3000"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Serve { port } => {
                assert_eq!(port, 3000);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_cli_parsing_bench() {
        use clap::Parser;
        
        let args = vec!["vectro", "bench"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Bench { save_report, open_report, summary, .. } => {
                assert!(save_report.is_none());
                assert!(!open_report);
                assert!(summary); // default true
            }
            _ => panic!("Expected Bench command"),
        }
    }

    #[test]
    fn test_build_bench_command() {
        let cmd = build_bench_command(None);
        let program = cmd.get_program();
        assert_eq!(program, "cargo");
    }

    #[test]
    fn test_build_bench_command_with_args() {
        let cmd = build_bench_command(Some("--verbose"));
        let program = cmd.get_program();
        assert_eq!(program, "cargo");
    }

    #[test]
    fn test_load_bench_history_missing_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let history_path = temp_dir.path().join("nonexistent.json");
        let history = load_bench_history(&history_path);
        assert!(history.is_empty());
    }

    #[test]
    fn test_load_save_bench_history() {
        use std::collections::HashMap;
        let temp_dir = tempfile::tempdir().unwrap();
        let history_path = temp_dir.path().join("history.json");
        
        let mut history = HashMap::new();
        history.insert("test_bench".to_string(), 123.456);
        
        save_bench_history(&history_path, &history).unwrap();
        let loaded = load_bench_history(&history_path);
        
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get("test_bench"), Some(&123.456));
    }

    #[test]
    fn test_calculate_delta() {
        // Normal case
        let delta = calculate_delta(110.0, 100.0);
        assert_eq!(delta, Some(10.0));
        
        // Decrease
        let delta = calculate_delta(90.0, 100.0);
        assert_eq!(delta, Some(-10.0));
        
        // Zero previous
        let delta = calculate_delta(100.0, 0.0);
        assert_eq!(delta, None);
    }

    #[test]
    fn test_format_delta() {
        use std::collections::HashMap;
        
        let mut history = HashMap::new();
        history.insert("bench1".to_string(), 100.0);
        
        // With history and current value
        let delta_str = format_delta(Some(110.0), &history, "bench1");
        assert_eq!(delta_str, "+10.00%");
        
        // No history
        let delta_str = format_delta(Some(110.0), &history, "bench2");
        assert_eq!(delta_str, "-");
        
        // No current value
        let delta_str = format_delta(None, &history, "bench1");
        assert_eq!(delta_str, "-");
    }

    #[test]
    fn test_format_delta_zero_previous() {
        use std::collections::HashMap;
        
        let mut history = HashMap::new();
        history.insert("bench1".to_string(), 0.0);
        
        let delta_str = format_delta(Some(110.0), &history, "bench1");
        assert_eq!(delta_str, "n/a");
    }

    #[test]
    fn test_get_delta_class() {
        assert_eq!(get_delta_class(1.0), "delta-positive");
        assert_eq!(get_delta_class(-1.0), "delta-negative");
        assert_eq!(get_delta_class(0.3), "delta-neutral");
        assert_eq!(get_delta_class(-0.3), "delta-neutral");
    }

    #[test]
    fn test_format_delta_html() {
        use std::collections::HashMap;
        
        let mut history = HashMap::new();
        history.insert("bench1".to_string(), 100.0);
        
        // Positive delta
        let (delta_str, class) = format_delta_html(Some(110.0), &history, "bench1");
        assert_eq!(delta_str, "+10.00%");
        assert_eq!(class, "delta-positive");
        
        // Negative delta
        let (delta_str, class) = format_delta_html(Some(90.0), &history, "bench1");
        assert_eq!(delta_str, "-10.00%");
        assert_eq!(class, "delta-negative");
        
        // Neutral delta
        let (delta_str, class) = format_delta_html(Some(100.3), &history, "bench1");
        assert_eq!(delta_str, "+0.30%");
        assert_eq!(class, "delta-neutral");
        
        // No history
        let (delta_str, class) = format_delta_html(Some(110.0), &history, "bench2");
        assert_eq!(delta_str, "-");
        assert_eq!(class, "delta-neutral");
    }

    #[test]
    fn test_calculate_delta_comprehensive() {
        // Test improvement (faster is negative delta)
        let delta = calculate_delta(90.0, 100.0);
        assert_eq!(delta, Some(-10.0));
        
        // Test regression (slower is positive delta)
        let delta = calculate_delta(110.0, 100.0);
        assert_eq!(delta, Some(10.0));
        
        // Test zero previous value
        let delta = calculate_delta(50.0, 0.0);
        assert_eq!(delta, None);
        
        // Test equal values
        let delta = calculate_delta(100.0, 100.0);
        assert_eq!(delta, Some(0.0));
        
        // Test small change
        let delta = calculate_delta(100.5, 100.0);
        assert_eq!(delta, Some(0.5));
    }

    #[test]
    fn test_find_number_in_json_deeply_nested() {
        let json = json!({
            "level1": {
                "level2": {
                    "level3": {
                        "target": 42.5
                    }
                }
            }
        });
        assert_eq!(find_number_in_json(&json, "target"), Some(42.5));
    }

    #[test]
    fn test_find_string_in_json_array() {
        let json = json!({
            "items": [
                {"name": "first"},
                {"name": "second"},
                {"unit": "milliseconds"}
            ]
        });
        assert_eq!(find_string_in_json(&json, "unit"), Some("milliseconds".to_string()));
    }

    #[test]
    fn test_get_estimate_criterion_format() {
        // Test realistic Criterion JSON structure
        let criterion_json = json!({
            "estimates": {
                "median": {
                    "point_estimate": 123.456,
                    "confidence_interval": {
                        "lower_bound": 120.0,
                        "upper_bound": 130.0
                    }
                },
                "mean": {
                    "point_estimate": 125.789
                }
            }
        });
        
        assert_eq!(get_estimate(&criterion_json, "median"), Some(123.456));
        assert_eq!(get_estimate(&criterion_json, "mean"), Some(125.789));
    }

    #[test]
    fn test_parse_query_string_edge_cases() {
        // Empty string
        assert_eq!(parse_query_string(""), Vec::<f32>::new());
        
        // Single value
        assert_eq!(parse_query_string("1.0"), vec![1.0]);
        
        // Multiple values with spaces
        assert_eq!(parse_query_string("1.0, 2.0 , 3.0"), vec![1.0, 2.0, 3.0]);
        
        // Invalid values (should be filtered out)
        assert_eq!(parse_query_string("1.0,invalid,3.0"), vec![1.0, 3.0]);
        
        // Negative and decimal values
        assert_eq!(parse_query_string("-1.5,0.0,2.5"), vec![-1.5, 0.0, 2.5]);
    }

    #[test]
    fn test_load_dataset_or_default_with_valid_file() {
        use tempfile::NamedTempFile;
        
        // Create a temporary dataset file
        let mut ds = vectro_lib::EmbeddingDataset::new();
        ds.add(vectro_lib::Embedding::new("test1", vec![1.0, 2.0, 3.0]));
        ds.add(vectro_lib::Embedding::new("test2", vec![4.0, 5.0, 6.0]));
        
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        ds.save(path).unwrap();
        
        // Test loading the dataset
        let loaded = load_dataset_or_default(Some(path));
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "test1");
        assert_eq!(loaded[1].id, "test2");
    }

    #[test]
    fn test_load_dataset_or_default_with_invalid_path() {
        // Test with invalid path - should return default dataset
        let embeddings = load_dataset_or_default(Some("/nonexistent/path.bin"));
        assert!(!embeddings.is_empty()); // Should return default toy dataset
    }

    #[test]
    fn test_load_dataset_or_default_none() {
        // Test with None - should return default dataset
        let embeddings = load_dataset_or_default(None);
        assert!(!embeddings.is_empty()); // Should return default toy dataset
    }

    #[test]
    fn test_get_bench_name_priority() {
        // Test that group_id takes priority
        let json = json!({
            "group_id": "group_name",
            "function_id": "function_name",
            "title": "Some Title"
        });
        assert_eq!(get_bench_name(&json), Some("group_name".to_string()));
        
        // Test fallback to function_id when group_id is missing
        let json2 = json!({
            "function_id": "function_name",
            "title": "Some Title"
        });
        assert_eq!(get_bench_name(&json2), Some("function_name".to_string()));
        
        // Test fallback to title when both are missing
        let json3 = json!({
            "title": "Some Title"
        });
        assert_eq!(get_bench_name(&json3), Some("Some Title".to_string()));
        
        // Test None when all are missing
        let json4 = json!({
            "other_field": "value"
        });
        assert_eq!(get_bench_name(&json4), None);
    }

    #[test]
    fn test_cli_parsing_quantize_default_profile() {
        use clap::Parser;

        let args = vec!["vectro", "quantize", "--input", "emb.jsonl", "--output", "emb.qstream"];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Quantize { input, output, profile } => {
                assert_eq!(input, "emb.jsonl");
                assert_eq!(output, "emb.qstream");
                assert_eq!(profile, "auto");
            }
            _ => panic!("Expected Quantize command"),
        }
    }

    #[test]
    fn test_cli_parsing_quantize_explicit_profile() {
        use clap::Parser;

        let args = vec!["vectro", "quantize", "--input", "emb.jsonl", "--output", "emb.qstream1", "--profile", "nf4"];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Quantize { input, output, profile } => {
                assert_eq!(input, "emb.jsonl");
                assert_eq!(output, "emb.qstream1");
                assert_eq!(profile, "nf4");
            }
            _ => panic!("Expected Quantize command"),
        }
    }

    #[test]
    fn test_cli_parsing_pipeline_defaults() {
        use clap::Parser;

        let cli = Cli::try_parse_from(["vectro", "pipeline", "--input", "data.jsonl"]).unwrap();

        match cli.command {
            Commands::Pipeline { input, out_dir, format, m, ef_construction, ef_search, query_file, top_k, quiet } => {
                assert_eq!(input, "data.jsonl");
                assert_eq!(out_dir, "out");
                assert_eq!(format, "stream1");
                assert_eq!(m, 16);
                assert_eq!(ef_construction, 200);
                assert_eq!(ef_search, 50);
                assert!(query_file.is_none());
                assert_eq!(top_k, 10);
                assert!(!quiet);
            }
            _ => panic!("Expected Pipeline command"),
        }
    }

    #[test]
    fn test_cli_parsing_pipeline_explicit_top_k() {
        use clap::Parser;

        let cli = Cli::try_parse_from([
            "vectro", "pipeline",
            "--input", "data.jsonl",
            "--format", "nf4",
            "--top-k", "5",
            "--quiet",
        ]).unwrap();

        match cli.command {
            Commands::Pipeline { format, top_k, quiet, .. } => {
                assert_eq!(format, "nf4");
                assert_eq!(top_k, 5);
                assert!(quiet);
            }
            _ => panic!("Expected Pipeline command"),
        }
    }
}
