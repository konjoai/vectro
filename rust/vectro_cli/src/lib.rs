use std::io::{BufRead, BufReader, Write};
use indicatif::{ProgressBar, ProgressStyle};

pub fn compress_stream(input: &str, output: &str, quantize: bool) -> anyhow::Result<usize> {
    use crossbeam_channel::{bounded, Sender, Receiver};
    use std::thread;

    let header = b"VECTRO+STREAM1\n";
    let infile = std::fs::File::open(input)?;
    let reader = BufReader::new(infile);

    let outfile = std::fs::File::create(output)?;
    let writer_buf = std::io::BufWriter::new(outfile);

    // channels
    let (item_tx, item_rx): (Sender<vectro_lib::Embedding>, Receiver<vectro_lib::Embedding>) = bounded(1024);
    let (bytes_tx, bytes_rx): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = bounded(1024);

    // writer thread (non-quantized path will spawn writer now; quantized path spawns writer after tables computed)
    let out_clone = output.to_string();
    let qheader = b"VECTRO+QSTREAM1\n";
    let mut writer_handle_opt = None;
    // prepare worker handles container
    let mut worker_handles: Vec<std::thread::JoinHandle<()>> = Vec::new();
    if !quantize {
        let mut w = writer_buf;
        let rx_for_writer = bytes_rx.clone();
        let out_for_writer = out_clone.clone();
        let header_local = *header;
        let handle = thread::spawn(move || -> anyhow::Result<()> {
            w.write_all(&header_local)?;
            let mut written = 0usize;
            while let Ok(bytes) = rx_for_writer.recv() {
                let len = (bytes.len() as u32).to_le_bytes();
                w.write_all(&len)?;
                w.write_all(&bytes)?;
                written += 1;
            }
            w.flush()?;
            eprintln!("wrote {} entries to {}", written, out_for_writer);
            Ok(())
        });
        writer_handle_opt = Some(handle);
        // spawn workers for non-quantized path
        let workers = num_cpus::get().max(1);
        for _ in 0..workers {
            let r = item_rx.clone();
            let tx = bytes_tx.clone();
            worker_handles.push(thread::spawn(move || {
                while let Ok(e) = r.recv() {
                    if let Ok(bytes) = bincode::serialize(&e) {
                        let _ = tx.send(bytes);
                    }
                }
            }));
        }
    }

    // don't spawn workers yet; will spawn depending on quantize mode

    // progress bar
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    if quantize {
        pb.set_message("parsing and computing quant tables...");
    } else {
        pb.set_message("compressing (streaming bincode)...");
    }

    // reader: parse lines and collect embeddings
    let mut parsed = 0usize;
    // collect embeddings when quantizing
    let mut collected_embeddings: Vec<vectro_lib::Embedding> = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() { continue; }

        // try JSON
        let mut pushed = false;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            if let (Some(id), Some(vec)) = (val.get("id"), val.get("vector")) {
                if let (Some(id_str), Some(arr)) = (id.as_str(), vec.as_array()) {
                    let mut v = Vec::with_capacity(arr.len());
                    for x in arr { if let Some(flt) = x.as_f64() { v.push(flt as f32); } }
                    let emb = vectro_lib::Embedding::new(id_str, v.clone());
                    if quantize { collected_embeddings.push(emb.clone()); } else { let _ = item_tx.send(emb); }
                    parsed += 1;
                    pushed = true;
                }
            }
        }
        if !pushed {
            // CSV
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let id = parts[0].to_string();
                let mut v = Vec::new();
                for p in &parts[1..] { if let Ok(f) = p.trim().parse::<f32>() { v.push(f); } }
                let emb = vectro_lib::Embedding::new(id, v.clone());
                if quantize { collected_embeddings.push(emb.clone()); } else { let _ = item_tx.send(emb); }
                parsed += 1;
            }
        }

        if parsed.is_multiple_of(100) { pb.set_message(format!("parsed {} entries", parsed)); }
    }

    if quantize {
        // compute tables using vectro_lib::search::quant::quantize_dataset
        let vectors: Vec<Vec<f32>> = collected_embeddings.iter().map(|e| e.vector.clone()).collect();
        let (tables, _qvecs) = vectro_lib::search::quant::quantize_dataset(&vectors);
        // serialize tables to bincode
        let tables_blob = bincode::serialize(&tables)?;

        // write header + tables to file, then spawn writer thread to append entries
        {
            // overwrite/create file and write header+tables
            let mut f = std::fs::File::create(output)?;
            let mut w = std::io::BufWriter::new(&mut f);
            w.write_all(qheader)?;
            let table_count = (tables.len() as u32).to_le_bytes();
            let dim = (if !tables.is_empty() { tables.len() as u32 } else { 0u32 }).to_le_bytes();
            let tables_len = (tables_blob.len() as u32).to_le_bytes();
            w.write_all(&table_count)?;
            w.write_all(&dim)?;
            w.write_all(&tables_len)?;
            w.write_all(&tables_blob)?;
            w.flush()?;
        }

        // spawn writer that appends entries
        let outfile = std::fs::OpenOptions::new().append(true).open(output)?;
        let writer_buf = std::io::BufWriter::new(outfile);
        let out_clone2 = out_clone.clone();
        let handle = thread::spawn(move || -> anyhow::Result<()> {
            let mut w = writer_buf;
            let mut written = 0usize;
            while let Ok(bytes) = bytes_rx.recv() {
                let len = (bytes.len() as u32).to_le_bytes();
                w.write_all(&len)?;
                w.write_all(&bytes)?;
                written += 1;
            }
            w.flush()?;
            eprintln!("wrote {} entries to {}", written, out_clone2);
            Ok(())
        });
        writer_handle_opt = Some(handle);

        // spawn workers to quantize embeddings
        let workers = num_cpus::get().max(1);
        use crossbeam_channel::bounded;
        let (item_tx2, item_rx2) = bounded::<vectro_lib::Embedding>(1024);
        // worker threads
        for _ in 0..workers {
            let r = item_rx2.clone();
            let tx = bytes_tx.clone();
            let tables = tables.clone();
            worker_handles.push(thread::spawn(move || {
                while let Ok(e) = r.recv() {
                    // quantize vector
                    let qv: Vec<u8> = e.vector.iter().enumerate().map(|(i, &x)| tables[i].quantize(x)).collect();
                    let rec = (e.id.clone(), qv);
                    if let Ok(bytes) = bincode::serialize(&rec) {
                        let _ = tx.send(bytes);
                    }
                }
            }));
        }

        // feed collected embeddings into item_tx2
        for emb in collected_embeddings {
            let _ = item_tx2.send(emb);
        }
        drop(item_tx2);

        // wait for workers
        drop(bytes_tx);
        for h in worker_handles { let _ = h.join(); }
        // wait for writer
        if let Some(h) = writer_handle_opt { let _ = h.join(); }

    } else {
        // close item_tx to signal workers to finish
        drop(item_tx);
        // wait for workers
        drop(bytes_tx);
        for h in worker_handles { let _ = h.join(); }
        // wait for writer
        if let Some(h) = writer_handle_opt { let _ = h.join(); }
    }
    if quantize {
    // If quantized, show a short summary including table count (attempt to read tables from file)
    // variable intentionally unused; underscore prefix to silence warnings
    let _table_count = 0usize;
        if let Ok(mut f) = std::fs::File::open(output) {
            use std::io::Read;
            let mut hdr = vec![0u8; 16];
            let _ = f.read(&mut hdr);
            // crude: read table_count at offset header.len()
            // header 'VECTRO+QSTREAM1\n' length is 14
            if hdr.len() >= 16 {
                // no-op; we will just display quantized
            }
        }
        pb.finish_with_message(format!("wrote {} quantized entries to {}", parsed, output));
    } else {
        pb.finish_with_message(format!("wrote {} entries to {}", parsed, output));
    }
    Ok(parsed)
}

// ---------------------------------------------------------------------------
// Helpers and compress functions ported from vectro-plus v2.1.0
// Adapted to vectro_lib v4.0.0 API (Nf4Vector, train_pq_codebook, pq_encode).
// ---------------------------------------------------------------------------

/// Parse a JSONL or CSV file into `Embedding` objects.
///
/// Accepts two line formats:
/// - JSONL: `{"id": "...", "vector": [f32, ...]}`
/// - CSV:   `id,f32,f32,...`
fn read_jsonl(input: &str) -> anyhow::Result<Vec<vectro_lib::Embedding>> {
    use std::io::{BufRead, BufReader};
    let infile = std::fs::File::open(input)
        .map_err(|e| anyhow::anyhow!("cannot open {input}: {e}"))?;
    let reader = BufReader::new(infile);
    let mut embeddings = Vec::new();

    for line in reader.lines().map_while(Result::ok) {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let mut added = false;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
            if let (Some(id), Some(vec_arr)) = (val.get("id"), val.get("vector")) {
                if let (Some(id_str), Some(arr)) = (id.as_str(), vec_arr.as_array()) {
                    let v: Vec<f32> = arr
                        .iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect();
                    embeddings.push(vectro_lib::Embedding::new(id_str, v));
                    added = true;
                }
            }
        }
        if !added {
            let parts: Vec<&str> = line.splitn(2, ',').collect();
            if parts.len() == 2 {
                let id = parts[0].to_string();
                let v: Vec<f32> = parts[1]
                    .split(',')
                    .filter_map(|p| p.trim().parse().ok())
                    .collect();
                if !v.is_empty() {
                    embeddings.push(vectro_lib::Embedding::new(id, v));
                }
            }
        }
    }
    Ok(embeddings)
}

/// Compress embeddings from `input` JSONL into `output` using NF4 quantization.
///
/// Output format: `VECTRO+NF4STREAM1\n` header, followed by a 4-byte LE `dim`,
/// a 4-byte LE `count`, then for each embedding a 4-byte LE record length and a
/// `bincode`-serialised `(id: String, packed: Vec<u8>, scale: f32, dim: u32)`.
pub fn compress_nf4(input: &str, output: &str) -> anyhow::Result<usize> {
    use std::io::Write;
    const HEADER: &[u8] = b"VECTRO+NF4STREAM1\n";

    let embeddings = read_jsonl(input)?;
    let count = embeddings.len();
    if count == 0 {
        return Ok(0);
    }

    let dim = embeddings[0].vector.len() as u32;

    let pb = ProgressBar::new(count as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner} [{bar:40}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_message("encoding NF4…");

    let f = std::fs::File::create(output)
        .map_err(|e| anyhow::anyhow!("cannot create {output}: {e}"))?;
    let mut w = std::io::BufWriter::new(f);

    w.write_all(HEADER)?;
    w.write_all(&dim.to_le_bytes())?;
    w.write_all(&(count as u32).to_le_bytes())?;

    for emb in &embeddings {
        let nv = vectro_lib::quant::nf4::Nf4Vector::encode_fast(&emb.vector);
        let rec: (&str, &[u8], f32, u32) = (&emb.id, &nv.packed, nv.scale, nv.dim as u32);
        let bytes = bincode::serialize(&rec)?;
        w.write_all(&(bytes.len() as u32).to_le_bytes())?;
        w.write_all(&bytes)?;
        pb.inc(1);
    }
    w.flush()?;

    pb.finish_with_message(format!("wrote {count} NF4-encoded entries to {output}"));
    Ok(count)
}

/// Compress embeddings from `input` JSONL into `output` using Product Quantization.
///
/// `m` = number of subspaces, `k` = number of centroids per subspace.
/// Output format: `VECTRO+PQSTREAM1\n` header, followed by a 4-byte LE codebook
/// blob length, the `bincode`-serialised `PQCodebook`, then for each embedding a
/// 4-byte LE record length and a `bincode`-serialised `(id: String, code: Vec<u8>)`.
pub fn compress_pq(input: &str, output: &str, m: usize, k: usize) -> anyhow::Result<usize> {
    use std::io::Write;
    const HEADER: &[u8] = b"VECTRO+PQSTREAM1\n";

    let embeddings = read_jsonl(input)?;
    let count = embeddings.len();
    if count == 0 {
        return Ok(0);
    }

    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb.set_message(format!("training PQ codebook on {count} vectors (m={m}, k={k})…"));

    let vecs: Vec<Vec<f32>> = embeddings.iter().map(|e| e.vector.clone()).collect();
    let codebook = vectro_lib::quant::pq::train_pq_codebook(&vecs, m, k, 25, 42)
        .map_err(|e| anyhow::anyhow!("PQ training failed: {e}"))?;

    pb.set_message("encoding and writing…");

    let codes = vectro_lib::quant::pq::pq_encode(&vecs, &codebook);
    let cb_blob = bincode::serialize(&codebook)?;

    let f = std::fs::File::create(output)
        .map_err(|e| anyhow::anyhow!("cannot create {output}: {e}"))?;
    let mut w = std::io::BufWriter::new(f);

    w.write_all(HEADER)?;
    w.write_all(&(cb_blob.len() as u32).to_le_bytes())?;
    w.write_all(&cb_blob)?;

    for (emb, code) in embeddings.iter().zip(codes.iter()) {
        let rec = (&emb.id, code.as_slice());
        let bytes = bincode::serialize(&rec)?;
        w.write_all(&(bytes.len() as u32).to_le_bytes())?;
        w.write_all(&bytes)?;
    }
    w.flush()?;

    pb.finish_with_message(format!(
        "wrote {count} PQ-encoded entries to {output} (m={m}, k={k})"
    ));
    Ok(count)
}

/// Compress `input` (JSONL) using Residual Quantization and write `VECTRO+RQSTREAM1`
/// to `output`.
///
/// * `n_passes` — residual passes (L ≥ 1; 2–4 recommended)
/// * `m` — PQ sub-spaces per pass (must divide embedding dimension)
/// * `k` — centroids per sub-space (≤ 256)
pub fn compress_rq(
    input: &str,
    output: &str,
    n_passes: usize,
    m: usize,
    k: usize,
) -> anyhow::Result<usize> {
    const HEADER: &[u8] = b"VECTRO+RQSTREAM1\n";

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}")
            .unwrap()
            .tick_strings(&["\u{2014}", "\\", "|", "/"]),
    );
    pb.set_message("loading embeddings for RQ training…");
    pb.enable_steady_tick(std::time::Duration::from_millis(80));

    // Read all embeddings
    let infile = std::fs::File::open(input)?;
    let reader = std::io::BufReader::new(infile);
    let mut embeddings: Vec<vectro_lib::Embedding> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let e: vectro_lib::Embedding = serde_json::from_str(&line)?;
        embeddings.push(e);
    }

    pb.set_message(format!(
        "training RQ codebook ({} vecs, n_passes={n_passes}, m={m}, k={k})…",
        embeddings.len()
    ));

    let vecs: Vec<Vec<f32>> = embeddings.iter().map(|e| e.vector.clone()).collect();
    // Train on at most 10 000 vectors for speed
    let train_vecs: Vec<Vec<f32>> = vecs.iter().take(10_000).cloned().collect();
    let codebook = vectro_lib::quant::rq::train_rq_codebook(
        &train_vecs, n_passes, m, k, 25, 42
    ).map_err(|e| anyhow::anyhow!(e))?;

    pb.set_message("encoding…");
    let flat_codes = vectro_lib::quant::rq::rq_encode_flat(&codebook, &vecs);
    let cb_blob = bincode::serialize(&codebook)?;

    let mut outfile = std::io::BufWriter::new(std::fs::File::create(output)?);
    outfile.write_all(HEADER)?;
    outfile.write_all(&(cb_blob.len() as u32).to_le_bytes())?;
    outfile.write_all(&cb_blob)?;

    let mut count = 0usize;
    for (emb, flat) in embeddings.iter().zip(flat_codes.iter()) {
        let rec = (&emb.id, flat.as_slice());
        let bytes = bincode::serialize(&rec)?;
        outfile.write_all(&(bytes.len() as u32).to_le_bytes())?;
        outfile.write_all(&bytes)?;
        count += 1;
    }

    pb.finish_with_message(format!(
        "wrote {count} RQ-encoded entries to {output} (n_passes={n_passes}, m={m}, k={k})"
    ));
    Ok(count)
}

/// Auto-select the best quantization format based on accuracy and compression goals
/// and compress `input` to `output`.
///
/// Delegates to [`vectro_lib::auto_select_format`] to pick the format, then
/// calls the corresponding `compress_*` function.
pub fn compress_auto(
    input: &str,
    output: &str,
    target_cosine: f32,
    target_compression: f32,
) -> anyhow::Result<usize> {
    match vectro_lib::auto_select_format(target_cosine, target_compression) {
        "int8" => compress_stream(input, output, true),
        "nf4"  => compress_nf4(input, output),
        "pq"   => compress_pq(input, output, 16, 64),
        _      => compress_rq(input, output, 2, 16, 64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn compress_small_file() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, r#"{"id":"one","vector":[1.0,0.0]}
{"id":"two","vector":[0.0,1.0]}"#).unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, false).expect("compress");
        assert_eq!(n, 2);

        let ds = vectro_lib::EmbeddingDataset::load(&out_path).expect("load");
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn compress_quantized() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, r#"{"id":"one","vector":[1.0,2.0,3.0]}
{"id":"two","vector":[4.0,5.0,6.0]}
{"id":"three","vector":[7.0,8.0,9.0]}"#).unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, true).expect("compress quantized");
        assert_eq!(n, 3);

        let ds = vectro_lib::EmbeddingDataset::load(&out_path).expect("load");
        assert_eq!(ds.len(), 3);
        // Quantized embeddings may not preserve order
        let ids: Vec<&str> = ds.embeddings.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"one"));
        assert!(ids.contains(&"two"));
        assert!(ids.contains(&"three"));
    }

    #[test]
    fn compress_csv_format() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, "id1,1.0,2.0,3.0\nid2,4.0,5.0,6.0\n").unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, false).expect("compress csv");
        assert_eq!(n, 2);

        let ds = vectro_lib::EmbeddingDataset::load(&out_path).expect("load");
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn compress_with_empty_lines() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, r#"
{"id":"one","vector":[1.0,0.0]}

{"id":"two","vector":[0.0,1.0]}

"#).unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, false).expect("compress");
        assert_eq!(n, 2);
    }
}
