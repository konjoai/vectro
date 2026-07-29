use std::io::Write;

#[test]
fn compress_quantized_and_load_roundtrip() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "{{\"id\": \"one\", \"vector\": [1.0, 0.0]}}").unwrap();
    writeln!(f, "{{\"id\": \"two\", \"vector\": [0.0, 1.0]}}").unwrap();
    f.flush().unwrap();

    let out = tempfile::NamedTempFile::new().unwrap();
    let outp = out.path().to_path_buf();

    // produce quantized stream
    let n = vectro_cli::compress_stream(path.to_str().unwrap(), outp.to_str().unwrap(), true)
        .expect("compress");
    assert_eq!(n, 2);

    // load using library (should read QSTREAM and dequantize)
    let ds = vectro_lib::EmbeddingDataset::load(outp.to_str().unwrap()).expect("load dataset");
    assert_eq!(ds.len(), 2);
    // basic checks on ids
    assert!(ds.embeddings.iter().any(|e| e.id == "one"));
    assert!(ds.embeddings.iter().any(|e| e.id == "two"));
}
