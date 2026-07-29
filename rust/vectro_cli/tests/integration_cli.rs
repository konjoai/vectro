use tempfile::NamedTempFile;

#[test]
fn test_search_command() {
    // Create a test dataset
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();

    let mut ds = vectro_lib::EmbeddingDataset::new();
    ds.add(vectro_lib::Embedding::new("test1", vec![1.0, 0.0, 0.0]));
    ds.add(vectro_lib::Embedding::new("test2", vec![0.0, 1.0, 0.0]));
    ds.add(vectro_lib::Embedding::new("test3", vec![0.0, 0.0, 1.0]));
    ds.save(&path).expect("save dataset");

    // Test that the file was created and can be loaded
    let loaded = vectro_lib::EmbeddingDataset::load(&path).expect("load dataset");
    assert_eq!(loaded.len(), 3);
}

#[test]
fn test_batch_search() {
    let mut ds = vectro_lib::EmbeddingDataset::new();
    ds.add(vectro_lib::Embedding::new("a", vec![1.0, 0.0]));
    ds.add(vectro_lib::Embedding::new("b", vec![0.0, 1.0]));

    let idx = vectro_lib::search::SearchIndex::from_dataset(&ds.embeddings);
    let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let results = idx.batch_top_k(&queries, 1);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0][0].0, "a");
    assert_eq!(results[1][0].0, "b");
}

#[test]
fn test_quantized_batch_search() {
    let mut ds = vectro_lib::EmbeddingDataset::new();
    ds.add(vectro_lib::Embedding::new("a", vec![1.0, 0.0]));
    ds.add(vectro_lib::Embedding::new("b", vec![0.0, 1.0]));

    let idx = vectro_lib::search::QuantizedIndex::from_dataset(&ds.embeddings);
    let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let results = idx.batch_top_k(&queries, 1);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0][0].0, "a");
    assert_eq!(results[1][0].0, "b");
}

#[test]
fn test_parse_query_vector() {
    // Test parsing comma-separated vectors
    let input = "1.0,2.0,3.0";
    let parts: Vec<&str> = input.split(',').collect();
    let vec: Vec<f32> = parts.iter().filter_map(|s| s.trim().parse().ok()).collect();
    assert_eq!(vec, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_json_parsing_helpers() {
    use serde_json::Value;

    let json = r#"{"results": [{"value": 123}, {"value": 456}]}"#;
    let val: Value = serde_json::from_str(json).unwrap();

    // Test finding nested numbers
    if let Some(arr) = val.get("results").and_then(|v| v.as_array()) {
        assert_eq!(arr.len(), 2);
        if let Some(num) = arr[0].get("value").and_then(|v| v.as_f64()) {
            assert_eq!(num, 123.0);
        }
    }
}
