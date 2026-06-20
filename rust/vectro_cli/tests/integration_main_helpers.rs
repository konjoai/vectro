/// Integration tests for main.rs helper functions to improve coverage
use vectro_lib::{Embedding, EmbeddingDataset};

#[test]
fn test_toy_dataset_generation() {
    // Test that the default toy dataset is properly generated
    let embeddings = [Embedding::new("apple", vec![1.0, 0.0, 0.0]),
        Embedding::new("banana", vec![0.9, 0.1, 0.0]),
        Embedding::new("orange", vec![0.8, 0.2, 0.0])];
    
    assert_eq!(embeddings.len(), 3);
    assert_eq!(embeddings[0].id, "apple");
    assert_eq!(embeddings[0].vector.len(), 3);
}

#[test]
fn test_search_with_toy_data() {
    use vectro_lib::search::SearchIndex;
    
    let embeddings = vec![
        Embedding::new("doc1", vec![1.0, 0.0, 0.0]),
        Embedding::new("doc2", vec![0.0, 1.0, 0.0]),
        Embedding::new("doc3", vec![0.0, 0.0, 1.0]),
    ];
    
    let index = SearchIndex::from_dataset(&embeddings);
    let results = index.top_k(&[0.9, 0.1, 0.0], 2);
    
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "doc1"); // Closest match
}

#[test]
fn test_compress_and_search_workflow() {
    use tempfile::NamedTempFile;
    use std::fs;
    
    // Create a test JSONL file
    let input = NamedTempFile::new().unwrap();
    let input_path = input.path().to_str().unwrap();
    fs::write(input_path, r#"{"id":"test1","vector":[1.0,0.0,0.0]}
{"id":"test2","vector":[0.0,1.0,0.0]}
{"id":"test3","vector":[0.0,0.0,1.0]}"#).unwrap();
    
    // Compress it
    let output = NamedTempFile::new().unwrap();
    let output_path = output.path().to_str().unwrap();
    
    let result = vectro_cli::compress_stream(input_path, output_path, false);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 3);
    
    // Load and search
    let ds = EmbeddingDataset::load(output_path).unwrap();
    assert_eq!(ds.len(), 3);
    
    let index = vectro_lib::search::SearchIndex::from_dataset(&ds.embeddings);
    let results = index.top_k(&[1.0, 0.0, 0.0], 1);
    assert_eq!(results[0].0, "test1");
}

#[test]
fn test_quantized_compress_and_search_workflow() {
    use tempfile::NamedTempFile;
    use std::fs;
    
    // Create a test JSONL file with more data for quantization
    let input = NamedTempFile::new().unwrap();
    let input_path = input.path().to_str().unwrap();
    let mut data = String::new();
    for i in 0..10 {
        let val = i as f32 / 10.0;
        data.push_str(&format!(r#"{{"id":"test{}","vector":[{},{},{}]}}"#, i, val, 1.0-val, val*0.5));
        data.push('\n');
    }
    fs::write(input_path, data).unwrap();
    
    // Compress with quantization
    let output = NamedTempFile::new().unwrap();
    let output_path = output.path().to_str().unwrap();
    
    let result = vectro_cli::compress_stream(input_path, output_path, true);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 10);
    
    // Load and verify quantized data
    let ds = EmbeddingDataset::load(output_path).unwrap();
    assert_eq!(ds.len(), 10);
}

#[test]
fn test_empty_jsonl_handling() {
    use tempfile::NamedTempFile;
    use std::fs;
    
    let input = NamedTempFile::new().unwrap();
    let input_path = input.path().to_str().unwrap();
    fs::write(input_path, "").unwrap();
    
    let output = NamedTempFile::new().unwrap();
    let output_path = output.path().to_str().unwrap();
    
    let result = vectro_cli::compress_stream(input_path, output_path, false);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

#[test]
fn test_malformed_json_lines() {
    use tempfile::NamedTempFile;
    use std::fs;
    
    let input = NamedTempFile::new().unwrap();
    let input_path = input.path().to_str().unwrap();
    fs::write(input_path, r#"{"id":"test1","vector":[1.0,0.0,0.0]}
invalid json line
{"id":"test2","vector":[0.0,1.0,0.0]}"#).unwrap();
    
    let output = NamedTempFile::new().unwrap();
    let output_path = output.path().to_str().unwrap();
    
    // Should handle malformed lines gracefully
    let result = vectro_cli::compress_stream(input_path, output_path, false);
    assert!(result.is_ok());
    // Should only process the 2 valid lines
    assert_eq!(result.unwrap(), 2);
}
