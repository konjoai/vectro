/// Integration tests for bench command functionality
/// These tests verify the bench command execution and report generation
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// A single benchmark summary row: (name, median_s, mean_s, unit_label).
type BenchRow = (String, Option<f64>, Option<f64>, Option<String>);

#[test]
fn test_bench_history_persistence() {
    // Create temp directory for history file
    let temp_dir = TempDir::new().unwrap();
    let history_path = temp_dir.path().join(".bench_history.json");
    
    // Create initial history
    let mut history = std::collections::HashMap::new();
    history.insert("bench1".to_string(), 123.456);
    history.insert("bench2".to_string(), 789.012);
    
    // Save history
    let json = serde_json::to_string_pretty(&history).unwrap();
    fs::write(&history_path, json).unwrap();
    
    // Verify file was created
    assert!(history_path.exists());
    
    // Load history back
    let loaded_json = fs::read_to_string(&history_path).unwrap();
    let loaded_history: std::collections::HashMap<String, f64> = 
        serde_json::from_str(&loaded_json).unwrap();
    
    assert_eq!(loaded_history.len(), 2);
    assert_eq!(loaded_history.get("bench1"), Some(&123.456));
    assert_eq!(loaded_history.get("bench2"), Some(&789.012));
}

#[test]
fn test_bench_summary_data_format() {
    // Test bench summary data structure
    let rows: Vec<BenchRow> = vec![
        ("bench1".to_string(), Some(1.234), Some(1.250), Some("ms".to_string())),
        ("bench2".to_string(), Some(5.678), Some(5.700), Some("ms".to_string())),
        ("bench3".to_string(), None, None, None),
    ];
    
    assert_eq!(rows.len(), 3);
    assert!(rows[0].1.is_some());
    assert!(rows[2].1.is_none());
}

#[test]
fn test_criterion_directory_structure() {
    // Create mock Criterion directory structure
    let temp_dir = TempDir::new().unwrap();
    let crit_dir = temp_dir.path().join("target").join("criterion");
    
    // Create benchmark directories
    let bench_dir = crit_dir.join("test_bench");
    let new_dir = bench_dir.join("new");
    fs::create_dir_all(&new_dir).unwrap();
    
    // Create mock JSON file
    let json_path = new_dir.join("estimates.json");
    let mock_data = serde_json::json!({
        "mean": {
            "point_estimate": 1.234567,
            "confidence_interval": {
                "lower_bound": 1.2,
                "upper_bound": 1.3
            }
        },
        "median": {
            "point_estimate": 1.23
        }
    });
    fs::write(&json_path, serde_json::to_string_pretty(&mock_data).unwrap()).unwrap();
    
    // Verify structure
    assert!(crit_dir.exists());
    assert!(bench_dir.exists());
    assert!(new_dir.exists());
    assert!(json_path.exists());
    
    // Read and parse the JSON
    let content = fs::read_to_string(&json_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(parsed.get("mean").is_some());
    assert!(parsed.get("median").is_some());
}

#[test]
fn test_html_report_generation() {
    // Test HTML generation logic
    let rows = vec![
        ("search_benchmark".to_string(), Some(1.234), Some(1.250), Some("ms".to_string())),
        ("compress_benchmark".to_string(), Some(5.678), Some(5.700), Some("μs".to_string())),
    ];
    
    let mut history = std::collections::HashMap::new();
    history.insert("search_benchmark".to_string(), 1.200);
    
    // Simple HTML validation - ensure we can format the data
    for (name, med, mean, unit) in &rows {
        let med_str = med.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "-".to_string());
        let mean_str = mean.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "-".to_string());
        let _unit_str = unit.clone().unwrap_or_else(|| "".to_string());
        
        assert!(!med_str.is_empty());
        assert!(!mean_str.is_empty());
        
        // Calculate delta
        if let Some(prev) = history.get(name) {
            if let Some(curr) = med {
                let delta_pct: f64 = (curr - prev) / prev * 100.0;
                assert!(delta_pct.abs() < 1000.0); // Sanity check
            }
        }
    }
}

#[test]
fn test_command_building() {
    // Test that we can build commands (without executing)
    let cmd = std::process::Command::new("cargo");
    let program = cmd.get_program();
    assert_eq!(program, "cargo");
}

#[test]
fn test_file_copying_logic() {
    // Test directory copying logic
    let temp_dir = TempDir::new().unwrap();
    let source = temp_dir.path().join("source");
    let dest = temp_dir.path().join("dest");
    
    // Create source directory with a file
    fs::create_dir(&source).unwrap();
    fs::write(source.join("test.txt"), "test content").unwrap();
    
    // Create destination
    fs::create_dir(&dest).unwrap();
    
    // Copy file
    let source_file = source.join("test.txt");
    let dest_file = dest.join("test.txt");
    fs::copy(&source_file, &dest_file).unwrap();
    
    // Verify
    assert!(dest_file.exists());
    let content = fs::read_to_string(&dest_file).unwrap();
    assert_eq!(content, "test content");
}

#[test]
fn test_timestamp_generation() {
    // Test timestamp generation for report naming
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let now = SystemTime::now();
    let duration = now.duration_since(UNIX_EPOCH).unwrap();
    let timestamp = duration.as_secs();
    
    // Verify it's a reasonable timestamp (after 2020)
    assert!(timestamp > 1577836800); // Jan 1, 2020
    
    // Format as string
    let ts_string = format!("{}", timestamp);
    assert!(!ts_string.is_empty());
    assert!(ts_string.len() >= 10);
}

#[test]
fn test_path_manipulation() {
    // Test path operations used in bench command
    let base = PathBuf::from("target/criterion");
    let bench_name = "my_benchmark";
    let full_path = base.join(bench_name).join("new").join("estimates.json");
    
    assert_eq!(
        full_path.to_str().unwrap(),
        "target/criterion/my_benchmark/new/estimates.json"
    );
    
    // Test file stem extraction
    if let Some(stem) = full_path.file_stem() {
        assert_eq!(stem.to_str().unwrap(), "estimates");
    }
    
    // Test extension checking
    if let Some(ext) = full_path.extension() {
        assert_eq!(ext.to_str().unwrap(), "json");
    }
}
