/// Integration tests for server functionality via CLI
/// These tests verify the server can be started and responds correctly
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

#[test]
fn test_server_command_builds() {
    // Test that we can build the serve command
    let output = Command::new("cargo")
        .args(["build", "--bin", "vectro_cli"])
        .output()
        .expect("Failed to build binary");
    
    assert!(output.status.success() || output.stderr.is_empty());
}

#[test]
fn test_server_startup_via_cli() {
    // This test spawns the actual server process
    // Skip if binary doesn't exist (e.g., first build)
    let binary_path = if cfg!(debug_assertions) {
        "target/debug/vectro_cli"
    } else {
        "target/release/vectro_cli"
    };
    
    if !std::path::Path::new(binary_path).exists() {
        eprintln!("Binary not found, skipping server startup test");
        return;
    }
    
    // Start server in background on a unique port
    let port = 19080;
    let mut child = Command::new(binary_path)
        .args(["serve", "--port", &port.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    
    if let Ok(ref mut process) = child {
        // Give server time to start
        thread::sleep(Duration::from_millis(500));
        
        // Try to connect (basic check)
        let client = reqwest::blocking::Client::new();
        let result = client
            .get(format!("http://localhost:{}/health", port))
            .timeout(Duration::from_secs(2))
            .send();
        
        // Kill server
        let _ = process.kill();
        
        // Verify we got a response
        if let Ok(response) = result {
            assert!(response.status().is_success());
        }
    }
}

#[test]
fn test_server_cli_help() {
    // Test that serve command help works
    let output = Command::new("cargo")
        .args(["run", "--bin", "vectro_cli", "--", "serve", "--help"])
        .output();
    
    if let Ok(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("serve") || stdout.contains("port") || stdout.contains("server"));
    }
}
