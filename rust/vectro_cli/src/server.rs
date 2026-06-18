use axum::{
    extract::{Json, Query, State},
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use vectro_lib::{Embedding, EmbeddingDataset, search::SearchIndex};

// Shared application state
#[derive(Clone)]
pub struct AppState {
    index: Arc<RwLock<Option<SearchIndex>>>,
    embeddings: Arc<RwLock<Vec<Embedding>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    pub fn new() -> Self {
        Self {
            index: Arc::new(RwLock::new(None)),
            embeddings: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

// API request/response types
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    #[serde(default = "default_top_k")]
    pub k: usize,
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub query_time_ms: f64,
}

#[derive(Debug, Deserialize)]
pub struct UploadRequest {
    pub embeddings: Vec<Embedding>,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub count: usize,
    pub dimensions: Option<usize>,
    pub index_loaded: bool,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

// Route handlers
async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn stats(State(state): State<AppState>) -> Json<StatsResponse> {
    let embeddings = state.embeddings.read().await;
    let index = state.index.read().await;
    
    let dimensions = embeddings.first().map(|e| e.vector.len());
    
    Json(StatsResponse {
        count: embeddings.len(),
        dimensions,
        index_loaded: index.is_some(),
    })
}

async fn upload_embeddings(
    State(state): State<AppState>,
    Json(payload): Json<UploadRequest>,
) -> Result<Json<StatsResponse>, (StatusCode, String)> {
    if payload.embeddings.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No embeddings provided".to_string()));
    }
    
    // Validate dimensions are consistent
    let first_dim = payload.embeddings[0].vector.len();
    for emb in &payload.embeddings {
        if emb.vector.len() != first_dim {
            return Err((
                StatusCode::BAD_REQUEST,
                "Inconsistent embedding dimensions".to_string(),
            ));
        }
    }
    
    // Update embeddings
    let mut embeddings = state.embeddings.write().await;
    *embeddings = payload.embeddings;
    
    // Rebuild index
    let new_index = SearchIndex::from_dataset(&embeddings);
    let mut index = state.index.write().await;
    *index = Some(new_index);
    
    let count = embeddings.len();
    drop(embeddings);
    drop(index);
    
    Ok(Json(StatsResponse {
        count,
        dimensions: Some(first_dim),
        index_loaded: true,
    }))
}

async fn search(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let index = state.index.read().await;
    
    if index.is_none() {
        return Err((StatusCode::NOT_FOUND, "No index loaded. Upload embeddings first.".to_string()));
    }
    
    let start = std::time::Instant::now();
    
    let idx = index.as_ref().unwrap();
    let results = idx.top_k(&payload.query, payload.k);
    
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    
    let search_results: Vec<SearchResult> = results
        .into_iter()
        .map(|(id, score)| SearchResult {
            id: id.to_string(),
            score,
        })
        .collect();
    
    Ok(Json(SearchResponse {
        results: search_results,
        query_time_ms: elapsed,
    }))
}

async fn load_dataset_endpoint(
    State(state): State<AppState>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<StatsResponse>, (StatusCode, String)> {
    let path = params.get("path").ok_or((
        StatusCode::BAD_REQUEST,
        "Missing 'path' query parameter".to_string(),
    ))?;
    
    let dataset = EmbeddingDataset::load(path).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to load dataset: {}", e))
    })?;
    
    let embeddings_vec = dataset.embeddings;
    let count = embeddings_vec.len();
    let dimensions = embeddings_vec.first().map(|e| e.vector.len());
    
    // Update state
    let new_index = SearchIndex::from_dataset(&embeddings_vec);
    let mut embeddings = state.embeddings.write().await;
    *embeddings = embeddings_vec;
    
    let mut index = state.index.write().await;
    *index = Some(new_index);
    
    Ok(Json(StatsResponse {
        count,
        dimensions,
        index_loaded: true,
    }))
}

async fn index_page() -> Html<String> {
    Html(include_str!("../static/index.html").to_string())
}

fn build_cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
}

fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index_page))
        .route("/health", get(health))
        .route("/api/stats", get(stats))
        .route("/api/search", post(search))
        .route("/api/upload", post(upload_embeddings))
        .route("/api/load", get(load_dataset_endpoint))
        .layer(build_cors_layer())
        .with_state(state)
}

fn print_server_info(port: u16) {
    println!("🚀 Vectro+ server starting on http://localhost:{}", port);
    println!("📊 Dashboard: http://localhost:{}", port);
    println!("🔍 API endpoints:");
    println!("   GET  /health");
    println!("   GET  /api/stats");
    println!("   POST /api/search");
    println!("   POST /api/upload");
    println!("   GET  /api/load?path=<path>");
}

pub async fn serve(port: u16) -> anyhow::Result<()> {
    let state = AppState::new();
    let app = build_router(state);
    let addr = format!("0.0.0.0:{}", port);
    
    print_server_info(port);
    
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_app_state_new() {
        let state = AppState::new();
        let embeddings = state.embeddings.read().await;
        assert_eq!(embeddings.len(), 0);
    }

    #[tokio::test]
    async fn test_health_handler() {
        let response = health().await;
        assert_eq!(response.0.status, "ok");
        assert!(!response.0.version.is_empty());
    }

    #[tokio::test]
    async fn test_stats_empty_state() {
        let state = AppState::new();
        let response = stats(State(state)).await;
        assert_eq!(response.0.count, 0);
        assert!(response.0.dimensions.is_none());
        assert!(!response.0.index_loaded);
    }

    #[tokio::test]
    async fn test_stats_with_embeddings() {
        let state = AppState::new();
        {
            let mut embeddings = state.embeddings.write().await;
            embeddings.push(Embedding::new("test", vec![1.0, 2.0, 3.0]));
        }
        
        let response = stats(State(state)).await;
        assert_eq!(response.0.count, 1);
        assert_eq!(response.0.dimensions, Some(3));
    }

    #[tokio::test]
    async fn test_upload_empty_embeddings() {
        let state = AppState::new();
        let payload = UploadRequest {
            embeddings: vec![],
        };
        
        let result = upload_embeddings(State(state), Json(payload)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_upload_valid_embeddings() {
        let state = AppState::new();
        let payload = UploadRequest {
            embeddings: vec![
                Embedding::new("a", vec![1.0, 0.0]),
                Embedding::new("b", vec![0.0, 1.0]),
            ],
        };
        
        let result = upload_embeddings(State(state.clone()), Json(payload)).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert_eq!(response.0.count, 2);
        assert_eq!(response.0.dimensions, Some(2));
        assert!(response.0.index_loaded);
    }

    #[tokio::test]
    async fn test_upload_inconsistent_dimensions() {
        let state = AppState::new();
        let payload = UploadRequest {
            embeddings: vec![
                Embedding::new("a", vec![1.0, 0.0]),
                Embedding::new("b", vec![0.0, 1.0, 2.0]), // Different dimension
            ],
        };
        
        let result = upload_embeddings(State(state), Json(payload)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_no_index() {
        let state = AppState::new();
        let payload = SearchRequest {
            query: vec![1.0, 0.0],
            k: 10,
        };
        
        let result = search(State(state), Json(payload)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_with_index() {
        let state = AppState::new();
        
        // Upload embeddings first
        let upload_payload = UploadRequest {
            embeddings: vec![
                Embedding::new("test1", vec![1.0, 0.0]),
                Embedding::new("test2", vec![0.0, 1.0]),
            ],
        };
        let _ = upload_embeddings(State(state.clone()), Json(upload_payload)).await.unwrap();
        
        // Now search
        let search_payload = SearchRequest {
            query: vec![1.0, 0.0],
            k: 1,
        };
        
        let result = search(State(state), Json(search_payload)).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert_eq!(response.0.results.len(), 1);
        assert_eq!(response.0.results[0].id, "test1");
    }

    #[tokio::test]
    async fn test_search_wrong_dimension() {
        let state = AppState::new();
        
        // Upload 2D embeddings
        let upload_payload = UploadRequest {
            embeddings: vec![
                Embedding::new("a", vec![1.0, 0.0]),
            ],
        };
        let _ = upload_embeddings(State(state.clone()), Json(upload_payload)).await.unwrap();
        
        // Search with 3D query - doesn't error, just gives poor results
        let search_payload = SearchRequest {
            query: vec![1.0, 0.0, 0.0],
            k: 1,
        };
        
        let result = search(State(state), Json(search_payload)).await;
        assert!(result.is_ok()); // No dimension validation in search
    }

    #[test]
    fn test_default_top_k() {
        assert_eq!(default_top_k(), 10);
    }

    #[test]
    fn test_search_request_serde() {
        let json = r#"{"query": [1.0, 2.0], "k": 5}"#;
        let req: SearchRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, vec![1.0, 2.0]);
        assert_eq!(req.k, 5);
    }

    #[test]
    fn test_search_request_default_k_serde() {
        let json = r#"{"query": [1.0, 2.0]}"#;
        let req: SearchRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.k, 10);
    }

    #[test]
    fn test_index_page_loads() {
        // Just test that the function exists and returns HTML
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let html = index_page().await;
            assert!(!html.0.is_empty());
        });
    }

    #[tokio::test]
    async fn test_load_dataset_endpoint() {
        use axum::extract::Query;
        use tempfile::NamedTempFile;
        
        let state = AppState::new();
        
        // Create temp dataset file
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        
        let mut ds = EmbeddingDataset::new();
        ds.add(Embedding::new("test1", vec![1.0, 0.0]));
        ds.add(Embedding::new("test2", vec![0.0, 1.0]));
        ds.save(&path).unwrap();
        
        // Load via endpoint
        let mut params = std::collections::HashMap::new();
        params.insert("path".to_string(), path.clone());
        
        let result = load_dataset_endpoint(State(state.clone()), Query(params)).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert_eq!(response.0.count, 2);
        assert_eq!(response.0.dimensions, Some(2));
        assert!(response.0.index_loaded);
        
        // Verify state was updated
        let embeddings = state.embeddings.read().await;
        assert_eq!(embeddings.len(), 2);
        
        let index = state.index.read().await;
        assert!(index.is_some());
    }

    #[tokio::test]
    async fn test_load_dataset_missing_path() {
        use axum::extract::Query;
        
        let state = AppState::new();
        let params = std::collections::HashMap::new();
        
        let result = load_dataset_endpoint(State(state), Query(params)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_load_dataset_invalid_file() {
        use axum::extract::Query;
        
        let state = AppState::new();
        let mut params = std::collections::HashMap::new();
        params.insert("path".to_string(), "/nonexistent/file.bin".to_string());
        
        let result = load_dataset_endpoint(State(state), Query(params)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_cors_configuration() {
        // Test build_cors_layer function
        let _cors = build_cors_layer();
        // If we got here, CORS construction succeeded
        // Test passes by not panicking
    }

    #[test]
    fn test_build_router() {
        // Test that we can build the router
        let state = AppState::new();
        let _router = build_router(state);
        // If we got here, router construction succeeded
    }

    #[test]
    fn test_print_server_info() {
        // Test that print_server_info doesn't panic
        print_server_info(8080);
        print_server_info(3000);
    }
}
