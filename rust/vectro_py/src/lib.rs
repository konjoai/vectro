// PyO3's `#[pymethods]` / `#[pyclass]` macros expand to `impl` blocks inside
// function bodies, which trips rustc's `non_local_definitions` lint. This is a
// known macro-expansion false positive (fixed by a future PyO3 bump); suppress
// it crate-wide so `cargo clippy -- -D warnings` stays clean.
#![allow(unknown_lints)]
#![allow(non_local_definitions)]

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use ndarray::{Array1, Array2, Array3};
use vectro_lib::{Embedding, EmbeddingDataset};
use vectro_lib::search::{SearchIndex, QuantizedIndex};
use std::collections::HashMap;

/// Python wrapper for Embedding
#[pyclass]
struct PyEmbedding {
    inner: Embedding,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    fn new(id: String, vector: PyReadonlyArray1<f32>) -> Self {
        let vector_vec = vector.as_array().to_vec();
        Self {
            inner: Embedding::new(id, vector_vec),
        }
    }

    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn vector(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f32>>> {
        let array = Array1::from(self.inner.vector.clone());
        Ok(array.into_pyarray(py).to_owned())
    }

    fn __repr__(&self) -> String {
        format!("PyEmbedding(id='{}', dim={})", self.inner.id, self.inner.vector.len())
    }
}

/// Python wrapper for EmbeddingDataset
#[pyclass(name = "EmbeddingDataset")]
struct PyEmbeddingDataset {
    inner: EmbeddingDataset,
}

#[pymethods]
impl PyEmbeddingDataset {
    #[new]
    fn new() -> Self {
        Self {
            inner: EmbeddingDataset::new(),
        }
    }

    fn add_embedding(&mut self, embedding: &PyEmbedding) {
        self.inner.add(embedding.inner.clone());
    }

    fn add_vector(&mut self, id: String, vector: PyReadonlyArray1<f32>) {
        let vector_vec = vector.as_array().to_vec();
        let embedding = Embedding::new(id, vector_vec);
        self.inner.add(embedding);
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn get_embedding(&self, index: usize) -> Option<PyEmbedding> {
        self.inner.embeddings.get(index).map(|e| PyEmbedding { inner: e.clone() })
    }

    fn get_vectors(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f32>>> {
        if self.inner.is_empty() {
            return Ok(Array2::zeros((0, 0)).into_pyarray(py).to_owned());
        }
        
        let dim = self.inner.embeddings[0].vector.len();
        let mut array = Array2::zeros((self.inner.len(), dim));
        
        for (i, embedding) in self.inner.embeddings.iter().enumerate() {
            for (j, &value) in embedding.vector.iter().enumerate() {
                array[[i, j]] = value;
            }
        }
        
        Ok(array.into_pyarray(py).to_owned())
    }

    fn get_ids(&self) -> Vec<String> {
        self.inner.embeddings.iter().map(|e| e.id.clone()).collect()
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __repr__(&self) -> String {
        format!("PyEmbeddingDataset(size={})", self.inner.len())
    }

    /// Construct an empty EmbeddingDataset (alias for `new()`).
    #[staticmethod]
    fn empty() -> Self {
        Self { inner: EmbeddingDataset::new() }
    }

    /// Build an EmbeddingDataset from parallel ids and vector lists.
    #[staticmethod]
    fn from_embeddings(
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
    ) -> PyResult<Self> {
        if ids.len() != vectors.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ids length ({}) != vectors length ({})",
                ids.len(),
                vectors.len()
            )));
        }
        let mut ds = EmbeddingDataset::new();
        for (id, vec) in ids.into_iter().zip(vectors.into_iter()) {
            ds.add(Embedding::new(id, vec));
        }
        Ok(Self { inner: ds })
    }

    /// Load an EmbeddingDataset from a .stream1 file on disk.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        EmbeddingDataset::load(path)
            .map(|inner| Self { inner })
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }
}

/// Python wrapper for SearchIndex
#[pyclass]
struct PySearchIndex {
    inner: SearchIndex,
    id_to_index: HashMap<String, usize>,
}

#[pymethods]
impl PySearchIndex {
    #[staticmethod]
    fn from_dataset(dataset: &PyEmbeddingDataset) -> PyResult<Self> {
        let index = SearchIndex::from_dataset(&dataset.inner.embeddings);
        
        // Build ID->index mapping
        let mut id_to_index = HashMap::new();
        for (idx, embedding) in dataset.inner.embeddings.iter().enumerate() {
            id_to_index.insert(embedding.id.clone(), idx);
        }
        
        Ok(Self { inner: index, id_to_index })
    }

    fn search_vector(&self, py: Python<'_>, query: PyReadonlyArray1<f32>, top_k: usize) -> PyResult<Py<PyTuple>> {
        let query_vec = query.as_array().to_vec();
        let results = self.inner.top_k(&query_vec, top_k);
        
        let mut indices = Vec::new();
        let mut similarities = Vec::new();
        
        // The results are (id, similarity) pairs, we need to convert to indices
        for (id, similarity) in results {
            // Find the index of this ID in the original dataset
            // Note: This is inefficient but works for the demo
            // In production, we'd want to store ID->index mapping
            if let Some(index) = self.find_id_index(id) {
                indices.push(index);
                similarities.push(similarity);
            }
        }
        
        let indices_array: &PyArray1<usize> = Array1::from(indices).into_pyarray(py);
        let similarities_array: &PyArray1<f32> = Array1::from(similarities).into_pyarray(py);
        
        Ok(PyTuple::new(py, [indices_array.as_ref(), similarities_array.as_ref()]).into())
    }

    fn batch_search(&self, py: Python<'_>, queries: PyReadonlyArray2<f32>, top_k: usize) -> PyResult<Py<PyList>> {
        let queries_array = queries.as_array();
        let mut all_results = Vec::new();
        
        for query_row in queries_array.outer_iter() {
            let query_vec = query_row.to_vec();
            let results = self.inner.top_k(&query_vec, top_k);
            
            let mut indices = Vec::new();
            let mut similarities = Vec::new();
            
            for (id, similarity) in results {
                if let Some(index) = self.find_id_index(id) {
                    indices.push(index);
                    similarities.push(similarity);
                }
            }
            
            let indices_array: &PyArray1<usize> = Array1::from(indices).into_pyarray(py);
            let similarities_array: &PyArray1<f32> = Array1::from(similarities).into_pyarray(py);
            let result_tuple = PyTuple::new(py, [indices_array.as_ref(), similarities_array.as_ref()]);
            
            all_results.push(result_tuple);
        }
        
        Ok(PyList::new(py, all_results).into())
    }

    fn __repr__(&self) -> String {
        // We can't access private fields, so use a simpler representation
        "PySearchIndex".to_string()
    }
}

impl PySearchIndex {
    fn find_id_index(&self, target_id: &str) -> Option<usize> {
        self.id_to_index.get(target_id).copied()
    }
}

/// Python wrapper for QuantizedIndex
#[pyclass]
struct PyQuantizedIndex {
    inner: QuantizedIndex,
    id_to_index: HashMap<String, usize>,
}

#[pymethods]
impl PyQuantizedIndex {
    #[staticmethod]
    fn from_dataset(dataset: &PyEmbeddingDataset) -> PyResult<Self> {
        let index = QuantizedIndex::from_dataset(&dataset.inner.embeddings);
        
        // Build ID->index mapping
        let mut id_to_index = HashMap::new();
        for (idx, embedding) in dataset.inner.embeddings.iter().enumerate() {
            id_to_index.insert(embedding.id.clone(), idx);
        }
        
        Ok(Self { inner: index, id_to_index })
    }

    fn search_vector(&self, py: Python<'_>, query: PyReadonlyArray1<f32>, top_k: usize) -> PyResult<Py<PyTuple>> {
        let query_vec = query.as_array().to_vec();
        let results = self.inner.top_k(&query_vec, top_k);
        
        let mut indices = Vec::new();
        let mut similarities = Vec::new();
        
        for (id, similarity) in results {
            if let Some(index) = self.find_id_index(id) {
                indices.push(index);
                similarities.push(similarity);
            }
        }
        
        let indices_array: &PyArray1<usize> = Array1::from(indices).into_pyarray(py);
        let similarities_array: &PyArray1<f32> = Array1::from(similarities).into_pyarray(py);
        
        Ok(PyTuple::new(py, [indices_array.as_ref(), similarities_array.as_ref()]).into())
    }

    fn compression_ratio(&self) -> f32 {
        // Estimate compression ratio: f32 (4 bytes) vs u8 (1 byte) per dimension
        // Plus some overhead for quantization tables
        4.0 // Simplified estimate
    }

    fn memory_usage_bytes(&self) -> usize {
        // Simplified calculation since we can't access private fields
        // This would need proper getter methods on QuantizedIndex
        1024 // Placeholder
    }

    fn __repr__(&self) -> String {
        format!("PyQuantizedIndex(ratio={:.2}x)", self.compression_ratio())
    }
}

impl PyQuantizedIndex {
    fn find_id_index(&self, target_id: &str) -> Option<usize> {
        self.id_to_index.get(target_id).copied()
    }
}

/// Compression utilities
#[pyfunction]
fn compress_embeddings(py: Python<'_>, vectors: PyReadonlyArray2<f32>, ids: Option<Vec<String>>) -> PyResult<Py<PyTuple>> {
    let vectors_array = vectors.as_array();
    let mut dataset = EmbeddingDataset::new();
    
    for (i, vector_row) in vectors_array.outer_iter().enumerate() {
        let id = ids.as_ref().and_then(|ids| ids.get(i).cloned())
                   .unwrap_or_else(|| format!("vec_{}", i));
        let vector_vec = vector_row.to_vec();
        dataset.add(Embedding::new(id, vector_vec));
    }
    
    // Create both regular and quantized indices
    let search_index = SearchIndex::from_dataset(&dataset.embeddings);
    let quantized_index = QuantizedIndex::from_dataset(&dataset.embeddings);
    
    // Build ID->index mapping
    let mut id_to_index = HashMap::new();
    for (idx, embedding) in dataset.embeddings.iter().enumerate() {
        id_to_index.insert(embedding.id.clone(), idx);
    }
    
    let py_search_index = PySearchIndex { 
        inner: search_index, 
        id_to_index: id_to_index.clone()
    };
    let py_quantized_index = PyQuantizedIndex { 
        inner: quantized_index, 
        id_to_index
    };
    
    Ok(PyTuple::new(py, &[
        py_search_index.into_py(py),
        py_quantized_index.into_py(py)
    ]).into())
}

/// Quality analysis utilities
#[pyfunction]
fn analyze_compression_quality(
    original: PyReadonlyArray2<f32>,
    compressed_index: &PyQuantizedIndex,
    num_samples: Option<usize>
) -> PyResult<HashMap<String, f32>> {
    let samples = num_samples.unwrap_or(100);
    let original_array = original.as_array();
    let mut total_similarity = 0.0f32;
    let mut max_similarity = 0.0f32;
    let mut min_similarity = 1.0f32;
    
    let actual_samples = samples.min(original_array.nrows());
    
    for i in 0..actual_samples {
        let query = original_array.row(i).to_vec();
        let results = compressed_index.inner.top_k(&query, 1);
        
        if let Some((_, similarity)) = results.first() {
            total_similarity += similarity;
            max_similarity = max_similarity.max(*similarity);
            min_similarity = min_similarity.min(*similarity);
        }
    }
    
    let avg_similarity = total_similarity / actual_samples as f32;
    let compression_ratio = compressed_index.compression_ratio();
    
    let mut analysis = HashMap::new();
    analysis.insert("average_similarity".to_string(), avg_similarity);
    analysis.insert("max_similarity".to_string(), max_similarity);
    analysis.insert("min_similarity".to_string(), min_similarity);
    analysis.insert("compression_ratio".to_string(), compression_ratio);
    analysis.insert("memory_savings_percent".to_string(), (1.0 - 1.0/compression_ratio) * 100.0);
    analysis.insert("samples_analyzed".to_string(), actual_samples as f32);
    
    Ok(analysis)
}

/// Performance benchmarking utilities
#[pyfunction]
fn benchmark_search_performance(
    index: &PySearchIndex,
    queries: PyReadonlyArray2<f32>,
    top_k: usize,
    num_runs: Option<usize>
) -> PyResult<HashMap<String, f32>> {
    use std::time::Instant;
    
    let runs = num_runs.unwrap_or(10);
    let queries_array = queries.as_array();
    let mut total_time = 0.0;
    let mut successful_queries = 0;
    
    for _ in 0..runs {
        for query_row in queries_array.outer_iter() {
            let start = Instant::now();
            let query_vec = query_row.to_vec();
            let _results = index.inner.top_k(&query_vec, top_k);
            let duration = start.elapsed();
            total_time += duration.as_secs_f32() * 1000.0; // Convert to milliseconds
            successful_queries += 1;
        }
    }
    
    let avg_latency_ms = if successful_queries > 0 {
        total_time / successful_queries as f32
    } else {
        0.0
    };
    
    let queries_per_second = if avg_latency_ms > 0.0 {
        1000.0 / avg_latency_ms
    } else {
        0.0
    };
    
    let mut benchmark = HashMap::new();
    benchmark.insert("average_latency_ms".to_string(), avg_latency_ms);
    benchmark.insert("queries_per_second".to_string(), queries_per_second);
    benchmark.insert("successful_queries".to_string(), successful_queries as f32);
    benchmark.insert("total_runs".to_string(), (runs * queries_array.nrows()) as f32);
    
    Ok(benchmark)
}

// ─────────────────────── Phase-16 algorithm bindings ──────────────────────

use vectro_lib::quant::{int8, nf4, binary, pq, bf16};
use vectro_lib::index::bm25::BM25Index;
use vectro_lib::index::hnsw::HnswIndex;
use vectro_lib::index::ivf::IvfIndex;
use vectro_lib::index::ivf_pq::IvfPqIndex;
use vectro_lib::index::quant_hnsw::{
    Bf16HnswIndex, Int8HnswIndex, Nf4HnswIndex, Sq2HnswIndex, BinaryHnswIndex,
};

/// INT8 symmetric abs-max quantizer (Python binding).
#[pyclass]
struct PyInt8Encoder {
    vectors: Vec<int8::Int8Vector>,
}

#[pymethods]
impl PyInt8Encoder {
    #[new]
    fn new() -> Self {
        Self { vectors: Vec::new() }
    }

    fn encode(&mut self, vectors: Vec<Vec<f32>>) {
        self.vectors = int8::encode_batch(&vectors);
    }

    /// Zero-copy encode from a numpy array (shape [N, D]).
    ///
    /// For C-contiguous arrays, row slices are read directly from the NumPy
    /// buffer without any copy.  Non-contiguous arrays fall back to a
    /// row-by-row copy.
    fn encode_np(&mut self, array: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = array.as_array();
        let (n, d) = (arr.nrows(), arr.ncols());
        self.vectors = match arr.as_slice() {
            Some(flat) => {
                // C-contiguous: each row is a direct slice of the buffer
                (0..n)
                    .map(|i| int8::Int8Vector::encode_fast(&flat[i * d..(i + 1) * d]))
                    .collect()
            }
            None => {
                // Non-contiguous layout: copy each row
                arr.rows()
                    .into_iter()
                    .map(|r| {
                        let v: Vec<f32> = r.iter().copied().collect();
                        int8::Int8Vector::encode_fast(&v)
                    })
                    .collect()
            }
        };
        Ok(())
    }

    fn search(&self, query: Vec<f32>, top_k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, int8::cosine_int8(&query, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    fn __repr__(&self) -> String {
        format!("PyInt8Encoder(n_vectors={})", self.vectors.len())
    }
}

/// NF4 4-bit normal-float quantizer (Python binding).
#[pyclass]
struct PyNf4Encoder {
    vectors: Vec<nf4::Nf4Vector>,
}

#[pymethods]
impl PyNf4Encoder {
    #[new]
    fn new() -> Self {
        Self { vectors: Vec::new() }
    }

    fn encode(&mut self, vectors: Vec<Vec<f32>>) {
        self.vectors = nf4::encode_batch(&vectors);
    }

    fn decode(&self) -> Vec<Vec<f32>> {
        nf4::decode_batch(&self.vectors)
    }

    fn __repr__(&self) -> String {
        format!("PyNf4Encoder(n_vectors={})", self.vectors.len())
    }
}

/// Binary 1-bit sign quantizer with Hamming search (Python binding).
#[pyclass]
struct PyBinaryEncoder {
    vectors: Vec<binary::BinaryVector>,
}

#[pymethods]
impl PyBinaryEncoder {
    #[new]
    fn new() -> Self {
        Self { vectors: Vec::new() }
    }

    fn encode(&mut self, vectors: Vec<Vec<f32>>) {
        self.vectors = binary::encode_batch(&vectors, true);
    }

    fn search(&self, query: Vec<f32>, top_k: usize) -> Vec<(usize, u32)> {
        binary::binary_search(&query, &self.vectors, top_k, true)
    }

    fn __repr__(&self) -> String {
        format!("PyBinaryEncoder(n_vectors={})", self.vectors.len())
    }
}

/// Product Quantization codebook + ADC search (Python binding).
#[pyclass]
struct PyPQCodebook {
    codebook: Option<pq::PQCodebook>,
    codes: Vec<Vec<u8>>,
}

#[pymethods]
impl PyPQCodebook {
    #[new]
    fn new() -> Self {
        Self { codebook: None, codes: Vec::new() }
    }

    fn train(
        &mut self,
        training_data: Vec<Vec<f32>>,
        n_subspaces: usize,
        n_centroids: usize,
        max_iter: usize,
        seed: u64,
    ) -> PyResult<()> {
        let cb = pq::train_pq_codebook(&training_data, n_subspaces, n_centroids, max_iter, seed)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.codebook = Some(cb);
        Ok(())
    }

    fn encode(&mut self, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        let cb = self
            .codebook
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("train() must be called first"))?;
        self.codes = pq::pq_encode(&vectors, cb);
        Ok(())
    }

    fn search(&self, query: Vec<f32>, top_k: usize) -> PyResult<Vec<(usize, f32)>> {
        let cb = self
            .codebook
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("train() must be called first"))?;
        Ok(pq::pq_search(&query, &self.codes, cb, top_k))
    }

    fn __repr__(&self) -> String {
        match &self.codebook {
            None => "PyPQCodebook(untrained)".to_string(),
            Some(cb) => format!(
                "PyPQCodebook(M={}, K={}, sub_dim={}, n_encoded={})",
                cb.n_subspaces, cb.n_centroids, cb.sub_dim, self.codes.len()
            ),
        }
    }
}

/// HNSW approximate nearest-neighbour index (Python binding).
#[pyclass]
struct PyHnswIndex {
    inner: HnswIndex,
}

#[pymethods]
impl PyHnswIndex {
    /// `metric`: `"cosine"` (default), `"l2"`/`"euclidean"`, or `"ip"`/`"inner_product"`.
    #[new]
    #[pyo3(signature = (m, ef_construction, metric = "cosine"))]
    fn new(m: usize, ef_construction: usize, metric: &str) -> PyResult<Self> {
        let metric = match metric.to_ascii_lowercase().as_str() {
            "cosine" | "angular" => vectro_lib::index::hnsw::Metric::Cosine,
            "l2" | "euclidean" => vectro_lib::index::hnsw::Metric::L2,
            "ip" | "inner_product" | "dot" => vectro_lib::index::hnsw::Metric::InnerProduct,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown metric '{other}' (expected cosine, l2/euclidean, or ip)"
                )))
            }
        };
        Ok(Self { inner: HnswIndex::with_metric(m, ef_construction, metric) })
    }

    fn add(&mut self, vector: Vec<f32>) {
        self.inner.add(&vector);
    }

    fn add_batch(&mut self, vectors: Vec<Vec<f32>>) {
        self.inner.add_batch(&vectors);
    }

    /// Batch insert from a numpy array (shape [N, D]).
    ///
    /// Routed through `add_batch` (not per-row `add`) so a large first batch
    /// uses the parallel graph build.
    fn add_np(&mut self, array: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = array.as_array();
        let (n, d) = (arr.nrows(), arr.ncols());
        let rows: Vec<Vec<f32>> = if let Some(flat) = arr.as_slice() {
            (0..n).map(|i| flat[i * d..(i + 1) * d].to_vec()).collect()
        } else {
            arr.rows().into_iter().map(|row| row.iter().copied().collect()).collect()
        };
        self.inner.add_batch(&rows);
        Ok(())
    }

    /// Zero-copy nearest-neighbour search from a 1-D numpy query vector.
    ///
    /// Copies the (tiny) query out and releases the GIL during the search, so a
    /// Python threadpool issuing concurrent queries scales across cores instead
    /// of serializing on the interpreter lock.
    fn search_np(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        let owned: Vec<f32> = query.as_array().iter().copied().collect();
        py.allow_threads(|| self.inner.search(&owned, k, ef))
    }

    fn search(&self, query: Vec<f32>, k: usize, ef: usize) -> Vec<(usize, f32)> {
        self.inner.search(&query, k, ef)
    }

    /// Single-query search returning two numpy arrays directly — node IDs
    /// (`int64`) and distances (`float32`) — with the GIL released during the
    /// search. Avoids the Python list-of-tuples allocation that bottlenecks the
    /// per-query hot path, cutting single-query latency well below the
    /// `search_np` → list → `np.array(...)` round-trip.
    fn search_arrays_np<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef: usize,
    ) -> (&'py PyArray1<i64>, &'py PyArray1<f32>) {
        // Copy the (tiny) query out before releasing the GIL so the pure-Rust
        // search holds no Python borrow.
        let owned: Vec<f32> = query.as_array().iter().copied().collect();
        let res = py.allow_threads(|| self.inner.search(&owned, k, ef));
        let ids: Vec<i64> = res.iter().map(|&(id, _)| id as i64).collect();
        let dists: Vec<f32> = res.iter().map(|&(_, d)| d).collect();
        (Array1::from(ids).into_pyarray(py), Array1::from(dists).into_pyarray(py))
    }

    /// Search with an allow-list of node IDs.
    ///
    /// Only nodes whose ID is in `allowed_ids` are eligible for the result set.
    fn search_filtered_np(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef: usize,
        allowed_ids: Vec<usize>,
    ) -> Vec<(usize, f32)> {
        use std::collections::HashSet;
        let allowed: HashSet<usize> = allowed_ids.into_iter().collect();
        let q = query.as_array();
        match q.as_slice() {
            Some(s) => self.inner.search_filtered(s, k, ef, |id| allowed.contains(&id)),
            None => {
                let v: Vec<f32> = q.iter().copied().collect();
                self.inner.search_filtered(&v, k, ef, |id| allowed.contains(&id))
            }
        }
    }

    /// Batch search: queries shape [Q, D], returns list of lists of (id, dist).
    ///
    /// Parallelised across queries (rayon, in the Rust core) — the GIL is
    /// released so other Python threads run while the search fans out.
    fn search_batch_np(
        &self,
        py: Python<'_>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        let arr = queries.as_array();
        let d = arr.ncols();
        if let Some(flat) = arr.as_slice() {
            py.allow_threads(|| self.inner.search_batch_flat(flat, d, k, ef))
        } else {
            // Non-contiguous fallback: materialise then search in parallel.
            let owned: Vec<f32> = arr.iter().copied().collect();
            py.allow_threads(|| self.inner.search_batch_flat(&owned, d, k, ef))
        }
    }

    /// Soft-delete a vector by ID.
    fn delete(&mut self, id: usize) {
        self.inner.delete(id);
    }

    /// Compact the index by permanently removing all soft-deleted nodes and
    /// rebuilding the graph.  Returns the number of nodes removed.
    fn vacuum(&mut self) -> usize {
        self.inner.vacuum()
    }

    /// Persist the index to a file (bincode format).
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load an index previously saved with `save()`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = HnswIndex::load(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("PyHnswIndex(n_vectors={})", self.inner.len())
    }
}

/// BFloat16 quantizer (Python binding).
///
/// Stores vectors as BF16 (2 bytes/dim, 2× memory savings vs f32) with
/// SimSIMD-accelerated cosine distance computation.
#[pyclass]
struct PyBf16Encoder {
    vectors: Vec<bf16::Bf16Vector>,
}

#[pymethods]
impl PyBf16Encoder {
    #[new]
    fn new() -> Self {
        Self { vectors: Vec::new() }
    }

    /// Encode a list of f32 vectors to BF16.
    fn encode(&mut self, vectors: Vec<Vec<f32>>) {
        self.vectors = vectors
            .iter()
            .map(|v| bf16::Bf16Vector::encode(v))
            .collect();
    }

    /// Zero-copy encode from a numpy array (shape [N, D]).
    fn encode_np(&mut self, array: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = array.as_array();
        let (n, d) = (arr.nrows(), arr.ncols());
        self.vectors = match arr.as_slice() {
            Some(flat) => (0..n)
                .map(|i| bf16::Bf16Vector::encode(&flat[i * d..(i + 1) * d]))
                .collect(),
            None => arr
                .rows()
                .into_iter()
                .map(|row| {
                    let v: Vec<f32> = row.iter().copied().collect();
                    bf16::Bf16Vector::encode(&v)
                })
                .collect(),
        };
        Ok(())
    }

    /// Decode all stored BF16 vectors back to f32.
    fn decode(&self) -> Vec<Vec<f32>> {
        self.vectors.iter().map(|v| v.decode()).collect()
    }

    /// Cosine distance between two stored vectors (by index).
    fn cosine_dist(&self, i: usize, j: usize) -> PyResult<f32> {
        if i >= self.vectors.len() || j >= self.vectors.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("index out of range: i={i}, j={j}, n={}", self.vectors.len()),
            ));
        }
        Ok(self.vectors[i].cosine_dist(&self.vectors[j]))
    }

    fn __len__(&self) -> usize {
        self.vectors.len()
    }

    fn __repr__(&self) -> String {
        format!("PyBf16Encoder(n_vectors={})", self.vectors.len())
    }
}

/// IVF-Flat approximate nearest-neighbour index (Python binding).
#[pyclass]
struct PyIvfIndex {
    inner: IvfIndex,
}

#[pymethods]
impl PyIvfIndex {
    #[new]
    fn new(n_lists: usize, n_probe: usize) -> Self {
        Self { inner: IvfIndex::new(n_lists, n_probe) }
    }

    /// Train the coarse quantizer from example vectors.
    fn train(&mut self, vectors: Vec<Vec<f32>>, max_iter: usize, seed: u64) -> PyResult<()> {
        self.inner
            .train(&vectors, max_iter, seed)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Zero-copy train from a numpy array (shape [N, D]).
    fn train_np(&mut self, array: PyReadonlyArray2<f32>, max_iter: usize, seed: u64) -> PyResult<()> {
        let arr = array.as_array();
        let (n, d) = (arr.nrows(), arr.ncols());
        // Borrow contiguous rows directly; only copy when the array is strided.
        match arr.as_slice() {
            Some(flat) => {
                let rows: Vec<&[f32]> = (0..n).map(|i| &flat[i * d..(i + 1) * d]).collect();
                self.inner.train(&rows, max_iter, seed)
            }
            None => {
                let owned: Vec<Vec<f32>> =
                    arr.rows().into_iter().map(|r| r.iter().copied().collect()).collect();
                self.inner.train(&owned, max_iter, seed)
            }
        }
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Add a single vector; returns its global id.
    fn add(&mut self, vector: Vec<f32>) -> usize {
        self.inner.add(&vector)
    }

    /// Zero-copy batch insert from a numpy array (shape [N, D]).
    fn add_np(&mut self, array: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = array.as_array();
        let (n, d) = (arr.nrows(), arr.ncols());
        if let Some(flat) = arr.as_slice() {
            for i in 0..n {
                self.inner.add(&flat[i * d..(i + 1) * d]);
            }
        } else {
            for row in arr.rows() {
                let v: Vec<f32> = row.iter().copied().collect();
                self.inner.add(&v);
            }
        }
        Ok(())
    }

    /// Search for the k nearest neighbours.  Returns list of (id, distance).
    fn search(&self, query: Vec<f32>, k: usize) -> Vec<(usize, f32)> {
        self.inner.search(&query, k)
    }

    /// Zero-copy search from a 1-D numpy query vector. Releases the GIL during
    /// the search so a Python threadpool scales across cores.
    fn search_np(&self, py: Python<'_>, query: PyReadonlyArray1<f32>, k: usize) -> Vec<(usize, f32)> {
        let owned: Vec<f32> = query.as_array().iter().copied().collect();
        py.allow_threads(|| self.inner.search(&owned, k))
    }

    /// Search with explicit n_probe override.
    fn search_with_probe(&self, query: Vec<f32>, k: usize, n_probe: usize) -> Vec<(usize, f32)> {
        self.inner.search_with_probe(&query, k, n_probe)
    }

    /// Soft-delete a vector by global id.
    fn delete(&mut self, id: usize) {
        self.inner.delete(id);
    }

    /// Compact the index by permanently removing soft-deleted vectors.
    /// Returns the number of vectors removed.
    fn vacuum(&mut self) -> usize {
        self.inner.vacuum()
    }

    /// Filtered search: only return vectors whose id is in `allowed_ids`.
    fn search_filtered_np(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        allowed_ids: Vec<usize>,
    ) -> Vec<(usize, f32)> {
        use std::collections::HashSet;
        let allowed: HashSet<usize> = allowed_ids.into_iter().collect();
        let q = query.as_array();
        match q.as_slice() {
            Some(s) => self.inner.search_filtered(s, k, |id| allowed.contains(&id)),
            None => {
                let v: Vec<f32> = q.iter().copied().collect();
                self.inner.search_filtered(&v, k, |id| allowed.contains(&id))
            }
        }
    }

    /// Find the minimum n_probe achieving `target_recall` for `query`.
    /// Returns `(results, n_probe_used)`.
    fn search_for_recall(
        &self,
        query: Vec<f32>,
        k: usize,
        target_recall: f32,
    ) -> (Vec<(usize, f32)>, usize) {
        self.inner.search_for_recall(&query, k, target_recall)
    }

    /// Persist to file (bincode).
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load from file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = IvfIndex::load(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("PyIvfIndex(n_lists={}, trained={})", self.inner.n_lists, self.inner.is_trained())
    }
}

/// IVF-PQ approximate nearest-neighbour index with ADC scoring (Python binding).
#[pyclass]
struct PyIvfPqIndex {
    inner: IvfPqIndex,
}

#[pymethods]
impl PyIvfPqIndex {
    #[new]
    fn new(n_lists: usize, n_probe: usize) -> Self {
        Self { inner: IvfPqIndex::new(n_lists, n_probe) }
    }

    /// Train the coarse quantizer and PQ codebook.
    fn train(
        &mut self,
        vectors: Vec<Vec<f32>>,
        n_subspaces: usize,
        n_centroids: usize,
        max_iter: usize,
        seed: u64,
    ) -> PyResult<()> {
        self.inner
            .train(&vectors, n_subspaces, n_centroids, max_iter, seed)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Zero-copy train from a numpy array (shape [N, D]).
    fn train_np(
        &mut self,
        array: PyReadonlyArray2<f32>,
        n_subspaces: usize,
        n_centroids: usize,
        max_iter: usize,
        seed: u64,
    ) -> PyResult<()> {
        let arr = array.as_array();
        let (n, d) = (arr.nrows(), arr.ncols());
        // Borrow contiguous rows directly; only copy when the array is strided.
        match arr.as_slice() {
            Some(flat) => {
                let rows: Vec<&[f32]> = (0..n).map(|i| &flat[i * d..(i + 1) * d]).collect();
                self.inner.train(&rows, n_subspaces, n_centroids, max_iter, seed)
            }
            None => {
                let owned: Vec<Vec<f32>> =
                    arr.rows().into_iter().map(|r| r.iter().copied().collect()).collect();
                self.inner.train(&owned, n_subspaces, n_centroids, max_iter, seed)
            }
        }
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Add a single vector; returns its global id.
    fn add(&mut self, vector: Vec<f32>) -> usize {
        self.inner.add(&vector)
    }

    /// Zero-copy batch insert from a numpy array (shape [N, D]).
    fn add_np(&mut self, array: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = array.as_array();
        let (n, d) = (arr.nrows(), arr.ncols());
        if let Some(flat) = arr.as_slice() {
            for i in 0..n {
                self.inner.add(&flat[i * d..(i + 1) * d]);
            }
        } else {
            for row in arr.rows() {
                let v: Vec<f32> = row.iter().copied().collect();
                self.inner.add(&v);
            }
        }
        Ok(())
    }

    /// Search for the k nearest neighbours using ADC.  Returns list of (id, distance).
    fn search(&self, query: Vec<f32>, k: usize) -> Vec<(usize, f32)> {
        self.inner.search(&query, k)
    }

    /// Zero-copy search from a 1-D numpy query vector. Releases the GIL during
    /// the search so a Python threadpool scales across cores.
    fn search_np(&self, py: Python<'_>, query: PyReadonlyArray1<f32>, k: usize) -> Vec<(usize, f32)> {
        let owned: Vec<f32> = query.as_array().iter().copied().collect();
        py.allow_threads(|| self.inner.search(&owned, k))
    }

    /// Search with explicit n_probe override.
    fn search_with_probe(&self, query: Vec<f32>, k: usize, n_probe: usize) -> Vec<(usize, f32)> {
        self.inner.search_with_probe(&query, k, n_probe)
    }

    /// Soft-delete a vector by global id.
    fn delete(&mut self, id: usize) {
        self.inner.delete(id);
    }

    /// Compact the index by permanently removing soft-deleted vectors.
    /// Returns the number of vectors removed.
    fn vacuum(&mut self) -> usize {
        self.inner.vacuum()
    }

    /// Find the minimum n_probe achieving `target_recall` for `query`.
    /// Returns `(results, n_probe_used)`.
    fn search_for_recall(
        &self,
        query: Vec<f32>,
        k: usize,
        target_recall: f32,
    ) -> (Vec<(usize, f32)>, usize) {
        self.inner.search_for_recall(&query, k, target_recall)
    }

    /// Batch search over a 2-D numpy query array [Q, D], parallelised across
    /// queries (rayon) with the GIL released. Returns one (id, dist) list per
    /// row — the throughput path that avoids the per-query Python call overhead
    /// of looping `search_with_probe` from Python.
    fn search_batch_np(
        &self,
        py: Python<'_>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        n_probe: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        let arr = queries.as_array();
        let (_q, d) = (arr.nrows(), arr.ncols());
        match arr.as_slice() {
            Some(flat) => py.allow_threads(|| self.inner.search_batch_flat(flat, d, k, n_probe)),
            None => {
                let owned: Vec<f32> = arr.iter().copied().collect();
                py.allow_threads(|| self.inner.search_batch_flat(&owned, d, k, n_probe))
            }
        }
    }

    /// Persist to file (bincode).
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load from file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = IvfPqIndex::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("PyIvfPqIndex(n_lists={}, trained={})", self.inner.n_lists(), self.inner.is_trained())
    }
}

/// IVF-PQ4 SIMD fast-scan index (Python binding).
///
/// Build-once: trains the coarse quantizer + K=16 PQ codebook and populates from
/// an `[N, D]` float32 array in the constructor, then serves approximate
/// nearest-neighbour queries via the `pshufb` fast-scan. ~5-6x the QPS of the
/// classic IVF-PQ scan at matched recall and memory budget.
#[pyclass]
struct PyIvfPq4Index {
    inner: vectro_lib::index::ivf_pq4::IvfPq4Index,
}

#[pymethods]
impl PyIvfPq4Index {
    /// Build from a numpy `[N, D]` float32 array.
    ///
    /// * `n_lists` — coarse Voronoi cells.  * `n_probe` — default cells per query.
    /// * `m`       — PQ subspaces (must divide D).
    #[new]
    #[pyo3(signature = (array, n_lists, n_probe, m, max_iter = 25, seed = 42))]
    fn new(
        py: Python<'_>,
        array: PyReadonlyArray2<f32>,
        n_lists: usize,
        n_probe: usize,
        m: usize,
        max_iter: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let arr = array.as_array();
        let (n, d) = (arr.nrows(), arr.ncols());
        // Own the data so the heavy build can run with the GIL released.
        let data: Vec<Vec<f32>> = match arr.as_slice() {
            Some(flat) => (0..n).map(|i| flat[i * d..(i + 1) * d].to_vec()).collect(),
            None => arr.rows().into_iter().map(|r| r.iter().copied().collect()).collect(),
        };
        let inner = py
            .allow_threads(|| {
                vectro_lib::index::ivf_pq4::IvfPq4Index::build(&data, n_lists, n_probe, m, max_iter, seed)
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Number of indexed vectors.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Search for the k nearest neighbours.  Returns a list of (id, distance).
    fn search(&self, query: Vec<f32>, k: usize) -> Vec<(usize, f32)> {
        self.inner.search(&query, k)
    }

    /// Zero-copy search from a 1-D numpy query, GIL released during the scan so a
    /// Python threadpool scales across cores.
    fn search_np(&self, py: Python<'_>, query: PyReadonlyArray1<f32>, k: usize) -> Vec<(usize, f32)> {
        let owned: Vec<f32> = query.as_array().iter().copied().collect();
        py.allow_threads(|| self.inner.search(&owned, k))
    }

    /// Search with an explicit probe width (GIL released).
    fn search_with_probe(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        n_probe: usize,
    ) -> Vec<(usize, f32)> {
        let owned: Vec<f32> = query.as_array().iter().copied().collect();
        py.allow_threads(|| self.inner.search_with_probe(&owned, k, n_probe))
    }

    fn __repr__(&self) -> String {
        format!("PyIvfPq4Index(n={})", self.inner.len())
    }
}

// ─────────────────────── Quantized HNSW Python bindings ──────────────────────

macro_rules! quant_hnsw_pyclass {
    ($pyname:ident, $inner:ty, $label:expr) => {
        #[doc = concat!($label, " quantized HNSW index (Python binding).")]
        #[pyclass]
        struct $pyname {
            inner: $inner,
        }

        #[pymethods]
        impl $pyname {
            #[new]
            fn new(m: usize, ef_construction: usize) -> Self {
                Self { inner: <$inner>::new(m, ef_construction) }
            }

            fn add(&mut self, vector: Vec<f32>) {
                self.inner.add(&vector);
            }

            fn add_batch(&mut self, vectors: Vec<Vec<f32>>) {
                self.inner.add_batch(&vectors);
            }

            /// Batch insert from a numpy array (shape [N, D]).
            ///
            /// Routed through `add_batch` (not per-row `add`) so quantizers that
            /// derive a per-index transform from the batch — e.g. binary
            /// mean-centering — can establish it before insertion.
            fn add_np(&mut self, array: PyReadonlyArray2<f32>) -> PyResult<()> {
                let arr = array.as_array();
                let (n, d) = (arr.nrows(), arr.ncols());
                let rows: Vec<Vec<f32>> = if let Some(flat) = arr.as_slice() {
                    (0..n).map(|i| flat[i * d..(i + 1) * d].to_vec()).collect()
                } else {
                    arr.rows().into_iter().map(|row| row.iter().copied().collect()).collect()
                };
                self.inner.add_batch(&rows);
                Ok(())
            }

            fn search(&self, query: Vec<f32>, k: usize, ef: usize) -> Vec<(usize, f32)> {
                self.inner.search(&query, k, ef)
            }

            /// Enable INT8 re-rank — must be called before `add`/`add_batch`.
            /// Retains a near-lossless INT8 copy of each vector (~¼ of an f32
            /// store) so `search_rerank*` can re-score graph candidates exactly,
            /// lifting low-recall (binary, NF4) graphs toward R@0.95.
            fn enable_rerank(&mut self) {
                self.inner.enable_rerank();
            }

            /// True once an INT8 re-rank store is populated.
            fn has_rerank(&self) -> bool {
                self.inner.has_rerank()
            }

            /// High-recall search: navigate the quantized graph for `rerank_k`
            /// candidates, then return the exact-cosine top-`k` re-scored against
            /// the INT8 store. Falls back to plain `search` if re-rank is off.
            fn search_rerank_np(
                &self,
                query: PyReadonlyArray1<f32>,
                k: usize,
                ef: usize,
                rerank_k: usize,
            ) -> Vec<(usize, f32)> {
                let q = query.as_array();
                match q.as_slice() {
                    Some(s) => self.inner.search_rerank(s, k, ef, rerank_k),
                    None => {
                        let v: Vec<f32> = q.iter().copied().collect();
                        self.inner.search_rerank(&v, k, ef, rerank_k)
                    }
                }
            }

            /// Batch high-recall re-rank search: queries `[Q, D]`, parallel over
            /// rows (rayon, GIL released).
            fn search_rerank_batch_np(
                &self,
                py: Python<'_>,
                queries: PyReadonlyArray2<f32>,
                k: usize,
                ef: usize,
                rerank_k: usize,
            ) -> Vec<Vec<(usize, f32)>> {
                let arr = queries.as_array();
                let d = arr.ncols();
                if let Some(flat) = arr.as_slice() {
                    py.allow_threads(|| self.inner.search_rerank_batch_flat(flat, d, k, ef, rerank_k))
                } else {
                    let owned: Vec<f32> = arr.iter().copied().collect();
                    py.allow_threads(|| {
                        self.inner.search_rerank_batch_flat(&owned, d, k, ef, rerank_k)
                    })
                }
            }

            /// Zero-copy search from a 1-D numpy query vector. Releases the GIL
            /// during the search so a Python threadpool scales across cores.
            fn search_np(
                &self,
                py: Python<'_>,
                query: PyReadonlyArray1<f32>,
                k: usize,
                ef: usize,
            ) -> Vec<(usize, f32)> {
                let owned: Vec<f32> = query.as_array().iter().copied().collect();
                py.allow_threads(|| self.inner.search(&owned, k, ef))
            }

            /// Batch search: queries shape [Q, D], parallelised across queries
            /// (rayon) with the GIL released. Returns one (id, dist) list per row.
            fn search_batch_np(
                &self,
                py: Python<'_>,
                queries: PyReadonlyArray2<f32>,
                k: usize,
                ef: usize,
            ) -> Vec<Vec<(usize, f32)>> {
                let arr = queries.as_array();
                let d = arr.ncols();
                if let Some(flat) = arr.as_slice() {
                    py.allow_threads(|| self.inner.search_batch_flat(flat, d, k, ef))
                } else {
                    let owned: Vec<f32> = arr.iter().copied().collect();
                    py.allow_threads(|| self.inner.search_batch_flat(&owned, d, k, ef))
                }
            }

            /// Search with an allow-list of node IDs.
            fn search_filtered_np(
                &self,
                query: PyReadonlyArray1<f32>,
                k: usize,
                ef: usize,
                allowed_ids: Vec<usize>,
            ) -> Vec<(usize, f32)> {
                use std::collections::HashSet;
                let allowed: HashSet<usize> = allowed_ids.into_iter().collect();
                let q = query.as_array();
                match q.as_slice() {
                    Some(s) => {
                        self.inner.search_filtered(s, k, ef, |id| allowed.contains(&id))
                    }
                    None => {
                        let v: Vec<f32> = q.iter().copied().collect();
                        self.inner.search_filtered(&v, k, ef, |id| allowed.contains(&id))
                    }
                }
            }

            fn delete(&mut self, id: usize) {
                self.inner.delete(id);
            }

            /// Compact the index by permanently removing soft-deleted nodes.
            /// Returns the number of nodes removed.
            fn vacuum(&mut self) -> usize {
                self.inner.vacuum()
            }

            fn save(&self, path: &str) -> PyResult<()> {
                self.inner
                    .save(std::path::Path::new(path))
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
            }

            #[staticmethod]
            fn load(path: &str) -> PyResult<Self> {
                let inner = <$inner>::load(std::path::Path::new(path))
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }

            fn __len__(&self) -> usize {
                self.inner.len()
            }

            fn __repr__(&self) -> String {
                format!(concat!($label, "HnswIndex(n_vectors={})"), self.inner.len())
            }
        }
    };
}

quant_hnsw_pyclass!(PyBf16HnswIndex,   Bf16HnswIndex,   "BF16");
quant_hnsw_pyclass!(PyInt8HnswIndex,   Int8HnswIndex,   "INT8");
quant_hnsw_pyclass!(PyNf4HnswIndex,    Nf4HnswIndex,    "NF4");
quant_hnsw_pyclass!(PySq2HnswIndex,    Sq2HnswIndex,    "SQ2");
quant_hnsw_pyclass!(PyBinaryHnswIndex, BinaryHnswIndex, "Binary");

/// Encode a single f32 vector to INT8 using SIMD-dispatched abs-max quantisation.
///
/// Returns `(codes, scale)` where `codes` is a `Vec<i8>` of quantised values and
/// `scale` is the per-vector abs-max used for dequantisation.
/// Runs in <1 ms p99 for d ≤ 4096 on Apple Silicon (NEON) and x86-64 (AVX2).
#[pyfunction]
fn encode_int8_fast(vec: Vec<f32>) -> PyResult<(Vec<i8>, f32)> {
    let q = vectro_lib::quant::int8::Int8Vector::encode_fast(&vec);
    Ok((q.codes, q.scale))
}

/// Encode a single f32 vector to packed NF4 (QLoRA-style) with SIMD abs-max scan.
///
/// Returns `(packed, scale, dim)` where `packed` holds `ceil(dim/2)` nibble-packed
/// bytes, `scale` is the per-vector abs-max, and `dim` is the original dimension.
/// Runs in <1 ms p99 for d ≤ 4096 on Apple Silicon (NEON) and x86-64 (AVX2).
#[pyfunction]
fn encode_nf4_fast(vec: Vec<f32>) -> PyResult<(Vec<u8>, f32, usize)> {
    let q = vectro_lib::quant::nf4::Nf4Vector::encode_fast(&vec);
    Ok((q.packed, q.scale, q.dim))
}

/// Batch encode a 2-D float32 numpy array `[N, D]` to packed NF4 with zero
/// per-row FFI crossings or boxed-float marshalling — the NF4 analogue of
/// [`quantize_int8_batch`]. Replaces the per-row `row.tolist()` Python loop
/// (N FFI calls + N·D boxed floats) with a single borrow + rayon-parallel pass.
///
/// Returns `(packed, scales)` where `packed` is shape `[N, ceil(D/2)]` dtype
/// `uint8` (low nibble = even dim, high nibble = odd dim) and `scales` is shape
/// `[N]` dtype `float32` (per-row abs-max).
#[pyfunction]
fn quantize_nf4_batch<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<f32>,
) -> PyResult<(&'py PyArray2<u8>, &'py PyArray1<f32>)> {
    let arr = vectors.as_array();
    let (n, d) = (arr.nrows(), arr.ncols());

    // Borrow the contiguous slice; own a copy only if the input isn't row-major.
    let owned: Option<Vec<f32>> = match arr.as_slice() {
        Some(_) => None,
        None => Some(arr.iter().copied().collect()),
    };
    let flat: &[f32] = match (&owned, arr.as_slice()) {
        (Some(v), _) => v,
        (None, Some(s)) => s,
        (None, None) => unreachable!("non-contiguous arrays were copied above"),
    };

    let bpv = d.div_ceil(2);
    // Uninitialised outputs filled entirely by the rayon kernel (no 0-init).
    // SAFETY: `batch_encode_packed_into` writes all `n*bpv` bytes and `n` scales
    // before we hand the arrays back to Python.
    let packed_arr = unsafe { PyArray2::<u8>::new(py, [n, bpv], false) };
    let scales_arr = unsafe { PyArray1::<f32>::new(py, [n], false) };
    {
        let packed_slice = unsafe { packed_arr.as_slice_mut()? };
        let scales_slice = unsafe { scales_arr.as_slice_mut()? };
        py.allow_threads(|| {
            vectro_lib::quant::nf4::batch_encode_packed_into(
                flat,
                n,
                d,
                packed_slice,
                scales_slice,
            )
        });
    }
    Ok((packed_arr, scales_arr))
}

/// Batch encode a 2-D float32 numpy array [N, D] to INT8 using rayon-parallel
/// abs-max quantisation with zero per-row heap allocation.
///
/// Returns `(codes, scales)` where `codes` is shape [N, D] dtype `int8` and
/// `scales` is shape [N] dtype `float32` (`abs_max / (127 · range_factor)` per
/// row).
///
/// `range_factor` (rf, in `(0, 1]`, default `1.0`) reproduces the Python
/// `VectroBatchProcessor` profiles: `1.0` = `fast` (max element → ±127),
/// `0.95` = `balanced`, `0.90` = `quality` (headroom below ±127).  Codes use
/// `round(v · 127 · rf / abs_max)`.
///
/// Zero-copy on C-contiguous input; auto-vectorised inner loop (NEON/AVX2).
/// Rejects non-finite (NaN/Inf) input with a `ValueError`, and a
/// `range_factor` outside `(0, 1]` with a `ValueError`.
#[pyfunction]
#[pyo3(signature = (vectors, range_factor = 1.0))]
fn quantize_int8_batch<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<f32>,
    range_factor: f32,
) -> PyResult<(&'py PyArray2<i8>, &'py PyArray1<f32>)> {
    if !(range_factor > 0.0 && range_factor <= 1.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "range_factor must be in (0, 1], got {range_factor}"
        )));
    }
    let arr = vectors.as_array();
    let (n, d) = (arr.nrows(), arr.ncols());

    // Own a contiguous copy only when the input isn't already row-major.
    let owned: Option<Vec<f32>> = match arr.as_slice() {
        Some(_) => None,
        None => Some(arr.iter().copied().collect()),
    };
    let flat: &[f32] = match (&owned, arr.as_slice()) {
        (Some(v), _) => v,
        (None, Some(s)) => s,
        (None, None) => unreachable!("non-contiguous arrays were copied above"),
    };

    // Allocate the outputs **uninitialised** and let the rayon kernel fill every
    // element — skipping the serial 0-init of an intermediate `Vec`. The NaN/Inf
    // validation is folded into the same parallel pass (no separate streaming
    // scan — that was the dominant cost). SAFETY: the kernel writes all `n*d`
    // codes and `n` scales before we read them back.
    let codes_arr = unsafe { PyArray2::<i8>::new(py, [n, d], false) };
    let scales_arr = unsafe { PyArray1::<f32>::new(py, [n], false) };
    let bad = {
        let codes_slice = unsafe { codes_arr.as_slice_mut()? };
        let scales_slice = unsafe { scales_arr.as_slice_mut()? };
        py.allow_threads(|| {
            vectro_lib::quant::int8::batch_encode_checked_into_with_range(
                flat, n, d, codes_slice, scales_slice, range_factor,
            )
        })
    };
    if let Some(pos) = bad {
        let (row, col) = if d > 0 { (pos / d, pos % d) } else { (0, pos) };
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "input contains a non-finite value (NaN or Inf) at row {row}, col {col}; \
             quantization requires finite float32 input"
        )));
    }
    Ok((codes_arr, scales_arr))
}

/// Wave 1.3 — Batch encode an L2-normalised f32 numpy array [N, D] to INT8
/// using the single-pass `batch_encode_normalized_into` kernel.
///
/// Caller asserts every row has ||·||_2 ≤ 1.  Skips the abs-max scan, ~1.4×
/// faster than `quantize_int8_batch` for memory-bandwidth-bound workloads.
/// Cosine recall floor is ~0.99 (vs 0.9999 for the abs-max path); use only
/// when the trade-off is acceptable.
///
/// Returns `(codes, scales)` — `scales` is filled with the constant
/// `1.0 / 127.0` for every row.
#[pyfunction]
fn quantize_int8_batch_normalized<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<f32>,
) -> PyResult<(&'py PyArray2<i8>, &'py PyArray1<f32>)> {
    let arr = vectors.as_array();
    let (n, d) = (arr.nrows(), arr.ncols());

    let owned: Option<Vec<f32>> = match arr.as_slice() {
        Some(_) => None,
        None => Some(arr.iter().copied().collect()),
    };
    let flat: &[f32] = match (&owned, arr.as_slice()) {
        (Some(v), _) => v,
        (None, Some(s)) => s,
        (None, None) => unreachable!("non-contiguous arrays were copied above"),
    };

    // Uninitialised outputs filled by the rayon kernel, with the NaN/Inf check
    // folded into the same pass (see `quantize_int8_batch`). SAFETY: the kernel
    // writes all `n*d` codes and `n` scales before we read them back.
    let codes_arr = unsafe { PyArray2::<i8>::new(py, [n, d], false) };
    let scales_arr = unsafe { PyArray1::<f32>::new(py, [n], false) };
    let bad = {
        let codes_slice = unsafe { codes_arr.as_slice_mut()? };
        let scales_slice = unsafe { scales_arr.as_slice_mut()? };
        py.allow_threads(|| {
            vectro_lib::quant::int8::batch_encode_normalized_checked_into(
                flat, n, d, codes_slice, scales_slice,
            )
        })
    };
    if let Some(pos) = bad {
        let (row, col) = if d > 0 { (pos / d, pos % d) } else { (0, pos) };
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "input contains a non-finite value (NaN or Inf) at row {row}, col {col}; \
             quantization requires finite float32 input"
        )));
    }
    Ok((codes_arr, scales_arr))
}

/// Wave 4 — Batch encode a float16 numpy array [N, D] directly to INT8 via
/// the standard abs-max path.  Halves the input bandwidth versus a separate
/// f16→f32 widening pass; the widening is fused into the per-row encode.
///
/// Accepts any C-contiguous PyArray2<half::f16>.  Returns `(codes, scales)`
/// with the same shape and dtype contract as `quantize_int8_batch`.
#[pyfunction]
fn quantize_int8_batch_from_f16<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<half::f16>,
) -> PyResult<(&'py PyArray2<i8>, &'py PyArray1<f32>)> {
    let arr = vectors.as_array();
    let (n, d) = (arr.nrows(), arr.ncols());

    // Own a contiguous f16 copy only when the input isn't already row-major.
    let owned: Option<Vec<half::f16>> = match arr.as_slice() {
        Some(_) => None,
        None => Some(arr.iter().copied().collect()),
    };
    let f16_flat: &[half::f16] = match (&owned, arr.as_slice()) {
        (Some(v), _) => v,
        (None, Some(s)) => s,
        (None, None) => unreachable!("non-contiguous arrays were copied above"),
    };

    // Widen + validate + abs-max encode in ONE fused parallel pass (see
    // `batch_encode_f16_checked_into`) — no separate serial widen / finite-scan /
    // 0-init. Output written straight into uninitialised numpy arrays.
    let codes_arr = unsafe { PyArray2::<i8>::new(py, [n, d], false) };
    let scales_arr = unsafe { PyArray1::<f32>::new(py, [n], false) };
    let bad = {
        let codes_slice = unsafe { codes_arr.as_slice_mut()? };
        let scales_slice = unsafe { scales_arr.as_slice_mut()? };
        py.allow_threads(|| {
            vectro_lib::quant::int8::batch_encode_f16_checked_into(
                f16_flat, n, d, codes_slice, scales_slice,
            )
        })
    };
    if let Some(pos) = bad {
        let (row, col) = if d > 0 { (pos / d, pos % d) } else { (0, pos) };
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "input contains a non-finite value (NaN or Inf) at row {row}, col {col}; \
             quantization requires finite float input"
        )));
    }
    Ok((codes_arr, scales_arr))
}

/// Batch dequantize INT8 codes back to float32.
///
/// `codes` shape [N, D] dtype `int8`, `scales` shape [N] dtype `float32`
/// (`abs_max / 127.0` convention from `quantize_int8_batch`).
/// Returns float32 array of shape [N, D].
#[pyfunction]
fn dequantize_int8_batch<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray2<i8>,
    scales: PyReadonlyArray1<f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let c = codes.as_array();
    let s = scales.as_array();
    let (n, d) = (c.nrows(), c.ncols());
    if s.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "scales length {} != n_vectors {}",
            s.len(),
            n
        )));
    }
    let mut out_flat = vec![0.0f32; n * d];
    match (c.as_slice(), s.as_slice()) {
        (Some(codes_flat), Some(scales_flat)) => {
            vectro_lib::quant::int8::batch_decode_into(codes_flat, scales_flat, d, &mut out_flat);
        }
        _ => {
            let codes_flat: Vec<i8> = c.iter().copied().collect();
            let scales_flat: Vec<f32> = s.iter().copied().collect();
            vectro_lib::quant::int8::batch_decode_into(&codes_flat, &scales_flat, d, &mut out_flat);
        }
    }
    let out_arr = Array2::from_shape_vec((n, d), out_flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out_arr.into_pyarray(py))
}

/// Batch PQ-encode an [N, D] float32 array against a trained centroid table.
///
/// `centroids` has shape `[M, K, sub_dim]` (`K ≤ 256`, `D == M * sub_dim`).
/// Returns codes of shape `[N, M]` dtype `uint8` — the nearest centroid index
/// per sub-space.  Rayon-parallel; numerically matches the NumPy reference in
/// `python/pq_api.py` (modulo equidistant-tie selection).  This is the fast
/// path that lets `pq_api.pq_encode` skip the per-sub-space NumPy loop.
#[pyfunction]
fn pq_encode_batch<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<f32>,
    centroids: PyReadonlyArray3<f32>,
) -> PyResult<&'py PyArray2<u8>> {
    let varr = vectors.as_array();
    let (n, d) = (varr.nrows(), varr.ncols());
    let cdims = centroids.shape();
    let (m, k, sub_dim) = (cdims[0], cdims[1], cdims[2]);

    if k > 256 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "PQ requires K ≤ 256 (got {k})"
        )));
    }
    if m * sub_dim != d {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "centroid shape M*sub_dim ({m}*{sub_dim}) != vector dim {d}"
        )));
    }

    let vflat: Vec<f32> = varr.iter().copied().collect();
    let cflat: Vec<f32> = centroids.as_array().iter().copied().collect();
    let cb = pq::PQCodebook { n_subspaces: m, n_centroids: k, sub_dim, centroids: cflat };

    let mut codes_flat = vec![0u8; n * m];
    pq::pq_encode_into(&vflat, &cb, &mut codes_flat);

    let codes_arr = Array2::from_shape_vec((n, m), codes_flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(codes_arr.into_pyarray(py))
}

/// Train a PQ codebook with the native (SIMD-accelerated, seeded) Lloyd's
/// k-means and return the centroids as a `[M, K, sub_dim]` float32 array.
///
/// This is the fast, dependency-free path behind `pq_api.train_pq_codebook`
/// (no scikit-learn required). The nearest-centroid assignment uses the
/// transposed-LUT kernel that vectorizes across the K axis (see `quant::pq`),
/// and sub-spaces train in parallel via rayon. Deterministic for a fixed seed.
#[pyfunction]
fn pq_train_batch<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<f32>,
    n_subspaces: usize,
    n_centroids: usize,
    max_iter: usize,
    seed: u64,
) -> PyResult<&'py PyArray3<f32>> {
    let varr = vectors.as_array();
    let d = varr.ncols();
    if n_centroids > 256 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "PQ requires K ≤ 256 (got {n_centroids})"
        )));
    }
    if n_subspaces == 0 || !d.is_multiple_of(n_subspaces) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "vector dim {d} not divisible by n_subspaces {n_subspaces}"
        )));
    }
    // Own the rows so training can run without holding the GIL.
    let rows: Vec<Vec<f32>> = varr.rows().into_iter().map(|r| r.to_vec()).collect();
    let cb = py
        .allow_threads(|| pq::train_pq_codebook(&rows, n_subspaces, n_centroids, max_iter, seed))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let arr = Array3::from_shape_vec((cb.n_subspaces, cb.n_centroids, cb.sub_dim), cb.centroids)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

// ─────────────────────── BM25 + Hybrid Search (v6.0.0) ─────────────────────

/// Okapi BM25 full-text index (Python binding).
#[pyclass(name = "BM25Index")]
struct PyBM25Index {
    inner: BM25Index,
}

#[pymethods]
impl PyBM25Index {
    /// Build a BM25 index from parallel lists of document IDs and texts.
    #[staticmethod]
    fn build(ids: Vec<String>, texts: Vec<String>) -> PyResult<Self> {
        if ids.len() != texts.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ids and texts must have the same length",
            ));
        }
        let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        Ok(Self {
            inner: BM25Index::build_from_texts(&id_refs, &text_refs),
        })
    }

    /// Build with custom k1 / b BM25 parameters.
    #[staticmethod]
    fn build_with_params(
        ids: Vec<String>,
        texts: Vec<String>,
        k1: f32,
        b: f32,
    ) -> PyResult<Self> {
        if ids.len() != texts.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ids and texts must have the same length",
            ));
        }
        let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        Ok(Self {
            inner: BM25Index::build_with_params(&id_refs, &text_refs, k1, b),
        })
    }

    /// Return the top-k documents for a query as `[(doc_id, score)]`.
    fn top_k(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        self.inner
            .top_k(query, k)
            .into_iter()
            .map(|(id, s)| (id.to_owned(), s))
            .collect()
    }

    /// Number of indexed documents.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// IDF for a single term (0.0 if unknown).
    fn idf(&self, term: &str) -> f32 {
        self.inner.idf(term).unwrap_or(0.0)
    }
}

/// Hybrid BM25 + dense cosine search.
///
/// Parameters
/// ----------
/// dataset : PyEmbeddingDataset
/// bm25    : BM25Index
/// query_vector : list[float]
/// query_text   : str
/// k            : int — number of results
/// alpha        : float — 1.0 = pure dense, 0.0 = pure BM25, default 0.7
///
/// Returns `[(doc_id, score)]` sorted descending by combined score.
#[pyfunction]
fn hybrid_search_py(
    dataset: &PyEmbeddingDataset,
    bm25: &PyBM25Index,
    query_vector: Vec<f32>,
    query_text: &str,
    k: usize,
    alpha: f32,
) -> Vec<(String, f32)> {
    use vectro_lib::search::hybrid_search;
    hybrid_search(
        &dataset.inner.embeddings,
        &bm25.inner,
        &query_vector,
        query_text,
        k,
        alpha,
    )
    .into_iter()
    .map(|(id, s)| (id.to_owned(), s))
    .collect()
}

/// Main Python module
/// Diagnostic: reset the global distance-eval counter (feature `distcount`).
#[cfg(feature = "distcount")]
#[pyfunction]
fn dist_evals_reset() {
    vectro_lib::index::hnsw::dist_evals_reset();
}

/// Diagnostic: read the global distance-eval counter (feature `distcount`).
#[cfg(feature = "distcount")]
#[pyfunction]
fn dist_evals_get() -> u64 {
    vectro_lib::index::hnsw::dist_evals_get()
}

#[pymodule]
fn vectro_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEmbedding>()?;
    m.add_class::<PyEmbeddingDataset>()?;
    m.add_class::<PySearchIndex>()?;
    m.add_class::<PyQuantizedIndex>()?;
    m.add_class::<PyInt8Encoder>()?;
    m.add_class::<PyNf4Encoder>()?;
    m.add_class::<PyBinaryEncoder>()?;
    m.add_class::<PyPQCodebook>()?;
    m.add_class::<PyHnswIndex>()?;
    m.add_class::<PyBf16Encoder>()?;
    m.add_class::<PyIvfIndex>()?;
    m.add_class::<PyIvfPqIndex>()?;
    m.add_class::<PyIvfPq4Index>()?;
    // Quantized HNSW variants (Phase 22)
    m.add_class::<PyBf16HnswIndex>()?;
    m.add_class::<PyInt8HnswIndex>()?;
    m.add_class::<PyNf4HnswIndex>()?;
    m.add_class::<PySq2HnswIndex>()?;
    m.add_class::<PyBinaryHnswIndex>()?;
    m.add_function(wrap_pyfunction!(compress_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_compression_quality, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_search_performance, m)?)?;
    m.add_function(wrap_pyfunction!(encode_int8_fast, m)?)?;
    m.add_function(wrap_pyfunction!(encode_nf4_fast, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_nf4_batch, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int8_batch, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int8_batch_normalized, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int8_batch_from_f16, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_int8_batch, m)?)?;
    m.add_function(wrap_pyfunction!(pq_encode_batch, m)?)?;
    m.add_function(wrap_pyfunction!(pq_train_batch, m)?)?;
    // BM25 + hybrid search (v6.0.0)
    m.add_class::<PyBM25Index>()?;
    m.add_function(wrap_pyfunction!(hybrid_search_py, m)?)?;
    #[cfg(feature = "distcount")]
    {
        m.add_function(wrap_pyfunction!(dist_evals_reset, m)?)?;
        m.add_function(wrap_pyfunction!(dist_evals_get, m)?)?;
    }

    // Add version info
    m.add("__version__", "4.10.0")?;
    m.add("__author__", "Wesley Scholl")?;
    m.add("__description__", "Python bindings for Vectro high-performance vector compression and search")?;

    Ok(())
}