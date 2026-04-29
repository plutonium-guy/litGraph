//! Python bindings for retrieval — BM25 for now; vector stores later.

use std::sync::Arc;

use litgraph_core::Document;
use litgraph_retrieval::store::VectorStore;
use litgraph_retrieval::{
    embedding_redundant_filter, long_context_reorder, mmr_select, AttributeInfo, Bm25Index,
    ChildSplitter, Compressor, ContextualCompressionRetriever, DocStore,
    retrieve_concurrent, EmbeddingsFilterCompressor, EnsembleReranker, EnsembleRetriever,
    HybridRetriever, LlmExtractCompressor, MemoryDocStore, HydeRetriever,
    MaxMarginalRelevanceRetriever, MultiQueryRetriever, MultiVectorItem, MultiVectorRetriever,
    ParentDocumentRetriever,
    PipelineCompressor, Reranker,
    RerankingRetriever, Retriever, SelfQueryRetriever, TimeWeightedRetriever, VectorRetriever,
};
use litgraph_rerankers_cohere::{CohereConfig, CohereReranker};
use litgraph_rerankers_jina::{JinaRerankConfig, JinaReranker};
use litgraph_rerankers_voyage::{VoyageRerankConfig, VoyageReranker};
use litgraph_stores_chroma::{ChromaConfig, ChromaVectorStore};
use litgraph_stores_hnsw::HnswVectorStore;
use litgraph_stores_memory::MemoryVectorStore;
use litgraph_stores_pgvector::PgVectorStore;
use litgraph_stores_qdrant::{QdrantConfig, QdrantVectorStore};
use litgraph_stores_weaviate::{WeaviateConfig, WeaviateVectorStore};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::cache::PyCachedEmbeddings;
use crate::embeddings::{
    PyBedrockEmbeddings, PyCohereEmbeddings, PyFunctionEmbeddings, PyGeminiEmbeddings,
    PyJinaEmbeddings, PyOpenAIEmbeddings, PyVoyageEmbeddings,
};
use crate::runtime::block_on_compat;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBm25Index>()?;
    m.add_class::<PyMemoryVectorStore>()?;
    m.add_class::<PyHnswVectorStore>()?;
    m.add_class::<PyQdrantVectorStore>()?;
    m.add_class::<PyWeaviateVectorStore>()?;
    m.add_class::<PyPgVectorStore>()?;
    m.add_class::<PyChromaVectorStore>()?;
    m.add_class::<PyVectorRetriever>()?;
    m.add_class::<PyCohereReranker>()?;
    m.add_class::<PyVoyageReranker>()?;
    m.add_class::<PyJinaReranker>()?;
    m.add_class::<PyEnsembleReranker>()?;
    m.add_class::<PyRerankingRetriever>()?;
    m.add_class::<PyHybridRetriever>()?;
    m.add_class::<PyEnsembleRetriever>()?;
    m.add_class::<PyParentDocumentRetriever>()?;
    m.add_class::<PyMultiVectorRetriever>()?;
    m.add_class::<PyMemoryDocStore>()?;
    m.add_class::<PyMultiQueryRetriever>()?;
    m.add_class::<PyHydeRetriever>()?;
    m.add_class::<PyMaxMarginalRelevanceRetriever>()?;
    m.add_class::<PyLlmExtractCompressor>()?;
    m.add_class::<PyEmbeddingsFilterCompressor>()?;
    m.add_class::<PyPipelineCompressor>()?;
    m.add_class::<PyContextualCompressionRetriever>()?;
    m.add_class::<PySelfQueryRetriever>()?;
    m.add_class::<PyTimeWeightedRetriever>()?;
    m.add_function(pyo3::wrap_pyfunction!(evaluate_retrieval, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(evaluate_generation, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_retrieve_concurrent, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(mmr_select_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(embedding_redundant_filter_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(long_context_reorder_py, m)?)?;
    Ok(())
}

/// Maximal Marginal Relevance — pick `k` candidates that are both
/// relevant to `query_embedding` and diverse from each other.
///
/// Args:
///   query_embedding: list[float] — the query's embedding
///   candidates: list[dict] — documents (with content, optional id/metadata)
///   candidate_embeddings: list[list[float]] — same length as candidates
///   k: number of documents to return
///   lambda_mult: 0.0–1.0 — 1.0 = pure relevance, 0.0 = pure diversity
///
/// ```python
/// from litgraph.retrieval import mmr_select
/// docs = mmr_select(qe, retrieved_docs, retrieved_embs, k=5, lambda_mult=0.5)
/// ```
#[pyfunction(name = "mmr_select")]
#[pyo3(signature = (query_embedding, candidates, candidate_embeddings, k, lambda_mult=0.5))]
fn mmr_select_py<'py>(
    py: Python<'py>,
    query_embedding: Vec<f32>,
    candidates: Bound<'py, PyList>,
    candidate_embeddings: Vec<Vec<f32>>,
    k: usize,
    lambda_mult: f32,
) -> PyResult<Bound<'py, PyList>> {
    let docs = parse_docs(&candidates)?;
    let out = mmr_select(
        &query_embedding,
        &docs,
        &candidate_embeddings,
        k,
        lambda_mult,
    );
    docs_to_pylist(py, out)
}

/// Drop documents whose embedding is within `threshold` cosine
/// similarity of an earlier kept document. Higher threshold → keeps
/// more docs. LangChain default ~0.95.
///
/// ```python
/// from litgraph.retrieval import embedding_redundant_filter
/// kept = embedding_redundant_filter(docs, embeddings, threshold=0.95)
/// ```
#[pyfunction(name = "embedding_redundant_filter")]
#[pyo3(signature = (candidates, embeddings, threshold=0.95))]
fn embedding_redundant_filter_py<'py>(
    py: Python<'py>,
    candidates: Bound<'py, PyList>,
    embeddings: Vec<Vec<f32>>,
    threshold: f32,
) -> PyResult<Bound<'py, PyList>> {
    let docs = parse_docs(&candidates)?;
    let out = embedding_redundant_filter(&docs, &embeddings, threshold);
    docs_to_pylist(py, out)
}

/// Reorder a relevance-sorted list to mitigate "lost in the middle"
/// (Liu et al 2023). Top docs end up at edges; least relevant in the
/// middle. Pure permutation — no embeddings needed.
///
/// ```python
/// from litgraph.retrieval import long_context_reorder
/// reordered = long_context_reorder(retrieved_docs)
/// ```
#[pyfunction(name = "long_context_reorder")]
fn long_context_reorder_py<'py>(
    py: Python<'py>,
    docs: Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    let parsed = parse_docs(&docs)?;
    let out = long_context_reorder(&parsed);
    docs_to_pylist(py, out)
}

/// Pure-Rust BM25 index. Rayon-parallel scoring. Release GIL during `search`.
#[pyclass(name = "Bm25Index", module = "litgraph.retrieval")]
pub struct PyBm25Index {
    inner: Arc<Bm25Index>,
}

#[pymethods]
impl PyBm25Index {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(Bm25Index::new()) }
    }

    /// Add documents. `docs` is a list of dicts with `content` (required) and
    /// optional `id` plus arbitrary metadata.
    fn add<'py>(&self, py: Python<'py>, docs: Bound<'py, PyList>) -> PyResult<()> {
        let parsed = parse_docs(&docs)?;
        let idx = self.inner.clone();
        py.allow_threads(|| {
            idx.add(parsed).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Search — returns list of dicts with `content`, `id`, `metadata`, `score`.
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let idx = self.inner.clone();
        let results = py.allow_threads(|| {
            idx.search(&query, k).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }

    fn __len__(&self) -> usize { self.inner.len() }
    fn __repr__(&self) -> String { format!("Bm25Index(docs={})", self.inner.len()) }
}

impl PyBm25Index {
    pub(crate) fn as_retriever(&self) -> Arc<dyn Retriever> {
        self.inner.clone() as Arc<dyn Retriever>
    }
}

/// In-memory vector store. Use for tests + <10k-doc corpora. For production,
/// swap in usearch/Qdrant (upcoming crates).
#[pyclass(name = "MemoryVectorStore", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyMemoryVectorStore {
    pub(crate) inner: Arc<MemoryVectorStore>,
}

#[pymethods]
impl PyMemoryVectorStore {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(MemoryVectorStore::new()) }
    }

    /// Add documents with pre-computed embeddings.
    /// `docs` is a list of dicts `{content, id?, metadata?}`;
    /// `embeddings` is a list of list[float] with matching length.
    fn add<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
        embeddings: Bound<'py, PyList>,
    ) -> PyResult<Vec<String>> {
        let parsed = parse_docs(&docs)?;
        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(embeddings.len());
        for item in embeddings.iter() {
            embs.push(item.extract::<Vec<f32>>()?);
        }
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.add(parsed, embs).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (query_embedding, k=4, filter=None))]
    fn similarity_search<'py>(
        &self,
        py: Python<'py>,
        query_embedding: Vec<f32>,
        k: usize,
        filter: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let filter_map = match filter {
            Some(d) => {
                let mut m = std::collections::HashMap::new();
                for (k, v) in d.iter() {
                    let key: String = k.extract()?;
                    let v_str: String = v.str()?.extract()?;
                    m.insert(key, serde_json::Value::String(v_str));
                }
                Some(m)
            }
            None => None,
        };
        let store = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move {
                store.similarity_search(&query_embedding, k, filter_map.as_ref()).await
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }

    fn delete<'py>(&self, py: Python<'py>, ids: Vec<String>) -> PyResult<()> {
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.delete(&ids).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        let store = self.inner.clone();
        py.allow_threads(|| block_on_compat(async move { store.len().await }))
    }
}

impl PyMemoryVectorStore {
    pub(crate) fn as_store(&self) -> Arc<dyn VectorStore> {
        self.inner.clone() as Arc<dyn VectorStore>
    }
}

/// Embedded HNSW VectorStore — sub-millisecond search over 100k+ docs.
/// Rebuild happens lazily on first search after any add/delete.
#[pyclass(name = "HnswVectorStore", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyHnswVectorStore {
    pub(crate) inner: Arc<HnswVectorStore>,
}

#[pymethods]
impl PyHnswVectorStore {
    #[new]
    #[pyo3(signature = (ef_search=64, ef_construction=200))]
    fn new(ef_search: usize, ef_construction: usize) -> Self {
        let store = HnswVectorStore::new()
            .with_ef_search(ef_search)
            .with_ef_construction(ef_construction);
        Self { inner: Arc::new(store) }
    }

    fn add<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
        embeddings: Bound<'py, PyList>,
    ) -> PyResult<Vec<String>> {
        let parsed = parse_docs(&docs)?;
        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(embeddings.len());
        for item in embeddings.iter() {
            embs.push(item.extract::<Vec<f32>>()?);
        }
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.add(parsed, embs).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (query_embedding, k=4, filter=None))]
    fn similarity_search<'py>(
        &self,
        py: Python<'py>,
        query_embedding: Vec<f32>,
        k: usize,
        filter: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let filter_map = match filter {
            Some(d) => {
                let mut m = std::collections::HashMap::new();
                for (k, v) in d.iter() {
                    let key: String = k.extract()?;
                    let v_str: String = v.str()?.extract()?;
                    m.insert(key, serde_json::Value::String(v_str));
                }
                Some(m)
            }
            None => None,
        };
        let store = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move {
                store.similarity_search(&query_embedding, k, filter_map.as_ref()).await
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }

    fn delete<'py>(&self, py: Python<'py>, ids: Vec<String>) -> PyResult<()> {
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.delete(&ids).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        let store = self.inner.clone();
        py.allow_threads(|| block_on_compat(async move { store.len().await }))
    }
}

impl PyHnswVectorStore {
    pub(crate) fn as_store(&self) -> Arc<dyn VectorStore> {
        self.inner.clone() as Arc<dyn VectorStore>
    }
}

/// Qdrant remote VectorStore (REST API). The collection must already exist, or
/// call `ensure_collection()` once after construction.
#[pyclass(name = "QdrantVectorStore", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyQdrantVectorStore {
    pub(crate) inner: Arc<QdrantVectorStore>,
}

#[pymethods]
impl PyQdrantVectorStore {
    #[new]
    #[pyo3(signature = (url, collection, api_key=None, vector_name=None, timeout_s=30))]
    fn new(
        url: String,
        collection: String,
        api_key: Option<String>,
        vector_name: Option<String>,
        timeout_s: u64,
    ) -> PyResult<Self> {
        let mut cfg = QdrantConfig::new(url, collection);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(k) = api_key { cfg = cfg.with_api_key(k); }
        if let Some(n) = vector_name { cfg = cfg.with_vector_name(n); }
        let store = QdrantVectorStore::new(cfg)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(store) })
    }

    /// Idempotently create the Qdrant collection.
    #[pyo3(signature = (dim, distance="Cosine"))]
    fn ensure_collection<'py>(
        &self,
        py: Python<'py>,
        dim: u64,
        distance: &str,
    ) -> PyResult<()> {
        let store = self.inner.clone();
        let dist = distance.to_string();
        py.allow_threads(|| {
            block_on_compat(async move { store.ensure_collection(dim, &dist).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn add<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
        embeddings: Bound<'py, PyList>,
    ) -> PyResult<Vec<String>> {
        let parsed = parse_docs(&docs)?;
        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(embeddings.len());
        for item in embeddings.iter() { embs.push(item.extract::<Vec<f32>>()?); }
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.add(parsed, embs).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (query_embedding, k=4, filter=None))]
    fn similarity_search<'py>(
        &self,
        py: Python<'py>,
        query_embedding: Vec<f32>,
        k: usize,
        filter: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let filter_map = match filter {
            Some(d) => {
                let mut m = std::collections::HashMap::new();
                for (k, v) in d.iter() {
                    let key: String = k.extract()?;
                    let v_str: String = v.str()?.extract()?;
                    m.insert(key, serde_json::Value::String(v_str));
                }
                Some(m)
            }
            None => None,
        };
        let store = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move {
                store.similarity_search(&query_embedding, k, filter_map.as_ref()).await
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }

    fn delete<'py>(&self, py: Python<'py>, ids: Vec<String>) -> PyResult<()> {
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.delete(&ids).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
}

impl PyQdrantVectorStore {
    pub(crate) fn as_store(&self) -> Arc<dyn VectorStore> {
        self.inner.clone() as Arc<dyn VectorStore>
    }
}

/// Weaviate VectorStore (REST + GraphQL v1 API). Class auto-creation via
/// `ensure_class()`. Caller-supplied document ids are mapped to deterministic
/// UUIDv5 (namespaced per class) so repeat upserts overwrite same object.
///
/// ```python
/// from litgraph.retrieval import WeaviateVectorStore
/// store = WeaviateVectorStore("http://localhost:8080", "Article",
///                              api_key="<wcs-key>")  # api_key optional
/// store.ensure_class()
/// store.add(docs, embeddings)
/// hits = store.similarity_search(query_emb, k=4, filter={"topic": "rust"})
/// ```
#[pyclass(name = "WeaviateVectorStore", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyWeaviateVectorStore {
    pub(crate) inner: Arc<WeaviateVectorStore>,
}

#[pymethods]
impl PyWeaviateVectorStore {
    #[new]
    #[pyo3(signature = (url, class, api_key=None, timeout_s=30))]
    fn new(url: String, class: String, api_key: Option<String>, timeout_s: u64) -> PyResult<Self> {
        let mut cfg = WeaviateConfig::new(url, class);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(k) = api_key { cfg = cfg.with_api_key(k); }
        let store = WeaviateVectorStore::new(cfg)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(store) })
    }

    /// Idempotently create the class with `vectorizer: none` (we always
    /// supply embeddings client-side).
    fn ensure_class<'py>(&self, py: Python<'py>) -> PyResult<()> {
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.ensure_class().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn add<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
        embeddings: Bound<'py, PyList>,
    ) -> PyResult<Vec<String>> {
        let parsed = parse_docs(&docs)?;
        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(embeddings.len());
        for item in embeddings.iter() { embs.push(item.extract::<Vec<f32>>()?); }
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.add(parsed, embs).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (query_embedding, k=4, filter=None))]
    fn similarity_search<'py>(
        &self,
        py: Python<'py>,
        query_embedding: Vec<f32>,
        k: usize,
        filter: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let filter_map = match filter {
            Some(d) => {
                let mut m = std::collections::HashMap::new();
                for (k, v) in d.iter() {
                    let key: String = k.extract()?;
                    // Preserve type info: bool/int → typed; everything else → string.
                    let val = if let Ok(b) = v.extract::<bool>() {
                        serde_json::Value::Bool(b)
                    } else if let Ok(i) = v.extract::<i64>() {
                        serde_json::Value::Number(i.into())
                    } else {
                        serde_json::Value::String(v.str()?.extract::<String>()?)
                    };
                    m.insert(key, val);
                }
                Some(m)
            }
            None => None,
        };
        let store = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move {
                store.similarity_search(&query_embedding, k, filter_map.as_ref()).await
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }

    fn delete<'py>(&self, py: Python<'py>, ids: Vec<String>) -> PyResult<()> {
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.delete(&ids).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
}

impl PyWeaviateVectorStore {
    pub(crate) fn as_store(&self) -> Arc<dyn VectorStore> {
        self.inner.clone() as Arc<dyn VectorStore>
    }
}

/// ChromaDB remote VectorStore (HTTP v1 API). Lazy collection creation on
/// first add/search/delete. Tenant + database default to chroma's startup
/// defaults; override for multi-tenant isolation.
#[pyclass(name = "ChromaVectorStore", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyChromaVectorStore {
    pub(crate) inner: Arc<ChromaVectorStore>,
}

#[pymethods]
impl PyChromaVectorStore {
    #[new]
    #[pyo3(signature = (url, collection, tenant="default_tenant", database="default_database", timeout_s=30))]
    fn new(
        url: String,
        collection: String,
        tenant: &str,
        database: &str,
        timeout_s: u64,
    ) -> PyResult<Self> {
        let cfg = ChromaConfig::new(url, collection)
            .with_tenant(tenant)
            .with_database(database)
            .with_timeout(std::time::Duration::from_secs(timeout_s));
        let store = ChromaVectorStore::new(cfg)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(store) })
    }

    fn add<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
        embeddings: Bound<'py, PyList>,
    ) -> PyResult<Vec<String>> {
        let parsed = parse_docs(&docs)?;
        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(embeddings.len());
        for item in embeddings.iter() { embs.push(item.extract::<Vec<f32>>()?); }
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.add(parsed, embs).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (query_embedding, k=4, filter=None))]
    fn similarity_search<'py>(
        &self,
        py: Python<'py>,
        query_embedding: Vec<f32>,
        k: usize,
        filter: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let filter_map = match filter {
            Some(d) => {
                let mut m = std::collections::HashMap::new();
                for (k, v) in d.iter() {
                    let key: String = k.extract()?;
                    let v_str: String = v.str()?.extract()?;
                    m.insert(key, serde_json::Value::String(v_str));
                }
                Some(m)
            }
            None => None,
        };
        let store = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move {
                store.similarity_search(&query_embedding, k, filter_map.as_ref()).await
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }

    fn delete<'py>(&self, py: Python<'py>, ids: Vec<String>) -> PyResult<()> {
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.delete(&ids).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
}

impl PyChromaVectorStore {
    pub(crate) fn as_store(&self) -> Arc<dyn VectorStore> {
        self.inner.clone() as Arc<dyn VectorStore>
    }
}

/// Postgres + pgvector-backed remote VectorStore. Requires `CREATE EXTENSION vector;`
/// and an ANN index (`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);`).
#[pyclass(name = "PgVectorStore", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyPgVectorStore {
    pub(crate) inner: Arc<PgVectorStore>,
}

#[pymethods]
impl PyPgVectorStore {
    /// Async construction is awkward from Python, so we expose a synchronous
    /// connect-and-initialize call. Blocks on the shared tokio runtime with the
    /// GIL released.
    #[staticmethod]
    #[pyo3(signature = (dsn, table, dim))]
    fn connect(py: Python<'_>, dsn: String, table: String, dim: usize) -> PyResult<Self> {
        let store = py.allow_threads(|| {
            block_on_compat(async move { PgVectorStore::connect(&dsn, table, dim).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(Self { inner: Arc::new(store) })
    }

    fn add<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
        embeddings: Bound<'py, PyList>,
    ) -> PyResult<Vec<String>> {
        let parsed = parse_docs(&docs)?;
        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(embeddings.len());
        for item in embeddings.iter() { embs.push(item.extract::<Vec<f32>>()?); }
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.add(parsed, embs).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (query_embedding, k=4, filter=None))]
    fn similarity_search<'py>(
        &self,
        py: Python<'py>,
        query_embedding: Vec<f32>,
        k: usize,
        filter: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let filter_map = match filter {
            Some(d) => {
                let mut m = std::collections::HashMap::new();
                for (k, v) in d.iter() {
                    let key: String = k.extract()?;
                    let v_str: String = v.str()?.extract()?;
                    m.insert(key, serde_json::Value::String(v_str));
                }
                Some(m)
            }
            None => None,
        };
        let store = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move {
                store.similarity_search(&query_embedding, k, filter_map.as_ref()).await
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }

    fn delete<'py>(&self, py: Python<'py>, ids: Vec<String>) -> PyResult<()> {
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.delete(&ids).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
}

impl PyPgVectorStore {
    pub(crate) fn as_store(&self) -> Arc<dyn VectorStore> {
        self.inner.clone() as Arc<dyn VectorStore>
    }
}

/// Dense retriever — embed query via `Embeddings`, search a `VectorStore`.
/// Accepts `MemoryVectorStore`, `HnswVectorStore`, `QdrantVectorStore`, or `PgVectorStore`.
#[pyclass(name = "VectorRetriever", module = "litgraph.retrieval")]
pub struct PyVectorRetriever {
    inner: Arc<VectorRetriever>,
}

#[pymethods]
impl PyVectorRetriever {
    #[new]
    fn new(embeddings: Bound<'_, PyAny>, store: Bound<'_, PyAny>) -> PyResult<Self> {
        let e: Arc<dyn litgraph_core::Embeddings> = if let Ok(fe) = embeddings.extract::<PyRef<PyFunctionEmbeddings>>() {
            fe.as_embeddings()
        } else if let Ok(oe) = embeddings.extract::<PyRef<PyOpenAIEmbeddings>>() {
            oe.as_embeddings()
        } else if let Ok(ce) = embeddings.extract::<PyRef<PyCohereEmbeddings>>() {
            ce.as_embeddings()
        } else if let Ok(ve) = embeddings.extract::<PyRef<PyVoyageEmbeddings>>() {
            ve.as_embeddings()
        } else if let Ok(ge) = embeddings.extract::<PyRef<PyGeminiEmbeddings>>() {
            ge.as_embeddings()
        } else if let Ok(be) = embeddings.extract::<PyRef<PyBedrockEmbeddings>>() {
            be.as_embeddings()
        } else if let Ok(je) = embeddings.extract::<PyRef<PyJinaEmbeddings>>() {
            je.as_embeddings()
        } else if let Ok(ce) = embeddings.extract::<PyRef<PyCachedEmbeddings>>() {
            ce.as_embeddings()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "embeddings must be FunctionEmbeddings, OpenAIEmbeddings, CohereEmbeddings, VoyageEmbeddings, GeminiEmbeddings, BedrockEmbeddings, JinaEmbeddings, or CachedEmbeddings",
            ));
        };
        let s: Arc<dyn VectorStore> = if let Ok(mem) = store.extract::<PyRef<PyMemoryVectorStore>>() {
            mem.as_store()
        } else if let Ok(hn) = store.extract::<PyRef<PyHnswVectorStore>>() {
            hn.as_store()
        } else if let Ok(qd) = store.extract::<PyRef<PyQdrantVectorStore>>() {
            qd.as_store()
        } else if let Ok(pg) = store.extract::<PyRef<PyPgVectorStore>>() {
            pg.as_store()
        } else if let Ok(ch) = store.extract::<PyRef<PyChromaVectorStore>>() {
            ch.as_store()
        } else if let Ok(wv) = store.extract::<PyRef<PyWeaviateVectorStore>>() {
            wv.as_store()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "store must be MemoryVectorStore, HnswVectorStore, QdrantVectorStore, PgVectorStore, ChromaVectorStore, or WeaviateVectorStore",
            ));
        };
        let inner = VectorRetriever::new(e, s);
        Ok(Self { inner: Arc::new(inner) })
    }

    fn retrieve<'py>(&self, py: Python<'py>, query: String, k: usize) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }
}

impl PyVectorRetriever {
    pub(crate) fn as_retriever(&self) -> Arc<dyn Retriever> {
        self.inner.clone() as Arc<dyn Retriever>
    }
}

pub(crate) fn parse_docs(docs: &Bound<'_, PyList>) -> PyResult<Vec<Document>> {
    let mut out = Vec::with_capacity(docs.len());
    for item in docs.iter() {
        let d: Bound<PyDict> = item.downcast_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("doc must be dict"))?;
        let content: String = d.get_item("content")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'content'"))?
            .extract()?;
        let mut doc = Document::new(content);
        if let Some(id) = d.get_item("id")? {
            let id: String = id.extract()?;
            doc.id = Some(id);
        }
        if let Some(meta) = d.get_item("metadata")? {
            let meta: Bound<PyDict> = meta.downcast_into()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("metadata must be dict"))?;
            for (k, v) in meta.iter() {
                let k: String = k.extract()?;
                let v_str: String = v.str()?.extract()?;
                doc.metadata.insert(k, serde_json::Value::String(v_str));
            }
        }
        out.push(doc);
    }
    Ok(out)
}

pub(crate) fn docs_to_pylist<'py>(py: Python<'py>, docs: Vec<Document>) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty_bound(py);
    for d in docs {
        let dict = PyDict::new_bound(py);
        dict.set_item("content", d.content)?;
        if let Some(id) = d.id { dict.set_item("id", id)?; }
        if let Some(s) = d.score { dict.set_item("score", s)?; }
        let meta = PyDict::new_bound(py);
        for (k, v) in d.metadata {
            let v_str = match v {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            };
            meta.set_item(k, v_str)?;
        }
        dict.set_item("metadata", meta)?;
        list.append(dict)?;
    }
    Ok(list)
}

/// Cohere `/v2/rerank` adapter. Pass to `RerankingRetriever` to wrap any base
/// retriever with cross-encoder reranking.
#[pyclass(name = "CohereReranker", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyCohereReranker {
    pub(crate) inner: Arc<CohereReranker>,
}

#[pymethods]
impl PyCohereReranker {
    #[new]
    #[pyo3(signature = (api_key, model="rerank-english-v3.0", base_url=None, timeout_s=60))]
    fn new(api_key: String, model: &str, base_url: Option<String>, timeout_s: u64) -> PyResult<Self> {
        let mut cfg = CohereConfig::new(api_key, model);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        let r = CohereReranker::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(r) })
    }

    fn rerank<'py>(
        &self,
        py: Python<'py>,
        query: String,
        docs: Bound<'py, PyList>,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let parsed = parse_docs(&docs)?;
        let r = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move { r.rerank(&query, parsed, top_k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }
}

impl PyCohereReranker {
    pub(crate) fn as_reranker(&self) -> Arc<dyn Reranker> {
        self.inner.clone() as Arc<dyn Reranker>
    }
}

/// Voyage AI `/v1/rerank` adapter (rerank-2 / rerank-2-lite). Pairs naturally
/// with `litgraph.embeddings.VoyageEmbeddings` but works against any base
/// retriever.
#[pyclass(name = "VoyageReranker", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyVoyageReranker {
    pub(crate) inner: Arc<VoyageReranker>,
}

#[pymethods]
impl PyVoyageReranker {
    #[new]
    #[pyo3(signature = (api_key, model="rerank-2", base_url=None, timeout_s=60, truncation=true))]
    fn new(
        api_key: String,
        model: &str,
        base_url: Option<String>,
        timeout_s: u64,
        truncation: bool,
    ) -> PyResult<Self> {
        let mut cfg = VoyageRerankConfig::new(api_key, model);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        cfg = cfg.with_truncation(truncation);
        let r = VoyageReranker::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(r) })
    }

    fn rerank<'py>(
        &self,
        py: Python<'py>,
        query: String,
        docs: Bound<'py, PyList>,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let parsed = parse_docs(&docs)?;
        let r = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move { r.rerank(&query, parsed, top_k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }
}

impl PyVoyageReranker {
    pub(crate) fn as_reranker(&self) -> Arc<dyn Reranker> {
        self.inner.clone() as Arc<dyn Reranker>
    }
}

/// Jina AI `/v1/rerank` adapter (`jina-reranker-v2-base-multilingual` and family).
#[pyclass(name = "JinaReranker", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyJinaReranker {
    pub(crate) inner: Arc<JinaReranker>,
}

#[pymethods]
impl PyJinaReranker {
    #[new]
    #[pyo3(signature = (api_key, model="jina-reranker-v2-base-multilingual", base_url=None, timeout_s=60))]
    fn new(api_key: String, model: &str, base_url: Option<String>, timeout_s: u64) -> PyResult<Self> {
        let mut cfg = JinaRerankConfig::new(api_key, model);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        let r = JinaReranker::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(r) })
    }

    fn rerank<'py>(
        &self,
        py: Python<'py>,
        query: String,
        docs: Bound<'py, PyList>,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let parsed = parse_docs(&docs)?;
        let r = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move { r.rerank(&query, parsed, top_k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }
}

impl PyJinaReranker {
    pub(crate) fn as_reranker(&self) -> Arc<dyn Reranker> {
        self.inner.clone() as Arc<dyn Reranker>
    }
}

/// Concurrent reranker ensemble — fan out N rerankers over the same
/// candidate set, fuse their orderings via weighted Reciprocal Rank
/// Fusion. All children invoke in parallel via `tokio::join_all`, so
/// the wall-clock latency is `max(t_i)` across rerankers, not the
/// sum.
///
/// Use this when one reranker isn't reliably better than another and
/// you want the ensemble to smooth out per-model bias — e.g. Cohere
/// + Voyage + a local fastembed cross-encoder. Pass `weights` to
/// emphasise the ones you trust more.
///
/// `weights` length must equal `children` length when given. A
/// weight of `0.0` mutes that branch entirely.
///
/// ```python
/// from litgraph.retrieval import (
///     CohereReranker, VoyageReranker, EnsembleReranker, RerankingRetriever,
/// )
/// rerank = EnsembleReranker(
///     [CohereReranker(...), VoyageReranker(...)],
///     weights=[0.6, 0.4],
/// )
/// retriever = RerankingRetriever(base=hybrid, reranker=rerank)
/// ```
#[pyclass(name = "EnsembleReranker", module = "litgraph.retrieval")]
pub struct PyEnsembleReranker {
    pub(crate) inner: Arc<EnsembleReranker>,
}

impl PyEnsembleReranker {
    pub(crate) fn as_reranker(&self) -> Arc<dyn Reranker> {
        self.inner.clone() as Arc<dyn Reranker>
    }
}

#[pymethods]
impl PyEnsembleReranker {
    #[new]
    #[pyo3(signature = (children, weights=None, rrf_k=60.0))]
    fn new(
        children: Bound<'_, PyList>,
        weights: Option<Vec<f32>>,
        rrf_k: f32,
    ) -> PyResult<Self> {
        if children.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "EnsembleReranker needs at least 2 children",
            ));
        }
        let mut child_arcs: Vec<Arc<dyn Reranker>> = Vec::with_capacity(children.len());
        for item in children.iter() {
            let r: Arc<dyn Reranker> = if let Ok(c) = item.extract::<PyRef<PyCohereReranker>>() {
                c.as_reranker()
            } else if let Ok(v) = item.extract::<PyRef<PyVoyageReranker>>() {
                v.as_reranker()
            } else if let Ok(j) = item.extract::<PyRef<PyJinaReranker>>() {
                j.as_reranker()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "each child must be CohereReranker, VoyageReranker, or JinaReranker",
                ));
            };
            child_arcs.push(r);
        }
        let mut e = match weights {
            Some(w) => EnsembleReranker::with_weights(child_arcs, w)
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?,
            None => EnsembleReranker::new(child_arcs),
        };
        e = e.with_rrf_k(rrf_k);
        Ok(Self { inner: Arc::new(e) })
    }

    fn rerank<'py>(
        &self,
        py: Python<'py>,
        query: String,
        docs: Bound<'py, PyList>,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let parsed = parse_docs(&docs)?;
        let r = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move { r.rerank(&query, parsed, top_k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }

    fn __repr__(&self) -> String {
        format!(
            "EnsembleReranker(children={}, rrf_k={})",
            self.inner.children.len(),
            self.inner.rrf_k,
        )
    }
}

/// Two-stage retriever: pull `over_fetch_k` candidates from a base, then
/// rerank-narrow to `k`. Standard cross-encoder pattern for production RAG.
#[pyclass(name = "RerankingRetriever", module = "litgraph.retrieval")]
pub struct PyRerankingRetriever {
    inner: Arc<RerankingRetriever>,
}

#[pymethods]
impl PyRerankingRetriever {
    /// `base` is any retriever (currently `VectorRetriever`); `reranker` is a
    /// `CohereReranker` or `VoyageReranker`. `over_fetch_k` defaults to `k * 4`.
    #[new]
    #[pyo3(signature = (base, reranker, over_fetch_k=None))]
    fn new(
        base: PyRef<'_, PyVectorRetriever>,
        reranker: Bound<'_, PyAny>,
        over_fetch_k: Option<usize>,
    ) -> PyResult<Self> {
        let r_arc: Arc<dyn Reranker> = if let Ok(c) = reranker.extract::<PyRef<PyCohereReranker>>() {
            c.as_reranker()
        } else if let Ok(v) = reranker.extract::<PyRef<PyVoyageReranker>>() {
            v.as_reranker()
        } else if let Ok(j) = reranker.extract::<PyRef<PyJinaReranker>>() {
            j.as_reranker()
        } else if let Ok(e) = reranker.extract::<PyRef<PyEnsembleReranker>>() {
            e.as_reranker()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "reranker must be CohereReranker, VoyageReranker, JinaReranker, or EnsembleReranker",
            ));
        };
        let mut r = RerankingRetriever::new(base.as_retriever(), r_arc);
        if let Some(n) = over_fetch_k { r = r.with_over_fetch_k(n); }
        Ok(Self { inner: Arc::new(r) })
    }

    fn retrieve<'py>(&self, py: Python<'py>, query: String, k: usize) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }
}

impl PyRerankingRetriever {
    pub(crate) fn as_retriever(&self) -> Arc<dyn Retriever> {
        self.inner.clone() as Arc<dyn Retriever>
    }
}

/// Evaluate a retriever against a labeled dataset of (query, relevant_ids).
/// Returns a dict with `recall_macro` / `mrr_macro` / `ndcg_macro` /
/// `n_queries` / `k` and a `per_query` list. `dataset` items shape:
/// `{"query": str, "relevant_ids": [str]}`.
#[pyfunction]
#[pyo3(signature = (retriever, dataset, k=10, max_concurrency=8))]
fn evaluate_retrieval<'py>(
    py: Python<'py>,
    retriever: Bound<'py, PyAny>,
    dataset: Bound<'py, PyList>,
    k: usize,
    max_concurrency: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r_arc: Arc<dyn Retriever> = if let Ok(v) = retriever.extract::<PyRef<PyVectorRetriever>>() {
        v.as_retriever()
    } else if let Ok(rr) = retriever.extract::<PyRef<PyRerankingRetriever>>() {
        rr.as_retriever()
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "retriever must be VectorRetriever or RerankingRetriever",
        ));
    };

    let mut cases: Vec<litgraph_retrieval::EvalCase> = Vec::with_capacity(dataset.len());
    for item in dataset.iter() {
        let d: Bound<PyDict> = item.downcast_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("each dataset entry must be a dict"))?;
        let query: String = d.get_item("query")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'query'"))?
            .extract()?;
        let relevant_ids: Vec<String> = d.get_item("relevant_ids")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'relevant_ids'"))?
            .extract()?;
        cases.push(litgraph_retrieval::EvalCase { query, relevant_ids });
    }

    let cfg = litgraph_retrieval::EvalConfig { k, max_concurrency };
    let report = py.allow_threads(|| {
        block_on_compat(async move {
            litgraph_retrieval::evaluate_retrieval(r_arc, &cases, cfg).await
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })?;

    let out = PyDict::new_bound(py);
    out.set_item("k", report.k)?;
    out.set_item("n_queries", report.n_queries)?;
    out.set_item("recall_macro", report.recall_macro)?;
    out.set_item("mrr_macro", report.mrr_macro)?;
    out.set_item("ndcg_macro", report.ndcg_macro)?;
    let per_query = PyList::empty_bound(py);
    for q in report.per_query {
        let qd = PyDict::new_bound(py);
        qd.set_item("query", q.query)?;
        qd.set_item("recall", q.recall)?;
        qd.set_item("mrr", q.mrr)?;
        qd.set_item("ndcg", q.ndcg)?;
        qd.set_item("returned_ids", q.returned_ids)?;
        per_query.append(qd)?;
    }
    out.set_item("per_query", per_query)?;
    Ok(out)
}

/// Hybrid retriever — fan out across N child retrievers concurrently,
/// fuse via Reciprocal Rank Fusion (RRF). Direct win over LangChain's
/// sequential ensemble pattern. Children can be any mix of
/// VectorRetriever / RerankingRetriever / Bm25Index.
#[pyclass(name = "HybridRetriever", module = "litgraph.retrieval")]
pub struct PyHybridRetriever {
    pub(crate) inner: Arc<HybridRetriever>,
}

impl PyHybridRetriever {
    pub(crate) fn as_retriever(&self) -> Arc<dyn Retriever> {
        self.inner.clone() as Arc<dyn Retriever>
    }
}

#[pymethods]
impl PyHybridRetriever {
    /// `children` is a list of retrievers (heterogeneous OK). `rrf_k` is the
    /// RRF discount constant (default 60.0; lower = sharper rank emphasis).
    /// `per_child_k` controls over-fetch from each branch (default `k * 2`).
    #[new]
    #[pyo3(signature = (children, rrf_k=60.0, per_child_k=None))]
    fn new(
        children: Bound<'_, PyList>,
        rrf_k: f32,
        per_child_k: Option<usize>,
    ) -> PyResult<Self> {
        if children.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "HybridRetriever needs at least 2 children",
            ));
        }
        let mut child_arcs: Vec<Arc<dyn Retriever>> = Vec::with_capacity(children.len());
        for item in children.iter() {
            let r: Arc<dyn Retriever> = if let Ok(v) = item.extract::<PyRef<PyVectorRetriever>>() {
                v.as_retriever()
            } else if let Ok(rr) = item.extract::<PyRef<PyRerankingRetriever>>() {
                rr.as_retriever()
            } else if let Ok(b) = item.extract::<PyRef<PyBm25Index>>() {
                b.as_retriever()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "each child must be VectorRetriever, RerankingRetriever, or Bm25Index",
                ));
            };
            child_arcs.push(r);
        }
        let mut h = HybridRetriever::new(child_arcs).with_rrf_k(rrf_k);
        if let Some(p) = per_child_k { h = h.with_per_child_k(p); }
        Ok(Self { inner: Arc::new(h) })
    }

    fn retrieve<'py>(&self, py: Python<'py>, query: String, k: usize) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }

    fn __repr__(&self) -> String {
        format!("HybridRetriever(children={}, rrf_k={})",
            self.inner.children.len(), self.inner.rrf_k)
    }
}

/// Weighted Reciprocal Rank Fusion across N child retrievers, fanned out
/// concurrently. Like `HybridRetriever`, but each child has its own
/// weight so a high-precision retriever can dominate a noisier one.
///
/// `weights` length must equal `children` length. Pass `weights=None` for
/// equal weights (equivalent to `HybridRetriever`). A weight of `0.0`
/// silences a branch entirely.
///
/// ```python
/// from litgraph.retrieval import EnsembleRetriever
/// # 70% weight on dense, 30% on BM25.
/// r = EnsembleRetriever([dense_r, bm25], weights=[0.7, 0.3])
/// docs = r.retrieve("how do transformers attend?", k=5)
/// ```
#[pyclass(name = "EnsembleRetriever", module = "litgraph.retrieval")]
pub struct PyEnsembleRetriever {
    pub(crate) inner: Arc<EnsembleRetriever>,
}

impl PyEnsembleRetriever {
    pub(crate) fn as_retriever(&self) -> Arc<dyn Retriever> {
        self.inner.clone() as Arc<dyn Retriever>
    }
}

#[pymethods]
impl PyEnsembleRetriever {
    #[new]
    #[pyo3(signature = (children, weights=None, rrf_k=60.0, per_child_k=None))]
    fn new(
        children: Bound<'_, PyList>,
        weights: Option<Vec<f32>>,
        rrf_k: f32,
        per_child_k: Option<usize>,
    ) -> PyResult<Self> {
        if children.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "EnsembleRetriever needs at least 2 children",
            ));
        }
        let mut child_arcs: Vec<Arc<dyn Retriever>> = Vec::with_capacity(children.len());
        for item in children.iter() {
            let r: Arc<dyn Retriever> = if let Ok(v) = item.extract::<PyRef<PyVectorRetriever>>() {
                v.as_retriever()
            } else if let Ok(rr) = item.extract::<PyRef<PyRerankingRetriever>>() {
                rr.as_retriever()
            } else if let Ok(b) = item.extract::<PyRef<PyBm25Index>>() {
                b.as_retriever()
            } else if let Ok(h) = item.extract::<PyRef<PyHybridRetriever>>() {
                h.as_retriever()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "each child must be VectorRetriever, RerankingRetriever, Bm25Index, or HybridRetriever",
                ));
            };
            child_arcs.push(r);
        }
        let mut e = match weights {
            Some(w) => EnsembleRetriever::with_weights(child_arcs, w)
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?,
            None => EnsembleRetriever::new(child_arcs),
        };
        e = e.with_rrf_k(rrf_k);
        if let Some(p) = per_child_k {
            e = e.with_per_child_k(p);
        }
        Ok(Self { inner: Arc::new(e) })
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let results = py.allow_threads(move || {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, results)
    }

    fn __repr__(&self) -> String {
        format!(
            "EnsembleRetriever(children={}, rrf_k={})",
            self.inner.children.len(),
            self.inner.rrf_k,
        )
    }
}

/// LLM-judge generation eval — faithfulness + answer-relevance + (optional)
/// correctness. Each case dispatches concurrent LLM calls bounded by
/// `max_concurrency`. Returns macro-averaged scores plus per-case breakdowns.
///
/// `judge` is any provider (use a CHEAP model — gpt-4o-mini, claude-haiku —
/// since each case = 2 or 3 LLM calls). `cases` shape:
/// `[{"query": str, "answer": str, "contexts": [str], "reference_answer"?: str}]`.
#[pyfunction]
#[pyo3(signature = (judge, cases, max_concurrency=8, skip_correctness=false))]
fn evaluate_generation<'py>(
    py: Python<'py>,
    judge: Bound<'py, PyAny>,
    cases: Bound<'py, PyList>,
    max_concurrency: usize,
    skip_correctness: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let chat = crate::agents::extract_chat_model(&judge)?;

    let mut parsed: Vec<litgraph_retrieval::GenerationCase> = Vec::with_capacity(cases.len());
    for item in cases.iter() {
        let d: Bound<PyDict> = item.downcast_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("each case must be a dict"))?;
        let query: String = d.get_item("query")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'query'"))?
            .extract()?;
        let answer: String = d.get_item("answer")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'answer'"))?
            .extract()?;
        let contexts: Vec<String> = d.get_item("contexts")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing 'contexts'"))?
            .extract()?;
        let reference_answer: Option<String> = d.get_item("reference_answer")?
            .map(|v| v.extract()).transpose()?;
        parsed.push(litgraph_retrieval::GenerationCase {
            query, answer, contexts, reference_answer,
        });
    }

    let cfg = litgraph_retrieval::GenEvalConfig { max_concurrency, skip_correctness };
    let report = py.allow_threads(|| {
        block_on_compat(async move {
            litgraph_retrieval::evaluate_generation(chat, &parsed, cfg).await
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })?;

    let out = PyDict::new_bound(py);
    out.set_item("n_cases", report.n_cases)?;
    out.set_item("faithfulness_macro", report.faithfulness_macro)?;
    out.set_item("answer_relevance_macro", report.answer_relevance_macro)?;
    out.set_item("correctness_macro", report.correctness_macro)?;
    let per_case = PyList::empty_bound(py);
    for c in report.per_case {
        let d = PyDict::new_bound(py);
        d.set_item("query", c.query)?;
        d.set_item("faithfulness", c.faithfulness)?;
        d.set_item("answer_relevance", c.answer_relevance)?;
        d.set_item("correctness", c.correctness)?;
        per_case.append(d)?;
    }
    out.set_item("per_case", per_case)?;
    Ok(out)
}

// ---------- ParentDocumentRetriever (iter 85) ----------

/// In-process HashMap-backed parent doc store. Drops with the process; for
/// durability across runs, use SqliteDocStore (TODO future iter).
#[pyclass(name = "MemoryDocStore", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyMemoryDocStore {
    pub(crate) inner: Arc<MemoryDocStore>,
}

#[pymethods]
impl PyMemoryDocStore {
    #[new]
    fn new() -> Self { Self { inner: Arc::new(MemoryDocStore::new()) } }

    fn __len__(&self) -> usize { self.inner.len() }

    fn __repr__(&self) -> String {
        format!("MemoryDocStore(len={})", self.inner.len())
    }
}

/// Adapter: any litgraph-splitters `Splitter` implementor → `ChildSplitter`
/// for ParentDocumentRetriever. Holds the splitter behind an Arc and
/// delegates `split()` to `Splitter::split_document()`.
struct SplitterAdapter<S: litgraph_splitters::Splitter + ?Sized> {
    inner: Arc<S>,
}

impl<S: litgraph_splitters::Splitter + ?Sized + 'static> ChildSplitter for SplitterAdapter<S> {
    fn split(&self, doc: &Document) -> Vec<Document> {
        self.inner.split_document(doc)
    }
}

/// ParentDocumentRetriever — embed small chunks for precision, return
/// large parents for context. Direct LangChain parity.
///
/// ```python
/// from litgraph.splitters import RecursiveCharacterSplitter
/// from litgraph.retrieval import (
///     ParentDocumentRetriever, MemoryDocStore, MemoryVectorStore,
///     VectorRetriever,
/// )
/// store = MemoryVectorStore(dim=1536)
/// docs_kv = MemoryDocStore()
/// pdr = ParentDocumentRetriever(
///     child_splitter=RecursiveCharacterSplitter(chunk_size=200),
///     vector_store=store,
///     embeddings=openai_embeddings,
///     parent_store=docs_kv,
/// )
/// pdr.index_documents(docs)
/// hits = pdr.retrieve("what's a vector?", k=2)   # returns full parent docs
/// ```
#[pyclass(name = "ParentDocumentRetriever", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyParentDocumentRetriever {
    pub(crate) inner: Arc<ParentDocumentRetriever>,
}

#[pymethods]
impl PyParentDocumentRetriever {
    /// `child_splitter` — accepts any litGraph splitter (`RecursiveCharacterSplitter`,
    /// `MarkdownHeaderSplitter`, `HtmlHeaderSplitter`, `JsonSplitter`).
    /// `vector_store` — accepts the same union as `VectorRetriever` (Memory / HNSW /
    /// Qdrant / Pgvector / Chroma / Weaviate).
    /// `embeddings` — same union as `VectorRetriever`.
    /// `parent_store` — `MemoryDocStore` (only impl in v1).
    #[new]
    #[pyo3(signature = (child_splitter, vector_store, embeddings, parent_store, child_k_factor=4))]
    fn new(
        child_splitter: Bound<'_, PyAny>,
        vector_store: Bound<'_, PyAny>,
        embeddings: Bound<'_, PyAny>,
        parent_store: PyRef<'_, PyMemoryDocStore>,
        child_k_factor: usize,
    ) -> PyResult<Self> {
        // Splitter adapter: extract the inner Rust splitter and wrap.
        let cs: Arc<dyn ChildSplitter> = if let Ok(s) = child_splitter.extract::<PyRef<crate::splitters::PyRecursiveSplitter>>() {
            Arc::new(SplitterAdapter { inner: Arc::new(s.inner.clone()) })
        } else if let Ok(s) = child_splitter.extract::<PyRef<crate::splitters::PyMarkdownSplitter>>() {
            Arc::new(SplitterAdapter { inner: Arc::new(s.inner.clone()) })
        } else if let Ok(s) = child_splitter.extract::<PyRef<crate::splitters::PyHtmlHeaderSplitter>>() {
            Arc::new(SplitterAdapter { inner: Arc::new(s.inner.clone()) })
        } else {
            // JsonSplitter intentionally omitted: it walks JSON structure and
            // doesn't implement the linear `Splitter` trait. Parent-document
            // pattern doesn't make sense for JSON anyway (no "parent doc" concept
            // for tree-shaped data).
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "child_splitter must be one of RecursiveCharacterSplitter, MarkdownHeaderSplitter, HtmlHeaderSplitter",
            ));
        };

        // Vector store: same polymorphism as PyVectorRetriever.
        let vs: Arc<dyn VectorStore> = if let Ok(mem) = vector_store.extract::<PyRef<PyMemoryVectorStore>>() {
            mem.as_store()
        } else if let Ok(hn) = vector_store.extract::<PyRef<PyHnswVectorStore>>() {
            hn.as_store()
        } else if let Ok(qd) = vector_store.extract::<PyRef<PyQdrantVectorStore>>() {
            qd.as_store()
        } else if let Ok(pg) = vector_store.extract::<PyRef<PyPgVectorStore>>() {
            pg.as_store()
        } else if let Ok(ch) = vector_store.extract::<PyRef<PyChromaVectorStore>>() {
            ch.as_store()
        } else if let Ok(wv) = vector_store.extract::<PyRef<PyWeaviateVectorStore>>() {
            wv.as_store()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "vector_store must be MemoryVectorStore / HnswVectorStore / QdrantVectorStore / PgVectorStore / ChromaVectorStore / WeaviateVectorStore",
            ));
        };

        // Embeddings: same polymorphism as PyVectorRetriever.
        let e: Arc<dyn litgraph_core::Embeddings> = if let Ok(fe) = embeddings.extract::<PyRef<PyFunctionEmbeddings>>() {
            fe.as_embeddings()
        } else if let Ok(oe) = embeddings.extract::<PyRef<PyOpenAIEmbeddings>>() {
            oe.as_embeddings()
        } else if let Ok(ce) = embeddings.extract::<PyRef<PyCohereEmbeddings>>() {
            ce.as_embeddings()
        } else if let Ok(ve) = embeddings.extract::<PyRef<PyVoyageEmbeddings>>() {
            ve.as_embeddings()
        } else if let Ok(ge) = embeddings.extract::<PyRef<PyGeminiEmbeddings>>() {
            ge.as_embeddings()
        } else if let Ok(be) = embeddings.extract::<PyRef<PyBedrockEmbeddings>>() {
            be.as_embeddings()
        } else if let Ok(je) = embeddings.extract::<PyRef<PyJinaEmbeddings>>() {
            je.as_embeddings()
        } else if let Ok(ce) = embeddings.extract::<PyRef<PyCachedEmbeddings>>() {
            ce.as_embeddings()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "embeddings must be a litgraph Embeddings type",
            ));
        };

        let ps: Arc<dyn DocStore> = parent_store.inner.clone() as Arc<dyn DocStore>;
        let pdr = ParentDocumentRetriever::new(cs, vs, e, ps)
            .with_child_k_factor(child_k_factor);
        Ok(Self { inner: Arc::new(pdr) })
    }

    /// Index parent documents — splits each into children, embeds them in
    /// one batch, upserts to vector store, and stores the parents.
    fn index_documents<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
    ) -> PyResult<Vec<String>> {
        let parsed = parse_docs(&docs)?;
        let r = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { r.index_documents(parsed).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Retrieve — returns full parent documents (NOT child chunks).
    #[pyo3(signature = (query, k=4))]
    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let docs = py.allow_threads(move || {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, docs)
    }
}

impl PyParentDocumentRetriever {
    pub(crate) fn as_retriever(&self) -> Arc<dyn Retriever> {
        self.inner.clone() as Arc<dyn Retriever>
    }
}

// ---------- MultiVectorRetriever (iter 188) ----------

/// Index N caller-supplied "perspectives" per parent doc (summary,
/// hypothetical questions, raw chunks, key-entities), retrieve the
/// parent on a hit against any perspective. Distinct from
/// `ParentDocumentRetriever`, which derives perspectives via a
/// `ChildSplitter`. Caller supplies the perspective generation —
/// typical pipelines produce them via a structured-output LLM call.
///
/// Indexing fans out the embedding work across Tokio chunks via the
/// `embed_documents_concurrent` primitive (iter 183) — a 10k-perspective
/// index finishes in O(chunks/concurrency) wall-clock.
///
/// ```python
/// from litgraph.retrieval import (
///     MultiVectorRetriever, MemoryDocStore, MemoryVectorStore,
/// )
/// from litgraph.embeddings import OpenAIEmbeddings
/// emb = OpenAIEmbeddings(api_key="sk-...")
/// store = MemoryVectorStore(dim=1536)
/// docs = MemoryDocStore()
/// mv = MultiVectorRetriever(store, emb, docs)
/// mv.index([
///     {
///         "parent": {"id": "p1", "content": "Long source doc..."},
///         "perspectives": [
///             "summary: this doc covers ...",
///             "Q: what is X?",
///             "Q: why does Y matter?",
///         ],
///     },
/// ])
/// hits = mv.retrieve("what is X?", k=2)
/// ```
#[pyclass(name = "MultiVectorRetriever", module = "litgraph.retrieval")]
pub struct PyMultiVectorRetriever {
    pub(crate) inner: Arc<MultiVectorRetriever>,
}

impl PyMultiVectorRetriever {
    pub(crate) fn as_retriever(&self) -> Arc<dyn Retriever> {
        self.inner.clone() as Arc<dyn Retriever>
    }
}

#[pymethods]
impl PyMultiVectorRetriever {
    #[new]
    #[pyo3(signature = (vector_store, embeddings, parent_store, child_k_factor=4, embed_chunk_size=1024, embed_concurrency=4))]
    fn new(
        vector_store: Bound<'_, PyAny>,
        embeddings: Bound<'_, PyAny>,
        parent_store: PyRef<'_, PyMemoryDocStore>,
        child_k_factor: usize,
        embed_chunk_size: usize,
        embed_concurrency: usize,
    ) -> PyResult<Self> {
        // Same vector-store polymorphism as PyVectorRetriever / PyParentDocumentRetriever.
        let vs: Arc<dyn VectorStore> = if let Ok(mem) = vector_store.extract::<PyRef<PyMemoryVectorStore>>() {
            mem.as_store()
        } else if let Ok(hn) = vector_store.extract::<PyRef<PyHnswVectorStore>>() {
            hn.as_store()
        } else if let Ok(qd) = vector_store.extract::<PyRef<PyQdrantVectorStore>>() {
            qd.as_store()
        } else if let Ok(pg) = vector_store.extract::<PyRef<PyPgVectorStore>>() {
            pg.as_store()
        } else if let Ok(ch) = vector_store.extract::<PyRef<PyChromaVectorStore>>() {
            ch.as_store()
        } else if let Ok(wv) = vector_store.extract::<PyRef<PyWeaviateVectorStore>>() {
            wv.as_store()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "vector_store must be MemoryVectorStore / HnswVectorStore / QdrantVectorStore / PgVectorStore / ChromaVectorStore / WeaviateVectorStore",
            ));
        };
        let e = crate::embeddings::extract_embeddings(&embeddings)?;
        let ps: Arc<dyn DocStore> = parent_store.inner.clone() as Arc<dyn DocStore>;
        let mvr = MultiVectorRetriever::new(vs, e, ps)
            .with_child_k_factor(child_k_factor)
            .with_embed_chunk_size(embed_chunk_size)
            .with_embed_concurrency(embed_concurrency);
        Ok(Self { inner: Arc::new(mvr) })
    }

    /// Index a list of dicts `{"parent": <doc-or-dict>, "perspectives": [str,...]}`.
    /// Returns the parent ids in input order. The embedding work runs in
    /// concurrent chunks via `embed_documents_concurrent`.
    fn index<'py>(
        &self,
        py: Python<'py>,
        items: Bound<'py, PyList>,
    ) -> PyResult<Vec<String>> {
        let mut parsed: Vec<MultiVectorItem> = Vec::with_capacity(items.len());
        for item in items.iter() {
            let d: Bound<PyDict> = item.downcast_into().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "each item must be a dict {parent, perspectives}",
                )
            })?;
            let parent_obj = d.get_item("parent")?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("missing 'parent' key")
            })?;
            let parent_doc = if let Ok(s) = parent_obj.extract::<String>() {
                Document::new(s)
            } else if let Ok(pd) = parent_obj.downcast::<PyDict>() {
                let mut docs = parse_docs(
                    &PyList::new_bound(py, &[pd.clone()]),
                )?;
                docs.pop().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("parent doc parse failed")
                })?
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "parent must be a str or doc dict",
                ));
            };
            let perspectives_obj = d.get_item("perspectives")?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("missing 'perspectives' key")
            })?;
            let perspectives_list: Bound<PyList> = perspectives_obj.downcast_into().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("'perspectives' must be a list of str")
            })?;
            let mut texts: Vec<String> = Vec::with_capacity(perspectives_list.len());
            for p in perspectives_list.iter() {
                texts.push(p.extract::<String>()?);
            }
            parsed.push(MultiVectorItem::new(parent_doc, texts));
        }

        let r = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { r.index(parsed).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (query, k=4))]
    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let docs = py.allow_threads(move || {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, docs)
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiVectorRetriever(child_k_factor={})",
            self.inner.child_k_factor
        )
    }
}

// ---------- MultiQueryRetriever (iter 86) ----------

/// MultiQueryRetriever — LLM paraphrases the query into N variations, runs
/// the base retriever for each in parallel, deduplicates the union, returns
/// top-k. Improves recall on queries that use different vocabulary than the
/// indexed documents.
///
/// Cost: 1 LLM call per `retrieve()` (cheap on flash/mini models) + N×base
/// retrieval calls in parallel.
///
/// ```python
/// from litgraph.providers import OpenAIChat
/// from litgraph.retrieval import MultiQueryRetriever
/// llm = OpenAIChat(model="gpt-4o-mini", api_key=...)
/// mqr = MultiQueryRetriever(base=vector_retriever, llm=llm, num_variations=4)
/// hits = mqr.retrieve("how does borrow checking work", k=5)
/// ```
#[pyclass(name = "MultiQueryRetriever", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyMultiQueryRetriever {
    pub(crate) inner: Arc<MultiQueryRetriever>,
}

#[pymethods]
impl PyMultiQueryRetriever {
    /// `base` — any litGraph Retriever (`VectorRetriever`, `Bm25Index`,
    /// `RerankingRetriever`, `HybridRetriever`, `ParentDocumentRetriever`,
    /// `MultiQueryRetriever`).
    /// `llm` — any litGraph chat model (same union as `ReactAgent.model`).
    /// `num_variations` — paraphrase count (default 4). The original query
    /// is also included unless `include_original=False`.
    #[new]
    #[pyo3(signature = (base, llm, num_variations=4, include_original=true, system_prompt=None))]
    fn new(
        base: Bound<'_, PyAny>,
        llm: Bound<'_, PyAny>,
        num_variations: usize,
        include_original: bool,
        system_prompt: Option<String>,
    ) -> PyResult<Self> {
        let base_arc: Arc<dyn Retriever> = if let Ok(v) = base.extract::<PyRef<PyVectorRetriever>>() {
            v.as_retriever()
        } else if let Ok(b) = base.extract::<PyRef<PyBm25Index>>() {
            b.as_retriever()
        } else if let Ok(r) = base.extract::<PyRef<PyRerankingRetriever>>() {
            r.as_retriever()
        } else if let Ok(h) = base.extract::<PyRef<PyHybridRetriever>>() {
            h.as_retriever()
        } else if let Ok(p) = base.extract::<PyRef<PyParentDocumentRetriever>>() {
            p.as_retriever()
        } else if let Ok(m) = base.extract::<PyRef<PyMultiQueryRetriever>>() {
            m.inner.clone() as Arc<dyn Retriever>
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "base must be a litGraph Retriever (VectorRetriever / Bm25Index / RerankingRetriever / HybridRetriever / ParentDocumentRetriever / MultiQueryRetriever)",
            ));
        };

        let llm_arc = crate::agents::extract_chat_model(&llm)?;
        let mut mqr = MultiQueryRetriever::new(base_arc, llm_arc)
            .with_num_variations(num_variations)
            .with_include_original(include_original);
        if let Some(p) = system_prompt {
            mqr = mqr.with_system_prompt(p);
        }
        Ok(Self { inner: Arc::new(mqr) })
    }

    /// Generate paraphrases without retrieving — useful for previewing /
    /// caching the LLM output outside the retrieval path.
    fn generate_queries<'py>(&self, py: Python<'py>, query: String) -> PyResult<Vec<String>> {
        let r = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { r.generate_queries(&query).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (query, k=4))]
    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let docs = py.allow_threads(move || {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, docs)
    }
}

// ---------- ContextualCompressionRetriever (iter 87) ----------

/// LLM-driven extractive compressor — for each retrieved doc, the LLM
/// rewrites the content to keep ONLY sentences relevant to the query.
/// Docs the LLM marks as `NO_OUTPUT` are dropped. One LLM call per doc,
/// fanned out in parallel.
#[pyclass(name = "LlmExtractCompressor", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyLlmExtractCompressor {
    pub(crate) inner: Arc<LlmExtractCompressor>,
}

#[pymethods]
impl PyLlmExtractCompressor {
    #[new]
    #[pyo3(signature = (llm, prompt=None, max_concurrency=8))]
    fn new(
        llm: Bound<'_, PyAny>,
        prompt: Option<String>,
        max_concurrency: usize,
    ) -> PyResult<Self> {
        let llm_arc = crate::agents::extract_chat_model(&llm)?;
        let mut c = LlmExtractCompressor::new(llm_arc).with_max_concurrency(max_concurrency);
        if let Some(p) = prompt { c = c.with_prompt(p); }
        Ok(Self { inner: Arc::new(c) })
    }
}

/// Cosine-similarity filter compressor — drops docs below the threshold.
/// Cheap pre-filter; combine with `LlmExtractCompressor` via `PipelineCompressor`.
#[pyclass(name = "EmbeddingsFilterCompressor", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyEmbeddingsFilterCompressor {
    pub(crate) inner: Arc<EmbeddingsFilterCompressor>,
}

#[pymethods]
impl PyEmbeddingsFilterCompressor {
    #[new]
    fn new(embeddings: Bound<'_, PyAny>, similarity_threshold: f32) -> PyResult<Self> {
        let e: Arc<dyn litgraph_core::Embeddings> = if let Ok(fe) = embeddings.extract::<PyRef<PyFunctionEmbeddings>>() {
            fe.as_embeddings()
        } else if let Ok(oe) = embeddings.extract::<PyRef<PyOpenAIEmbeddings>>() {
            oe.as_embeddings()
        } else if let Ok(ce) = embeddings.extract::<PyRef<PyCohereEmbeddings>>() {
            ce.as_embeddings()
        } else if let Ok(ve) = embeddings.extract::<PyRef<PyVoyageEmbeddings>>() {
            ve.as_embeddings()
        } else if let Ok(ge) = embeddings.extract::<PyRef<PyGeminiEmbeddings>>() {
            ge.as_embeddings()
        } else if let Ok(be) = embeddings.extract::<PyRef<PyBedrockEmbeddings>>() {
            be.as_embeddings()
        } else if let Ok(je) = embeddings.extract::<PyRef<PyJinaEmbeddings>>() {
            je.as_embeddings()
        } else if let Ok(ce) = embeddings.extract::<PyRef<PyCachedEmbeddings>>() {
            ce.as_embeddings()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "embeddings must be a litGraph Embeddings type",
            ));
        };
        Ok(Self { inner: Arc::new(EmbeddingsFilterCompressor::new(e, similarity_threshold)) })
    }
}

/// Pipeline of compressors run in order. Empty result short-circuits
/// the rest of the pipeline (no point running an LLM on 0 docs).
#[pyclass(name = "PipelineCompressor", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyPipelineCompressor {
    pub(crate) inner: Arc<PipelineCompressor>,
}

fn extract_compressor(any: &Bound<'_, PyAny>) -> PyResult<Arc<dyn Compressor>> {
    if let Ok(c) = any.extract::<PyRef<PyLlmExtractCompressor>>() {
        Ok(c.inner.clone() as Arc<dyn Compressor>)
    } else if let Ok(c) = any.extract::<PyRef<PyEmbeddingsFilterCompressor>>() {
        Ok(c.inner.clone() as Arc<dyn Compressor>)
    } else if let Ok(c) = any.extract::<PyRef<PyPipelineCompressor>>() {
        Ok(c.inner.clone() as Arc<dyn Compressor>)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "compressor must be LlmExtractCompressor, EmbeddingsFilterCompressor, or PipelineCompressor",
        ))
    }
}

#[pymethods]
impl PyPipelineCompressor {
    #[new]
    fn new(steps: Bound<'_, PyList>) -> PyResult<Self> {
        let mut s: Vec<Arc<dyn Compressor>> = Vec::with_capacity(steps.len());
        for item in steps.iter() {
            s.push(extract_compressor(&item)?);
        }
        Ok(Self { inner: Arc::new(PipelineCompressor::new(s)) })
    }
}

/// Wraps a base Retriever with a Compressor — base over-fetches, compressor
/// filters / extracts. Direct LangChain `ContextualCompressionRetriever`
/// parity.
///
/// ```python
/// from litgraph.providers import OpenAIChat
/// from litgraph.retrieval import (
///     LlmExtractCompressor, ContextualCompressionRetriever,
/// )
/// llm = OpenAIChat(model="gpt-4o-mini", api_key=...)
/// ccr = ContextualCompressionRetriever(
///     base=vector_retriever,
///     compressor=LlmExtractCompressor(llm=llm),
/// )
/// hits = ccr.retrieve("how does X work?", k=4)  # → token-trimmed docs
/// ```
#[pyclass(name = "ContextualCompressionRetriever", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyContextualCompressionRetriever {
    pub(crate) inner: Arc<ContextualCompressionRetriever>,
}

#[pymethods]
impl PyContextualCompressionRetriever {
    /// `base` — same retriever polymorphism as other compositional retrievers.
    /// `compressor` — `LlmExtractCompressor` / `EmbeddingsFilterCompressor` /
    /// `PipelineCompressor`.
    /// `over_fetch_factor` — base.retrieve called with `k * factor`.
    #[new]
    #[pyo3(signature = (base, compressor, over_fetch_factor=2))]
    fn new(
        base: Bound<'_, PyAny>,
        compressor: Bound<'_, PyAny>,
        over_fetch_factor: usize,
    ) -> PyResult<Self> {
        let base_arc: Arc<dyn Retriever> = if let Ok(v) = base.extract::<PyRef<PyVectorRetriever>>() {
            v.as_retriever()
        } else if let Ok(b) = base.extract::<PyRef<PyBm25Index>>() {
            b.as_retriever()
        } else if let Ok(r) = base.extract::<PyRef<PyRerankingRetriever>>() {
            r.as_retriever()
        } else if let Ok(h) = base.extract::<PyRef<PyHybridRetriever>>() {
            h.as_retriever()
        } else if let Ok(p) = base.extract::<PyRef<PyParentDocumentRetriever>>() {
            p.as_retriever()
        } else if let Ok(m) = base.extract::<PyRef<PyMultiQueryRetriever>>() {
            m.inner.clone() as Arc<dyn Retriever>
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "base must be a litGraph Retriever",
            ));
        };
        let comp = extract_compressor(&compressor)?;
        let r = ContextualCompressionRetriever::new(base_arc, comp)
            .with_over_fetch_factor(over_fetch_factor);
        Ok(Self { inner: Arc::new(r) })
    }

    #[pyo3(signature = (query, k=4))]
    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let docs = py.allow_threads(move || {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, docs)
    }
}

// ---------- SelfQueryRetriever (iter 90) ----------

/// SelfQueryRetriever — LLM extracts a metadata filter from the user's
/// natural-language query before vector search. Direct LangChain parity.
///
/// ```python
/// from litgraph.providers import OpenAIChat
/// from litgraph.retrieval import SelfQueryRetriever, MemoryVectorStore
/// chat = OpenAIChat(model="gpt-4o-mini", api_key=...)
/// store = MemoryVectorStore()
/// store.add(docs, embeddings)  # docs carry `language`, `year` metadata
/// sqr = SelfQueryRetriever(
///     embeddings=openai_embed,
///     store=store,
///     llm=chat,
///     document_contents="rust crate descriptions",
///     attributes=[
///         {"name": "language", "description": "Programming language", "type": "string"},
///         {"name": "year", "description": "Publication year", "type": "integer"},
///     ],
/// )
/// hits = sqr.retrieve("rust crates from 2024", k=5)
/// # LLM extracts {query:"crates", filter:{language:"rust", year:2024}}
/// # Then vector search runs ONLY over docs matching the filter.
/// ```
#[pyclass(name = "SelfQueryRetriever", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PySelfQueryRetriever {
    pub(crate) inner: Arc<SelfQueryRetriever>,
}

#[pymethods]
impl PySelfQueryRetriever {
    /// `attributes` — list of dicts: `{"name", "description", "type"}` where
    /// type is one of `"string"` / `"integer"` / `"number"` / `"boolean"`.
    #[new]
    #[pyo3(signature = (embeddings, store, llm, document_contents, attributes, system_prompt=None))]
    fn new(
        embeddings: Bound<'_, PyAny>,
        store: Bound<'_, PyAny>,
        llm: Bound<'_, PyAny>,
        document_contents: String,
        attributes: Bound<'_, PyList>,
        system_prompt: Option<String>,
    ) -> PyResult<Self> {
        let e: Arc<dyn litgraph_core::Embeddings> = if let Ok(fe) = embeddings.extract::<PyRef<PyFunctionEmbeddings>>() {
            fe.as_embeddings()
        } else if let Ok(oe) = embeddings.extract::<PyRef<PyOpenAIEmbeddings>>() {
            oe.as_embeddings()
        } else if let Ok(ce) = embeddings.extract::<PyRef<PyCohereEmbeddings>>() {
            ce.as_embeddings()
        } else if let Ok(ve) = embeddings.extract::<PyRef<PyVoyageEmbeddings>>() {
            ve.as_embeddings()
        } else if let Ok(ge) = embeddings.extract::<PyRef<PyGeminiEmbeddings>>() {
            ge.as_embeddings()
        } else if let Ok(be) = embeddings.extract::<PyRef<PyBedrockEmbeddings>>() {
            be.as_embeddings()
        } else if let Ok(je) = embeddings.extract::<PyRef<PyJinaEmbeddings>>() {
            je.as_embeddings()
        } else if let Ok(ce) = embeddings.extract::<PyRef<PyCachedEmbeddings>>() {
            ce.as_embeddings()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "embeddings must be a litGraph Embeddings type",
            ));
        };

        let s: Arc<dyn litgraph_retrieval::VectorStore> = if let Ok(mem) = store.extract::<PyRef<PyMemoryVectorStore>>() {
            mem.as_store()
        } else if let Ok(hn) = store.extract::<PyRef<PyHnswVectorStore>>() {
            hn.as_store()
        } else if let Ok(qd) = store.extract::<PyRef<PyQdrantVectorStore>>() {
            qd.as_store()
        } else if let Ok(pg) = store.extract::<PyRef<PyPgVectorStore>>() {
            pg.as_store()
        } else if let Ok(ch) = store.extract::<PyRef<PyChromaVectorStore>>() {
            ch.as_store()
        } else if let Ok(wv) = store.extract::<PyRef<PyWeaviateVectorStore>>() {
            wv.as_store()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "store must be a litGraph VectorStore type",
            ));
        };

        let llm_arc = crate::agents::extract_chat_model(&llm)?;

        let mut attrs: Vec<AttributeInfo> = Vec::with_capacity(attributes.len());
        for item in attributes.iter() {
            let d = item.downcast::<PyDict>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("each attribute must be a dict")
            })?;
            let name: String = d
                .get_item("name")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("attribute missing 'name'"))?
                .extract()?;
            let description: String = d
                .get_item("description")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("attribute missing 'description'"))?
                .extract()?;
            let field_type: String = d
                .get_item("type")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("attribute missing 'type'"))?
                .extract()?;
            attrs.push(AttributeInfo::new(name, description, field_type));
        }

        let mut sqr = SelfQueryRetriever::new(e, s, llm_arc, document_contents, attrs);
        if let Some(p) = system_prompt {
            sqr = sqr.with_system_prompt(p);
        }
        Ok(Self { inner: Arc::new(sqr) })
    }

    /// Run JUST the LLM extraction step — preview what the retriever would
    /// search for. Returns `{"query": str, "filter": dict}`.
    fn extract_query_and_filter<'py>(
        &self,
        py: Python<'py>,
        query: String,
    ) -> PyResult<Bound<'py, PyDict>> {
        let r = self.inner.clone();
        let (q, f) = py.allow_threads(|| {
            block_on_compat(async move { r.extract_query_and_filter(&query).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let out = PyDict::new_bound(py);
        out.set_item("query", q)?;
        let f_py = PyDict::new_bound(py);
        for (k, v) in f {
            f_py.set_item(k, crate::graph::json_to_py(py, &v)?)?;
        }
        out.set_item("filter", f_py)?;
        Ok(out)
    }

    #[pyo3(signature = (query, k=4))]
    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let docs = py.allow_threads(move || {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, docs)
    }
}

// ---------- TimeWeightedRetriever (iter 93) ----------

/// TimeWeightedVectorStoreRetriever — boost recently-accessed docs via
/// exponential decay. Direct LangChain parity. Use case: agent memory /
/// chat-history retrieval where recent turns matter more than older ones
/// even at lower cosine similarity.
///
/// `combined_score = similarity_score + (1 - decay_rate)^hours_since_access`
///
/// Side effect: every retrieved doc has its `last_accessed` bumped to now,
/// so frequently-used docs stay surfaced even as time passes.
///
/// ```python
/// from litgraph.retrieval import TimeWeightedRetriever, MemoryVectorStore
/// store = MemoryVectorStore()
/// twr = TimeWeightedRetriever(embeddings=embed, store=store, decay_rate=0.01)
/// twr.add_documents(docs, embeddings)
/// hits = twr.retrieve("query", k=5)  # → fresh + relevant docs ranked first
/// ```
#[pyclass(name = "TimeWeightedRetriever", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyTimeWeightedRetriever {
    pub(crate) inner: Arc<TimeWeightedRetriever>,
}

#[pymethods]
impl PyTimeWeightedRetriever {
    #[new]
    #[pyo3(signature = (embeddings, store, decay_rate=0.01, over_fetch_factor=4))]
    fn new(
        embeddings: Bound<'_, PyAny>,
        store: Bound<'_, PyAny>,
        decay_rate: f32,
        over_fetch_factor: usize,
    ) -> PyResult<Self> {
        let e: Arc<dyn litgraph_core::Embeddings> = if let Ok(fe) = embeddings.extract::<PyRef<PyFunctionEmbeddings>>() {
            fe.as_embeddings()
        } else if let Ok(oe) = embeddings.extract::<PyRef<PyOpenAIEmbeddings>>() {
            oe.as_embeddings()
        } else if let Ok(ce) = embeddings.extract::<PyRef<PyCohereEmbeddings>>() {
            ce.as_embeddings()
        } else if let Ok(ve) = embeddings.extract::<PyRef<PyVoyageEmbeddings>>() {
            ve.as_embeddings()
        } else if let Ok(ge) = embeddings.extract::<PyRef<PyGeminiEmbeddings>>() {
            ge.as_embeddings()
        } else if let Ok(be) = embeddings.extract::<PyRef<PyBedrockEmbeddings>>() {
            be.as_embeddings()
        } else if let Ok(je) = embeddings.extract::<PyRef<PyJinaEmbeddings>>() {
            je.as_embeddings()
        } else if let Ok(ce) = embeddings.extract::<PyRef<PyCachedEmbeddings>>() {
            ce.as_embeddings()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "embeddings must be a litGraph Embeddings type",
            ));
        };
        let s: Arc<dyn litgraph_retrieval::VectorStore> = if let Ok(mem) = store.extract::<PyRef<PyMemoryVectorStore>>() {
            mem.as_store()
        } else if let Ok(hn) = store.extract::<PyRef<PyHnswVectorStore>>() {
            hn.as_store()
        } else if let Ok(qd) = store.extract::<PyRef<PyQdrantVectorStore>>() {
            qd.as_store()
        } else if let Ok(pg) = store.extract::<PyRef<PyPgVectorStore>>() {
            pg.as_store()
        } else if let Ok(ch) = store.extract::<PyRef<PyChromaVectorStore>>() {
            ch.as_store()
        } else if let Ok(wv) = store.extract::<PyRef<PyWeaviateVectorStore>>() {
            wv.as_store()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "store must be a litGraph VectorStore type",
            ));
        };
        let twr = TimeWeightedRetriever::new(e, s)
            .with_decay_rate(decay_rate)
            .with_over_fetch_factor(over_fetch_factor);
        Ok(Self { inner: Arc::new(twr) })
    }

    /// Add docs + initialize their last_accessed timestamps to "now".
    /// Returns the assigned ids.
    fn add_documents<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
        embeddings: Bound<'py, PyList>,
    ) -> PyResult<Vec<String>> {
        let parsed = parse_docs(&docs)?;
        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(embeddings.len());
        for item in embeddings.iter() { embs.push(item.extract::<Vec<f32>>()?); }
        let r = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { r.add_documents(parsed, embs).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Snapshot of (id, last_accessed_ms) — useful for observability
    /// dashboards showing which docs the system is actually using.
    fn access_log<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let log = self.inner.access_log();
        let out = PyList::empty_bound(py);
        for (id, ts) in log {
            let d = PyDict::new_bound(py);
            d.set_item("id", id)?;
            d.set_item("last_accessed_ms", ts)?;
            out.append(d)?;
        }
        Ok(out)
    }

    #[pyo3(signature = (query, k=4))]
    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let docs = py.allow_threads(move || {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, docs)
    }
}

// ---------- HydeRetriever (iter 132) ----------
//
// Hypothetical Document Embeddings. Wraps a base retriever + ChatModel.
// Asks the LLM to write a hypothetical answer to the user's question,
// then retrieves using THAT answer's embedding. Boosts recall on
// abstract / conceptual queries where question + document vocabulary
// diverge.
//
// Orthogonal to MultiQueryRetriever — MultiQuery generates N question
// paraphrases; HyDE generates ONE answer passage. Stack both for
// maximum recall on high-precision workloads.
#[pyclass(name = "HydeRetriever", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyHydeRetriever { pub(crate) inner: Arc<HydeRetriever> }

#[pymethods]
impl PyHydeRetriever {
    /// `base` — any litGraph Retriever.
    /// `llm` — any litGraph chat model (pair with a cheap model;
    /// the hypothetical doesn't need to be factually correct).
    /// `include_original=True` also retrieves with the raw user query
    /// (belt-and-suspenders — good for off-topic hypotheticals).
    /// `system_prompt=None` uses the default encyclopedia-style prompt.
    #[new]
    #[pyo3(signature = (base, llm, include_original=true, system_prompt=None))]
    fn new(
        base: Bound<'_, PyAny>,
        llm: Bound<'_, PyAny>,
        include_original: bool,
        system_prompt: Option<String>,
    ) -> PyResult<Self> {
        let base_arc: Arc<dyn Retriever> = if let Ok(v) = base.extract::<PyRef<PyVectorRetriever>>() {
            v.as_retriever()
        } else if let Ok(b) = base.extract::<PyRef<PyBm25Index>>() {
            b.as_retriever()
        } else if let Ok(r) = base.extract::<PyRef<PyRerankingRetriever>>() {
            r.as_retriever()
        } else if let Ok(h) = base.extract::<PyRef<PyHybridRetriever>>() {
            h.as_retriever()
        } else if let Ok(p) = base.extract::<PyRef<PyParentDocumentRetriever>>() {
            p.as_retriever()
        } else if let Ok(m) = base.extract::<PyRef<PyMultiQueryRetriever>>() {
            m.inner.clone() as Arc<dyn Retriever>
        } else if let Ok(hy) = base.extract::<PyRef<PyHydeRetriever>>() {
            hy.inner.clone() as Arc<dyn Retriever>
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "base must be a litGraph Retriever (VectorRetriever / Bm25Index / RerankingRetriever / HybridRetriever / ParentDocumentRetriever / MultiQueryRetriever / HydeRetriever)",
            ));
        };

        let llm_arc = crate::agents::extract_chat_model(&llm)?;
        let mut hyde = HydeRetriever::new(base_arc, llm_arc)
            .with_include_original(include_original);
        if let Some(p) = system_prompt {
            hyde = hyde.with_system_prompt(p);
        }
        Ok(Self { inner: Arc::new(hyde) })
    }

    /// Generate the hypothetical answer without retrieving — useful for
    /// previewing / caching outside the retrieval path.
    fn generate_hypothetical<'py>(
        &self,
        py: Python<'py>,
        query: String,
    ) -> PyResult<String> {
        let r = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { r.generate_hypothetical(&query).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (query, k=4))]
    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let docs = py.allow_threads(|| {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, docs)
    }
}

/// Max-Marginal-Relevance retriever — over-fetches K from base, then
/// MMR-selects final K balancing relevance vs novelty. Reduces near-
/// duplicate results in RAG. LangChain `MaxMarginalRelevanceRetriever`.
///
/// `lambda_mult ∈ [0,1]`: 1.0 = pure relevance (top-K), 0.5 = balanced
/// (LangChain default), 0.0 = pure diversity. `fetch_k` controls how many
/// candidates the base retriever surfaces before MMR; default 20.
///
/// ```python
/// from litgraph.retrieval import MaxMarginalRelevanceRetriever
/// mmr = MaxMarginalRelevanceRetriever(
///     base=vector_retriever, embeddings=embedder,
///     fetch_k=20, lambda_mult=0.5,
/// )
/// docs = mmr.retrieve("multi-aspect question", k=5)
/// ```
#[pyclass(name = "MaxMarginalRelevanceRetriever", module = "litgraph.retrieval")]
#[derive(Clone)]
pub struct PyMaxMarginalRelevanceRetriever {
    pub(crate) inner: Arc<MaxMarginalRelevanceRetriever>,
}

#[pymethods]
impl PyMaxMarginalRelevanceRetriever {
    #[new]
    #[pyo3(signature = (base, embeddings, fetch_k=20, lambda_mult=0.5))]
    fn new(
        base: Bound<'_, PyAny>,
        embeddings: Bound<'_, PyAny>,
        fetch_k: usize,
        lambda_mult: f32,
    ) -> PyResult<Self> {
        let base_arc: Arc<dyn Retriever> = if let Ok(v) = base.extract::<PyRef<PyVectorRetriever>>() {
            v.as_retriever()
        } else if let Ok(b) = base.extract::<PyRef<PyBm25Index>>() {
            b.as_retriever()
        } else if let Ok(r) = base.extract::<PyRef<PyRerankingRetriever>>() {
            r.as_retriever()
        } else if let Ok(h) = base.extract::<PyRef<PyHybridRetriever>>() {
            h.as_retriever()
        } else if let Ok(p) = base.extract::<PyRef<PyParentDocumentRetriever>>() {
            p.as_retriever()
        } else if let Ok(m) = base.extract::<PyRef<PyMultiQueryRetriever>>() {
            m.inner.clone() as Arc<dyn Retriever>
        } else if let Ok(hy) = base.extract::<PyRef<PyHydeRetriever>>() {
            hy.inner.clone() as Arc<dyn Retriever>
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "base must be a litGraph Retriever",
            ));
        };
        let emb_arc = crate::embeddings::extract_embeddings(&embeddings)?;
        let mmr = MaxMarginalRelevanceRetriever::new(base_arc, emb_arc)
            .with_fetch_k(fetch_k)
            .with_lambda(lambda_mult);
        Ok(Self { inner: Arc::new(mmr) })
    }

    #[pyo3(signature = (query, k=4))]
    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let r = self.inner.clone();
        let docs = py.allow_threads(|| {
            block_on_compat(async move { r.retrieve(&query, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, docs)
    }

    fn __repr__(&self) -> String {
        format!(
            "MaxMarginalRelevanceRetriever(fetch_k={}, lambda_mult={})",
            self.inner.fetch_k, self.inner.lambda_mult,
        )
    }
}

// ---------- retrieve_concurrent (iter 190) ----------

/// Extract an `Arc<dyn Retriever>` from any of the supported py
/// retriever wrappers. Used by `retrieve_concurrent` and any future
/// helpers that need polymorphic retriever input.
fn extract_retriever_arc(item: &Bound<'_, PyAny>) -> PyResult<Arc<dyn Retriever>> {
    if let Ok(v) = item.extract::<PyRef<PyVectorRetriever>>() {
        return Ok(v.as_retriever());
    }
    if let Ok(b) = item.extract::<PyRef<PyBm25Index>>() {
        return Ok(b.as_retriever());
    }
    if let Ok(rr) = item.extract::<PyRef<PyRerankingRetriever>>() {
        return Ok(rr.as_retriever());
    }
    if let Ok(h) = item.extract::<PyRef<PyHybridRetriever>>() {
        return Ok(h.as_retriever());
    }
    if let Ok(e) = item.extract::<PyRef<PyEnsembleRetriever>>() {
        return Ok(e.as_retriever());
    }
    if let Ok(p) = item.extract::<PyRef<PyParentDocumentRetriever>>() {
        return Ok(p.as_retriever());
    }
    if let Ok(m) = item.extract::<PyRef<PyMultiVectorRetriever>>() {
        return Ok(m.as_retriever());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "retriever must be VectorRetriever, Bm25Index, RerankingRetriever, HybridRetriever, \
         EnsembleRetriever, ParentDocumentRetriever, or MultiVectorRetriever",
    ))
}

/// Run a single retriever against N caller-supplied queries
/// concurrently, capped at `max_concurrency` in flight. Output is
/// aligned 1:1 with `queries`. Per-query failures isolate by default
/// (slot becomes `{"error": "..."}`); pass `fail_fast=True` to raise
/// on the first error and abort the rest.
///
/// Distinct from `MultiQueryRetriever` (which uses an LLM to expand
/// ONE query into N variants) and `EnsembleRetriever` (which uses
/// DIFFERENT retrievers on the same query). This helper is the
/// "evaluate / batch" path: same retriever, many caller queries.
///
/// ```python
/// from litgraph.retrieval import retrieve_concurrent
/// queries = [c["query"] for c in eval_cases]
/// results = retrieve_concurrent(retriever, queries, k=10, max_concurrency=16)
/// for q, hits in zip(queries, results):
///     if isinstance(hits, dict) and "error" in hits:
///         print(f"failed: {q}: {hits['error']}")
///     else:
///         print(f"{q}: top-1 = {hits[0]['content']}")
/// ```
#[pyfunction]
#[pyo3(name = "retrieve_concurrent", signature = (retriever, queries, k=10, max_concurrency=8, fail_fast=false))]
fn py_retrieve_concurrent<'py>(
    py: Python<'py>,
    retriever: Bound<'py, PyAny>,
    queries: Vec<String>,
    k: usize,
    max_concurrency: usize,
    fail_fast: bool,
) -> PyResult<Bound<'py, PyList>> {
    let r_arc = extract_retriever_arc(&retriever)?;
    let results = py.allow_threads(|| {
        block_on_compat(async move {
            Ok::<_, litgraph_core::Error>(
                retrieve_concurrent(r_arc, queries, k, max_concurrency).await,
            )
        })
        .map_err(|e: litgraph_core::Error| PyRuntimeError::new_err(e.to_string()))
    })?;

    let out = PyList::empty_bound(py);
    for r in results {
        match r {
            Ok(docs) => out.append(docs_to_pylist(py, docs)?)?,
            Err(e) => {
                if fail_fast {
                    return Err(PyRuntimeError::new_err(e.to_string()));
                }
                let d = PyDict::new_bound(py);
                d.set_item("error", e.to_string())?;
                out.append(d)?;
            }
        }
    }
    Ok(out)
}
