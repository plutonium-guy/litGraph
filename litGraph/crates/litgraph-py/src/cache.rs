//! Python bindings for cache backends. `CachedModel` wrapping is done on the
//! Python side of provider classes once they support instrumentation; this
//! module exposes just the backends for now.

use std::sync::Arc;

use litgraph_cache::{
    Cache, CachedEmbeddings, EmbeddingCache, MemoryCache, MemoryEmbeddingCache, SemanticCache,
    SqliteCache, SqliteEmbeddingCache,
};
use litgraph_core::Embeddings;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::embeddings::{
    PyBedrockEmbeddings, PyCohereEmbeddings, PyFunctionEmbeddings, PyGeminiEmbeddings,
    PyJinaEmbeddings, PyOpenAIEmbeddings, PyVoyageEmbeddings,
};
use crate::runtime::block_on_compat;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoryCache>()?;
    m.add_class::<PySqliteCache>()?;
    m.add_class::<PySemanticCache>()?;
    m.add_class::<PyMemoryEmbeddingCache>()?;
    m.add_class::<PySqliteEmbeddingCache>()?;
    m.add_class::<PyCachedEmbeddings>()?;
    Ok(())
}

/// SQLite-backed embedding cache. Durable across process restarts — indexing
/// jobs that span multiple runs get the full cost savings. Pair with
/// `CachedEmbeddings` to wrap any embeddings provider.
///
/// `path` = file path (creates+opens with WAL journal); use `in_memory()`
/// staticmethod for ephemeral testing. Same `EmbeddingCache` trait as
/// `MemoryEmbeddingCache`, so they're interchangeable behind `CachedEmbeddings`.
#[pyclass(name = "SqliteEmbeddingCache", module = "litgraph.cache")]
#[derive(Clone)]
pub struct PySqliteEmbeddingCache {
    pub(crate) inner: Arc<dyn EmbeddingCache>,
}

#[pymethods]
impl PySqliteEmbeddingCache {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let cache = SqliteEmbeddingCache::open(&path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(cache) })
    }

    /// Open an in-memory SQLite cache — for tests. Not durable.
    #[staticmethod]
    fn in_memory() -> PyResult<Self> {
        let cache = SqliteEmbeddingCache::in_memory()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(cache) })
    }

    fn clear<'py>(&self, py: Python<'py>) -> PyResult<()> {
        let cache = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { cache.clear().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> String { "SqliteEmbeddingCache()".into() }
}

/// In-memory LRU cache for embedding vectors. TTL optional. Pair with
/// `CachedEmbeddings` to wrap any embeddings provider with read-through
/// caching keyed on `(model, text)`.
#[pyclass(name = "MemoryEmbeddingCache", module = "litgraph.cache")]
#[derive(Clone)]
pub struct PyMemoryEmbeddingCache {
    pub(crate) inner: Arc<dyn EmbeddingCache>,
}

#[pymethods]
impl PyMemoryEmbeddingCache {
    #[new]
    #[pyo3(signature = (max_capacity=100_000, ttl_s=None))]
    fn new(max_capacity: u64, ttl_s: Option<u64>) -> Self {
        let inner: Arc<dyn EmbeddingCache> = match ttl_s {
            Some(s) => Arc::new(MemoryEmbeddingCache::with_ttl(
                max_capacity, std::time::Duration::from_secs(s)
            )),
            None => Arc::new(MemoryEmbeddingCache::new(max_capacity)),
        };
        Self { inner }
    }

    fn clear<'py>(&self, py: Python<'py>) -> PyResult<()> {
        let cache = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { cache.clear().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> String { "MemoryEmbeddingCache()".into() }
}

/// Wraps any embeddings provider with read-through caching. Cache is keyed
/// on `(provider.name, text)` so swapping providers cleanly invalidates.
/// `embed_documents` only sends MISSES to the inner provider — agents and
/// indexing pipelines reliably re-embed corpora that mostly didn't change.
#[pyclass(name = "CachedEmbeddings", module = "litgraph.cache")]
#[derive(Clone)]
pub struct PyCachedEmbeddings {
    pub(crate) inner: Arc<CachedEmbeddings>,
    /// Cached for fast getter access; matches the inner provider's name.
    name: String,
    dim: usize,
}

#[pymethods]
impl PyCachedEmbeddings {
    /// `embeddings` is any of the 7 embeddings providers (OpenAI/Cohere/
    /// Voyage/Gemini/Bedrock/Jina/Function). `cache` is either
    /// `MemoryEmbeddingCache` (ephemeral) or `SqliteEmbeddingCache` (durable).
    #[new]
    fn new(embeddings: Bound<'_, PyAny>, cache: Bound<'_, PyAny>) -> PyResult<Self> {
        let cache_arc: Arc<dyn EmbeddingCache> =
            if let Ok(mem) = cache.extract::<PyRef<PyMemoryEmbeddingCache>>() {
                mem.inner.clone()
            } else if let Ok(sql) = cache.extract::<PyRef<PySqliteEmbeddingCache>>() {
                sql.inner.clone()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "cache must be MemoryEmbeddingCache or SqliteEmbeddingCache",
                ));
            };
        let inner_emb: Arc<dyn Embeddings> =
            if let Ok(fe) = embeddings.extract::<PyRef<PyFunctionEmbeddings>>() {
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
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "embeddings must be FunctionEmbeddings, OpenAIEmbeddings, CohereEmbeddings, VoyageEmbeddings, GeminiEmbeddings, BedrockEmbeddings, or JinaEmbeddings",
                ));
            };
        let name = inner_emb.name().to_string();
        let dim = inner_emb.dimensions();
        let wrapped = CachedEmbeddings::new(inner_emb, cache_arc);
        Ok(Self { inner: Arc::new(wrapped), name, dim })
    }

    #[getter] fn name(&self) -> &str { &self.name }
    #[getter] fn dimensions(&self) -> usize { self.dim }

    fn embed_query<'py>(&self, py: Python<'py>, text: String) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.embed_query(&text).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn embed_documents<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.embed_documents(&texts).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> String {
        format!("CachedEmbeddings(model='{}', dimensions={})", self.name, self.dim)
    }
}

impl PyCachedEmbeddings {
    pub(crate) fn as_embeddings(&self) -> Arc<dyn Embeddings> {
        self.inner.clone() as Arc<dyn Embeddings>
    }
}

#[pyclass(name = "MemoryCache", module = "litgraph.cache")]
pub struct PyMemoryCache {
    pub(crate) inner: Arc<dyn Cache>,
}

#[pymethods]
impl PyMemoryCache {
    #[new]
    #[pyo3(signature = (max_capacity=10_000))]
    fn new(max_capacity: u64) -> Self {
        Self { inner: Arc::new(MemoryCache::new(max_capacity)) }
    }

    fn clear(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.clear().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> String { "MemoryCache(...)".into() }
}

#[pyclass(name = "SqliteCache", module = "litgraph.cache")]
pub struct PySqliteCache {
    pub(crate) inner: Arc<dyn Cache>,
}

#[pymethods]
impl PySqliteCache {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let inner = SqliteCache::open(&path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(inner) })
    }

    #[staticmethod]
    fn in_memory() -> PyResult<Self> {
        let inner = SqliteCache::in_memory()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(inner) })
    }

    fn clear(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.clear().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> String { "SqliteCache(...)".into() }
}

/// Semantic cache — looks up cached responses by embedding cosine similarity.
/// Use a conservative `threshold` (≥ 0.95) in production. Not yet wired into
/// provider `.with_cache()`; use via the Rust API or store/fetch manually.
#[pyclass(name = "SemanticCache", module = "litgraph.cache")]
pub struct PySemanticCache {
    #[allow(dead_code)]
    pub(crate) inner: Arc<SemanticCache>,
}

#[pymethods]
impl PySemanticCache {
    #[new]
    #[pyo3(signature = (embeddings, threshold=0.95, max_entries=10_000))]
    fn new(embeddings: PyFunctionEmbeddings, threshold: f32, max_entries: usize) -> Self {
        let inner = SemanticCache::new(embeddings.as_embeddings(), threshold)
            .with_max_entries(max_entries);
        Self { inner: Arc::new(inner) }
    }

    fn __len__(&self) -> usize { self.inner.len() }

    fn clear(&self) { self.inner.clear() }

    fn __repr__(&self) -> String {
        format!("SemanticCache(entries={})", self.inner.len())
    }
}
