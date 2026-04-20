//! Python bindings for cache backends. `CachedModel` wrapping is done on the
//! Python side of provider classes once they support instrumentation; this
//! module exposes just the backends for now.

use std::sync::Arc;

use litgraph_cache::{Cache, MemoryCache, SemanticCache, SqliteCache};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::embeddings::PyFunctionEmbeddings;
use crate::runtime::block_on_compat;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoryCache>()?;
    m.add_class::<PySqliteCache>()?;
    m.add_class::<PySemanticCache>()?;
    Ok(())
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
