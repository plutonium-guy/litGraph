//! Python bindings for retrieval — BM25 for now; vector stores later.

use std::sync::Arc;

use litgraph_core::Document;
use litgraph_retrieval::store::VectorStore;
use litgraph_retrieval::{
    Bm25Index, HybridRetriever, Reranker, RerankingRetriever, Retriever, VectorRetriever,
};
use litgraph_rerankers_cohere::{CohereConfig, CohereReranker};
use litgraph_rerankers_jina::{JinaRerankConfig, JinaReranker};
use litgraph_rerankers_voyage::{VoyageRerankConfig, VoyageReranker};
use litgraph_stores_chroma::{ChromaConfig, ChromaVectorStore};
use litgraph_stores_hnsw::HnswVectorStore;
use litgraph_stores_memory::MemoryVectorStore;
use litgraph_stores_pgvector::PgVectorStore;
use litgraph_stores_qdrant::{QdrantConfig, QdrantVectorStore};
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
    m.add_class::<PyPgVectorStore>()?;
    m.add_class::<PyChromaVectorStore>()?;
    m.add_class::<PyVectorRetriever>()?;
    m.add_class::<PyCohereReranker>()?;
    m.add_class::<PyVoyageReranker>()?;
    m.add_class::<PyJinaReranker>()?;
    m.add_class::<PyRerankingRetriever>()?;
    m.add_class::<PyHybridRetriever>()?;
    m.add_function(pyo3::wrap_pyfunction!(evaluate_retrieval, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(evaluate_generation, m)?)?;
    Ok(())
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
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "store must be MemoryVectorStore, HnswVectorStore, QdrantVectorStore, PgVectorStore, or ChromaVectorStore",
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
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "reranker must be CohereReranker, VoyageReranker, or JinaReranker",
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
    inner: Arc<HybridRetriever>,
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
