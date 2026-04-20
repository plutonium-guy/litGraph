//! Python `Embeddings` provider. Users give a Python callable
//! `list[str] -> list[list[float]]`; we wrap it in an `Embeddings` implementation
//! for use with `VectorRetriever` + any `VectorStore`.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{Embeddings, Result as LgResult};
use litgraph_providers_bedrock::{
    AwsCredentials, BedrockEmbedFormat, BedrockEmbeddings, BedrockEmbeddingsConfig,
};
use litgraph_providers_cohere::{CohereEmbeddings, CohereEmbeddingsConfig};
use litgraph_providers_gemini::{GeminiEmbeddings, GeminiEmbeddingsConfig};
use litgraph_providers_jina::{JinaEmbeddings, JinaEmbeddingsConfig};
use litgraph_providers_openai::{OpenAIEmbeddings, OpenAIEmbeddingsConfig};
use litgraph_providers_voyage::{VoyageEmbeddings, VoyageEmbeddingsConfig};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;

use crate::runtime::block_on_compat;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFunctionEmbeddings>()?;
    m.add_class::<PyOpenAIEmbeddings>()?;
    m.add_class::<PyCohereEmbeddings>()?;
    m.add_class::<PyVoyageEmbeddings>()?;
    m.add_class::<PyGeminiEmbeddings>()?;
    m.add_class::<PyBedrockEmbeddings>()?;
    m.add_class::<PyJinaEmbeddings>()?;
    m.add_function(wrap_pyfunction!(tei_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(together_embeddings, m)?)?;
    Ok(())
}

/// HuggingFace TEI (Text Embeddings Inference) — self-hostable embeddings
/// server with an OpenAI-compatible `/embeddings` endpoint. Pass the URL of
/// the running TEI instance. No api_key needed by default; set one only if
/// you have an auth proxy in front.
#[pyfunction]
#[pyo3(signature = (base_url, dimensions, model="tei", api_key="", timeout_s=120))]
fn tei_embeddings(
    base_url: String,
    dimensions: usize,
    model: &str,
    api_key: &str,
    timeout_s: u64,
) -> PyResult<PyOpenAIEmbeddings> {
    PyOpenAIEmbeddings::new(
        api_key.to_string(),
        model.to_string(),
        dimensions,
        Some(base_url),
        timeout_s,
        None,
    )
}

/// Together AI embeddings via the OpenAI-compatible endpoint.
/// Default base_url: `https://api.together.xyz/v1`.
#[pyfunction]
#[pyo3(signature = (api_key, model, dimensions, base_url=None, timeout_s=120))]
fn together_embeddings(
    api_key: String,
    model: String,
    dimensions: usize,
    base_url: Option<String>,
    timeout_s: u64,
) -> PyResult<PyOpenAIEmbeddings> {
    let url = base_url.unwrap_or_else(|| "https://api.together.xyz/v1".into());
    PyOpenAIEmbeddings::new(api_key, model, dimensions, Some(url), timeout_s, None)
}

/// Native Jina AI Embeddings provider (`/v1/embeddings`). Task-aware
/// retrieval: defaults to `retrieval.passage` for documents and
/// `retrieval.query` for queries (other tasks: `text-matching`,
/// `classification`, `separation`). Optional `output_dimensions` for
/// Matryoshka truncation.
#[pyclass(name = "JinaEmbeddings", module = "litgraph.embeddings")]
#[derive(Clone)]
pub struct PyJinaEmbeddings {
    pub(crate) inner: Arc<JinaEmbeddings>,
    model: String,
    dim: usize,
}

#[pymethods]
impl PyJinaEmbeddings {
    #[new]
    #[pyo3(signature = (
        api_key, model, dimensions, base_url=None, timeout_s=120,
        task_document=Some("retrieval.passage".to_string()),
        task_query=Some("retrieval.query".to_string()),
        output_dimensions=None,
    ))]
    fn new(
        api_key: String,
        model: String,
        dimensions: usize,
        base_url: Option<String>,
        timeout_s: u64,
        task_document: Option<String>,
        task_query: Option<String>,
        output_dimensions: Option<usize>,
    ) -> PyResult<Self> {
        let mut cfg = JinaEmbeddingsConfig::new(api_key, &model, dimensions);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        cfg.task_document = task_document;
        cfg.task_query = task_query;
        if let Some(d) = output_dimensions { cfg = cfg.with_output_dimensions(d); }
        let dim = cfg.dimensions;
        let emb = JinaEmbeddings::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(emb), model, dim })
    }

    #[getter] fn name(&self) -> &str { &self.model }
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
        format!("JinaEmbeddings(model='{}', dimensions={})", self.model, self.dim)
    }
}

impl PyJinaEmbeddings {
    pub(crate) fn as_embeddings(&self) -> Arc<dyn Embeddings> {
        self.inner.clone() as Arc<dyn Embeddings>
    }
}

/// Wraps a Python callable as an `Embeddings` provider. The callable receives a
/// `list[str]` and must return a `list[list[float]]` of matching length.
#[pyclass(name = "FunctionEmbeddings", module = "litgraph.embeddings")]
#[derive(Clone)]
pub struct PyFunctionEmbeddings {
    pub(crate) inner: Arc<FunctionEmbeddingsImpl>,
}

pub(crate) struct FunctionEmbeddingsImpl {
    pub name: String,
    pub dimensions: usize,
    pub func: Py<PyAny>,
}

#[pymethods]
impl PyFunctionEmbeddings {
    #[new]
    #[pyo3(signature = (func, dimensions, name="custom"))]
    fn new(func: Py<PyAny>, dimensions: usize, name: &str) -> Self {
        Self {
            inner: Arc::new(FunctionEmbeddingsImpl {
                name: name.into(),
                dimensions,
                func,
            }),
        }
    }

    #[getter]
    fn name(&self) -> &str { &self.inner.name }

    #[getter]
    fn dimensions(&self) -> usize { self.inner.dimensions }

    /// Embed a single query — used by `VectorRetriever`. Useful to test.
    fn embed_query<'py>(&self, py: Python<'py>, text: String) -> PyResult<Vec<f32>> {
        let texts = PyList::new_bound(py, &[text]);
        let out = self.inner.func.call1(py, (texts,))?;
        let bound = out.bind(py);
        let first = bound.downcast::<PyList>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("expected list[list[float]]"))?
            .get_item(0)?;
        first.extract::<Vec<f32>>()
    }

    fn __repr__(&self) -> String {
        format!(
            "FunctionEmbeddings(name='{}', dimensions={})",
            self.inner.name, self.inner.dimensions
        )
    }
}

#[async_trait]
impl Embeddings for FunctionEmbeddingsImpl {
    fn name(&self) -> &str { &self.name }
    fn dimensions(&self) -> usize { self.dimensions }

    async fn embed_query(&self, text: &str) -> LgResult<Vec<f32>> {
        let text = text.to_string();
        let out: Result<Vec<f32>, String> = Python::with_gil(|py| {
            let list = PyList::new_bound(py, &[text]);
            let ret = self.func.call1(py, (list,)).map_err(|e| e.to_string())?;
            let bound = ret.bind(py);
            let outer = bound.downcast::<PyList>()
                .map_err(|_| "expected list[list[float]]".to_string())?;
            let first = outer.get_item(0).map_err(|e| e.to_string())?;
            first.extract::<Vec<f32>>().map_err(|e| e.to_string())
        });
        out.map_err(|e| litgraph_core::Error::other(format!("embed_query: {e}")))
    }

    async fn embed_documents(&self, texts: &[String]) -> LgResult<Vec<Vec<f32>>> {
        let texts = texts.to_vec();
        let out: Result<Vec<Vec<f32>>, String> = Python::with_gil(|py| {
            let list = PyList::new_bound(py, &texts);
            let ret = self.func.call1(py, (list,)).map_err(|e| e.to_string())?;
            let bound = ret.bind(py);
            let outer = bound.downcast::<PyList>()
                .map_err(|_| "expected list[list[float]]".to_string())?;
            let mut collected = Vec::with_capacity(outer.len());
            for item in outer.iter() {
                collected.push(item.extract::<Vec<f32>>().map_err(|e| e.to_string())?);
            }
            Ok(collected)
        });
        out.map_err(|e| litgraph_core::Error::other(format!("embed_documents: {e}")))
    }
}

impl PyFunctionEmbeddings {
    pub(crate) fn as_embeddings(&self) -> Arc<dyn Embeddings> {
        self.inner.clone() as Arc<dyn Embeddings>
    }
}

/// Native OpenAI Embeddings provider. Hits `/embeddings` directly — no Python
/// callable round-trip per query. Pass `dimensions` so VectorStore sizing
/// works without an extra warm-up call.
#[pyclass(name = "OpenAIEmbeddings", module = "litgraph.embeddings")]
#[derive(Clone)]
pub struct PyOpenAIEmbeddings {
    pub(crate) inner: Arc<OpenAIEmbeddings>,
    model: String,
    dim: usize,
}

#[pymethods]
impl PyOpenAIEmbeddings {
    /// `override_dimensions` only applies to text-embedding-3+ models that
    /// support truncation server-side. When set, the server returns vectors
    /// of that length and `dimensions` is updated to match.
    #[new]
    #[pyo3(signature = (
        api_key, model, dimensions, base_url=None, timeout_s=120, override_dimensions=None,
    ))]
    fn new(
        api_key: String,
        model: String,
        dimensions: usize,
        base_url: Option<String>,
        timeout_s: u64,
        override_dimensions: Option<usize>,
    ) -> PyResult<Self> {
        let mut cfg = OpenAIEmbeddingsConfig::new(api_key, &model, dimensions);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        if let Some(d) = override_dimensions { cfg = cfg.with_override_dimensions(d); }
        let dim = cfg.dimensions;
        let emb = OpenAIEmbeddings::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(emb), model, dim })
    }

    #[getter]
    fn name(&self) -> &str { &self.model }

    #[getter]
    fn dimensions(&self) -> usize { self.dim }

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
        format!("OpenAIEmbeddings(model='{}', dimensions={})", self.model, self.dim)
    }
}

impl PyOpenAIEmbeddings {
    pub(crate) fn as_embeddings(&self) -> Arc<dyn Embeddings> {
        self.inner.clone() as Arc<dyn Embeddings>
    }
}

/// Native Cohere Embeddings provider (`/v2/embed`). Cohere requires a per-request
/// `input_type`; we send `search_document` for `embed_documents` and
/// `search_query` for `embed_query`. Override via constructor kwargs.
#[pyclass(name = "CohereEmbeddings", module = "litgraph.embeddings")]
#[derive(Clone)]
pub struct PyCohereEmbeddings {
    pub(crate) inner: Arc<CohereEmbeddings>,
    model: String,
    dim: usize,
}

#[pymethods]
impl PyCohereEmbeddings {
    #[new]
    #[pyo3(signature = (
        api_key, model, dimensions, base_url=None, timeout_s=120,
        input_type_document=None, input_type_query=None,
    ))]
    fn new(
        api_key: String,
        model: String,
        dimensions: usize,
        base_url: Option<String>,
        timeout_s: u64,
        input_type_document: Option<String>,
        input_type_query: Option<String>,
    ) -> PyResult<Self> {
        let mut cfg = CohereEmbeddingsConfig::new(api_key, &model, dimensions);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        if let Some(d) = input_type_document {
            cfg.input_type_document = d;
        }
        if let Some(q) = input_type_query {
            cfg.input_type_query = q;
        }
        let dim = cfg.dimensions;
        let emb = CohereEmbeddings::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(emb), model, dim })
    }

    #[getter]
    fn name(&self) -> &str { &self.model }

    #[getter]
    fn dimensions(&self) -> usize { self.dim }

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
        format!("CohereEmbeddings(model='{}', dimensions={})", self.model, self.dim)
    }
}

impl PyCohereEmbeddings {
    pub(crate) fn as_embeddings(&self) -> Arc<dyn Embeddings> {
        self.inner.clone() as Arc<dyn Embeddings>
    }
}

/// Native Voyage AI Embeddings provider (`/v1/embeddings`). Voyage's RAG-tuned
/// models (voyage-3, voyage-3-large, voyage-code-3) are Anthropic's recommended
/// embedder for Claude RAG. Defaults: `input_type_query="query"`,
/// `input_type_document="document"`. Pass `input_type_*=None` to omit the
/// field (Voyage default = generic).
#[pyclass(name = "VoyageEmbeddings", module = "litgraph.embeddings")]
#[derive(Clone)]
pub struct PyVoyageEmbeddings {
    pub(crate) inner: Arc<VoyageEmbeddings>,
    model: String,
    dim: usize,
}

#[pymethods]
impl PyVoyageEmbeddings {
    #[new]
    #[pyo3(signature = (
        api_key, model, dimensions, base_url=None, timeout_s=120,
        input_type_document=Some("document".to_string()),
        input_type_query=Some("query".to_string()),
    ))]
    fn new(
        api_key: String,
        model: String,
        dimensions: usize,
        base_url: Option<String>,
        timeout_s: u64,
        input_type_document: Option<String>,
        input_type_query: Option<String>,
    ) -> PyResult<Self> {
        let mut cfg = VoyageEmbeddingsConfig::new(api_key, &model, dimensions);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        cfg.input_type_document = input_type_document;
        cfg.input_type_query = input_type_query;
        let dim = cfg.dimensions;
        let emb = VoyageEmbeddings::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(emb), model, dim })
    }

    #[getter]
    fn name(&self) -> &str { &self.model }

    #[getter]
    fn dimensions(&self) -> usize { self.dim }

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
        format!("VoyageEmbeddings(model='{}', dimensions={})", self.model, self.dim)
    }
}

impl PyVoyageEmbeddings {
    pub(crate) fn as_embeddings(&self) -> Arc<dyn Embeddings> {
        self.inner.clone() as Arc<dyn Embeddings>
    }
}

/// Native Gemini Embeddings provider (`/v1beta/models/{model}:batchEmbedContents`).
/// Per-request `task_type` (RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT /
/// SEMANTIC_SIMILARITY / CLASSIFICATION / CLUSTERING / QUESTION_ANSWERING /
/// FACT_VERIFICATION). Defaults follow RAG: query→RETRIEVAL_QUERY,
/// documents→RETRIEVAL_DOCUMENT.
#[pyclass(name = "GeminiEmbeddings", module = "litgraph.embeddings")]
#[derive(Clone)]
pub struct PyGeminiEmbeddings {
    pub(crate) inner: Arc<GeminiEmbeddings>,
    model: String,
    dim: usize,
}

#[pymethods]
impl PyGeminiEmbeddings {
    #[new]
    #[pyo3(signature = (
        api_key, model, dimensions, base_url=None, timeout_s=120,
        task_type_document=Some("RETRIEVAL_DOCUMENT".to_string()),
        task_type_query=Some("RETRIEVAL_QUERY".to_string()),
        output_dimensionality=None,
    ))]
    fn new(
        api_key: String,
        model: String,
        dimensions: usize,
        base_url: Option<String>,
        timeout_s: u64,
        task_type_document: Option<String>,
        task_type_query: Option<String>,
        output_dimensionality: Option<usize>,
    ) -> PyResult<Self> {
        let mut cfg = GeminiEmbeddingsConfig::new(api_key, &model, dimensions);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        cfg.task_type_document = task_type_document;
        cfg.task_type_query = task_type_query;
        if let Some(d) = output_dimensionality { cfg = cfg.with_output_dimensionality(d); }
        let dim = cfg.dimensions;
        let emb = GeminiEmbeddings::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(emb), model, dim })
    }

    #[getter]
    fn name(&self) -> &str { &self.model }

    #[getter]
    fn dimensions(&self) -> usize { self.dim }

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
        format!("GeminiEmbeddings(model='{}', dimensions={})", self.model, self.dim)
    }
}

impl PyGeminiEmbeddings {
    pub(crate) fn as_embeddings(&self) -> Arc<dyn Embeddings> {
        self.inner.clone() as Arc<dyn Embeddings>
    }
}

/// Native Bedrock Embeddings provider — supports the Titan Embed family
/// (`amazon.titan-embed-text-v2:0`, etc.) and Cohere-on-Bedrock
/// (`cohere.embed-*`). Wire format auto-detected from the model id; pass
/// `format="titan"` or `format="cohere"` to override.
///
/// Titan endpoints take a single `inputText` per call, so `embed_documents`
/// fans the batch out across `max_concurrency` parallel HTTP requests
/// (`buffered()` to preserve order). Cohere-on-Bedrock natively batches
/// `texts: [...]` so the call is a single round-trip.
#[pyclass(name = "BedrockEmbeddings", module = "litgraph.embeddings")]
#[derive(Clone)]
pub struct PyBedrockEmbeddings {
    pub(crate) inner: Arc<BedrockEmbeddings>,
    model: String,
    dim: usize,
}

#[pymethods]
impl PyBedrockEmbeddings {
    #[new]
    #[pyo3(signature = (
        access_key_id, secret_access_key, region, model_id, dimensions,
        session_token=None, timeout_s=120, endpoint_override=None,
        format=None, max_concurrency=8, normalize=true,
        cohere_input_type_document=None, cohere_input_type_query=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        access_key_id: String,
        secret_access_key: String,
        region: String,
        model_id: String,
        dimensions: usize,
        session_token: Option<String>,
        timeout_s: u64,
        endpoint_override: Option<String>,
        format: Option<String>,
        max_concurrency: usize,
        normalize: bool,
        cohere_input_type_document: Option<String>,
        cohere_input_type_query: Option<String>,
    ) -> PyResult<Self> {
        let creds = AwsCredentials { access_key_id, secret_access_key, session_token };
        let mut cfg = BedrockEmbeddingsConfig::new(creds, region, &model_id, dimensions);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = endpoint_override { cfg = cfg.with_endpoint(url); }
        if let Some(fmt) = format {
            let f = match fmt.to_ascii_lowercase().as_str() {
                "titan" => BedrockEmbedFormat::Titan,
                "cohere" => BedrockEmbedFormat::Cohere,
                other => return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("format must be 'titan' or 'cohere', got '{other}'"))),
            };
            cfg = cfg.with_format(f);
        }
        cfg = cfg.with_max_concurrency(max_concurrency);
        cfg.normalize = normalize;
        if let Some(d) = cohere_input_type_document {
            cfg.cohere_input_type_document = d;
        }
        if let Some(q) = cohere_input_type_query {
            cfg.cohere_input_type_query = q;
        }
        let dim = cfg.dimensions;
        let emb = BedrockEmbeddings::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(emb), model: model_id, dim })
    }

    #[getter]
    fn name(&self) -> &str { &self.model }

    #[getter]
    fn dimensions(&self) -> usize { self.dim }

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
        format!("BedrockEmbeddings(model='{}', dimensions={})", self.model, self.dim)
    }
}

impl PyBedrockEmbeddings {
    pub(crate) fn as_embeddings(&self) -> Arc<dyn Embeddings> {
        self.inner.clone() as Arc<dyn Embeddings>
    }
}

