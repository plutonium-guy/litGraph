//! Python bindings for text splitters.

use std::sync::Arc;

use litgraph_core::{Document, Embeddings};
use litgraph_splitters::{
    HtmlHeaderSplitter, JsonSplitter, Language, MarkdownHeaderSplitter, RecursiveCharacterSplitter,
    SemanticChunker, Splitter,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::embeddings::{
    PyBedrockEmbeddings, PyCohereEmbeddings, PyFunctionEmbeddings, PyGeminiEmbeddings,
    PyOpenAIEmbeddings, PyVoyageEmbeddings,
};
use crate::retrieval::{docs_to_pylist, parse_docs};
use crate::runtime::block_on_compat;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRecursiveSplitter>()?;
    m.add_class::<PyMarkdownSplitter>()?;
    m.add_class::<PySemanticChunker>()?;
    m.add_class::<PyJsonSplitter>()?;
    m.add_class::<PyHtmlHeaderSplitter>()?;
    Ok(())
}

/// HTML header-aware splitter — emits one chunk per `<h1..h6>` section
/// (depth bounded by `max_depth`). Each chunk carries its heading
/// breadcrumb in metadata (`h1`, `h2`, …). Mirrors `MarkdownHeaderSplitter`.
#[pyclass(name = "HtmlHeaderSplitter", module = "litgraph.splitters")]
pub struct PyHtmlHeaderSplitter {
    inner: HtmlHeaderSplitter,
}

#[pymethods]
impl PyHtmlHeaderSplitter {
    #[new]
    #[pyo3(signature = (max_depth=3, strip_headers=false))]
    fn new(max_depth: u8, strip_headers: bool) -> Self {
        let inner = HtmlHeaderSplitter::new(max_depth).strip_headers(strip_headers);
        Self { inner }
    }

    fn split_text(&self, py: Python<'_>, text: String) -> Vec<String> {
        py.allow_threads(|| self.inner.split_text(&text))
    }

    fn split_documents<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        let parsed: Vec<Document> = parse_docs(&docs)?;
        let chunks = py.allow_threads(|| self.inner.split_documents(&parsed));
        docs_to_pylist(py, chunks)
    }

    fn __repr__(&self) -> String {
        format!(
            "HtmlHeaderSplitter(max_depth={}, strip_headers={})",
            self.inner.max_depth, self.inner.strip_headers
        )
    }
}

/// Recursive JSON splitter — keeps each chunk under `max_chunk_size` while
/// preserving structural validity (each chunk is parseable JSON) and
/// path context (chunks carry a `_path` field pointing back to their
/// originating subtree). Mirrors LangChain's `RecursiveJsonSplitter`.
#[pyclass(name = "JsonSplitter", module = "litgraph.splitters")]
pub struct PyJsonSplitter {
    inner: JsonSplitter,
}

#[pymethods]
impl PyJsonSplitter {
    #[new]
    #[pyo3(signature = (max_chunk_size=2000))]
    fn new(max_chunk_size: usize) -> Self {
        Self { inner: JsonSplitter::new(max_chunk_size) }
    }

    /// Split a JSON-serialized string into a list of valid JSON-serialized
    /// chunks. Invalid JSON inputs are returned unchanged as a single chunk.
    fn split_text(&self, py: Python<'_>, text: String) -> Vec<String> {
        py.allow_threads(|| self.inner.split_text(&text))
    }

    fn __repr__(&self) -> String {
        format!("JsonSplitter(max_chunk_size={})", self.inner.max_chunk_size)
    }
}

#[pyclass(name = "RecursiveCharacterSplitter", module = "litgraph.splitters")]
pub struct PyRecursiveSplitter {
    inner: RecursiveCharacterSplitter,
}

#[pymethods]
impl PyRecursiveSplitter {
    #[new]
    #[pyo3(signature = (chunk_size=1000, chunk_overlap=200))]
    fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self { inner: RecursiveCharacterSplitter::new(chunk_size, chunk_overlap) }
    }

    /// Construct a splitter with separator priorities tuned for a programming
    /// language. Pass the language name as a string — accepts (case-insensitive):
    /// `python`, `rust`, `javascript`, `typescript`, `go`, `java`, `cpp`,
    /// `ruby`, `php`, `markdown`, `html`. Mirrors LangChain's
    /// `RecursiveCharacterTextSplitter.from_language()`.
    #[staticmethod]
    #[pyo3(signature = (language, chunk_size=1000, chunk_overlap=200))]
    fn for_language(language: &str, chunk_size: usize, chunk_overlap: usize) -> PyResult<Self> {
        let lang = match language.to_ascii_lowercase().as_str() {
            "python" | "py" => Language::Python,
            "rust" | "rs" => Language::Rust,
            "javascript" | "js" => Language::JavaScript,
            "typescript" | "ts" => Language::TypeScript,
            "go" | "golang" => Language::Go,
            "java" => Language::Java,
            "cpp" | "c++" => Language::Cpp,
            "ruby" | "rb" => Language::Ruby,
            "php" => Language::Php,
            "markdown" | "md" => Language::Markdown,
            "html" | "htm" => Language::Html,
            other => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("unknown language: '{other}' (supported: python, rust, javascript, typescript, go, java, cpp, ruby, php, markdown, html)")
            )),
        };
        Ok(Self { inner: RecursiveCharacterSplitter::for_language(chunk_size, chunk_overlap, lang) })
    }

    /// Split a single string into chunks.
    fn split_text(&self, py: Python<'_>, text: String) -> Vec<String> {
        py.allow_threads(|| self.inner.split_text(&text))
    }

    /// Split a list of doc dicts, returning a list of chunk dicts (metadata carried).
    fn split_documents<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        let parsed: Vec<Document> = parse_docs(&docs)?;
        let chunks = py.allow_threads(|| self.inner.split_documents(&parsed));
        docs_to_pylist(py, chunks)
    }

    fn __repr__(&self) -> String {
        format!(
            "RecursiveCharacterSplitter(chunk_size={}, chunk_overlap={})",
            self.inner.chunk_size, self.inner.chunk_overlap
        )
    }
}

#[pyclass(name = "MarkdownHeaderSplitter", module = "litgraph.splitters")]
pub struct PyMarkdownSplitter {
    inner: MarkdownHeaderSplitter,
}

#[pymethods]
impl PyMarkdownSplitter {
    #[new]
    #[pyo3(signature = (max_depth=3, strip_headers=false))]
    fn new(max_depth: u8, strip_headers: bool) -> Self {
        let inner = MarkdownHeaderSplitter::new(max_depth).strip_headers(strip_headers);
        Self { inner }
    }

    fn split_text(&self, py: Python<'_>, text: String) -> Vec<String> {
        py.allow_threads(|| self.inner.split_text(&text))
    }

    fn split_documents<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        let parsed: Vec<Document> = parse_docs(&docs)?;
        let chunks = py.allow_threads(|| self.inner.split_documents(&parsed));
        docs_to_pylist(py, chunks)
    }

    fn __repr__(&self) -> String {
        format!(
            "MarkdownHeaderSplitter(max_depth={}, strip_headers={})",
            self.inner.max_depth, self.inner.strip_headers
        )
    }
}

/// Embedding-based semantic chunker (Greg Kamradt / LangChain experimental).
/// Splits at sentences whose cosine distance to the next sentence exceeds the
/// `breakpoint_percentile`th percentile of all consecutive distances —
/// adaptive per document, no global threshold to tune.
///
/// Accepts any embeddings provider (`FunctionEmbeddings` / `OpenAIEmbeddings`
/// / `CohereEmbeddings` / `VoyageEmbeddings` / `GeminiEmbeddings` /
/// `BedrockEmbeddings`).
#[pyclass(name = "SemanticChunker", module = "litgraph.splitters")]
pub struct PySemanticChunker {
    inner: SemanticChunker,
}

#[pymethods]
impl PySemanticChunker {
    #[new]
    #[pyo3(signature = (
        embeddings, buffer_size=1, breakpoint_percentile=95.0, min_sentences_per_chunk=1,
    ))]
    fn new(
        embeddings: Bound<'_, PyAny>,
        buffer_size: usize,
        breakpoint_percentile: f64,
        min_sentences_per_chunk: usize,
    ) -> PyResult<Self> {
        let e: Arc<dyn Embeddings> = if let Ok(fe) = embeddings.extract::<PyRef<PyFunctionEmbeddings>>() {
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
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "embeddings must be one of FunctionEmbeddings, OpenAIEmbeddings, CohereEmbeddings, VoyageEmbeddings, GeminiEmbeddings, or BedrockEmbeddings",
            ));
        };
        let inner = SemanticChunker::new(e)
            .with_buffer_size(buffer_size)
            .with_breakpoint_percentile(breakpoint_percentile)
            .with_min_sentences_per_chunk(min_sentences_per_chunk);
        Ok(Self { inner })
    }

    fn split_text(&self, py: Python<'_>, text: String) -> PyResult<Vec<String>> {
        let chunker = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { chunker.split_text(&text).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn split_documents<'py>(
        &self,
        py: Python<'py>,
        docs: Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        let parsed: Vec<Document> = parse_docs(&docs)?;
        let chunker = self.inner.clone();
        let chunks = py.allow_threads(move || {
            block_on_compat(async move {
                let mut all = Vec::new();
                for d in &parsed {
                    all.extend(chunker.split_document(d).await?);
                }
                Ok::<_, litgraph_core::Error>(all)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, chunks)
    }

    fn __repr__(&self) -> String {
        format!(
            "SemanticChunker(buffer_size={}, breakpoint_percentile={}, min_sentences_per_chunk={})",
            self.inner.buffer_size, self.inner.breakpoint_percentile, self.inner.min_sentences_per_chunk
        )
    }
}
