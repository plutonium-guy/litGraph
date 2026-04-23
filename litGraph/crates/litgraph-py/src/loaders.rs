//! Python bindings for loaders. All `load()` calls release the GIL and run the
//! rayon-parallel directory traversal on Rust threads.

use litgraph_loaders::{
    CsvLoader, DirectoryLoader, DocxLoader, HtmlLoader, JsonLinesLoader, JsonLoader, Loader,
    MarkdownLoader, PdfLoader, TextLoader, WebLoader, default_dispatcher,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::retrieval::docs_to_pylist;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTextLoader>()?;
    m.add_class::<PyJsonLinesLoader>()?;
    m.add_class::<PyMarkdownLoader>()?;
    m.add_class::<PyDirectoryLoader>()?;
    m.add_class::<PyWebLoader>()?;
    m.add_class::<PyCsvLoader>()?;
    m.add_class::<PyHtmlLoader>()?;
    m.add_class::<PyJsonLoader>()?;
    m.add_class::<PyPdfLoader>()?;
    m.add_class::<PyDocxLoader>()?;
    Ok(())
}

/// PDF text loader — pure-Rust via lopdf. `per_page=True` (default) yields
/// one Document per page with `metadata["page"]` 1-indexed; `per_page=False`
/// returns a single Document joined with form-feed (`\f`) separators.
///
/// ```python
/// from litgraph.loaders import PdfLoader
/// docs = PdfLoader("report.pdf").load()
/// # docs[0].metadata["page"] == 1, docs[0].metadata["source"] == "report.pdf"
/// ```
#[pyclass(name = "PdfLoader", module = "litgraph.loaders")]
pub struct PyPdfLoader { inner: PdfLoader }

#[pymethods]
impl PyPdfLoader {
    #[new]
    #[pyo3(signature = (path, per_page=true))]
    fn new(path: String, per_page: bool) -> Self {
        Self { inner: PdfLoader::new(path).with_per_page(per_page) }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

/// DOCX text loader — extracts the flowing text from `word/document.xml`.
/// Pure-Rust via `zip` + `quick-xml`; no native deps. Single Document per
/// file (DOCX has no first-class page concept). Paragraph breaks become
/// `\n`; explicit `<w:br/>` becomes `\n`; `<w:tab/>` becomes `\t`.
///
/// ```python
/// from litgraph.loaders import DocxLoader
/// docs = DocxLoader("memo.docx").load()
/// ```
#[pyclass(name = "DocxLoader", module = "litgraph.loaders")]
pub struct PyDocxLoader { inner: DocxLoader }

#[pymethods]
impl PyDocxLoader {
    #[new]
    fn new(path: String) -> Self {
        Self { inner: DocxLoader::new(path) }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

/// Single-file JSON loader. `pointer` (dot-separated, supports numeric array
/// indices) drills into a nested location. `content_field` selects a string
/// field from each record; if omitted, content = serialized JSON.
#[pyclass(name = "JsonLoader", module = "litgraph.loaders")]
pub struct PyJsonLoader { inner: JsonLoader }

#[pymethods]
impl PyJsonLoader {
    #[new]
    #[pyo3(signature = (path, pointer=None, content_field=None))]
    fn new(path: String, pointer: Option<String>, content_field: Option<String>) -> Self {
        let mut inner = JsonLoader::new(path);
        if let Some(p) = pointer { inner = inner.with_pointer(p); }
        if let Some(f) = content_field { inner = inner.with_content_field(f); }
        Self { inner }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

/// HTML loader — strips script/style/comments + (by default) nav/header/
/// footer/aside, decodes common entities, returns plaintext. Pass either
/// `path=` for file-backed input or `html=` for an in-memory string. Set
/// `strip_boilerplate=False` to keep nav/header/footer.
#[pyclass(name = "HtmlLoader", module = "litgraph.loaders")]
pub struct PyHtmlLoader { inner: HtmlLoader }

#[pymethods]
impl PyHtmlLoader {
    #[new]
    #[pyo3(signature = (path=None, html=None, strip_boilerplate=true))]
    fn new(
        path: Option<String>,
        html: Option<String>,
        strip_boilerplate: bool,
    ) -> PyResult<Self> {
        let inner = match (path, html) {
            (Some(p), None) => HtmlLoader::new(p),
            (None, Some(s)) => HtmlLoader::from_string(s),
            (Some(_), Some(_)) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "pass exactly one of `path` or `html`, not both",
                ));
            }
            (None, None) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "must pass either `path` or `html`",
                ));
            }
        };
        let inner = if strip_boilerplate { inner } else { inner.keep_boilerplate() };
        Ok(Self { inner })
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

/// CSV / TSV / pipe-delimited loader. `content_column` selects the value to
/// embed; other columns become metadata. Pass `delimiter="\t"` for TSV.
#[pyclass(name = "CsvLoader", module = "litgraph.loaders")]
pub struct PyCsvLoader { inner: CsvLoader }

#[pymethods]
impl PyCsvLoader {
    #[new]
    #[pyo3(signature = (path, content_column=None, delimiter=",", max_rows=None))]
    fn new(
        path: String,
        content_column: Option<String>,
        delimiter: &str,
        max_rows: Option<usize>,
    ) -> PyResult<Self> {
        let bytes = delimiter.as_bytes();
        if bytes.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "delimiter must be a single byte (e.g. ',', '\\t', '|')",
            ));
        }
        let mut inner = CsvLoader::new(path).with_delimiter(bytes[0]);
        if let Some(c) = content_column { inner = inner.with_content_column(c); }
        if let Some(n) = max_rows { inner = inner.with_max_rows(n); }
        Ok(Self { inner })
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

#[pyclass(name = "WebLoader", module = "litgraph.loaders")]
pub struct PyWebLoader { inner: WebLoader }

#[pymethods]
impl PyWebLoader {
    #[new]
    #[pyo3(signature = (url, timeout_s=30, user_agent=None))]
    fn new(url: String, timeout_s: u64, user_agent: Option<String>) -> Self {
        let mut inner = WebLoader::new(url)
            .with_timeout(std::time::Duration::from_secs(timeout_s));
        if let Some(ua) = user_agent { inner = inner.with_user_agent(ua); }
        Self { inner }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

#[pyclass(name = "TextLoader", module = "litgraph.loaders")]
pub struct PyTextLoader { inner: TextLoader }

#[pymethods]
impl PyTextLoader {
    #[new]
    fn new(path: String) -> Self { Self { inner: TextLoader::new(path) } }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

#[pyclass(name = "JsonLinesLoader", module = "litgraph.loaders")]
pub struct PyJsonLinesLoader { inner: JsonLinesLoader }

#[pymethods]
impl PyJsonLinesLoader {
    #[new]
    #[pyo3(signature = (path, content_field="content"))]
    fn new(path: String, content_field: &str) -> Self {
        Self { inner: JsonLinesLoader::new(path, content_field) }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

#[pyclass(name = "MarkdownLoader", module = "litgraph.loaders")]
pub struct PyMarkdownLoader { inner: MarkdownLoader }

#[pymethods]
impl PyMarkdownLoader {
    #[new]
    fn new(path: String) -> Self { Self { inner: MarkdownLoader::new(path) } }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

/// Parallel directory loader using the default dispatcher (.txt / .md / .jsonl).
#[pyclass(name = "DirectoryLoader", module = "litgraph.loaders")]
pub struct PyDirectoryLoader {
    root: String,
    glob: String,
    follow_symlinks: bool,
}

#[pymethods]
impl PyDirectoryLoader {
    #[new]
    #[pyo3(signature = (root, glob="**/*", follow_symlinks=false))]
    fn new(root: String, glob: &str, follow_symlinks: bool) -> Self {
        Self { root, glob: glob.to_string(), follow_symlinks }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let root = self.root.clone();
        let glob = self.glob.clone();
        let follow = self.follow_symlinks;
        let docs = py.allow_threads(move || {
            let loader = DirectoryLoader::new(&root, &glob, default_dispatcher)
                .follow_symlinks(follow);
            loader.load().map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        docs_to_pylist(py, docs)
    }
}
