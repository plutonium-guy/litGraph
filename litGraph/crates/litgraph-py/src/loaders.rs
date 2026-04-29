//! Python bindings for loaders. All `load()` calls release the GIL and run the
//! rayon-parallel directory traversal on Rust threads.

use litgraph_loaders::{
    ConfluenceLoader, CsvLoader, DirectoryLoader, DocxLoader, GithubFilesLoader,
    GithubIssuesLoader, GmailLoader, GoogleDriveLoader, HtmlLoader, JiraIssuesLoader,
    JsonLinesLoader, JsonLoader, LinearIssuesLoader,
    Loader, MarkdownLoader, NotionLoader, PdfLoader, S3Loader, SlackLoader, TextLoader, WebLoader,
    default_dispatcher,
};
use litgraph_providers_bedrock::AwsCredentials;
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
    m.add_class::<PyNotionLoader>()?;
    m.add_class::<PySlackLoader>()?;
    m.add_class::<PyConfluenceLoader>()?;
    m.add_class::<PyGithubIssuesLoader>()?;
    m.add_class::<PyGithubFilesLoader>()?;
    m.add_class::<PyGmailLoader>()?;
    m.add_class::<PyGoogleDriveLoader>()?;
    m.add_class::<PyLinearIssuesLoader>()?;
    m.add_class::<PyJiraIssuesLoader>()?;
    m.add_class::<PyS3Loader>()?;
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

// ---------- NotionLoader (iter 94) ----------

/// Notion loader — pulls pages from a Notion database OR a list of page ids.
/// Direct LangChain `NotionDBLoader` parity. Pure Rust via reqwest blocking
/// + the official Notion v1 REST API. Pages are flattened to text with
/// markdown affordances for headings / lists / quotes / code blocks.
///
/// ```python
/// from litgraph.loaders import NotionLoader
/// # Load all pages in a database (the most common pattern):
/// docs = NotionLoader.from_database(api_key="secret_xxx", database_id="abcd1234").load()
/// # Or load specific pages by id:
/// docs = NotionLoader.from_pages(api_key="secret_xxx", page_ids=["p1", "p2"]).load()
/// ```
#[pyclass(name = "NotionLoader", module = "litgraph.loaders")]
pub struct PyNotionLoader { inner: NotionLoader }

#[pymethods]
impl PyNotionLoader {
    /// Construct a loader for a Notion database (queries paginated).
    #[staticmethod]
    #[pyo3(signature = (api_key, database_id, base_url=None, max_pages=Some(1000), timeout_s=30))]
    fn from_database(
        api_key: String,
        database_id: String,
        base_url: Option<String>,
        max_pages: Option<usize>,
        timeout_s: u64,
    ) -> Self {
        let mut inner = NotionLoader::from_database(api_key, database_id);
        if let Some(url) = base_url { inner = inner.with_base_url(url); }
        inner = inner.with_max_pages(max_pages);
        inner.timeout = std::time::Duration::from_secs(timeout_s);
        Self { inner }
    }

    /// Construct a loader for an explicit list of page ids.
    #[staticmethod]
    #[pyo3(signature = (api_key, page_ids, base_url=None, max_pages=None, timeout_s=30))]
    fn from_pages(
        api_key: String,
        page_ids: Vec<String>,
        base_url: Option<String>,
        max_pages: Option<usize>,
        timeout_s: u64,
    ) -> Self {
        let mut inner = NotionLoader::from_pages(api_key, page_ids);
        if let Some(url) = base_url { inner = inner.with_base_url(url); }
        inner = inner.with_max_pages(max_pages);
        inner.timeout = std::time::Duration::from_secs(timeout_s);
        Self { inner }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

// ---------- SlackLoader (iter 95) ----------

/// Slack loader — pull channel message history via the Slack Web API.
/// Each message → one `Document`. Threads optionally flattened inline.
///
/// ```python
/// from litgraph.loaders import SlackLoader
/// # Bot token (xoxb-...) with channels:history scope.
/// docs = SlackLoader(
///     api_key="xoxb-...",
///     channel_id="C0123ABCD",
///     include_threads=True,  # follow replies
///     max_messages=500,
/// ).load()
/// ```
#[pyclass(name = "SlackLoader", module = "litgraph.loaders")]
pub struct PySlackLoader { inner: SlackLoader }

#[pymethods]
impl PySlackLoader {
    #[new]
    #[pyo3(signature = (
        api_key, channel_id, base_url=None, max_messages=Some(1000),
        include_threads=false, oldest=None, latest=None, timeout_s=30,
    ))]
    fn new(
        api_key: String,
        channel_id: String,
        base_url: Option<String>,
        max_messages: Option<usize>,
        include_threads: bool,
        oldest: Option<String>,
        latest: Option<String>,
        timeout_s: u64,
    ) -> Self {
        let mut inner = SlackLoader::new(api_key, channel_id);
        if let Some(url) = base_url { inner = inner.with_base_url(url); }
        inner = inner.with_max_messages(max_messages);
        inner = inner.with_include_threads(include_threads);
        if let Some(o) = oldest { inner = inner.with_oldest(o); }
        if let Some(l) = latest { inner = inner.with_latest(l); }
        inner.timeout = std::time::Duration::from_secs(timeout_s);
        Self { inner }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

// ---------- ConfluenceLoader (iter 96) ----------

/// Confluence loader — pull pages from a Confluence space via REST API.
/// Supports BOTH Cloud (Basic email+token) and Server/DC (Bearer Personal
/// Access Token) auth. Direct LangChain `ConfluenceLoader` parity.
///
/// ```python
/// from litgraph.loaders import ConfluenceLoader
/// # Cloud:
/// docs = ConfluenceLoader.from_space_cloud(
///     base_url="https://acme.atlassian.net",
///     email="ada@example.com",
///     api_token="...",
///     space_key="ENG",
/// ).load()
/// # Server/DC:
/// docs = ConfluenceLoader.from_space_bearer(
///     base_url="https://wiki.internal.corp",
///     token="pat-...",
///     space_key="ENG",
/// ).load()
/// ```
#[pyclass(name = "ConfluenceLoader", module = "litgraph.loaders")]
pub struct PyConfluenceLoader { inner: ConfluenceLoader }

#[pymethods]
impl PyConfluenceLoader {
    /// Cloud auth (Basic email + API token). `base_url` like
    /// `"https://acme.atlassian.net"`.
    #[staticmethod]
    #[pyo3(signature = (base_url, email, api_token, space_key, max_pages=Some(1000), timeout_s=30))]
    fn from_space_cloud(
        base_url: String,
        email: String,
        api_token: String,
        space_key: String,
        max_pages: Option<usize>,
        timeout_s: u64,
    ) -> Self {
        let inner = ConfluenceLoader::from_space_cloud(base_url, email, api_token, space_key)
            .with_max_pages(max_pages)
            .with_timeout(std::time::Duration::from_secs(timeout_s));
        Self { inner }
    }

    /// Server/DC auth (Bearer with a Personal Access Token).
    #[staticmethod]
    #[pyo3(signature = (base_url, token, space_key, max_pages=Some(1000), timeout_s=30))]
    fn from_space_bearer(
        base_url: String,
        token: String,
        space_key: String,
        max_pages: Option<usize>,
        timeout_s: u64,
    ) -> Self {
        let inner = ConfluenceLoader::from_space_bearer(base_url, token, space_key)
            .with_max_pages(max_pages)
            .with_timeout(std::time::Duration::from_secs(timeout_s));
        Self { inner }
    }

    /// Load specific pages by id (Cloud auth).
    #[staticmethod]
    #[pyo3(signature = (base_url, email, api_token, page_ids, max_pages=None, timeout_s=30))]
    fn from_pages_cloud(
        base_url: String,
        email: String,
        api_token: String,
        page_ids: Vec<String>,
        max_pages: Option<usize>,
        timeout_s: u64,
    ) -> Self {
        let inner = ConfluenceLoader::from_pages_cloud(base_url, email, api_token, page_ids)
            .with_max_pages(max_pages)
            .with_timeout(std::time::Duration::from_secs(timeout_s));
        Self { inner }
    }

    /// Load specific pages by id (Bearer auth).
    #[staticmethod]
    #[pyo3(signature = (base_url, token, page_ids, max_pages=None, timeout_s=30))]
    fn from_pages_bearer(
        base_url: String,
        token: String,
        page_ids: Vec<String>,
        max_pages: Option<usize>,
        timeout_s: u64,
    ) -> Self {
        let inner = ConfluenceLoader::from_pages_bearer(base_url, token, page_ids)
            .with_max_pages(max_pages)
            .with_timeout(std::time::Duration::from_secs(timeout_s));
        Self { inner }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

// ---------- GithubIssuesLoader (iter 98) ----------

/// GitHub Issues loader — pull issues + PRs from a repo via the GitHub REST
/// API. Direct LangChain `GitHubIssuesLoader` parity. Includes PRs in results
/// (tagged via `metadata["is_pull_request"]`) since GitHub's /issues endpoint
/// returns both — callers filter downstream if they want only one kind.
///
/// ```python
/// from litgraph.loaders import GithubIssuesLoader
/// docs = GithubIssuesLoader(
///     token="ghp_...",
///     owner="acme",
///     repo="app",
///     state="open",
///     include_comments=True,
///     labels=["bug", "p1"],
/// ).load()
/// ```
#[pyclass(name = "GithubIssuesLoader", module = "litgraph.loaders")]
pub struct PyGithubIssuesLoader { inner: GithubIssuesLoader }

#[pymethods]
impl PyGithubIssuesLoader {
    #[new]
    #[pyo3(signature = (
        token, owner, repo,
        state="all", include_comments=false, labels=None,
        max_issues=Some(1000), base_url=None, timeout_s=30,
    ))]
    fn new(
        token: String,
        owner: String,
        repo: String,
        state: &str,
        include_comments: bool,
        labels: Option<Vec<String>>,
        max_issues: Option<usize>,
        base_url: Option<String>,
        timeout_s: u64,
    ) -> Self {
        let mut inner = GithubIssuesLoader::from_repo_issues(token, owner, repo)
            .with_state(state)
            .with_include_comments(include_comments)
            .with_max_issues(max_issues);
        if let Some(url) = base_url { inner = inner.with_base_url(url); }
        if let Some(ls) = labels { inner = inner.with_labels(ls); }
        inner.timeout = std::time::Duration::from_secs(timeout_s);
        Self { inner }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

// ---------- GithubFilesLoader (iter 102) ----------

/// GitHub files loader — walk a repo tree and pull file contents for
/// code-RAG / docs-RAG. Complements `GithubIssuesLoader`.
///
/// ```python
/// from litgraph.loaders import GithubFilesLoader
/// docs = GithubFilesLoader(
///     token="ghp_...",
///     owner="acme",
///     repo="app",
///     ref="main",
///     extensions=[".rs", ".md"],
///     max_files=500,
///     max_file_size_bytes=1024 * 1024,
/// ).load()
/// ```
#[pyclass(name = "GithubFilesLoader", module = "litgraph.loaders")]
pub struct PyGithubFilesLoader { inner: GithubFilesLoader }

#[pymethods]
impl PyGithubFilesLoader {
    #[new]
    #[pyo3(signature = (
        token, owner, repo,
        r#ref="main", extensions=None, exclude_paths=None,
        max_files=Some(500), max_file_size_bytes=1024 * 1024,
        base_url=None, timeout_s=30,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        token: String,
        owner: String,
        repo: String,
        r#ref: &str,
        extensions: Option<Vec<String>>,
        exclude_paths: Option<Vec<String>>,
        max_files: Option<usize>,
        max_file_size_bytes: u64,
        base_url: Option<String>,
        timeout_s: u64,
    ) -> Self {
        let mut inner = GithubFilesLoader::from_repo_tree(token, owner, repo)
            .with_ref(r#ref)
            .with_max_files(max_files)
            .with_max_file_size_bytes(max_file_size_bytes);
        if let Some(url) = base_url { inner = inner.with_base_url(url); }
        if let Some(e) = extensions { inner = inner.with_extensions(e); }
        if let Some(ex) = exclude_paths { inner = inner.with_exclude_paths(ex); }
        inner.timeout = std::time::Duration::from_secs(timeout_s);
        Self { inner }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

// ---------- GmailLoader (iter 103) ----------

/// Gmail loader — pull messages via the Gmail REST API. OAuth2 Bearer auth
/// (caller mints + refreshes the token externally). Body extraction walks
/// multipart payload, prefers `text/plain`, falls back to stripped HTML.
///
/// ```python
/// from litgraph.loaders import GmailLoader
/// docs = GmailLoader(
///     access_token="ya29.a0A...",
///     query="from:alice after:2024/01/01",
///     max_messages=50,
/// ).load()
/// ```
#[pyclass(name = "GmailLoader", module = "litgraph.loaders")]
pub struct PyGmailLoader { inner: GmailLoader }

#[pymethods]
impl PyGmailLoader {
    #[new]
    #[pyo3(signature = (
        access_token, user_id="me", query=None,
        include_body=true, max_messages=Some(100),
        base_url=None, timeout_s=30,
    ))]
    fn new(
        access_token: String,
        user_id: &str,
        query: Option<String>,
        include_body: bool,
        max_messages: Option<usize>,
        base_url: Option<String>,
        timeout_s: u64,
    ) -> Self {
        let mut inner = GmailLoader::new(access_token)
            .with_user_id(user_id)
            .with_include_body(include_body)
            .with_max_messages(max_messages);
        if let Some(q) = query { inner = inner.with_query(q); }
        if let Some(url) = base_url { inner = inner.with_base_url(url); }
        inner.timeout = std::time::Duration::from_secs(timeout_s);
        Self { inner }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

// ---------- GoogleDriveLoader (iter 104) ----------

/// Google Drive loader — pull files via the Drive REST API. OAuth2 Bearer
/// auth; caller mints + refreshes the token externally. Google native
/// formats (Docs/Sheets/Slides) are auto-exported to text/plain or
/// text/csv; plain textual uploads use `alt=media`. Binary files skipped
/// by default (enable via `include_binaries=True` for metadata-only docs).
///
/// ```python
/// from litgraph.loaders import GoogleDriveLoader
/// docs = GoogleDriveLoader(
///     access_token="ya29.a0A...",
///     folder_id="1abcDEF...",  # or use query=
///     mime_types=["application/vnd.google-apps.document", "text/markdown"],
///     max_files=200,
/// ).load()
/// ```
#[pyclass(name = "GoogleDriveLoader", module = "litgraph.loaders")]
pub struct PyGoogleDriveLoader { inner: GoogleDriveLoader }

#[pymethods]
impl PyGoogleDriveLoader {
    #[new]
    #[pyo3(signature = (
        access_token, folder_id=None, query=None, mime_types=None,
        include_binaries=false, max_files=Some(500),
        base_url=None, timeout_s=30,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        access_token: String,
        folder_id: Option<String>,
        query: Option<String>,
        mime_types: Option<Vec<String>>,
        include_binaries: bool,
        max_files: Option<usize>,
        base_url: Option<String>,
        timeout_s: u64,
    ) -> Self {
        let mut inner = GoogleDriveLoader::new(access_token)
            .with_include_binaries(include_binaries)
            .with_max_files(max_files);
        if let Some(fid) = folder_id { inner = inner.with_folder_id(fid); }
        if let Some(q) = query { inner = inner.with_query(q); }
        if let Some(mts) = mime_types { inner = inner.with_mime_types(mts); }
        if let Some(url) = base_url { inner = inner.with_base_url(url); }
        inner.timeout = std::time::Duration::from_secs(timeout_s);
        Self { inner }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

// ---------- LinearIssuesLoader (iter 124) ----------
//
// First GraphQL-backed loader. Auth quirk: raw API key in Authorization
// header — NO `Bearer` prefix. Filters (team_key / state_names /
// label_names) stack AND-semantics in the GraphQL filter.
#[pyclass(name = "LinearIssuesLoader", module = "litgraph.loaders")]
pub struct PyLinearIssuesLoader { inner: LinearIssuesLoader }

#[pymethods]
impl PyLinearIssuesLoader {
    #[new]
    #[pyo3(signature = (
        api_key, team_key=None, state_names=None, label_names=None,
        max_issues=500, base_url=None, timeout_s=30,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        api_key: String,
        team_key: Option<String>,
        state_names: Option<Vec<String>>,
        label_names: Option<Vec<String>>,
        max_issues: usize,
        base_url: Option<String>,
        timeout_s: u64,
    ) -> Self {
        let mut inner = LinearIssuesLoader::new(api_key)
            .with_timeout(std::time::Duration::from_secs(timeout_s))
            .with_max_issues(max_issues);
        if let Some(tk) = team_key { inner = inner.with_team(tk); }
        if let Some(sn) = state_names { inner = inner.with_states(sn); }
        if let Some(ln) = label_names { inner = inner.with_labels(ln); }
        if let Some(url) = base_url { inner = inner.with_base_url(url); }
        Self { inner }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

// ---------- JiraIssuesLoader (iter 125) ----------
//
// Auth: Cloud (email + API token → Basic) or Data Center (Bearer PAT).
// Dispatch via `token`: when `email` is set, Basic; otherwise Bearer.
// JQL is required — caller picks the project/filter scope.
#[pyclass(name = "JiraIssuesLoader", module = "litgraph.loaders")]
pub struct PyJiraIssuesLoader { inner: JiraIssuesLoader }

#[pymethods]
impl PyJiraIssuesLoader {
    #[new]
    #[pyo3(signature = (
        base_url, jql, email=None, api_token=None, bearer_token=None,
        max_issues=500, timeout_s=30,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base_url: String,
        jql: String,
        email: Option<String>,
        api_token: Option<String>,
        bearer_token: Option<String>,
        max_issues: usize,
        timeout_s: u64,
    ) -> PyResult<Self> {
        let inner = match (email, api_token, bearer_token) {
            (Some(e), Some(t), None) => JiraIssuesLoader::cloud(base_url, e, t, jql),
            (None, None, Some(bt)) => JiraIssuesLoader::with_bearer_token(base_url, bt, jql),
            _ => {
                return Err(PyRuntimeError::new_err(
                    "JiraIssuesLoader: pass either (email, api_token) for Cloud OR (bearer_token) for Data Center — not both, not neither"
                ));
            }
        };
        let inner = inner
            .with_max_issues(max_issues)
            .with_timeout(std::time::Duration::from_secs(timeout_s));
        Ok(Self { inner })
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}

// ---------- S3Loader (iter 126) ----------
//
// AWS S3 bucket loader. SigV4 auth via access_key_id + secret_access_key
// (optional session_token for STS-scoped creds). Filters: prefix,
// extensions allowlist, exclude-path substrings, size cap, max-files.
// Works with S3-compatible stores (MinIO, Cloudflare R2, Backblaze B2)
// via `base_url` override.
#[pyclass(name = "S3Loader", module = "litgraph.loaders")]
pub struct PyS3Loader { inner: S3Loader }

#[pymethods]
impl PyS3Loader {
    #[new]
    #[pyo3(signature = (
        access_key_id, secret_access_key, region, bucket,
        session_token=None, prefix=None, extensions=None, exclude_paths=None,
        max_files=500, max_file_size_bytes=10_485_760,
        base_url=None, timeout_s=30,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        access_key_id: String,
        secret_access_key: String,
        region: String,
        bucket: String,
        session_token: Option<String>,
        prefix: Option<String>,
        extensions: Option<Vec<String>>,
        exclude_paths: Option<Vec<String>>,
        max_files: usize,
        max_file_size_bytes: u64,
        base_url: Option<String>,
        timeout_s: u64,
    ) -> Self {
        let creds = AwsCredentials { access_key_id, secret_access_key, session_token };
        let mut inner = S3Loader::new(creds, region, bucket)
            .with_max_files(max_files)
            .with_max_file_size_bytes(max_file_size_bytes)
            .with_timeout(std::time::Duration::from_secs(timeout_s));
        if let Some(p) = prefix { inner = inner.with_prefix(p); }
        if let Some(e) = extensions { inner = inner.with_extensions(e); }
        if let Some(ex) = exclude_paths { inner = inner.with_exclude_paths(ex); }
        if let Some(url) = base_url { inner = inner.with_base_url(url); }
        Self { inner }
    }

    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let docs = py.allow_threads(|| self.inner.load()
            .map_err(|e| PyRuntimeError::new_err(e.to_string())))?;
        docs_to_pylist(py, docs)
    }
}
