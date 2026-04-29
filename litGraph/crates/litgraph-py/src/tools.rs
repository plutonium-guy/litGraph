//! Bridge Python callables into Rust `Tool`s.
//!
//! A `PyFunctionTool` wraps an arbitrary Python callable with an explicit
//! JSON-Schema `parameters`. When the agent executes the tool, we acquire the
//! GIL, parse args JSON → Python dict, call, convert return → JSON.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::Result as LgResult;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_tools_search::{
    BraveSearch, BraveSearchConfig, DuckDuckGoConfig, DuckDuckGoSearch, TavilyConfig, TavilyExtract,
    TavilySearch,
};
use litgraph_tools_utils::{
    CachedTool, CalculatorTool, DalleConfig, DalleImageTool, FsRoot, HttpRequestConfig,
    HttpRequestTool, ListDirectoryTool, PythonReplConfig, PythonReplTool, ReadFileTool, ShellTool,
    SqliteQueryTool, TtsAudioTool, TtsConfig, WebhookConfig, WebhookTool, WhisperConfig,
    WhisperTranscribeTool, WriteFileTool,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use crate::graph::{json_to_py, py_to_json};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFunctionTool>()?;
    m.add_class::<PyBraveSearchTool>()?;
    m.add_class::<PyTavilySearchTool>()?;
    m.add_class::<PyTavilyExtractTool>()?;
    m.add_class::<PyCalculatorTool>()?;
    m.add_class::<PyHttpRequestTool>()?;
    m.add_class::<PyReadFileTool>()?;
    m.add_class::<PyWriteFileTool>()?;
    m.add_class::<PyListDirectoryTool>()?;
    m.add_class::<PyShellTool>()?;
    m.add_class::<PyDuckDuckGoSearchTool>()?;
    m.add_class::<PySqliteQueryTool>()?;
    m.add_class::<PyWhisperTranscribeTool>()?;
    m.add_class::<PyDalleImageTool>()?;
    m.add_class::<PyTtsAudioTool>()?;
    m.add_class::<PyCachedTool>()?;
    m.add_class::<PyPythonReplTool>()?;
    m.add_class::<PyWebhookTool>()?;
    m.add_function(pyo3::wrap_pyfunction!(tool, m)?)?;
    Ok(())
}

/// Extract an `Arc<dyn Tool>` from any of the supported `Py*Tool` types.
/// Centralizes the type-dispatch so wrappers (CachedTool, etc) and
/// agent constructors don't duplicate it.
pub(crate) fn extract_tool_arc(bound: &Bound<'_, PyAny>) -> PyResult<Arc<dyn Tool>> {
    if let Ok(t) = bound.extract::<PyRef<PyFunctionTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyBraveSearchTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyTavilySearchTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyTavilyExtractTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyDuckDuckGoSearchTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyCalculatorTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyHttpRequestTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyReadFileTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyWriteFileTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyListDirectoryTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyShellTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PySqliteQueryTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyWhisperTranscribeTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyDalleImageTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyTtsAudioTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyCachedTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyPythonReplTool>>() {
        return Ok(t.as_tool());
    }
    if let Ok(t) = bound.extract::<PyRef<PyWebhookTool>>() {
        return Ok(t.as_tool());
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "expected a litgraph.tools tool (FunctionTool, BraveSearchTool, ...)",
    ))
}

/// Decorator that converts a Python function into a `FunctionTool` with the
/// JSON Schema auto-derived from type hints + docstring. Behaviour:
///
/// - The function's `__name__` becomes the tool name (override with `name=`).
/// - The function's docstring's first paragraph becomes the description.
///   If no docstring, the description is the function name.
/// - Each non-default parameter is `required`; `Optional[X]` and parameters
///   with defaults are omitted from `required`.
/// - Type → JSON-schema mapping: `str → string`, `int → integer`,
///   `float → number`, `bool → boolean`, `list/tuple → array`, `dict → object`.
///   Anything else → `string` (no failure: agents can usually still produce
///   sensible strings; we don't want decoration to crash on exotic types).
///
/// ```python
/// @tool
/// def search(query: str, limit: int = 10) -> list[str]:
///     "Search the docs for `query`."
///     ...
/// agent = ReactAgent(model=chat, tools=[search])  # search is now a FunctionTool
/// ```
#[pyfunction]
#[pyo3(signature = (func, name=None))]
fn tool<'py>(
    py: Python<'py>,
    func: Bound<'py, PyAny>,
    name: Option<String>,
) -> PyResult<PyFunctionTool> {
    use pyo3::types::PyTuple;
    let inspect = py.import_bound("inspect")?;
    let sig = inspect.call_method1("signature", (&func,))?;
    let parameters_attr = sig.getattr("parameters")?;
    let items: Vec<(String, Bound<'_, PyAny>)> = parameters_attr
        .call_method0("items")?
        .iter()?
        .map(|item| {
            let pair: Bound<'_, PyTuple> = item?.downcast_into()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err("bad signature item"))?;
            let key: String = pair.get_item(0)?.extract()?;
            let val = pair.get_item(1)?;
            Ok((key, val))
        })
        .collect::<PyResult<_>>()?;

    let mut props = serde_json::Map::new();
    let mut required: Vec<Value> = Vec::new();
    let empty = inspect.getattr("Parameter")?.getattr("empty")?;

    for (param_name, param) in &items {
        // Skip *args, **kwargs — they're variadic; agents shouldn't use them.
        let kind: i32 = param.getattr("kind")?.extract().unwrap_or(0);
        // VAR_POSITIONAL=2, VAR_KEYWORD=4 in Python's inspect.
        if kind == 2 || kind == 4 { continue; }

        let annotation = param.getattr("annotation")?;
        let json_type = annotation_to_json_type(py, &annotation, &empty)?;

        let mut prop = serde_json::Map::new();
        prop.insert("type".into(), Value::String(json_type));
        // For lists, add a permissive items schema (agents usually emit homogeneous arrays).
        if let Some(t) = prop.get("type").and_then(|v| v.as_str()) {
            if t == "array" {
                prop.insert("items".into(), serde_json::json!({}));
            }
        }
        props.insert(param_name.clone(), Value::Object(prop));

        let default = param.getattr("default")?;
        let has_default = !default.is(&empty);
        if !has_default {
            required.push(Value::String(param_name.clone()));
        }
    }

    let schema = serde_json::json!({
        "type": "object",
        "properties": props,
        "required": required,
    });

    let final_name = name
        .or_else(|| func.getattr("__name__").ok().and_then(|n| n.extract::<String>().ok()))
        .unwrap_or_else(|| "tool".into());
    let description = func.getattr("__doc__").ok()
        .and_then(|d| d.extract::<String>().ok())
        .map(|d| {
            // Take first paragraph (stop at first blank line) and trim.
            d.split("\n\n").next().unwrap_or("").trim().to_string()
        })
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| final_name.clone());

    Ok(PyFunctionTool {
        inner: Arc::new(FunctionToolImpl {
            name: final_name,
            description,
            parameters: schema,
            func: func.unbind(),
        }),
    })
}

/// Map a Python type annotation to a JSON-schema type string.
/// Conservative + lossy on purpose — exotic types fall back to "string"
/// rather than crashing decoration. The agent will usually figure it out.
fn annotation_to_json_type(
    py: Python<'_>,
    annotation: &Bound<'_, PyAny>,
    empty: &Bound<'_, PyAny>,
) -> PyResult<String> {
    if annotation.is(empty) { return Ok("string".into()); }

    // Strip Optional[X] → X for type-name purposes; rely on `default=` to
    // mark optional-ness in the JSON-schema `required` list.
    let typing = py.import_bound("typing")?;
    let get_origin = typing.getattr("get_origin")?;
    let get_args = typing.getattr("get_args")?;
    let union_type = typing.getattr("Union")?;

    let origin = get_origin.call1((annotation,))?;
    let args = get_args.call1((annotation,))?;
    if !origin.is_none() && origin.is(&union_type) {
        // Union: find the first non-None arg.
        let none_type = py.None().into_bound(py).get_type();
        for a in args.iter()? {
            let a = a?;
            if !a.is(&none_type) {
                return annotation_to_json_type(py, &a, empty);
            }
        }
    }
    // For generics like `list[str]` or `dict[str, int]`, the origin tells us
    // the container shape.
    if !origin.is_none() {
        if let Ok(name) = origin.getattr("__name__").and_then(|n| n.extract::<String>()) {
            return Ok(match name.as_str() {
                "list" | "tuple" | "set" | "frozenset" => "array",
                "dict" => "object",
                _ => "string",
            }.into());
        }
    }
    // Plain types (no generic args). `int`, `str`, etc.
    if let Ok(name) = annotation.getattr("__name__").and_then(|n| n.extract::<String>()) {
        return Ok(match name.as_str() {
            "str" => "string",
            "int" => "integer",
            "float" => "number",
            "bool" => "boolean",
            "bytes" => "string",
            "list" | "tuple" | "set" | "frozenset" => "array",
            "dict" => "object",
            _ => "string",
        }.into());
    }
    Ok("string".into())
}

/// Read-only SQLite query tool with table allowlist + row caps. Three guard
/// rails: engine-level READONLY open, statement-keyword gate (SELECT/WITH
/// only by default; PRAGMA opt-in), table allowlist. Blocks `WITH ... DELETE`
/// CTE writes via dedicated regex.
#[pyclass(name = "SqliteQueryTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PySqliteQueryTool { pub(crate) inner: Arc<SqliteQueryTool> }

#[pymethods]
impl PySqliteQueryTool {
    #[new]
    #[pyo3(signature = (
        db_path, allowed_tables, read_only=true, max_rows=1000,
        max_output_bytes=262_144, allow_pragma=false,
    ))]
    fn new(
        db_path: String,
        allowed_tables: Vec<String>,
        read_only: bool,
        max_rows: usize,
        max_output_bytes: usize,
        allow_pragma: bool,
    ) -> PyResult<Self> {
        let t = SqliteQueryTool::new(db_path, allowed_tables)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .with_read_only(read_only)
            .with_max_rows(max_rows)
            .with_max_output_bytes(max_output_bytes)
            .with_pragma(allow_pragma);
        Ok(Self { inner: Arc::new(t) })
    }
    #[getter] fn name(&self) -> &'static str { "sqlite_query" }
    fn __repr__(&self) -> String {
        format!("SqliteQueryTool(allowlist={}, read_only={})",
            self.inner.allowed_tables.len(), self.inner.read_only)
    }
}

impl PySqliteQueryTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// Whisper transcription tool — POST audio file to OpenAI-compatible
/// `/audio/transcriptions` endpoint. Works with OpenAI directly + every
/// provider that mimics that surface (Groq, Together, OpenRouter,
/// self-hosted whisper.cpp + vLLM-with-Whisper).
///
/// Tool args (the LLM passes these as JSON):
///   - audio_path (required): local file path
///   - language (optional): ISO-639-1 code like "en"
///   - prompt (optional): transcript-style hint to bias the recognizer
///   - response_format: "json" (default) | "text" | "verbose_json"
///
/// ```python
/// from litgraph.tools import WhisperTranscribeTool
/// from litgraph.agents import ReactAgent
///
/// whisper = WhisperTranscribeTool(api_key=os.environ["OPENAI_API_KEY"])
/// agent = ReactAgent(model, tools=[whisper])
/// result = agent.invoke("Transcribe /tmp/podcast.mp3 and summarize it.")
/// ```
#[pyclass(name = "WhisperTranscribeTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyWhisperTranscribeTool { pub(crate) inner: Arc<WhisperTranscribeTool> }

#[pymethods]
impl PyWhisperTranscribeTool {
    #[new]
    #[pyo3(signature = (api_key, model="whisper-1", base_url=None, timeout_s=120, max_file_size_bytes=26_214_400))]
    fn new(
        api_key: String,
        model: &str,
        base_url: Option<String>,
        timeout_s: u64,
        max_file_size_bytes: u64,
    ) -> PyResult<Self> {
        let mut cfg = WhisperConfig::new(api_key)
            .with_model(model)
            .with_timeout(std::time::Duration::from_secs(timeout_s))
            .with_max_file_size(max_file_size_bytes);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        let t = WhisperTranscribeTool::new(cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(t) })
    }
    #[getter] fn name(&self) -> &'static str { "whisper_transcribe" }
    fn __repr__(&self) -> String { "WhisperTranscribeTool()".into() }
}

impl PyWhisperTranscribeTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// DALL-E / image-generation tool — POST text prompts to a DALL-E
/// compatible `/images/generations` endpoint. Works with OpenAI directly
/// + self-hosted Stable Diffusion through OpenAI-compat proxies.
///
/// Tool args (the LLM passes these as JSON):
///   - prompt (required): description of the desired image
///   - size (optional): "1024x1024" (default), "1024x1792", "1792x1024"
///   - quality (optional): "standard" | "hd" (DALL-E 3 only; hd ~2x cost)
///   - n (optional): 1..10 (DALL-E 3 enforces 1)
///   - response_format (optional): "url" (default, ~1h CDN URL) | "b64_json" (inline base64)
///
/// Returns: `{"images": [{"url"|"b64_json": "..."}, ...]}`.
///
/// ```python
/// from litgraph.tools import DalleImageTool
/// from litgraph.agents import ReactAgent
///
/// dalle = DalleImageTool(api_key=os.environ["OPENAI_API_KEY"])
/// agent = ReactAgent(model, tools=[dalle])
/// result = agent.invoke("draw me a watercolor cat in a sunbeam.")
/// ```
#[pyclass(name = "DalleImageTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyDalleImageTool { pub(crate) inner: Arc<DalleImageTool> }

#[pymethods]
impl PyDalleImageTool {
    #[new]
    #[pyo3(signature = (api_key, model="dall-e-3", base_url=None, timeout_s=120))]
    fn new(
        api_key: String,
        model: &str,
        base_url: Option<String>,
        timeout_s: u64,
    ) -> PyResult<Self> {
        let mut cfg = DalleConfig::new(api_key)
            .with_model(model)
            .with_timeout(std::time::Duration::from_secs(timeout_s));
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        let t = DalleImageTool::new(cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(t) })
    }
    #[getter] fn name(&self) -> &'static str { "image_generate" }
    fn __repr__(&self) -> String { "DalleImageTool()".into() }
}

impl PyDalleImageTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// Text-to-speech tool — POST text to OpenAI-compatible `/audio/speech`,
/// write the binary audio response to disk, return the file path. Closes
/// the audio-OUTPUT modality.
///
/// Tool args (the LLM passes these as JSON):
///   - text (required): what to say
///   - voice (required): "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer"
///   - output_path (required): local file path for the audio file
///   - format (optional): "mp3" (default) | "opus" | "aac" | "flac" | "wav" | "pcm"
///   - speed (optional): 0.25..4.0 (default 1.0)
///   - model (optional): "tts-1" (default) | "tts-1-hd"
///
/// Returns: `{"audio_path": "/tmp/x.mp3", "format": "mp3", "size_bytes": N}`.
///
/// ```python
/// from litgraph.tools import TtsAudioTool
/// from litgraph.agents import ReactAgent
///
/// tts = TtsAudioTool(api_key=os.environ["OPENAI_API_KEY"])
/// agent = ReactAgent(model, tools=[tts])
/// result = agent.invoke("Read this paragraph aloud and save to /tmp/out.mp3: ...")
/// ```
#[pyclass(name = "TtsAudioTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyTtsAudioTool { pub(crate) inner: Arc<TtsAudioTool> }

#[pymethods]
impl PyTtsAudioTool {
    #[new]
    #[pyo3(signature = (
        api_key,
        model="tts-1",
        base_url=None,
        timeout_s=120,
        max_text_len=4096,
    ))]
    fn new(
        api_key: String,
        model: &str,
        base_url: Option<String>,
        timeout_s: u64,
        max_text_len: usize,
    ) -> PyResult<Self> {
        let mut cfg = TtsConfig::new(api_key)
            .with_model(model)
            .with_timeout(std::time::Duration::from_secs(timeout_s))
            .with_max_text_len(max_text_len);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        let t = TtsAudioTool::new(cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(t) })
    }
    #[getter] fn name(&self) -> &'static str { "tts_speak" }
    fn __repr__(&self) -> String { "TtsAudioTool()".into() }
}

impl PyTtsAudioTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// TTL + LRU cache wrapper around any tool. Identical-arg calls within
/// the TTL window return the cached prior result without invoking the
/// inner tool. Use for: web search, DB queries, deterministic API
/// lookups. Don't use for: side-effecting tools (write_file, shell,
/// post-to-slack), non-deterministic tools (random sampling, current-
/// time lookup).
///
/// ```python
/// from litgraph.tools import BraveSearchTool, CachedTool
/// raw = BraveSearchTool(api_key=...)
/// cached = CachedTool(raw, ttl_seconds=3600, max_entries=256)
/// agent = ReactAgent(model, tools=[cached])
/// ```
///
/// Cache key = `{tool_name}\0{canonical_json(args)}` — `{"a":1,"b":2}`
/// and `{"b":2,"a":1}` collide as the same entry.
#[pyclass(name = "CachedTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyCachedTool { pub(crate) inner: Arc<CachedTool> }

#[pymethods]
impl PyCachedTool {
    #[new]
    #[pyo3(signature = (tool, ttl_seconds=3600, max_entries=256))]
    fn new(tool: Bound<'_, PyAny>, ttl_seconds: u64, max_entries: usize) -> PyResult<Self> {
        let inner_tool = extract_tool_arc(&tool)?;
        let cached = CachedTool::wrap(
            inner_tool,
            std::time::Duration::from_secs(ttl_seconds),
            max_entries,
        );
        Ok(Self { inner: cached })
    }

    /// Drop all entries — useful for tests + manual invalidation.
    fn clear(&self) {
        self.inner.clear();
    }

    /// Number of cached entries (TTL-expired entries removed lazily, so
    /// this may overcount briefly until the next access).
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("CachedTool(inner_size={})", self.inner.len())
    }
}

impl PyCachedTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// Execute a Python snippet in a sandboxed subprocess. The agent passes
/// `code` (Python source) and gets back `{exit_code, stdout, stderr,
/// elapsed_ms}`. Use for: math beyond Calculator's expression grammar,
/// data wrangling, JSON / CSV transforms, regex extraction.
///
/// Sandbox properties:
/// - subprocess (NOT eval) — failures stay in the child
/// - parent env stripped — only PATH/HOME/LANG/LC_ALL/TMPDIR pass through
/// - `with_extra_env` opts in additional vars (use sparingly — secrets!)
/// - working_dir is mandatory; child's CWD is set there
/// - timeout via SIGKILL (default 30s; per-call override capped at config)
/// - stdout/stderr each capped (default 64 KiB)
/// - stdin closed; code can't `input()`-block
///
/// **NOT** sandboxed against: network access, filesystem reads outside
/// working_dir (HOME is in scope), CPU/RAM exhaustion (timeout-only).
/// Run in a chroot/jail/container for adversarial input.
///
/// ```python
/// from litgraph.tools import PythonReplTool
/// import tempfile
/// repl = PythonReplTool(working_dir=tempfile.mkdtemp())
/// agent = ReactAgent(model, tools=[repl])
/// result = agent.invoke("compute the mean of [3, 5, 8, 13, 21]")
/// ```
#[pyclass(name = "PythonReplTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyPythonReplTool { pub(crate) inner: Arc<PythonReplTool> }

#[pymethods]
impl PyPythonReplTool {
    #[new]
    #[pyo3(signature = (
        working_dir,
        python="python3",
        timeout_s=30,
        max_output_bytes=65536,
        extra_env=None,
    ))]
    fn new(
        working_dir: String,
        python: &str,
        timeout_s: u64,
        max_output_bytes: usize,
        extra_env: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut cfg = PythonReplConfig::new(&working_dir)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .with_python(python)
            .with_timeout(std::time::Duration::from_secs(timeout_s))
            .with_max_output_bytes(max_output_bytes);
        if let Some(keys) = extra_env {
            cfg = cfg.with_extra_env(keys);
        }
        Ok(Self { inner: Arc::new(PythonReplTool::new(cfg)) })
    }
    #[getter] fn name(&self) -> &'static str { "python_repl" }
    fn __repr__(&self) -> String { "PythonReplTool()".into() }
}

impl PyPythonReplTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// Post agent messages to Slack / Discord / generic incoming webhooks.
/// URL is hard-coded at construction (NOT a tool arg) so a prompt-
/// injected agent can't pivot the target channel — it only picks the
/// message text.
///
/// Use different `preset`:
///   - `"slack"` → payload shape `{"text": "...", "username"?: "..."}`
///   - `"discord"` → `{"content": "...", "username"?: "..."}`
///   - `"generic"` → agent's `message` is forwarded verbatim as the
///     POST body (must be valid JSON)
///
/// For multi-channel, construct multiple `WebhookTool`s with distinct
/// `name=` values (`slack_oncall`, `slack_release`, etc) and let the
/// agent pick by tool name.
///
/// ```python
/// from litgraph.tools import WebhookTool
/// oncall = WebhookTool(
///     url=os.environ["SLACK_ONCALL_WEBHOOK"],
///     preset="slack",
///     name="notify_oncall",
///     description="Page the on-call engineer. Use ONLY for P1 incidents.",
/// )
/// agent = ReactAgent(model, tools=[oncall])
/// ```
#[pyclass(name = "WebhookTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyWebhookTool { pub(crate) inner: Arc<WebhookTool> }

#[pymethods]
impl PyWebhookTool {
    #[new]
    #[pyo3(signature = (url, preset="slack", name=None, description=None, timeout_s=10))]
    fn new(
        url: String,
        preset: &str,
        name: Option<String>,
        description: Option<String>,
        timeout_s: u64,
    ) -> PyResult<Self> {
        let mut cfg = match preset {
            "slack" => WebhookConfig::slack(url),
            "discord" => WebhookConfig::discord(url),
            "generic" => WebhookConfig::generic(url),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "WebhookTool preset must be 'slack' | 'discord' | 'generic', got '{other}'"
                )));
            }
        };
        cfg = cfg.with_timeout(std::time::Duration::from_secs(timeout_s));
        if let Some(n) = name {
            cfg = cfg.with_name(n);
        }
        if let Some(d) = description {
            cfg = cfg.with_description(d);
        }
        let t = WebhookTool::new(cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(t) })
    }
    #[getter]
    fn name(&self) -> String {
        self.inner.schema().name
    }
    fn __repr__(&self) -> String {
        format!("WebhookTool(name='{}')", self.inner.schema().name)
    }
}

impl PyWebhookTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> {
        self.inner.clone() as Arc<dyn Tool>
    }
}

/// DuckDuckGo Instant Answer search — no API key required. Limited results
/// vs Brave/Tavily (best for definitions + disambiguation queries) but the
/// only zero-credentials web-search escape hatch.
#[pyclass(name = "DuckDuckGoSearchTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyDuckDuckGoSearchTool { pub(crate) inner: Arc<DuckDuckGoSearch> }

#[pymethods]
impl PyDuckDuckGoSearchTool {
    #[new]
    #[pyo3(signature = (base_url=None, timeout_s=20))]
    fn new(base_url: Option<String>, timeout_s: u64) -> PyResult<Self> {
        let mut cfg = DuckDuckGoConfig::default();
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        let t = DuckDuckGoSearch::new(cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(t) })
    }
    #[getter] fn name(&self) -> &'static str { "web_search" }
    fn __repr__(&self) -> String { "DuckDuckGoSearchTool()".into() }
}

impl PyDuckDuckGoSearchTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// Sandboxed shell with strict allowlist + working-dir + timeout. Args are
/// passed verbatim — no shell expansion. The allowlist is mandatory; without
/// a non-empty list of allowed programs nothing runs.
#[pyclass(name = "ShellTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyShellTool { pub(crate) inner: Arc<ShellTool> }

#[pymethods]
impl PyShellTool {
    #[new]
    #[pyo3(signature = (working_dir, allowed_commands, timeout_s=30, max_output_bytes=65536))]
    fn new(
        working_dir: String,
        allowed_commands: Vec<String>,
        timeout_s: u64,
        max_output_bytes: usize,
    ) -> PyResult<Self> {
        if allowed_commands.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "allowed_commands must be non-empty (no shell access without an explicit allowlist)",
            ));
        }
        let t = ShellTool::new(working_dir, allowed_commands)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .with_timeout(std::time::Duration::from_secs(timeout_s))
            .with_max_output_bytes(max_output_bytes);
        Ok(Self { inner: Arc::new(t) })
    }
    #[getter] fn name(&self) -> &'static str { "shell" }
    fn __repr__(&self) -> String {
        format!("ShellTool(allowlist={})", self.inner.allowed_commands.len())
    }
}

impl PyShellTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// Read a UTF-8 text file from a sandbox root. Mandatory `sandbox_root` —
/// every requested path is resolved against this root and rejected if it
/// escapes via `..` or absolute paths.
#[pyclass(name = "ReadFileTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyReadFileTool { pub(crate) inner: Arc<ReadFileTool> }

#[pymethods]
impl PyReadFileTool {
    #[new]
    #[pyo3(signature = (sandbox_root, max_bytes=1_048_576))]
    fn new(sandbox_root: String, max_bytes: usize) -> PyResult<Self> {
        let root = FsRoot::new(&sandbox_root)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(ReadFileTool::new(root).with_max_bytes(max_bytes)) })
    }
    #[getter] fn name(&self) -> &'static str { "read_file" }
    fn __repr__(&self) -> String { "ReadFileTool()".into() }
}

impl PyReadFileTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// Write a UTF-8 text file inside the sandbox. By default refuses to
/// overwrite; pass `overwrite=true` per call to allow.
#[pyclass(name = "WriteFileTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyWriteFileTool { pub(crate) inner: Arc<WriteFileTool> }

#[pymethods]
impl PyWriteFileTool {
    #[new]
    fn new(sandbox_root: String) -> PyResult<Self> {
        let root = FsRoot::new(&sandbox_root)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(WriteFileTool::new(root)) })
    }
    #[getter] fn name(&self) -> &'static str { "write_file" }
    fn __repr__(&self) -> String { "WriteFileTool()".into() }
}

impl PyWriteFileTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// List directory contents inside the sandbox. Returns `[{name, kind}, ...]`
/// where `kind` is `"file" | "dir" | "symlink"`.
#[pyclass(name = "ListDirectoryTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyListDirectoryTool { pub(crate) inner: Arc<ListDirectoryTool> }

#[pymethods]
impl PyListDirectoryTool {
    #[new]
    fn new(sandbox_root: String) -> PyResult<Self> {
        let root = FsRoot::new(&sandbox_root)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(ListDirectoryTool::new(root)) })
    }
    #[getter] fn name(&self) -> &'static str { "list_directory" }
    fn __repr__(&self) -> String { "ListDirectoryTool()".into() }
}

impl PyListDirectoryTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// Built-in math expression evaluator (sandboxed via `evalexpr`). No network,
/// no I/O. Pass to `ReactAgent` as `tools=[CalculatorTool()]`.
#[pyclass(name = "CalculatorTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyCalculatorTool { pub(crate) inner: Arc<CalculatorTool> }

#[pymethods]
impl PyCalculatorTool {
    #[new]
    fn new() -> Self { Self { inner: Arc::new(CalculatorTool::new()) } }
    #[getter] fn name(&self) -> &'static str { "calculator" }
    fn __repr__(&self) -> String { "CalculatorTool()".into() }
}

impl PyCalculatorTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// Generic HTTP request tool. Allowed methods default to GET only — extend
/// via `allowed_methods=...`. Optional `allowed_hosts` allowlist for
/// defense-in-depth.
#[pyclass(name = "HttpRequestTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyHttpRequestTool { pub(crate) inner: Arc<HttpRequestTool> }

#[pymethods]
impl PyHttpRequestTool {
    #[new]
    #[pyo3(signature = (timeout_s=20, allowed_methods=None, allowed_hosts=None))]
    fn new(
        timeout_s: u64,
        allowed_methods: Option<Vec<String>>,
        allowed_hosts: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut cfg = HttpRequestConfig::default();
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(methods) = allowed_methods { cfg = cfg.with_methods(methods); }
        if let Some(hosts) = allowed_hosts { cfg = cfg.with_allowed_hosts(hosts); }
        let t = HttpRequestTool::new(cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(t) })
    }
    #[getter] fn name(&self) -> &'static str { "http_request" }
    fn __repr__(&self) -> String { "HttpRequestTool()".into() }
}

impl PyHttpRequestTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() as Arc<dyn Tool> }
}

/// Wraps a Python callable with a tool schema. Pass to `ReactAgent` as a tool.
#[pyclass(name = "FunctionTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyFunctionTool {
    pub(crate) inner: Arc<FunctionToolImpl>,
}

pub(crate) struct FunctionToolImpl {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub func: Py<PyAny>,
}

#[pymethods]
impl PyFunctionTool {
    /// `parameters` — JSON Schema for the args (dict or JSON string).
    #[new]
    fn new(
        name: String,
        description: String,
        parameters: Bound<'_, PyAny>,
        func: Py<PyAny>,
    ) -> PyResult<Self> {
        let params_val: Value = if let Ok(d) = parameters.downcast::<PyDict>() {
            crate::graph::py_dict_to_json(d)?
        } else if let Ok(s) = parameters.extract::<String>() {
            serde_json::from_str(&s)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad JSON schema: {e}")))?
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "parameters must be a dict or a JSON string",
            ));
        };
        Ok(Self {
            inner: Arc::new(FunctionToolImpl {
                name,
                description,
                parameters: params_val,
                func,
            }),
        })
    }

    #[getter]
    fn name(&self) -> &str { &self.inner.name }

    fn __repr__(&self) -> String { format!("FunctionTool(name='{}')", self.inner.name) }
}

#[async_trait]
impl Tool for FunctionToolImpl {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }

    async fn run(&self, args: Value) -> LgResult<Value> {
        let name = self.name.clone();
        let val: Result<Value, String> = Python::with_gil(|py| {
            let py_args = json_to_py(py, &args).map_err(|e| e.to_string())?;
            // Tool args arrive as a JSON object — unpack to kwargs so a
            // `def add(a, b)` tool gets `add(a=2, b=3)`, not `add({"a":2,"b":3})`.
            // Non-object args (rare) fall back to single positional.
            let ret = if let Ok(d) = py_args.downcast::<PyDict>() {
                self.func
                    .call_bound(py, pyo3::types::PyTuple::empty_bound(py), Some(d))
                    .map_err(|e| e.to_string())?
            } else {
                self.func.call1(py, (py_args,)).map_err(|e| e.to_string())?
            };
            let mut bound = ret.bind(py).clone();
            if bound.hasattr("__await__").map_err(|e| e.to_string())? {
                let asyncio = py.import_bound("asyncio").map_err(|e| e.to_string())?;
                bound = asyncio
                    .call_method1("run", (bound,))
                    .map_err(|e| e.to_string())?;
            }
            py_to_json(py, &bound).map_err(|e| e.to_string())
        });
        val.map_err(|e| litgraph_core::Error::other(format!("tool `{name}` failed: {e}")))
    }
}

impl PyFunctionTool {
    /// Internal: obtain a `dyn Tool` for the agent layer.
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> {
        self.inner.clone() as Arc<dyn Tool>
    }
}

/// Brave Search built-in tool. Pass to `ReactAgent` like a `FunctionTool`.
#[pyclass(name = "BraveSearchTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyBraveSearchTool {
    pub(crate) inner: Arc<BraveSearch>,
}

#[pymethods]
impl PyBraveSearchTool {
    #[new]
    #[pyo3(signature = (api_key, base_url=None, timeout_s=20))]
    fn new(api_key: String, base_url: Option<String>, timeout_s: u64) -> PyResult<Self> {
        let mut cfg = BraveSearchConfig::new(api_key);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        let t = BraveSearch::new(cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(t) })
    }

    #[getter]
    fn name(&self) -> &'static str { "web_search" }
    fn __repr__(&self) -> String { "BraveSearchTool()".into() }
}

impl PyBraveSearchTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> {
        self.inner.clone() as Arc<dyn Tool>
    }
}

/// Tavily Search built-in tool. Same args schema as Brave; either is a drop-in.
#[pyclass(name = "TavilySearchTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyTavilySearchTool {
    pub(crate) inner: Arc<TavilySearch>,
}

#[pymethods]
impl PyTavilySearchTool {
    #[new]
    #[pyo3(signature = (api_key, base_url=None, timeout_s=30, search_depth="basic"))]
    fn new(
        api_key: String,
        base_url: Option<String>,
        timeout_s: u64,
        search_depth: &str,
    ) -> PyResult<Self> {
        let mut cfg = TavilyConfig::new(api_key);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        cfg = cfg.with_search_depth(search_depth);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        let t = TavilySearch::new(cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(t) })
    }

    #[getter]
    fn name(&self) -> &'static str { "web_search" }
    fn __repr__(&self) -> String { "TavilySearchTool()".into() }
}

impl PyTavilySearchTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> {
        self.inner.clone() as Arc<dyn Tool>
    }
}

/// Tavily /extract — fetch full article text for a list of URLs. Pair with
/// `TavilySearchTool` for search → extract → answer loops. Reuses the same
/// API key as TavilySearch (Tavily uses one key for both endpoints).
///
/// ```python
/// from litgraph.tools import TavilyExtractTool
/// extract = TavilyExtractTool(api_key="tvly-...")
/// # Agent can pass:
/// #   {"urls": ["https://foo/bar", "https://baz/qux"], "extract_depth": "basic"}
/// ```
#[pyclass(name = "TavilyExtractTool", module = "litgraph.tools")]
#[derive(Clone)]
pub struct PyTavilyExtractTool {
    pub(crate) inner: Arc<TavilyExtract>,
}

#[pymethods]
impl PyTavilyExtractTool {
    #[new]
    #[pyo3(signature = (api_key, base_url=None, timeout_s=30))]
    fn new(api_key: String, base_url: Option<String>, timeout_s: u64) -> PyResult<Self> {
        let mut cfg = TavilyConfig::new(api_key);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url {
            cfg = cfg.with_base_url(url);
        }
        let t = TavilyExtract::new(cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(t) })
    }

    #[getter]
    fn name(&self) -> &'static str { "web_extract" }
    fn __repr__(&self) -> String { "TavilyExtractTool()".into() }
}

impl PyTavilyExtractTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> {
        self.inner.clone() as Arc<dyn Tool>
    }
}
