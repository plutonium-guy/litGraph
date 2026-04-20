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
    BraveSearch, BraveSearchConfig, DuckDuckGoConfig, DuckDuckGoSearch, TavilyConfig, TavilySearch,
};
use litgraph_tools_utils::{
    CalculatorTool, FsRoot, HttpRequestConfig, HttpRequestTool, ListDirectoryTool, ReadFileTool,
    ShellTool, WriteFileTool,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use crate::graph::{json_to_py, py_to_json};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFunctionTool>()?;
    m.add_class::<PyBraveSearchTool>()?;
    m.add_class::<PyTavilySearchTool>()?;
    m.add_class::<PyCalculatorTool>()?;
    m.add_class::<PyHttpRequestTool>()?;
    m.add_class::<PyReadFileTool>()?;
    m.add_class::<PyWriteFileTool>()?;
    m.add_class::<PyListDirectoryTool>()?;
    m.add_class::<PyShellTool>()?;
    m.add_class::<PyDuckDuckGoSearchTool>()?;
    Ok(())
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
            let ret = self.func.call1(py, (py_args,)).map_err(|e| e.to_string())?;
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
