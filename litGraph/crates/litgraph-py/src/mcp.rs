//! Python bindings for `litgraph-mcp` — Model Context Protocol client.
//!
//! ```python
//! from litgraph.mcp import McpClient
//! client = McpClient.connect_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
//! tools = client.tools()                       # ready for ReactAgent
//! agent = ReactAgent(model=chat, tools=tools)
//! ```

use std::sync::Arc;
use std::time::Duration;

use litgraph_core::tool::Tool;
use litgraph_mcp::{McpClient, McpTool};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::graph::{json_to_py, py_to_json};
use crate::runtime::block_on_compat;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMcpClient>()?;
    m.add_class::<PyMcpTool>()?;
    Ok(())
}

/// MCP client over a stdio child process. Python builds the client via the
/// `connect_stdio` classmethod (we can't take an async constructor in PyO3).
#[pyclass(name = "McpClient", module = "litgraph.mcp")]
pub struct PyMcpClient {
    inner: Arc<McpClient>,
}

#[pymethods]
impl PyMcpClient {
    /// Spawn `program` with `args`, run the MCP `initialize` handshake, and
    /// return a connected client. `timeout_s` controls per-request timeout
    /// (default 30s).
    #[staticmethod]
    #[pyo3(signature = (program, args=Vec::new(), timeout_s=30))]
    fn connect_stdio<'py>(
        py: Python<'py>,
        program: String,
        args: Vec<String>,
        timeout_s: u64,
    ) -> PyResult<Self> {
        let inner: McpClient = py.allow_threads(|| {
            block_on_compat(async move {
                let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
                let c = McpClient::connect_stdio(&program, &arg_refs)
                    .await
                    .map_err(|e| litgraph_core::Error::other(e.to_string()))?;
                Ok::<_, litgraph_core::Error>(c.with_timeout(Duration::from_secs(timeout_s)))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(Self { inner: Arc::new(inner) })
    }

    /// Query `tools/list` — returns `[{name, description, input_schema}, ...]`.
    fn list_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let inner = self.inner.clone();
        let descs = py.allow_threads(|| {
            block_on_compat(async move { inner.list_tools().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let out = PyList::empty_bound(py);
        for d in descs {
            let dict = PyDict::new_bound(py);
            dict.set_item("name", d.name)?;
            if let Some(desc) = d.description { dict.set_item("description", desc)?; }
            dict.set_item("input_schema", json_to_py(py, &d.input_schema)?)?;
            out.append(dict)?;
        }
        Ok(out)
    }

    /// Invoke a tool by name. `args` is a dict matching the tool's
    /// `input_schema`. Returns the raw MCP result envelope (typically
    /// `{"content":[...], "isError":false}`).
    fn call_tool<'py>(
        &self,
        py: Python<'py>,
        name: String,
        args: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let args_json = py_to_json(py, &args.as_any())?;
        let inner = self.inner.clone();
        let raw = py.allow_threads(|| {
            block_on_compat(async move { inner.call_tool(&name, args_json).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        json_to_py(py, &raw)
    }

    /// Convenience: list MCP tools and wrap each as an `McpTool` ready for
    /// `ReactAgent(tools=...)`. One round-trip; Python never sees the raw
    /// descriptor list unless you also call `list_tools()`.
    fn tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let inner = self.inner.clone();
        let tool_arcs: Vec<Arc<dyn Tool>> = py.allow_threads(|| {
            block_on_compat(async move { inner.into_tools().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let out = PyList::empty_bound(py);
        for t in tool_arcs {
            let py_tool = PyMcpTool { inner: t };
            out.append(Py::new(py, py_tool)?)?;
        }
        Ok(out)
    }

    fn __repr__(&self) -> String { "McpClient(stdio)".into() }
}

/// Python wrapper around a single MCP-server-side tool (an `Arc<dyn Tool>`).
/// Acceptable to `ReactAgent(tools=...)` like any other tool type.
#[pyclass(name = "McpTool", module = "litgraph.mcp")]
pub struct PyMcpTool {
    pub(crate) inner: Arc<dyn Tool>,
}

#[pymethods]
impl PyMcpTool {
    #[getter]
    fn name(&self) -> String { self.inner.schema().name }

    #[getter]
    fn description(&self) -> String { self.inner.schema().description }

    fn __repr__(&self) -> String {
        format!("McpTool(name='{}')", self.inner.schema().name)
    }
}

impl PyMcpTool {
    pub(crate) fn as_tool(&self) -> Arc<dyn Tool> { self.inner.clone() }
}

// Suppress unused-import warning when no consumer uses these in this file.
#[allow(dead_code)]
fn _suppress_unused(_: &PyValueError, _: &McpTool) {}
