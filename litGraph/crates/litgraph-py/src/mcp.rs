//! Python bindings for `litgraph-mcp` — Model Context Protocol client.
//!
//! ```python
//! from litgraph.mcp import McpClient
//! # Local subprocess server:
//! client = McpClient.connect_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
//! # Hosted HTTP server (auth via headers):
//! client = McpClient.connect_http("https://mcp.example.com/v1",
//!                                 headers={"Authorization": "Bearer ..."})
//! tools = client.tools()                       # ready for ReactAgent
//! agent = ReactAgent(model=chat, tools=tools)
//! ```

use std::sync::Arc;
use std::time::Duration;

use litgraph_core::tool::Tool;
use litgraph_mcp::{
    serve_stdio, McpClient, McpPrompt, McpPromptArg, McpPromptMessage, McpResource, McpServer,
    McpTool,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::graph::{json_to_py, py_to_json};
use crate::runtime::block_on_compat;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMcpClient>()?;
    m.add_class::<PyMcpTool>()?;
    m.add_class::<PyMcpServer>()?;
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

    /// Connect to a hosted MCP server over HTTP (Streamable HTTP transport).
    /// `url` is the JSON-RPC POST endpoint. `headers` is an optional dict of
    /// header name → value sent on every request (typical use:
    /// `{"Authorization": "Bearer ..."}`). The session id from the
    /// `initialize` response is captured and echoed automatically.
    #[staticmethod]
    #[pyo3(signature = (url, headers=None, timeout_s=30))]
    fn connect_http<'py>(
        py: Python<'py>,
        url: String,
        headers: Option<Bound<'py, PyDict>>,
        timeout_s: u64,
    ) -> PyResult<Self> {
        let header_pairs: Vec<(String, String)> = if let Some(d) = headers {
            let mut v = Vec::with_capacity(d.len());
            for (k, val) in d.iter() {
                let kk: String = k.extract()?;
                let vv: String = val.extract()?;
                v.push((kk, vv));
            }
            v
        } else {
            Vec::new()
        };
        let inner: McpClient = py.allow_threads(|| {
            block_on_compat(async move {
                let c = McpClient::connect_http(url, header_pairs)
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

    fn __repr__(&self) -> String { "McpClient()".into() }
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

/// MCP server — expose Python-defined tools (from @tool, BraveSearch,
/// etc) to Claude Desktop / Cursor / Zed / any MCP-aware host over
/// stdio. Typical setup: one-line `main()` that hands the server a
/// list of tools and calls `.serve_stdio()`.
///
/// ```python
/// # my_server.py — invoked by the host via `python my_server.py`
/// from litgraph.mcp import McpServer
/// from litgraph.tools import tool
///
/// @tool
/// def add(a: int, b: int) -> int:
///     """Add two integers."""
///     return a + b
///
/// if __name__ == "__main__":
///     McpServer([add]).serve_stdio()
/// ```
///
/// In Claude Desktop config, point at the script:
/// ```json
/// {
///   "mcpServers": {
///     "my-server": {
///       "command": "python",
///       "args": ["/path/to/my_server.py"]
///     }
///   }
/// }
/// ```
///
/// The server handles `initialize`, `tools/list`, `tools/call`, `ping`,
/// and silently accepts `notifications/*`. Unknown methods return JSON-RPC
/// -32601. Tool errors surface as `isError: true` in the result, not as
/// JSON-RPC errors (per MCP spec).
#[pyclass(name = "McpServer", module = "litgraph.mcp")]
pub struct PyMcpServer {
    tools: Vec<Arc<dyn Tool>>,
    resources: Vec<McpResource>,
    prompts: Vec<McpPrompt>,
    server_name: String,
    server_version: String,
}

#[pymethods]
impl PyMcpServer {
    /// `resources` is a list of dicts: `{uri, name, description, mime_type, reader}`.
    /// `reader` is a 0-arg callable returning the resource's text.
    ///
    /// `prompts` is a list of dicts: `{name, description, arguments, renderer}`.
    /// `arguments` is a list of `{name, description, required}` dicts.
    /// `renderer` is a 1-arg callable: takes the arguments dict, returns a list
    /// of `(role, text)` tuples.
    #[new]
    #[pyo3(signature = (
        tools,
        resources=None,
        prompts=None,
        server_name="litgraph-mcp-server",
        server_version="0.1.0",
    ))]
    fn new(
        tools: Bound<'_, PyList>,
        resources: Option<Bound<'_, PyList>>,
        prompts: Option<Bound<'_, PyList>>,
        server_name: &str,
        server_version: &str,
    ) -> PyResult<Self> {
        let mut tool_vec: Vec<Arc<dyn Tool>> = Vec::with_capacity(tools.len());
        for item in tools.iter() {
            let t = crate::tools::extract_tool_arc(&item)?;
            tool_vec.push(t);
        }
        let resource_vec = match resources {
            Some(list) => parse_resources(&list)?,
            None => Vec::new(),
        };
        let prompt_vec = match prompts {
            Some(list) => parse_prompts(&list)?,
            None => Vec::new(),
        };
        Ok(Self {
            tools: tool_vec,
            resources: resource_vec,
            prompts: prompt_vec,
            server_name: server_name.to_string(),
            server_version: server_version.to_string(),
        })
    }

    /// Run the server on stdin / stdout. Blocks until stdin closes.
    /// Typical `if __name__ == "__main__":` entry point.
    fn serve_stdio(&self, py: Python<'_>) -> PyResult<()> {
        let mut server = McpServer::new(self.tools.clone())
            .with_server_name(&self.server_name)
            .with_server_version(&self.server_version);
        for r in &self.resources {
            server = server.with_resource(r.clone());
        }
        for p in &self.prompts {
            server = server.with_prompt(p.clone());
        }
        py.allow_threads(|| {
            block_on_compat(async move { serve_stdio(server).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "McpServer(tools={}, resources={}, prompts={}, name='{}')",
            self.tools.len(),
            self.resources.len(),
            self.prompts.len(),
            self.server_name
        )
    }
}

fn parse_resources(list: &Bound<'_, PyList>) -> PyResult<Vec<McpResource>> {
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let d: Bound<'_, pyo3::types::PyDict> = item.downcast_into()?;
        let uri: String = d.get_item("uri")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("resource missing `uri`"))?
            .extract()?;
        let name: String = d.get_item("name")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_else(|| uri.clone());
        let description: String = d.get_item("description")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();
        let mime_type: String = d.get_item("mime_type")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_else(|| "text/plain".to_string());
        let reader: Py<PyAny> = d.get_item("reader")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("resource missing `reader`"))?
            .into();
        let reader_arc = Arc::new(reader);
        let reader_fn: litgraph_mcp::ResourceReader = Arc::new(move || {
            Python::with_gil(|py| {
                match reader_arc.call0(py) {
                    Ok(v) => v.extract::<String>(py).map_err(|e| e.to_string()),
                    Err(e) => Err(e.to_string()),
                }
            })
        });
        out.push(McpResource {
            uri,
            name,
            description,
            mime_type,
            reader: reader_fn,
        });
    }
    Ok(out)
}

fn parse_prompts(list: &Bound<'_, PyList>) -> PyResult<Vec<McpPrompt>> {
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let d: Bound<'_, pyo3::types::PyDict> = item.downcast_into()?;
        let name: String = d.get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("prompt missing `name`"))?
            .extract()?;
        let description: String = d.get_item("description")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();
        let arguments = match d.get_item("arguments")? {
            Some(arg_list) => {
                let arg_pylist: Bound<'_, PyList> = arg_list.downcast_into()?;
                let mut args = Vec::with_capacity(arg_pylist.len());
                for a in arg_pylist.iter() {
                    let ad: Bound<'_, pyo3::types::PyDict> = a.downcast_into()?;
                    let a_name: String = ad.get_item("name")?
                        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("arg missing `name`"))?
                        .extract()?;
                    let a_desc: String = ad.get_item("description")?
                        .map(|v| v.extract())
                        .transpose()?
                        .unwrap_or_default();
                    let a_required: bool = ad.get_item("required")?
                        .map(|v| v.extract())
                        .transpose()?
                        .unwrap_or(false);
                    args.push(McpPromptArg {
                        name: a_name,
                        description: a_desc,
                        required: a_required,
                    });
                }
                args
            }
            None => Vec::new(),
        };
        let renderer: Py<PyAny> = d.get_item("renderer")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("prompt missing `renderer`"))?
            .into();
        let renderer_arc = Arc::new(renderer);
        let renderer_fn: litgraph_mcp::PromptRenderer = Arc::new(move |args: &serde_json::Value| {
            Python::with_gil(|py| -> std::result::Result<Vec<McpPromptMessage>, String> {
                // Convert serde_json Value to a Python dict via json module.
                let json_mod = py.import_bound("json").map_err(|e| e.to_string())?;
                let args_str = args.to_string();
                let py_args = json_mod
                    .call_method1("loads", (args_str,))
                    .map_err(|e| e.to_string())?;
                let result = renderer_arc
                    .call1(py, (py_args,))
                    .map_err(|e| e.to_string())?;
                // Expect list of (role, text) tuples OR list of {"role","text"} dicts.
                let list: Bound<'_, PyList> = result
                    .bind(py)
                    .downcast()
                    .map_err(|e| e.to_string())?
                    .clone();
                let mut msgs = Vec::with_capacity(list.len());
                for m in list.iter() {
                    // Try tuple first.
                    if let Ok(tup) = m.extract::<(String, String)>() {
                        msgs.push(McpPromptMessage { role: tup.0, text: tup.1 });
                        continue;
                    }
                    // Fall back to dict.
                    let d: Bound<'_, pyo3::types::PyDict> = m
                        .downcast_into()
                        .map_err(|e| e.to_string())?;
                    let role: String = d.get_item("role")
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "msg missing `role`".to_string())?
                        .extract()
                        .map_err(|e| e.to_string())?;
                    let text: String = d.get_item("text")
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "msg missing `text`".to_string())?
                        .extract()
                        .map_err(|e| e.to_string())?;
                    msgs.push(McpPromptMessage { role, text });
                }
                Ok(msgs)
            })
        });
        out.push(McpPrompt {
            name,
            description,
            arguments,
            renderer: renderer_fn,
        });
    }
    Ok(out)
}
