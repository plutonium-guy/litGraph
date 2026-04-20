//! `ReactAgent` + `SupervisorAgent` Python bindings.

use std::collections::HashMap;
use std::sync::Arc;

use litgraph_agents::{ReactAgent, ReactAgentConfig, SupervisorAgent, SupervisorConfig};
use litgraph_core::{ChatModel, ChatOptions, Message, Role};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::providers::{
    PyAnthropicChat, PyBedrockChat, PyCohereChat, PyGeminiChat, PyOpenAIChat,
};
use crate::runtime::block_on_compat;
use crate::tools::{
    PyBraveSearchTool, PyCalculatorTool, PyDuckDuckGoSearchTool, PyFunctionTool, PyHttpRequestTool,
    PyListDirectoryTool, PyReadFileTool, PyShellTool, PyTavilySearchTool, PyWriteFileTool,
};
use crate::mcp::PyMcpTool;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReactAgent>()?;
    m.add_class::<PySupervisorAgent>()?;
    Ok(())
}

pub(crate) fn extract_chat_model(bound: &Bound<'_, PyAny>) -> PyResult<Arc<dyn ChatModel>> {
    if let Ok(o) = bound.extract::<PyRef<PyOpenAIChat>>() {
        Ok(o.chat_model())
    } else if let Ok(a) = bound.extract::<PyRef<PyAnthropicChat>>() {
        Ok(a.chat_model())
    } else if let Ok(g) = bound.extract::<PyRef<PyGeminiChat>>() {
        Ok(g.chat_model())
    } else if let Ok(b) = bound.extract::<PyRef<PyBedrockChat>>() {
        Ok(b.chat_model())
    } else if let Ok(c) = bound.extract::<PyRef<PyCohereChat>>() {
        Ok(c.chat_model())
    } else {
        Err(PyValueError::new_err(
            "model must be OpenAIChat, AnthropicChat, GeminiChat, BedrockChat, or CohereChat",
        ))
    }
}

/// Prebuilt ReAct-style tool-calling agent. Give it a chat model and a list of
/// tools; `invoke(user_message)` runs the tool-call loop until the model stops.
#[pyclass(name = "ReactAgent", module = "litgraph.agents")]
pub struct PyReactAgent {
    pub(crate) inner: Arc<ReactAgent>,
}

#[pymethods]
impl PyReactAgent {
    #[new]
    #[pyo3(signature = (model, tools, system_prompt=None, max_iterations=10))]
    fn new(
        model: Py<PyAny>,
        tools: Bound<'_, PyList>,
        system_prompt: Option<String>,
        max_iterations: u32,
    ) -> PyResult<Self> {
        let chat_model: Arc<dyn ChatModel> =
            Python::with_gil(|py| extract_chat_model(model.bind(py)))?;

        let mut tool_vec: Vec<Arc<dyn litgraph_core::tool::Tool>> = Vec::new();
        for item in tools.iter() {
            if let Ok(ft) = item.extract::<PyRef<PyFunctionTool>>() {
                tool_vec.push(ft.as_tool());
            } else if let Ok(b) = item.extract::<PyRef<PyBraveSearchTool>>() {
                tool_vec.push(b.as_tool());
            } else if let Ok(t) = item.extract::<PyRef<PyTavilySearchTool>>() {
                tool_vec.push(t.as_tool());
            } else if let Ok(c) = item.extract::<PyRef<PyCalculatorTool>>() {
                tool_vec.push(c.as_tool());
            } else if let Ok(h) = item.extract::<PyRef<PyHttpRequestTool>>() {
                tool_vec.push(h.as_tool());
            } else if let Ok(r) = item.extract::<PyRef<PyReadFileTool>>() {
                tool_vec.push(r.as_tool());
            } else if let Ok(w) = item.extract::<PyRef<PyWriteFileTool>>() {
                tool_vec.push(w.as_tool());
            } else if let Ok(l) = item.extract::<PyRef<PyListDirectoryTool>>() {
                tool_vec.push(l.as_tool());
            } else if let Ok(m) = item.extract::<PyRef<PyMcpTool>>() {
                tool_vec.push(m.as_tool());
            } else if let Ok(s) = item.extract::<PyRef<PyShellTool>>() {
                tool_vec.push(s.as_tool());
            } else if let Ok(d) = item.extract::<PyRef<PyDuckDuckGoSearchTool>>() {
                tool_vec.push(d.as_tool());
            } else {
                return Err(PyValueError::new_err(
                    "tools must be FunctionTool, BraveSearchTool, TavilySearchTool, \
                     DuckDuckGoSearchTool, CalculatorTool, HttpRequestTool, ReadFileTool, \
                     WriteFileTool, ListDirectoryTool, ShellTool, or McpTool",
                ));
            }
        }

        let cfg = ReactAgentConfig {
            system_prompt,
            max_iterations,
            chat_options: ChatOptions::default(),
            max_parallel_tools: 16,
        };

        let agent = ReactAgent::new(chat_model, tool_vec, cfg)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(agent) })
    }

    fn invoke<'py>(&self, py: Python<'py>, user: String) -> PyResult<Bound<'py, PyDict>> {
        let agent = self.inner.clone();
        let state = py.allow_threads(|| {
            block_on_compat(async move { agent.invoke(user).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        state_to_py_dict(py, &state)
    }
}

/// Supervisor multi-agent — a supervisor LLM delegates to named worker ReactAgents
/// via handoff/finish tools.
#[pyclass(name = "SupervisorAgent", module = "litgraph.agents")]
pub struct PySupervisorAgent {
    inner: Arc<SupervisorAgent>,
}

#[pymethods]
impl PySupervisorAgent {
    /// `model` — supervisor chat model (OpenAI/Anthropic/Gemini).
    /// `workers` — dict mapping name → ReactAgent.
    #[new]
    #[pyo3(signature = (model, workers, system_prompt=None, max_hops=6))]
    fn new(
        model: Py<PyAny>,
        workers: Bound<'_, PyDict>,
        system_prompt: Option<String>,
        max_hops: u32,
    ) -> PyResult<Self> {
        let sup_model: Arc<dyn ChatModel> =
            Python::with_gil(|py| extract_chat_model(model.bind(py)))?;

        let mut map: HashMap<String, Arc<ReactAgent>> = HashMap::new();
        for (k, v) in workers.iter() {
            let name: String = k.extract()?;
            let worker: PyRef<PyReactAgent> = v.extract().map_err(|_| {
                PyValueError::new_err("worker must be a ReactAgent")
            })?;
            map.insert(name, worker.inner.clone());
        }

        let cfg = SupervisorConfig { system_prompt, max_hops };
        let sup = SupervisorAgent::new(sup_model, map, cfg)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(sup) })
    }

    fn invoke<'py>(&self, py: Python<'py>, user: String) -> PyResult<Bound<'py, PyDict>> {
        let sup = self.inner.clone();
        let state = py.allow_threads(|| {
            block_on_compat(async move { sup.invoke(user).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        state_to_py_dict(py, &state)
    }

    fn worker_names(&self) -> Vec<String> {
        self.inner.worker_names()
    }
}

fn state_to_py_dict<'py>(
    py: Python<'py>,
    state: &litgraph_agents::AgentState,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    out.set_item("iterations", state.iterations)?;
    let msgs = PyList::empty_bound(py);
    for m in &state.messages {
        msgs.append(message_to_py_dict(py, m)?)?;
    }
    out.set_item("messages", msgs)?;
    Ok(out)
}

fn message_to_py_dict<'py>(py: Python<'py>, m: &Message) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("role", match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    })?;
    d.set_item("content", m.text_content())?;
    if !m.tool_calls.is_empty() {
        let list = PyList::empty_bound(py);
        for tc in &m.tool_calls {
            let tcd = PyDict::new_bound(py);
            tcd.set_item("id", &tc.id)?;
            tcd.set_item("name", &tc.name)?;
            tcd.set_item("arguments", tc.arguments.to_string())?;
            list.append(tcd)?;
        }
        d.set_item("tool_calls", list)?;
    }
    if let Some(ref id) = m.tool_call_id {
        d.set_item("tool_call_id", id)?;
    }
    Ok(d)
}
