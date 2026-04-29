//! `ReactAgent` + `SupervisorAgent` Python bindings.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use litgraph_agents::{
    AgentEvent, ReactAgent, ReactAgentConfig, StoppedReason, SupervisorAgent, SupervisorConfig,
    TextReActAgent, TextReactAgentConfig, TextReactEvent, TextReactTurn,
};
use litgraph_core::{ChatModel, ChatOptions, Message, Role};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::sync::mpsc;

use crate::graph::json_to_py;

use crate::providers::{
    PyAnthropicChat, PyBedrockChat, PyBedrockConverseChat, PyCohereChat, PyCostCappedChat,
    PyFallbackChat, PyGeminiChat, PyOpenAIChat, PyOpenAIResponses, PyPiiScrubbingChat,
    PyPromptCachingChat, PySelfConsistencyChat, PyStructuredChatModel, PyTokenBudgetChat,
};
use crate::runtime::{block_on_compat, rt};
use crate::tools::{
    PyBraveSearchTool, PyCachedTool, PyCalculatorTool, PyDalleImageTool, PyDuckDuckGoSearchTool,
    PyFunctionTool, PyHttpRequestTool, PyListDirectoryTool, PyPythonReplTool, PyReadFileTool,
    PyGmailSendTool, PyShellTool, PySqliteQueryTool, PyTavilyExtractTool, PyTavilySearchTool,
    PyTtsAudioTool, PyWebFetchTool, PyWebhookTool, PyWhisperTranscribeTool, PyWriteFileTool,
};
use crate::mcp::PyMcpTool;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReactAgent>()?;
    m.add_class::<PySupervisorAgent>()?;
    m.add_class::<PyAgentEventStream>()?;
    m.add_class::<PyTextReActAgent>()?;
    m.add_class::<PyTextReactEventStream>()?;
    Ok(())
}

fn extract_tools(tools: &Bound<'_, PyList>) -> PyResult<Vec<Arc<dyn litgraph_core::tool::Tool>>> {
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
        } else if let Ok(sq) = item.extract::<PyRef<PySqliteQueryTool>>() {
            tool_vec.push(sq.as_tool());
        } else if let Ok(wh) = item.extract::<PyRef<PyWhisperTranscribeTool>>() {
            tool_vec.push(wh.as_tool());
        } else if let Ok(dl) = item.extract::<PyRef<PyDalleImageTool>>() {
            tool_vec.push(dl.as_tool());
        } else if let Ok(tt) = item.extract::<PyRef<PyTtsAudioTool>>() {
            tool_vec.push(tt.as_tool());
        } else if let Ok(ct) = item.extract::<PyRef<PyCachedTool>>() {
            tool_vec.push(ct.as_tool());
        } else if let Ok(pr) = item.extract::<PyRef<PyPythonReplTool>>() {
            tool_vec.push(pr.as_tool());
        } else if let Ok(wh) = item.extract::<PyRef<PyWebhookTool>>() {
            tool_vec.push(wh.as_tool());
        } else if let Ok(te) = item.extract::<PyRef<PyTavilyExtractTool>>() {
            tool_vec.push(te.as_tool());
        } else if let Ok(gs) = item.extract::<PyRef<PyGmailSendTool>>() {
            tool_vec.push(gs.as_tool());
        } else if let Ok(wf) = item.extract::<PyRef<PyWebFetchTool>>() {
            tool_vec.push(wf.as_tool());
        } else {
            return Err(PyValueError::new_err(
                "tools must be FunctionTool, BraveSearchTool, TavilySearchTool, \
                 TavilyExtractTool, DuckDuckGoSearchTool, CalculatorTool, HttpRequestTool, \
                 ReadFileTool, WriteFileTool, ListDirectoryTool, ShellTool, SqliteQueryTool, \
                 WhisperTranscribeTool, DalleImageTool, TtsAudioTool, CachedTool, \
                 PythonReplTool, WebhookTool, or McpTool",
            ));
        }
    }
    Ok(tool_vec)
}

pub(crate) fn extract_chat_model(bound: &Bound<'_, PyAny>) -> PyResult<Arc<dyn ChatModel>> {
    if let Ok(o) = bound.extract::<PyRef<PyOpenAIChat>>() {
        Ok(o.chat_model())
    } else if let Ok(r) = bound.extract::<PyRef<PyOpenAIResponses>>() {
        Ok(r.chat_model())
    } else if let Ok(a) = bound.extract::<PyRef<PyAnthropicChat>>() {
        Ok(a.chat_model())
    } else if let Ok(g) = bound.extract::<PyRef<PyGeminiChat>>() {
        Ok(g.chat_model())
    } else if let Ok(b) = bound.extract::<PyRef<PyBedrockChat>>() {
        Ok(b.chat_model())
    } else if let Ok(bc) = bound.extract::<PyRef<PyBedrockConverseChat>>() {
        Ok(bc.chat_model())
    } else if let Ok(c) = bound.extract::<PyRef<PyCohereChat>>() {
        Ok(c.chat_model())
    } else if let Ok(s) = bound.extract::<PyRef<PyStructuredChatModel>>() {
        Ok(s.chat_model())
    } else if let Ok(f) = bound.extract::<PyRef<PyFallbackChat>>() {
        Ok(f.chat_model())
    } else if let Ok(tb) = bound.extract::<PyRef<PyTokenBudgetChat>>() {
        Ok(tb.chat_model())
    } else if let Ok(ps) = bound.extract::<PyRef<PyPiiScrubbingChat>>() {
        Ok(ps.chat_model())
    } else if let Ok(pc) = bound.extract::<PyRef<PyPromptCachingChat>>() {
        Ok(pc.chat_model())
    } else if let Ok(cc) = bound.extract::<PyRef<PyCostCappedChat>>() {
        Ok(cc.chat_model())
    } else if let Ok(sc) = bound.extract::<PyRef<PySelfConsistencyChat>>() {
        Ok(sc.chat_model())
    } else {
        Err(PyValueError::new_err(
            "model must be OpenAIChat, OpenAIResponses, AnthropicChat, GeminiChat, BedrockChat, CohereChat, StructuredChatModel, FallbackChat, TokenBudgetChat, PiiScrubbingChat, PromptCachingChat, CostCappedChat, or SelfConsistencyChat",
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
            } else if let Ok(sq) = item.extract::<PyRef<PySqliteQueryTool>>() {
                tool_vec.push(sq.as_tool());
            } else if let Ok(wh) = item.extract::<PyRef<PyWhisperTranscribeTool>>() {
                tool_vec.push(wh.as_tool());
            } else if let Ok(dl) = item.extract::<PyRef<PyDalleImageTool>>() {
                tool_vec.push(dl.as_tool());
            } else if let Ok(tt) = item.extract::<PyRef<PyTtsAudioTool>>() {
                tool_vec.push(tt.as_tool());
            } else if let Ok(ct) = item.extract::<PyRef<PyCachedTool>>() {
                tool_vec.push(ct.as_tool());
            } else if let Ok(pr) = item.extract::<PyRef<PyPythonReplTool>>() {
                tool_vec.push(pr.as_tool());
            } else if let Ok(wh) = item.extract::<PyRef<PyWebhookTool>>() {
                tool_vec.push(wh.as_tool());
            } else if let Ok(te) = item.extract::<PyRef<PyTavilyExtractTool>>() {
                tool_vec.push(te.as_tool());
            } else if let Ok(gs) = item.extract::<PyRef<PyGmailSendTool>>() {
                tool_vec.push(gs.as_tool());
            } else if let Ok(wf) = item.extract::<PyRef<PyWebFetchTool>>() {
                tool_vec.push(wf.as_tool());
            } else {
                return Err(PyValueError::new_err(
                    "tools must be FunctionTool, BraveSearchTool, TavilySearchTool, \
                     TavilyExtractTool, DuckDuckGoSearchTool, CalculatorTool, HttpRequestTool, \
                     ReadFileTool, WriteFileTool, ListDirectoryTool, ShellTool, SqliteQueryTool, \
                     WhisperTranscribeTool, DalleImageTool, TtsAudioTool, CachedTool, \
                     PythonReplTool, WebhookTool, or McpTool",
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

    /// Stream per-turn events. Returns an iterator of dicts — each carries a
    /// `type` field (`iteration_start`, `llm_message`, `tool_call_start`,
    /// `tool_call_result`, `final`, `max_iterations_reached`) plus
    /// event-specific payload fields. Iteration terminates after `final` or
    /// `max_iterations_reached`.
    ///
    /// ```python
    /// for ev in agent.stream("what is 2+3"):
    ///     print(ev["type"], ev)
    /// ```
    fn stream<'py>(&self, _py: Python<'py>, user: String) -> PyResult<Py<PyAgentEventStream>> {
        let agent = self.inner.clone();
        // Spawn the stream pump on the shared runtime so Python's __next__
        // can block on the receiver. Buffer 64 events — plenty for agent
        // granularity (token streaming this is NOT).
        let (tx, rx) = mpsc::channel::<litgraph_graph::Result<AgentEvent>>(64);
        rt().spawn(async move {
            let mut s = agent.stream(user);
            while let Some(ev) = s.next().await {
                if tx.send(ev).await.is_err() {
                    break;
                }
            }
        });
        Python::with_gil(|py| {
            Py::new(py, PyAgentEventStream { rx: Arc::new(Mutex::new(Some(rx))) })
        })
    }

    /// Token-streaming variant of `stream()`. Same event shape, plus a new
    /// `token_delta` event ({"type":"token_delta", "text": "..."}) emitted
    /// during each model turn as the LLM generates characters. Use this for
    /// chat UIs that render the assistant reply progressively.
    ///
    /// Buffer is 256 (vs 64 for `stream()`) because token deltas can arrive
    /// faster than Python can drain.
    fn stream_tokens<'py>(&self, _py: Python<'py>, user: String) -> PyResult<Py<PyAgentEventStream>> {
        let agent = self.inner.clone();
        let (tx, rx) = mpsc::channel::<litgraph_graph::Result<AgentEvent>>(256);
        rt().spawn(async move {
            let mut s = agent.stream_tokens(user);
            while let Some(ev) = s.next().await {
                if tx.send(ev).await.is_err() {
                    break;
                }
            }
        });
        Python::with_gil(|py| {
            Py::new(py, PyAgentEventStream { rx: Arc::new(Mutex::new(Some(rx))) })
        })
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

/// Python-facing agent event stream. Each `__next__` blocks on the receiver
/// with GIL released, converts one `AgentEvent` into a dict.
#[pyclass(name = "AgentEventStream", module = "litgraph.agents")]
pub struct PyAgentEventStream {
    rx: Arc<Mutex<Option<mpsc::Receiver<litgraph_graph::Result<AgentEvent>>>>>,
}

#[pymethods]
impl PyAgentEventStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let slot = self.rx.clone();
        let ev = py.allow_threads(|| {
            let mut guard = slot.lock().expect("poisoned");
            let mut rx = match guard.take() {
                Some(r) => r,
                None => return None,
            };
            let got = block_on_compat(async { rx.recv().await });
            *guard = Some(rx);
            got
        });
        match ev {
            Some(Ok(e)) => agent_event_to_py_dict(py, &e),
            Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
            None => {
                *self.rx.lock().expect("poisoned") = None;
                Err(PyStopIteration::new_err("agent event stream exhausted"))
            }
        }
    }
}

fn agent_event_to_py_dict<'py>(
    py: Python<'py>,
    ev: &AgentEvent,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    match ev {
        AgentEvent::IterationStart { iteration } => {
            d.set_item("type", "iteration_start")?;
            d.set_item("iteration", *iteration)?;
        }
        AgentEvent::TokenDelta { text } => {
            d.set_item("type", "token_delta")?;
            d.set_item("text", text)?;
        }
        AgentEvent::LlmMessage { message } => {
            d.set_item("type", "llm_message")?;
            d.set_item("message", message_to_py_dict(py, message)?)?;
        }
        AgentEvent::ToolCallStart { call_id, name, arguments } => {
            d.set_item("type", "tool_call_start")?;
            d.set_item("call_id", call_id)?;
            d.set_item("name", name)?;
            d.set_item("arguments", arguments.to_string())?;
        }
        AgentEvent::ToolCallResult { call_id, name, result, is_error, duration_ms } => {
            d.set_item("type", "tool_call_result")?;
            d.set_item("call_id", call_id)?;
            d.set_item("name", name)?;
            d.set_item("result", result)?;
            d.set_item("is_error", *is_error)?;
            d.set_item("duration_ms", *duration_ms)?;
        }
        AgentEvent::Final { messages, iterations } => {
            d.set_item("type", "final")?;
            d.set_item("iterations", *iterations)?;
            let msgs = PyList::empty_bound(py);
            for m in messages {
                msgs.append(message_to_py_dict(py, m)?)?;
            }
            d.set_item("messages", msgs)?;
        }
        AgentEvent::MaxIterationsReached { messages, iterations } => {
            d.set_item("type", "max_iterations_reached")?;
            d.set_item("iterations", *iterations)?;
            let msgs = PyList::empty_bound(py);
            for m in messages {
                msgs.append(message_to_py_dict(py, m)?)?;
            }
            d.set_item("messages", msgs)?;
        }
    }
    Ok(d)
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

/// Text-mode ReAct agent for LLMs WITHOUT native tool-calling (Ollama
/// local, base-completion checkpoints, older open-weight fine-tunes).
/// Parses Thought/Action/Action Input/Final Answer prose each turn; feeds
/// tool observations back as "Observation: ..." user messages.
///
/// **For GPT-4, Claude, Gemini, Cohere R+, Mistral Large, etc — use
/// `ReactAgent` (native tool-calling API).** This class is for models
/// that don't expose a tool-calling endpoint.
///
/// ```python
/// from litgraph.agents import TextReActAgent
/// from litgraph.providers import OpenAIChat  # pointed at an Ollama server
/// agent = TextReActAgent(model, tools=[...], max_iterations=10)
/// result = agent.invoke("what's the weather in Paris?")
/// print(result["final_answer"])
/// for turn in result["trace"]:
///     print(turn)
/// ```
#[pyclass(name = "TextReActAgent", module = "litgraph.agents")]
pub struct PyTextReActAgent {
    inner: Arc<TextReActAgent>,
}

#[pymethods]
impl PyTextReActAgent {
    #[new]
    #[pyo3(signature = (
        model,
        tools,
        system_prompt=None,
        max_iterations=10,
        auto_format_instructions=true,
    ))]
    fn new(
        model: Py<PyAny>,
        tools: Bound<'_, PyList>,
        system_prompt: Option<String>,
        max_iterations: u32,
        auto_format_instructions: bool,
    ) -> PyResult<Self> {
        let chat_model: Arc<dyn ChatModel> =
            Python::with_gil(|py| extract_chat_model(model.bind(py)))?;
        let tool_vec = extract_tools(&tools)?;
        let cfg = TextReactAgentConfig {
            max_iterations,
            system_prompt,
            chat_options: ChatOptions::default(),
            auto_format_instructions,
        };
        let agent = TextReActAgent::new(chat_model, tool_vec, cfg);
        Ok(Self {
            inner: Arc::new(agent),
        })
    }

    fn invoke<'py>(&self, py: Python<'py>, user: String) -> PyResult<Bound<'py, PyDict>> {
        let agent = self.inner.clone();
        let result = py.allow_threads(|| {
            block_on_compat(async move { agent.invoke(user).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let out = PyDict::new_bound(py);
        out.set_item("final_answer", result.final_answer)?;
        out.set_item("iterations", result.iterations)?;
        out.set_item(
            "stopped_reason",
            match result.stopped_reason {
                StoppedReason::FinalAnswer => "final_answer",
                StoppedReason::MaxIterations => "max_iterations",
                StoppedReason::ParseError => "parse_error",
                StoppedReason::ToolNotFound => "tool_not_found",
            },
        )?;
        let trace = PyList::empty_bound(py);
        for turn in &result.trace {
            trace.append(text_react_turn_to_py(py, turn)?)?;
        }
        out.set_item("trace", trace)?;
        Ok(out)
    }

    /// Stream per-turn events. Returns an iterator of dicts; each carries
    /// a `type` field. Variants:
    ///   - `iteration_start` {iteration}
    ///   - `llm_response` {iteration, text}
    ///   - `parsed_action` {iteration, thought, tool, input}
    ///   - `parsed_final` {iteration, thought, answer}
    ///   - `parse_error` {iteration, error, raw_response}  [terminal]
    ///   - `tool_start` {iteration, tool, input}
    ///   - `tool_result` {iteration, tool, observation, is_error, duration_ms}
    ///   - `tool_not_found` {iteration, tool, available}  [terminal]
    ///   - `final` {answer, iterations}                   [terminal]
    ///   - `max_iterations` {iterations}                  [terminal]
    ///
    /// ```python
    /// for ev in agent.stream("solve this"):
    ///     print(ev["type"], ev)
    /// ```
    fn stream<'py>(&self, _py: Python<'py>, user: String) -> PyResult<Py<PyTextReactEventStream>> {
        let agent = self.inner.clone();
        let (tx, rx) = mpsc::channel::<litgraph_core::Result<TextReactEvent>>(64);
        rt().spawn(async move {
            let mut s = agent.stream(user);
            while let Some(ev) = s.next().await {
                if tx.send(ev).await.is_err() {
                    break;
                }
            }
        });
        Python::with_gil(|py| {
            Py::new(
                py,
                PyTextReactEventStream {
                    rx: Arc::new(Mutex::new(Some(rx))),
                },
            )
        })
    }
}

/// Iterator of TextReActAgent events. `__next__` blocks on the receiver
/// with the GIL released, returns one event dict, or raises StopIteration.
#[pyclass(name = "TextReactEventStream", module = "litgraph.agents")]
pub struct PyTextReactEventStream {
    rx: Arc<Mutex<Option<mpsc::Receiver<litgraph_core::Result<TextReactEvent>>>>>,
}

#[pymethods]
impl PyTextReactEventStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let slot = self.rx.clone();
        let ev = py.allow_threads(|| {
            let mut guard = slot.lock().expect("poisoned");
            let mut rx = match guard.take() {
                Some(r) => r,
                None => return None,
            };
            let got = block_on_compat(async { rx.recv().await });
            *guard = Some(rx);
            got
        });
        match ev {
            Some(Ok(e)) => text_react_event_to_py(py, &e),
            Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
            None => {
                *self.rx.lock().expect("poisoned") = None;
                Err(PyStopIteration::new_err("text-react event stream exhausted"))
            }
        }
    }
}

fn text_react_event_to_py<'py>(
    py: Python<'py>,
    ev: &TextReactEvent,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    match ev {
        TextReactEvent::IterationStart { iteration } => {
            d.set_item("type", "iteration_start")?;
            d.set_item("iteration", *iteration)?;
        }
        TextReactEvent::LlmResponse { iteration, text } => {
            d.set_item("type", "llm_response")?;
            d.set_item("iteration", *iteration)?;
            d.set_item("text", text)?;
        }
        TextReactEvent::ParsedAction {
            iteration,
            thought,
            tool,
            input,
        } => {
            d.set_item("type", "parsed_action")?;
            d.set_item("iteration", *iteration)?;
            d.set_item("thought", thought.clone())?;
            d.set_item("tool", tool)?;
            d.set_item("input", json_to_py(py, input)?)?;
        }
        TextReactEvent::ParsedFinal {
            iteration,
            thought,
            answer,
        } => {
            d.set_item("type", "parsed_final")?;
            d.set_item("iteration", *iteration)?;
            d.set_item("thought", thought.clone())?;
            d.set_item("answer", answer)?;
        }
        TextReactEvent::ParseError {
            iteration,
            error,
            raw_response,
        } => {
            d.set_item("type", "parse_error")?;
            d.set_item("iteration", *iteration)?;
            d.set_item("error", error)?;
            d.set_item("raw_response", raw_response)?;
        }
        TextReactEvent::ToolStart {
            iteration,
            tool,
            input,
        } => {
            d.set_item("type", "tool_start")?;
            d.set_item("iteration", *iteration)?;
            d.set_item("tool", tool)?;
            d.set_item("input", json_to_py(py, input)?)?;
        }
        TextReactEvent::ToolResult {
            iteration,
            tool,
            observation,
            is_error,
            duration_ms,
        } => {
            d.set_item("type", "tool_result")?;
            d.set_item("iteration", *iteration)?;
            d.set_item("tool", tool)?;
            d.set_item("observation", observation)?;
            d.set_item("is_error", *is_error)?;
            d.set_item("duration_ms", *duration_ms)?;
        }
        TextReactEvent::Final { answer, iterations } => {
            d.set_item("type", "final")?;
            d.set_item("answer", answer)?;
            d.set_item("iterations", *iterations)?;
        }
        TextReactEvent::MaxIterations { iterations } => {
            d.set_item("type", "max_iterations")?;
            d.set_item("iterations", *iterations)?;
        }
        TextReactEvent::ToolNotFound {
            iteration,
            tool,
            available,
        } => {
            d.set_item("type", "tool_not_found")?;
            d.set_item("iteration", *iteration)?;
            d.set_item("tool", tool)?;
            d.set_item("available", available.clone())?;
        }
    }
    Ok(d)
}

fn text_react_turn_to_py<'py>(
    py: Python<'py>,
    turn: &TextReactTurn,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    match turn {
        TextReactTurn::Action {
            raw_response,
            thought,
            tool,
            input,
            observation,
            is_error,
        } => {
            d.set_item("kind", "action")?;
            d.set_item("raw_response", raw_response)?;
            d.set_item("thought", thought.clone())?;
            d.set_item("tool", tool)?;
            d.set_item("input", json_to_py(py, input)?)?;
            d.set_item("observation", observation)?;
            d.set_item("is_error", *is_error)?;
        }
        TextReactTurn::Final {
            raw_response,
            thought,
            answer,
        } => {
            d.set_item("kind", "final")?;
            d.set_item("raw_response", raw_response)?;
            d.set_item("thought", thought.clone())?;
            d.set_item("answer", answer)?;
        }
        TextReactTurn::ParseError {
            raw_response,
            error,
        } => {
            d.set_item("kind", "parse_error")?;
            d.set_item("raw_response", raw_response)?;
            d.set_item("error", error)?;
        }
    }
    Ok(d)
}
