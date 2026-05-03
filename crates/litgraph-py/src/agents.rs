//! `ReactAgent` + `SupervisorAgent` Python bindings.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use litgraph_agents::{
    AgentEvent, PlanAndExecuteAgent, PlanAndExecuteConfig, ReactAgent, ReactAgentConfig,
    StoppedReason, SupervisorAgent, SupervisorConfig,
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
use crate::middleware::PyMiddlewareChat;
use crate::runtime::{block_on_compat, rt};
use crate::tools::{
    PyBraveSearchTool, PyCachedTool, PyCalculatorTool, PyDalleImageTool, PyDuckDuckGoSearchTool,
    PyFunctionTool, PyHttpRequestTool, PyListDirectoryTool, PyPythonReplTool, PyReadFileTool,
    PyGmailSendTool, PyPlanningTool, PyRetryTool, PyShellTool, PySqliteQueryTool,
    PySubagentTool, PyTavilyExtractTool, PyTavilySearchTool, PyTimeoutTool, PyTtsAudioTool,
    PyVirtualFilesystemTool, PyWebFetchTool, PyWebhookTool, PyWhisperTranscribeTool,
    PyWriteFileTool,
};
use crate::mcp::PyMcpTool;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReactAgent>()?;
    m.add_class::<PySupervisorAgent>()?;
    m.add_class::<PyPlanAndExecuteAgent>()?;
    m.add_class::<PyAgentEventStream>()?;
    m.add_class::<PyTextReActAgent>()?;
    m.add_class::<PyTextReactEventStream>()?;
    m.add_function(wrap_pyfunction!(batch_chat, m)?)?;
    m.add_class::<PyMultiplexStream>()?;
    m.add_function(wrap_pyfunction!(multiplex_chat_streams, m)?)?;
    m.add_function(wrap_pyfunction!(py_tool_dispatch_concurrent, m)?)?;
    m.add_class::<PyBroadcastHandle>()?;
    m.add_class::<PyBroadcastSubscriber>()?;
    m.add_function(wrap_pyfunction!(broadcast_chat_stream, m)?)?;
    Ok(())
}

/// Broadcast a single chat model's token stream to N concurrent
/// subscribers. Inverse of `multiplex_chat_streams` (iter 189): that
/// fans-IN N streams into one; broadcast fans-OUT one stream to N.
///
/// Use cases:
/// - Live UI + audit log subscribing to the same agent stream.
/// - Multi-pane debugger showing the same tokens to many clients.
/// - Sidecar policy evaluator watching tokens in flight.
///
/// All subscribers must call `.subscribe()` BEFORE the first iteration
/// hits the runtime — the pump is spawned on the first `subscribe()`,
/// so additional subscribers must register before yielding control.
///
/// ```python
/// from litgraph.agents import broadcast_chat_stream
/// handle = broadcast_chat_stream(model, [{"role": "user", "content": "hi"}], capacity=64)
/// sub_a = handle.subscribe()
/// sub_b = handle.subscribe()
/// # Now drain both. Each receives every event.
/// for ev in sub_a:
///     print("a:", ev.get("text", ""))
/// ```
#[pyfunction]
#[pyo3(signature = (model, messages, capacity=64, temperature=None, max_tokens=None))]
fn broadcast_chat_stream(
    py: Python<'_>,
    model: Bound<'_, PyAny>,
    messages: Bound<'_, PyList>,
    capacity: usize,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> PyResult<Py<PyBroadcastHandle>> {
    let chat = extract_chat_model(&model)?;
    let msgs = crate::providers::parse_messages_from_pylist(&messages)?;
    let opts = ChatOptions {
        temperature,
        max_tokens,
        ..Default::default()
    };

    // Build an "upstream" by spawning model.stream() inside the
    // broadcast pump. We need a `ChatStream` here — call
    // `model.stream(...)` once and pass the resulting stream into
    // `litgraph_core::broadcast_chat_stream`.
    let upstream_future = async move { chat.stream(msgs, &opts).await };
    // Resolve the stream synchronously on this thread so we have a
    // concrete `ChatStream` to pass in. `model.stream` returns the
    // stream future immediately; awaiting once gives us the actual
    // Stream impl.
    let upstream = py.allow_threads(|| {
        block_on_compat(upstream_future)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })?;

    let handle = litgraph_core::broadcast_chat_stream(upstream, capacity)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Py::new(
        py,
        PyBroadcastHandle {
            inner: Arc::new(handle),
        },
    )
}

/// Handle to a configured broadcast. Call `.subscribe()` to obtain
/// a fresh subscriber iterator. First `.subscribe()` call spawns the
/// pump.
#[pyclass(name = "BroadcastHandle", module = "litgraph.agents")]
pub struct PyBroadcastHandle {
    inner: Arc<litgraph_core::BroadcastHandle>,
}

#[pymethods]
impl PyBroadcastHandle {
    fn subscribe<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBroadcastSubscriber>> {
        let stream = self.inner.subscribe();
        // Bridge: drain the BroadcastSubscriberStream into an
        // mpsc::Receiver that Python can `__next__` block on.
        let (tx, rx) = mpsc::channel::<litgraph_core::BroadcastEvent>(64);
        crate::runtime::rt().spawn(async move {
            let mut s = stream;
            while let Some(ev) = s.next().await {
                if tx.send(ev).await.is_err() {
                    break;
                }
            }
        });
        Py::new(
            py,
            PyBroadcastSubscriber {
                rx: Arc::new(Mutex::new(Some(rx))),
            },
        )
    }

    fn receiver_count(&self) -> usize {
        self.inner.receiver_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "BroadcastHandle(receivers={})",
            self.inner.receiver_count()
        )
    }
}

/// One Python iterator over a broadcast subscriber's events. Each
/// `__next__` returns a dict; iteration ends with `StopIteration` when
/// the upstream pump finishes.
#[pyclass(name = "BroadcastSubscriber", module = "litgraph.agents")]
pub struct PyBroadcastSubscriber {
    rx: Arc<Mutex<Option<mpsc::Receiver<litgraph_core::BroadcastEvent>>>>,
}

#[pymethods]
impl PyBroadcastSubscriber {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let slot = self.rx.clone();
        let item = py.allow_threads(|| {
            let mut guard = slot.lock().expect("poisoned");
            let mut rx = match guard.take() {
                Some(r) => r,
                None => return None,
            };
            let got = block_on_compat(async { rx.recv().await });
            *guard = Some(rx);
            got
        });
        match item {
            Some(ev) => broadcast_event_to_py(py, &ev),
            None => {
                *self.rx.lock().expect("poisoned") = None;
                Err(PyStopIteration::new_err("broadcast subscriber exhausted"))
            }
        }
    }
}

fn broadcast_event_to_py<'py>(
    py: Python<'py>,
    ev: &litgraph_core::BroadcastEvent,
) -> PyResult<Bound<'py, PyDict>> {
    use litgraph_core::{BroadcastEvent, ChatStreamEvent};
    let d = PyDict::new_bound(py);
    match ev {
        BroadcastEvent::Lagged { skipped } => {
            d.set_item("type", "lagged")?;
            d.set_item("skipped", skipped)?;
        }
        BroadcastEvent::Event(Err(msg)) => {
            d.set_item("type", "error")?;
            d.set_item("error", msg)?;
        }
        BroadcastEvent::Event(Ok(ChatStreamEvent::Delta { text })) => {
            d.set_item("type", "delta")?;
            d.set_item("text", text)?;
        }
        BroadcastEvent::Event(Ok(ChatStreamEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments_delta,
        })) => {
            d.set_item("type", "tool_call_delta")?;
            d.set_item("index", index)?;
            d.set_item("id", id.clone())?;
            d.set_item("name", name.clone())?;
            d.set_item("arguments_delta", arguments_delta.clone())?;
        }
        BroadcastEvent::Event(Ok(ChatStreamEvent::Done { response })) => {
            d.set_item("type", "done")?;
            d.set_item("text", response.message.text_content())?;
            d.set_item(
                "finish_reason",
                format!("{:?}", response.finish_reason).to_lowercase(),
            )?;
            d.set_item("model", &response.model)?;
        }
    }
    Ok(d)
}

/// Run N tool calls concurrently, capped at `max_concurrency` in
/// flight. Each call routes to a tool by name and gets its own
/// `Result` slot — a failed or unknown-tool call doesn't tank the
/// rest by default. Pass `fail_fast=True` to raise on the first
/// error instead.
///
/// `tools` is a list of any registered tool pyclass (the same set
/// `ReactAgent` accepts). `calls` is a list of `{"name", "args"}`
/// dicts (or `(name, args)` tuples) — one per intended invocation.
/// Output is aligned 1:1 with `calls`: each slot is either the
/// tool's JSON result or, on per-call failure (fail_fast=False), a
/// dict `{"error": "..."}`.
///
/// Useful outside the React agent loop — Plan-and-Execute style
/// orchestrators, custom batch-tool drivers, eval harnesses.
///
/// ```python
/// from litgraph.tools import CalculatorTool, HttpRequestTool
/// from litgraph.agents import tool_dispatch_concurrent
/// tools = [CalculatorTool(), HttpRequestTool()]
/// calls = [
///     {"name": "calculator", "args": {"expr": "1+1"}},
///     {"name": "http_request", "args": {"url": "https://api.example.com/x"}},
/// ]
/// results = tool_dispatch_concurrent(tools, calls, max_concurrency=4)
/// ```
#[pyfunction]
#[pyo3(name = "tool_dispatch_concurrent", signature = (tools, calls, max_concurrency=8, fail_fast=false))]
fn py_tool_dispatch_concurrent<'py>(
    py: Python<'py>,
    tools: Bound<'py, PyList>,
    calls: Bound<'py, PyList>,
    max_concurrency: usize,
    fail_fast: bool,
) -> PyResult<Bound<'py, PyList>> {
    let tool_vec = extract_tools(&tools)?;
    let mut registry: HashMap<String, Arc<dyn litgraph_core::tool::Tool>> = HashMap::new();
    for t in tool_vec {
        registry.insert(t.name(), t);
    }

    let mut parsed_calls: Vec<(String, serde_json::Value)> = Vec::with_capacity(calls.len());
    for item in calls.iter() {
        // Accept either (name, args) tuple OR {"name", "args"} dict.
        let (name, args): (String, serde_json::Value) =
            if let Ok(t) = item.downcast::<pyo3::types::PyTuple>() {
                if t.len() != 2 {
                    return Err(PyValueError::new_err(
                        "tuple call must be (name, args)",
                    ));
                }
                let name: String = t.get_item(0)?.extract()?;
                let args = crate::graph::py_to_json(t.py(), &t.get_item(1)?)?;
                (name, args)
            } else if let Ok(d) = item.downcast::<PyDict>() {
                let name: String = d
                    .get_item("name")?
                    .ok_or_else(|| PyValueError::new_err("missing 'name'"))?
                    .extract()?;
                let args = match d.get_item("args")? {
                    Some(v) => crate::graph::py_to_json(d.py(), &v)?,
                    None => serde_json::Value::Object(serde_json::Map::new()),
                };
                (name, args)
            } else {
                return Err(PyValueError::new_err(
                    "each call must be a (name, args) tuple or a {name, args} dict",
                ));
            };
        parsed_calls.push((name, args));
    }

    let results = py.allow_threads(|| {
        block_on_compat(async move {
            Ok::<_, litgraph_core::Error>(
                litgraph_core::tool_dispatch_concurrent(
                    registry,
                    parsed_calls,
                    max_concurrency,
                )
                .await,
            )
        })
        .map_err(|e: litgraph_core::Error| PyRuntimeError::new_err(e.to_string()))
    })?;

    let out = PyList::empty_bound(py);
    for r in results {
        match r {
            Ok(v) => out.append(crate::graph::json_to_py(py, &v)?)?,
            Err(e) => {
                if fail_fast {
                    return Err(PyRuntimeError::new_err(e.to_string()));
                }
                let d = PyDict::new_bound(py);
                d.set_item("error", e.to_string())?;
                out.append(d)?;
            }
        }
    }
    Ok(out)
}

/// Run N chat-model streams concurrently, fan-in token deltas via a
/// Tokio mpsc channel, return a Python iterator of tagged events.
/// Each `__next__` yields a dict `{model_label, type, text?, error?,
/// finish_reason?, ...}` — see the inner `ChatStreamEvent` discriminants.
///
/// `models` is a list of `(label_str, chat_model)` tuples. Each model
/// receives the same `messages` + (temperature, max_tokens) options.
///
/// Distinct from `RaceChat` (iter 184) — race returns one winning
/// response. Multiplex returns **all** models' streams interleaved
/// in arrival order, useful for live side-by-side rendering.
///
/// ```python
/// from litgraph.agents import multiplex_chat_streams
/// for ev in multiplex_chat_streams(
///     [("openai", openai_model), ("anthropic", anth_model)],
///     [{"role": "user", "content": "hi"}],
/// ):
///     print(ev["model_label"], ev.get("text", ""))
/// ```
#[pyfunction]
#[pyo3(signature = (models, messages, temperature=None, max_tokens=None))]
fn multiplex_chat_streams(
    py: Python<'_>,
    models: Bound<'_, PyList>,
    messages: Bound<'_, PyList>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> PyResult<Py<PyMultiplexStream>> {
    if models.is_empty() {
        return Err(PyValueError::new_err(
            "multiplex_chat_streams: need at least one (label, model) pair",
        ));
    }
    let mut parsed: Vec<(String, Arc<dyn ChatModel>)> = Vec::with_capacity(models.len());
    for item in models.iter() {
        // Accept either (label, model) tuple OR a 2-element list.
        let pair: pyo3::Bound<pyo3::types::PyTuple> = match item.downcast::<pyo3::types::PyTuple>() {
            Ok(t) => t.clone(),
            Err(_) => {
                return Err(PyValueError::new_err(
                    "each model entry must be a (label, model) tuple",
                ));
            }
        };
        if pair.len() != 2 {
            return Err(PyValueError::new_err(
                "each tuple must be exactly (label, model)",
            ));
        }
        let label: String = pair.get_item(0)?.extract()?;
        let model_obj = pair.get_item(1)?;
        let model = extract_chat_model(&model_obj)?;
        parsed.push((label, model));
    }

    let msgs = crate::providers::parse_messages_from_pylist(&messages)?;
    let opts = ChatOptions {
        temperature,
        max_tokens,
        ..Default::default()
    };

    let cap = (parsed.len() * 16).max(8);
    let (tx, rx) = mpsc::channel::<litgraph_core::Result<litgraph_core::MultiplexEvent>>(cap);
    crate::runtime::rt().spawn(async move {
        let mut s = litgraph_core::multiplex_chat_streams(parsed, msgs, opts);
        while let Some(item) = s.next().await {
            if tx.send(item).await.is_err() {
                break;
            }
        }
    });
    Py::new(
        py,
        PyMultiplexStream {
            rx: Arc::new(Mutex::new(Some(rx))),
        },
    )
}

/// Iterator over multiplexed chat-stream events. Each `__next__`
/// returns a dict; iteration ends with `StopIteration` once every
/// inner model has finished.
#[pyclass(name = "MultiplexStream", module = "litgraph.agents")]
pub struct PyMultiplexStream {
    rx: Arc<Mutex<Option<mpsc::Receiver<litgraph_core::Result<litgraph_core::MultiplexEvent>>>>>,
}

#[pymethods]
impl PyMultiplexStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let slot = self.rx.clone();
        let item = py.allow_threads(|| {
            let mut guard = slot.lock().expect("poisoned");
            let mut rx = match guard.take() {
                Some(r) => r,
                None => return None,
            };
            let got = block_on_compat(async { rx.recv().await });
            *guard = Some(rx);
            got
        });
        match item {
            Some(Ok(ev)) => multiplex_event_to_py(py, &ev),
            Some(Err(e)) => {
                // Per-model failure surfaces as a tagged-error dict so
                // the iterator keeps going. Embedded label is in the
                // error text (set by `multiplex_chat_streams`).
                let d = PyDict::new_bound(py);
                let msg = e.to_string();
                // Try to recover the [label] tag; otherwise fall back.
                let (label, body) = match (msg.find('['), msg.find(']')) {
                    (Some(l), Some(r)) if r > l + 1 => {
                        (msg[l + 1..r].to_string(), msg[r + 1..].trim().to_string())
                    }
                    _ => ("?".into(), msg.clone()),
                };
                d.set_item("type", "error")?;
                d.set_item("model_label", label)?;
                d.set_item("error", body)?;
                Ok(d)
            }
            None => {
                *self.rx.lock().expect("poisoned") = None;
                Err(PyStopIteration::new_err("multiplex stream exhausted"))
            }
        }
    }
}

fn multiplex_event_to_py<'py>(
    py: Python<'py>,
    ev: &litgraph_core::MultiplexEvent,
) -> PyResult<Bound<'py, PyDict>> {
    use litgraph_core::ChatStreamEvent;
    let d = PyDict::new_bound(py);
    d.set_item("model_label", &ev.model_label)?;
    match &ev.event {
        ChatStreamEvent::Delta { text } => {
            d.set_item("type", "delta")?;
            d.set_item("text", text)?;
        }
        ChatStreamEvent::ToolCallDelta { index, id, name, arguments_delta } => {
            d.set_item("type", "tool_call_delta")?;
            d.set_item("index", index)?;
            d.set_item("id", id.clone())?;
            d.set_item("name", name.clone())?;
            d.set_item("arguments_delta", arguments_delta.clone())?;
        }
        ChatStreamEvent::Done { response } => {
            d.set_item("type", "done")?;
            d.set_item("text", response.message.text_content())?;
            d.set_item(
                "finish_reason",
                format!("{:?}", response.finish_reason).to_lowercase(),
            )?;
            d.set_item("model", &response.model)?;
        }
    }
    Ok(d)
}

/// Concurrent batched chat — fan out N invocations across a Tokio task
/// pool capped at `max_concurrency` in flight. Output order matches
/// input order. Per-call failures are isolated by default; pass
/// `fail_fast=True` to raise on the first error and cancel the rest.
///
/// Each input is a list of `{"role", "content"}` dicts (same shape as
/// `model.invoke`). Each output is a dict `{text, finish_reason, usage,
/// model}` for successes or `{error: "..."}` for failures (when
/// `fail_fast=False`).
///
/// ```python
/// from litgraph.agents import batch_chat
/// inputs = [
///     [{"role": "user", "content": "translate hello to french"}],
///     [{"role": "user", "content": "translate hello to spanish"}],
///     [{"role": "user", "content": "translate hello to german"}],
/// ]
/// results = batch_chat(model, inputs, max_concurrency=8)
/// for r in results:
///     print(r.get("text") or f"err: {r['error']}")
/// ```
#[pyfunction]
#[pyo3(signature = (model, inputs, max_concurrency=8, fail_fast=false))]
fn batch_chat<'py>(
    py: Python<'py>,
    model: Bound<'py, PyAny>,
    inputs: Bound<'py, PyList>,
    max_concurrency: usize,
    fail_fast: bool,
) -> PyResult<Bound<'py, PyList>> {
    let chat = extract_chat_model(&model)?;
    let mut parsed: Vec<Vec<Message>> = Vec::with_capacity(inputs.len());
    for item in inputs.iter() {
        let lst: Bound<PyList> = item
            .downcast_into()
            .map_err(|_| PyValueError::new_err("each input must be a list of message dicts"))?;
        parsed.push(crate::providers::parse_messages_from_pylist(&lst)?);
    }

    if fail_fast {
        let resps = py.allow_threads(|| {
            block_on_compat(async move {
                litgraph_core::batch_concurrent_fail_fast(
                    chat,
                    parsed,
                    ChatOptions::default(),
                    max_concurrency,
                )
                .await
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let out = PyList::empty_bound(py);
        for r in resps {
            out.append(crate::providers::response_to_py_dict(py, &r)?)?;
        }
        return Ok(out);
    }

    let results = py.allow_threads(|| {
        block_on_compat(async move {
            Ok::<_, litgraph_core::Error>(
                litgraph_core::batch_concurrent(
                    chat,
                    parsed,
                    ChatOptions::default(),
                    max_concurrency,
                )
                .await,
            )
        })
        .map_err(|e: litgraph_core::Error| PyRuntimeError::new_err(e.to_string()))
    })?;

    let out = PyList::empty_bound(py);
    for r in results {
        match r {
            Ok(resp) => out.append(crate::providers::response_to_py_dict(py, &resp)?)?,
            Err(e) => {
                let d = PyDict::new_bound(py);
                d.set_item("error", e.to_string())?;
                out.append(d)?;
            }
        }
    }
    Ok(out)
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
        } else if let Ok(tt) = item.extract::<PyRef<PyTimeoutTool>>() {
            tool_vec.push(tt.as_tool());
        } else if let Ok(rt) = item.extract::<PyRef<PyRetryTool>>() {
            tool_vec.push(rt.as_tool());
        } else if let Ok(p) = item.extract::<PyRef<PyPlanningTool>>() {
            tool_vec.push(p.as_tool());
        } else if let Ok(v) = item.extract::<PyRef<PyVirtualFilesystemTool>>() {
            tool_vec.push(v.as_tool());
        } else if let Ok(s) = item.extract::<PyRef<PySubagentTool>>() {
            tool_vec.push(s.as_tool());
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
    } else if let Ok(rc) = bound.extract::<PyRef<crate::providers::PyRaceChat>>() {
        Ok(rc.chat_model())
    } else if let Ok(tc) = bound.extract::<PyRef<crate::providers::PyTimeoutChat>>() {
        Ok(tc.chat_model())
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
    } else if let Ok(mw) = bound.extract::<PyRef<PyMiddlewareChat>>() {
        Ok(mw.chat_model())
    } else {
        Err(PyValueError::new_err(
            "model must be OpenAIChat, OpenAIResponses, AnthropicChat, GeminiChat, BedrockChat, CohereChat, StructuredChatModel, FallbackChat, TokenBudgetChat, PiiScrubbingChat, PromptCachingChat, CostCappedChat, SelfConsistencyChat, or MiddlewareChat",
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
            } else if let Ok(tt) = item.extract::<PyRef<PyTimeoutTool>>() {
                tool_vec.push(tt.as_tool());
            } else if let Ok(rt) = item.extract::<PyRef<PyRetryTool>>() {
                tool_vec.push(rt.as_tool());
            } else if let Ok(p) = item.extract::<PyRef<PyPlanningTool>>() {
                tool_vec.push(p.as_tool());
            } else if let Ok(v) = item.extract::<PyRef<PyVirtualFilesystemTool>>() {
                tool_vec.push(v.as_tool());
            } else if let Ok(s) = item.extract::<PyRef<PySubagentTool>>() {
                tool_vec.push(s.as_tool());
            } else {
                return Err(PyValueError::new_err(
                    "tools must be FunctionTool, BraveSearchTool, TavilySearchTool, \
                     TavilyExtractTool, DuckDuckGoSearchTool, CalculatorTool, HttpRequestTool, \
                     ReadFileTool, WriteFileTool, ListDirectoryTool, ShellTool, SqliteQueryTool, \
                     WhisperTranscribeTool, DalleImageTool, TtsAudioTool, CachedTool, \
                     PythonReplTool, WebhookTool, PlanningTool, VirtualFilesystemTool, or McpTool",
                ));
            }
        }

        let cfg = ReactAgentConfig {
            system_prompt,
            max_iterations,
            chat_options: ChatOptions::default(),
            max_parallel_tools: 16,
            tool_middleware: litgraph_agents::middleware::ToolMiddlewareChain::new(),
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

/// Plan-and-Execute agent. Two-phase: planner LLM emits a numbered list
/// of steps; per-step ReactAgent worker executes each in sequence.
/// Different model for planner vs executor allows cheap-planner /
/// capable-executor cost optimization.
///
/// ```python
/// from litgraph.agents import PlanAndExecuteAgent
/// from litgraph.providers import OpenAIChat
///
/// planner  = OpenAIChat(api_key=..., model="gpt-4o-mini")  # cheap
/// executor = OpenAIChat(api_key=..., model="gpt-4o")       # capable
/// agent = PlanAndExecuteAgent(
///     planner=planner, executor=executor, tools=[],
///     max_steps=5, max_iterations_per_step=4,
/// )
/// result = agent.invoke("Research X then write a brief.")
/// print(result["plan"])          # ["step1", "step2", ...]
/// print(result["steps"])         # [{"step": ..., "output": ...}, ...]
/// print(result["final_answer"])  # last step's output
/// ```
#[pyclass(name = "PlanAndExecuteAgent", module = "litgraph.agents")]
pub struct PyPlanAndExecuteAgent {
    inner: PlanAndExecuteAgent,
}

#[pymethods]
impl PyPlanAndExecuteAgent {
    #[new]
    #[pyo3(signature = (
        planner,
        executor=None,
        tools=None,
        planner_system=None,
        executor_system=None,
        max_steps=7,
        max_iterations_per_step=5,
    ))]
    fn new(
        planner: Bound<'_, PyAny>,
        executor: Option<Bound<'_, PyAny>>,
        tools: Option<Bound<'_, pyo3::types::PyList>>,
        planner_system: Option<String>,
        executor_system: Option<String>,
        max_steps: usize,
        max_iterations_per_step: u32,
    ) -> PyResult<Self> {
        let planner_arc = extract_chat_model(&planner)?;
        let executor_arc = match executor {
            Some(e) => extract_chat_model(&e)?,
            None => planner_arc.clone(),
        };
        let mut tool_vec: Vec<Arc<dyn litgraph_core::tool::Tool>> = Vec::new();
        if let Some(tlist) = tools {
            for item in tlist.iter() {
                if let Ok(t) = item.extract::<PyRef<crate::tools::PyFunctionTool>>() {
                    tool_vec.push(t.as_tool());
                } else {
                    let arc = crate::tools::extract_tool_arc(&item)?;
                    tool_vec.push(arc);
                }
            }
        }
        let mut cfg = PlanAndExecuteConfig::default();
        if let Some(s) = planner_system {
            cfg.planner_system = s;
        }
        cfg.executor_system = executor_system;
        cfg.max_steps = max_steps;
        cfg.max_iterations_per_step = max_iterations_per_step;
        Ok(Self {
            inner: PlanAndExecuteAgent::new(planner_arc, executor_arc, tool_vec, cfg),
        })
    }

    fn invoke<'py>(
        &self,
        py: Python<'py>,
        task: String,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let result = py.allow_threads(|| {
            crate::runtime::block_on_compat(async { self.inner.invoke(task).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let d = pyo3::types::PyDict::new_bound(py);
        let plan_list = pyo3::types::PyList::empty_bound(py);
        for s in &result.plan {
            plan_list.append(s)?;
        }
        d.set_item("plan", plan_list)?;
        let steps_list = pyo3::types::PyList::empty_bound(py);
        for s in &result.steps {
            let sd = pyo3::types::PyDict::new_bound(py);
            sd.set_item("step", &s.step)?;
            sd.set_item("output", &s.output)?;
            steps_list.append(sd)?;
        }
        d.set_item("steps", steps_list)?;
        d.set_item("final_answer", &result.final_answer)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        "PlanAndExecuteAgent()".into()
    }
}
