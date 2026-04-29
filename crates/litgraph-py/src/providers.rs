//! Python-facing provider bindings. Wraps the core `ChatModel` trait implementors
//! and exposes them as Python classes. Provides `.with_cache(cache)` and
//! `.instrument(tracker)` sugar that chain `CachedModel` + `InstrumentedChatModel`
//! around the provider without re-wrapping on every call.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use litgraph_cache::{CachedModel, SemanticCachedModel};
use litgraph_resilience::{
    CostCappedChatModel, FallbackChatModel, PiiScrubbingChatModel, PromptCachingChatModel,
    RateLimitConfig, RateLimitedChatModel, RetryConfig, RetryingChatModel,
    SelfConsistencyChatModel, TokenBudgetChatModel,
};
use litgraph_core::model::ChatStreamEvent;
use litgraph_core::{ChatModel, ChatOptions, Message, PiiScrubber, Role};
use litgraph_observability::{CallbackBus, InstrumentedChatModel};
use litgraph_providers_anthropic::{AnthropicChat, AnthropicConfig};
use litgraph_providers_bedrock::{AwsCredentials, BedrockChat, BedrockConfig, BedrockConverseChat};
use litgraph_providers_cohere::{CohereChat, CohereChatConfig};
use litgraph_providers_gemini::{GeminiChat, GeminiConfig};
use litgraph_providers_openai::{OpenAIChat, OpenAIConfig, OpenAIResponses, OpenAIResponsesConfig};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::sync::mpsc;

use crate::cache::{PyMemoryCache, PySemanticCache, PySqliteCache};
use crate::observability::PyCostTracker;
use crate::runtime::{block_on_compat, rt};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOpenAIChat>()?;
    m.add_class::<PyOpenAIResponses>()?;
    m.add_class::<PyAnthropicChat>()?;
    m.add_class::<PyGeminiChat>()?;
    m.add_class::<PyBedrockChat>()?;
    m.add_class::<PyBedrockConverseChat>()?;
    m.add_class::<PyCohereChat>()?;
    m.add_class::<PyChatStream>()?;
    m.add_class::<PyStructuredChatModel>()?;
    m.add_class::<PyFallbackChat>()?;
    m.add_class::<PyTokenBudgetChat>()?;
    m.add_class::<PyPiiScrubbingChat>()?;
    m.add_class::<PyPromptCachingChat>()?;
    m.add_class::<PyCostCappedChat>()?;
    m.add_class::<PySelfConsistencyChat>()?;
    m.add_function(wrap_pyfunction!(with_structured_output, m)?)?;
    m.add_function(wrap_pyfunction!(ollama_chat, m)?)?;
    m.add_function(wrap_pyfunction!(groq_chat, m)?)?;
    m.add_function(wrap_pyfunction!(together_chat, m)?)?;
    m.add_function(wrap_pyfunction!(mistral_chat, m)?)?;
    m.add_function(wrap_pyfunction!(deepseek_chat, m)?)?;
    m.add_function(wrap_pyfunction!(xai_chat, m)?)?;
    m.add_function(wrap_pyfunction!(fireworks_chat, m)?)?;
    Ok(())
}

#[pyclass(name = "CohereChat", module = "litgraph.providers")]
pub struct PyCohereChat {
    inner: Arc<dyn ChatModel>,
    model: String,
    bus_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
}

#[pymethods]
impl PyCohereChat {
    #[new]
    #[pyo3(signature = (api_key, model, base_url=None, timeout_s=120, on_request=None))]
    fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        timeout_s: u64,
        on_request: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut cfg = CohereChatConfig::new(api_key, &model);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        if let Some(cb) = on_request {
            cfg = cfg.with_on_request(wrap_py_inspector(cb));
        }
        let chat = CohereChat::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(chat), model, bus_handle: None })
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let chat = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { chat.invoke(msgs, &opts).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn stream<'py>(
        &self,
        _py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<PyChatStream> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let chat = self.inner.clone();
        let rx = spawn_stream(chat, msgs, opts);
        Ok(PyChatStream { rx: Arc::new(Mutex::new(Some(rx))) })
    }

    fn with_cache(&mut self, py: Python<'_>, cache: Bound<'_, PyAny>) -> PyResult<()> {
        let backend = unify_cache_backend(py, &cache)?;
        self.inner = Arc::new(CachedModel::new(self.inner.clone(), backend));
        Ok(())
    }

    fn with_semantic_cache(&mut self, cache: PyRef<'_, PySemanticCache>) -> PyResult<()> {
        self.inner = Arc::new(SemanticCachedModel::new(self.inner.clone(), cache.inner.clone()));
        Ok(())
    }

    #[pyo3(signature = (max_times=5, min_delay_ms=200, max_delay_ms=30_000, factor=2.0, jitter=true))]
    fn with_retry(&mut self, max_times: usize, min_delay_ms: u64, max_delay_ms: u64,
                  factor: f32, jitter: bool) -> PyResult<()> {
        let cfg = RetryConfig {
            min_delay: std::time::Duration::from_millis(min_delay_ms),
            max_delay: std::time::Duration::from_millis(max_delay_ms),
            factor, max_times, jitter,
        };
        self.inner = Arc::new(RetryingChatModel::new(self.inner.clone(), cfg));
        Ok(())
    }

    /// Token-bucket rate limit. `requests_per_minute` sets steady-state rate;
    /// `burst` (default = rpm) is the bucket capacity for idle accumulation.
    /// Set `burst=1` for a strict non-bursty cadence.
    #[pyo3(signature = (requests_per_minute, burst=None))]
    fn with_rate_limit(&mut self, requests_per_minute: u32, burst: Option<u32>) -> PyResult<()> {
        let mut cfg = RateLimitConfig::per_minute(requests_per_minute);
        if let Some(b) = burst { cfg = cfg.with_burst(b); }
        self.inner = Arc::new(RateLimitedChatModel::new(self.inner.clone(), cfg));
        Ok(())
    }

    fn instrument(&mut self, tracker: PyRef<'_, PyCostTracker>) -> PyResult<()> {
        let bus = CallbackBus::new();
        bus.subscribe(tracker.inner.clone());
        let _g = rt().enter();
        let (handle, join) = bus.start();
        self.inner = Arc::new(InstrumentedChatModel::new(self.inner.clone(), handle));
        self.bus_handle = Some(Arc::new(join));
        Ok(())
    }

    fn __repr__(&self) -> String { format!("CohereChat(model='{}')", self.model) }
}

impl PyCohereChat {
    pub(crate) fn chat_model(&self) -> std::sync::Arc<dyn litgraph_core::ChatModel> {
        self.inner.clone()
    }
}

/// Convenience constructor for a local Ollama server. Same as
/// `OpenAIChat(api_key="ollama", model=..., base_url="http://localhost:11434/v1")`.
///
/// Ollama implements the OpenAI-compatible API, so all `OpenAIChat` methods
/// (`invoke`, `stream`, `with_cache`, `instrument`, …) work unchanged.
#[pyfunction]
#[pyo3(signature = (model, base_url=None, timeout_s=120))]
fn ollama_chat(
    model: String,
    base_url: Option<String>,
    timeout_s: u64,
) -> PyResult<PyOpenAIChat> {
    let url = base_url.unwrap_or_else(|| "http://localhost:11434/v1".into());
    PyOpenAIChat::new("ollama".into(), model, Some(url), timeout_s, None)
}

/// Groq — OpenAI-compatible inference for Llama / Mixtral / Whisper running
/// on Groq's LPU. Default: `https://api.groq.com/openai/v1`.
#[pyfunction]
#[pyo3(signature = (api_key, model, base_url=None, timeout_s=120))]
fn groq_chat(
    api_key: String,
    model: String,
    base_url: Option<String>,
    timeout_s: u64,
) -> PyResult<PyOpenAIChat> {
    let url = base_url.unwrap_or_else(|| "https://api.groq.com/openai/v1".into());
    PyOpenAIChat::new(api_key, model, Some(url), timeout_s, None)
}

/// Together AI — OpenAI-compatible serverless inference for hundreds of OSS
/// models. Default: `https://api.together.xyz/v1`.
#[pyfunction]
#[pyo3(signature = (api_key, model, base_url=None, timeout_s=120))]
fn together_chat(
    api_key: String,
    model: String,
    base_url: Option<String>,
    timeout_s: u64,
) -> PyResult<PyOpenAIChat> {
    let url = base_url.unwrap_or_else(|| "https://api.together.xyz/v1".into());
    PyOpenAIChat::new(api_key, model, Some(url), timeout_s, None)
}

/// Mistral La Plateforme — OpenAI-compatible. Default:
/// `https://api.mistral.ai/v1`. Common models: `mistral-large-latest`,
/// `mistral-small-latest`, `codestral-latest`, `pixtral-large-latest`.
#[pyfunction]
#[pyo3(signature = (api_key, model, base_url=None, timeout_s=120))]
fn mistral_chat(
    api_key: String,
    model: String,
    base_url: Option<String>,
    timeout_s: u64,
) -> PyResult<PyOpenAIChat> {
    let url = base_url.unwrap_or_else(|| "https://api.mistral.ai/v1".into());
    PyOpenAIChat::new(api_key, model, Some(url), timeout_s, None)
}

/// DeepSeek — OpenAI-compatible. Default: `https://api.deepseek.com/v1`.
/// Common models: `deepseek-chat`, `deepseek-reasoner`.
#[pyfunction]
#[pyo3(signature = (api_key, model, base_url=None, timeout_s=120))]
fn deepseek_chat(
    api_key: String,
    model: String,
    base_url: Option<String>,
    timeout_s: u64,
) -> PyResult<PyOpenAIChat> {
    let url = base_url.unwrap_or_else(|| "https://api.deepseek.com/v1".into());
    PyOpenAIChat::new(api_key, model, Some(url), timeout_s, None)
}

/// xAI Grok — OpenAI-compatible. Default: `https://api.x.ai/v1`.
/// Common models: `grok-2-latest`, `grok-2-vision-latest`.
#[pyfunction]
#[pyo3(signature = (api_key, model, base_url=None, timeout_s=120))]
fn xai_chat(
    api_key: String,
    model: String,
    base_url: Option<String>,
    timeout_s: u64,
) -> PyResult<PyOpenAIChat> {
    let url = base_url.unwrap_or_else(|| "https://api.x.ai/v1".into());
    PyOpenAIChat::new(api_key, model, Some(url), timeout_s, None)
}

/// Fireworks AI — OpenAI-compatible serverless inference for OSS models.
/// Default: `https://api.fireworks.ai/inference/v1`.
#[pyfunction]
#[pyo3(signature = (api_key, model, base_url=None, timeout_s=120))]
fn fireworks_chat(
    api_key: String,
    model: String,
    base_url: Option<String>,
    timeout_s: u64,
) -> PyResult<PyOpenAIChat> {
    let url = base_url.unwrap_or_else(|| "https://api.fireworks.ai/inference/v1".into());
    PyOpenAIChat::new(api_key, model, Some(url), timeout_s, None)
}

/// Wrap a Python callable as a Rust `Fn(&str, &serde_json::Value)` request
/// inspector. Closure re-acquires the GIL inside and dispatches to the
/// callable with `(model: str, body: dict)`. Errors raised inside the callable
/// are logged via tracing — never want a debug hook to crash the request path.
fn wrap_py_inspector(cb: Py<PyAny>) -> impl Fn(&str, &serde_json::Value) + Send + Sync + 'static {
    let cb = Arc::new(cb);
    move |model: &str, body: &serde_json::Value| {
        let cb = cb.clone();
        let model = model.to_string();
        let body = body.clone();
        Python::with_gil(|py| {
            let py_body = match crate::graph::json_to_py(py, &body) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(error = %e, "on_request: failed to convert body to Python");
                    return;
                }
            };
            if let Err(e) = cb.call1(py, (model, py_body)) {
                tracing::warn!(error = %e, "on_request: Python callback raised");
            }
        });
    }
}

/// Extract the underlying `Arc<dyn ChatModel>` from whichever provider subtype
/// was passed. Central point for future providers (Gemini, Bedrock).
fn unify_cache_backend(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn litgraph_cache::Cache>> {
    if let Ok(c) = obj.extract::<PyRef<PyMemoryCache>>() {
        return Ok(c.inner.clone());
    }
    if let Ok(c) = obj.extract::<PyRef<PySqliteCache>>() {
        return Ok(c.inner.clone());
    }
    let _ = py;
    Err(PyValueError::new_err("expected MemoryCache or SqliteCache"))
}

#[pyclass(name = "OpenAIChat", module = "litgraph.providers")]
pub struct PyOpenAIChat {
    inner: Arc<dyn ChatModel>,
    model: String,
    /// Keeps the callback bus alive as long as the instrumented model is used.
    bus_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
}

#[pymethods]
impl PyOpenAIChat {
    /// `on_request` — optional Python callable `(model: str, body: dict) -> None`
    /// invoked just before each HTTP request. Use to log / snapshot / assert
    /// the final wire body. Solves "what is the model actually seeing?" without
    /// monkey-patching reqwest. Errors raised inside the callback propagate as
    /// Rust panics by design.
    #[new]
    #[pyo3(signature = (api_key, model, base_url=None, timeout_s=120, on_request=None))]
    fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        timeout_s: u64,
        on_request: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut cfg = OpenAIConfig::new(api_key, &model);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url {
            cfg = cfg.with_base_url(url);
        }
        if let Some(cb) = on_request {
            cfg = cfg.with_on_request(wrap_py_inspector(cb));
        }
        let chat = OpenAIChat::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(chat), model, bus_handle: None })
    }

    /// Invoke. Optional `response_format` enables structured output. Pass either:
    /// - `{"type": "json_object"}` — model returns valid JSON, no schema constraint
    /// - `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}` — model adheres to schema
    /// Provider maps this to OpenAI's native `response_format` field.
    #[pyo3(signature = (messages, temperature=None, max_tokens=None, response_format=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        response_format: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let rf = response_format
            .map(|d| crate::graph::py_dict_to_json(&d))
            .transpose()?;
        let opts = ChatOptions { temperature, max_tokens, response_format: rf, ..Default::default() };
        let chat = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { chat.invoke(msgs, &opts).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    /// Streaming — returns a `ChatStream` iterator. Each item is a dict:
    /// - `{"type": "delta", "text": "..."}` — partial assistant text
    /// - `{"type": "tool_call_delta", "index": i, ...}` — streaming tool call
    /// - `{"type": "done", ...}` — final response (text, finish_reason, usage)
    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn stream<'py>(
        &self,
        _py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<PyChatStream> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let chat = self.inner.clone();
        let rx = spawn_stream(chat, msgs, opts);
        Ok(PyChatStream { rx: Arc::new(Mutex::new(Some(rx))) })
    }

    /// Wrap this provider with a Cache. Subsequent invocations hit the cache.
    fn with_cache(&mut self, py: Python<'_>, cache: Bound<'_, PyAny>) -> PyResult<()> {
        let backend = unify_cache_backend(py, &cache)?;
        self.inner = Arc::new(CachedModel::new(self.inner.clone(), backend));
        Ok(())
    }

    /// Wrap with a SemanticCache — looks up responses by embedding cosine.
    /// Use a conservative threshold (≥ 0.95) in production.
    fn with_semantic_cache(&mut self, cache: PyRef<'_, PySemanticCache>) -> PyResult<()> {
        self.inner = Arc::new(SemanticCachedModel::new(self.inner.clone(), cache.inner.clone()));
        Ok(())
    }

    /// Wrap with retry + jittered exponential backoff. Retries 429 (rate-limited),
    /// timeouts, and 5xx responses. Does NOT retry 4xx (client bug) or stream calls.
    #[pyo3(signature = (max_times=5, min_delay_ms=200, max_delay_ms=30_000, factor=2.0, jitter=true))]
    fn with_retry(
        &mut self,
        max_times: usize,
        min_delay_ms: u64,
        max_delay_ms: u64,
        factor: f32,
        jitter: bool,
    ) -> PyResult<()> {
        let cfg = RetryConfig {
            min_delay: std::time::Duration::from_millis(min_delay_ms),
            max_delay: std::time::Duration::from_millis(max_delay_ms),
            factor,
            max_times,
            jitter,
        };
        self.inner = Arc::new(RetryingChatModel::new(self.inner.clone(), cfg));
        Ok(())
    }

    /// Token-bucket rate limit. `requests_per_minute` sets steady-state rate;
    /// `burst` (default = rpm) is the bucket capacity for idle accumulation.
    /// Set `burst=1` for a strict non-bursty cadence.
    #[pyo3(signature = (requests_per_minute, burst=None))]
    fn with_rate_limit(&mut self, requests_per_minute: u32, burst: Option<u32>) -> PyResult<()> {
        let mut cfg = RateLimitConfig::per_minute(requests_per_minute);
        if let Some(b) = burst { cfg = cfg.with_burst(b); }
        self.inner = Arc::new(RateLimitedChatModel::new(self.inner.clone(), cfg));
        Ok(())
    }

    /// Wrap this provider so it emits events to a `CostTracker` (and any other
    /// subscribers on the bus). The bus owns a drain task that lives as long as
    /// the returned provider.
    fn instrument(&mut self, tracker: PyRef<'_, PyCostTracker>) -> PyResult<()> {
        let bus = CallbackBus::new();
        bus.subscribe(tracker.inner.clone());
        let _g = rt().enter();
        let (handle, join) = bus.start();
        self.inner = Arc::new(InstrumentedChatModel::new(self.inner.clone(), handle));
        self.bus_handle = Some(Arc::new(join));
        Ok(())
    }

    fn __repr__(&self) -> String { format!("OpenAIChat(model='{}')", self.model) }
}

impl PyOpenAIChat {
    pub(crate) fn chat_model(&self) -> std::sync::Arc<dyn litgraph_core::ChatModel> {
        self.inner.clone()
    }
}

/// OpenAI's `/v1/responses` endpoint — the new agentic chat API. Same
/// `ChatModel` surface as `OpenAIChat`; supports server-side stateful
/// chains via `previous_response_id` and a top-level `instructions` field.
/// Streaming is not yet implemented (the Responses SSE event taxonomy
/// differs from chat completions); use `invoke()`.
///
/// ```python
/// from litgraph.providers import OpenAIResponses
/// chat = OpenAIResponses(api_key=..., model="gpt-4o",
///                        instructions="Be terse.",
///                        previous_response_id="resp_abc")  # optional
/// out = chat.invoke([{"role": "user", "content": "hi"}])
/// ```
#[pyclass(name = "OpenAIResponses", module = "litgraph.providers")]
pub struct PyOpenAIResponses {
    inner: Arc<dyn ChatModel>,
    model: String,
}

#[pymethods]
impl PyOpenAIResponses {
    #[new]
    #[pyo3(signature = (api_key, model, base_url=None, timeout_s=120,
                        instructions=None, previous_response_id=None,
                        on_request=None))]
    fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        timeout_s: u64,
        instructions: Option<String>,
        previous_response_id: Option<String>,
        on_request: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut cfg = OpenAIResponsesConfig::new(api_key, &model);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        if let Some(s) = instructions { cfg = cfg.with_instructions(s); }
        if let Some(id) = previous_response_id { cfg = cfg.with_previous_response_id(id); }
        if let Some(cb) = on_request { cfg = cfg.with_on_request(wrap_py_inspector(cb)); }
        let chat = OpenAIResponses::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(chat), model })
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let chat = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { chat.invoke(msgs, &opts).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    fn __repr__(&self) -> String { format!("OpenAIResponses(model='{}')", self.model) }
}

impl PyOpenAIResponses {
    pub(crate) fn chat_model(&self) -> std::sync::Arc<dyn litgraph_core::ChatModel> {
        self.inner.clone()
    }
}

#[pyclass(name = "AnthropicChat", module = "litgraph.providers")]
pub struct PyAnthropicChat {
    inner: Arc<dyn ChatModel>,
    model: String,
    bus_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
}

#[pymethods]
impl PyAnthropicChat {
    #[new]
    #[pyo3(signature = (api_key, model, base_url=None, timeout_s=120, max_tokens=4096, on_request=None, thinking_budget=None))]
    fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        timeout_s: u64,
        max_tokens: u32,
        on_request: Option<Py<PyAny>>,
        thinking_budget: Option<u32>,
    ) -> PyResult<Self> {
        let mut cfg = AnthropicConfig::new(api_key, &model);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        cfg.max_tokens = max_tokens;
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        if let Some(cb) = on_request {
            cfg = cfg.with_on_request(wrap_py_inspector(cb));
        }
        if thinking_budget.is_some() {
            cfg = cfg.with_thinking(thinking_budget);
        }
        let chat = AnthropicChat::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(chat), model, bus_handle: None })
    }

    /// Invoke. `response_format = {"type":"json_schema","json_schema":{"schema":{...}}}`
    /// triggers Anthropic's tool-call workaround: a synthesized tool with that
    /// schema is forced via tool_choice, and the response is unwrapped back
    /// into the text field. `{"type":"json_object"}` is no-op for Anthropic
    /// (no native equivalent — use a system prompt to ask for JSON).
    #[pyo3(signature = (messages, temperature=None, max_tokens=None, response_format=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        response_format: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let rf = response_format
            .map(|d| crate::graph::py_dict_to_json(&d))
            .transpose()?;
        let opts = ChatOptions { temperature, max_tokens, response_format: rf, ..Default::default() };
        let chat = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { chat.invoke(msgs, &opts).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn stream<'py>(
        &self,
        _py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<PyChatStream> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let chat = self.inner.clone();
        let rx = spawn_stream(chat, msgs, opts);
        Ok(PyChatStream { rx: Arc::new(Mutex::new(Some(rx))) })
    }

    fn with_cache(&mut self, py: Python<'_>, cache: Bound<'_, PyAny>) -> PyResult<()> {
        let backend = unify_cache_backend(py, &cache)?;
        self.inner = Arc::new(CachedModel::new(self.inner.clone(), backend));
        Ok(())
    }

    /// Wrap with a SemanticCache — looks up responses by embedding cosine.
    /// Use a conservative threshold (≥ 0.95) in production.
    fn with_semantic_cache(&mut self, cache: PyRef<'_, PySemanticCache>) -> PyResult<()> {
        self.inner = Arc::new(SemanticCachedModel::new(self.inner.clone(), cache.inner.clone()));
        Ok(())
    }

    #[pyo3(signature = (max_times=5, min_delay_ms=200, max_delay_ms=30_000, factor=2.0, jitter=true))]
    fn with_retry(&mut self, max_times: usize, min_delay_ms: u64, max_delay_ms: u64,
                  factor: f32, jitter: bool) -> PyResult<()> {
        let cfg = RetryConfig {
            min_delay: std::time::Duration::from_millis(min_delay_ms),
            max_delay: std::time::Duration::from_millis(max_delay_ms),
            factor, max_times, jitter,
        };
        self.inner = Arc::new(RetryingChatModel::new(self.inner.clone(), cfg));
        Ok(())
    }

    #[pyo3(signature = (requests_per_minute, burst=None))]
    fn with_rate_limit(&mut self, requests_per_minute: u32, burst: Option<u32>) -> PyResult<()> {
        let mut cfg = RateLimitConfig::per_minute(requests_per_minute);
        if let Some(b) = burst { cfg = cfg.with_burst(b); }
        self.inner = Arc::new(RateLimitedChatModel::new(self.inner.clone(), cfg));
        Ok(())
    }

    fn instrument(&mut self, tracker: PyRef<'_, PyCostTracker>) -> PyResult<()> {
        let bus = CallbackBus::new();
        bus.subscribe(tracker.inner.clone());
        let _g = rt().enter();
        let (handle, join) = bus.start();
        self.inner = Arc::new(InstrumentedChatModel::new(self.inner.clone(), handle));
        self.bus_handle = Some(Arc::new(join));
        Ok(())
    }

    fn __repr__(&self) -> String { format!("AnthropicChat(model='{}')", self.model) }
}

impl PyAnthropicChat {
    pub(crate) fn chat_model(&self) -> std::sync::Arc<dyn litgraph_core::ChatModel> {
        self.inner.clone()
    }
}

#[pyclass(name = "GeminiChat", module = "litgraph.providers")]
pub struct PyGeminiChat {
    inner: Arc<dyn ChatModel>,
    model: String,
    bus_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
}

#[pymethods]
impl PyGeminiChat {
    #[new]
    #[pyo3(signature = (api_key, model, base_url=None, timeout_s=120, on_request=None))]
    fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        timeout_s: u64,
        on_request: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut cfg = GeminiConfig::new(api_key, &model);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        if let Some(cb) = on_request {
            cfg = cfg.with_on_request(wrap_py_inspector(cb));
        }
        let chat = GeminiChat::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(chat), model, bus_handle: None })
    }

    /// Vertex AI mode — enterprise-grade Gemini via Google Cloud. Uses
    /// OAuth2 Bearer auth (from a service account / ADC / gcloud), the
    /// region-specific `{location}-aiplatform.googleapis.com` endpoint,
    /// and a project+location-scoped URL path. Required by orgs with
    /// VPC Service Controls / Cloud Audit / per-request IAM requirements.
    ///
    /// `access_token` is a Google OAuth2 token — caller mints it
    /// externally (this adapter does NOT refresh). Typical sources:
    /// `gcloud auth print-access-token`, Application Default Credentials,
    /// workload identity on GKE/GCE, or a signed JWT → token exchange.
    ///
    /// ```python
    /// from litgraph.providers import GeminiChat
    /// chat = GeminiChat.vertex(
    ///     project="my-gcp-project",
    ///     location="us-central1",
    ///     access_token="ya29.a0A...",
    ///     model="gemini-1.5-pro",
    /// )
    /// out = chat.invoke([{"role":"user","content":"hi"}])
    /// ```
    #[staticmethod]
    #[pyo3(signature = (project, location, access_token, model, base_url=None, timeout_s=120, on_request=None))]
    fn vertex(
        project: String,
        location: String,
        access_token: String,
        model: String,
        base_url: Option<String>,
        timeout_s: u64,
        on_request: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut cfg = GeminiConfig::new_vertex(project, location, access_token, &model);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        if let Some(url) = base_url { cfg = cfg.with_base_url(url); }
        if let Some(cb) = on_request {
            cfg = cfg.with_on_request(wrap_py_inspector(cb));
        }
        let chat = GeminiChat::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(chat), model, bus_handle: None })
    }

    /// Invoke. Optional `response_format` enables structured output. Mapped to
    /// Gemini's `generationConfig.responseMimeType` + `responseSchema`.
    #[pyo3(signature = (messages, temperature=None, max_tokens=None, response_format=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        response_format: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let rf = response_format
            .map(|d| crate::graph::py_dict_to_json(&d))
            .transpose()?;
        let opts = ChatOptions { temperature, max_tokens, response_format: rf, ..Default::default() };
        let chat = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { chat.invoke(msgs, &opts).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn stream<'py>(
        &self,
        _py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<PyChatStream> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let chat = self.inner.clone();
        let rx = spawn_stream(chat, msgs, opts);
        Ok(PyChatStream { rx: Arc::new(Mutex::new(Some(rx))) })
    }

    fn with_cache(&mut self, py: Python<'_>, cache: Bound<'_, PyAny>) -> PyResult<()> {
        let backend = unify_cache_backend(py, &cache)?;
        self.inner = Arc::new(CachedModel::new(self.inner.clone(), backend));
        Ok(())
    }

    /// Wrap with a SemanticCache — looks up responses by embedding cosine.
    /// Use a conservative threshold (≥ 0.95) in production.
    fn with_semantic_cache(&mut self, cache: PyRef<'_, PySemanticCache>) -> PyResult<()> {
        self.inner = Arc::new(SemanticCachedModel::new(self.inner.clone(), cache.inner.clone()));
        Ok(())
    }

    #[pyo3(signature = (max_times=5, min_delay_ms=200, max_delay_ms=30_000, factor=2.0, jitter=true))]
    fn with_retry(&mut self, max_times: usize, min_delay_ms: u64, max_delay_ms: u64,
                  factor: f32, jitter: bool) -> PyResult<()> {
        let cfg = RetryConfig {
            min_delay: std::time::Duration::from_millis(min_delay_ms),
            max_delay: std::time::Duration::from_millis(max_delay_ms),
            factor, max_times, jitter,
        };
        self.inner = Arc::new(RetryingChatModel::new(self.inner.clone(), cfg));
        Ok(())
    }

    #[pyo3(signature = (requests_per_minute, burst=None))]
    fn with_rate_limit(&mut self, requests_per_minute: u32, burst: Option<u32>) -> PyResult<()> {
        let mut cfg = RateLimitConfig::per_minute(requests_per_minute);
        if let Some(b) = burst { cfg = cfg.with_burst(b); }
        self.inner = Arc::new(RateLimitedChatModel::new(self.inner.clone(), cfg));
        Ok(())
    }

    fn instrument(&mut self, tracker: PyRef<'_, PyCostTracker>) -> PyResult<()> {
        let bus = CallbackBus::new();
        bus.subscribe(tracker.inner.clone());
        let _g = rt().enter();
        let (handle, join) = bus.start();
        self.inner = Arc::new(InstrumentedChatModel::new(self.inner.clone(), handle));
        self.bus_handle = Some(Arc::new(join));
        Ok(())
    }

    fn __repr__(&self) -> String { format!("GeminiChat(model='{}')", self.model) }
}

impl PyGeminiChat {
    pub(crate) fn chat_model(&self) -> std::sync::Arc<dyn litgraph_core::ChatModel> {
        self.inner.clone()
    }
}

#[pyclass(name = "BedrockChat", module = "litgraph.providers")]
pub struct PyBedrockChat {
    inner: Arc<dyn ChatModel>,
    model_id: String,
    bus_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
}

#[pymethods]
impl PyBedrockChat {
    /// `access_key_id` / `secret_access_key` / `session_token` are static creds.
    /// For instance-profile / SSO / IRSA, resolve creds with the AWS SDK in your
    /// app and pass the materialized values here.
    #[new]
    #[pyo3(signature = (
        access_key_id, secret_access_key, region, model_id,
        session_token=None, timeout_s=120, max_tokens=4096,
        endpoint_override=None, on_request=None,
    ))]
    fn new(
        access_key_id: String,
        secret_access_key: String,
        region: String,
        model_id: String,
        session_token: Option<String>,
        timeout_s: u64,
        max_tokens: u32,
        endpoint_override: Option<String>,
        on_request: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let creds = AwsCredentials { access_key_id, secret_access_key, session_token };
        let mut cfg = BedrockConfig::new(creds, region, &model_id);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        cfg.max_tokens = max_tokens;
        if let Some(url) = endpoint_override { cfg = cfg.with_endpoint(url); }
        if let Some(cb) = on_request {
            cfg = cfg.with_on_request(wrap_py_inspector(cb));
        }
        let chat = BedrockChat::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(chat), model_id, bus_handle: None })
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let chat = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { chat.invoke(msgs, &opts).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    /// Bedrock streaming returns a single `Done` event in v1 (event-stream
    /// frame parser is non-trivial; track the GH issue for status). Provided
    /// for API parity so users can swap providers without rewriting consumers.
    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn stream<'py>(
        &self,
        _py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<PyChatStream> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let chat = self.inner.clone();
        let rx = spawn_stream(chat, msgs, opts);
        Ok(PyChatStream { rx: Arc::new(Mutex::new(Some(rx))) })
    }

    fn with_cache(&mut self, py: Python<'_>, cache: Bound<'_, PyAny>) -> PyResult<()> {
        let backend = unify_cache_backend(py, &cache)?;
        self.inner = Arc::new(CachedModel::new(self.inner.clone(), backend));
        Ok(())
    }

    fn with_semantic_cache(&mut self, cache: PyRef<'_, PySemanticCache>) -> PyResult<()> {
        self.inner = Arc::new(SemanticCachedModel::new(self.inner.clone(), cache.inner.clone()));
        Ok(())
    }

    #[pyo3(signature = (max_times=5, min_delay_ms=200, max_delay_ms=30_000, factor=2.0, jitter=true))]
    fn with_retry(&mut self, max_times: usize, min_delay_ms: u64, max_delay_ms: u64,
                  factor: f32, jitter: bool) -> PyResult<()> {
        let cfg = RetryConfig {
            min_delay: std::time::Duration::from_millis(min_delay_ms),
            max_delay: std::time::Duration::from_millis(max_delay_ms),
            factor, max_times, jitter,
        };
        self.inner = Arc::new(RetryingChatModel::new(self.inner.clone(), cfg));
        Ok(())
    }

    #[pyo3(signature = (requests_per_minute, burst=None))]
    fn with_rate_limit(&mut self, requests_per_minute: u32, burst: Option<u32>) -> PyResult<()> {
        let mut cfg = RateLimitConfig::per_minute(requests_per_minute);
        if let Some(b) = burst { cfg = cfg.with_burst(b); }
        self.inner = Arc::new(RateLimitedChatModel::new(self.inner.clone(), cfg));
        Ok(())
    }

    fn instrument(&mut self, tracker: PyRef<'_, PyCostTracker>) -> PyResult<()> {
        let bus = CallbackBus::new();
        bus.subscribe(tracker.inner.clone());
        let _g = rt().enter();
        let (handle, join) = bus.start();
        self.inner = Arc::new(InstrumentedChatModel::new(self.inner.clone(), handle));
        self.bus_handle = Some(Arc::new(join));
        Ok(())
    }

    fn __repr__(&self) -> String { format!("BedrockChat(model_id='{}')", self.model_id) }
}

impl PyBedrockChat {
    pub(crate) fn chat_model(&self) -> std::sync::Arc<dyn litgraph_core::ChatModel> {
        self.inner.clone()
    }
}

/// AWS Bedrock Converse API — works across ALL model families (Llama, Titan,
/// Mistral, Command, Nova, Anthropic) via one unified wire format. Direct
/// Bedrock `/converse` parity. Use this when you want to swap between model
/// families without rewriting prompt-adapter code.
///
/// Streaming not yet implemented — call `invoke()`.
///
/// ```python
/// from litgraph.providers import BedrockConverseChat
/// chat = BedrockConverseChat(
///     access_key_id="AKIA...",
///     secret_access_key="...",
///     region="us-east-1",
///     model_id="meta.llama3-70b-instruct-v1:0",
/// )
/// out = chat.invoke([{"role": "user", "content": "hello"}])
/// ```
#[pyclass(name = "BedrockConverseChat", module = "litgraph.providers")]
pub struct PyBedrockConverseChat {
    inner: Arc<dyn ChatModel>,
    model: String,
}

#[pymethods]
impl PyBedrockConverseChat {
    #[new]
    #[pyo3(signature = (
        access_key_id, secret_access_key, region, model_id,
        session_token=None, endpoint_override=None, timeout_s=120, max_tokens=4096,
        on_request=None,
    ))]
    fn new(
        access_key_id: String,
        secret_access_key: String,
        region: String,
        model_id: String,
        session_token: Option<String>,
        endpoint_override: Option<String>,
        timeout_s: u64,
        max_tokens: u32,
        on_request: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let creds = AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token,
        };
        let mut cfg = BedrockConfig::new(creds, region, &model_id);
        cfg.timeout = std::time::Duration::from_secs(timeout_s);
        cfg.max_tokens = max_tokens;
        if let Some(url) = endpoint_override {
            cfg = cfg.with_endpoint(url);
        }
        if let Some(cb) = on_request {
            cfg = cfg.with_on_request(wrap_py_inspector(cb));
        }
        let chat = BedrockConverseChat::new(cfg)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(chat), model: model_id })
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let chat = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { chat.invoke(msgs, &opts).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    fn __repr__(&self) -> String {
        format!("BedrockConverseChat(model='{}')", self.model)
    }
}

impl PyBedrockConverseChat {
    pub(crate) fn chat_model(&self) -> Arc<dyn ChatModel> {
        self.inner.clone()
    }
}

/// Python-facing chat stream. Each `__next__` blocks on the receiver with GIL
/// released, converts one `ChatStreamEvent` to a dict.
#[pyclass(name = "ChatStream", module = "litgraph.providers")]
pub struct PyChatStream {
    rx: Arc<Mutex<Option<mpsc::Receiver<litgraph_core::Result<ChatStreamEvent>>>>>,
}

#[pymethods]
impl PyChatStream {
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
            Some(Ok(e)) => chat_event_to_py_dict(py, &e),
            Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
            None => {
                *self.rx.lock().expect("poisoned") = None;
                Err(PyStopIteration::new_err("chat stream exhausted"))
            }
        }
    }
}

/// Spawn the provider `stream()` on our runtime, forward events into an mpsc.
fn spawn_stream(
    chat: Arc<dyn ChatModel>,
    messages: Vec<Message>,
    opts: ChatOptions,
) -> mpsc::Receiver<litgraph_core::Result<ChatStreamEvent>> {
    let (tx, rx) = mpsc::channel::<litgraph_core::Result<ChatStreamEvent>>(64);
    rt().spawn(async move {
        match chat.stream(messages, &opts).await {
            Err(e) => { let _ = tx.send(Err(e)).await; }
            Ok(mut s) => {
                while let Some(item) = s.next().await {
                    if tx.send(item).await.is_err() { break; }
                }
            }
        }
    });
    rx
}

fn chat_event_to_py_dict<'py>(py: Python<'py>, ev: &ChatStreamEvent) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    match ev {
        ChatStreamEvent::Delta { text } => {
            d.set_item("type", "delta")?;
            d.set_item("text", text)?;
        }
        ChatStreamEvent::ToolCallDelta { index, id, name, arguments_delta } => {
            d.set_item("type", "tool_call_delta")?;
            d.set_item("index", *index)?;
            if let Some(v) = id { d.set_item("id", v)?; }
            if let Some(v) = name { d.set_item("name", v)?; }
            if let Some(v) = arguments_delta { d.set_item("arguments_delta", v)?; }
        }
        ChatStreamEvent::Done { response } => {
            d.set_item("type", "done")?;
            d.set_item("text", response.message.text_content())?;
            d.set_item("finish_reason", format!("{:?}", response.finish_reason).to_lowercase())?;
            let usage = PyDict::new_bound(py);
            usage.set_item("prompt", response.usage.prompt)?;
            usage.set_item("completion", response.usage.completion)?;
            usage.set_item("total", response.usage.total)?;
            d.set_item("usage", usage)?;
            d.set_item("model", &response.model)?;
        }
    }
    Ok(d)
}

pub(crate) fn response_to_py_dict<'py>(
    py: Python<'py>,
    resp: &litgraph_core::ChatResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    out.set_item("text", resp.message.text_content())?;
    out.set_item("finish_reason", format!("{:?}", resp.finish_reason).to_lowercase())?;
    let usage = PyDict::new_bound(py);
    usage.set_item("prompt", resp.usage.prompt)?;
    usage.set_item("completion", resp.usage.completion)?;
    usage.set_item("total", resp.usage.total)?;
    usage.set_item("cache_creation", resp.usage.cache_creation)?;
    usage.set_item("cache_read", resp.usage.cache_read)?;
    out.set_item("usage", usage)?;
    out.set_item("model", &resp.model)?;
    Ok(out)
}

/// Public re-export so other modules (memory, agents, …) can reuse the same
/// dict→Message conversion without duplicating it.
pub(crate) fn parse_messages_from_pylist(py_msgs: &Bound<'_, PyList>) -> PyResult<Vec<Message>> {
    parse_messages(py_msgs)
}

/// Render a slice of `Message` back to a Python list of `{role, content}` dicts.
/// Multimodal content collapses to its concatenated text view — sufficient for
/// the round-trip use case (memory snapshots, callback payloads). For an
/// invoke() the full structured form is preserved internally regardless.
pub(crate) fn messages_to_py_list<'py>(
    py: Python<'py>,
    msgs: &[Message],
) -> PyResult<Bound<'py, PyList>> {
    use pyo3::types::PyDict;
    let list = PyList::empty_bound(py);
    for m in msgs {
        let d = PyDict::new_bound(py);
        let role = match m.role {
            litgraph_core::Role::System => "system",
            litgraph_core::Role::User => "user",
            litgraph_core::Role::Assistant => "assistant",
            litgraph_core::Role::Tool => "tool",
        };
        d.set_item("role", role)?;
        d.set_item("content", m.text_content())?;
        if m.cache { d.set_item("cache", true)?; }
        if let Some(id) = &m.tool_call_id { d.set_item("tool_call_id", id)?; }
        list.append(d)?;
    }
    Ok(list)
}

fn parse_messages(py_msgs: &Bound<'_, PyList>) -> PyResult<Vec<Message>> {
    use litgraph_core::{ContentPart, ImageSource};
    let mut out = Vec::with_capacity(py_msgs.len());
    for item in py_msgs.iter() {
        let d: Bound<PyDict> = item.downcast_into()
            .map_err(|_| PyValueError::new_err("message must be dict"))?;
        let role_s: String = d
            .get_item("role")?
            .ok_or_else(|| PyValueError::new_err("missing 'role'"))?
            .extract()?;
        let role = match role_s.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => return Err(PyValueError::new_err(format!("bad role: {other}"))),
        };
        let content_obj = d
            .get_item("content")?
            .ok_or_else(|| PyValueError::new_err("missing 'content'"))?;

        // Accept either:
        //   1) plain string  → single Text part
        //   2) list of dicts → multimodal parts
        //      {"type": "text", "text": "..."}
        //      {"type": "image_url", "image_url": {"url": "..."}}
        //      {"type": "image", "image": {"media_type": "...", "data": "..."}}
        let cache: bool = d
            .get_item("cache")?
            .map(|v| v.extract::<bool>().unwrap_or(false))
            .unwrap_or(false);

        if let Ok(s) = content_obj.extract::<String>() {
            let mut m = Message::text(role, s);
            m.cache = cache;
            out.push(m);
            continue;
        }
        let parts_list: Bound<PyList> = content_obj.downcast_into()
            .map_err(|_| PyValueError::new_err("'content' must be a string or list of part dicts"))?;
        let mut parts: Vec<ContentPart> = Vec::with_capacity(parts_list.len());
        for raw_part in parts_list.iter() {
            let pd: Bound<PyDict> = raw_part.downcast_into()
                .map_err(|_| PyValueError::new_err("each content part must be a dict"))?;
            let ptype: String = pd
                .get_item("type")?
                .ok_or_else(|| PyValueError::new_err("part missing 'type'"))?
                .extract()?;
            match ptype.as_str() {
                "text" => {
                    let t: String = pd
                        .get_item("text")?
                        .ok_or_else(|| PyValueError::new_err("text part missing 'text'"))?
                        .extract()?;
                    parts.push(ContentPart::Text { text: t });
                }
                "image_url" => {
                    let inner: Bound<PyDict> = pd
                        .get_item("image_url")?
                        .ok_or_else(|| PyValueError::new_err("image_url part missing 'image_url'"))?
                        .downcast_into()
                        .map_err(|_| PyValueError::new_err("'image_url' must be a dict"))?;
                    let url: String = inner
                        .get_item("url")?
                        .ok_or_else(|| PyValueError::new_err("image_url missing 'url'"))?
                        .extract()?;
                    parts.push(ContentPart::Image { source: ImageSource::Url { url } });
                }
                "image" => {
                    let inner: Bound<PyDict> = pd
                        .get_item("image")?
                        .ok_or_else(|| PyValueError::new_err("image part missing 'image'"))?
                        .downcast_into()
                        .map_err(|_| PyValueError::new_err("'image' must be a dict"))?;
                    let media_type: String = inner
                        .get_item("media_type")?
                        .ok_or_else(|| PyValueError::new_err("image missing 'media_type'"))?
                        .extract()?;
                    let data: String = inner
                        .get_item("data")?
                        .ok_or_else(|| PyValueError::new_err("image missing 'data'"))?
                        .extract()?;
                    parts.push(ContentPart::Image { source: ImageSource::Base64 { media_type, data } });
                }
                other => return Err(PyValueError::new_err(format!("unknown part type: {other}"))),
            }
        }
        out.push(Message {
            role,
            content: parts,
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            cache,
        });
    }
    Ok(out)
}

// ---------- StructuredChatModel (iter 89) ----------

use litgraph_core::StructuredChatModel;

/// Wraps a chat model with a JSON Schema. `invoke_structured()` returns
/// the parsed dict directly; `invoke()` returns a normal ChatResponse but
/// the text is guaranteed to be valid JSON matching the schema (errors
/// otherwise — fail fast at the wrapper, not in user code).
///
/// Direct LangChain `ChatModel.with_structured_output(schema)` parity.
///
/// ```python
/// from litgraph.providers import OpenAIChat, with_structured_output
/// chat = OpenAIChat(api_key=..., model="gpt-4o")
/// schema = {
///     "type": "object",
///     "properties": {
///         "name": {"type": "string"},
///         "age": {"type": "integer"},
///     },
///     "required": ["name", "age"],
/// }
/// structured = with_structured_output(chat, schema, name="Person")
/// out = structured.invoke_structured([{"role": "user", "content": "Ada Lovelace, age 36"}])
/// # out == {"name": "Ada Lovelace", "age": 36}
/// ```
#[pyclass(name = "StructuredChatModel", module = "litgraph.providers")]
pub struct PyStructuredChatModel {
    inner: Arc<StructuredChatModel>,
}

#[pymethods]
impl PyStructuredChatModel {
    /// Returns the parsed JSON dict directly. Errors when the LLM returns
    /// malformed JSON; the error message includes the schema name + raw
    /// response for debugging.
    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke_structured<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let m = self.inner.clone();
        let v = py.allow_threads(|| {
            block_on_compat(async move { m.invoke_structured(msgs, &opts).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        crate::graph::json_to_py(py, &v)
    }

    /// Standard `invoke()` — returns the normal ChatResponse dict. Text is
    /// guaranteed to be valid JSON matching the schema (errors otherwise).
    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let m = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { m.invoke(msgs, &opts).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    fn __repr__(&self) -> String {
        format!("StructuredChatModel(model='{}')", self.inner.name())
    }
}

impl PyStructuredChatModel {
    pub(crate) fn chat_model(&self) -> Arc<dyn ChatModel> {
        self.inner.clone() as Arc<dyn ChatModel>
    }
}

/// Wrap any chat model with a JSON Schema for structured output. `schema`
/// is a dict (raw JSON Schema) or a Pydantic model class (with
/// `model_json_schema()`).
///
/// `name` defaults to "Output" if not given. `strict=True` (default) makes
/// providers refuse non-conforming responses where supported (OpenAI gpt-4o+);
/// pass `strict=False` for OSS / older models that don't support strict mode.
#[pyfunction]
#[pyo3(signature = (model, schema, name=None, strict=true))]
fn with_structured_output<'py>(
    py: Python<'py>,
    model: Bound<'py, PyAny>,
    schema: Bound<'py, PyAny>,
    name: Option<String>,
    strict: bool,
) -> PyResult<PyStructuredChatModel> {
    let inner = crate::agents::extract_chat_model(&model)?;
    // Schema input: dict (raw schema), or class with model_json_schema (pydantic).
    let schema_value: serde_json::Value = if let Ok(d) = schema.downcast::<PyDict>() {
        crate::graph::py_dict_to_json(d)?
    } else if let Ok(method) = schema.getattr("model_json_schema") {
        // Pydantic v2: cls.model_json_schema() returns a dict.
        let result = method.call0()?;
        let d = result.downcast_into::<PyDict>().map_err(|_| {
            PyValueError::new_err("schema.model_json_schema() must return a dict")
        })?;
        crate::graph::py_dict_to_json(&d)?
    } else if let Ok(method) = schema.getattr("schema") {
        // Pydantic v1 fallback: cls.schema() returns a dict.
        let result = method.call0()?;
        let d = result.downcast_into::<PyDict>().map_err(|_| {
            PyValueError::new_err("schema.schema() must return a dict")
        })?;
        crate::graph::py_dict_to_json(&d)?
    } else {
        return Err(PyValueError::new_err(
            "schema must be a dict (JSON Schema), a Pydantic model class, or have a .schema()/.model_json_schema() method",
        ));
    };
    // Default schema name: pull from the schema's "title" if present, else "Output".
    let schema_name = name
        .or_else(|| schema_value.get("title").and_then(|v| v.as_str()).map(String::from))
        .unwrap_or_else(|| "Output".into());
    let _ = py;
    let s = StructuredChatModel::new(inner, schema_value, schema_name).with_strict(strict);
    Ok(PyStructuredChatModel { inner: Arc::new(s) })
}

/// Auto-failover chat model. Wraps an ordered list of chat models. On
/// transient failure (rate-limit / timeout / 5xx), routes to the next
/// model in the list. The LAST model's error is propagated.
///
/// Use for provider failover (GPT-4 primary, Claude backup), cost
/// shedding (GPT-4 primary, GPT-3.5 backup), or region failover.
///
/// `fall_through_on_all=True` falls through on EVERY error (including
/// 4xx / parse errors). Default `False` is safer because a malformed
/// request will likely fail the same way against backup providers.
///
/// ```python
/// from litgraph.providers import OpenAIChat, AnthropicChat, FallbackChat
/// chat = FallbackChat([
///     OpenAIChat(api_key=..., model="gpt-4o"),
///     AnthropicChat(api_key=..., model="claude-sonnet-4-6"),
/// ])
/// resp = chat.invoke([{"role": "user", "content": "hi"}])
/// ```
#[pyclass(name = "FallbackChat", module = "litgraph.providers")]
pub struct PyFallbackChat {
    inner: Arc<dyn ChatModel>,
    name_label: String,
}

#[pymethods]
impl PyFallbackChat {
    #[new]
    #[pyo3(signature = (models, fall_through_on_all=false))]
    fn new(models: Bound<'_, PyList>, fall_through_on_all: bool) -> PyResult<Self> {
        if models.is_empty() {
            return Err(PyValueError::new_err(
                "FallbackChat: chain must have at least one model",
            ));
        }
        let mut inners: Vec<Arc<dyn ChatModel>> = Vec::with_capacity(models.len());
        let mut labels: Vec<String> = Vec::with_capacity(models.len());
        for item in models.iter() {
            let m = crate::agents::extract_chat_model(&item)?;
            labels.push(m.name().to_string());
            inners.push(m);
        }
        let mut chain = FallbackChatModel::new(inners);
        if fall_through_on_all {
            chain = chain.fall_through_on_all();
        }
        let name_label = format!("fallback({})", labels.join(", "));
        Ok(Self {
            inner: Arc::new(chain),
            name_label,
        })
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions {
            temperature,
            max_tokens,
            ..Default::default()
        };
        let m = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { m.invoke(msgs, &opts).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    fn __repr__(&self) -> String {
        self.name_label.clone()
    }
}

impl PyFallbackChat {
    pub(crate) fn chat_model(&self) -> Arc<dyn ChatModel> {
        self.inner.clone()
    }
}

/// Token-budget wrapper. Enforces a per-invocation token cap to prevent
/// runaway history growth from blowing up LLM bills. Two modes:
///
///   - Strict (default): messages over budget raise `ValueError`.
///   - Auto-trim: drops oldest non-system messages until under budget.
///     System + last message always preserved.
///
/// ```python
/// from litgraph.providers import OpenAIChat, TokenBudgetChat
/// raw = OpenAIChat(api_key=..., model="gpt-4o-mini")
/// chat = TokenBudgetChat(raw, max_tokens=4096, auto_trim=True)
/// resp = chat.invoke(long_message_history)  # auto-trimmed if over 4096
/// ```
#[pyclass(name = "TokenBudgetChat", module = "litgraph.providers")]
pub struct PyTokenBudgetChat {
    inner: Arc<dyn ChatModel>,
}

#[pymethods]
impl PyTokenBudgetChat {
    #[new]
    #[pyo3(signature = (model, max_tokens, auto_trim=false))]
    fn new(model: Py<PyAny>, max_tokens: usize, auto_trim: bool) -> PyResult<Self> {
        let chat_model: Arc<dyn ChatModel> =
            Python::with_gil(|py| crate::agents::extract_chat_model(model.bind(py)))?;
        let mut wrapper = TokenBudgetChatModel::new(chat_model, max_tokens);
        if auto_trim {
            wrapper = wrapper.auto_trim();
        }
        Ok(Self { inner: Arc::new(wrapper) })
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let m = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { m.invoke(msgs, &opts).await })
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    fn __repr__(&self) -> String {
        format!("TokenBudgetChat(inner={})", self.inner.name())
    }
}

impl PyTokenBudgetChat {
    pub(crate) fn chat_model(&self) -> Arc<dyn ChatModel> {
        self.inner.clone()
    }
}

/// `PiiScrubbingChat` — wrap any `ChatModel` to redact PII before send.
///
/// Scrubs User role messages (and optionally System) with a `PiiScrubber`,
/// then forwards to the inner model. Assistant + Tool messages in the
/// history pass through untouched so agent traces stay intact.
///
/// - `scrub_system=False` by default (operator prompts trusted).
/// - `scrub_outputs=False` by default (response formatting preserved).
///
/// ```python
/// from litgraph.providers import OpenAIChat, PiiScrubbingChat
/// raw = OpenAIChat(api_key=..., model="gpt-4o-mini")
/// chat = PiiScrubbingChat(raw)           # scrubs outgoing user messages
/// chat.invoke([("user", "Email me at a@b.com")])  # sends "<EMAIL>"
/// ```
#[pyclass(name = "PiiScrubbingChat", module = "litgraph.providers")]
pub struct PyPiiScrubbingChat {
    inner: Arc<dyn ChatModel>,
}

#[pymethods]
impl PyPiiScrubbingChat {
    #[new]
    #[pyo3(signature = (model, scrubber=None, scrub_system=false, scrub_outputs=false))]
    fn new(
        model: Py<PyAny>,
        scrubber: Option<Py<PyAny>>,
        scrub_system: bool,
        scrub_outputs: bool,
    ) -> PyResult<Self> {
        let chat_model: Arc<dyn ChatModel> =
            Python::with_gil(|py| crate::agents::extract_chat_model(model.bind(py)))?;
        let scrub = Python::with_gil(|py| -> PyResult<PiiScrubber> {
            match scrubber {
                Some(obj) => {
                    let bound = obj.bind(py);
                    let s = bound.extract::<PyRef<crate::evaluators::PyPiiScrubber>>()?;
                    Ok(s.clone_inner())
                }
                None => Ok(PiiScrubber::new()),
            }
        })?;
        let mut wrapper = PiiScrubbingChatModel::new(chat_model)
            .with_scrubber(Arc::new(scrub));
        if scrub_system {
            wrapper = wrapper.with_system_scrubbing();
        }
        if scrub_outputs {
            wrapper = wrapper.with_output_scrubbing();
        }
        Ok(Self { inner: Arc::new(wrapper) })
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let m = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { m.invoke(msgs, &opts).await })
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    fn __repr__(&self) -> String {
        format!("PiiScrubbingChat(inner={})", self.inner.name())
    }
}

impl PyPiiScrubbingChat {
    pub(crate) fn chat_model(&self) -> Arc<dyn ChatModel> {
        self.inner.clone()
    }
}

/// `PromptCachingChat` — auto-mark messages as Anthropic prompt-cache breakpoints.
///
/// Wraps any `ChatModel`. On every invoke, sets `cache=True` on messages
/// matching the policy BEFORE forwarding. Anthropic/Bedrock-on-Anthropic
/// providers attach `cache_control: {"type":"ephemeral"}` to those messages'
/// last content blocks — cached input tokens cost ~0.1×. Other providers
/// ignore the flag, so stacking is safe.
///
/// - `cache_system=True` (default) — mark the first system message.
/// - `cache_last_user_if_over=N` — mark the LAST user message if its text
///   exceeds N bytes (long-context-in-user pattern).
/// - `cache_indices=[i, ...]` — manual: mark specific message indices.
///
/// Anthropic caps at 4 breakpoints per request; keep policies minimal.
///
/// ```python
/// from litgraph.providers import AnthropicChat, PromptCachingChat
/// raw = AnthropicChat(api_key=..., model="claude-opus-4-7")
/// chat = PromptCachingChat(raw, cache_last_user_if_over=4096)
/// chat.invoke([
///     {"role": "system", "content": "<long system prompt>"},
///     {"role": "user",   "content": "<long context>"},
/// ])
/// ```
#[pyclass(name = "PromptCachingChat", module = "litgraph.providers")]
pub struct PyPromptCachingChat {
    inner: Arc<dyn ChatModel>,
}

#[pymethods]
impl PyPromptCachingChat {
    #[new]
    #[pyo3(signature = (
        model,
        cache_system=true,
        cache_last_user_if_over=None,
        cache_indices=None,
    ))]
    fn new(
        model: Py<PyAny>,
        cache_system: bool,
        cache_last_user_if_over: Option<usize>,
        cache_indices: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let chat_model: Arc<dyn ChatModel> =
            Python::with_gil(|py| crate::agents::extract_chat_model(model.bind(py)))?;
        let mut wrapper = PromptCachingChatModel::new(chat_model);
        if !cache_system {
            wrapper = wrapper.without_system();
        }
        if let Some(b) = cache_last_user_if_over {
            wrapper = wrapper.cache_last_user_if_over(b);
        }
        if let Some(idxs) = cache_indices {
            wrapper = wrapper.cache_indices(idxs);
        }
        Ok(Self { inner: Arc::new(wrapper) })
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let m = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { m.invoke(msgs, &opts).await })
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    fn __repr__(&self) -> String {
        format!("PromptCachingChat(inner={})", self.inner.name())
    }
}

impl PyPromptCachingChat {
    pub(crate) fn chat_model(&self) -> Arc<dyn ChatModel> {
        self.inner.clone()
    }
}

/// `CostCappedChat` — hard USD cap on cumulative spend across calls.
///
/// Wraps any `ChatModel`. On each invoke, checks the running total against
/// `max_usd`. Once at or above the cap, further calls fail with ValueError
/// BEFORE hitting the provider — the failing call burns no tokens. Prices
/// default to the litGraph built-in table (all major hosted LLMs as of
/// 2026-04). Override with `prices={"model": (prompt_usd_per_mtok, completion_usd_per_mtok)}`.
///
/// Unpriced models add $0 per call (fail-open — custom / internal models
/// don't stall the pipeline; add them to `prices` to opt in).
///
/// ```python
/// from litgraph.providers import OpenAIChat, CostCappedChat
/// raw = OpenAIChat(api_key=..., model="gpt-4o-mini")
/// chat = CostCappedChat(raw, max_usd=1.00)  # $1 daily cap
/// try:
///     chat.invoke(...)
/// except ValueError as e:
///     if "cost cap" in str(e): reset_daily_budget()
/// print(chat.total_usd(), chat.remaining_usd())
/// ```
#[pyclass(name = "CostCappedChat", module = "litgraph.providers")]
pub struct PyCostCappedChat {
    inner: Arc<dyn ChatModel>,
    capped: Arc<CostCappedChatModel>,
}

#[pymethods]
impl PyCostCappedChat {
    #[new]
    #[pyo3(signature = (model, max_usd, prices=None))]
    fn new(
        model: Py<PyAny>,
        max_usd: f64,
        prices: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let chat_model: Arc<dyn ChatModel> =
            Python::with_gil(|py| crate::agents::extract_chat_model(model.bind(py)))?;
        let sheet = match prices {
            Some(d) => {
                let mut s = litgraph_observability::cost::PriceSheet::new();
                for (k, v) in d.iter() {
                    let model_key: String = k.extract()?;
                    let tup: (f64, f64) = v.extract()?;
                    s.set(
                        model_key,
                        litgraph_observability::cost::ModelPrice {
                            prompt_per_mtok: tup.0,
                            completion_per_mtok: tup.1,
                        },
                    );
                }
                s
            }
            None => litgraph_observability::cost::default_prices(),
        };
        let capped = Arc::new(CostCappedChatModel::new(chat_model, sheet, max_usd));
        Ok(Self {
            inner: capped.clone(),
            capped,
        })
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let m = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { m.invoke(msgs, &opts).await })
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    /// Cumulative USD spent through this wrapper.
    fn total_usd(&self) -> f64 {
        self.capped.total_usd()
    }

    /// `max_usd - total_usd`, floored at 0.
    fn remaining_usd(&self) -> f64 {
        self.capped.remaining_usd()
    }

    /// Reset the spend counter — e.g. at midnight UTC for a daily budget.
    fn reset(&self) {
        self.capped.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "CostCappedChat(inner={}, spent=${:.4}, remaining=${:.4})",
            self.inner.name(),
            self.capped.total_usd(),
            self.capped.remaining_usd()
        )
    }
}

impl PyCostCappedChat {
    pub(crate) fn chat_model(&self) -> Arc<dyn ChatModel> {
        self.inner.clone()
    }
}

/// `SelfConsistencyChat` — sample the model N times at elevated temperature,
/// pick the majority answer (Wang et al 2022). Classic reasoning-accuracy
/// boost for math/code/multi-step tasks, at the cost of N× tokens.
///
/// Defaults: `samples=5`, `temperature=0.7`. The default voter picks the
/// most-common normalized response text (trim + lowercase + collapse
/// whitespace). For tasks where raw-text majority never converges (long
/// reasoning chains that reach the same answer through different wording),
/// pass `voter=callable` that extracts the answer field from each response
/// and returns the majority value.
///
/// `voter`: 1-arg callable receiving a list of response dicts (each with
/// `{"text","model","usage",...}`), returning a string-comparable value
/// per response. The wrapper counts these values and picks the majority.
/// Return `None` from the callable for responses where the field is missing
/// — those get excluded from the vote.
///
/// ```python
/// from litgraph.providers import OpenAIChat, SelfConsistencyChat
/// raw = OpenAIChat(api_key=..., model="gpt-4o-mini")
///
/// # Default: text-majority.
/// voter_chat = SelfConsistencyChat(raw, samples=5)
/// resp = voter_chat.invoke([{"role": "user", "content": "2+2?"}])
///
/// # Custom: extract the last number.
/// import re
/// def last_number(r):
///     nums = re.findall(r"-?\d+", r["text"])
///     return nums[-1] if nums else None
/// smart = SelfConsistencyChat(raw, samples=5, voter=last_number)
/// ```
#[pyclass(name = "SelfConsistencyChat", module = "litgraph.providers")]
pub struct PySelfConsistencyChat {
    inner: Arc<dyn ChatModel>,
}

#[pymethods]
impl PySelfConsistencyChat {
    #[new]
    #[pyo3(signature = (model, samples=5, temperature=0.7, voter=None))]
    fn new(
        model: Py<PyAny>,
        samples: usize,
        temperature: f32,
        voter: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let chat_model: Arc<dyn ChatModel> =
            Python::with_gil(|py| crate::agents::extract_chat_model(model.bind(py)))?;
        let mut wrapper = SelfConsistencyChatModel::new(chat_model, samples)
            .with_temperature(temperature);
        if let Some(cb) = voter {
            let cb_arc = Arc::new(cb);
            let voter_fn: litgraph_resilience::ConsistencyVoter =
                Arc::new(move |responses: &[litgraph_core::ChatResponse]| {
                    Python::with_gil(|py| -> Option<usize> {
                        let mut extracted: Vec<Option<String>> = Vec::with_capacity(responses.len());
                        for r in responses {
                            let d = pyo3::types::PyDict::new_bound(py);
                            d.set_item("text", r.message.text_content()).ok()?;
                            d.set_item("model", &r.model).ok()?;
                            let u = pyo3::types::PyDict::new_bound(py);
                            u.set_item("prompt", r.usage.prompt).ok()?;
                            u.set_item("completion", r.usage.completion).ok()?;
                            u.set_item("total", r.usage.total).ok()?;
                            d.set_item("usage", u).ok()?;
                            match cb_arc.call1(py, (d,)) {
                                Ok(v) => {
                                    if v.is_none(py) {
                                        extracted.push(None);
                                    } else {
                                        extracted.push(v.extract::<String>(py).ok());
                                    }
                                }
                                Err(_) => extracted.push(None),
                            }
                        }
                        use std::collections::HashMap;
                        let mut counts: HashMap<&str, usize> = HashMap::new();
                        for e in &extracted {
                            if let Some(v) = e {
                                *counts.entry(v.as_str()).or_insert(0) += 1;
                            }
                        }
                        let (best, _) = counts.iter().max_by_key(|(_, c)| *c)?;
                        extracted
                            .iter()
                            .position(|e| e.as_deref() == Some(*best))
                    })
                });
            wrapper = wrapper.with_voter(voter_fn);
        }
        Ok(Self { inner: Arc::new(wrapper) })
    }

    #[pyo3(signature = (messages, temperature=None, max_tokens=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyList>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let msgs = parse_messages(&messages)?;
        let opts = ChatOptions { temperature, max_tokens, ..Default::default() };
        let m = self.inner.clone();
        let resp = py.allow_threads(|| {
            block_on_compat(async move { m.invoke(msgs, &opts).await })
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        response_to_py_dict(py, &resp)
    }

    fn __repr__(&self) -> String {
        format!("SelfConsistencyChat(inner={})", self.inner.name())
    }
}

impl PySelfConsistencyChat {
    pub(crate) fn chat_model(&self) -> Arc<dyn ChatModel> {
        self.inner.clone()
    }
}
