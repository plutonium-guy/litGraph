//! Python bindings for `MiddlewareChain` + built-in middlewares + the
//! `MiddlewareChat` wrapper. Mirrors LangChain 1.0's middleware abstraction:
//! before/after-model hooks composed in declared order.

use std::sync::Arc;

use litgraph_core::middleware::{
    AgentMiddleware, LoggingMiddleware, MessageWindowMiddleware, MiddlewareChain,
    MiddlewareChatModel, SystemPromptMiddleware,
};
use litgraph_core::ChatModel;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::agents::extract_chat_model;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMiddlewareChain>()?;
    m.add_class::<PyMiddlewareChat>()?;
    m.add_class::<PyLoggingMiddleware>()?;
    m.add_class::<PyMessageWindowMiddleware>()?;
    m.add_class::<PySystemPromptMiddleware>()?;
    Ok(())
}

fn extract_middleware(bound: &Bound<'_, PyAny>) -> PyResult<Arc<dyn AgentMiddleware>> {
    if let Ok(m) = bound.extract::<PyRef<PyLoggingMiddleware>>() {
        return Ok(m.inner.clone());
    }
    if let Ok(m) = bound.extract::<PyRef<PyMessageWindowMiddleware>>() {
        return Ok(m.inner.clone());
    }
    if let Ok(m) = bound.extract::<PyRef<PySystemPromptMiddleware>>() {
        return Ok(m.inner.clone());
    }
    Err(PyValueError::new_err(
        "expected a litgraph middleware instance (LoggingMiddleware, MessageWindowMiddleware, SystemPromptMiddleware)",
    ))
}

#[pyclass(name = "MiddlewareChain", module = "litgraph.middleware")]
pub(crate) struct PyMiddlewareChain {
    pub(crate) inner: MiddlewareChain,
}

#[pymethods]
impl PyMiddlewareChain {
    #[new]
    fn new() -> Self {
        Self {
            inner: MiddlewareChain::new(),
        }
    }

    /// Append a middleware to the chain. Returns `self` so callers can chain
    /// `.with(...).with(...)` fluently.
    fn with_<'py>(
        mut slf: PyRefMut<'py, Self>,
        middleware: Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mw = extract_middleware(&middleware)?;
        slf.inner.push(mw);
        Ok(slf)
    }

    fn append(&mut self, middleware: Bound<'_, PyAny>) -> PyResult<()> {
        let mw = extract_middleware(&middleware)?;
        self.inner.push(mw);
        Ok(())
    }

    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("MiddlewareChain({})", self.inner.names().join(", "))
    }
}

/// Wraps any litgraph chat model with a middleware chain. Implements the
/// chat-model protocol so it plugs into `ReactAgent`, `SupervisorAgent`, etc.
#[pyclass(name = "MiddlewareChat", module = "litgraph.middleware")]
pub struct PyMiddlewareChat {
    pub(crate) inner: Arc<MiddlewareChatModel>,
}

impl PyMiddlewareChat {
    pub(crate) fn chat_model(&self) -> Arc<dyn ChatModel> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyMiddlewareChat {
    #[new]
    fn new(model: Bound<'_, PyAny>, chain: PyRef<'_, PyMiddlewareChain>) -> PyResult<Self> {
        let inner_model = extract_chat_model(&model)?;
        let wrapped = MiddlewareChatModel::new(inner_model, chain.inner.clone());
        Ok(Self {
            inner: Arc::new(wrapped),
        })
    }

    fn names(&self) -> Vec<String> {
        // The wrapper doesn't store the chain post-construction, so this
        // returns the underlying model's name instead. Use MiddlewareChain
        // for chain introspection.
        vec![self.inner.name().to_string()]
    }

    fn __repr__(&self) -> String {
        format!("MiddlewareChat(model={})", self.inner.name())
    }
}

#[pyclass(name = "LoggingMiddleware", module = "litgraph.middleware")]
pub(crate) struct PyLoggingMiddleware {
    inner: Arc<dyn AgentMiddleware>,
}

#[pymethods]
impl PyLoggingMiddleware {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(LoggingMiddleware::new()),
        }
    }

    fn __repr__(&self) -> String {
        "LoggingMiddleware()".into()
    }
}

#[pyclass(name = "MessageWindowMiddleware", module = "litgraph.middleware")]
pub(crate) struct PyMessageWindowMiddleware {
    inner: Arc<dyn AgentMiddleware>,
    keep_last: usize,
}

#[pymethods]
impl PyMessageWindowMiddleware {
    #[new]
    fn new(keep_last: usize) -> Self {
        Self {
            inner: Arc::new(MessageWindowMiddleware::new(keep_last)),
            keep_last,
        }
    }

    fn __repr__(&self) -> String {
        format!("MessageWindowMiddleware(keep_last={})", self.keep_last)
    }
}

#[pyclass(name = "SystemPromptMiddleware", module = "litgraph.middleware")]
pub(crate) struct PySystemPromptMiddleware {
    inner: Arc<dyn AgentMiddleware>,
    prompt: String,
}

#[pymethods]
impl PySystemPromptMiddleware {
    #[new]
    fn new(prompt: String) -> Self {
        Self {
            inner: Arc::new(SystemPromptMiddleware::new(prompt.clone())),
            prompt,
        }
    }

    fn __repr__(&self) -> String {
        let preview: String = self.prompt.chars().take(40).collect();
        format!("SystemPromptMiddleware({:?})", preview)
    }
}
