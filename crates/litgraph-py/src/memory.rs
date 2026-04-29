//! Python bindings for `ConversationMemory`. The Rust types stay sync — this
//! module just owns a `Mutex` so Python's threading model can safely call
//! `append`/`messages` from multiple threads.

use std::sync::{Arc, Mutex};

use litgraph_core::{
    BufferMemory, ConversationMemory, Message, SummaryBufferMemory, TokenBufferMemory, TokenCounter,
    VectorStoreMemory, summarize_conversation as core_summarize_conversation,
};
use litgraph_memory_postgres::PostgresChatHistory;
use litgraph_memory_sqlite::SqliteChatHistory;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;

use crate::agents::extract_chat_model;
use crate::providers::{messages_to_py_list, parse_messages_from_pylist};
use crate::runtime::block_on_compat;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBufferMemory>()?;
    m.add_class::<PyTokenBufferMemory>()?;
    m.add_class::<PySummaryBufferMemory>()?;
    m.add_class::<PyVectorStoreMemory>()?;
    m.add_class::<PySqliteChatHistory>()?;
    m.add_class::<PyPostgresChatHistory>()?;
    m.add_function(wrap_pyfunction!(summarize_conversation, m)?)?;
    Ok(())
}

/// Distill a list of messages into a brief running summary via `model`.
/// `prior_summary` (optional) lets callers iterate — pass the previous summary
/// and the model extends it with new turns.
#[pyfunction]
#[pyo3(signature = (model, messages, prior_summary=None))]
fn summarize_conversation<'py>(
    py: Python<'py>,
    model: Bound<'py, PyAny>,
    messages: Bound<'py, PyList>,
    prior_summary: Option<String>,
) -> PyResult<String> {
    let chat = extract_chat_model(&model)?;
    let msgs = parse_messages_from_pylist(&messages)?;
    py.allow_threads(|| {
        block_on_compat(async move {
            core_summarize_conversation(&*chat, &msgs, prior_summary.as_deref()).await
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

/// Keep the last `max_messages` non-system messages in a sliding window.
/// System messages set via `append({"role":"system",...})` or `set_system(...)`
/// are pinned and always returned first. Drop a system message in to update
/// the pin (matches LangChain's `ConversationBufferMemory`).
#[pyclass(name = "BufferMemory", module = "litgraph.memory")]
pub struct PyBufferMemory {
    inner: Arc<Mutex<BufferMemory>>,
}

#[pymethods]
impl PyBufferMemory {
    #[new]
    #[pyo3(signature = (max_messages=20))]
    fn new(max_messages: usize) -> Self {
        Self { inner: Arc::new(Mutex::new(BufferMemory::new(max_messages))) }
    }

    /// Append one message dict (`{"role": "...", "content": "..."}`).
    fn append<'py>(&self, py: Python<'py>, message: Bound<'py, PyAny>) -> PyResult<()> {
        let m = parse_one_message(message)?;
        py.allow_threads(|| {
            let mut g = self.inner.lock().map_err(|_| PyRuntimeError::new_err("memory poisoned"))?;
            g.append(m);
            Ok(())
        })
    }

    /// Snapshot the conversation. System pin first (if any), then history.
    fn messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let snapshot = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .messages();
        messages_to_py_list(py, &snapshot)
    }

    /// Drop all non-pinned messages.
    fn clear(&self) -> PyResult<()> {
        self.inner.lock().map_err(|_| PyRuntimeError::new_err("memory poisoned"))?.clear();
        Ok(())
    }

    /// Replace the pinned system message; pass `None` to remove it.
    #[pyo3(signature = (message=None))]
    fn set_system<'py>(&self, message: Option<Bound<'py, PyAny>>) -> PyResult<()> {
        let m = message.map(parse_one_message).transpose()?;
        self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .set_system(m);
        Ok(())
    }

    fn __repr__(&self) -> String {
        let g = self.inner.lock();
        let n = g.as_ref().map(|m| m.messages().len()).unwrap_or(0);
        format!("BufferMemory(messages={n})")
    }

    /// Serialize the current state to bytes (JSON). Pair with the matching
    /// `from_bytes` classmethod to restore.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .to_bytes()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new_bound(py, &bytes))
    }

    /// Restore from a previous `to_bytes()` blob. The current `max_messages`
    /// cap on this instance is preserved; a snapshot larger than the cap will
    /// drop oldest messages until it fits.
    fn restore(&self, bytes: &[u8]) -> PyResult<()> {
        let snap = litgraph_core::MemorySnapshot::from_bytes(bytes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .restore(snap);
        Ok(())
    }

    /// Construct a new `BufferMemory(max_messages=...)` populated from a
    /// previous `to_bytes()` blob.
    #[staticmethod]
    fn from_bytes(max_messages: usize, bytes: &[u8]) -> PyResult<Self> {
        let m = BufferMemory::from_bytes(max_messages, bytes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(Mutex::new(m)) })
    }

    /// Summarize the oldest `summarize_count` messages via `model` and replace
    /// them with a single synthetic system message (folding any prior system
    /// pin into the new summary). No-op if history is shorter than the count.
    fn summarize_and_compact<'py>(
        &self,
        py: Python<'py>,
        model: Bound<'py, PyAny>,
        summarize_count: usize,
    ) -> PyResult<()> {
        let chat = extract_chat_model(&model)?;
        // Take the snapshot + summarize OUTSIDE the std Mutex (the inner
        // `summarize_and_compact` borrows the model across an `.await`, so
        // holding a non-Send std Mutex guard across that would not compile —
        // and even if it did, holding a sync lock across an await deadlocks
        // any parallel reader). Instead: snapshot → await → restore.
        let snap = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .snapshot();
        let new_snap = py.allow_threads(|| {
            block_on_compat(async move {
                let mut tmp = BufferMemory::new(usize::MAX);
                tmp.restore(snap);
                tmp.summarize_and_compact(&*chat, summarize_count).await?;
                Ok::<_, litgraph_core::Error>(tmp.snapshot())
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        // Re-apply onto the live instance — this preserves the user's original
        // max_messages cap (see `restore` for eviction semantics).
        self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .restore(new_snap);
        Ok(())
    }
}

/// Keep messages under `max_tokens`; evict oldest when over budget. Token
/// counting is delegated to the user-supplied `counter(message_dict) -> int`
/// callable so this binding stays free of any tokenizer dep. For OpenAI, pass
/// a `tiktoken` callback; for rough estimates, `lambda m: len(m["content"]) // 4`
/// is fine.
#[pyclass(name = "TokenBufferMemory", module = "litgraph.memory")]
pub struct PyTokenBufferMemory {
    inner: Arc<Mutex<TokenBufferMemory>>,
}

#[pymethods]
impl PyTokenBufferMemory {
    /// `counter` receives one Rust-side `Message` rendered as `{"role","content"}`
    /// dict and returns an integer token count.
    #[new]
    fn new(max_tokens: usize, counter: Py<PyAny>) -> Self {
        let counter_py = Arc::new(counter);
        let counter_fn: TokenCounter = Arc::new(move |m: &Message| {
            // Keep the dispatch dead-simple: just `{role, content}` text.
            // Multimodal content would need a richer schema; for token counting
            // the text-only view is the only useful signal anyway.
            let role = match m.role {
                litgraph_core::Role::System => "system",
                litgraph_core::Role::User => "user",
                litgraph_core::Role::Assistant => "assistant",
                litgraph_core::Role::Tool => "tool",
            };
            let content = m.text_content();
            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new_bound(py);
                let _ = dict.set_item("role", role);
                let _ = dict.set_item("content", &content);
                match counter_py.call1(py, (dict,)) {
                    Ok(v) => v.extract::<i64>(py).map(|n| n.max(0) as usize).unwrap_or(0),
                    // Fall back to a length-based approx if the user callable blows up.
                    Err(_) => content.chars().count() / 4,
                }
            })
        });
        Self {
            inner: Arc::new(Mutex::new(TokenBufferMemory::new(max_tokens, counter_fn))),
        }
    }

    fn append<'py>(&self, py: Python<'py>, message: Bound<'py, PyAny>) -> PyResult<()> {
        let m = parse_one_message(message)?;
        py.allow_threads(|| {
            let mut g = self.inner.lock()
                .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?;
            g.append(m);
            Ok(())
        })
    }

    fn messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let snapshot = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .messages();
        messages_to_py_list(py, &snapshot)
    }

    fn clear(&self) -> PyResult<()> {
        self.inner.lock().map_err(|_| PyRuntimeError::new_err("memory poisoned"))?.clear();
        Ok(())
    }

    fn set_system<'py>(&self, message: Option<Bound<'py, PyAny>>) -> PyResult<()> {
        let m = message.map(parse_one_message).transpose()?;
        self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .set_system(m);
        Ok(())
    }

    fn __repr__(&self) -> String {
        let g = self.inner.lock();
        let n = g.as_ref().map(|m| m.messages().len()).unwrap_or(0);
        format!("TokenBufferMemory(messages={n})")
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .to_bytes()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new_bound(py, &bytes))
    }

    /// Restore from a `to_bytes()` blob into the existing instance — the
    /// counter callback bound at construction is reused, so persistence
    /// doesn't need to know about it.
    fn restore(&self, bytes: &[u8]) -> PyResult<()> {
        let snap = litgraph_core::MemorySnapshot::from_bytes(bytes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .restore(snap);
        Ok(())
    }
}

/// Rolling buffer + LLM-distilled running summary of evicted turns.
/// LangChain parity: `ConversationSummaryBufferMemory`.
///
/// ```python
/// from litgraph.memory import SummaryBufferMemory
/// mem = SummaryBufferMemory(max_recent_messages=6, summarize_chunk=4)
/// mem.append({"role": "user", "content": "tell me about Rust"})
/// mem.append({"role": "assistant", "content": "Rust is a systems..."})
/// # ... many turns later ...
/// mem.compact(model)            # fold oldest 4 msgs into running summary
/// msgs = mem.messages()         # [system_pin?, summary_msg?, ...recent]
/// ```
///
/// Summarization is decoupled from `append` so the sync Memory API stays
/// non-blocking. Call `compact(model)` before invoking the model if
/// `needs_compact` is True (or just call it unconditionally — it's a
/// no-op when under cap).
#[pyclass(name = "SummaryBufferMemory", module = "litgraph.memory")]
pub struct PySummaryBufferMemory {
    inner: Arc<Mutex<SummaryBufferMemory>>,
}

#[pymethods]
impl PySummaryBufferMemory {
    #[new]
    #[pyo3(signature = (max_recent_messages=20, summarize_chunk=10))]
    fn new(max_recent_messages: usize, summarize_chunk: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SummaryBufferMemory::new(
                max_recent_messages,
                summarize_chunk,
            ))),
        }
    }

    fn append<'py>(&self, py: Python<'py>, message: Bound<'py, PyAny>) -> PyResult<()> {
        let m = parse_one_message(message)?;
        py.allow_threads(|| {
            let mut g = self.inner.lock().map_err(|_| PyRuntimeError::new_err("memory poisoned"))?;
            g.append(m);
            Ok(())
        })
    }

    fn messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let snapshot = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .messages();
        messages_to_py_list(py, &snapshot)
    }

    fn clear(&self) -> PyResult<()> {
        self.inner.lock().map_err(|_| PyRuntimeError::new_err("memory poisoned"))?.clear();
        Ok(())
    }

    #[pyo3(signature = (message=None))]
    fn set_system<'py>(&self, message: Option<Bound<'py, PyAny>>) -> PyResult<()> {
        let m = message.map(parse_one_message).transpose()?;
        self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .set_system(m);
        Ok(())
    }

    /// Current running summary (may be empty string if none yet).
    fn running_summary(&self) -> PyResult<Option<String>> {
        Ok(self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .running_summary()
            .map(|s| s.to_string()))
    }

    fn recent_len(&self) -> PyResult<usize> {
        Ok(self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .recent_len())
    }

    fn needs_compact(&self) -> PyResult<bool> {
        Ok(self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .needs_compact())
    }

    /// Summarize the oldest `summarize_chunk` messages into the running
    /// summary IFF the buffer exceeds `max_recent_messages`. Returns True
    /// if compaction ran, False if no-op.
    fn compact<'py>(
        &self,
        py: Python<'py>,
        model: Bound<'py, PyAny>,
    ) -> PyResult<bool> {
        let chat = extract_chat_model(&model)?;
        // Same pattern as PyBufferMemory::summarize_and_compact — snapshot,
        // await, restore. Avoids holding std::sync::Mutex across an await.
        let (snap, max_recent, chunk) = {
            let g = self.inner.lock()
                .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?;
            (g.snapshot(), g.max_recent_messages, g.summarize_chunk)
        };
        let (new_snap, ran) = py.allow_threads(|| {
            block_on_compat(async move {
                let mut tmp = SummaryBufferMemory::new(max_recent, chunk);
                tmp.restore(snap);
                let r = tmp.compact(&*chat).await?;
                Ok::<_, litgraph_core::Error>((tmp.snapshot(), r))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .restore(new_snap);
        Ok(ran)
    }

    /// Force-summarize the ENTIRE recent buffer into the running summary.
    /// Useful at session boundaries.
    fn compact_all<'py>(
        &self,
        py: Python<'py>,
        model: Bound<'py, PyAny>,
    ) -> PyResult<bool> {
        let chat = extract_chat_model(&model)?;
        let (snap, max_recent, chunk) = {
            let g = self.inner.lock()
                .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?;
            (g.snapshot(), g.max_recent_messages, g.summarize_chunk)
        };
        let (new_snap, ran) = py.allow_threads(|| {
            block_on_compat(async move {
                let mut tmp = SummaryBufferMemory::new(max_recent, chunk);
                tmp.restore(snap);
                let r = tmp.compact_all(&*chat).await?;
                Ok::<_, litgraph_core::Error>((tmp.snapshot(), r))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .restore(new_snap);
        Ok(ran)
    }

    fn __repr__(&self) -> String {
        let g = self.inner.lock();
        let (n, has_sum) = g.as_ref()
            .map(|m| (m.recent_len(), m.running_summary().is_some()))
            .unwrap_or((0, false));
        format!("SummaryBufferMemory(recent={n}, summary={})", if has_sum { "yes" } else { "no" })
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .to_bytes()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new_bound(py, &bytes))
    }

    fn restore(&self, bytes: &[u8]) -> PyResult<()> {
        let snap = litgraph_core::MemorySnapshot::from_bytes(bytes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("memory poisoned"))?
            .restore(snap);
        Ok(())
    }

    #[staticmethod]
    fn from_bytes(max_recent_messages: usize, summarize_chunk: usize, bytes: &[u8]) -> PyResult<Self> {
        let m = SummaryBufferMemory::from_bytes(max_recent_messages, summarize_chunk, bytes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(Mutex::new(m)) })
    }
}

/// Reuse `parse_messages` (which expects a list) by wrapping a single dict in
/// a one-element list.
fn parse_one_message(message: Bound<'_, PyAny>) -> PyResult<Message> {
    let py = message.py();
    let list = PyList::new_bound(py, &[message]);
    let mut parsed = parse_messages_from_pylist(&list)?;
    parsed.pop().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("empty message"))
}

// ---------- SqliteChatHistory (iter 91) ----------

/// Durable conversation history backed by SQLite. Survives process restarts;
/// per-message storage; multi-session isolation by `session_id` key.
///
/// ```python
/// from litgraph.memory import SqliteChatHistory
/// h = SqliteChatHistory.open("/var/lib/myapp/chats.db", session_id="user-42")
/// h.set_system({"role": "system", "content": "You are helpful."})
/// h.append({"role": "user", "content": "hi"})
/// h.append({"role": "assistant", "content": "hello"})
/// # Process restart...
/// h2 = SqliteChatHistory.open("/var/lib/myapp/chats.db", session_id="user-42")
/// msgs = h2.messages()  # → [system, user, assistant]
/// ```
#[pyclass(name = "SqliteChatHistory", module = "litgraph.memory")]
#[derive(Clone)]
pub struct PySqliteChatHistory {
    inner: SqliteChatHistory,
}

#[pymethods]
impl PySqliteChatHistory {
    /// Open a sqlite file at `path` for the given `session_id`. Creates the
    /// schema idempotently. Multiple opens against the same file are safe
    /// (sqlite WAL mode + per-handle Mutex).
    #[staticmethod]
    fn open(path: String, session_id: String) -> PyResult<Self> {
        let inner = SqliteChatHistory::open(&path, session_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// In-memory variant — no file. For tests or ephemeral sessions.
    #[staticmethod]
    fn in_memory(session_id: String) -> PyResult<Self> {
        let inner = SqliteChatHistory::in_memory(session_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Same backing store, addressed at a different session id. Cheap —
    /// no new sqlite connection.
    fn session(&self, session_id: String) -> Self {
        Self { inner: self.inner.session(session_id) }
    }

    #[getter]
    fn session_id(&self) -> String { self.inner.session_id().into() }

    fn append<'py>(&self, py: Python<'py>, message: Bound<'py, PyAny>) -> PyResult<()> {
        let msg = parse_one_message(message)?;
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.append(msg).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn append_all<'py>(&self, py: Python<'py>, messages: Bound<'py, PyList>) -> PyResult<()> {
        let msgs = parse_messages_from_pylist(&messages)?;
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.append_all(msgs).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let inner = self.inner.clone();
        let msgs = py.allow_threads(|| {
            block_on_compat(async move { inner.messages().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        messages_to_py_list(py, &msgs)
    }

    /// Drops all conversation messages for this session. The system pin
    /// is preserved (use `delete_session()` to wipe everything).
    fn clear<'py>(&self, py: Python<'py>) -> PyResult<()> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.clear().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn delete_session<'py>(&self, py: Python<'py>) -> PyResult<()> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.delete_session().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Set or clear the system pin. Pass `None` to remove.
    #[pyo3(signature = (message=None))]
    fn set_system<'py>(&self, py: Python<'py>, message: Option<Bound<'py, PyAny>>) -> PyResult<()> {
        let msg = match message {
            Some(m) => Some(parse_one_message(m)?),
            None => None,
        };
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.set_system(msg).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn message_count<'py>(&self, py: Python<'py>) -> PyResult<usize> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.message_count().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn list_sessions<'py>(&self, py: Python<'py>) -> PyResult<Vec<String>> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.list_sessions().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> String {
        format!("SqliteChatHistory(session='{}')", self.inner.session_id())
    }
}

/// Vector-store memory — embeddings of each turn, retrieve top-K most-
/// relevant past messages by cosine. Long-running agents need topic-
/// relevance memory, not just recency. LangChain `VectorStoreRetrieverMemory`.
///
/// `append()` is sync and queues the message; `flush()` (or auto-flush
/// on `retrieve_for`) embeds pending messages in a single batch call.
///
/// ```python
/// from litgraph.memory import VectorStoreMemory
/// from litgraph.embeddings import OpenAIEmbeddings
///
/// emb = OpenAIEmbeddings(api_key=..., model="text-embedding-3-small")
/// mem = VectorStoreMemory(embeddings=emb, default_top_k=4)
/// mem.append({"role": "user", "content": "rust borrow checker fix"})
/// mem.append({"role": "assistant", "content": "use clone or rework lifetimes"})
/// # Later turn:
/// retrieved = mem.retrieve_for("rust borrow", k=2)
/// for r in retrieved:
///     print(r["score"], r["message"])
/// # Or build the next-turn context in one call:
/// ctx = mem.build_context("rust borrow", k=2,
///                          current={"role": "user", "content": "another fix?"})
/// ```
#[pyclass(name = "VectorStoreMemory", module = "litgraph.memory")]
pub struct PyVectorStoreMemory {
    inner: Arc<VectorStoreMemory>,
}

#[pymethods]
impl PyVectorStoreMemory {
    #[new]
    #[pyo3(signature = (embeddings, default_top_k=4))]
    fn new(embeddings: Bound<'_, PyAny>, default_top_k: usize) -> PyResult<Self> {
        let emb = crate::embeddings::extract_embeddings(&embeddings)?;
        Ok(Self {
            inner: Arc::new(VectorStoreMemory::new(emb, default_top_k)),
        })
    }

    fn append<'py>(&self, py: Python<'py>, message: Bound<'py, PyAny>) -> PyResult<()> {
        let m = parse_one_message(message)?;
        py.allow_threads(|| {
            self.inner.append(m);
            Ok(())
        })
    }

    #[pyo3(signature = (message=None))]
    fn set_system<'py>(&self, message: Option<Bound<'py, PyAny>>) -> PyResult<()> {
        let m = message.map(parse_one_message).transpose()?;
        self.inner.set_system(m);
        Ok(())
    }

    fn embedded_len(&self) -> usize { self.inner.embedded_len() }
    fn pending_len(&self) -> usize { self.inner.pending_len() }

    fn clear(&self) {
        self.inner.clear();
    }

    /// Embed any pending messages now (in one batch). Returns the new
    /// total number of embedded messages.
    fn flush<'py>(&self, py: Python<'py>) -> PyResult<usize> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.flush().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Top-K most-relevant past messages. `k=0` uses `default_top_k`.
    /// Returns list of dicts: `[{"message": {...}, "score": float}, ...]`.
    #[pyo3(signature = (query, k=0))]
    fn retrieve_for<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let inner = self.inner.clone();
        let q = query.to_string();
        let retrieved = py.allow_threads(|| {
            block_on_compat(async move { inner.retrieve_for(&q, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let out = PyList::empty_bound(py);
        for r in retrieved {
            let d = pyo3::types::PyDict::new_bound(py);
            // messages_to_py_list emits a list — pull the single dict back out.
            let msg_pylist = messages_to_py_list(py, &[r.message])?;
            let msg_dict = msg_pylist.get_item(0)?;
            d.set_item("message", msg_dict)?;
            d.set_item("score", r.score as f64)?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// Build the next-turn message list: `[system?, ...top_k, current]`.
    #[pyo3(signature = (query, current, k=0))]
    fn build_context<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        current: Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let cur = parse_one_message(current)?;
        let inner = self.inner.clone();
        let q = query.to_string();
        let msgs = py.allow_threads(|| {
            block_on_compat(async move { inner.build_context(&q, k, cur).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        messages_to_py_list(py, &msgs)
    }

    fn __repr__(&self) -> String {
        format!(
            "VectorStoreMemory(embedded={}, pending={})",
            self.inner.embedded_len(),
            self.inner.pending_len(),
        )
    }
}

/// Postgres-backed durable conversation history. Distributed counterpart
/// to `SqliteChatHistory` — for cloud / multi-instance deployments where
/// chat sessions need to be readable + writable from multiple workers.
///
/// Uses `deadpool-postgres` for connection pooling. `connect(dsn, session_id)`
/// is the entry point; DSN accepts the standard libpq format
/// `postgres://user:pass@host:5432/db`.
///
/// ```python
/// from litgraph.memory import PostgresChatHistory
/// h = PostgresChatHistory.connect(
///     "postgres://user:pass@localhost:5432/myapp",
///     session_id="user-42",
/// )
/// h.set_system({"role": "system", "content": "You are helpful."})
/// h.append({"role": "user", "content": "hi"})
/// h.append({"role": "assistant", "content": "hello"})
/// # Across worker processes / restarts:
/// h2 = PostgresChatHistory.connect(dsn, session_id="user-42")
/// msgs = h2.messages()  # → [system, user, assistant]
/// ```
#[pyclass(name = "PostgresChatHistory", module = "litgraph.memory")]
#[derive(Clone)]
pub struct PyPostgresChatHistory {
    inner: PostgresChatHistory,
}

#[pymethods]
impl PyPostgresChatHistory {
    /// Connect via libpq DSN. Schema is created idempotently on first
    /// connect.
    #[staticmethod]
    fn connect(py: Python<'_>, dsn: String, session_id: String) -> PyResult<Self> {
        let inner = py.allow_threads(|| {
            block_on_compat(async move {
                PostgresChatHistory::connect(&dsn, session_id).await
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(Self { inner })
    }

    /// Same backing pool, addressed at a different session id. Cheap.
    fn session(&self, session_id: String) -> Self {
        Self { inner: self.inner.session(session_id) }
    }

    #[getter]
    fn session_id(&self) -> String { self.inner.session_id().into() }

    fn append<'py>(&self, py: Python<'py>, message: Bound<'py, PyAny>) -> PyResult<()> {
        let msg = parse_one_message(message)?;
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.append(msg).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn append_all<'py>(&self, py: Python<'py>, messages: Bound<'py, PyList>) -> PyResult<()> {
        let msgs = parse_messages_from_pylist(&messages)?;
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.append_all(msgs).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let inner = self.inner.clone();
        let msgs = py.allow_threads(|| {
            block_on_compat(async move { inner.messages().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        messages_to_py_list(py, &msgs)
    }

    /// Drop all conversation messages for this session. The system pin
    /// is preserved — use `delete_session()` to wipe everything.
    fn clear<'py>(&self, py: Python<'py>) -> PyResult<()> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.clear().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn delete_session<'py>(&self, py: Python<'py>) -> PyResult<()> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.delete_session().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Set or clear the system pin. Pass `None` to remove.
    #[pyo3(signature = (message=None))]
    fn set_system<'py>(&self, py: Python<'py>, message: Option<Bound<'py, PyAny>>) -> PyResult<()> {
        let msg = match message {
            Some(m) => Some(parse_one_message(m)?),
            None => None,
        };
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.set_system(msg).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn message_count<'py>(&self, py: Python<'py>) -> PyResult<usize> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.message_count().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn list_sessions<'py>(&self, py: Python<'py>) -> PyResult<Vec<String>> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { inner.list_sessions().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> String {
        format!("PostgresChatHistory(session='{}')", self.inner.session_id())
    }
}
