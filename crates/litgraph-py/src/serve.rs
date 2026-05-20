//! Python binding for `litgraph-serve` (iter 384). Closes the
//! "recipes.serve actually spawns the binary" Tier-2 gap.
//!
//! Pre-iter-384, `litgraph.recipes.serve(...)` returned a shell-command
//! string that didn't correspond to anything runnable — the crate
//! lives as a library only, no `main.rs` ships. To actually serve a
//! chat model the user had to wire `axum::serve` themselves.
//!
//! This module exposes:
//!
//! * `litgraph.serve.spawn_chat(model, host, port)` — binds the
//!   listener synchronously (so the caller sees `OSError` fast if the
//!   port is taken), then spawns the axum server on the shared tokio
//!   runtime and returns a [`ServeHandle`].
//! * `ServeHandle.address()` / `.url()` — the bound address (useful
//!   when `port=0` and the OS picks a free port).
//! * `ServeHandle.shutdown()` — cancels the server gracefully via a
//!   `CancellationToken`; idempotent.
//!
//! `recipes.serve(...)` calls into this binding for `ChatModel` inputs.
//! Graph-shaped inputs still raise — graph serialization across the
//! HTTP boundary is a separate scope item (`/threads` API on the
//! `studio` feature).

use std::net::SocketAddr;
use std::sync::Arc;

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::agents::extract_chat_model;
use crate::runtime::rt;

/// Handle to a running litgraph-serve instance. Drops do NOT cancel
/// the server — call `.shutdown()` explicitly. Why: Python's GC
/// timing is unpredictable, and a server that vanished mid-request
/// would surface as a connection-reset on the client without an
/// obvious cause. Explicit shutdown stays loud.
#[pyclass(name = "ServeHandle", module = "litgraph.serve")]
pub struct PyServeHandle {
    address: SocketAddr,
    model_name: String,
    cancel: CancellationToken,
    /// Server future's join handle. `None` after shutdown so repeat
    /// calls to `.shutdown()` don't double-await.
    join: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

#[pymethods]
impl PyServeHandle {
    fn address(&self) -> String {
        self.address.to_string()
    }

    fn url(&self) -> String {
        // `0.0.0.0` is the wildcard — return `http://localhost:<port>`
        // so a clipboard-paste lands on a working URL. Other bind
        // addresses pass through unchanged.
        if self.address.ip().is_unspecified() {
            format!("http://localhost:{}", self.address.port())
        } else {
            format!("http://{}", self.address)
        }
    }

    fn model(&self) -> String {
        self.model_name.clone()
    }

    /// Cancel the server. Idempotent: a second call is a no-op.
    /// Blocks until the axum task observes the cancel token and
    /// returns — typically a few milliseconds.
    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<()> {
        self.cancel.cancel();
        // Drain the join handle. Use `try_lock` so a double-shutdown
        // from two threads doesn't deadlock; the second caller sees
        // `None` and returns.
        py.allow_threads(|| {
            rt().block_on(async {
                let mut guard = self.join.lock().await;
                if let Some(h) = guard.take() {
                    let _ = h.await;
                }
            });
        });
        Ok(())
    }
}

/// Spawn `model` as an HTTP server on `host:port`. Returns a
/// [`PyServeHandle`] once the listener is bound — failures here
/// (bad host, port in use) surface as `OSError` before the function
/// returns, matching Python's `socket.bind` convention.
///
/// `port=0` asks the OS for a free port; read it back from
/// `handle.address()`.
#[pyfunction]
#[pyo3(signature = (model, host="127.0.0.1", port=8080))]
pub fn spawn_chat<'py>(
    py: Python<'py>,
    model: Bound<'py, PyAny>,
    host: &str,
    port: u16,
) -> PyResult<PyServeHandle> {
    let chat: Arc<dyn litgraph_core::ChatModel> = extract_chat_model(&model)?;
    let model_name = chat.name().to_string();
    let addr_str = format!("{host}:{port}");

    // Bind synchronously so a port-in-use error surfaces before we
    // return — Python callers expect bind failures to look like
    // `OSError`, not "the future I never awaited had an error".
    let (listener, bound_addr) = py.allow_threads(|| {
        rt().block_on(async {
            let parsed: SocketAddr = addr_str
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("bad addr {addr_str}: {e}")))?;
            let listener = tokio::net::TcpListener::bind(parsed).await?;
            let bound = listener.local_addr()?;
            Ok::<_, std::io::Error>((listener, bound))
        })
    })
    .map_err(|e: std::io::Error| PyOSError::new_err(e.to_string()))?;

    debug!("litgraph-serve bound on {bound_addr} (model={model_name})");

    let cancel = CancellationToken::new();
    let cancel_for_task = cancel.clone();
    let router = litgraph_serve::router_for(chat);

    let join = rt().spawn(async move {
        // Drive axum::serve with graceful-shutdown wired to the
        // cancellation token. When `cancel` fires, axum stops
        // accepting new connections and waits for in-flight requests
        // before returning.
        let serve = axum::serve(listener, router)
            .with_graceful_shutdown(async move { cancel_for_task.cancelled().await });
        if let Err(e) = serve.await {
            tracing::warn!("litgraph-serve exited with error: {e}");
        }
    });

    Ok(PyServeHandle {
        address: bound_addr,
        model_name,
        cancel,
        join: Mutex::new(Some(join)),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyServeHandle>()?;
    m.add_function(wrap_pyfunction!(spawn_chat, m)?)?;
    Ok(())
}

// pyo3 macro re-import shim — `wrap_pyfunction!` lives in `pyo3::prelude`
// which we already imported.
use pyo3::wrap_pyfunction;
