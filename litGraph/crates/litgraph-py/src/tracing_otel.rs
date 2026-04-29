//! Python bindings for the `litgraph-tracing-otel` crate. Exposes
//! `litgraph.tracing.{init_otlp, init_stdout, shutdown}` so Python apps
//! can wire their existing `tracing` spans to an OTLP collector with a
//! single call.

use std::sync::Mutex;

use litgraph_tracing_otel::{
    init_langsmith, init_otlp, init_otlp_http, init_stdout, OtelGuard,
};
use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::runtime::rt;

/// Global guard slot. Init stashes the guard here so it survives the
/// Python caller dropping the return value (common in one-shot startup
/// code). Shutdown takes the guard out and drops it.
static GUARD: Lazy<Mutex<Option<OtelGuard>>> = Lazy::new(|| Mutex::new(None));

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_otlp_py, m)?)?;
    m.add_function(wrap_pyfunction!(init_stdout_py, m)?)?;
    m.add_function(wrap_pyfunction!(init_langsmith_py, m)?)?;
    m.add_function(wrap_pyfunction!(init_otlp_http_py, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown_py, m)?)?;
    Ok(())
}

/// Install a global tracing subscriber that exports spans via OTLP gRPC
/// to `endpoint`. Call once at app startup.
///
/// `endpoint` defaults to `http://localhost:4317` (standard OTel
/// collector gRPC port). `service_name` defaults to `"litgraph"`.
/// Explicit args override the `OTEL_EXPORTER_OTLP_ENDPOINT` and
/// `OTEL_SERVICE_NAME` env vars respectively.
///
/// ```python
/// from litgraph.tracing import init_otlp, shutdown
/// init_otlp(endpoint="http://localhost:4317", service_name="my-agent")
/// # ... your code emits tracing spans ...
/// shutdown()  # flush before process exit (optional — atexit also works)
/// ```
#[pyfunction(name = "init_otlp")]
#[pyo3(signature = (endpoint="http://localhost:4317", service_name="litgraph"))]
fn init_otlp_py(endpoint: &str, service_name: &str) -> PyResult<()> {
    // Batch span processor spawns a background task, which requires a
    // running Tokio runtime in context. Enter the shared runtime.
    let _guard = rt().enter();
    let guard = init_otlp(endpoint, service_name)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    *GUARD.lock().expect("poisoned") = Some(guard);
    Ok(())
}

/// Install a stdout exporter. Pretty-prints spans to stderr as they
/// close. For local dev; verifies instrumentation before wiring a
/// collector.
#[pyfunction(name = "init_stdout")]
#[pyo3(signature = (service_name="litgraph"))]
fn init_stdout_py(service_name: &str) -> PyResult<()> {
    let guard = init_stdout(service_name)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    *GUARD.lock().expect("poisoned") = Some(guard);
    Ok(())
}

/// Flush pending spans and shut down the tracer provider. Idempotent —
/// safe to call multiple times or without a prior init. Hook into
/// `atexit.register(shutdown)` in production apps.
#[pyfunction(name = "shutdown")]
fn shutdown_py() {
    let mut slot = GUARD.lock().expect("poisoned");
    if let Some(mut g) = slot.take() {
        g.shutdown();
    }
}

/// LangSmith migration shim. Traces flow to the LangSmith UI with your
/// existing `tracing::info_span!()` instrumentation — no re-plumbing.
///
/// `api_key` is your LangSmith API key (smith.langchain.com/settings).
/// `project_name` tags every span with the project it belongs to.
///
/// ```python
/// import os
/// from litgraph.tracing import init_langsmith, shutdown
/// init_langsmith(
///     api_key=os.environ["LANGSMITH_API_KEY"],
///     project_name="my-agent",
/// )
/// # ... litGraph code emits tracing spans ...
/// shutdown()
/// ```
///
/// For self-hosted LangSmith, use `init_otlp_http` directly with your
/// endpoint + `{"x-api-key": "..."}` header dict.
#[pyfunction(name = "init_langsmith")]
#[pyo3(signature = (api_key, project_name="litgraph"))]
fn init_langsmith_py(api_key: &str, project_name: &str) -> PyResult<()> {
    let _guard = rt().enter();
    let guard = init_langsmith(api_key, project_name)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    *GUARD.lock().expect("poisoned") = Some(guard);
    Ok(())
}

/// Generic OTLP/HTTP exporter with arbitrary headers. Use for Honeycomb,
/// Grafana Cloud, Dynatrace, or any endpoint that needs API-key headers.
///
/// `headers` is a dict[str, str] — passed to every export request.
#[pyfunction(name = "init_otlp_http")]
fn init_otlp_http_py(
    endpoint: &str,
    service_name: &str,
    headers: Bound<'_, PyDict>,
) -> PyResult<()> {
    let mut map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for (k, v) in headers.iter() {
        map.insert(k.extract::<String>()?, v.extract::<String>()?);
    }
    let _guard = rt().enter();
    let guard = init_otlp_http(endpoint, service_name, map)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    *GUARD.lock().expect("poisoned") = Some(guard);
    Ok(())
}
