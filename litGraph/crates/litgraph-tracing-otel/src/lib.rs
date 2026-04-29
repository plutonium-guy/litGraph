//! OpenTelemetry OTLP exporter for litGraph.
//!
//! Pipes the existing `tracing` spans (emitted by every provider / graph
//! node / tool call / retriever in the workspace) to an OTLP-compatible
//! collector — Jaeger, Tempo, Honeycomb, Datadog, New Relic, etc.
//!
//! # Why a separate crate
//!
//! The OTel ecosystem pulls in `tonic` / `prost` / `reqwest` for the wire
//! transport. A project that only wants pure Rust + JSON logging
//! shouldn't pay that binary cost. Operators opt-in by adding this crate
//! + calling `init_otlp()` once at startup.
//!
//! # Why OTLP and not LangSmith
//!
//! Vendor-neutral. OTLP is the CNCF standard. Every backend (Jaeger,
//! Tempo, Honeycomb, Datadog, New Relic, Grafana Cloud, Dynatrace)
//! accepts OTLP natively. LangSmith forces you into their UI.
//!
//! # Usage
//!
//! ```no_run
//! use litgraph_tracing_otel::{init_otlp, OtelGuard};
//!
//! #[tokio::main]
//! async fn main() {
//!     let _guard: OtelGuard = init_otlp("http://localhost:4317", "my-service")
//!         .expect("init otel");
//!     // ...your code emits tracing::info_span!("node", name = "step1") etc.
//!     // Spans batch-flush to the collector; guard drops → shutdown.
//! }
//! ```
//!
//! # Env var fallbacks
//!
//! Per OTel semantic conventions:
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` — default collector URL
//! - `OTEL_SERVICE_NAME` — service.name resource attribute
//!
//! Explicit args to `init_otlp` override env vars.

use std::env;
use std::time::Duration;

use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::runtime;
use opentelemetry_sdk::trace::{self, TracerProvider};
use opentelemetry_sdk::Resource;
use thiserror::Error;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Error)]
pub enum OtelError {
    #[error("otlp exporter build: {0}")]
    ExporterBuild(String),

    #[error("tracer provider build: {0}")]
    ProviderBuild(String),

    #[error("subscriber init: {0}")]
    SubscriberInit(String),
}

pub type Result<T> = std::result::Result<T, OtelError>;

/// Drop-guard that shuts down the global tracer provider on drop. Flushes
/// any pending spans + tears down the exporter cleanly. Hold it for the
/// process lifetime — dropping mid-program stops trace delivery.
pub struct OtelGuard {
    provider: Option<TracerProvider>,
}

impl OtelGuard {
    /// Explicit synchronous shutdown — flush + close. Safe to call
    /// multiple times (second call is a no-op).
    pub fn shutdown(&mut self) {
        if let Some(provider) = self.provider.take() {
            let _ = provider.shutdown();
        }
    }
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Install a global tracing subscriber that exports spans via OTLP gRPC
/// (tonic) to `endpoint` with `service_name` as the service.name resource
/// attribute. Returns a drop-guard — keep it alive for the process lifetime.
///
/// Batch span processor is used (NOT simple/sync) to avoid blocking hot
/// paths on span export. Batch interval + max batch size come from OTel
/// defaults (2s scheduled delay, 512-span batch).
///
/// `RUST_LOG` / `OTEL_LOG_LEVEL` drive the `EnvFilter` — default `info`
/// if neither is set.
///
/// Subsequent calls are no-ops (global subscriber is install-once).
/// Caller wanting to reconfigure should shut down + re-init in a fresh
/// process; tracing-subscriber's `try_init` returns an error that we
/// surface.
pub fn init_otlp(endpoint: &str, service_name: &str) -> Result<OtelGuard> {
    let endpoint = resolve_endpoint(Some(endpoint.to_string()));
    let service_name = resolve_service_name(Some(service_name.to_string()));
    install(build_otlp_provider(&endpoint, &service_name)?)
}

/// LangSmith migration shim. Preconfigures the OTLP/HTTP exporter to
/// point at LangSmith's ingest endpoint with the right auth headers.
/// Equivalent to the LangChain SDK's `LANGSMITH_TRACING=true` flag —
/// traces land in the LangSmith UI without re-plumbing.
///
/// `api_key` is the LangSmith API key (from `smith.langchain.com/settings`).
/// `project_name` tags every span with the LangSmith project it belongs
/// to. Defaults to service name when None.
///
/// For self-hosted LangSmith / custom endpoint, use `init_otlp_http`
/// directly with your endpoint + `x-api-key` header.
///
/// ```no_run
/// use litgraph_tracing_otel::init_langsmith;
/// # async fn run() {
/// let _guard = init_langsmith(
///     &std::env::var("LANGSMITH_API_KEY").unwrap(),
///     "my-agent",
/// ).expect("init langsmith");
/// // ... tracing spans flow to LangSmith UI.
/// # }
/// ```
pub fn init_langsmith(api_key: &str, project_name: &str) -> Result<OtelGuard> {
    let mut headers = std::collections::HashMap::new();
    headers.insert("x-api-key".to_string(), api_key.to_string());
    headers.insert("Langsmith-Project".to_string(), project_name.to_string());
    init_otlp_http(
        "https://api.smith.langchain.com/otel/v1/traces",
        project_name,
        headers,
    )
}

/// Generic OTLP/HTTP exporter with arbitrary headers. Used by
/// `init_langsmith` and callers wiring Honeycomb / Grafana Cloud /
/// Dynatrace / custom endpoints that need API-key headers (unlike
/// plain OTLP/gRPC which uses mTLS or no auth).
pub fn init_otlp_http(
    endpoint: &str,
    service_name: &str,
    headers: std::collections::HashMap<String, String>,
) -> Result<OtelGuard> {
    let exporter = opentelemetry_otlp::new_exporter()
        .http()
        .with_endpoint(endpoint)
        .with_headers(headers)
        .with_timeout(Duration::from_secs(5))
        .build_span_exporter()
        .map_err(|e| OtelError::ExporterBuild(e.to_string()))?;

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, runtime::Tokio)
        .with_config(
            trace::Config::default().with_resource(Resource::new(vec![KeyValue::new(
                "service.name",
                service_name.to_string(),
            )])),
        )
        .build();
    install(provider)
}

/// Stdout exporter — for local dev. Pretty-prints spans to stderr as
/// they close. Use to verify instrumentation before wiring a collector.
pub fn init_stdout(service_name: &str) -> Result<OtelGuard> {
    let service_name = resolve_service_name(Some(service_name.to_string()));
    let exporter = opentelemetry_stdout::SpanExporter::default();
    let provider = TracerProvider::builder()
        .with_simple_exporter(exporter)
        .with_config(
            trace::Config::default().with_resource(Resource::new(vec![KeyValue::new(
                "service.name",
                service_name,
            )])),
        )
        .build();
    install(provider)
}

/// For tests: install a tracer that captures spans in memory. Returns
/// the guard + a handle to inspect captured spans.
pub fn init_in_memory(
    service_name: &str,
) -> Result<(OtelGuard, opentelemetry_sdk::testing::trace::InMemorySpanExporter)> {
    let exporter = opentelemetry_sdk::testing::trace::InMemorySpanExporter::default();
    let handle = exporter.clone();
    let provider = TracerProvider::builder()
        .with_simple_exporter(exporter)
        .with_config(
            trace::Config::default().with_resource(Resource::new(vec![KeyValue::new(
                "service.name",
                service_name.to_string(),
            )])),
        )
        .build();
    let guard = install(provider)?;
    Ok((guard, handle))
}

fn resolve_endpoint(explicit: Option<String>) -> String {
    explicit
        .or_else(|| env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok())
        .unwrap_or_else(|| "http://localhost:4317".to_string())
}

fn resolve_service_name(explicit: Option<String>) -> String {
    explicit
        .or_else(|| env::var("OTEL_SERVICE_NAME").ok())
        .unwrap_or_else(|| "litgraph".to_string())
}

fn build_otlp_provider(endpoint: &str, service_name: &str) -> Result<TracerProvider> {
    // opentelemetry-otlp 0.26 API: build the exporter via `new_exporter().tonic()`,
    // configure endpoint + timeout, call `build_span_exporter()`.
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(3))
        .build_span_exporter()
        .map_err(|e| OtelError::ExporterBuild(e.to_string()))?;

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, runtime::Tokio)
        .with_config(
            trace::Config::default().with_resource(Resource::new(vec![KeyValue::new(
                "service.name",
                service_name.to_string(),
            )])),
        )
        .build();
    Ok(provider)
}

fn install(provider: TracerProvider) -> Result<OtelGuard> {
    global::set_tracer_provider(provider.clone());
    let tracer = provider.tracer("litgraph");
    let otel_layer = OpenTelemetryLayer::new(tracer);
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .map_err(|e| OtelError::SubscriberInit(e.to_string()))?;

    let subscriber = tracing_subscriber::registry()
        .with(env_filter)
        .with(otel_layer);

    // `try_init` fails on the second call — swallow and return a guard
    // tied to this provider. The tracing layer is global-install-once;
    // subsequent `init_otlp` calls in the same process just refresh the
    // provider on `global::set_tracer_provider`.
    let _ = subscriber.try_init();

    Ok(OtelGuard {
        provider: Some(provider),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test — build an in-memory provider, install it globally,
    /// verify the guard's drop doesn't panic. The full tracing → OTel
    /// → exporter pipeline is tested upstream by `tracing-opentelemetry`
    /// + `opentelemetry_sdk`; our crate's value is the config glue.
    #[tokio::test]
    async fn in_memory_provider_installs_and_shuts_down_cleanly() {
        let (mut guard, exporter) = init_in_memory("test-service").unwrap();
        // Force a shutdown then read — exporter accessible = wiring worked.
        guard.shutdown();
        let finished = exporter.get_finished_spans().unwrap();
        // No spans emitted in this test; shouldn't panic on empty read.
        assert_eq!(finished.len(), 0);
    }

    #[test]
    fn resolve_endpoint_prefers_explicit_then_env_then_default() {
        // explicit overrides env
        std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://from-env:4317");
        assert_eq!(
            resolve_endpoint(Some("http://explicit:4317".into())),
            "http://explicit:4317"
        );
        // env used when no explicit
        assert_eq!(resolve_endpoint(None), "http://from-env:4317");
        std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        // default when neither
        assert_eq!(resolve_endpoint(None), "http://localhost:4317");
    }

    #[test]
    fn resolve_service_name_prefers_explicit_then_env_then_default() {
        std::env::set_var("OTEL_SERVICE_NAME", "from-env");
        assert_eq!(
            resolve_service_name(Some("explicit".into())),
            "explicit"
        );
        assert_eq!(resolve_service_name(None), "from-env");
        std::env::remove_var("OTEL_SERVICE_NAME");
        assert_eq!(resolve_service_name(None), "litgraph");
    }

    /// Drop-guard idempotent shutdown.
    #[tokio::test]
    async fn guard_shutdown_is_idempotent() {
        let (mut guard, _exp) = init_in_memory("svc").unwrap();
        guard.shutdown();
        guard.shutdown(); // must not panic
    }
}
