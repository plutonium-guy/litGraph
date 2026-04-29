//! OpenTelemetry wiring (feature = `otel`). One function — `init_tracer` —
//! connects the `tracing` crate to an OTLP exporter. Consumers add `.init()` in
//! main and every `tracing::info_span!` in the workspace becomes an OTel span.
//!
//! Not LangSmith. Not any vendor. OTLP means Grafana Tempo, Honeycomb, Datadog,
//! Signoz, AWS X-Ray adapter, or LangSmith's OTel endpoint — user's choice.

#[cfg(feature = "otel")]
use opentelemetry::global;
#[cfg(feature = "otel")]
use opentelemetry::KeyValue;
#[cfg(feature = "otel")]
use opentelemetry_sdk::trace::{self, Sampler, TracerProvider};
#[cfg(feature = "otel")]
use opentelemetry_sdk::Resource;
#[cfg(feature = "otel")]
use tracing_subscriber::layer::SubscriberExt;
#[cfg(feature = "otel")]
use tracing_subscriber::util::SubscriberInitExt;

#[cfg(feature = "otel")]
#[derive(Debug, Clone)]
pub struct OtelConfig {
    pub service_name: String,
    pub sampler_ratio: f64,
}

#[cfg(feature = "otel")]
impl Default for OtelConfig {
    fn default() -> Self {
        Self { service_name: "litgraph".into(), sampler_ratio: 1.0 }
    }
}

/// Install a basic tracer that emits spans via the `tracing` crate. Intended as
/// a starting point — callers who need an OTLP HTTP/gRPC exporter bring their
/// own exporter crate and wire the resulting `TracerProvider` here.
#[cfg(feature = "otel")]
pub fn init_stdout_tracer(cfg: OtelConfig) -> Result<(), Box<dyn std::error::Error>> {
    let resource = Resource::new(vec![KeyValue::new("service.name", cfg.service_name.clone())]);
    let provider = TracerProvider::builder()
        .with_config(trace::Config::default()
            .with_sampler(Sampler::TraceIdRatioBased(cfg.sampler_ratio))
            .with_resource(resource))
        .build();
    let tracer = opentelemetry::trace::TracerProvider::tracer(&provider, cfg.service_name.clone());
    global::set_tracer_provider(provider);

    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(otel_layer)
        .try_init()?;
    Ok(())
}

#[cfg(not(feature = "otel"))]
pub fn init_stdout_tracer() {
    // no-op stub so downstream code compiles without the feature
}

/// LangSmith config — pass to `langsmith_otlp_endpoint` to format the OTLP
/// endpoint URL, and `langsmith_otlp_headers` for the auth headers.
///
/// LangSmith accepts OTLP-shaped traces at:
///     https://api.smith.langchain.com/otel/v1/traces
///
/// with auth headers `x-api-key: <LANGSMITH_API_KEY>` and an optional
/// `Langsmith-Project` header to route traces to a specific project.
///
/// Integration is OTel-native: configure your existing OTLP HTTP exporter
/// (e.g. `opentelemetry-otlp` with `http-proto`) using these helpers, and
/// `tracing` spans flow into LangSmith without any LangSmith SDK in your
/// dep tree. No vendor lock-in.
#[derive(Debug, Clone)]
pub struct LangSmithConfig {
    pub api_key: String,
    pub project: Option<String>,
    /// Override for self-hosted / regional endpoints.
    pub base_url: Option<String>,
}

impl LangSmithConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self { api_key: api_key.into(), project: None, base_url: None }
    }

    pub fn with_project(mut self, p: impl Into<String>) -> Self {
        self.project = Some(p.into());
        self
    }
}

/// Build the OTLP/HTTP traces endpoint URL for LangSmith. Pass to your
/// `opentelemetry-otlp` `HttpExporterBuilder::with_endpoint(...)`.
pub fn langsmith_otlp_endpoint(cfg: &LangSmithConfig) -> String {
    let base = cfg
        .base_url
        .clone()
        .unwrap_or_else(|| "https://api.smith.langchain.com/otel".into());
    format!("{}/v1/traces", base.trim_end_matches('/'))
}

/// Build the auth headers required by LangSmith's OTLP endpoint. Pass to your
/// `HttpExporterBuilder::with_headers(...)` as a `HashMap<String, String>`.
pub fn langsmith_otlp_headers(cfg: &LangSmithConfig) -> std::collections::HashMap<String, String> {
    let mut h = std::collections::HashMap::new();
    h.insert("x-api-key".into(), cfg.api_key.clone());
    if let Some(p) = &cfg.project {
        h.insert("Langsmith-Project".into(), p.clone());
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn langsmith_endpoint_default() {
        let cfg = LangSmithConfig::new("ls-test");
        assert_eq!(
            langsmith_otlp_endpoint(&cfg),
            "https://api.smith.langchain.com/otel/v1/traces"
        );
    }

    #[test]
    fn langsmith_endpoint_self_hosted() {
        let mut cfg = LangSmithConfig::new("k");
        cfg.base_url = Some("https://ls.internal.example.com/otel".into());
        assert_eq!(
            langsmith_otlp_endpoint(&cfg),
            "https://ls.internal.example.com/otel/v1/traces"
        );
    }

    #[test]
    fn langsmith_headers_include_api_key_and_project() {
        let cfg = LangSmithConfig::new("sk-ls").with_project("my-app");
        let h = langsmith_otlp_headers(&cfg);
        assert_eq!(h.get("x-api-key"), Some(&"sk-ls".to_string()));
        assert_eq!(h.get("Langsmith-Project"), Some(&"my-app".to_string()));
    }
}
