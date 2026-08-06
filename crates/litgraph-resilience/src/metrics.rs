use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, Counter, Embeddings, Gauge, Histogram, Message,
    MetricsRegistry, Result,
};

/// Default histogram buckets in seconds. Geometric-ish spread
/// covering the typical LLM/embed latency range from 5ms to 30s.
pub const DEFAULT_LATENCY_BUCKETS_SECS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

/// Pre-resolved metric handles so the hot path is pure atomic
/// ops (no HashMap lookup).
struct MetricsHandles {
    invocations: Arc<Counter>,
    errors: Arc<Counter>,
    in_flight: Arc<Gauge>,
    latency: Arc<Histogram>,
}

/// RAII guard that decrements the in-flight gauge on drop. This
/// makes the gauge correct even if the inner future panics or is
/// cancelled mid-await.
struct InFlightGuard {
    gauge: Arc<Gauge>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.gauge.dec();
    }
}

fn resolve_handles(
    registry: &MetricsRegistry,
    prefix: &str,
    buckets: &[f64],
) -> MetricsHandles {
    MetricsHandles {
        invocations: registry.counter(&format!("{prefix}_invocations_total")),
        errors: registry.counter(&format!("{prefix}_errors_total")),
        in_flight: registry.gauge(&format!("{prefix}_in_flight")),
        latency: registry.histogram(
            &format!("{prefix}_latency_seconds"),
            buckets,
        ),
    }
}

/// Wrap any [`ChatModel`] with auto-instrumentation against an
/// iter-260 [`MetricsRegistry`]. Bumps the standard four
/// metrics on every `invoke`:
///
/// - `<prefix>_invocations_total` — counter, every call.
/// - `<prefix>_errors_total` — counter, calls that returned `Err`.
/// - `<prefix>_in_flight` — gauge, current concurrent calls
///   (incremented on entry, decremented on exit via RAII guard
///   so it stays correct under cancellation/panic).
/// - `<prefix>_latency_seconds` — histogram of `Instant::elapsed`
///   across the inner call.
///
/// Default prefix: `"chat"`. Override with `with_prefix`.
/// Default histogram buckets: [`DEFAULT_LATENCY_BUCKETS_SECS`].
/// Override with `with_buckets`.
///
/// Metric handles are resolved once at construction via the
/// registry's get-or-create lookup, so the hot path is pure
/// atomic ops — no HashMap lookup per call.
///
/// # Streaming
///
/// `stream()` records the same metrics at the **handshake**:
/// invocations / errors / latency reflect the time until
/// `stream()` returns its outer Result, not the time the inner
/// stream takes to drain. In-flight is incremented before
/// handshake and decremented before stream() returns —
/// per-token timing is the consumer's responsibility (typical
/// patten: caller wraps `s.next()` with its own timing).
pub struct MetricsChatModel {
    pub inner: Arc<dyn ChatModel>,
    handles: MetricsHandles,
}

impl MetricsChatModel {
    pub fn new(inner: Arc<dyn ChatModel>, registry: &MetricsRegistry) -> Self {
        Self {
            inner,
            handles: resolve_handles(registry, "chat", DEFAULT_LATENCY_BUCKETS_SECS),
        }
    }

    pub fn with_prefix(
        inner: Arc<dyn ChatModel>,
        registry: &MetricsRegistry,
        prefix: &str,
    ) -> Self {
        Self {
            inner,
            handles: resolve_handles(registry, prefix, DEFAULT_LATENCY_BUCKETS_SECS),
        }
    }

    pub fn with_buckets(
        inner: Arc<dyn ChatModel>,
        registry: &MetricsRegistry,
        prefix: &str,
        buckets: &[f64],
    ) -> Self {
        Self {
            inner,
            handles: resolve_handles(registry, prefix, buckets),
        }
    }
}

#[async_trait]
impl ChatModel for MetricsChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.handles.invocations.inc();
        self.handles.in_flight.inc();
        let _guard = InFlightGuard {
            gauge: self.handles.in_flight.clone(),
        };
        let started = std::time::Instant::now();
        let r = self.inner.invoke(messages, opts).await;
        self.handles
            .latency
            .observe(started.elapsed().as_secs_f64());
        if r.is_err() {
            self.handles.errors.inc();
        }
        r
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        self.handles.invocations.inc();
        self.handles.in_flight.inc();
        let _guard = InFlightGuard {
            gauge: self.handles.in_flight.clone(),
        };
        let started = std::time::Instant::now();
        let r = self.inner.stream(messages, opts).await;
        self.handles
            .latency
            .observe(started.elapsed().as_secs_f64());
        if r.is_err() {
            self.handles.errors.inc();
        }
        r
    }
}

/// Embed-axis sibling. Same four-metric instrumentation; default
/// prefix is `"embed"`. Both `embed_query` and `embed_documents`
/// share the metrics — distinguish in caller code if you need
/// per-method breakdown.
pub struct MetricsEmbeddings {
    pub inner: Arc<dyn Embeddings>,
    handles: MetricsHandles,
}

impl MetricsEmbeddings {
    pub fn new(inner: Arc<dyn Embeddings>, registry: &MetricsRegistry) -> Self {
        Self {
            inner,
            handles: resolve_handles(registry, "embed", DEFAULT_LATENCY_BUCKETS_SECS),
        }
    }

    pub fn with_prefix(
        inner: Arc<dyn Embeddings>,
        registry: &MetricsRegistry,
        prefix: &str,
    ) -> Self {
        Self {
            inner,
            handles: resolve_handles(registry, prefix, DEFAULT_LATENCY_BUCKETS_SECS),
        }
    }

    pub fn with_buckets(
        inner: Arc<dyn Embeddings>,
        registry: &MetricsRegistry,
        prefix: &str,
        buckets: &[f64],
    ) -> Self {
        Self {
            inner,
            handles: resolve_handles(registry, prefix, buckets),
        }
    }
}

#[async_trait]
impl Embeddings for MetricsEmbeddings {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.handles.invocations.inc();
        self.handles.in_flight.inc();
        let _guard = InFlightGuard {
            gauge: self.handles.in_flight.clone(),
        };
        let started = std::time::Instant::now();
        let r = self.inner.embed_query(text).await;
        self.handles
            .latency
            .observe(started.elapsed().as_secs_f64());
        if r.is_err() {
            self.handles.errors.inc();
        }
        r
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.handles.invocations.inc();
        self.handles.in_flight.inc();
        let _guard = InFlightGuard {
            gauge: self.handles.in_flight.clone(),
        };
        let started = std::time::Instant::now();
        let r = self.inner.embed_documents(texts).await;
        self.handles
            .latency
            .observe(started.elapsed().as_secs_f64());
        if r.is_err() {
            self.handles.errors.inc();
        }
        r
    }
}

/// Wrap any [`litgraph_core::tool::Tool`] with auto-instrumentation
/// against an iter-260 [`MetricsRegistry`]. Same four metrics as
/// [`MetricsChatModel`] / [`MetricsEmbeddings`]: `<prefix>_invocations_total`,
/// `<prefix>_errors_total`, `<prefix>_in_flight`, `<prefix>_latency_seconds`.
///
/// Default prefix is the tool's own name (from its schema), with
/// disallowed characters sanitized to `_`. So a tool named `"http.get"`
/// produces metrics under `http_get_*`. Override with `with_prefix`
/// for explicit per-call-site labeling.
///
/// Metric handles are pre-resolved at construction so the hot path is
/// pure atomic ops — same design as iter 261.
///
/// Per-tool metrics are especially valuable for agent debugging: agents
/// make many tool calls per session, and knowing which tools fail /
/// are slow / are hot is the first thing you want from a `/metrics`
/// dashboard.
pub struct MetricsTool {
    pub inner: Arc<dyn litgraph_core::tool::Tool>,
    handles: MetricsHandles,
}

impl MetricsTool {
    pub fn new(
        inner: Arc<dyn litgraph_core::tool::Tool>,
        registry: &MetricsRegistry,
    ) -> Self {
        let prefix = sanitize_metric_prefix(&inner.name());
        Self {
            inner,
            handles: resolve_handles(registry, &prefix, DEFAULT_LATENCY_BUCKETS_SECS),
        }
    }

    pub fn with_prefix(
        inner: Arc<dyn litgraph_core::tool::Tool>,
        registry: &MetricsRegistry,
        prefix: &str,
    ) -> Self {
        Self {
            inner,
            handles: resolve_handles(registry, prefix, DEFAULT_LATENCY_BUCKETS_SECS),
        }
    }

    pub fn with_buckets(
        inner: Arc<dyn litgraph_core::tool::Tool>,
        registry: &MetricsRegistry,
        prefix: &str,
        buckets: &[f64],
    ) -> Self {
        Self {
            inner,
            handles: resolve_handles(registry, prefix, buckets),
        }
    }
}

#[async_trait]
impl litgraph_core::tool::Tool for MetricsTool {
    fn schema(&self) -> litgraph_core::tool::ToolSchema {
        self.inner.schema()
    }

    async fn run(
        &self,
        args: serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.handles.invocations.inc();
        self.handles.in_flight.inc();
        let _guard = InFlightGuard {
            gauge: self.handles.in_flight.clone(),
        };
        let started = std::time::Instant::now();
        let r = self.inner.run(args).await;
        self.handles
            .latency
            .observe(started.elapsed().as_secs_f64());
        if r.is_err() {
            self.handles.errors.inc();
        }
        r
    }
}

/// Sanitize a tool name into a Prometheus-compatible metric prefix.
/// Mirrors the rules used by `MetricsRegistry::to_prometheus` so the
/// same allowed-character set applies. Returns "tool" if the input
/// produces an empty result (defensive default).
fn sanitize_metric_prefix(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, c) in name.chars().enumerate() {
        let ok = if i == 0 {
            c.is_ascii_alphabetic() || c == '_'
        } else {
            c.is_ascii_alphanumeric() || c == '_' || c == ':'
        };
        out.push(if ok { c } else { '_' });
    }
    if out.is_empty() {
        "tool".into()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    #[allow(unused_imports)]
    use litgraph_core::tool::Tool as _;
    #[allow(unused_imports)]
    use litgraph_core::{ContentPart, Message, Role};
    #[allow(unused_imports)]
    use std::sync::atomic::{AtomicU32, Ordering};
    #[allow(unused_imports)]
    use litgraph_core::Error;
    #[allow(unused_imports)]
    use std::time::Duration;

    /// Always-succeeds model for closed-path test.
    struct AlwaysOkModel;

    #[async_trait]
    impl ChatModel for AlwaysOkModel {
        fn name(&self) -> &str {
            "always-ok"
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text: "hi".into() }],
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "always-ok".into(),
            })
        }
        async fn stream(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    /// Sleeps `delay_ms` per call and tracks peak concurrent calls.
    /// Lets keyed-mutex tests verify same-key serialization vs
    /// different-key parallelism.
    struct DelayChatModel {
        delay_ms: u64,
        in_flight: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl ChatModel for DelayChatModel {
        fn name(&self) -> &str {
            "delay-chat"
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut p = self.peak.load(Ordering::SeqCst);
            while now > p {
                match self.peak.compare_exchange(
                    p,
                    now,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(actual) => p = actual,
                }
            }
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text: "ok".into() }],
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "delay-chat".into(),
            })
        }
        async fn stream(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    /// Embed provider that fails the first N calls (transient) then succeeds.
    struct FlakyEmbed {
        fails_remaining: AtomicU32,
        kind: EmbedFlakyKind,
        dim: usize,
        total_calls: AtomicU32,
    }

    #[allow(dead_code)]
    enum EmbedFlakyKind {
        RateLimited,
        Provider5xx,
        BadRequest,
    }

    #[async_trait]
    impl Embeddings for FlakyEmbed {
        fn name(&self) -> &str {
            "flaky-embed"
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            self.total_calls.fetch_add(1, Ordering::SeqCst);
            let n = self.fails_remaining.load(Ordering::SeqCst);
            if n > 0 {
                self.fails_remaining.fetch_sub(1, Ordering::SeqCst);
                return Err(match self.kind {
                    EmbedFlakyKind::RateLimited => {
                        Error::RateLimited { retry_after_ms: None }
                    }
                    EmbedFlakyKind::Provider5xx => Error::provider("502 bad gateway"),
                    EmbedFlakyKind::BadRequest => Error::invalid("bad request"),
                });
            }
            Ok(vec![0.1; self.dim])
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.total_calls.fetch_add(1, Ordering::SeqCst);
            let n = self.fails_remaining.load(Ordering::SeqCst);
            if n > 0 {
                self.fails_remaining.fetch_sub(1, Ordering::SeqCst);
                return Err(match self.kind {
                    EmbedFlakyKind::RateLimited => {
                        Error::RateLimited { retry_after_ms: None }
                    }
                    EmbedFlakyKind::Provider5xx => Error::provider("503 service unavailable"),
                    EmbedFlakyKind::BadRequest => Error::invalid("malformed batch"),
                });
            }
            Ok(vec![vec![0.1; self.dim]; texts.len()])
        }
    }

    fn flaky_embed(fails: u32, kind: EmbedFlakyKind, dim: usize) -> Arc<FlakyEmbed> {
        Arc::new(FlakyEmbed {
            fails_remaining: AtomicU32::new(fails),
            kind,
            dim,
            total_calls: AtomicU32::new(0),
        })
    }


    // ---- MetricsChatModel / MetricsEmbeddings tests --------------------

    #[tokio::test]
    async fn metrics_chat_records_invocations_and_latency() {
        let registry = MetricsRegistry::new();
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let mc = MetricsChatModel::new(inner, &registry);
        for _ in 0..3 {
            mc.invoke(vec![Message::user("hi")], &ChatOptions::default())
                .await
                .unwrap();
        }
        assert_eq!(registry.counter("chat_invocations_total").get(), 3);
        assert_eq!(registry.counter("chat_errors_total").get(), 0);
        // in_flight should be 0 after all calls return.
        assert_eq!(registry.gauge("chat_in_flight").get(), 0);
        // Latency histogram observed 3 times.
        assert_eq!(
            registry.histogram("chat_latency_seconds", DEFAULT_LATENCY_BUCKETS_SECS).count(),
            3,
        );
    }

    #[tokio::test]
    async fn metrics_chat_counts_errors() {
        let registry = MetricsRegistry::new();
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysFailModel {
            seen: AtomicU32::new(0),
        });
        let mc = MetricsChatModel::new(inner, &registry);
        for _ in 0..4 {
            let _ = mc
                .invoke(vec![Message::user("hi")], &ChatOptions::default())
                .await;
        }
        assert_eq!(registry.counter("chat_invocations_total").get(), 4);
        assert_eq!(registry.counter("chat_errors_total").get(), 4);
    }

    #[tokio::test]
    async fn metrics_chat_in_flight_gauge_tracks_concurrent_calls() {
        use std::sync::atomic::AtomicUsize;
        let registry = Arc::new(MetricsRegistry::new());
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn ChatModel> = Arc::new(DelayChatModel {
            delay_ms: 30,
            in_flight: in_flight.clone(),
            peak: peak.clone(),
        });
        let mc = Arc::new(MetricsChatModel::new(inner, registry.as_ref()));
        // Spawn 3 concurrent invokes; capture the gauge mid-flight.
        let mut handles = Vec::new();
        for _ in 0..3 {
            let mc = mc.clone();
            handles.push(tokio::spawn(async move {
                mc.invoke(vec![Message::user("hi")], &ChatOptions::default())
                    .await
            }));
        }
        // Sample mid-flight.
        tokio::time::sleep(Duration::from_millis(10)).await;
        let mid = registry.gauge("chat_in_flight").get();
        for h in handles {
            h.await.unwrap().unwrap();
        }
        // Some workers were in flight at sample time.
        assert!(mid >= 1, "in_flight gauge never observed concurrent work");
        // After all complete, gauge is back to 0.
        assert_eq!(registry.gauge("chat_in_flight").get(), 0);
    }

    #[tokio::test]
    async fn metrics_chat_with_prefix_uses_custom_name() {
        let registry = MetricsRegistry::new();
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let mc = MetricsChatModel::with_prefix(inner, &registry, "openai_gpt4");
        mc.invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(registry.counter("openai_gpt4_invocations_total").get(), 1);
        // Default chat_* counters should NOT be created.
        let prom = registry.to_prometheus();
        assert!(prom.contains("openai_gpt4_invocations_total 1"));
        assert!(!prom.contains("\nchat_invocations_total "));
    }

    #[tokio::test]
    async fn metrics_chat_name_proxies_inner() {
        let registry = MetricsRegistry::new();
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let mc = MetricsChatModel::new(inner, &registry);
        assert_eq!(mc.name(), "always-ok");
    }

    #[tokio::test]
    async fn metrics_embed_records_both_methods() {
        let registry = MetricsRegistry::new();
        let inner = flaky_embed(0, EmbedFlakyKind::Provider5xx, 4);
        let me = MetricsEmbeddings::new(inner as Arc<dyn Embeddings>, &registry);
        me.embed_query("hi").await.unwrap();
        me.embed_documents(&["a".into(), "b".into()]).await.unwrap();
        assert_eq!(registry.counter("embed_invocations_total").get(), 2);
        assert_eq!(registry.counter("embed_errors_total").get(), 0);
    }

    #[tokio::test]
    async fn metrics_embed_dimensions_proxy_inner() {
        let registry = MetricsRegistry::new();
        let inner = flaky_embed(0, EmbedFlakyKind::Provider5xx, 9);
        let me = MetricsEmbeddings::new(inner as Arc<dyn Embeddings>, &registry);
        assert_eq!(me.dimensions(), 9);
        assert_eq!(me.name(), "flaky-embed");
    }


    /// Echo tool: returns args under {"echo": ...}. Counts invocations.
    struct EchoTool {
        seen: AtomicU32,
    }

    #[async_trait]
    impl litgraph_core::tool::Tool for EchoTool {
        fn schema(&self) -> litgraph_core::tool::ToolSchema {
            litgraph_core::tool::ToolSchema {
                name: "echo".into(),
                description: "Echo args".into(),
                parameters: serde_json::json!({"type": "object"}),
            }
        }
        async fn run(
            &self,
            args: serde_json::Value,
        ) -> Result<serde_json::Value> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Ok(serde_json::json!({"echo": args}))
        }
    }

    /// Always errs with provider("502 ...") so the breaker counts every call
    /// as a failure. Counts invocations so we can verify the breaker
    /// short-circuits without invoking inner.
    struct AlwaysFailModel {
        seen: AtomicU32,
    }

    #[async_trait]
    impl ChatModel for AlwaysFailModel {
        fn name(&self) -> &str {
            "always-fail"
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Err(Error::provider("502 sick upstream"))
        }
        async fn stream(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatStream> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Err(Error::provider("502 sick upstream"))
        }
    }

    #[tokio::test]
    async fn metrics_tool_records_invocations_and_errors() {
        let registry = MetricsRegistry::new();
        let inner = Arc::new(EchoTool {
            seen: AtomicU32::new(0),
        });
        let mt = MetricsTool::new(
            inner as Arc<dyn litgraph_core::tool::Tool>,
            &registry,
        );
        // Tool name is "echo" → prefix sanitizes to "echo".
        for _ in 0..3 {
            mt.run(serde_json::json!({"x": 1})).await.unwrap();
        }
        assert_eq!(registry.counter("echo_invocations_total").get(), 3);
        assert_eq!(registry.counter("echo_errors_total").get(), 0);
        assert_eq!(registry.gauge("echo_in_flight").get(), 0);
        assert_eq!(
            registry
                .histogram("echo_latency_seconds", DEFAULT_LATENCY_BUCKETS_SECS)
                .count(),
            3,
        );
    }

    /// Always-fails tool to verify error counter and in_flight RAII guard.
    struct AlwaysFailTool;

    #[async_trait]
    impl litgraph_core::tool::Tool for AlwaysFailTool {
        fn schema(&self) -> litgraph_core::tool::ToolSchema {
            litgraph_core::tool::ToolSchema {
                name: "fail".into(),
                description: "always fails".into(),
                parameters: serde_json::json!({"type":"object"}),
            }
        }
        async fn run(
            &self,
            _args: serde_json::Value,
        ) -> Result<serde_json::Value> {
            Err(Error::other("synthetic"))
        }
    }

    #[tokio::test]
    async fn metrics_tool_counts_errors_and_decs_gauge() {
        let registry = MetricsRegistry::new();
        let inner: Arc<dyn litgraph_core::tool::Tool> = Arc::new(AlwaysFailTool);
        let mt = MetricsTool::new(inner, &registry);
        for _ in 0..4 {
            let _ = mt.run(serde_json::json!({})).await;
        }
        assert_eq!(registry.counter("fail_invocations_total").get(), 4);
        assert_eq!(registry.counter("fail_errors_total").get(), 4);
        // RAII guard decremented on every error path.
        assert_eq!(registry.gauge("fail_in_flight").get(), 0);
    }

    #[tokio::test]
    async fn metrics_tool_with_prefix_uses_custom_name() {
        let registry = MetricsRegistry::new();
        let inner = Arc::new(EchoTool {
            seen: AtomicU32::new(0),
        });
        let mt = MetricsTool::with_prefix(
            inner as Arc<dyn litgraph_core::tool::Tool>,
            &registry,
            "google_search",
        );
        mt.run(serde_json::json!({"q": "test"})).await.unwrap();
        assert_eq!(registry.counter("google_search_invocations_total").get(), 1);
        // Default echo_* metric should NOT be present.
        let prom = registry.to_prometheus();
        assert!(prom.contains("google_search_invocations_total 1"));
        assert!(!prom.contains("\necho_invocations_total "));
    }

    #[tokio::test]
    async fn metrics_tool_sanitizes_tool_name_for_prefix() {
        // A tool named with a dot (`http.get`) should produce metrics
        // under `http_get_*` — Prometheus disallows dots in names.
        struct DottedTool;
        #[async_trait]
        impl litgraph_core::tool::Tool for DottedTool {
            fn schema(&self) -> litgraph_core::tool::ToolSchema {
                litgraph_core::tool::ToolSchema {
                    name: "http.get".into(),
                    description: String::new(),
                    parameters: serde_json::json!({"type":"object"}),
                }
            }
            async fn run(
                &self,
                _args: serde_json::Value,
            ) -> Result<serde_json::Value> {
                Ok(serde_json::json!({}))
            }
        }
        let registry = MetricsRegistry::new();
        let mt = MetricsTool::new(
            Arc::new(DottedTool) as Arc<dyn litgraph_core::tool::Tool>,
            &registry,
        );
        mt.run(serde_json::json!({})).await.unwrap();
        assert_eq!(registry.counter("http_get_invocations_total").get(), 1);
    }

    #[tokio::test]
    async fn metrics_tool_schema_proxies_inner() {
        let registry = MetricsRegistry::new();
        let inner: Arc<dyn litgraph_core::tool::Tool> = Arc::new(EchoTool {
            seen: AtomicU32::new(0),
        });
        let mt = MetricsTool::new(inner, &registry);
        assert_eq!(mt.schema().name, "echo");
    }

    #[tokio::test]
    async fn metrics_chat_in_flight_decrements_on_error_path() {
        let registry = MetricsRegistry::new();
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysFailModel {
            seen: AtomicU32::new(0),
        });
        let mc = MetricsChatModel::new(inner, &registry);
        // 5 errors back-to-back; gauge must end at 0 (RAII guard
        // decs even on error).
        for _ in 0..5 {
            let _ = mc
                .invoke(vec![Message::user("hi")], &ChatOptions::default())
                .await;
        }
        assert_eq!(registry.gauge("chat_in_flight").get(), 0);
    }
}
