use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{Embeddings, Error, Result, Singleflight};

/// Wrap any [`litgraph_core::tool::Tool`] so concurrent calls
/// with identical args share ONE upstream `run`. Bridges the
/// iter-252 [`Singleflight`] primitive into the tool family —
/// closes the request-coalescing matrix across chat (intentionally
/// not coalesced), embed (iter 253), and tool (this iter).
///
/// # Hash key
///
/// blake3 over canonical JSON of `args` (same hash function as
/// iter 256 `tool_args_hash`). Identical args → identical hash
/// → coalesced call.
///
/// # Error handling
///
/// Errors broadcast as `Arc<String>` (the inner `Error`'s
/// `to_string()`) — same lossy-by-design tradeoff as
/// `SingleflightEmbeddings`. Variant info collapses to
/// `Error::Provider(s)` on the caller side.
///
/// # Real prod use
///
/// - **Idempotent expensive lookups**: `lookup_user("alice")`
///   called 10× concurrently from different agent steps → 1
///   DB round-trip.
/// - **Hot search query coalescing**: a popular search-tool
///   query embedded by 50 concurrent eval rows → 1 SerpAPI call.
/// - **Stable function tools**: tools whose output is a pure
///   function of args benefit; tools with side effects MUST
///   NOT be coalesced (would deduplicate the side effects).
///
/// # When NOT to coalesce
///
/// Tools with side effects (writes, sends, mutations) must NOT
/// be wrapped — coalescing collapses N intent-distinct calls
/// into one execution. Use only for idempotent reads /
/// pure-function tools.
pub struct SingleflightTool {
    pub inner: Arc<dyn litgraph_core::tool::Tool>,
    sf: Arc<Singleflight<String, Arc<std::result::Result<serde_json::Value, String>>>>,
}

impl SingleflightTool {
    pub fn new(inner: Arc<dyn litgraph_core::tool::Tool>) -> Self {
        Self {
            inner,
            sf: Arc::new(Singleflight::new()),
        }
    }
}

#[async_trait]
impl litgraph_core::tool::Tool for SingleflightTool {
    fn schema(&self) -> litgraph_core::tool::ToolSchema {
        self.inner.schema()
    }

    async fn run(
        &self,
        args: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let key = blake3::hash(
            serde_json::to_string(&args).unwrap_or_default().as_bytes(),
        )
        .to_hex()
        .to_string();
        let inner = self.inner.clone();
        let args_for_compute = args.clone();
        let r = self
            .sf
            .get_or_compute(key, move || async move {
                let res = inner.run(args_for_compute).await;
                Arc::new(match res {
                    Ok(v) => Ok(v),
                    Err(e) => Err(e.to_string()),
                })
            })
            .await;
        match &*r {
            Ok(v) => Ok(v.clone()),
            Err(s) => Err(Error::Provider(s.clone())),
        }
    }
}

/// Wrap any [`Embeddings`] so concurrent identical `embed_query`
/// calls share ONE upstream HTTP call. Bridges the iter-252
/// [`Singleflight`] primitive into the embeddings family.
///
/// # Why embed_query, not embed_documents
///
/// Embedding queries are repeated often (the same user query
/// from many threads, the same system-prompt prefix from many
/// agents). Coalescing them is high-value. Multi-doc batches in
/// `embed_documents` are typically distinct per call (a chunked
/// indexer's chunks differ batch-to-batch); coalescing them is
/// rare-win and would require a hashable key over `Vec<String>`.
/// `embed_documents` passes through unchanged.
///
/// # Error handling
///
/// Errors broadcast as `Arc<String>` (the original `Error`'s
/// `to_string()`) — the variant info (`RateLimited`,
/// `Timeout`, `InvalidInput`, etc.) collapses to
/// `Error::Provider(s)` on the caller side. This is lossy by
/// design: `Error` isn't `Clone`, and the alternative (running
/// each follower's compute when leader fails) defeats the whole
/// purpose of coalescing under a flapping upstream. Callers who
/// need exact error variants should not coalesce.
///
/// # Real prod use
///
/// - **System-prompt cache priming**: 50 agents start; each
///   embeds the same long system prompt. One HTTP call.
/// - **Hot query dedup**: a popular search query gets embedded
///   100×/sec from different threads. One call per unique query
///   per Singleflight window.
/// - **Eval harness deduplication**: golden-set runner
///   evaluates the same query against many retrievers; the
///   query embedding only needs computing once.
pub struct SingleflightEmbeddings {
    pub inner: Arc<dyn Embeddings>,
    sf: Arc<Singleflight<String, Arc<std::result::Result<Vec<f32>, String>>>>,
}

impl SingleflightEmbeddings {
    pub fn new(inner: Arc<dyn Embeddings>) -> Self {
        Self {
            inner,
            sf: Arc::new(Singleflight::new()),
        }
    }
}

#[async_trait]
impl Embeddings for SingleflightEmbeddings {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let inner = self.inner.clone();
        let text_owned = text.to_string();
        let r = self
            .sf
            .get_or_compute(text.to_string(), move || async move {
                let res = inner.embed_query(&text_owned).await;
                Arc::new(match res {
                    Ok(v) => Ok(v),
                    Err(e) => Err(e.to_string()),
                })
            })
            .await;
        match &*r {
            Ok(v) => Ok(v.clone()),
            Err(s) => Err(Error::Provider(s.clone())),
        }
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Pass-through (see doc above).
        self.inner.embed_documents(texts).await
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


    // ---- SingleflightEmbeddings tests ----------------------------------

    /// Counts inner calls per text. Lets tests verify dedup happened.
    struct SfCountingEmbed {
        delay_ms: u64,
        seen: AtomicU32,
        dim: usize,
    }

    #[async_trait]
    impl Embeddings for SfCountingEmbed {
        fn name(&self) -> &str {
            "sf-counting-embed"
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            // Embedding is deterministic from text length so we can
            // verify followers got the leader's actual value.
            Ok(vec![text.len() as f32; self.dim])
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Ok(vec![vec![0.0; self.dim]; texts.len()])
        }
    }

    #[tokio::test]
    async fn singleflight_concurrent_same_query_one_inner_call() {
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 30,
            seen: AtomicU32::new(0),
            dim: 4,
        });
        let sf = Arc::new(SingleflightEmbeddings::new(
            inner.clone() as Arc<dyn Embeddings>,
        ));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let sf = sf.clone();
            handles.push(tokio::spawn(async move {
                sf.embed_query("same query").await
            }));
        }
        for h in handles {
            let v = h.await.unwrap().unwrap();
            // All callers got the leader's deterministic value.
            assert_eq!(v, vec!["same query".len() as f32; 4]);
        }
        assert_eq!(
            inner.seen.load(Ordering::SeqCst),
            1,
            "inner.embed_query ran more than once",
        );
    }

    #[tokio::test]
    async fn singleflight_different_queries_run_independently() {
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 5,
            seen: AtomicU32::new(0),
            dim: 3,
        });
        let sf = Arc::new(SingleflightEmbeddings::new(
            inner.clone() as Arc<dyn Embeddings>,
        ));
        let mut handles = Vec::new();
        for q in ["a", "bb", "ccc", "dddd"] {
            let sf = sf.clone();
            handles.push(tokio::spawn(async move {
                sf.embed_query(q).await
            }));
        }
        for h in handles {
            let _ = h.await.unwrap().unwrap();
        }
        assert_eq!(inner.seen.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn singleflight_embed_documents_passes_through() {
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 0,
            seen: AtomicU32::new(0),
            dim: 2,
        });
        let sf = SingleflightEmbeddings::new(inner.clone() as Arc<dyn Embeddings>);
        let _ = sf.embed_documents(&["a".into(), "b".into()]).await.unwrap();
        let _ = sf.embed_documents(&["a".into(), "b".into()]).await.unwrap();
        // Pass-through, no dedup.
        assert_eq!(inner.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn singleflight_propagates_errors_as_provider_error() {
        // FlakyEmbed with 1 fail then OK; first call gets the
        // 502 error, second call past the in-flight window starts
        // fresh and succeeds.
        let inner = flaky_embed(1, EmbedFlakyKind::Provider5xx, 3);
        let sf = SingleflightEmbeddings::new(inner.clone() as Arc<dyn Embeddings>);
        let r1 = sf.embed_query("q").await;
        // Error variant collapses to Error::Provider(...) on the
        // singleflight path (lossy-by-design).
        match r1 {
            Err(Error::Provider(s)) => {
                assert!(s.contains("502") || s.contains("provider"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
        // Next call: in-flight window closed, fresh compute, succeeds.
        let r2 = sf.embed_query("q").await;
        assert!(r2.is_ok());
    }

    #[tokio::test]
    async fn singleflight_name_and_dimensions_proxy_inner() {
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 0,
            seen: AtomicU32::new(0),
            dim: 7,
        });
        let sf = SingleflightEmbeddings::new(inner as Arc<dyn Embeddings>);
        assert_eq!(sf.name(), "sf-counting-embed");
        assert_eq!(sf.dimensions(), 7);
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

    // ---- SingleflightTool tests ----------------------------------------

    /// Slow echo tool: sleeps `delay_ms` then echoes args. Counts
    /// inner invocations so tests verify dedup happened.
    struct SlowEchoTool {
        seen: AtomicU32,
        delay_ms: u64,
    }

    #[async_trait]
    impl litgraph_core::tool::Tool for SlowEchoTool {
        fn schema(&self) -> litgraph_core::tool::ToolSchema {
            litgraph_core::tool::ToolSchema {
                name: "slow_echo".into(),
                description: "echo with delay".into(),
                parameters: serde_json::json!({"type":"object"}),
            }
        }
        async fn run(
            &self,
            args: serde_json::Value,
        ) -> Result<serde_json::Value> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(serde_json::json!({"echo": args}))
        }
    }

    #[tokio::test]
    async fn singleflight_tool_concurrent_same_args_one_call() {
        let inner = Arc::new(SlowEchoTool {
            seen: AtomicU32::new(0),
            delay_ms: 30,
        });
        let sf = Arc::new(SingleflightTool::new(
            inner.clone() as Arc<dyn litgraph_core::tool::Tool>,
        ));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let sf = sf.clone();
            handles.push(tokio::spawn(async move {
                sf.run(serde_json::json!({"key": "shared"})).await
            }));
        }
        for h in handles {
            let v = h.await.unwrap().unwrap();
            assert_eq!(v, serde_json::json!({"echo": {"key": "shared"}}));
        }
        // Single inner call despite 10 concurrent identical args.
        assert_eq!(
            inner.seen.load(Ordering::SeqCst),
            1,
            "inner ran more than once",
        );
    }

    #[tokio::test]
    async fn singleflight_tool_different_args_run_independently() {
        let inner = Arc::new(SlowEchoTool {
            seen: AtomicU32::new(0),
            delay_ms: 5,
        });
        let sf = Arc::new(SingleflightTool::new(
            inner.clone() as Arc<dyn litgraph_core::tool::Tool>,
        ));
        let mut handles = Vec::new();
        for i in 0..5 {
            let sf = sf.clone();
            handles.push(tokio::spawn(async move {
                sf.run(serde_json::json!({"i": i})).await
            }));
        }
        for h in handles {
            let _ = h.await.unwrap().unwrap();
        }
        assert_eq!(inner.seen.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn singleflight_tool_propagates_errors_as_provider_error() {
        let inner: Arc<dyn litgraph_core::tool::Tool> = Arc::new(AlwaysFailTool);
        let sf = SingleflightTool::new(inner);
        let r = sf.run(serde_json::json!({"q": "bad"})).await;
        match r {
            Err(Error::Provider(msg)) => {
                assert!(msg.contains("synthetic"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn singleflight_tool_window_closes_after_completion() {
        // First call runs compute; second call after the in-flight
        // window closes runs compute again.
        let inner = Arc::new(SlowEchoTool {
            seen: AtomicU32::new(0),
            delay_ms: 0,
        });
        let sf = SingleflightTool::new(
            inner.clone() as Arc<dyn litgraph_core::tool::Tool>,
        );
        let _ = sf.run(serde_json::json!({"q": "a"})).await.unwrap();
        let _ = sf.run(serde_json::json!({"q": "a"})).await.unwrap();
        // Both calls ran independently — coalescing only inside an
        // in-flight window.
        assert_eq!(inner.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn singleflight_tool_schema_proxies_inner() {
        let inner: Arc<dyn litgraph_core::tool::Tool> = Arc::new(EchoTool {
            seen: AtomicU32::new(0),
        });
        let sf = SingleflightTool::new(inner);
        assert_eq!(sf.schema().name, "echo");
    }
}
