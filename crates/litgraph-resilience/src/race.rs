use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{
    hedged_call, ChatModel, ChatOptions, ChatResponse, Embeddings, Error, Message, Result,
};

/// Invokes N inner `ChatModel`s **concurrently**, returns the first one
/// that succeeds. Outstanding invocations are aborted as soon as a
/// winner emerges.
///
/// Use this when latency matters more than cost: paying for N parallel
/// requests buys you `min(t_1, .., t_N)` end-to-end. Typical patterns:
///
/// - Multi-region failover where the request hedges across regions
///   instead of waiting for one to time out.
/// - Latency-critical paths backed by both a fast cheap model and a
///   slow strong model — race them and take whichever finishes first.
/// - Cross-provider speculative serving (OpenAI + Anthropic +
///   Bedrock) for tail-latency reduction.
///
/// # Cost model
///
/// All N requests are *issued*. Cancellation aborts the in-flight
/// future, but providers have already begun processing — most bill for
/// any tokens generated before the connection closes. Budget for
/// `cost_floor ≈ N × p50_inference_cost` even though latency is `p_min`.
///
/// # Failure
///
/// Returns `Ok` as soon as **any** inner returns `Ok`. Returns `Err`
/// only if **every** inner fails — error message aggregates all
/// failures (newline-separated) so the caller can debug.
///
/// # Streaming
///
/// `stream()` falls through to `inners[0]`. Racing token streams is
/// possible (race for first chunk, then commit) but inviting in
/// practice — providers' first chunks vary wildly in latency and
/// quality, and switching mid-stream is impossible. Consumers who
/// need it can race `invoke()` calls themselves.
///
/// # Composition
///
/// `Race(Retry(A), Retry(B))` is the typical shape — let each branch
/// handle its own transient errors, race the steady-state outcomes.
/// Avoid `Race(Race(A,B), C)` — flatten the chain into a single
/// `Race(A, B, C)` so cancellation works across the whole set.
///
/// ```ignore
/// use std::sync::Arc;
/// use litgraph_resilience::RaceChatModel;
/// let race = RaceChatModel::new(vec![openai_arc, anthropic_arc, bedrock_arc]);
/// // First to return wins; the other two are aborted.
/// let resp = race.invoke(messages, &opts).await?;
/// ```
pub struct RaceChatModel {
    pub inners: Vec<Arc<dyn ChatModel>>,
}

impl RaceChatModel {
    /// Build a race set. Panics if `inners` is empty (a race with no
    /// runners can't yield a winner).
    pub fn new(inners: Vec<Arc<dyn ChatModel>>) -> Self {
        assert!(
            !inners.is_empty(),
            "RaceChatModel: need at least one inner model",
        );
        Self { inners }
    }
}

#[async_trait]
impl ChatModel for RaceChatModel {
    fn name(&self) -> &str {
        "race"
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        use tokio::task::JoinSet;

        // Single-inner shortcut keeps the spawn-overhead off the hot
        // path for users who probe with a one-element race.
        if self.inners.len() == 1 {
            return self.inners[0].invoke(messages, opts).await;
        }

        let mut set: JoinSet<Result<ChatResponse>> = JoinSet::new();
        for inner in self.inners.iter() {
            let inner = inner.clone();
            let msgs = messages.clone();
            let opts = opts.clone();
            set.spawn(async move { inner.invoke(msgs, &opts).await });
        }

        let mut errors: Vec<String> = Vec::with_capacity(self.inners.len());
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(resp)) => {
                    // Winner — abort everything else and return.
                    set.abort_all();
                    return Ok(resp);
                }
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(e) => errors.push(format!("task join: {e}")),
            }
        }
        Err(Error::other(format!(
            "RaceChatModel: all {} inners failed:\n  - {}",
            self.inners.len(),
            errors.join("\n  - "),
        )))
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // Mid-stream switching is unworkable. See module doc.
        self.inners[0].stream(messages, opts).await
    }
}

/// Embeddings analogue of [`RaceChatModel`]: invokes every inner
/// `Embeddings` provider concurrently, returns the first successful
/// result, aborts the others.
///
/// # Why use this over `FallbackEmbeddings`
///
/// `FallbackEmbeddings` is **sequential** — try A, on failure try B
/// (cost-min, but pays A's full latency before B even starts).
/// `RaceEmbeddings` is **concurrent** — issue A, B, C at the same
/// instant, take whoever returns first (latency-min, pays for all
/// requests but cuts the p95).
///
/// Typical patterns:
/// - Hedge a remote provider (OpenAI / Voyage / Cohere) against a
///   local fastembed cross-encoder. Local usually wins on warm
///   cache; remote wins when local hits CPU pressure.
/// - Multi-region embedding: us-east + eu-west, race for whichever
///   is closer to the caller's pod.
/// - Tail-latency reduction on the embed-query critical path —
///   `embed_query` blocks every retrieval, so shaving 50ms off p95
///   compounds over a session.
///
/// # Cost
///
/// All N providers are *issued* — most bill for tokens generated
/// before the connection closes. Budget `cost_floor ≈ N × p50_cost`
/// even though latency is `p_min`.
///
/// # Dimensions
///
/// All inners must agree on `dimensions()` — checked at construction
/// (panic). Race semantics on the result mean a 1536-dim winner this
/// call vs a 512-dim winner next call would silently corrupt your
/// vector index, so we forbid the configuration outright.
pub struct RaceEmbeddings {
    pub inners: Vec<Arc<dyn Embeddings>>,
    dim: usize,
    name_label: String,
}

impl RaceEmbeddings {
    /// Build a race set. Panics if `inners` is empty or any two
    /// inners disagree on `dimensions()`.
    pub fn new(inners: Vec<Arc<dyn Embeddings>>) -> Self {
        assert!(
            !inners.is_empty(),
            "RaceEmbeddings: need at least one inner provider",
        );
        let dim = inners[0].dimensions();
        for (i, e) in inners.iter().enumerate().skip(1) {
            assert_eq!(
                e.dimensions(),
                dim,
                "RaceEmbeddings: inner #{} has dim {} but inner #0 has dim {} — \
                 race semantics + dim mismatch would silently corrupt the vector index",
                i,
                e.dimensions(),
                dim,
            );
        }
        let labels: Vec<String> = inners.iter().map(|e| e.name().to_string()).collect();
        Self {
            inners,
            dim,
            name_label: format!("race({})", labels.join(", ")),
        }
    }
}

#[async_trait]
impl Embeddings for RaceEmbeddings {
    fn name(&self) -> &str {
        &self.name_label
    }
    fn dimensions(&self) -> usize {
        self.dim
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        use tokio::task::JoinSet;

        if self.inners.len() == 1 {
            return self.inners[0].embed_query(text).await;
        }

        let mut set: JoinSet<Result<Vec<f32>>> = JoinSet::new();
        for inner in self.inners.iter() {
            let inner = inner.clone();
            let text = text.to_string();
            set.spawn(async move { inner.embed_query(&text).await });
        }

        let mut errors: Vec<String> = Vec::with_capacity(self.inners.len());
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(v)) => {
                    set.abort_all();
                    return Ok(v);
                }
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(e) => errors.push(format!("task join: {e}")),
            }
        }
        Err(Error::other(format!(
            "RaceEmbeddings.embed_query: all {} inners failed:\n  - {}",
            self.inners.len(),
            errors.join("\n  - "),
        )))
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        use tokio::task::JoinSet;

        if self.inners.len() == 1 {
            return self.inners[0].embed_documents(texts).await;
        }

        let mut set: JoinSet<Result<Vec<Vec<f32>>>> = JoinSet::new();
        for inner in self.inners.iter() {
            let inner = inner.clone();
            let owned: Vec<String> = texts.to_vec();
            set.spawn(async move { inner.embed_documents(&owned).await });
        }

        let mut errors: Vec<String> = Vec::with_capacity(self.inners.len());
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(v)) => {
                    set.abort_all();
                    return Ok(v);
                }
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(e) => errors.push(format!("task join: {e}")),
            }
        }
        Err(Error::other(format!(
            "RaceEmbeddings.embed_documents: all {} inners failed:\n  - {}",
            self.inners.len(),
            errors.join("\n  - "),
        )))
    }
}

/// Wrap two [`ChatModel`]s so each call goes to `primary` first;
/// if primary hasn't finished within `hedge_delay`, also issue
/// the same call to `backup` and return whichever finishes first.
/// Bridges the iter-250 [`hedged_call`] combinator into the chat
/// family.
///
/// # Distinct from `RaceChatModel`
///
/// `RaceChatModel` (iter 184) issues to both providers
/// simultaneously — every call doubles cost. Hedge only pays
/// the second-call cost when primary is slow, which is the
/// right trade-off for tail-latency mitigation where the
/// median is fine and you only want to insure against the p99.
///
/// # Real prod use
///
/// - **LLM tail latency**: provider with 500ms p50 / 30s p99 —
///   set `hedge_delay = 2s`; calls under 2s use only primary,
///   slow tail covered by backup.
/// - **Multi-region failover**: primary in us-east-1, backup
///   in us-west-2 — hedge after 1s.
/// - **Replica hedging**: same provider, two API keys;
///   distributes load + insures against per-key throttling.
///
/// # Streaming
///
/// `stream()` is NOT hedged — token streams can't be cleanly
/// raced (chunks can't be merged or chosen between mid-stream).
/// Stream calls pass through to `primary` only. Callers needing
/// stream tail-latency mitigation should run their own
/// per-chunk timeout + restart-on-different-provider logic.
pub struct HedgedChatModel {
    pub primary: Arc<dyn ChatModel>,
    pub backup: Arc<dyn ChatModel>,
    pub hedge_delay: Duration,
}

impl HedgedChatModel {
    pub fn new(
        primary: Arc<dyn ChatModel>,
        backup: Arc<dyn ChatModel>,
        hedge_delay: Duration,
    ) -> Self {
        Self {
            primary,
            backup,
            hedge_delay,
        }
    }
}

#[async_trait]
impl ChatModel for HedgedChatModel {
    fn name(&self) -> &str {
        self.primary.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let primary = self.primary.clone();
        let backup = self.backup.clone();
        let messages_p = messages.clone();
        let opts_p = opts.clone();
        let opts_b = opts.clone();
        hedged_call(
            move || async move { primary.invoke(messages_p, &opts_p).await },
            move || async move { backup.invoke(messages, &opts_b).await },
            self.hedge_delay,
        )
        .await
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // Streams aren't hedged (see doc above). Pass through to primary.
        self.primary.stream(messages, opts).await
    }
}

/// Embed-axis sibling. Hedges both `embed_query` and
/// `embed_documents` against the backup embedder. Backup is
/// only invoked when the primary call exceeds `hedge_delay`.
pub struct HedgedEmbeddings {
    pub primary: Arc<dyn Embeddings>,
    pub backup: Arc<dyn Embeddings>,
    pub hedge_delay: Duration,
}

impl HedgedEmbeddings {
    pub fn new(
        primary: Arc<dyn Embeddings>,
        backup: Arc<dyn Embeddings>,
        hedge_delay: Duration,
    ) -> Self {
        Self {
            primary,
            backup,
            hedge_delay,
        }
    }
}

#[async_trait]
impl Embeddings for HedgedEmbeddings {
    fn name(&self) -> &str {
        self.primary.name()
    }
    fn dimensions(&self) -> usize {
        self.primary.dimensions()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let primary = self.primary.clone();
        let backup = self.backup.clone();
        let text_p = text.to_owned();
        let text_b = text.to_owned();
        hedged_call(
            move || async move { primary.embed_query(&text_p).await },
            move || async move { backup.embed_query(&text_b).await },
            self.hedge_delay,
        )
        .await
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let primary = self.primary.clone();
        let backup = self.backup.clone();
        let texts_p = texts.to_vec();
        let texts_b = texts.to_vec();
        hedged_call(
            move || async move { primary.embed_documents(&texts_p).await },
            move || async move { backup.embed_documents(&texts_b).await },
            self.hedge_delay,
        )
        .await
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

    // ---- RaceChatModel tests ----------------------------------------------

    /// Sleeps `delay_ms` then either returns a labelled response or errors.
    struct DelayedModel {
        label: &'static str,
        delay_ms: u64,
        succeed: bool,
        invocations: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl ChatModel for DelayedModel {
        fn name(&self) -> &str {
            self.label
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.invocations
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            if self.succeed {
                Ok(ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content: vec![ContentPart::Text {
                            text: self.label.into(),
                        }],
                        tool_calls: vec![],
                        tool_call_id: None,
                        name: None,
                        cache: false,
                    },
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage::default(),
                    model: self.label.into(),
                })
            } else {
                Err(Error::provider(format!("{} failed", self.label)))
            }
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn arc_dm(
        label: &'static str,
        delay_ms: u64,
        succeed: bool,
    ) -> (Arc<dyn ChatModel>, Arc<std::sync::atomic::AtomicUsize>) {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let m = DelayedModel {
            label,
            delay_ms,
            succeed,
            invocations: count.clone(),
        };
        (Arc::new(m), count)
    }

    #[tokio::test]
    #[should_panic(expected = "at least one inner model")]
    async fn race_panics_on_empty() {
        let _ = RaceChatModel::new(vec![]);
    }

    #[tokio::test]
    async fn race_returns_first_winner() {
        // A finishes in 5ms, B in 50ms — A must win.
        let (a, _) = arc_dm("a", 5, true);
        let (b, _) = arc_dm("b", 50, true);
        let race = RaceChatModel::new(vec![a, b]);
        let resp = race
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(resp.model, "a");
        assert_eq!(resp.message.text_content(), "a");
    }

    #[tokio::test]
    async fn race_falls_through_failures() {
        // A fails immediately, B succeeds slowly — B is the answer.
        let (a, _) = arc_dm("a", 1, false);
        let (b, _) = arc_dm("b", 10, true);
        let race = RaceChatModel::new(vec![a, b]);
        let resp = race
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(resp.model, "b");
    }

    #[tokio::test]
    async fn race_aggregates_when_all_fail() {
        let (a, _) = arc_dm("a", 1, false);
        let (b, _) = arc_dm("b", 2, false);
        let (c, _) = arc_dm("c", 3, false);
        let race = RaceChatModel::new(vec![a, b, c]);
        let err = race
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("all 3 inners failed"), "got: {s}");
        assert!(s.contains("a failed"));
        assert!(s.contains("b failed"));
        assert!(s.contains("c failed"));
    }

    #[tokio::test]
    async fn race_single_inner_passes_through() {
        let (a, count) = arc_dm("a", 0, true);
        let race = RaceChatModel::new(vec![a]);
        let _ = race
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn race_aborts_losers_after_winner() {
        // A finishes in 5ms; B sleeps 500ms. After A wins, B's task
        // should be aborted — measured by the wall-clock total being
        // closer to A's 5ms than B's 500ms.
        let (a, _) = arc_dm("a", 5, true);
        let (b, _) = arc_dm("b", 500, true);
        let race = RaceChatModel::new(vec![a, b]);
        let started = std::time::Instant::now();
        let _ = race
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        let elapsed_ms = started.elapsed().as_millis() as u64;
        // Allow generous slack for CI variance, but B's 500ms must NOT
        // dominate.
        assert!(
            elapsed_ms < 200,
            "elapsed {elapsed_ms}ms — losers were not aborted",
        );
    }


    // ---- RaceEmbeddings tests ------------------------------------------

    /// Embed provider that sleeps `delay_ms` then either returns a
    /// `dim`-vector of `marker` floats or errors. Lets us pin which
    /// inner wins a race deterministically.
    struct DelayedEmbed {
        label: &'static str,
        dim: usize,
        delay_ms: u64,
        marker: f32,
        succeed: bool,
    }

    #[async_trait]
    impl Embeddings for DelayedEmbed {
        fn name(&self) -> &str {
            self.label
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            if self.succeed {
                Ok(vec![self.marker; self.dim])
            } else {
                Err(Error::provider(format!("{} failed", self.label)))
            }
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            if self.succeed {
                Ok(vec![vec![self.marker; self.dim]; texts.len()])
            } else {
                Err(Error::provider(format!("{} failed", self.label)))
            }
        }
    }

    fn delayed(
        label: &'static str,
        dim: usize,
        delay_ms: u64,
        marker: f32,
        succeed: bool,
    ) -> Arc<dyn Embeddings> {
        Arc::new(DelayedEmbed {
            label,
            dim,
            delay_ms,
            marker,
            succeed,
        })
    }

    #[tokio::test]
    #[should_panic(expected = "at least one inner provider")]
    async fn race_embed_panics_on_empty() {
        let _ = RaceEmbeddings::new(vec![]);
    }

    #[tokio::test]
    #[should_panic(expected = "dim mismatch")]
    async fn race_embed_panics_on_dim_mismatch() {
        let a = delayed("a", 1536, 1, 0.1, true);
        let b = delayed("b", 768, 1, 0.2, true);
        let _ = RaceEmbeddings::new(vec![a, b]);
    }

    #[tokio::test]
    async fn race_embed_query_returns_first_winner() {
        // a finishes in 5ms, b in 50ms — a wins, marker=0.42 ↦ all 1536 dims = 0.42
        let a = delayed("a", 4, 5, 0.42, true);
        let b = delayed("b", 4, 50, 0.99, true);
        let race = RaceEmbeddings::new(vec![a, b]);
        let out = race.embed_query("hi").await.unwrap();
        assert_eq!(out, vec![0.42; 4]);
    }

    #[tokio::test]
    async fn race_embed_query_falls_through_failures() {
        // a fails fast, b succeeds slowly → b's vector is the result.
        let a = delayed("a", 4, 1, 0.0, false);
        let b = delayed("b", 4, 10, 0.5, true);
        let race = RaceEmbeddings::new(vec![a, b]);
        let out = race.embed_query("q").await.unwrap();
        assert_eq!(out, vec![0.5; 4]);
    }

    #[tokio::test]
    async fn race_embed_query_aggregates_when_all_fail() {
        let a = delayed("a", 4, 1, 0.0, false);
        let b = delayed("b", 4, 2, 0.0, false);
        let race = RaceEmbeddings::new(vec![a, b]);
        let err = race.embed_query("q").await.unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("all 2 inners failed"), "got: {s}");
        assert!(s.contains("a failed"));
        assert!(s.contains("b failed"));
    }

    #[tokio::test]
    async fn race_embed_documents_returns_first_winner() {
        let a = delayed("a", 4, 5, 0.42, true);
        let b = delayed("b", 4, 50, 0.99, true);
        let race = RaceEmbeddings::new(vec![a, b]);
        let out = race
            .embed_documents(&["hi".into(), "there".into()])
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], vec![0.42; 4]);
        assert_eq!(out[1], vec![0.42; 4]);
    }

    #[tokio::test]
    async fn race_embed_single_inner_passes_through() {
        let a = delayed("only", 4, 0, 0.7, true);
        let race = RaceEmbeddings::new(vec![a]);
        let out = race.embed_query("q").await.unwrap();
        assert_eq!(out, vec![0.7; 4]);
    }

    #[tokio::test]
    async fn race_embed_aborts_losers_after_winner() {
        // a wins in 5ms, b sleeps 500ms — total wall-clock should be
        // closer to 5ms than 500ms.
        let a = delayed("a", 4, 5, 0.42, true);
        let b = delayed("b", 4, 500, 0.99, true);
        let race = RaceEmbeddings::new(vec![a, b]);
        let started = std::time::Instant::now();
        let _ = race.embed_query("q").await.unwrap();
        let elapsed_ms = started.elapsed().as_millis() as u64;
        assert!(
            elapsed_ms < 200,
            "elapsed {elapsed_ms}ms — losers were not aborted",
        );
    }

    #[tokio::test]
    async fn race_embed_name_includes_all_inners() {
        let a = delayed("openai", 4, 0, 0.0, true);
        let b = delayed("voyage", 4, 0, 0.0, true);
        let race = RaceEmbeddings::new(vec![a, b]);
        assert!(race.name().contains("openai"));
        assert!(race.name().contains("voyage"));
        assert!(race.name().starts_with("race("));
    }


    // ---- HedgedChatModel / HedgedEmbeddings tests ----------------------

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

    /// Returns a fixed text after `delay_ms`. Counts invocations so
    /// tests can assert whether backup was actually invoked.
    struct LabeledDelayChat {
        delay_ms: u64,
        label: String,
        seen: AtomicU32,
    }

    #[async_trait]
    impl ChatModel for LabeledDelayChat {
        fn name(&self) -> &str {
            "labeled-delay-chat"
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text {
                        text: self.label.clone(),
                    }],
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "labeled-delay-chat".into(),
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

    #[tokio::test]
    async fn hedged_chat_primary_wins_when_fast_no_backup_invoked() {
        let primary = Arc::new(LabeledDelayChat {
            delay_ms: 10,
            label: "PRIMARY".into(),
            seen: AtomicU32::new(0),
        });
        let backup = Arc::new(LabeledDelayChat {
            delay_ms: 10,
            label: "BACKUP".into(),
            seen: AtomicU32::new(0),
        });
        let hc = HedgedChatModel::new(
            primary.clone() as Arc<dyn ChatModel>,
            backup.clone() as Arc<dyn ChatModel>,
            Duration::from_millis(50),
        );
        let resp = hc
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(resp.message.text_content(), "PRIMARY");
        assert_eq!(primary.seen.load(Ordering::SeqCst), 1);
        // Backup never even invoked.
        assert_eq!(backup.seen.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn hedged_chat_backup_wins_when_primary_slow() {
        let primary = Arc::new(LabeledDelayChat {
            delay_ms: 100,
            label: "PRIMARY".into(),
            seen: AtomicU32::new(0),
        });
        let backup = Arc::new(LabeledDelayChat {
            delay_ms: 10,
            label: "BACKUP".into(),
            seen: AtomicU32::new(0),
        });
        let hc = HedgedChatModel::new(
            primary.clone() as Arc<dyn ChatModel>,
            backup.clone() as Arc<dyn ChatModel>,
            Duration::from_millis(20),
        );
        let resp = hc
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        // Primary started at 0ms, finishes at 100ms.
        // Backup started at 20ms, finishes at 30ms → wins.
        assert_eq!(resp.message.text_content(), "BACKUP");
        assert_eq!(primary.seen.load(Ordering::SeqCst), 1);
        assert_eq!(backup.seen.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn hedged_chat_stream_passes_through_to_primary_only() {
        // stream() is documented as primary-only. Use a model that
        // panics on stream so we'd notice if backup was hit.
        struct StreamCounter {
            seen: AtomicU32,
            text: String,
        }
        #[async_trait]
        impl ChatModel for StreamCounter {
            fn name(&self) -> &str {
                "stream-counter"
            }
            async fn invoke(
                &self,
                _m: Vec<Message>,
                _o: &ChatOptions,
            ) -> Result<ChatResponse> {
                Ok(ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content: vec![ContentPart::Text {
                            text: self.text.clone(),
                        }],
                        tool_calls: vec![],
                        tool_call_id: None,
                        name: None,
                        cache: false,
                    },
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage::default(),
                    model: "stream-counter".into(),
                })
            }
            async fn stream(
                &self,
                _m: Vec<Message>,
                _o: &ChatOptions,
            ) -> Result<ChatStream> {
                self.seen.fetch_add(1, Ordering::SeqCst);
                Err(Error::other("stream not implemented for test"))
            }
        }
        let primary = Arc::new(StreamCounter {
            seen: AtomicU32::new(0),
            text: "P".into(),
        });
        let backup = Arc::new(StreamCounter {
            seen: AtomicU32::new(0),
            text: "B".into(),
        });
        let hc = HedgedChatModel::new(
            primary.clone() as Arc<dyn ChatModel>,
            backup.clone() as Arc<dyn ChatModel>,
            Duration::from_millis(0),
        );
        let _ = hc
            .stream(vec![Message::user("hi")], &ChatOptions::default())
            .await;
        // Stream hit primary exactly once; backup not touched.
        assert_eq!(primary.seen.load(Ordering::SeqCst), 1);
        assert_eq!(backup.seen.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn hedged_chat_name_proxies_primary() {
        let primary: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let backup: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let hc = HedgedChatModel::new(primary, backup, Duration::from_millis(50));
        assert_eq!(hc.name(), "always-ok");
    }

    #[tokio::test]
    async fn hedged_embed_primary_wins_when_fast() {
        // Re-use FlakyEmbed with fails=0 (always succeeds) and a
        // wall-clock delay shim. Simplest: make a delayed dummy embedder.
        struct DelayEmbed {
            delay_ms: u64,
            tag: f32,
            seen: AtomicU32,
        }
        #[async_trait]
        impl Embeddings for DelayEmbed {
            fn name(&self) -> &str {
                "delay-embed"
            }
            fn dimensions(&self) -> usize {
                3
            }
            async fn embed_query(&self, _t: &str) -> Result<Vec<f32>> {
                self.seen.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
                Ok(vec![self.tag; 3])
            }
            async fn embed_documents(
                &self,
                texts: &[String],
            ) -> Result<Vec<Vec<f32>>> {
                self.seen.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
                Ok(vec![vec![self.tag; 3]; texts.len()])
            }
        }
        let primary = Arc::new(DelayEmbed {
            delay_ms: 5,
            tag: 1.0,
            seen: AtomicU32::new(0),
        });
        let backup = Arc::new(DelayEmbed {
            delay_ms: 5,
            tag: 9.0,
            seen: AtomicU32::new(0),
        });
        let he = HedgedEmbeddings::new(
            primary.clone() as Arc<dyn Embeddings>,
            backup.clone() as Arc<dyn Embeddings>,
            Duration::from_millis(50),
        );
        let v = he.embed_query("hi").await.unwrap();
        assert_eq!(v[0], 1.0);
        assert_eq!(backup.seen.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn hedged_embed_backup_wins_when_primary_slow() {
        struct DelayEmbed {
            delay_ms: u64,
            tag: f32,
            seen: AtomicU32,
        }
        #[async_trait]
        impl Embeddings for DelayEmbed {
            fn name(&self) -> &str {
                "delay-embed"
            }
            fn dimensions(&self) -> usize {
                2
            }
            async fn embed_query(&self, _t: &str) -> Result<Vec<f32>> {
                self.seen.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
                Ok(vec![self.tag; 2])
            }
            async fn embed_documents(
                &self,
                texts: &[String],
            ) -> Result<Vec<Vec<f32>>> {
                self.seen.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
                Ok(vec![vec![self.tag; 2]; texts.len()])
            }
        }
        let primary = Arc::new(DelayEmbed {
            delay_ms: 100,
            tag: 1.0,
            seen: AtomicU32::new(0),
        });
        let backup = Arc::new(DelayEmbed {
            delay_ms: 5,
            tag: 9.0,
            seen: AtomicU32::new(0),
        });
        let he = HedgedEmbeddings::new(
            primary.clone() as Arc<dyn Embeddings>,
            backup.clone() as Arc<dyn Embeddings>,
            Duration::from_millis(20),
        );
        let docs = he.embed_documents(&["a".into(), "b".into()]).await.unwrap();
        // Backup wins, returning tag=9.0.
        assert_eq!(docs[0][0], 9.0);
        assert_eq!(primary.seen.load(Ordering::SeqCst), 1);
        assert_eq!(backup.seen.load(Ordering::SeqCst), 1);
    }

}
