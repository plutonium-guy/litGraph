use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Embeddings, Message, Result};
use tracing::debug;

use crate::cassette::exchange_hash;

/// One cache entry: vector + insertion time (for TTL).
struct CachedEmbedQueryEntry {
    vec: Vec<f32>,
    inserted_at: std::time::Instant,
}

struct CachedEmbedDocumentsEntry {
    vecs: Vec<Vec<f32>>,
    inserted_at: std::time::Instant,
}

/// Wrap any [`Embeddings`] with an exact-match string-key cache.
/// Distinct from iter-253 `SingleflightEmbeddings`: that
/// coalesces *concurrent* identical calls (same text arriving at
/// the same time share one HTTP call). `CachedEmbeddings` caches
/// *across* calls — once a query is embedded, subsequent calls
/// with the exact same query string skip the upstream entirely
/// until the entry expires or is evicted.
///
/// Real prod use:
/// - Popular search queries embedded once per cache window.
/// - Repeated agent runs over the same query corpus during
///   eval / development.
/// - Pre-warming cache from a known-popular query list at
///   startup.
///
/// # Knobs
///
/// - `with_max_entries(n)` — LRU cap (per-method: query and
///   documents have separate caps but share the bound). Default
///   1000.
/// - `with_ttl(d)` — entries expire after `d`. Default `None`
///   (cache forever; useful when the embedding model is fixed).
///
/// # Composition
///
/// Stack on top of `SingleflightEmbeddings` for the full "cache
/// + dedup-concurrent" stack:
///
/// ```ignore
/// let inner = Arc::new(real_embeddings);
/// let dedup = Arc::new(SingleflightEmbeddings::new(inner));
/// let cached = Arc::new(CachedEmbeddings::new(dedup));
/// ```
///
/// Order matters: cache outside, dedup inside. Cache hits skip
/// even the dedup step; cache misses go through dedup so
/// concurrent misses still coalesce.
pub struct CachedEmbeddings {
    pub inner: Arc<dyn Embeddings>,
    pub max_entries: usize,
    pub ttl: Option<Duration>,
    query_cache: parking_lot::Mutex<Vec<(String, CachedEmbedQueryEntry)>>,
    doc_cache: parking_lot::Mutex<Vec<(Vec<String>, CachedEmbedDocumentsEntry)>>,
}

impl CachedEmbeddings {
    pub fn new(inner: Arc<dyn Embeddings>) -> Self {
        Self {
            inner,
            max_entries: 1000,
            ttl: None,
            query_cache: parking_lot::Mutex::new(Vec::new()),
            doc_cache: parking_lot::Mutex::new(Vec::new()),
        }
    }

    pub fn with_max_entries(mut self, n: usize) -> Self {
        self.max_entries = n.max(1);
        self
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Approximate query-cache size (for telemetry / tests).
    pub fn query_cache_len(&self) -> usize {
        self.query_cache.lock().len()
    }

    /// Drop all cached entries.
    pub fn clear(&self) {
        self.query_cache.lock().clear();
        self.doc_cache.lock().clear();
    }
}

#[async_trait]
impl Embeddings for CachedEmbeddings {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let now = std::time::Instant::now();
        // Lookup phase.
        {
            let mut cache = self.query_cache.lock();
            if let Some(ttl) = self.ttl {
                cache.retain(|(_, e)| now.duration_since(e.inserted_at) <= ttl);
            }
            if let Some((_, entry)) = cache.iter().find(|(k, _)| k == text) {
                return Ok(entry.vec.clone());
            }
        }
        // Miss — call inner.
        let vec = self.inner.embed_query(text).await?;
        // Insert.
        {
            let mut cache = self.query_cache.lock();
            cache.push((
                text.to_string(),
                CachedEmbedQueryEntry {
                    vec: vec.clone(),
                    inserted_at: now,
                },
            ));
            while cache.len() > self.max_entries {
                cache.remove(0);
            }
        }
        Ok(vec)
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let now = std::time::Instant::now();
        // Lookup phase.
        {
            let mut cache = self.doc_cache.lock();
            if let Some(ttl) = self.ttl {
                cache.retain(|(_, e)| now.duration_since(e.inserted_at) <= ttl);
            }
            if let Some((_, entry)) = cache.iter().find(|(k, _)| k.as_slice() == texts) {
                return Ok(entry.vecs.clone());
            }
        }
        let vecs = self.inner.embed_documents(texts).await?;
        {
            let mut cache = self.doc_cache.lock();
            cache.push((
                texts.to_vec(),
                CachedEmbedDocumentsEntry {
                    vecs: vecs.clone(),
                    inserted_at: now,
                },
            ));
            while cache.len() > self.max_entries {
                cache.remove(0);
            }
        }
        Ok(vecs)
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

    // ---- CachedEmbeddings tests ----------------------------------------

    #[tokio::test]
    async fn cached_embed_same_query_hits_cache() {
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 0,
            seen: AtomicU32::new(0),
            dim: 4,
        });
        let cached = CachedEmbeddings::new(inner.clone() as Arc<dyn Embeddings>);
        let v1 = cached.embed_query("hello").await.unwrap();
        let v2 = cached.embed_query("hello").await.unwrap();
        assert_eq!(v1, v2);
        // Inner called once; second call hit cache.
        assert_eq!(inner.seen.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cached_embed_different_query_misses_cache() {
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 0,
            seen: AtomicU32::new(0),
            dim: 3,
        });
        let cached = CachedEmbeddings::new(inner.clone() as Arc<dyn Embeddings>);
        cached.embed_query("first").await.unwrap();
        cached.embed_query("second").await.unwrap();
        cached.embed_query("third").await.unwrap();
        assert_eq!(inner.seen.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn cached_embed_documents_caches_by_full_vec() {
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 0,
            seen: AtomicU32::new(0),
            dim: 2,
        });
        let cached = CachedEmbeddings::new(inner.clone() as Arc<dyn Embeddings>);
        cached
            .embed_documents(&["a".into(), "b".into()])
            .await
            .unwrap();
        cached
            .embed_documents(&["a".into(), "b".into()])
            .await
            .unwrap();
        // Same Vec → cache hit.
        assert_eq!(inner.seen.load(Ordering::SeqCst), 1);
        // Different Vec ordering → cache miss (Vec equality is
        // ordered).
        cached
            .embed_documents(&["b".into(), "a".into()])
            .await
            .unwrap();
        assert_eq!(inner.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn cached_embed_query_and_documents_separate_caches() {
        // embed_query "x" doesn't satisfy embed_documents(["x"]).
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 0,
            seen: AtomicU32::new(0),
            dim: 2,
        });
        let cached = CachedEmbeddings::new(inner.clone() as Arc<dyn Embeddings>);
        cached.embed_query("x").await.unwrap();
        cached.embed_documents(&["x".into()]).await.unwrap();
        assert_eq!(inner.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn cached_embed_ttl_expires_old_entries() {
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 0,
            seen: AtomicU32::new(0),
            dim: 2,
        });
        let cached = CachedEmbeddings::new(inner.clone() as Arc<dyn Embeddings>)
            .with_ttl(Duration::from_millis(20));
        cached.embed_query("q").await.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        cached.embed_query("q").await.unwrap();
        assert_eq!(inner.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn cached_embed_max_entries_evicts_oldest() {
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 0,
            seen: AtomicU32::new(0),
            dim: 2,
        });
        let cached = CachedEmbeddings::new(inner.clone() as Arc<dyn Embeddings>)
            .with_max_entries(2);
        cached.embed_query("a").await.unwrap();
        cached.embed_query("b").await.unwrap();
        cached.embed_query("c").await.unwrap();
        assert_eq!(cached.query_cache_len(), 2);
        // "a" was evicted; re-querying it misses cache.
        let baseline = inner.seen.load(Ordering::SeqCst);
        cached.embed_query("a").await.unwrap();
        assert_eq!(inner.seen.load(Ordering::SeqCst), baseline + 1);
    }

    #[tokio::test]
    async fn cached_embed_clear_drops_all() {
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 0,
            seen: AtomicU32::new(0),
            dim: 2,
        });
        let cached = CachedEmbeddings::new(inner.clone() as Arc<dyn Embeddings>);
        cached.embed_query("q").await.unwrap();
        cached.clear();
        cached.embed_query("q").await.unwrap();
        assert_eq!(inner.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn cached_embed_dimensions_proxy_inner() {
        let inner = Arc::new(SfCountingEmbed {
            delay_ms: 0,
            seen: AtomicU32::new(0),
            dim: 9,
        });
        let cached = CachedEmbeddings::new(inner as Arc<dyn Embeddings>);
        assert_eq!(cached.dimensions(), 9);
        assert_eq!(cached.name(), "sf-counting-embed");
    }
}


/// Wrap any `ChatModel` to auto-set `Message.cache = true` on messages matching
/// a policy before forwarding. Anthropic (and Bedrock-on-Anthropic) providers
/// read the flag and attach `cache_control: {"type":"ephemeral"}` to the
/// message's last content block; other providers ignore it, so stacking is safe.
///
/// # Why
///
/// Anthropic's prompt cache discounts cached input tokens to ~0.1× (writes are
/// 1.25×). For agents with stable system prompts or long pinned context
/// (RAG docs, tool specs, style guides), a single cache hit can cut input
/// cost by 80–90%. The wrapper sidesteps having to flag `.cached()` at every
/// call site — declare the policy once on construction.
///
/// # Policy knobs
///
/// - `cache_system=true` (default) — mark the first System message.
/// - `cache_last_user_over=Some(N)` — mark the LAST User message if its text
///   exceeds N bytes (typical long-context-pinned-in-user pattern).
/// - `extra_indices` — manual: mark specific message indices. Overrides other
///   policies if set (advanced use).
///
/// Anthropic allows up to 4 breakpoints per request — the wrapper doesn't
/// enforce this cap (provider surfaces the error); keep policies minimal.
///
/// ```rust,ignore
/// use litgraph_resilience::PromptCachingChatModel;
/// let chat = PromptCachingChatModel::new(inner)
///     .cache_last_user_if_over(4096);  // cache system + long user
/// ```
pub struct PromptCachingChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub cache_system: bool,
    pub cache_last_user_over: Option<usize>,
    pub extra_indices: Vec<usize>,
}

impl PromptCachingChatModel {
    /// Default policy: cache the system message only. Most common pattern
    /// (stable system prompt across many turns).
    pub fn new(inner: Arc<dyn ChatModel>) -> Self {
        Self {
            inner,
            cache_system: true,
            cache_last_user_over: None,
            extra_indices: Vec::new(),
        }
    }

    /// Disable system-message caching. Use when the system prompt varies
    /// per-call and only user-side context is worth caching.
    pub fn without_system(mut self) -> Self {
        self.cache_system = false;
        self
    }

    /// Also mark the LAST User message as a cache breakpoint if its text
    /// content exceeds `bytes`. Threshold guards against caching short
    /// user turns (cache writes cost ~1.25× — pointless for small inputs).
    pub fn cache_last_user_if_over(mut self, bytes: usize) -> Self {
        self.cache_last_user_over = Some(bytes);
        self
    }

    /// Manually mark message indices as cache breakpoints. Indices out of
    /// range are silently ignored.
    pub fn cache_indices(mut self, indices: Vec<usize>) -> Self {
        self.extra_indices = indices;
        self
    }

    fn apply_policy(&self, mut messages: Vec<Message>) -> Vec<Message> {
        use litgraph_core::Role;

        if self.cache_system {
            if let Some(first_sys) = messages.iter_mut().find(|m| matches!(m.role, Role::System)) {
                first_sys.cache = true;
            }
        }

        if let Some(threshold) = self.cache_last_user_over {
            if let Some(last_user) = messages
                .iter_mut()
                .rev()
                .find(|m| matches!(m.role, Role::User))
            {
                if last_user.text_content().len() > threshold {
                    last_user.cache = true;
                }
            }
        }

        for &idx in &self.extra_indices {
            if let Some(m) = messages.get_mut(idx) {
                m.cache = true;
            }
        }

        messages
    }
}

#[async_trait]
impl ChatModel for PromptCachingChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.inner.invoke(self.apply_policy(messages), opts).await
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        self.inner.stream(self.apply_policy(messages), opts).await
    }
}

#[cfg(test)]
mod prompt_cache_tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ChatResponse, Message};
    use std::sync::Mutex;

    /// Records messages seen on invoke; returns a canned reply.
    struct SpyModel {
        seen: Mutex<Vec<Vec<Message>>>,
    }

    impl SpyModel {
        fn new() -> Arc<Self> {
            Arc::new(Self { seen: Mutex::new(Vec::new()) })
        }
    }

    #[async_trait]
    impl ChatModel for SpyModel {
        fn name(&self) -> &str { "spy" }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen.lock().unwrap().push(messages);
            Ok(ChatResponse {
                message: Message::assistant("ok"),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage { prompt: 1, completion: 1, total: 2, cache_creation: 0, cache_read: 0 },
                model: "spy".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn default_caches_system_only() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone());
        chat.invoke(
            vec![
                Message::system("you are helpful"),
                Message::user("hi"),
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(seen[0].cache, "system should be cached");
        assert!(!seen[1].cache, "user should NOT be cached");
    }

    #[tokio::test]
    async fn without_system_leaves_system_alone() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone()).without_system();
        chat.invoke(
            vec![
                Message::system("you are helpful"),
                Message::user("hi"),
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(!seen[0].cache);
        assert!(!seen[1].cache);
    }

    #[tokio::test]
    async fn cache_last_user_only_if_over_threshold() {
        let spy = SpyModel::new();
        let long_ctx = "x".repeat(5000);
        let chat = PromptCachingChatModel::new(spy.clone())
            .cache_last_user_if_over(4096);
        chat.invoke(
            vec![
                Message::system("sys"),
                Message::user("hi"),              // short — NOT cached
                Message::assistant("hello"),
                Message::user(long_ctx.clone()),  // long — cached
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(seen[0].cache);   // system (default on)
        assert!(!seen[1].cache);  // short user
        assert!(!seen[2].cache);  // assistant not touched
        assert!(seen[3].cache);   // long user
    }

    #[tokio::test]
    async fn short_user_not_cached_even_with_policy() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone())
            .cache_last_user_if_over(4096);
        chat.invoke(
            vec![
                Message::system("sys"),
                Message::user("short"),
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(seen[0].cache);   // system
        assert!(!seen[1].cache);  // user too short
    }

    #[tokio::test]
    async fn extra_indices_marks_specific_messages() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone())
            .without_system()
            .cache_indices(vec![1, 3]);
        chat.invoke(
            vec![
                Message::system("sys"),
                Message::user("first"),
                Message::assistant("a"),
                Message::user("second"),
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(!seen[0].cache);
        assert!(seen[1].cache);
        assert!(!seen[2].cache);
        assert!(seen[3].cache);
    }

    #[tokio::test]
    async fn no_system_message_still_works() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone());
        chat.invoke(
            vec![Message::user("hi")],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(!seen[0].cache);  // no system to cache; nothing crashes
    }

    #[tokio::test]
    async fn out_of_range_indices_ignored() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone())
            .without_system()
            .cache_indices(vec![99, 100]);
        chat.invoke(
            vec![Message::user("hi")],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(!seen[0].cache);
    }

    #[tokio::test]
    async fn caches_first_system_only_when_multiple() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone());
        chat.invoke(
            vec![
                Message::system("first sys"),
                Message::system("second sys"),
                Message::user("hi"),
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(seen[0].cache);
        assert!(!seen[1].cache, "only FIRST system marked");
    }

    #[tokio::test]
    async fn preserves_existing_cache_flag() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone()).without_system();
        let pre_cached = Message::user("big ctx").cached();
        chat.invoke(
            vec![pre_cached],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(seen[0].cache, "policy never clears existing cache flags");
    }

    #[tokio::test]
    async fn name_delegates_to_inner() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy);
        assert_eq!(chat.name(), "spy");
    }
}



/// Cache entry for [`CachedChatModel`].
#[derive(Clone)]
struct CachedChatEntry {
    response: ChatResponse,
    inserted_at: std::time::Instant,
}

/// Wrap any [`ChatModel`] with an in-memory response cache keyed
/// by canonical hash of `(messages, ChatOptions)`.
///
/// # Distinct from neighboring primitives
///
/// - **`RecordingChatModel` / `ReplayingChatModel`** (iter 254) are
///   for *test workflows* — record once, replay forever in CI.
///   `CachedChatModel` is for *production* — cache hits skip the
///   provider call to save tokens / latency, with an LRU cap and
///   optional TTL so the cache doesn't grow unbounded.
/// - **`PromptCachingChatModel`** (iter 136) controls Anthropic's
///   *server-side* prompt cache via cache_control headers — that
///   reduces input-token cost on the provider side. This caches
///   the entire response *client-side* — zero provider call on hit.
///   The two compose: even with prompt caching enabled upstream,
///   client-side caching is still a win because identical requests
///   skip the network roundtrip entirely.
/// - **`SingleflightChatModel`** doesn't exist (we have it for
///   tools/embeddings/retrievers). If we add it, it'd coalesce
///   *concurrent* identical calls; this caches *across* calls.
///   The two compose: cache-outside / dedup-inside.
///
/// # When this is a win
///
/// - Eval harnesses that re-run the same dataset against the same
///   model multiple times (parameter sweeps, scorer iteration).
/// - Agents over a fixed FAQ where users phrase the same question
///   identically (verbatim — for fuzzy matches use a semantic
///   layer; this is exact-key only).
/// - Demo / dev environments where the same prompt is replayed
///   while iterating on downstream UI / parsing logic.
/// - Multi-stage agent loops where the same upstream call gets
///   re-invoked due to retry / control-flow restart — cache short-
///   circuits the redundant work.
///
/// # When this is NOT a win
///
/// - Stochastic-output workflows (high temperature, creative
///   generation) — cache-hit rate is near zero because each call
///   has slightly different opts metadata or message wording.
/// - Long-tail prompts where each user query is unique. The cache
///   just adds memory pressure without earning hits.
///
/// # Streaming
///
/// `stream()` is NOT cached — token streams can't be replayed in
/// a useful way without preserving inter-chunk timing, and the
/// caller of `stream()` typically wants the streaming UX (otherwise
/// they'd call `invoke()`). Stream calls pass through to inner.
/// Tests verifying cache behavior should use `invoke()`.
///
/// # Hash determinism
///
/// Cache key uses [`exchange_hash`], the same blake3-over-canonical-
/// JSON function as the cassette infrastructure. Reusing it ensures
/// the cache and the cassette agree on what counts as "the same
/// request" — same hash function, same canonicalization rules.
pub struct CachedChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub max_entries: usize,
    pub ttl: Option<Duration>,
    cache: parking_lot::Mutex<Vec<(String, CachedChatEntry)>>,
}

impl CachedChatModel {
    pub fn new(inner: Arc<dyn ChatModel>) -> Self {
        Self {
            inner,
            max_entries: 1000,
            ttl: None,
            cache: parking_lot::Mutex::new(Vec::new()),
        }
    }

    pub fn with_max_entries(mut self, n: usize) -> Self {
        self.max_entries = n.max(1);
        self
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Approximate cache size — for telemetry / tests.
    pub fn cache_len(&self) -> usize {
        self.cache.lock().len()
    }

    /// Drop all cached entries.
    pub fn clear(&self) {
        self.cache.lock().clear();
    }
}

#[async_trait]
impl ChatModel for CachedChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let key = exchange_hash(&messages, opts);
        let now = std::time::Instant::now();
        // Lookup phase. Sweep TTL-expired entries while holding lock.
        {
            let mut cache = self.cache.lock();
            if let Some(ttl) = self.ttl {
                cache.retain(|(_, e)| now.duration_since(e.inserted_at) <= ttl);
            }
            if let Some((_, entry)) = cache.iter().find(|(k, _)| k == &key) {
                debug!(target: "litgraph_resilience::cached_chat", "cache hit");
                return Ok(entry.response.clone());
            }
        }
        // Miss — call inner.
        let response = self.inner.invoke(messages, opts).await?;
        // Insert + evict.
        {
            let mut cache = self.cache.lock();
            cache.push((
                key,
                CachedChatEntry {
                    response: response.clone(),
                    inserted_at: now,
                },
            ));
            while cache.len() > self.max_entries {
                cache.remove(0);
            }
        }
        Ok(response)
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // See module doc — streams pass through uncached.
        self.inner.stream(messages, opts).await
    }
}

#[cfg(test)]
mod cached_chat_tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ChatResponse, Message};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Records call count + returns "resp-N" where N is the call index.
    struct CountingModel {
        calls: AtomicUsize,
    }

    impl CountingModel {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
            })
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ChatModel for CountingModel {
        fn name(&self) -> &str {
            "counting"
        }
        async fn invoke(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ChatResponse {
                message: Message::assistant(format!("resp-{n}")),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage {
                    prompt: 10,
                    completion: 5,
                    total: 15,
                    cache_creation: 0,
                    cache_read: 0,
                },
                model: "counting".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn identical_calls_hit_cache() {
        let inner = CountingModel::new();
        let cached = CachedChatModel::new(inner.clone());
        let opts = ChatOptions::default();
        let msgs = vec![Message::user("hello")];

        let r1 = cached.invoke(msgs.clone(), &opts).await.unwrap();
        let r2 = cached.invoke(msgs.clone(), &opts).await.unwrap();
        let r3 = cached.invoke(msgs, &opts).await.unwrap();

        // Inner saw exactly one call; subsequent two were cache hits.
        assert_eq!(inner.calls(), 1);
        // All three responses are byte-identical (the cached value).
        assert_eq!(r1.message.text_content(), "resp-0");
        assert_eq!(r2.message.text_content(), "resp-0");
        assert_eq!(r3.message.text_content(), "resp-0");
        assert_eq!(cached.cache_len(), 1);
    }

    #[tokio::test]
    async fn different_messages_miss_cache() {
        let inner = CountingModel::new();
        let cached = CachedChatModel::new(inner.clone());
        let opts = ChatOptions::default();

        cached
            .invoke(vec![Message::user("a")], &opts)
            .await
            .unwrap();
        cached
            .invoke(vec![Message::user("b")], &opts)
            .await
            .unwrap();
        cached
            .invoke(vec![Message::user("c")], &opts)
            .await
            .unwrap();

        assert_eq!(inner.calls(), 3);
        assert_eq!(cached.cache_len(), 3);
    }

    #[tokio::test]
    async fn different_opts_miss_cache() {
        let inner = CountingModel::new();
        let cached = CachedChatModel::new(inner.clone());
        let msgs = vec![Message::user("same")];

        let mut o1 = ChatOptions::default();
        o1.temperature = Some(0.0);
        let mut o2 = ChatOptions::default();
        o2.temperature = Some(0.7);

        cached.invoke(msgs.clone(), &o1).await.unwrap();
        cached.invoke(msgs.clone(), &o2).await.unwrap();
        // Same opts as o1 — cache hit.
        cached.invoke(msgs, &o1).await.unwrap();

        // Two distinct opt fingerprints → two inner calls; the third was a hit.
        assert_eq!(inner.calls(), 2);
    }

    #[tokio::test]
    async fn lru_eviction_caps_size() {
        let inner = CountingModel::new();
        let cached = CachedChatModel::new(inner.clone()).with_max_entries(2);
        let opts = ChatOptions::default();

        cached
            .invoke(vec![Message::user("a")], &opts)
            .await
            .unwrap();
        cached
            .invoke(vec![Message::user("b")], &opts)
            .await
            .unwrap();
        cached
            .invoke(vec![Message::user("c")], &opts)
            .await
            .unwrap();

        // Only the two most-recent entries remain; "a" was evicted.
        assert_eq!(cached.cache_len(), 2);
        // Now re-invoking "a" misses (it was evicted) → 4th inner call.
        cached
            .invoke(vec![Message::user("a")], &opts)
            .await
            .unwrap();
        assert_eq!(inner.calls(), 4);
    }

    #[tokio::test]
    async fn ttl_expires_entries() {
        let inner = CountingModel::new();
        let cached = CachedChatModel::new(inner.clone()).with_ttl(Duration::from_millis(50));
        let opts = ChatOptions::default();
        let msgs = vec![Message::user("ephemeral")];

        cached.invoke(msgs.clone(), &opts).await.unwrap();
        // Within TTL — cache hit.
        cached.invoke(msgs.clone(), &opts).await.unwrap();
        assert_eq!(inner.calls(), 1);

        tokio::time::sleep(Duration::from_millis(80)).await;
        // After TTL — entry expired, re-invokes inner.
        cached.invoke(msgs, &opts).await.unwrap();
        assert_eq!(inner.calls(), 2);
    }

    #[tokio::test]
    async fn clear_drops_all_entries() {
        let inner = CountingModel::new();
        let cached = CachedChatModel::new(inner.clone());
        let opts = ChatOptions::default();

        cached
            .invoke(vec![Message::user("x")], &opts)
            .await
            .unwrap();
        cached
            .invoke(vec![Message::user("y")], &opts)
            .await
            .unwrap();
        assert_eq!(cached.cache_len(), 2);

        cached.clear();
        assert_eq!(cached.cache_len(), 0);

        // Post-clear: previously-cached call is now a miss.
        cached
            .invoke(vec![Message::user("x")], &opts)
            .await
            .unwrap();
        assert_eq!(inner.calls(), 3);
    }

    #[tokio::test]
    async fn name_delegates_to_inner() {
        let inner = CountingModel::new();
        let cached = CachedChatModel::new(inner);
        assert_eq!(cached.name(), "counting");
    }

    #[tokio::test]
    async fn with_max_entries_zero_clamps_to_one() {
        let inner = CountingModel::new();
        let cached = CachedChatModel::new(inner).with_max_entries(0);
        assert_eq!(cached.max_entries, 1);
    }
}
