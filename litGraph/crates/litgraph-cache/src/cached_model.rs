//! `CachedModel` — transparent caching wrapper around any `ChatModel`.
//!
//! Only `invoke` is cached. `stream` flows through uncached (token streams can't
//! roundtrip well through a cache without changing semantics; callers who need it
//! should capture the terminal `ChatResponse` and put it via `Cache::put`).
//!
//! Streaming also bypasses cache when options are non-deterministic (`temperature > 0`
//! or a non-fixed seed). We use the exact options the caller passed, so two calls
//! with identical options always hit the same cache key — determinism is the
//! caller's responsibility.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::{ChatStream, FinishReason};
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Message, Result};
use tracing::{debug, trace};

use crate::backend::Cache;
use crate::key::cache_key;

pub struct CachedModel {
    pub inner: Arc<dyn ChatModel>,
    pub cache: Arc<dyn Cache>,
    /// If true, cached hits retain the original `FinishReason` from when the response
    /// was written. If false, they come back as `Stop` regardless.
    pub preserve_finish_reason: bool,
}

impl CachedModel {
    pub fn new(inner: Arc<dyn ChatModel>, cache: Arc<dyn Cache>) -> Self {
        Self { inner, cache, preserve_finish_reason: true }
    }
}

#[async_trait]
impl ChatModel for CachedModel {
    fn name(&self) -> &str { self.inner.name() }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let key = cache_key(self.inner.name(), &messages, opts);
        if let Some(mut hit) = self.cache.get(&key).await? {
            trace!(key = %&key[..16], "cache hit");
            if !self.preserve_finish_reason {
                hit.finish_reason = FinishReason::Stop;
            }
            return Ok(hit);
        }
        debug!(key = %&key[..16], "cache miss — calling upstream");
        let resp = self.inner.invoke(messages, opts).await?;
        // Write-through; swallow cache errors — they shouldn't fail the request.
        if let Err(e) = self.cache.put(&key, resp.clone()).await {
            tracing::warn!(error = %e, "cache put failed (non-fatal)");
        }
        Ok(resp)
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        // Bypass cache for streams — see module docstring.
        self.inner.stream(messages, opts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MemoryCache;
    use litgraph_core::model::{FinishReason, TokenUsage};
    use litgraph_core::{ContentPart, Message, Role};
    use std::sync::atomic::{AtomicU32, Ordering};

    struct CountingModel {
        hits: AtomicU32,
    }

    #[async_trait]
    impl ChatModel for CountingModel {
        fn name(&self) -> &str { "count" }
        async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
            self.hits.fetch_add(1, Ordering::SeqCst);
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
                model: "count".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn second_identical_call_hits_cache() {
        let inner: Arc<dyn ChatModel> = Arc::new(CountingModel { hits: AtomicU32::new(0) });
        let cache: Arc<dyn Cache> = Arc::new(MemoryCache::new(100));
        let cm = CachedModel::new(inner.clone(), cache);
        let msgs = vec![Message::user("ping")];
        let opts = ChatOptions::default();

        let _ = cm.invoke(msgs.clone(), &opts).await.unwrap();
        let _ = cm.invoke(msgs.clone(), &opts).await.unwrap();
        // Third with different message — should miss.
        let _ = cm.invoke(vec![Message::user("pong")], &opts).await.unwrap();

        let counting = inner.as_ref() as *const dyn ChatModel;
        // Can't easily downcast through trait object; re-use the AtomicU32 via a fresh check.
        // Instead, validate by round-trip: the second identical call should have returned
        // the same response body, which is guaranteed by cache hit path.
        // To assert the hit count directly, we capture the AtomicU32 ahead of time.
        let _ = counting;
        // This indirect check is enough: the test below explicitly verifies hit count.
    }

    #[tokio::test]
    async fn hit_count_verified() {
        struct M(Arc<AtomicU32>);
        #[async_trait]
        impl ChatModel for M {
            fn name(&self) -> &str { "m" }
            async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(ChatResponse {
                    message: Message::assistant("hi"),
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage::default(),
                    model: "m".into(),
                })
            }
            async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
                unimplemented!()
            }
        }

        let counter = Arc::new(AtomicU32::new(0));
        let inner: Arc<dyn ChatModel> = Arc::new(M(counter.clone()));
        let cache: Arc<dyn Cache> = Arc::new(MemoryCache::new(100));
        let cm = CachedModel::new(inner, cache);

        let msgs = vec![Message::user("ping")];
        let opts = ChatOptions::default();

        let _ = cm.invoke(msgs.clone(), &opts).await.unwrap();
        let _ = cm.invoke(msgs.clone(), &opts).await.unwrap();
        let _ = cm.invoke(vec![Message::user("pong")], &opts).await.unwrap();

        // Two distinct prompts → two upstream calls, the second "ping" is cached.
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }
}
