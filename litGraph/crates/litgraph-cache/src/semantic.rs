//! Semantic cache — look up responses by *embedding similarity* instead of
//! exact-prompt hash. When a new prompt has cosine similarity ≥ `threshold` to a
//! past prompt, the past response is returned (saving an LLM call).
//!
//! Sound usage:
//! - Best for FAQ-like workloads where prompts rephrase the same intent.
//! - Avoid for tool-calling where arg precision matters.
//! - Always set a conservative threshold (≥ 0.95 for production).
//!
//! Implementation lives on top of any `VectorStore`-like lookup: we use a small
//! inline in-memory store + rayon-parallel cosine, because the cache set is
//! usually orders of magnitude smaller than your RAG corpus.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Embeddings, Message, Result};
use parking_lot::RwLock;
use rayon::prelude::*;
use tracing::{debug, warn};

struct Entry {
    embedding: Vec<f32>,
    response: ChatResponse,
}

pub struct SemanticCache {
    embeddings: Arc<dyn Embeddings>,
    threshold: f32,
    max_entries: usize,
    entries: RwLock<Vec<Entry>>,
}

impl SemanticCache {
    pub fn new(embeddings: Arc<dyn Embeddings>, threshold: f32) -> Self {
        Self {
            embeddings,
            threshold: threshold.clamp(0.0, 1.0),
            max_entries: 10_000,
            entries: RwLock::new(Vec::new()),
        }
    }

    pub fn with_max_entries(mut self, n: usize) -> Self { self.max_entries = n; self }

    pub fn len(&self) -> usize { self.entries.read().len() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn clear(&self) { self.entries.write().clear(); }

    pub async fn lookup(&self, prompt: &str) -> Result<Option<ChatResponse>> {
        let snap = {
            // Cheap clone of f32 vecs — acceptable; keeps the lock uncontended across async awaits.
            self.entries.read().iter()
                .map(|e| (e.embedding.clone(), e.response.clone()))
                .collect::<Vec<_>>()
        };
        if snap.is_empty() { return Ok(None); }
        let q = self.embeddings.embed_query(prompt).await?;
        let q_norm = norm(&q);
        let best = snap
            .par_iter()
            .map(|(e, resp)| (cosine(&q, e, q_norm), resp))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        match best {
            Some((sim, resp)) if sim >= self.threshold => {
                debug!(sim, threshold = self.threshold, "semantic cache hit");
                Ok(Some(resp.clone()))
            }
            _ => Ok(None),
        }
    }

    pub async fn insert(&self, prompt: &str, response: ChatResponse) -> Result<()> {
        let emb = self.embeddings.embed_query(prompt).await?;
        let mut g = self.entries.write();
        // Simple FIFO eviction — good enough; swap for LRU if memory pressure becomes a concern.
        if g.len() >= self.max_entries { g.remove(0); }
        let _ = prompt; // retained for potential debug logging
        g.push(Entry { embedding: emb, response });
        Ok(())
    }
}

fn norm(v: &[f32]) -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() }

fn cosine(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0f32;
    let mut bn = 0f32;
    for i in 0..len {
        dot += a[i] * b[i];
        bn += b[i] * b[i];
    }
    let bn = bn.sqrt();
    if a_norm == 0.0 || bn == 0.0 { return 0.0; }
    dot / (a_norm * bn)
}

/// Wraps a `ChatModel` with a `SemanticCache`. Cache key is the concatenated
/// text of the last user message — good enough for chat QA. Multi-turn
/// caching is the caller's responsibility (they'd key on something richer).
pub struct SemanticCachedModel {
    pub inner: Arc<dyn ChatModel>,
    pub cache: Arc<SemanticCache>,
}

impl SemanticCachedModel {
    pub fn new(inner: Arc<dyn ChatModel>, cache: Arc<SemanticCache>) -> Self {
        Self { inner, cache }
    }
}

#[async_trait]
impl ChatModel for SemanticCachedModel {
    fn name(&self) -> &str { self.inner.name() }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let Some(prompt) = last_user_text(&messages) else {
            return self.inner.invoke(messages, opts).await;
        };
        if let Some(hit) = self.cache.lookup(&prompt).await? {
            return Ok(hit);
        }
        let resp = self.inner.invoke(messages, opts).await?;
        if let Err(e) = self.cache.insert(&prompt, resp.clone()).await {
            warn!(error = %e, "semantic cache insert failed (non-fatal)");
        }
        Ok(resp)
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        // Bypass cache for streams — see CachedModel docstring for rationale.
        self.inner.stream(messages, opts).await
    }
}

fn last_user_text(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| matches!(m.role, litgraph_core::Role::User))
        .map(|m| m.text_content())
}

#[cfg(test)]
mod tests {
    use super::*;
    use litgraph_core::model::{FinishReason, TokenUsage};
    use litgraph_core::{ContentPart, Message, Role};
    use std::sync::atomic::{AtomicU32, Ordering};

    struct FakeEmbed;
    #[async_trait]
    impl Embeddings for FakeEmbed {
        fn name(&self) -> &str { "fake" }
        fn dimensions(&self) -> usize { 3 }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            Ok(text_vec(text))
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| text_vec(t)).collect())
        }
    }
    fn text_vec(t: &str) -> Vec<f32> {
        let l = t.to_lowercase();
        vec![
            if l.contains("cat") { 1.0 } else { 0.0 },
            if l.contains("dog") { 1.0 } else { 0.0 },
            if l.contains("car") { 1.0 } else { 0.0 },
        ]
    }

    struct M(Arc<AtomicU32>);
    #[async_trait]
    impl ChatModel for M {
        fn name(&self) -> &str { "m" }
        async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text: "cats are fine".into() }],
                    tool_calls: vec![], tool_call_id: None, name: None, cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "m".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn similar_prompt_hits_cache() {
        let counter = Arc::new(AtomicU32::new(0));
        let inner: Arc<dyn ChatModel> = Arc::new(M(counter.clone()));
        let cache = Arc::new(SemanticCache::new(Arc::new(FakeEmbed), 0.90));
        let cm = SemanticCachedModel::new(inner, cache);

        // First call populates cache.
        cm.invoke(vec![Message::user("I like cats")], &ChatOptions::default()).await.unwrap();
        // Identical semantic signature (contains "cat") → cache hit.
        cm.invoke(vec![Message::user("do you like cats too")], &ChatOptions::default()).await.unwrap();
        // Different semantic signature ("car") → cache miss.
        cm.invoke(vec![Message::user("fast cars go brr")], &ChatOptions::default()).await.unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }
}
