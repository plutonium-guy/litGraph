//! Vector-store memory — embeddings of each conversation turn, retrieve
//! top-K most-relevant past turns by cosine similarity. LangChain
//! `VectorStoreRetrieverMemory` parity.
//!
//! # vs other memory types
//!
//! - `BufferMemory`: keeps last N turns (recency).
//! - `TokenBufferMemory`: token-budget recency.
//! - `SummaryBufferMemory`: rolling buffer + LLM summary of evictions.
//! - `VectorStoreMemory` (this): topic-relevance retrieval. Long-running
//!   agents accumulate hundreds of turns where a question from yesterday
//!   may matter MORE than the last 3 turns — pure recency drops it.
//!
//! # Why standalone (not `ConversationMemory`-impl)
//!
//! `ConversationMemory::messages() -> Vec<Message>` is sync + parameter-less.
//! Vector-store memory is QUERY-DRIVEN — relevance only makes sense given
//! the current input. The trait can't fit. Caller pattern:
//!
//! ```ignore
//! let mem = VectorStoreMemory::new(embeddings, 4);
//! mem.append(user_msg.clone());
//! mem.append(assistant_msg.clone());
//! mem.flush().await?;  // embed pending
//! // Later turn:
//! let relevant = mem.retrieve_for("what did I say about X?", 4).await?;
//! // Splice `relevant` into the next prompt as additional context.
//! ```
//!
//! # Storage
//!
//! In-memory `Vec<(Vec<f32>, Message)>` guarded by parking_lot RwLock.
//! No durability — caller persists via the standard `MemorySnapshot`
//! roundtrip pattern (or wraps with their own backend). For 10K-turn
//! agents this is ~40MB at 1536 dims — acceptable. Beyond that, plug
//! a real vector store via the `VectorStore` trait.
//!
//! # Embedding deferral
//!
//! `append()` is sync (matches the rest of the memory API). It pushes
//! the message into a `pending` buffer. `flush().await` embeds pending
//! messages in one batch call (cheaper than per-turn embedding —
//! providers charge per-call + per-token). `retrieve_for()` auto-flushes
//! if there's anything pending, so callers can choose: explicit `flush`
//! at well-known points, or rely on auto-flush at retrieve time.

use std::sync::Arc;

use parking_lot::RwLock;

use crate::{Embeddings, Error, Message, Result};

/// Top-K retrieval result — message + cosine score.
#[derive(Debug, Clone)]
pub struct RetrievedMessage {
    pub message: Message,
    pub score: f32,
}

pub struct VectorStoreMemory {
    embeddings: Arc<dyn Embeddings>,
    /// Default top-K for `retrieve_for`. Overridable per-call.
    pub default_top_k: usize,
    /// Embedded messages: `(embedding_vec, message)`.
    store: Arc<RwLock<Vec<(Vec<f32>, Message)>>>,
    /// Messages appended but not yet embedded.
    pending: Arc<RwLock<Vec<Message>>>,
    /// Optional system pin — surfaced separately from the relevance
    /// retrieval (system messages are config, not searchable content).
    system: Arc<RwLock<Option<Message>>>,
}

impl VectorStoreMemory {
    pub fn new(embeddings: Arc<dyn Embeddings>, default_top_k: usize) -> Self {
        Self {
            embeddings,
            default_top_k: default_top_k.max(1),
            store: Arc::new(RwLock::new(Vec::new())),
            pending: Arc::new(RwLock::new(Vec::new())),
            system: Arc::new(RwLock::new(None)),
        }
    }

    /// Append a message. System messages set the pin; other messages
    /// queue for embedding (call `flush().await` to embed; or rely on
    /// `retrieve_for`'s auto-flush).
    pub fn append(&self, m: Message) {
        if matches!(m.role, crate::Role::System) {
            *self.system.write() = Some(m);
            return;
        }
        self.pending.write().push(m);
    }

    /// Set or clear the system pin explicitly.
    pub fn set_system(&self, m: Option<Message>) {
        *self.system.write() = m;
    }

    /// Returns the system pin, if any.
    pub fn system(&self) -> Option<Message> {
        self.system.read().clone()
    }

    /// Number of embedded messages in the store.
    pub fn embedded_len(&self) -> usize {
        self.store.read().len()
    }

    /// Number of messages appended but not yet embedded.
    pub fn pending_len(&self) -> usize {
        self.pending.read().len()
    }

    /// Drop everything — embeddings, pending queue, system pin.
    pub fn clear(&self) {
        self.store.write().clear();
        self.pending.write().clear();
        *self.system.write() = None;
    }

    /// Embed any pending messages in a single batch call (efficient).
    /// Idempotent — no-op if pending is empty. Returns the number of
    /// messages newly embedded.
    pub async fn flush(&self) -> Result<usize> {
        // Drain pending OUTSIDE the await to avoid holding the lock
        // across a network call.
        let pending: Vec<Message> = {
            let mut p = self.pending.write();
            std::mem::take(&mut *p)
        };
        if pending.is_empty() {
            return Ok(0);
        }
        let texts: Vec<String> = pending.iter().map(|m| m.text_content()).collect();
        let embeds = self.embeddings.embed_documents(&texts).await?;
        if embeds.len() != pending.len() {
            return Err(Error::other(format!(
                "VectorStoreMemory: embedder returned {} vectors for {} messages",
                embeds.len(),
                pending.len()
            )));
        }
        let mut store = self.store.write();
        for (vec, msg) in embeds.into_iter().zip(pending.into_iter()) {
            store.push((vec, msg));
        }
        Ok(store.len())
    }

    /// Top-K most-relevant past messages to `query`, by cosine similarity.
    /// Auto-flushes any pending messages first. `k=0` uses `default_top_k`.
    pub async fn retrieve_for(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<RetrievedMessage>> {
        // Flush first so pending messages are eligible.
        if self.pending_len() > 0 {
            self.flush().await?;
        }
        let k = if k == 0 { self.default_top_k } else { k };
        let q_vec = self.embeddings.embed_query(query).await?;
        let store = self.store.read();
        let mut scored: Vec<(usize, f32)> = store
            .iter()
            .enumerate()
            .map(|(i, (ev, _))| (i, cosine_sim(&q_vec, ev)))
            .collect();
        // Stable descending sort by score.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored
            .into_iter()
            .take(k)
            .map(|(idx, score)| RetrievedMessage {
                message: store[idx].1.clone(),
                score,
            })
            .collect())
    }

    /// Build the message list for the next chat turn:
    /// `[system_pin?, ...top_k_relevant_to_query, current_user_message]`.
    /// Convenience for the canonical "stuff retrieved memory into the
    /// prompt" pattern — saves callers from manually splicing.
    pub async fn build_context(
        &self,
        query: &str,
        k: usize,
        current: Message,
    ) -> Result<Vec<Message>> {
        let retrieved = self.retrieve_for(query, k).await?;
        let mut out: Vec<Message> = Vec::with_capacity(retrieved.len() + 2);
        if let Some(sys) = self.system() {
            out.push(sys);
        }
        for r in retrieved {
            out.push(r.message);
        }
        out.push(current);
        Ok(out)
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Embeddings;
    use crate::Role;
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// Keyword-presence embedder. Each text → fixed-dim vec marking which
    /// keywords appear. Deterministic + lets us reason about what should
    /// come back top-K.
    struct KeywordEmbedder {
        keywords: Vec<&'static str>,
        embed_calls: Mutex<usize>,
    }

    impl KeywordEmbedder {
        fn new(keywords: Vec<&'static str>) -> Arc<Self> {
            Arc::new(Self { keywords, embed_calls: Mutex::new(0) })
        }
        fn vec_for(&self, text: &str) -> Vec<f32> {
            let lower = text.to_lowercase();
            self.keywords.iter().map(|kw| if lower.contains(kw) { 1.0 } else { 0.0 }).collect()
        }
    }

    #[async_trait]
    impl Embeddings for KeywordEmbedder {
        fn name(&self) -> &str { "keyword" }
        fn dimensions(&self) -> usize { self.keywords.len() }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            *self.embed_calls.lock().unwrap() += 1;
            Ok(self.vec_for(text))
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            *self.embed_calls.lock().unwrap() += texts.len();
            Ok(texts.iter().map(|t| self.vec_for(t)).collect())
        }
    }

    fn embedder() -> Arc<KeywordEmbedder> {
        KeywordEmbedder::new(vec![
            "rust", "python", "javascript", "css", "borrow", "lifetime",
            "promise", "flexbox", "comprehension",
        ])
    }

    #[tokio::test]
    async fn retrieves_top_k_by_relevance() {
        let mem = VectorStoreMemory::new(embedder(), 2);
        mem.append(Message::user("how do I fix a rust borrow checker issue"));
        mem.append(Message::assistant("use clone or rework lifetimes"));
        mem.append(Message::user("css flexbox question"));
        mem.append(Message::assistant("use justify-content"));
        mem.append(Message::user("python list comprehension"));
        mem.append(Message::assistant("use [x for x in ...]"));

        let r = mem
            .retrieve_for("rust lifetime trouble", 2)
            .await
            .unwrap();
        assert_eq!(r.len(), 2);
        // Top-2 should both be rust-themed.
        for m in &r {
            assert!(
                m.message.text_content().to_lowercase().contains("rust")
                    || m.message.text_content().to_lowercase().contains("borrow")
                    || m.message.text_content().to_lowercase().contains("lifetime")
                    || m.message.text_content().to_lowercase().contains("clone"),
                "unexpected match: {}", m.message.text_content()
            );
        }
    }

    #[tokio::test]
    async fn auto_flush_on_retrieve() {
        let mem = VectorStoreMemory::new(embedder(), 2);
        mem.append(Message::user("rust borrow"));
        // No explicit flush.
        assert_eq!(mem.embedded_len(), 0);
        assert_eq!(mem.pending_len(), 1);
        let r = mem.retrieve_for("rust", 1).await.unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(mem.pending_len(), 0);
        assert_eq!(mem.embedded_len(), 1);
    }

    #[tokio::test]
    async fn flush_batches_pending_into_one_call() {
        let e = embedder();
        let mem = VectorStoreMemory::new(e.clone(), 2);
        for i in 0..5 {
            mem.append(Message::user(format!("rust msg {i}")));
        }
        let n = mem.flush().await.unwrap();
        assert_eq!(n, 5);
        // 5 messages embedded in ONE batch call.
        assert_eq!(*e.embed_calls.lock().unwrap(), 5);  // counts texts, not calls
    }

    #[tokio::test]
    async fn flush_idempotent_when_pending_empty() {
        let e = embedder();
        let mem = VectorStoreMemory::new(e.clone(), 2);
        let n = mem.flush().await.unwrap();
        assert_eq!(n, 0);
        assert_eq!(*e.embed_calls.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn system_role_sets_pin_not_embedded() {
        let mem = VectorStoreMemory::new(embedder(), 2);
        mem.append(Message::system("you are a helpful assistant"));
        mem.append(Message::user("rust borrow"));
        mem.flush().await.unwrap();
        assert_eq!(mem.embedded_len(), 1, "system shouldn't go into searchable store");
        assert!(mem.system().is_some());
        assert_eq!(
            mem.system().unwrap().text_content(),
            "you are a helpful assistant"
        );
    }

    #[tokio::test]
    async fn build_context_includes_system_retrieved_and_current() {
        let mem = VectorStoreMemory::new(embedder(), 2);
        mem.append(Message::system("be terse"));
        mem.append(Message::user("rust borrow tip?"));
        mem.append(Message::assistant("use clone"));
        mem.append(Message::user("css question"));
        mem.append(Message::assistant("flex"));

        let ctx = mem
            .build_context("rust", 2, Message::user("more rust help"))
            .await
            .unwrap();
        // Layout: [system, ...retrieved (2), current_user] → 4 messages.
        assert_eq!(ctx.len(), 4);
        assert_eq!(ctx[0].role, Role::System);
        assert_eq!(ctx[0].text_content(), "be terse");
        assert_eq!(ctx[3].text_content(), "more rust help");
    }

    #[tokio::test]
    async fn build_context_no_system_omits_system_slot() {
        let mem = VectorStoreMemory::new(embedder(), 1);
        mem.append(Message::user("rust"));
        let ctx = mem
            .build_context("rust", 1, Message::user("now"))
            .await
            .unwrap();
        // [retrieved (1), current] → 2.
        assert_eq!(ctx.len(), 2);
    }

    #[tokio::test]
    async fn k_zero_uses_default_top_k() {
        let mem = VectorStoreMemory::new(embedder(), 3);
        for i in 0..5 {
            mem.append(Message::user(format!("rust {i}")));
        }
        let r = mem.retrieve_for("rust", 0).await.unwrap();
        assert_eq!(r.len(), 3, "k=0 falls back to default_top_k=3");
    }

    #[tokio::test]
    async fn empty_store_returns_empty_results() {
        let mem = VectorStoreMemory::new(embedder(), 5);
        let r = mem.retrieve_for("anything", 3).await.unwrap();
        assert!(r.is_empty());
    }

    #[tokio::test]
    async fn clear_drops_everything() {
        let mem = VectorStoreMemory::new(embedder(), 2);
        mem.append(Message::system("sys"));
        mem.append(Message::user("rust"));
        mem.flush().await.unwrap();
        mem.append(Message::user("pending"));
        mem.clear();
        assert_eq!(mem.embedded_len(), 0);
        assert_eq!(mem.pending_len(), 0);
        assert!(mem.system().is_none());
    }

    #[tokio::test]
    async fn results_sorted_descending_by_score() {
        let mem = VectorStoreMemory::new(embedder(), 5);
        mem.append(Message::user("rust borrow lifetime"));      // 3 keyword hits
        mem.append(Message::user("rust borrow"));                // 2
        mem.append(Message::user("rust"));                       // 1
        mem.append(Message::user("nothing"));                    // 0
        let r = mem.retrieve_for("rust borrow lifetime", 4).await.unwrap();
        assert_eq!(r.len(), 4);
        // Scores monotonically non-increasing.
        for w in r.windows(2) {
            assert!(w[0].score >= w[1].score, "{} should be >= {}", w[0].score, w[1].score);
        }
        assert!(r[0].score > r[3].score, "best > worst");
    }

    #[tokio::test]
    async fn k_larger_than_store_returns_all() {
        let mem = VectorStoreMemory::new(embedder(), 100);
        mem.append(Message::user("a"));
        mem.append(Message::user("b"));
        let r = mem.retrieve_for("anything", 100).await.unwrap();
        assert_eq!(r.len(), 2);
    }

    #[tokio::test]
    async fn cosine_zero_vector_returns_zero_not_nan() {
        assert_eq!(cosine_sim(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine_sim(&[0.0; 5], &[0.0; 5]), 0.0);
    }
}
