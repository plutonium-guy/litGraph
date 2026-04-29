//! Episodic memory SDK — auto-extract memorable facts from conversations
//! and recall them in future turns. LangMem parity, but built on the
//! existing [`Store`] trait so any backend (in-memory, Postgres, Redis)
//! works without a custom integration.
//!
//! # The pattern
//!
//! ```text
//! conversation messages
//!         │
//!         ▼
//!   MemoryExtractor (LLM + structured output)
//!         │
//!         ▼
//!     Vec<Memory>
//!         │
//!         ▼
//!   EpisodicMemory.observe()
//!         │
//!         ▼
//!     Store::put (namespaced per user/agent)
//!
//!  ── later ──
//!
//!   EpisodicMemory.recall(query, k)
//!         │
//!         ▼
//!   Store::search (query_text-driven)
//!         │
//!         ▼
//!     Vec<Memory>  → prepend to next prompt
//! ```
//!
//! # Why this lives in core (not its own crate)
//!
//! - It's pure glue around `ChatModel` + `Store` + `StructuredChatModel`,
//!   which all live here.
//! - No new dependencies — the heavy lifting (vector search, persistence)
//!   is delegated to whatever `Store` impl the caller plugs in.
//!
//! # Production checklist
//!
//! - **Backend**: pair with a `Store` that supports `query_text` for
//!   semantic recall. The bundled `InMemoryStore` does prefix-substring
//!   matching only — fine for tests. For prod, plug in
//!   `litgraph-stores-*` (Postgres/Chroma/Weaviate) or roll your own.
//! - **Namespace strategy**: scope memories per-user (`["mem", user_id]`)
//!   so different users can't see each other's. The cross-user case
//!   (shared knowledge base) is `["mem", "shared"]`.
//! - **Cost control**: extraction is one LLM call per `observe()`. Don't
//!   call it per-message — call it at conversation end or on a timer.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::message::Message;
use crate::model::{ChatModel, ChatOptions};
use crate::store::{SearchFilter, Store};
use crate::structured::StructuredChatModel;
use crate::{Error, Result};

/// Default system prompt for the extractor. Designed to err on the
/// side of fewer-but-higher-quality memories — the LangMem failure
/// mode is "the agent's memory is full of trivia" so we explicitly
/// instruct against it.
pub const DEFAULT_EXTRACTION_SYSTEM_PROMPT: &str = "\
You distil long-term memorable facts from a conversation. Output ONLY \
the JSON object specified by the schema. \

Rules:
- Memorable means: information that would help a future conversation \
  with the same user (preferences, goals, identity, decisions made).
- Skip: trivia, small talk, things the user said once and contradicted \
  later, anything the assistant said about itself.
- Each memory is self-contained — readable without the original turn.
- Prefer fewer, higher-quality memories. If nothing memorable, return \
  an empty `memories` array.
- `kind` is one of: \"preference\", \"fact\", \"goal\", \"identity\", \
  \"decision\". Use \"fact\" if unsure.
- `importance` is 0.0 (forgettable) to 1.0 (critical context).
";

/// One persisted memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Memory {
    pub id: String,
    pub content: String,
    /// Free-form category (preference, fact, goal, identity, decision).
    /// Stays a String rather than an enum so the LLM can return values
    /// the caller registers later without recompiling.
    pub kind: String,
    /// 0.0..=1.0 — caller can use as a threshold for what to surface.
    pub importance: f32,
    pub created_at_ms: u64,
    /// Optional thread/session id the memory came from. Useful for
    /// "show only memories from this conversation" UIs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_thread: Option<String>,
}

impl Memory {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: new_memory_id(),
            content: content.into(),
            kind: "fact".into(),
            importance: 0.5,
            created_at_ms: now_ms(),
            source_thread: None,
        }
    }

    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = kind.into();
        self
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    pub fn with_source_thread(mut self, thread: impl Into<String>) -> Self {
        self.source_thread = Some(thread.into());
        self
    }
}

/// Shape the LLM is asked to produce. Kept separate from `Memory` so the
/// LLM doesn't have to invent an id or timestamp — we add those server-
/// side after parsing.
#[derive(Debug, Clone, Deserialize)]
struct ExtractedMemory {
    content: String,
    #[serde(default = "default_kind")]
    kind: String,
    #[serde(default = "default_importance")]
    importance: f32,
}

fn default_kind() -> String {
    "fact".into()
}
fn default_importance() -> f32 {
    0.5
}

#[derive(Debug, Clone, Deserialize)]
struct ExtractionBatch {
    #[serde(default)]
    memories: Vec<ExtractedMemory>,
}

/// LLM-driven memory extractor. Calls the model once per `extract()`
/// with structured output forced.
pub struct MemoryExtractor {
    model: Arc<dyn ChatModel>,
    system_prompt: String,
    /// Maximum memories the LLM is allowed to return. Defends against
    /// runaway extractions. Default 8.
    max_memories: usize,
}

impl MemoryExtractor {
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            model,
            system_prompt: DEFAULT_EXTRACTION_SYSTEM_PROMPT.to_string(),
            max_memories: 8,
        }
    }

    pub fn with_system_prompt(mut self, p: impl Into<String>) -> Self {
        self.system_prompt = p.into();
        self
    }

    pub fn with_max_memories(mut self, n: usize) -> Self {
        self.max_memories = n;
        self
    }

    /// Extract memorable facts from `messages`. The conversation is
    /// embedded as a single user-role payload so the system prompt
    /// stays in control of behaviour.
    pub async fn extract(&self, messages: &[Message]) -> Result<Vec<Memory>> {
        if messages.is_empty() {
            return Ok(Vec::new());
        }
        let schema = json!({
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "minLength": 1},
                            "kind": {
                                "type": "string",
                                "enum": ["preference", "fact", "goal", "identity", "decision"]
                            },
                            "importance": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["content"]
                    }
                }
            },
            "required": ["memories"]
        });
        let structured =
            StructuredChatModel::new(self.model.clone(), schema, "MemoryBatch")
                .with_strict(true);

        let convo = format_conversation(messages);
        let user = format!(
            "Distil up to {} memorable facts from this conversation.\n\n\
             Conversation:\n{convo}\n\n\
             Output JSON object {{\"memories\": [...]}}.",
            self.max_memories
        );

        let raw = structured
            .invoke_structured(
                vec![
                    Message::system(self.system_prompt.clone()),
                    Message::user(user),
                ],
                &ChatOptions::default(),
            )
            .await?;

        let batch: ExtractionBatch = serde_json::from_value(raw)
            .map_err(|e| Error::other(format!("langmem: bad batch: {e}")))?;

        let mut out = Vec::with_capacity(batch.memories.len().min(self.max_memories));
        for m in batch.memories.into_iter().take(self.max_memories) {
            if m.content.trim().is_empty() {
                continue;
            }
            out.push(
                Memory::new(m.content.trim())
                    .with_kind(m.kind)
                    .with_importance(m.importance),
            );
        }
        Ok(out)
    }
}

/// Render a conversation as plain text for the extractor prompt.
/// Compact format — the LLM doesn't need our internal Message struct.
fn format_conversation(messages: &[Message]) -> String {
    let mut s = String::new();
    for m in messages {
        let role = match m.role {
            crate::message::Role::System => "system",
            crate::message::Role::User => "user",
            crate::message::Role::Assistant => "assistant",
            crate::message::Role::Tool => "tool",
        };
        let text = m.text_content();
        if text.trim().is_empty() {
            continue;
        }
        s.push_str(role);
        s.push_str(": ");
        s.push_str(&text);
        s.push('\n');
    }
    s
}

// ---- High-level orchestrator ----------------------------------------------

/// Glue between the extractor, the store, and the recall path.
/// Cheap to clone (Arc inside).
#[derive(Clone)]
pub struct EpisodicMemory {
    store: Arc<dyn Store>,
    extractor: Arc<MemoryExtractor>,
    namespace: Vec<String>,
    importance_threshold: f32,
    source_thread: Option<String>,
}

impl EpisodicMemory {
    pub fn new(
        store: Arc<dyn Store>,
        extractor: Arc<MemoryExtractor>,
        namespace: Vec<String>,
    ) -> Self {
        Self {
            store,
            extractor,
            namespace,
            importance_threshold: 0.0,
            source_thread: None,
        }
    }

    /// Drop memories the LLM scored below `t` before persisting. Default
    /// 0.0 = keep everything the LLM returned.
    pub fn with_importance_threshold(mut self, t: f32) -> Self {
        self.importance_threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Tag every observed memory with this thread id so a future caller
    /// can filter by it. Set per-conversation, then call `observe`.
    pub fn with_source_thread(mut self, s: impl Into<String>) -> Self {
        self.source_thread = Some(s.into());
        self
    }

    pub fn namespace(&self) -> &[String] {
        &self.namespace
    }

    /// Extract memories from `messages` and persist them. Returns the
    /// memories that were stored (post-filter).
    pub async fn observe(&self, messages: &[Message]) -> Result<Vec<Memory>> {
        let mut mems = self.extractor.extract(messages).await?;
        // Apply threshold.
        mems.retain(|m| m.importance >= self.importance_threshold);
        // Annotate source thread.
        if let Some(t) = &self.source_thread {
            for m in &mut mems {
                m.source_thread = Some(t.clone());
            }
        }
        for m in &mems {
            let value = serde_json::to_value(m)
                .map_err(|e| Error::other(format!("langmem serialise: {e}")))?;
            self.store
                .put(&self.namespace, &m.id, &value, None)
                .await?;
        }
        Ok(mems)
    }

    /// Recall memories relevant to `query`. Backed by `Store::search`
    /// with `query_text` — implementations that support semantic search
    /// (vector backends) will rank by relevance; simple stores fall
    /// back to substring match.
    pub async fn recall(&self, query: &str, k: usize) -> Result<Vec<Memory>> {
        let filter = SearchFilter {
            limit: Some(k),
            offset: None,
            query_text: if query.trim().is_empty() {
                None
            } else {
                Some(query.to_string())
            },
            matches: Vec::new(),
        };
        let items = self.store.search(&self.namespace, &filter).await?;
        let mut out = Vec::with_capacity(items.len());
        for it in items {
            // Skip items that don't deserialise — defensive against a
            // shared namespace polluted by other writers.
            if let Ok(m) = serde_json::from_value::<Memory>(it.value) {
                out.push(m);
            }
        }
        Ok(out)
    }

    /// All memories in the namespace, in store order. Useful for admin
    /// UIs / debugging.
    pub async fn list(&self, limit: Option<usize>) -> Result<Vec<Memory>> {
        let filter = SearchFilter {
            limit,
            ..Default::default()
        };
        let items = self.store.search(&self.namespace, &filter).await?;
        let mut out = Vec::with_capacity(items.len());
        for it in items {
            if let Ok(m) = serde_json::from_value::<Memory>(it.value) {
                out.push(m);
            }
        }
        Ok(out)
    }

    pub async fn forget(&self, memory_id: &str) -> Result<bool> {
        self.store.delete(&self.namespace, memory_id).await
    }

    /// Render `recall(query, k)` as a system-message string ready to
    /// prepend to the next prompt. Format:
    ///
    /// ```text
    /// You have these memories about the user:
    /// - [preference] foo
    /// - [goal] bar
    /// ```
    ///
    /// Returns `None` if no memories match.
    pub async fn recall_as_system_message(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Option<Message>> {
        let mems = self.recall(query, k).await?;
        if mems.is_empty() {
            return Ok(None);
        }
        let mut s = String::from("You have these memories about the user:\n");
        for m in &mems {
            s.push_str(&format!("- [{}] {}\n", m.kind, m.content));
        }
        Ok(Some(Message::system(s)))
    }
}

// ---- helpers ---------------------------------------------------------------

fn new_memory_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("mem-{:x}-{:x}", now_ms(), n)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ChatStream, FinishReason, TokenUsage};
    use crate::store::InMemoryStore;
    use async_trait::async_trait;

    /// Stub model returning a canned structured payload as the assistant
    /// message text. `StructuredChatModel` parses that text as JSON.
    struct CannedModel {
        json_payload: String,
    }

    #[async_trait]
    impl ChatModel for CannedModel {
        fn name(&self) -> &str {
            "canned"
        }
        async fn invoke(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<crate::ChatResponse> {
            Ok(crate::ChatResponse {
                message: Message::assistant(self.json_payload.clone()),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "canned".into(),
            })
        }
        async fn stream(&self, _: Vec<Message>, _: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn convo() -> Vec<Message> {
        vec![
            Message::user("Hi, my name is Alice and I prefer Python over Rust."),
            Message::assistant("Got it, Alice."),
            Message::user("My goal this quarter is to ship a CLI tool."),
        ]
    }

    fn extractor_with(json_payload: &str) -> Arc<MemoryExtractor> {
        let m = Arc::new(CannedModel {
            json_payload: json_payload.into(),
        }) as Arc<dyn ChatModel>;
        Arc::new(MemoryExtractor::new(m))
    }

    #[tokio::test]
    async fn empty_messages_yields_empty_extraction() {
        let ex = extractor_with(r#"{"memories": []}"#);
        let out = ex.extract(&[]).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn extract_parses_memories_and_assigns_ids() {
        let ex = extractor_with(
            r#"{"memories": [
                {"content": "User name is Alice", "kind": "identity", "importance": 0.9},
                {"content": "Prefers Python over Rust", "kind": "preference", "importance": 0.7}
            ]}"#,
        );
        let out = ex.extract(&convo()).await.unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].id.starts_with("mem-"));
        assert_ne!(out[0].id, out[1].id, "ids must be distinct");
        assert_eq!(out[0].content, "User name is Alice");
        assert_eq!(out[0].kind, "identity");
        assert!((out[0].importance - 0.9).abs() < 1e-6);
        assert!(out[0].created_at_ms > 0);
    }

    #[tokio::test]
    async fn extract_caps_at_max_memories() {
        let payload = r#"{"memories": [
            {"content": "a"}, {"content": "b"}, {"content": "c"},
            {"content": "d"}, {"content": "e"}, {"content": "f"}
        ]}"#;
        let m = Arc::new(CannedModel {
            json_payload: payload.into(),
        }) as Arc<dyn ChatModel>;
        let ex = MemoryExtractor::new(m).with_max_memories(3);
        let out = ex.extract(&convo()).await.unwrap();
        assert_eq!(out.len(), 3);
    }

    #[tokio::test]
    async fn extract_drops_blank_content() {
        let ex = extractor_with(
            r#"{"memories": [
                {"content": "real fact"},
                {"content": "   "},
                {"content": ""}
            ]}"#,
        );
        let out = ex.extract(&convo()).await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, "real fact");
    }

    #[tokio::test]
    async fn extract_uses_default_kind_and_importance() {
        let ex = extractor_with(r#"{"memories": [{"content": "x"}]}"#);
        let out = ex.extract(&convo()).await.unwrap();
        assert_eq!(out[0].kind, "fact");
        assert!((out[0].importance - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn observe_persists_to_store() {
        let store: Arc<dyn Store> = Arc::new(InMemoryStore::new());
        let ex = extractor_with(
            r#"{"memories": [
                {"content": "Alice prefers Python", "kind": "preference", "importance": 0.8}
            ]}"#,
        );
        let mem = EpisodicMemory::new(
            store.clone(),
            ex,
            vec!["mem".into(), "user-alice".into()],
        );
        let observed = mem.observe(&convo()).await.unwrap();
        assert_eq!(observed.len(), 1);
        // Listed back via the store directly to confirm persistence shape.
        let listed = mem.list(None).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].content, "Alice prefers Python");
    }

    #[tokio::test]
    async fn observe_filters_by_importance_threshold() {
        let store: Arc<dyn Store> = Arc::new(InMemoryStore::new());
        let ex = extractor_with(
            r#"{"memories": [
                {"content": "important", "importance": 0.9},
                {"content": "trivial", "importance": 0.1}
            ]}"#,
        );
        let mem = EpisodicMemory::new(store, ex, vec!["mem".into(), "u1".into()])
            .with_importance_threshold(0.5);
        let observed = mem.observe(&convo()).await.unwrap();
        assert_eq!(observed.len(), 1);
        assert_eq!(observed[0].content, "important");
    }

    #[tokio::test]
    async fn observe_tags_with_source_thread() {
        let store: Arc<dyn Store> = Arc::new(InMemoryStore::new());
        let ex = extractor_with(r#"{"memories": [{"content": "x"}]}"#);
        let mem = EpisodicMemory::new(store, ex, vec!["mem".into(), "u1".into()])
            .with_source_thread("session-42");
        let observed = mem.observe(&convo()).await.unwrap();
        assert_eq!(observed[0].source_thread.as_deref(), Some("session-42"));
    }

    #[tokio::test]
    async fn recall_returns_substring_matches_against_inmem_store() {
        let store: Arc<dyn Store> = Arc::new(InMemoryStore::new());
        let ex = extractor_with(
            r#"{"memories": [
                {"content": "Alice prefers Python"},
                {"content": "Bob prefers Rust"}
            ]}"#,
        );
        let mem = EpisodicMemory::new(
            store,
            ex,
            vec!["mem".into(), "shared".into()],
        );
        mem.observe(&convo()).await.unwrap();
        let hits = mem.recall("Alice", 5).await.unwrap();
        // InMemoryStore search treats query_text as a substring match
        // over the value JSON. "Alice" should hit the first item.
        assert!(hits.iter().any(|m| m.content.contains("Alice")));
    }

    #[tokio::test]
    async fn recall_as_system_message_renders_bullet_list() {
        let store: Arc<dyn Store> = Arc::new(InMemoryStore::new());
        let ex = extractor_with(
            r#"{"memories": [
                {"content": "Alice prefers Python", "kind": "preference"},
                {"content": "Goal: ship CLI", "kind": "goal"}
            ]}"#,
        );
        let mem = EpisodicMemory::new(store, ex, vec!["mem".into(), "u1".into()]);
        mem.observe(&convo()).await.unwrap();
        let msg = mem.recall_as_system_message("", 10).await.unwrap().unwrap();
        let text = msg.text_content();
        assert!(text.starts_with("You have these memories"));
        assert!(text.contains("[preference] Alice prefers Python"));
        assert!(text.contains("[goal] Goal: ship CLI"));
    }

    #[tokio::test]
    async fn recall_as_system_message_returns_none_when_empty() {
        let store: Arc<dyn Store> = Arc::new(InMemoryStore::new());
        let ex = extractor_with(r#"{"memories": []}"#);
        let mem = EpisodicMemory::new(store, ex, vec!["mem".into(), "u1".into()]);
        let msg = mem.recall_as_system_message("anything", 5).await.unwrap();
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn forget_removes_memory() {
        let store: Arc<dyn Store> = Arc::new(InMemoryStore::new());
        let ex = extractor_with(r#"{"memories": [{"content": "x"}]}"#);
        let mem = EpisodicMemory::new(store, ex, vec!["mem".into(), "u1".into()]);
        let observed = mem.observe(&convo()).await.unwrap();
        let id = observed[0].id.clone();
        assert!(mem.forget(&id).await.unwrap());
        assert!(!mem.forget(&id).await.unwrap()); // idempotent
        assert!(mem.list(None).await.unwrap().is_empty());
    }

    #[test]
    fn memory_builder_sets_fields() {
        let m = Memory::new("hi")
            .with_kind("goal")
            .with_importance(2.0) // clamped
            .with_source_thread("t1");
        assert_eq!(m.kind, "goal");
        assert!((m.importance - 1.0).abs() < 1e-6);
        assert_eq!(m.source_thread.as_deref(), Some("t1"));
    }

    #[test]
    fn memory_importance_clamps_negatives() {
        let m = Memory::new("hi").with_importance(-0.5);
        assert!((m.importance - 0.0).abs() < 1e-6);
    }

    #[test]
    fn format_conversation_skips_empty_messages() {
        let msgs = vec![
            Message::user("real"),
            Message::user("   "),
            Message::assistant("also real"),
        ];
        let out = format_conversation(&msgs);
        assert!(out.contains("user: real"));
        assert!(out.contains("assistant: also real"));
        // Empty message dropped.
        assert_eq!(out.lines().count(), 2);
    }

    #[test]
    fn new_memory_id_collision_proof_within_process() {
        let n = 1000;
        let ids: std::collections::HashSet<String> =
            (0..n).map(|_| new_memory_id()).collect();
        assert_eq!(ids.len(), n, "id generator collided");
    }
}
