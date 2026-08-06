use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Embeddings, Error, Message, Result};

/// One captured request/response pair. Hash key is computed from
/// canonical JSON of `(messages, opts)`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CassetteExchange {
    pub request_hash: String,
    pub messages: Vec<Message>,
    pub opts: ChatOptions,
    pub response: ChatResponse,
}

/// Persistable record of LLM interactions. Round-trips via
/// `serde_json` so cassettes can live in a `tests/` directory
/// next to the test file.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Cassette {
    #[serde(default = "default_version")]
    pub version: u32,
    pub exchanges: Vec<CassetteExchange>,
}

fn default_version() -> u32 {
    1
}

impl Cassette {
    /// Load a cassette from a JSON file. Convenience wrapper around
    /// `std::fs::read_to_string` + `serde_json::from_str`. Errors
    /// surface as `Error::Other(...)`.
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let s = std::fs::read_to_string(path)
            .map_err(|e| Error::other(format!("read cassette {path:?}: {e}")))?;
        serde_json::from_str(&s)
            .map_err(|e| Error::other(format!("parse cassette {path:?}: {e}")))
    }

    /// Save a cassette as pretty JSON. Creates parent directory if
    /// needed.
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    Error::other(format!("mkdir {parent:?}: {e}"))
                })?;
            }
        }
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| Error::other(format!("serialize cassette: {e}")))?;
        std::fs::write(path, s)
            .map_err(|e| Error::other(format!("write cassette {path:?}: {e}")))?;
        Ok(())
    }
}

/// Compute a deterministic hash of a `(messages, opts)` request.
/// Uses canonical JSON (no field ordering, no whitespace) over
/// blake3 → hex string.
pub fn exchange_hash(messages: &[Message], opts: &ChatOptions) -> String {
    // serde_json::to_string is stable for our types because the
    // structs use `#[derive(Serialize)]` with deterministic field
    // order. There are no `HashMap` fields in messages or opts.
    let req = serde_json::json!({ "messages": messages, "opts": opts });
    let s = serde_json::to_string(&req).unwrap_or_default();
    let h = blake3::hash(s.as_bytes());
    h.to_hex().to_string()
}

/// Wrap any [`ChatModel`] to record every `invoke` call into a
/// shared [`Cassette`]. Use during a real-traffic test run; serialize
/// the cassette to disk; replay in CI via [`ReplayingChatModel`].
///
/// VCR-style determinism for agent tests — no API calls in CI, no
/// flaky stochastic outputs, no quota burn for golden-set
/// regression tests.
///
/// # Streaming
///
/// `stream()` is NOT recorded — token streams aren't replayable
/// in a useful way without preserving inter-chunk timing. Stream
/// calls pass through to inner. Tests that want streaming
/// determinism should use [`ReplayingChatModel`] in invoke-mode
/// and manually re-emit a synthetic stream.
pub struct RecordingChatModel {
    pub inner: Arc<dyn ChatModel>,
    cassette: Arc<parking_lot::Mutex<Cassette>>,
}

impl RecordingChatModel {
    pub fn new(
        inner: Arc<dyn ChatModel>,
        cassette: Arc<parking_lot::Mutex<Cassette>>,
    ) -> Self {
        Self { inner, cassette }
    }
}

#[async_trait]
impl ChatModel for RecordingChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let response = self.inner.invoke(messages.clone(), opts).await?;
        let request_hash = exchange_hash(&messages, opts);
        let exchange = CassetteExchange {
            request_hash,
            messages,
            opts: opts.clone(),
            response: response.clone(),
        };
        self.cassette.lock().exchanges.push(exchange);
        Ok(response)
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // See doc — streams pass through unrecorded.
        self.inner.stream(messages, opts).await
    }
}

/// Replay recorded LLM interactions from a [`Cassette`]. Lookup
/// is by request hash — exact match on canonical JSON of
/// `(messages, opts)`.
///
/// On miss: returns `Error::Provider("no recorded response for
/// hash <…>")` if no `passthrough` is set, or invokes
/// `passthrough` (typically the live model) otherwise. The
/// passthrough branch is useful for "record-then-replay-with-
/// gap-fill" workflows.
pub struct ReplayingChatModel {
    pub cassette: Cassette,
    pub passthrough: Option<Arc<dyn ChatModel>>,
}

impl ReplayingChatModel {
    pub fn new(cassette: Cassette, passthrough: Option<Arc<dyn ChatModel>>) -> Self {
        Self {
            cassette,
            passthrough,
        }
    }
}

#[async_trait]
impl ChatModel for ReplayingChatModel {
    fn name(&self) -> &str {
        "replaying"
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let hash = exchange_hash(&messages, opts);
        if let Some(ex) = self
            .cassette
            .exchanges
            .iter()
            .find(|e| e.request_hash == hash)
        {
            return Ok(ex.response.clone());
        }
        if let Some(pt) = &self.passthrough {
            return pt.invoke(messages, opts).await;
        }
        Err(Error::Provider(format!(
            "no recorded response for hash {hash}",
        )))
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // Streaming replay isn't supported (see RecordingChatModel
        // doc). Defer to passthrough or error.
        if let Some(pt) = &self.passthrough {
            return pt.stream(messages, opts).await;
        }
        Err(Error::Provider(
            "ReplayingChatModel does not support stream() without passthrough".into(),
        ))
    }
}

/// One captured tool call. Hash key: blake3 over canonical JSON
/// of `args`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolExchange {
    pub request_hash: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub response: serde_json::Value,
}

/// Persistable record of `Tool::run` calls. Same load/save shape
/// as [`Cassette`] / [`EmbedCassette`].
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ToolCassette {
    #[serde(default = "default_version")]
    pub version: u32,
    pub exchanges: Vec<ToolExchange>,
}

impl ToolCassette {
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let s = std::fs::read_to_string(path)
            .map_err(|e| Error::other(format!("read tool cassette {path:?}: {e}")))?;
        serde_json::from_str(&s)
            .map_err(|e| Error::other(format!("parse tool cassette {path:?}: {e}")))
    }

    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    Error::other(format!("mkdir {parent:?}: {e}"))
                })?;
            }
        }
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| Error::other(format!("serialize tool cassette: {e}")))?;
        std::fs::write(path, s)
            .map_err(|e| Error::other(format!("write tool cassette {path:?}: {e}")))?;
        Ok(())
    }
}

/// blake3 over canonical JSON of `args` for tool exchange keys.
pub fn tool_args_hash(args: &serde_json::Value) -> String {
    let s = serde_json::to_string(args).unwrap_or_default();
    blake3::hash(s.as_bytes()).to_hex().to_string()
}

/// Wrap any [`litgraph_core::tool::Tool`] to record every `run`
/// call into a shared [`ToolCassette`]. Closes the third axis of
/// the record/replay matrix (after iters 254 chat and 255 embed).
///
/// Real prod use: agent integration tests with deterministic
/// tool side effects. Record real tool runs against a staging
/// API; replay in CI without hitting the real service.
pub struct RecordingTool {
    pub inner: Arc<dyn litgraph_core::tool::Tool>,
    cassette: Arc<parking_lot::Mutex<ToolCassette>>,
}

impl RecordingTool {
    pub fn new(
        inner: Arc<dyn litgraph_core::tool::Tool>,
        cassette: Arc<parking_lot::Mutex<ToolCassette>>,
    ) -> Self {
        Self { inner, cassette }
    }
}

#[async_trait]
impl litgraph_core::tool::Tool for RecordingTool {
    fn schema(&self) -> litgraph_core::tool::ToolSchema {
        self.inner.schema()
    }

    async fn run(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let response = self.inner.run(args.clone()).await?;
        let exchange = ToolExchange {
            request_hash: tool_args_hash(&args),
            tool_name: self.inner.name(),
            args,
            response: response.clone(),
        };
        self.cassette.lock().exchanges.push(exchange);
        Ok(response)
    }
}

/// Replay recorded tool runs from a [`ToolCassette`]. Matches by
/// args hash. On miss: returns an error or falls through to
/// `passthrough` if set. Schema is inherited from `passthrough`
/// when present, otherwise a synthesized schema with the
/// configured `name` and an empty parameters object — sufficient
/// to satisfy callers that only need the cassette's responses.
pub struct ReplayingTool {
    pub cassette: ToolCassette,
    pub passthrough: Option<Arc<dyn litgraph_core::tool::Tool>>,
    pub name: String,
    pub description: String,
}

impl ReplayingTool {
    pub fn new(
        cassette: ToolCassette,
        passthrough: Option<Arc<dyn litgraph_core::tool::Tool>>,
    ) -> Self {
        Self {
            cassette,
            passthrough,
            name: "replaying-tool".into(),
            description: "Replay-only tool backed by a ToolCassette".into(),
        }
    }

    /// Override the tool's reported name (controls the synthesized
    /// schema when no passthrough is configured).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Override the tool's reported description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

#[async_trait]
impl litgraph_core::tool::Tool for ReplayingTool {
    fn schema(&self) -> litgraph_core::tool::ToolSchema {
        if let Some(pt) = &self.passthrough {
            return pt.schema();
        }
        litgraph_core::tool::ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }
    }

    async fn run(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let hash = tool_args_hash(&args);
        if let Some(ex) = self
            .cassette
            .exchanges
            .iter()
            .find(|e| e.request_hash == hash)
        {
            return Ok(ex.response.clone());
        }
        if let Some(pt) = &self.passthrough {
            return pt.run(args).await;
        }
        Err(Error::Provider(format!(
            "no recorded tool response for hash {hash}",
        )))
    }
}

/// One captured embed call. The two variants correspond to the
/// two `Embeddings` trait methods.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EmbedExchange {
    Query {
        request_hash: String,
        text: String,
        response: Vec<f32>,
    },
    Documents {
        request_hash: String,
        texts: Vec<String>,
        response: Vec<Vec<f32>>,
    },
}

/// Persistable record of `embed_query` / `embed_documents` calls.
/// Same load/save shape as [`Cassette`].
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EmbedCassette {
    #[serde(default = "default_version")]
    pub version: u32,
    pub exchanges: Vec<EmbedExchange>,
}

impl EmbedCassette {
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let s = std::fs::read_to_string(path)
            .map_err(|e| Error::other(format!("read embed cassette {path:?}: {e}")))?;
        serde_json::from_str(&s)
            .map_err(|e| Error::other(format!("parse embed cassette {path:?}: {e}")))
    }

    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    Error::other(format!("mkdir {parent:?}: {e}"))
                })?;
            }
        }
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| Error::other(format!("serialize embed cassette: {e}")))?;
        std::fs::write(path, s)
            .map_err(|e| Error::other(format!("write embed cassette {path:?}: {e}")))?;
        Ok(())
    }
}

/// blake3 over `text` for `embed_query` exchange keys.
pub fn embed_query_hash(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

/// blake3 over canonical JSON of `texts` for `embed_documents`
/// exchange keys.
pub fn embed_documents_hash(texts: &[String]) -> String {
    let s = serde_json::to_string(texts).unwrap_or_default();
    blake3::hash(s.as_bytes()).to_hex().to_string()
}

/// Wrap any [`Embeddings`] to record every `embed_query` /
/// `embed_documents` call into a shared [`EmbedCassette`].
pub struct RecordingEmbeddings {
    pub inner: Arc<dyn Embeddings>,
    cassette: Arc<parking_lot::Mutex<EmbedCassette>>,
}

impl RecordingEmbeddings {
    pub fn new(
        inner: Arc<dyn Embeddings>,
        cassette: Arc<parking_lot::Mutex<EmbedCassette>>,
    ) -> Self {
        Self { inner, cassette }
    }
}

#[async_trait]
impl Embeddings for RecordingEmbeddings {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let response = self.inner.embed_query(text).await?;
        let exchange = EmbedExchange::Query {
            request_hash: embed_query_hash(text),
            text: text.to_owned(),
            response: response.clone(),
        };
        self.cassette.lock().exchanges.push(exchange);
        Ok(response)
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let response = self.inner.embed_documents(texts).await?;
        let exchange = EmbedExchange::Documents {
            request_hash: embed_documents_hash(texts),
            texts: texts.to_vec(),
            response: response.clone(),
        };
        self.cassette.lock().exchanges.push(exchange);
        Ok(response)
    }
}

/// Replay recorded embedding interactions. Like
/// [`ReplayingChatModel`]: hash lookup, optional passthrough on
/// miss. `dimensions()` defaults to whatever the first
/// `Documents` / `Query` exchange returns; if the cassette is
/// empty it returns 0 (caller should pass through to a live
/// embedder if dimensions matter pre-recording).
pub struct ReplayingEmbeddings {
    pub cassette: EmbedCassette,
    pub passthrough: Option<Arc<dyn Embeddings>>,
    pub name: String,
}

impl ReplayingEmbeddings {
    pub fn new(
        cassette: EmbedCassette,
        passthrough: Option<Arc<dyn Embeddings>>,
    ) -> Self {
        Self {
            cassette,
            passthrough,
            name: "replaying-embed".into(),
        }
    }

    fn first_dim(&self) -> usize {
        self.cassette.exchanges.iter().find_map(|e| match e {
            EmbedExchange::Query { response, .. } => Some(response.len()),
            EmbedExchange::Documents { response, .. } => {
                response.first().map(|v| v.len())
            }
        }).unwrap_or(0)
    }
}

#[async_trait]
impl Embeddings for ReplayingEmbeddings {
    fn name(&self) -> &str {
        &self.name
    }
    fn dimensions(&self) -> usize {
        if let Some(pt) = &self.passthrough {
            return pt.dimensions();
        }
        self.first_dim()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let hash = embed_query_hash(text);
        for ex in &self.cassette.exchanges {
            if let EmbedExchange::Query {
                request_hash,
                response,
                ..
            } = ex
            {
                if request_hash == &hash {
                    return Ok(response.clone());
                }
            }
        }
        if let Some(pt) = &self.passthrough {
            return pt.embed_query(text).await;
        }
        Err(Error::Provider(format!(
            "no recorded embed_query response for hash {hash}",
        )))
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let hash = embed_documents_hash(texts);
        for ex in &self.cassette.exchanges {
            if let EmbedExchange::Documents {
                request_hash,
                response,
                ..
            } = ex
            {
                if request_hash == &hash {
                    return Ok(response.clone());
                }
            }
        }
        if let Some(pt) = &self.passthrough {
            return pt.embed_documents(texts).await;
        }
        Err(Error::Provider(format!(
            "no recorded embed_documents response for hash {hash}",
        )))
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

    // ---- RecordingChatModel / ReplayingChatModel tests -----------------

    #[tokio::test]
    async fn record_then_replay_round_trip() {
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let cassette = Arc::new(parking_lot::Mutex::new(Cassette::default()));
        let recorder = RecordingChatModel::new(inner, cassette.clone());
        // Make 3 calls with different inputs.
        recorder
            .invoke(vec![Message::user("a")], &ChatOptions::default())
            .await
            .unwrap();
        recorder
            .invoke(vec![Message::user("b")], &ChatOptions::default())
            .await
            .unwrap();
        recorder
            .invoke(vec![Message::user("a")], &ChatOptions::default())
            .await
            .unwrap();
        let snap = cassette.lock().clone();
        assert_eq!(snap.exchanges.len(), 3);
        // Now replay using the cassette.
        let player = ReplayingChatModel::new(snap, None);
        let r1 = player
            .invoke(vec![Message::user("a")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r1.message.text_content(), "hi"); // AlwaysOkModel returns "hi"
        // Same input replayed twice — should hit twice.
        let r2 = player
            .invoke(vec![Message::user("a")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r2.message.text_content(), "hi");
    }

    #[tokio::test]
    async fn replay_miss_returns_error_when_no_passthrough() {
        let cassette = Cassette::default();
        let player = ReplayingChatModel::new(cassette, None);
        let r = player
            .invoke(vec![Message::user("nope")], &ChatOptions::default())
            .await;
        match r {
            Err(Error::Provider(msg)) => {
                assert!(msg.contains("no recorded response"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn replay_miss_falls_through_to_passthrough() {
        let cassette = Cassette::default();
        let live: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let player = ReplayingChatModel::new(cassette, Some(live));
        let r = player
            .invoke(vec![Message::user("anything")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r.message.text_content(), "hi");
    }

    #[tokio::test]
    async fn cassette_serializes_to_json_round_trip() {
        // Round-trip via JSON to verify the cassette is portable.
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let cass = Arc::new(parking_lot::Mutex::new(Cassette::default()));
        let recorder = RecordingChatModel::new(inner, cass.clone());
        recorder
            .invoke(vec![Message::user("hello")], &ChatOptions::default())
            .await
            .unwrap();
        let snap = cass.lock().clone();
        let s = serde_json::to_string(&snap).unwrap();
        let restored: Cassette = serde_json::from_str(&s).unwrap();
        assert_eq!(restored.exchanges.len(), 1);
        // Replay using the restored cassette.
        let player = ReplayingChatModel::new(restored, None);
        let r = player
            .invoke(vec![Message::user("hello")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r.message.text_content(), "hi");
    }

    #[tokio::test]
    async fn request_hash_is_deterministic_across_orderings() {
        // Two requests with semantically-identical canonical JSON
        // must hash the same regardless of HashMap iteration order
        // for opts (response_format etc). AlwaysOkModel ignores opts
        // anyway, so the test verifies the hash function via direct
        // comparison.
        let m1 = vec![Message::user("hi")];
        let m2 = vec![Message::user("hi")];
        let opts = ChatOptions {
            temperature: Some(0.7),
            ..Default::default()
        };
        let h1 = exchange_hash(&m1, &opts);
        let h2 = exchange_hash(&m2, &opts);
        assert_eq!(h1, h2);
        let opts2 = ChatOptions {
            temperature: Some(0.5),
            ..Default::default()
        };
        let h3 = exchange_hash(&m1, &opts2);
        assert_ne!(h1, h3);
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


    // ---- Cassette file IO + Embeddings record/replay tests --------------

    #[tokio::test]
    async fn cassette_save_and_load_round_trip_through_disk() {
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let cass = Arc::new(parking_lot::Mutex::new(Cassette::default()));
        let recorder = RecordingChatModel::new(inner, cass.clone());
        recorder
            .invoke(vec![Message::user("disk-test")], &ChatOptions::default())
            .await
            .unwrap();
        let snap = cass.lock().clone();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        snap.save_to_file(&path).unwrap();
        let restored = Cassette::load_from_file(&path).unwrap();
        assert_eq!(restored.exchanges.len(), 1);
        let player = ReplayingChatModel::new(restored, None);
        let r = player
            .invoke(vec![Message::user("disk-test")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r.message.text_content(), "hi");
    }

    #[tokio::test]
    async fn cassette_save_creates_parent_directory() {
        let cass = Cassette::default();
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("a").join("b").join("c.json");
        cass.save_to_file(&nested).unwrap();
        assert!(nested.exists());
    }

    #[tokio::test]
    async fn embed_record_then_replay_round_trip() {
        let inner = flaky_embed(0, EmbedFlakyKind::Provider5xx, 4);
        let cass = Arc::new(parking_lot::Mutex::new(EmbedCassette::default()));
        let recorder =
            RecordingEmbeddings::new(inner.clone() as Arc<dyn Embeddings>, cass.clone());
        // Record both embed_query and embed_documents.
        recorder.embed_query("hello").await.unwrap();
        recorder
            .embed_documents(&["a".into(), "b".into()])
            .await
            .unwrap();
        recorder.embed_query("world").await.unwrap();
        let snap = cass.lock().clone();
        assert_eq!(snap.exchanges.len(), 3);
        // Replay via the cassette.
        let player = ReplayingEmbeddings::new(snap, None);
        let v1 = player.embed_query("hello").await.unwrap();
        assert_eq!(v1.len(), 4); // FlakyEmbed dim
        let v2 = player.embed_query("world").await.unwrap();
        assert_eq!(v2.len(), 4);
        let docs = player
            .embed_documents(&["a".into(), "b".into()])
            .await
            .unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[tokio::test]
    async fn embed_replay_miss_returns_error_when_no_passthrough() {
        let cass = EmbedCassette::default();
        let player = ReplayingEmbeddings::new(cass, None);
        let r = player.embed_query("nope").await;
        match r {
            Err(Error::Provider(msg)) => {
                assert!(msg.contains("no recorded embed_query"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
        let r2 = player.embed_documents(&["x".into()]).await;
        match r2 {
            Err(Error::Provider(msg)) => {
                assert!(msg.contains("no recorded embed_documents"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn embed_replay_miss_falls_through_to_passthrough() {
        let cass = EmbedCassette::default();
        let live = flaky_embed(0, EmbedFlakyKind::Provider5xx, 5);
        let player =
            ReplayingEmbeddings::new(cass, Some(live as Arc<dyn Embeddings>));
        let v = player.embed_query("anything").await.unwrap();
        assert_eq!(v.len(), 5);
    }

    #[tokio::test]
    async fn embed_replay_dimensions_proxy_or_first_exchange() {
        // Without passthrough: dimensions from the first exchange.
        let mut cass = EmbedCassette::default();
        cass.exchanges.push(EmbedExchange::Query {
            request_hash: embed_query_hash("x"),
            text: "x".into(),
            response: vec![0.0; 11],
        });
        let player = ReplayingEmbeddings::new(cass, None);
        assert_eq!(player.dimensions(), 11);

        // With passthrough: dimensions delegated.
        let live = flaky_embed(0, EmbedFlakyKind::Provider5xx, 13);
        let player2 = ReplayingEmbeddings::new(
            EmbedCassette::default(),
            Some(live as Arc<dyn Embeddings>),
        );
        assert_eq!(player2.dimensions(), 13);
    }

    #[tokio::test]
    async fn embed_cassette_save_and_load_through_disk() {
        let inner = flaky_embed(0, EmbedFlakyKind::Provider5xx, 3);
        let cass = Arc::new(parking_lot::Mutex::new(EmbedCassette::default()));
        let recorder =
            RecordingEmbeddings::new(inner as Arc<dyn Embeddings>, cass.clone());
        recorder.embed_query("disk-text").await.unwrap();
        let snap = cass.lock().clone();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        snap.save_to_file(tmp.path()).unwrap();
        let restored = EmbedCassette::load_from_file(tmp.path()).unwrap();
        let player = ReplayingEmbeddings::new(restored, None);
        let v = player.embed_query("disk-text").await.unwrap();
        assert_eq!(v.len(), 3);
    }

    #[tokio::test]
    async fn embed_query_hash_is_deterministic() {
        assert_eq!(embed_query_hash("a"), embed_query_hash("a"));
        assert_ne!(embed_query_hash("a"), embed_query_hash("b"));
    }

    #[tokio::test]
    async fn embed_documents_hash_distinguishes_order() {
        // Order matters: ["a","b"] != ["b","a"]. This is intentional —
        // embed_documents returns aligned vectors so order is part of
        // the request semantically.
        let h1 = embed_documents_hash(&["a".into(), "b".into()]);
        let h2 = embed_documents_hash(&["b".into(), "a".into()]);
        assert_ne!(h1, h2);
    }

    // ---- Tool record/replay tests --------------------------------------

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

    #[tokio::test]
    async fn tool_record_then_replay_round_trip() {
        let inner = Arc::new(EchoTool {
            seen: AtomicU32::new(0),
        });
        let cass = Arc::new(parking_lot::Mutex::new(ToolCassette::default()));
        let recorder = RecordingTool::new(
            inner.clone() as Arc<dyn litgraph_core::tool::Tool>,
            cass.clone(),
        );
        // Record 3 calls.
        let r1 = recorder.run(serde_json::json!({"x": 1})).await.unwrap();
        let _ = recorder.run(serde_json::json!({"x": 2})).await.unwrap();
        let _ = recorder.run(serde_json::json!({"x": 1})).await.unwrap();
        assert_eq!(r1, serde_json::json!({"echo": {"x": 1}}));
        let snap = cass.lock().clone();
        assert_eq!(snap.exchanges.len(), 3);
        assert_eq!(inner.seen.load(Ordering::SeqCst), 3);
        // Replay.
        let player = ReplayingTool::new(snap, None);
        let r1b = player.run(serde_json::json!({"x": 1})).await.unwrap();
        assert_eq!(r1b, serde_json::json!({"echo": {"x": 1}}));
        let r2b = player.run(serde_json::json!({"x": 2})).await.unwrap();
        assert_eq!(r2b, serde_json::json!({"echo": {"x": 2}}));
        // Replay didn't bump inner.seen.
        assert_eq!(inner.seen.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn tool_replay_miss_returns_error_when_no_passthrough() {
        let cass = ToolCassette::default();
        let player = ReplayingTool::new(cass, None);
        let r = player.run(serde_json::json!({"q": "nope"})).await;
        match r {
            Err(Error::Provider(msg)) => {
                assert!(msg.contains("no recorded tool response"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_replay_miss_falls_through_to_passthrough() {
        let cass = ToolCassette::default();
        let live = Arc::new(EchoTool {
            seen: AtomicU32::new(0),
        });
        let player = ReplayingTool::new(
            cass,
            Some(live.clone() as Arc<dyn litgraph_core::tool::Tool>),
        );
        let r = player.run(serde_json::json!({"x": 7})).await.unwrap();
        assert_eq!(r, serde_json::json!({"echo": {"x": 7}}));
        assert_eq!(live.seen.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn tool_cassette_save_and_load_through_disk() {
        let inner = Arc::new(EchoTool {
            seen: AtomicU32::new(0),
        });
        let cass = Arc::new(parking_lot::Mutex::new(ToolCassette::default()));
        let recorder = RecordingTool::new(
            inner as Arc<dyn litgraph_core::tool::Tool>,
            cass.clone(),
        );
        recorder
            .run(serde_json::json!({"disk": "test"}))
            .await
            .unwrap();
        let snap = cass.lock().clone();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        snap.save_to_file(tmp.path()).unwrap();
        let restored = ToolCassette::load_from_file(tmp.path()).unwrap();
        let player = ReplayingTool::new(restored, None);
        let r = player
            .run(serde_json::json!({"disk": "test"}))
            .await
            .unwrap();
        assert_eq!(r, serde_json::json!({"echo": {"disk": "test"}}));
    }

    #[tokio::test]
    async fn tool_args_hash_is_deterministic() {
        let h1 = tool_args_hash(&serde_json::json!({"x": 1, "y": 2}));
        let h2 = tool_args_hash(&serde_json::json!({"x": 1, "y": 2}));
        assert_eq!(h1, h2);
        let h3 = tool_args_hash(&serde_json::json!({"x": 1, "y": 3}));
        assert_ne!(h1, h3);
    }

    #[tokio::test]
    async fn tool_replay_schema_synthesized_when_no_passthrough() {
        let cass = ToolCassette::default();
        let player = ReplayingTool::new(cass, None)
            .with_name("custom_tool")
            .with_description("desc");
        let s = player.schema();
        assert_eq!(s.name, "custom_tool");
        assert_eq!(s.description, "desc");
        assert!(s.parameters.is_object());
    }

    #[tokio::test]
    async fn tool_replay_schema_proxied_to_passthrough() {
        let live = Arc::new(EchoTool {
            seen: AtomicU32::new(0),
        });
        let cass = ToolCassette::default();
        let player = ReplayingTool::new(
            cass,
            Some(live as Arc<dyn litgraph_core::tool::Tool>),
        );
        let s = player.schema();
        assert_eq!(s.name, "echo");
    }

}
