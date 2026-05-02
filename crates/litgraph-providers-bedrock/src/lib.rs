//! AWS Bedrock provider adapter. Targets the Anthropic Claude family on
//! Bedrock — the wire format is the same Messages API, with `anthropic_version`
//! pinned to a Bedrock-specific value and the model ID encoded in the URL.
//!
//! # What's supported (v1)
//! - `invoke` (POST /model/{id}/invoke) — full chat + tool use.
//!
//! # Not yet supported
//! - `stream` (POST /model/{id}/invoke-with-response-stream) — Bedrock returns
//!   an AWS event-stream binary frame format that requires a non-trivial parser.
//!   `stream` falls back to a single-shot `invoke` wrapped as a `Done` event so
//!   the API works, just without progressive deltas.
//! - Non-Anthropic model families (Titan, Jurassic, Llama, Mistral) — each has
//!   its own request schema; PRs welcome to add them.
//!
//! # Auth
//! Static credentials only for now: `access_key_id`, `secret_access_key`,
//! optional `session_token`. Users on EC2/ECS/Lambda should resolve creds with
//! the AWS SDK at app startup, then pass them in.

mod event_stream;
pub mod sigv4;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use chrono::Utc;
use futures_util::{StreamExt, TryStreamExt};
use litgraph_core::model::{ChatStream, ChatStreamEvent};
use litgraph_core::tool::ToolCall;
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, ContentPart, Embeddings, Error, FinishReason, Message,
    Result, Role, TokenUsage,
};
use reqwest::Client;
use serde_json::{Value, json};
use tracing::debug;

use crate::event_stream::parse_frame;

pub use sigv4::AwsCredentials;

/// Bedrock-specific Anthropic version pin. (Bedrock does NOT accept the
/// public Anthropic version like "2023-06-01"; it has its own constant.)
pub const BEDROCK_ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

pub type RequestInspector = Arc<dyn Fn(&str, &Value) + Send + Sync>;

#[derive(Clone)]
pub struct BedrockConfig {
    pub credentials: AwsCredentials,
    pub region: String,
    pub model_id: String,
    pub timeout: Duration,
    pub max_tokens: u32,
    /// Override the URL prefix (default `https://bedrock-runtime.{region}.amazonaws.com`).
    /// Useful for VPC PrivateLink endpoints, AWS GovCloud, integration tests
    /// against a fake server. When set, the request still gets SigV4-signed
    /// against the override host (NOT the AWS public hostname).
    pub endpoint_override: Option<String>,
    /// Pre-flight inspector. Fires *before* SigV4 signing — gets the unsigned
    /// JSON body so callers see exactly what the model will see.
    pub on_request: Option<RequestInspector>,
}

impl std::fmt::Debug for BedrockConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockConfig")
            .field("region", &self.region)
            .field("model_id", &self.model_id)
            .field("timeout", &self.timeout)
            .field("max_tokens", &self.max_tokens)
            .field("endpoint_override", &self.endpoint_override)
            .field("on_request", &self.on_request.as_ref().map(|_| "<callback>"))
            .finish()
    }
}

impl BedrockConfig {
    pub fn new(creds: AwsCredentials, region: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            credentials: creds,
            region: region.into(),
            model_id: model_id.into(),
            timeout: Duration::from_secs(120),
            max_tokens: 4096,
            endpoint_override: None,
            on_request: None,
        }
    }

    pub fn with_endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint_override = Some(url.into());
        self
    }

    pub fn with_on_request<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &Value) + Send + Sync + 'static,
    {
        self.on_request = Some(Arc::new(f));
        self
    }
}

/// Returned `(host, base_url)` — host is what we sign against (no scheme,
/// no port), base_url is the URL prefix we POST to.
fn resolve_endpoint(cfg: &BedrockConfig) -> (String, String) {
    let base = cfg
        .endpoint_override
        .clone()
        .unwrap_or_else(|| format!("https://bedrock-runtime.{}.amazonaws.com", cfg.region));
    let base = base.trim_end_matches('/').to_string();
    let host = base
        .split_once("://")
        .map(|(_, after)| after)
        .unwrap_or(&base)
        .split('/')
        .next()
        .unwrap_or(&base)
        .to_string();
    (host, base)
}

pub struct BedrockChat {
    cfg: BedrockConfig,
    http: Client,
}

impl BedrockChat {
    pub fn new(cfg: BedrockConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    fn body(&self, messages: &[Message], opts: &ChatOptions) -> Value {
        // Identical shape to public Anthropic Messages API except the version pin.
        let (system, rest): (Vec<&Message>, Vec<&Message>) = messages
            .iter()
            .partition(|m| matches!(m.role, Role::System));

        let max = opts.max_tokens.unwrap_or(self.cfg.max_tokens);
        let mut body = json!({
            "anthropic_version": BEDROCK_ANTHROPIC_VERSION,
            "max_tokens": max,
            "messages": rest.iter().map(|m| message_to_anthropic(m)).collect::<Vec<_>>(),
        });
        if !system.is_empty() {
            // Typed-block array so cache_control can attach per-block.
            let parts: Vec<Value> = system.iter().filter_map(|m| {
                let t = m.text_content();
                if t.is_empty() { return None; }
                let mut p = json!({"type": "text", "text": t});
                if m.cache {
                    p["cache_control"] = json!({"type": "ephemeral"});
                }
                Some(p)
            }).collect();
            if !parts.is_empty() {
                body["system"] = Value::Array(parts);
            }
        }
        if let Some(t) = opts.temperature { body["temperature"] = json!(t); }
        if let Some(t) = opts.top_p { body["top_p"] = json!(t); }
        if let Some(ref s) = opts.stop { body["stop_sequences"] = json!(s); }
        if !opts.tools.is_empty() {
            body["tools"] = json!(opts.tools.iter().map(|t| json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            })).collect::<Vec<_>>());
        }
        body
    }

    async fn invoke_inner(&self, body: &Value) -> Result<Value> {
        if let Some(cb) = &self.cfg.on_request {
            cb(&self.cfg.model_id, body);
        }
        let (host, base) = resolve_endpoint(&self.cfg);
        let path = format!("/model/{}/invoke", urlencoded(&self.cfg.model_id));
        let url = format!("{base}{path}");
        let body_bytes = serde_json::to_vec(body).map_err(Error::from)?;

        let signed = sigv4::sign(
            &self.cfg.credentials,
            &sigv4::SigningInputs {
                method: "POST",
                host: &host,
                path: &path,
                query: "",
                body: &body_bytes,
                extra_headers: &[("content-type".into(), "application/json".into())],
                region: &self.cfg.region,
                service: "bedrock",
                now: Utc::now(),
            },
        );

        debug!(model = %self.cfg.model_id, host, "bedrock invoke");
        let mut req = self
            .http
            .post(&url)
            .header("content-type", "application/json")
            .header("authorization", &signed.authorization)
            .header("x-amz-date", &signed.x_amz_date)
            .header("x-amz-content-sha256", &signed.x_amz_content_sha256)
            .body(body_bytes);
        if let Some(tok) = &signed.x_amz_security_token {
            req = req.header("x-amz-security-token", tok);
        }
        let resp = req.send().await.map_err(|e| Error::provider(format!("send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(Error::RateLimited { retry_after_ms: None });
            }
            return Err(Error::provider(format!("{status}: {txt}")));
        }
        resp.json().await.map_err(|e| Error::provider(format!("decode: {e}")))
    }
}

#[async_trait]
impl ChatModel for BedrockChat {
    fn name(&self) -> &str { &self.cfg.model_id }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let body = self.body(&messages, opts);
        let v = self.invoke_inner(&body).await?;
        parse_response(&self.cfg.model_id, v)
    }

    /// Bedrock streaming via `InvokeModelWithResponseStream`. Parses the AWS
    /// event-stream binary frame format, base64-decodes each chunk's payload,
    /// then interprets the inner JSON as Anthropic Messages-API SSE events
    /// (the Anthropic-on-Bedrock pattern; non-Anthropic models would need
    /// their own inner-payload parser).
    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        let body = self.body(&messages, opts);
        if let Some(cb) = &self.cfg.on_request {
            cb(&self.cfg.model_id, &body);
        }
        let (host, base) = resolve_endpoint(&self.cfg);
        let path = format!("/model/{}/invoke-with-response-stream", urlencoded(&self.cfg.model_id));
        let url = format!("{base}{path}");
        let body_bytes = serde_json::to_vec(&body).map_err(Error::from)?;

        let signed = sigv4::sign(
            &self.cfg.credentials,
            &sigv4::SigningInputs {
                method: "POST",
                host: &host,
                path: &path,
                query: "",
                body: &body_bytes,
                extra_headers: &[("content-type".into(), "application/json".into())],
                region: &self.cfg.region,
                service: "bedrock",
                now: Utc::now(),
            },
        );

        debug!(model = %self.cfg.model_id, "bedrock stream");
        let mut req = self
            .http
            .post(&url)
            .header("content-type", "application/json")
            .header("authorization", &signed.authorization)
            .header("x-amz-date", &signed.x_amz_date)
            .header("x-amz-content-sha256", &signed.x_amz_content_sha256)
            .body(body_bytes);
        if let Some(tok) = &signed.x_amz_security_token {
            req = req.header("x-amz-security-token", tok);
        }
        let resp = req.send().await.map_err(|e| Error::provider(format!("send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::provider(format!("bedrock stream {status}: {txt}")));
        }

        let model_name = self.cfg.model_id.clone();
        let mut bytes_stream = resp.bytes_stream();

        let stream = async_stream::try_stream! {
            let mut buf: Vec<u8> = Vec::with_capacity(8 * 1024);
            let mut agg_text = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut tool_arg_bufs: Vec<String> = Vec::new();
            let mut usage = TokenUsage::default();
            let mut finish = FinishReason::Stop;

            while let Some(chunk) = bytes_stream.next().await {
                let chunk = chunk.map_err(|e| Error::provider(format!("stream chunk: {e}")))?;
                buf.extend_from_slice(&chunk);

                // Drain as many complete frames as the buffer holds.
                loop {
                    match parse_frame(&buf) {
                        Err(e) => Err(Error::provider(format!("event-stream parse: {e:?}")))?,
                        Ok(None) => break,
                        Ok(Some((frame, consumed))) => {
                            let _drained: Vec<u8> = buf.drain(..consumed).collect();
                            let event_type = frame.headers.get(":event-type").cloned().unwrap_or_default();
                            if event_type != "chunk" { continue; }
                            // Outer payload: { "bytes": "<base64>" }
                            let outer: Value = serde_json::from_slice(&frame.payload)
                                .map_err(|e| Error::provider(format!("chunk outer json: {e}")))?;
                            let b64 = outer.get("bytes").and_then(|v| v.as_str())
                                .ok_or_else(|| Error::provider("chunk missing bytes"))?;
                            let decoded = base64::engine::general_purpose::STANDARD
                                .decode(b64)
                                .map_err(|e| Error::provider(format!("chunk base64: {e}")))?;
                            // Inner: Anthropic-on-Bedrock SSE event shape
                            let v: Value = serde_json::from_slice(&decoded)
                                .map_err(|e| Error::provider(format!("inner json: {e}")))?;
                            let etype = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
                            match etype {
                                "content_block_start" => {
                                    if let Some(block) = v.get("content_block") {
                                        if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                                            tool_calls.push(ToolCall {
                                                id: block.get("id").and_then(|v| v.as_str()).unwrap_or("").into(),
                                                name: block.get("name").and_then(|v| v.as_str()).unwrap_or("").into(),
                                                arguments: Value::Null,
                                            });
                                            tool_arg_bufs.push(String::new());
                                        }
                                    }
                                }
                                "content_block_delta" => {
                                    if let Some(delta) = v.get("delta") {
                                        match delta.get("type").and_then(|t| t.as_str()) {
                                            Some("text_delta") => {
                                                if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
                                                    if !t.is_empty() {
                                                        agg_text.push_str(t);
                                                        yield ChatStreamEvent::Delta { text: t.into() };
                                                    }
                                                }
                                            }
                                            Some("input_json_delta") => {
                                                if let Some(idx) = v.get("index").and_then(|v| v.as_u64()) {
                                                    let idx = idx as usize;
                                                    if let Some(b) = tool_arg_bufs.get_mut(idx) {
                                                        if let Some(pj) = delta.get("partial_json").and_then(|v| v.as_str()) {
                                                            b.push_str(pj);
                                                            yield ChatStreamEvent::ToolCallDelta {
                                                                index: idx as u32, id: None, name: None,
                                                                arguments_delta: Some(pj.into()),
                                                            };
                                                        }
                                                    }
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                "message_delta" => {
                                    if let Some(d) = v.get("delta") {
                                        if let Some(sr) = d.get("stop_reason").and_then(|v| v.as_str()) {
                                            finish = match sr {
                                                "end_turn" => FinishReason::Stop,
                                                "max_tokens" => FinishReason::Length,
                                                "tool_use" => FinishReason::ToolCalls,
                                                _ => FinishReason::Other,
                                            };
                                        }
                                    }
                                    if let Some(u) = v.get("usage") {
                                        usage.completion = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                    }
                                }
                                "message_start" => {
                                    if let Some(m) = v.get("message") {
                                        if let Some(u) = m.get("usage") {
                                            usage.prompt = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // Finalize aggregated tool calls.
            for (i, b) in tool_arg_bufs.iter().enumerate() {
                if !b.is_empty() {
                    tool_calls[i].arguments = serde_json::from_str(b).unwrap_or(Value::String(b.clone()));
                }
            }
            usage.total = usage.prompt + usage.completion;
            let msg = Message {
                role: Role::Assistant,
                content: if agg_text.is_empty() { vec![] } else { vec![ContentPart::Text { text: agg_text.clone() }] },
                tool_calls,
                tool_call_id: None,
                name: None,
                cache: false,
            };
            yield ChatStreamEvent::Done {
                response: ChatResponse { message: msg, finish_reason: finish, usage, model: model_name },
            };
        };

        Ok(Box::pin(stream.map_err(|e: Error| e)))
    }
}


fn urlencoded(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' | '/' => c.to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect()
}

fn add_cache_breakpoint(blocks: &mut [Value]) {
    if let Some(last) = blocks.last_mut() {
        if let Some(obj) = last.as_object_mut() {
            obj.insert("cache_control".into(), json!({"type": "ephemeral"}));
        }
    }
}

fn message_to_anthropic(m: &Message) -> Value {
    let role = match m.role {
        Role::User | Role::System => "user",
        Role::Assistant => "assistant",
        Role::Tool => "user",
    };

    if matches!(m.role, Role::Tool) {
        let mut blocks = vec![json!({
            "type": "tool_result",
            "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
            "content": m.text_content(),
        })];
        if m.cache { add_cache_breakpoint(&mut blocks); }
        return json!({"role": "user", "content": blocks});
    }

    if matches!(m.role, Role::Assistant) && !m.tool_calls.is_empty() {
        let mut blocks = Vec::new();
        if !m.text_content().is_empty() {
            blocks.push(json!({"type": "text", "text": m.text_content()}));
        }
        for tc in &m.tool_calls {
            blocks.push(json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            }));
        }
        if m.cache { add_cache_breakpoint(&mut blocks); }
        return json!({"role": role, "content": blocks});
    }

    let mut blocks: Vec<Value> = if m.content.iter().all(|p| matches!(p, ContentPart::Text { .. })) {
        vec![json!({"type": "text", "text": m.text_content()})]
    } else {
        m.content.iter().map(|p| match p {
            ContentPart::Text { text } => json!({"type": "text", "text": text}),
            ContentPart::Image { source } => match source {
                litgraph_core::ImageSource::Url { url } => json!({
                    "type": "image",
                    "source": {"type": "url", "url": url}
                }),
                litgraph_core::ImageSource::Base64 { media_type, data } => json!({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data}
                }),
            },
        }).collect()
    };
    if m.cache { add_cache_breakpoint(&mut blocks); }
    json!({"role": role, "content": Value::Array(blocks)})
}

fn parse_response(model: &str, v: Value) -> Result<ChatResponse> {
    let blocks = v
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or_else(|| Error::provider("bedrock: missing content blocks"))?;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for b in blocks {
        match b.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = b.get("text").and_then(|v| v.as_str()) { text.push_str(t); }
            }
            Some("tool_use") => {
                tool_calls.push(ToolCall {
                    id: b.get("id").and_then(|v| v.as_str()).unwrap_or("").into(),
                    name: b.get("name").and_then(|v| v.as_str()).unwrap_or("").into(),
                    arguments: b.get("input").cloned().unwrap_or(Value::Null),
                });
            }
            _ => {}
        }
    }
    let finish = match v.get("stop_reason").and_then(|v| v.as_str()).unwrap_or("end_turn") {
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        _ => FinishReason::Other,
    };
    let usage = v
        .get("usage")
        .map(|u| {
            let prompt = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let completion = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let cache_creation = u.get("cache_creation_input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let cache_read = u.get("cache_read_input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            TokenUsage { prompt, completion, total: prompt + completion, cache_creation, cache_read }
        })
        .unwrap_or_default();

    Ok(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: if text.is_empty() { vec![] } else { vec![ContentPart::Text { text }] },
            tool_calls,
            tool_call_id: None,
            name: None,
            cache: false,
        },
        finish_reason: finish,
        usage,
        model: model.to_string(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Bedrock Embeddings (Titan Embed family)
// ─────────────────────────────────────────────────────────────────────────────

/// Wire format dispatch. Titan = single `inputText` per call (we parallelize
/// for batches). Cohere-on-Bedrock = batched `texts` array per call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BedrockEmbedFormat {
    /// `amazon.titan-embed-text-v2:0`, `amazon.titan-embed-text-v1` etc.
    /// Body: `{ "inputText": "...", "dimensions": N, "normalize": true }`.
    /// Response: `{ "embedding": [...] }`.
    Titan,
    /// `cohere.embed-english-v3`, `cohere.embed-multilingual-v3`.
    /// Body: `{ "texts": [...], "input_type": "...", "embedding_types": ["float"] }`.
    /// Response: `{ "embeddings": [[...], ...] }` for v1 or
    /// `{ "embeddings": { "float": [[...]] } }` when embedding_types is set.
    Cohere,
}

impl BedrockEmbedFormat {
    /// Heuristic from the model id. Override at config time if wrong.
    fn detect(model_id: &str) -> Self {
        let m = model_id.to_ascii_lowercase();
        if m.starts_with("cohere.") { Self::Cohere } else { Self::Titan }
    }
}

#[derive(Clone, Debug)]
pub struct BedrockEmbeddingsConfig {
    pub credentials: AwsCredentials,
    pub region: String,
    pub model_id: String,
    pub timeout: Duration,
    pub dimensions: usize,
    /// Optional URL prefix override (PrivateLink / fake server). When set, host
    /// is derived from this URL and SigV4 signs against that host.
    pub endpoint_override: Option<String>,
    /// Wire format. Defaults from `model_id` (cohere.* → Cohere, else Titan).
    pub format: BedrockEmbedFormat,
    /// Max parallel requests for Titan-style providers (one HTTP call per
    /// input). Bounded to avoid exhausting the connection pool. Ignored for
    /// Cohere since it batches. Defaults to 8.
    pub max_concurrency: usize,
    /// Titan-only: send `normalize: true` (recommended for cosine sim).
    pub normalize: bool,
    /// Cohere-on-Bedrock: per-request `input_type`. Same vocabulary as the
    /// direct Cohere API (`search_document` / `search_query` / `classification`
    /// / `clustering`).
    pub cohere_input_type_document: String,
    pub cohere_input_type_query: String,
}

impl BedrockEmbeddingsConfig {
    pub fn new(
        creds: AwsCredentials,
        region: impl Into<String>,
        model_id: impl Into<String>,
        dimensions: usize,
    ) -> Self {
        let model_id = model_id.into();
        let format = BedrockEmbedFormat::detect(&model_id);
        Self {
            credentials: creds,
            region: region.into(),
            model_id,
            timeout: Duration::from_secs(120),
            dimensions,
            endpoint_override: None,
            format,
            max_concurrency: 8,
            normalize: true,
            cohere_input_type_document: "search_document".into(),
            cohere_input_type_query: "search_query".into(),
        }
    }
    pub fn with_endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint_override = Some(url.into());
        self
    }
    pub fn with_format(mut self, f: BedrockEmbedFormat) -> Self {
        self.format = f;
        self
    }
    pub fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self
    }
}

pub struct BedrockEmbeddings {
    cfg: BedrockEmbeddingsConfig,
    http: Client,
}

impl BedrockEmbeddings {
    pub fn new(cfg: BedrockEmbeddingsConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    fn endpoint_host_base(&self) -> (String, String) {
        // Reuse the chat endpoint resolver semantics by faking a BedrockConfig.
        let synthetic = BedrockConfig {
            credentials: self.cfg.credentials.clone(),
            region: self.cfg.region.clone(),
            model_id: self.cfg.model_id.clone(),
            timeout: self.cfg.timeout,
            max_tokens: 1,
            endpoint_override: self.cfg.endpoint_override.clone(),
            on_request: None,
        };
        resolve_endpoint(&synthetic)
    }

    async fn invoke_signed(&self, body: &Value) -> Result<Value> {
        let (host, base) = self.endpoint_host_base();
        let path = format!("/model/{}/invoke", urlencoded(&self.cfg.model_id));
        let url = format!("{base}{path}");
        let body_bytes = serde_json::to_vec(body).map_err(Error::from)?;
        let signed = sigv4::sign(
            &self.cfg.credentials,
            &sigv4::SigningInputs {
                method: "POST",
                host: &host,
                path: &path,
                query: "",
                body: &body_bytes,
                extra_headers: &[("content-type".into(), "application/json".into())],
                region: &self.cfg.region,
                service: "bedrock",
                now: Utc::now(),
            },
        );
        let mut req = self
            .http
            .post(&url)
            .header("content-type", "application/json")
            .header("authorization", &signed.authorization)
            .header("x-amz-date", &signed.x_amz_date)
            .header("x-amz-content-sha256", &signed.x_amz_content_sha256)
            .body(body_bytes);
        if let Some(tok) = &signed.x_amz_security_token {
            req = req.header("x-amz-security-token", tok);
        }
        let resp = req.send().await.map_err(|e| Error::provider(format!("send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(Error::RateLimited { retry_after_ms: None });
            }
            return Err(Error::provider(format!("bedrock embed {status}: {txt}")));
        }
        resp.json().await.map_err(|e| Error::provider(format!("decode: {e}")))
    }

    async fn embed_titan_one(&self, text: &str) -> Result<Vec<f32>> {
        let body = json!({
            "inputText": text,
            "dimensions": self.cfg.dimensions,
            "normalize": self.cfg.normalize,
        });
        let v = self.invoke_signed(&body).await?;
        let arr = v
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| Error::provider("titan embed: missing `embedding`"))?;
        Ok(arr.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect())
    }

    async fn embed_titan_batch(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        use futures_util::stream::{self, StreamExt};
        let limit = self.cfg.max_concurrency.max(1);
        // FuturesOrdered would also work; buffered() preserves input order.
        let outputs: Vec<Result<Vec<f32>>> = stream::iter(inputs.into_iter())
            .map(|t| async move { self.embed_titan_one(&t).await })
            .buffered(limit)
            .collect()
            .await;
        outputs.into_iter().collect()
    }

    async fn embed_cohere_batch(&self, inputs: Vec<String>, input_type: &str) -> Result<Vec<Vec<f32>>> {
        let body = json!({
            "texts": inputs,
            "input_type": input_type,
            "embedding_types": ["float"],
        });
        let v = self.invoke_signed(&body).await?;
        // Two response shapes are possible: flat `embeddings: [[...]]` (older
        // models, or no embedding_types) and nested `embeddings.float[][]` when
        // embedding_types was sent. Handle both.
        let arr = v
            .get("embeddings")
            .and_then(|e| e.get("float").and_then(|f| f.as_array()).or_else(|| e.as_array()))
            .ok_or_else(|| Error::provider("cohere bedrock embed: missing `embeddings`"))?;
        let mut out = Vec::with_capacity(arr.len());
        for row in arr {
            let r = row
                .as_array()
                .ok_or_else(|| Error::provider("cohere bedrock embed: row not array"))?;
            out.push(r.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect());
        }
        Ok(out)
    }
}

#[async_trait]
impl Embeddings for BedrockEmbeddings {
    fn name(&self) -> &str { &self.cfg.model_id }
    fn dimensions(&self) -> usize { self.cfg.dimensions }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        match self.cfg.format {
            BedrockEmbedFormat::Titan => self.embed_titan_one(text).await,
            BedrockEmbedFormat::Cohere => {
                let it = self.cfg.cohere_input_type_query.clone();
                let mut out = self.embed_cohere_batch(vec![text.to_string()], &it).await?;
                out.pop().ok_or_else(|| Error::provider("embed_query: empty result"))
            }
        }
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(vec![]); }
        match self.cfg.format {
            BedrockEmbedFormat::Titan => self.embed_titan_batch(texts.to_vec()).await,
            BedrockEmbedFormat::Cohere => {
                let it = self.cfg.cohere_input_type_document.clone();
                self.embed_cohere_batch(texts.to_vec(), &it).await
            }
        }
    }
}

// ============================================================================
// BedrockConverseChat — unified Bedrock Converse API (iter 99)
// ============================================================================
//
// AWS's Converse API normalizes chat across model families (Anthropic, Llama,
// Titan, Mistral, Command, Nova). One request/response shape regardless of
// which model's foundation. Great for apps that want to swap model families
// without rewriting wire-format code.
//
// Wire shape (POST /model/{id}/converse):
// - Request: {messages:[...], system:[...]?, inferenceConfig:{maxTokens,
//            temperature, topP, stopSequences}?, toolConfig:{tools:[...]}?}
// - Response: {output:{message:{role:"assistant", content:[{text,toolUse...}]}},
//              stopReason, usage:{inputTokens, outputTokens, totalTokens}}
// - Content blocks: {text:"..."} for plain text, {toolUse:{toolUseId,name,input}}
//   for function calls, {toolResult:{...}} for tool replies (reverse direction).
//
// This iter ships invoke only — streaming (`/converse-stream`) uses the same
// AWS event-stream framing as iter-0 Anthropic streaming but with different
// event shapes (contentBlockStart/Delta/Stop, messageStart/Stop, metadata).
// Added in a future iter.

pub struct BedrockConverseChat {
    cfg: BedrockConfig,
    http: Client,
}

impl BedrockConverseChat {
    pub fn new(cfg: BedrockConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    /// Build a Converse API request body from messages + options. System
    /// messages go into a separate top-level `system` array; tool_use /
    /// tool_result messages flow into assistant / user message content
    /// blocks per the Converse content-block spec.
    pub(crate) fn body(&self, messages: &[Message], opts: &ChatOptions) -> Value {
        let (system, rest): (Vec<&Message>, Vec<&Message>) = messages
            .iter()
            .partition(|m| matches!(m.role, Role::System));

        let mut body = json!({
            "messages": rest
                .iter()
                .map(|m| message_to_converse(m))
                .collect::<Vec<_>>(),
        });

        if !system.is_empty() {
            body["system"] = Value::Array(
                system
                    .iter()
                    .filter_map(|m| {
                        let t = m.text_content();
                        if t.is_empty() {
                            None
                        } else {
                            Some(json!({"text": t}))
                        }
                    })
                    .collect(),
            );
        }

        let mut inference = serde_json::Map::new();
        let max = opts.max_tokens.unwrap_or(self.cfg.max_tokens);
        inference.insert("maxTokens".into(), json!(max));
        if let Some(t) = opts.temperature {
            inference.insert("temperature".into(), json!(t));
        }
        if let Some(t) = opts.top_p {
            inference.insert("topP".into(), json!(t));
        }
        if let Some(ref s) = opts.stop {
            inference.insert("stopSequences".into(), json!(s));
        }
        body["inferenceConfig"] = Value::Object(inference);

        if !opts.tools.is_empty() {
            body["toolConfig"] = json!({
                "tools": opts.tools.iter().map(|t| json!({
                    "toolSpec": {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": { "json": t.parameters },
                    }
                })).collect::<Vec<_>>(),
            });
        }
        body
    }

    async fn invoke_inner(&self, body: &Value) -> Result<Value> {
        if let Some(cb) = &self.cfg.on_request {
            cb(&self.cfg.model_id, body);
        }
        let (host, base) = resolve_endpoint(&self.cfg);
        let path = format!("/model/{}/converse", urlencoded(&self.cfg.model_id));
        let url = format!("{base}{path}");
        let body_bytes = serde_json::to_vec(body).map_err(Error::from)?;

        let signed = sigv4::sign(
            &self.cfg.credentials,
            &sigv4::SigningInputs {
                method: "POST",
                host: &host,
                path: &path,
                query: "",
                body: &body_bytes,
                extra_headers: &[("content-type".into(), "application/json".into())],
                region: &self.cfg.region,
                service: "bedrock",
                now: Utc::now(),
            },
        );

        debug!(model = %self.cfg.model_id, host, "bedrock converse invoke");
        let mut req = self
            .http
            .post(&url)
            .header("content-type", "application/json")
            .header("authorization", &signed.authorization)
            .header("x-amz-date", &signed.x_amz_date)
            .header("x-amz-content-sha256", &signed.x_amz_content_sha256)
            .body(body_bytes);
        if let Some(tok) = &signed.x_amz_security_token {
            req = req.header("x-amz-security-token", tok);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::provider(format!("send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(Error::RateLimited { retry_after_ms: None });
            }
            return Err(Error::provider(format!("{status}: {txt}")));
        }
        resp.json()
            .await
            .map_err(|e| Error::provider(format!("decode: {e}")))
    }
}

#[async_trait]
impl ChatModel for BedrockConverseChat {
    fn name(&self) -> &str {
        &self.cfg.model_id
    }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let body = self.body(&messages, opts);
        let v = self.invoke_inner(&body).await?;
        parse_converse_response(&self.cfg.model_id, v)
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        let body = self.body(&messages, opts);
        if let Some(cb) = &self.cfg.on_request {
            cb(&self.cfg.model_id, &body);
        }
        let (host, base) = resolve_endpoint(&self.cfg);
        let path = format!("/model/{}/converse-stream", urlencoded(&self.cfg.model_id));
        let url = format!("{base}{path}");
        let body_bytes = serde_json::to_vec(&body).map_err(Error::from)?;

        let signed = sigv4::sign(
            &self.cfg.credentials,
            &sigv4::SigningInputs {
                method: "POST",
                host: &host,
                path: &path,
                query: "",
                body: &body_bytes,
                extra_headers: &[("content-type".into(), "application/json".into())],
                region: &self.cfg.region,
                service: "bedrock",
                now: Utc::now(),
            },
        );

        debug!(model = %self.cfg.model_id, "bedrock converse stream");
        let mut req = self
            .http
            .post(&url)
            .header("content-type", "application/json")
            .header("authorization", &signed.authorization)
            .header("x-amz-date", &signed.x_amz_date)
            .header("x-amz-content-sha256", &signed.x_amz_content_sha256)
            .body(body_bytes);
        if let Some(tok) = &signed.x_amz_security_token {
            req = req.header("x-amz-security-token", tok);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::provider(format!("send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::provider(format!(
                "bedrock converse stream {status}: {txt}"
            )));
        }

        let model_name = self.cfg.model_id.clone();
        let mut bytes_stream = resp.bytes_stream();

        // Converse stream events — differ from Anthropic-on-Bedrock!
        // Frame payload is the JSON event directly (no nested base64 `bytes`
        // wrapper like invoke-with-response-stream uses). Event-type header:
        //   messageStart          → {role}
        //   contentBlockStart     → {start: {toolUse: {toolUseId, name}}, contentBlockIndex}
        //   contentBlockDelta     → {delta: {text | toolUse.input}, contentBlockIndex}
        //   contentBlockStop      → {contentBlockIndex}
        //   messageStop           → {stopReason}
        //   metadata              → {usage: {inputTokens, outputTokens, totalTokens}}
        let stream = async_stream::try_stream! {
            let mut buf: Vec<u8> = Vec::with_capacity(8 * 1024);
            let mut agg_text = String::new();
            // Each contentBlockIndex maps to at most one tool_use accumulator.
            // We store by index so out-of-order blocks don't clobber each other.
            let mut tool_calls_by_index: std::collections::BTreeMap<u64, ToolCall> =
                std::collections::BTreeMap::new();
            let mut tool_arg_bufs: std::collections::BTreeMap<u64, String> =
                std::collections::BTreeMap::new();
            let mut usage = TokenUsage::default();
            let mut finish = FinishReason::Stop;

            while let Some(chunk) = bytes_stream.next().await {
                let chunk = chunk.map_err(|e| Error::provider(format!("stream chunk: {e}")))?;
                buf.extend_from_slice(&chunk);

                loop {
                    match parse_frame(&buf) {
                        Err(e) => Err(Error::provider(format!("event-stream parse: {e:?}")))?,
                        Ok(None) => break,
                        Ok(Some((frame, consumed))) => {
                            let _drained: Vec<u8> = buf.drain(..consumed).collect();
                            let event_type = frame
                                .headers
                                .get(":event-type")
                                .cloned()
                                .unwrap_or_default();
                            // :message-type header is "event" for payload frames,
                            // "exception" for error frames. Surface exceptions
                            // loudly — they typically carry ValidationException
                            // / ThrottlingException / ModelNotReadyException.
                            let msg_type = frame
                                .headers
                                .get(":message-type")
                                .cloned()
                                .unwrap_or_default();
                            if msg_type == "exception" {
                                let payload = String::from_utf8_lossy(&frame.payload).to_string();
                                Err(Error::provider(format!(
                                    "bedrock converse stream {event_type}: {payload}"
                                )))?;
                            }
                            // Parse the JSON payload directly — no nested base64.
                            let v: Value = serde_json::from_slice(&frame.payload)
                                .map_err(|e| Error::provider(format!(
                                    "converse event json ({event_type}): {e}"
                                )))?;
                            match event_type.as_str() {
                                "messageStart" => {
                                    // {"role": "assistant"} — nothing to do
                                    // (we always set Assistant on the final msg).
                                }
                                "contentBlockStart" => {
                                    // Start of a new block. If it's a toolUse
                                    // block, stash id + name at this index.
                                    let idx = v
                                        .get("contentBlockIndex")
                                        .and_then(|n| n.as_u64())
                                        .unwrap_or(0);
                                    if let Some(tu) = v.pointer("/start/toolUse") {
                                        let id = tu
                                            .get("toolUseId")
                                            .and_then(|s| s.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let name = tu
                                            .get("name")
                                            .and_then(|s| s.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        tool_calls_by_index.insert(idx, ToolCall {
                                            id: id.clone(),
                                            name: name.clone(),
                                            arguments: Value::Null,
                                        });
                                        tool_arg_bufs.insert(idx, String::new());
                                        yield ChatStreamEvent::ToolCallDelta {
                                            index: idx as u32,
                                            id: Some(id),
                                            name: Some(name),
                                            arguments_delta: None,
                                        };
                                    }
                                }
                                "contentBlockDelta" => {
                                    let idx = v
                                        .get("contentBlockIndex")
                                        .and_then(|n| n.as_u64())
                                        .unwrap_or(0);
                                    if let Some(delta) = v.get("delta") {
                                        // Text delta.
                                        if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
                                            if !t.is_empty() {
                                                agg_text.push_str(t);
                                                yield ChatStreamEvent::Delta { text: t.into() };
                                            }
                                        }
                                        // Tool-use input delta (partial JSON).
                                        if let Some(tu) = delta.get("toolUse") {
                                            if let Some(pj) = tu
                                                .get("input")
                                                .and_then(|v| v.as_str())
                                            {
                                                if let Some(b) = tool_arg_bufs.get_mut(&idx) {
                                                    b.push_str(pj);
                                                    yield ChatStreamEvent::ToolCallDelta {
                                                        index: idx as u32,
                                                        id: None,
                                                        name: None,
                                                        arguments_delta: Some(pj.into()),
                                                    };
                                                }
                                            }
                                        }
                                    }
                                }
                                "contentBlockStop" => {
                                    // Finalize an accumulator block. For tool_use,
                                    // parse the buffered partial-json → Value.
                                    let idx = v
                                        .get("contentBlockIndex")
                                        .and_then(|n| n.as_u64())
                                        .unwrap_or(0);
                                    if let Some(buf) = tool_arg_bufs.get(&idx) {
                                        if let Some(tc) = tool_calls_by_index.get_mut(&idx) {
                                            tc.arguments = serde_json::from_str(buf)
                                                .unwrap_or(Value::String(buf.clone()));
                                        }
                                    }
                                }
                                "messageStop" => {
                                    if let Some(sr) = v
                                        .get("stopReason")
                                        .and_then(|v| v.as_str())
                                    {
                                        finish = match sr {
                                            "end_turn" => FinishReason::Stop,
                                            "tool_use" => FinishReason::ToolCalls,
                                            "max_tokens" => FinishReason::Length,
                                            "content_filtered" => FinishReason::ContentFilter,
                                            "stop_sequence" => FinishReason::Stop,
                                            _ => FinishReason::Other,
                                        };
                                    }
                                }
                                "metadata" => {
                                    if let Some(u) = v.get("usage") {
                                        usage.prompt = u
                                            .get("inputTokens")
                                            .and_then(|n| n.as_u64())
                                            .unwrap_or(0) as u32;
                                        usage.completion = u
                                            .get("outputTokens")
                                            .and_then(|n| n.as_u64())
                                            .unwrap_or(0) as u32;
                                        usage.total = u
                                            .get("totalTokens")
                                            .and_then(|n| n.as_u64())
                                            .unwrap_or(0) as u32;
                                    }
                                }
                                _ => {
                                    // Unknown event type — forward-compat silent
                                    // skip, consistent with how the iter-0
                                    // Anthropic path handles future event kinds.
                                }
                            }
                        }
                    }
                }
            }

            // Assemble final message — tool_calls in stable index order.
            let tool_calls: Vec<ToolCall> = tool_calls_by_index.into_values().collect();
            let content = if agg_text.is_empty() {
                vec![]
            } else {
                vec![ContentPart::Text { text: agg_text.clone() }]
            };
            yield ChatStreamEvent::Done {
                response: ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content,
                        tool_calls,
                        tool_call_id: None,
                        name: None,
                        cache: false,
                    },
                    finish_reason: finish,
                    usage,
                    model: model_name,
                },
            };
        };

        Ok(Box::pin(stream.map_err(|e: Error| e)))
    }
}

/// Convert a `Message` to a Converse API content-block message.
fn message_to_converse(m: &Message) -> Value {
    let role = match m.role {
        Role::Assistant => "assistant",
        _ => "user",
    };
    let mut content: Vec<Value> = Vec::new();

    // Tool-role messages become a `toolResult` content block under `user`
    // (per Converse: tool outputs are injected as user-turn content).
    if matches!(m.role, Role::Tool) {
        let call_id = m.tool_call_id.clone().unwrap_or_default();
        let text = m.text_content();
        content.push(json!({
            "toolResult": {
                "toolUseId": call_id,
                "content": [{"text": text}],
                // Converse lets callers mark error vs success — we leave it
                // unset (success) since the tool adapter surfaces errors
                // via Tool::run's Err path, not as a role:tool message.
            }
        }));
    } else {
        // Text parts.
        let text = m.text_content();
        if !text.is_empty() {
            content.push(json!({ "text": text }));
        }
        // Assistant turns with tool_calls → `toolUse` content blocks.
        for tc in &m.tool_calls {
            content.push(json!({
                "toolUse": {
                    "toolUseId": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            }));
        }
    }

    json!({ "role": role, "content": content })
}

fn parse_converse_response(model: &str, v: Value) -> Result<ChatResponse> {
    let output = v
        .get("output")
        .ok_or_else(|| Error::provider("converse: missing output"))?;
    let msg = output
        .get("message")
        .ok_or_else(|| Error::provider("converse: missing output.message"))?;
    let content = msg
        .get("content")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();

    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    for block in &content {
        if let Some(t) = block.get("text").and_then(|x| x.as_str()) {
            text.push_str(t);
        } else if let Some(tu) = block.get("toolUse") {
            let id = tu
                .get("toolUseId")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let name = tu
                .get("name")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let arguments = tu.get("input").cloned().unwrap_or(Value::Null);
            tool_calls.push(ToolCall { id, name, arguments });
        }
    }

    let stop_reason = v
        .get("stopReason")
        .and_then(|s| s.as_str())
        .unwrap_or("end_turn");
    let finish = match stop_reason {
        "end_turn" => FinishReason::Stop,
        "tool_use" => FinishReason::ToolCalls,
        "max_tokens" => FinishReason::Length,
        "content_filtered" => FinishReason::ContentFilter,
        "stop_sequence" => FinishReason::Stop,
        _ => FinishReason::Other,
    };

    // Usage: Converse uses `inputTokens` / `outputTokens` (no underscore).
    let usage = v
        .get("usage")
        .map(|u| TokenUsage {
            prompt: u.get("inputTokens").and_then(|n| n.as_u64()).unwrap_or(0) as u32,
            completion: u.get("outputTokens").and_then(|n| n.as_u64()).unwrap_or(0) as u32,
            total: u.get("totalTokens").and_then(|n| n.as_u64()).unwrap_or(0) as u32,
            cache_creation: 0,
            cache_read: 0,
        })
        .unwrap_or_default();

    let content_parts = if text.is_empty() {
        vec![]
    } else {
        vec![ContentPart::Text { text }]
    };

    Ok(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: content_parts,
            tool_calls,
            tool_call_id: None,
            name: None,
            cache: false,
        },
        finish_reason: finish,
        usage,
        model: model.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_endpoint_default_uses_aws() {
        let cfg = BedrockConfig::new(
            AwsCredentials { access_key_id: "x".into(), secret_access_key: "y".into(), session_token: None },
            "us-west-2",
            "anthropic.claude-opus-4-7-v1:0",
        );
        let (host, base) = resolve_endpoint(&cfg);
        assert_eq!(host, "bedrock-runtime.us-west-2.amazonaws.com");
        assert_eq!(base, "https://bedrock-runtime.us-west-2.amazonaws.com");
    }

    #[test]
    fn resolve_endpoint_strips_scheme_and_path_for_host() {
        let cfg = BedrockConfig::new(
            AwsCredentials { access_key_id: "x".into(), secret_access_key: "y".into(), session_token: None },
            "us-east-1",
            "anthropic.claude-opus-4-7-v1:0",
        ).with_endpoint("http://127.0.0.1:8080/proxy/");
        let (host, base) = resolve_endpoint(&cfg);
        assert_eq!(host, "127.0.0.1:8080");
        assert_eq!(base, "http://127.0.0.1:8080/proxy");
    }

    #[test]
    fn body_pins_bedrock_anthropic_version() {
        let chat = BedrockChat::new(BedrockConfig::new(
            AwsCredentials {
                access_key_id: "AKID".into(),
                secret_access_key: "secret".into(),
                session_token: None,
            },
            "us-east-1",
            "anthropic.claude-opus-4-7-v1:0",
        )).unwrap();
        let body = chat.body(&[Message::user("hi")], &ChatOptions::default());
        assert_eq!(body["anthropic_version"], json!(BEDROCK_ANTHROPIC_VERSION));
        assert_eq!(body["messages"][0]["role"], json!("user"));
    }

    #[test]
    fn embed_format_detect_titan_vs_cohere() {
        assert_eq!(BedrockEmbedFormat::detect("amazon.titan-embed-text-v2:0"), BedrockEmbedFormat::Titan);
        assert_eq!(BedrockEmbedFormat::detect("amazon.titan-embed-text-v1"), BedrockEmbedFormat::Titan);
        assert_eq!(BedrockEmbedFormat::detect("cohere.embed-english-v3"), BedrockEmbedFormat::Cohere);
        assert_eq!(BedrockEmbedFormat::detect("cohere.embed-multilingual-v3"), BedrockEmbedFormat::Cohere);
        assert_eq!(BedrockEmbedFormat::detect("Cohere.Embed-Multi"), BedrockEmbedFormat::Cohere);
    }

    #[test]
    fn embed_config_defaults_titan_normalize_and_concurrency() {
        let cfg = BedrockEmbeddingsConfig::new(
            AwsCredentials { access_key_id: "AKID".into(), secret_access_key: "s".into(), session_token: None },
            "us-east-1",
            "amazon.titan-embed-text-v2:0",
            1024,
        );
        assert!(cfg.normalize);
        assert_eq!(cfg.format, BedrockEmbedFormat::Titan);
        assert_eq!(cfg.max_concurrency, 8);
        let cfg = cfg.with_max_concurrency(0);
        assert_eq!(cfg.max_concurrency, 1, "0 must clamp to 1");
    }

    // ---------- BedrockConverseChat (iter 99) ----------

    use litgraph_core::tool::ToolSchema;

    fn test_creds() -> AwsCredentials {
        AwsCredentials {
            access_key_id: "AKIAEXAMPLE".into(),
            secret_access_key: "secretEXAMPLE".into(),
            session_token: None,
        }
    }

    fn chat(model_id: &str) -> BedrockConverseChat {
        let cfg = BedrockConfig::new(test_creds(), "us-east-1", model_id);
        BedrockConverseChat::new(cfg).unwrap()
    }

    #[test]
    fn converse_body_flattens_system_to_top_level_array() {
        let c = chat("amazon.titan-text-lite-v1");
        let body = c.body(
            &[
                Message::system("You are terse."),
                Message::user("hi"),
            ],
            &ChatOptions::default(),
        );
        // System lives at top-level `system`, NOT inside `messages`.
        let sys = body["system"].as_array().unwrap();
        assert_eq!(sys.len(), 1);
        assert_eq!(sys[0]["text"].as_str(), Some("You are terse."));
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"].as_str(), Some("user"));
    }

    #[test]
    fn converse_body_maps_temperature_topp_stop_to_inference_config() {
        let c = chat("meta.llama3-70b-instruct-v1:0");
        let opts = ChatOptions {
            temperature: Some(0.3),
            top_p: Some(0.9),
            max_tokens: Some(128),
            stop: Some(vec!["END".into(), "STOP".into()]),
            ..Default::default()
        };
        let body = c.body(&[Message::user("x")], &opts);
        let ic = &body["inferenceConfig"];
        assert_eq!(ic["maxTokens"], 128);
        // Floats round-trip through serde_json — tolerance compare.
        let t = ic["temperature"].as_f64().unwrap();
        assert!((t - 0.3).abs() < 1e-6);
        let p = ic["topP"].as_f64().unwrap();
        assert!((p - 0.9).abs() < 1e-6);
        assert_eq!(ic["stopSequences"], json!(["END", "STOP"]));
    }

    #[test]
    fn converse_body_tools_nest_under_tool_config_with_tool_spec() {
        // Converse API wraps each tool as `{toolSpec: {name, description,
        // inputSchema: {json: {...}}}}` — NOT flat like chat completions.
        let c = chat("mistral.mistral-large-2407-v1:0");
        let opts = ChatOptions {
            tools: vec![ToolSchema {
                name: "lookup".into(),
                description: "Look it up.".into(),
                parameters: json!({"type":"object","properties":{"q":{"type":"string"}}}),
            }],
            ..Default::default()
        };
        let body = c.body(&[Message::user("x")], &opts);
        let tools = body["toolConfig"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["toolSpec"]["name"], "lookup");
        assert_eq!(tools[0]["toolSpec"]["description"], "Look it up.");
        assert!(tools[0]["toolSpec"]["inputSchema"]["json"].is_object());
    }

    #[test]
    fn converse_body_has_no_tool_config_when_opts_tools_empty() {
        let c = chat("amazon.nova-lite-v1:0");
        let body = c.body(&[Message::user("x")], &ChatOptions::default());
        assert!(body.get("toolConfig").is_none());
    }

    #[test]
    fn message_to_converse_emits_text_plus_tool_use_content_blocks() {
        let assistant_with_tool_calls = Message {
            role: Role::Assistant,
            content: vec![ContentPart::Text {
                text: "Let me search.".into(),
            }],
            tool_calls: vec![ToolCall {
                id: "tu_1".into(),
                name: "search".into(),
                arguments: json!({"q": "rust async"}),
            }],
            tool_call_id: None,
            name: None,
            cache: false,
        };
        let v = message_to_converse(&assistant_with_tool_calls);
        assert_eq!(v["role"], "assistant");
        let content = v["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "Let me search.");
        assert_eq!(content[1]["toolUse"]["toolUseId"], "tu_1");
        assert_eq!(content[1]["toolUse"]["name"], "search");
        assert_eq!(content[1]["toolUse"]["input"]["q"], "rust async");
    }

    #[test]
    fn message_to_converse_tool_role_becomes_tool_result_in_user_content() {
        // Converse injects tool outputs via a user-role message carrying a
        // `toolResult` content block — NOT role:tool (which Converse doesn't
        // accept). `toolUseId` links back to the assistant's toolUse block.
        let tool_msg = Message::tool_response("tu_1", "42 found");
        let v = message_to_converse(&tool_msg);
        assert_eq!(v["role"], "user");
        let content = v["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        let tr = &content[0]["toolResult"];
        assert_eq!(tr["toolUseId"], "tu_1");
        assert_eq!(tr["content"][0]["text"], "42 found");
    }

    #[test]
    fn parse_converse_response_extracts_text_and_tool_calls() {
        let v = json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Let me look that up."},
                        {"toolUse": {
                            "toolUseId": "tu_42",
                            "name": "search",
                            "input": {"q": "rust async"}
                        }}
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 25, "outputTokens": 12, "totalTokens": 37}
        });
        let resp = parse_converse_response("meta.llama3", v).unwrap();
        assert_eq!(resp.message.text_content(), "Let me look that up.");
        assert_eq!(resp.message.tool_calls.len(), 1);
        assert_eq!(resp.message.tool_calls[0].id, "tu_42");
        assert_eq!(resp.message.tool_calls[0].name, "search");
        assert_eq!(resp.message.tool_calls[0].arguments["q"], "rust async");
        assert!(matches!(resp.finish_reason, FinishReason::ToolCalls));
        assert_eq!(resp.usage.prompt, 25);
        assert_eq!(resp.usage.completion, 12);
        assert_eq!(resp.usage.total, 37);
    }

    #[test]
    fn parse_converse_response_maps_stop_reasons_to_finish_reason() {
        fn mk(reason: &str) -> Value {
            json!({
                "output": {"message": {"role":"assistant", "content":[{"text":"ok"}]}},
                "stopReason": reason,
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
            })
        }
        let cases = [
            ("end_turn", FinishReason::Stop),
            ("tool_use", FinishReason::ToolCalls),
            ("max_tokens", FinishReason::Length),
            ("content_filtered", FinishReason::ContentFilter),
            ("stop_sequence", FinishReason::Stop),
            ("weird_unknown", FinishReason::Other),
        ];
        for (reason, expected) in cases {
            let r = parse_converse_response("m", mk(reason)).unwrap();
            assert!(
                std::mem::discriminant(&r.finish_reason) == std::mem::discriminant(&expected),
                "reason={reason}: got {:?}", r.finish_reason
            );
        }
    }

    #[test]
    fn parse_converse_response_handles_missing_usage() {
        // Some Bedrock responses omit usage on error paths. Default to zeros.
        let v = json!({
            "output": {"message": {"role":"assistant","content":[{"text":"ok"}]}},
            "stopReason": "end_turn"
        });
        let r = parse_converse_response("m", v).unwrap();
        assert_eq!(r.usage.prompt, 0);
        assert_eq!(r.usage.completion, 0);
    }

    // ---------- BedrockConverseChat::stream (iter 100) ----------

    /// Build an AWS event-stream frame for testing. Matches the wire format
    /// parsed by `event_stream::parse_frame`:
    /// total_len u32 | headers_len u32 | prelude_crc u32 | headers | payload | crc32
    fn build_event_stream_frame(
        headers: &[(&str, &str)],
        payload: &[u8],
    ) -> Vec<u8> {
        use crc32fast::Hasher;
        // Encode headers: each = u8 name_len | name | u8 type(7=string) |
        //                       u16 value_len | value
        let mut headers_buf: Vec<u8> = Vec::new();
        for (name, value) in headers {
            let nb = name.as_bytes();
            let vb = value.as_bytes();
            headers_buf.push(nb.len() as u8);
            headers_buf.extend_from_slice(nb);
            headers_buf.push(7u8); // header value type = string
            headers_buf.extend_from_slice(&(vb.len() as u16).to_be_bytes());
            headers_buf.extend_from_slice(vb);
        }
        let total_len = (4 + 4 + 4 + headers_buf.len() + payload.len() + 4) as u32;
        let headers_len = headers_buf.len() as u32;
        let prelude = {
            let mut p = Vec::with_capacity(8);
            p.extend_from_slice(&total_len.to_be_bytes());
            p.extend_from_slice(&headers_len.to_be_bytes());
            p
        };
        let prelude_crc = {
            let mut h = Hasher::new();
            h.update(&prelude);
            h.finalize()
        };
        let mut out: Vec<u8> = Vec::with_capacity(total_len as usize);
        out.extend_from_slice(&prelude);
        out.extend_from_slice(&prelude_crc.to_be_bytes());
        out.extend_from_slice(&headers_buf);
        out.extend_from_slice(payload);
        let msg_crc = {
            let mut h = Hasher::new();
            h.update(&out);
            h.finalize()
        };
        out.extend_from_slice(&msg_crc.to_be_bytes());
        out
    }

    fn converse_event(event_type: &str, payload: Value) -> Vec<u8> {
        let body = serde_json::to_vec(&payload).unwrap();
        build_event_stream_frame(
            &[
                (":event-type", event_type),
                (":content-type", "application/json"),
                (":message-type", "event"),
            ],
            &body,
        )
    }

    /// Spawn a tiny TCP server that accepts ONE POST /model/.../converse-stream
    /// and streams back a canned sequence of AWS event-stream frames.
    fn spawn_fake_converse_stream(events: Vec<Vec<u8>>) -> (String, std::sync::mpsc::Sender<()>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        listener.set_nonblocking(true).unwrap();
        std::thread::spawn(move || {
            loop {
                if rx.try_recv().is_ok() { break; }
                match listener.accept() {
                    Ok((mut s, _)) => {
                        s.set_nonblocking(false).ok();
                        let mut buf = [0u8; 8192];
                        // Read request (don't bother parsing — we just need the body done).
                        let _ = s.read(&mut buf);
                        let header = b"HTTP/1.1 200 OK\r\nContent-Type: application/vnd.amazon.eventstream\r\nTransfer-Encoding: chunked\r\n\r\n";
                        let _ = s.write_all(header);
                        // Chunked-encode each frame so reqwest's bytes_stream()
                        // yields them progressively.
                        for frame in &events {
                            let chunk_len = format!("{:x}\r\n", frame.len());
                            let _ = s.write_all(chunk_len.as_bytes());
                            let _ = s.write_all(frame);
                            let _ = s.write_all(b"\r\n");
                        }
                        let _ = s.write_all(b"0\r\n\r\n");
                        break;
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        (format!("http://127.0.0.1:{port}"), tx)
    }

    async fn drain<S>(mut stream: S) -> Vec<ChatStreamEvent>
    where
        S: futures_util::Stream<Item = Result<ChatStreamEvent>> + Unpin,
    {
        use futures_util::StreamExt;
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(item.unwrap());
        }
        out
    }

    #[tokio::test]
    async fn converse_stream_emits_text_deltas_then_done() {
        // Canned sequence: messageStart → 2 text deltas → messageStop → metadata.
        let frames = vec![
            converse_event("messageStart", json!({"role": "assistant"})),
            converse_event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 0, "delta": {"text": "Hello, "}}),
            ),
            converse_event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 0, "delta": {"text": "world!"}}),
            ),
            converse_event("contentBlockStop", json!({"contentBlockIndex": 0})),
            converse_event("messageStop", json!({"stopReason": "end_turn"})),
            converse_event(
                "metadata",
                json!({"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}),
            ),
        ];
        let (url, shutdown) = spawn_fake_converse_stream(frames);
        let cfg = BedrockConfig::new(test_creds(), "us-east-1", "meta.llama3")
            .with_endpoint(url);
        let c = BedrockConverseChat::new(cfg).unwrap();
        let s = c.stream(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        let events = drain(s).await;
        // 2 deltas then Done.
        let deltas: Vec<&str> = events.iter().filter_map(|e| match e {
            ChatStreamEvent::Delta { text } => Some(text.as_str()),
            _ => None,
        }).collect();
        assert_eq!(deltas, vec!["Hello, ", "world!"]);
        // Last event is Done with assembled state.
        match events.last().unwrap() {
            ChatStreamEvent::Done { response } => {
                assert_eq!(response.message.text_content(), "Hello, world!");
                assert!(matches!(response.finish_reason, FinishReason::Stop));
                assert_eq!(response.usage.prompt, 10);
                assert_eq!(response.usage.completion, 5);
                assert_eq!(response.usage.total, 15);
                assert!(response.message.tool_calls.is_empty());
            }
            other => panic!("expected Done, got {other:?}"),
        }
        let _ = shutdown.send(());
    }

    #[tokio::test]
    async fn converse_stream_accumulates_tool_use_input_from_partial_json() {
        // contentBlockStart → toolUse meta; then 2 contentBlockDelta events
        // carry the input split across two fragments; contentBlockStop
        // finalizes; messageStop sets tool_use.
        let frames = vec![
            converse_event("messageStart", json!({"role": "assistant"})),
            converse_event("contentBlockStart", json!({
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "tu_1", "name": "search"}}
            })),
            converse_event("contentBlockDelta", json!({
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "{\"q\":\"rust"}}
            })),
            converse_event("contentBlockDelta", json!({
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": " async\"}"}}
            })),
            converse_event("contentBlockStop", json!({"contentBlockIndex": 0})),
            converse_event("messageStop", json!({"stopReason": "tool_use"})),
            converse_event("metadata", json!({
                "usage": {"inputTokens": 5, "outputTokens": 8, "totalTokens": 13}
            })),
        ];
        let (url, shutdown) = spawn_fake_converse_stream(frames);
        let cfg = BedrockConfig::new(test_creds(), "us-east-1", "mistral.mistral-large")
            .with_endpoint(url);
        let c = BedrockConverseChat::new(cfg).unwrap();
        let s = c.stream(vec![Message::user("find stuff")], &ChatOptions::default())
            .await
            .unwrap();
        let events = drain(s).await;
        // ToolCallDelta events present (at least one for the start, two for partials).
        let tool_deltas: Vec<&ChatStreamEvent> = events.iter()
            .filter(|e| matches!(e, ChatStreamEvent::ToolCallDelta { .. }))
            .collect();
        assert!(tool_deltas.len() >= 2, "expected tool_call deltas, got: {tool_deltas:?}");
        // Done event carries the assembled tool call.
        match events.last().unwrap() {
            ChatStreamEvent::Done { response } => {
                assert!(matches!(response.finish_reason, FinishReason::ToolCalls));
                assert_eq!(response.message.tool_calls.len(), 1);
                let tc = &response.message.tool_calls[0];
                assert_eq!(tc.id, "tu_1");
                assert_eq!(tc.name, "search");
                assert_eq!(tc.arguments["q"], "rust async");
            }
            other => panic!("expected Done, got {other:?}"),
        }
        let _ = shutdown.send(());
    }

    #[tokio::test]
    async fn converse_stream_exception_frame_surfaces_as_error() {
        // Exception frames have :message-type: exception. Must surface as
        // an Error on the stream, not silently swallow.
        let exc_payload = br#"{"message":"throttled"}"#.to_vec();
        let exc_frame = build_event_stream_frame(
            &[
                (":event-type", "throttlingException"),
                (":message-type", "exception"),
                (":content-type", "application/json"),
            ],
            &exc_payload,
        );
        let (url, shutdown) = spawn_fake_converse_stream(vec![exc_frame]);
        let cfg = BedrockConfig::new(test_creds(), "us-east-1", "m").with_endpoint(url);
        let c = BedrockConverseChat::new(cfg).unwrap();
        let s = c.stream(vec![Message::user("x")], &ChatOptions::default())
            .await
            .unwrap();
        use futures_util::StreamExt;
        let mut s = s;
        let mut saw_error = false;
        while let Some(item) = s.next().await {
            if item.is_err() {
                let msg = format!("{}", item.unwrap_err());
                assert!(msg.contains("throttlingException") || msg.contains("throttled"));
                saw_error = true;
                break;
            }
        }
        assert!(saw_error, "expected exception frame to surface as Error");
        let _ = shutdown.send(());
    }

    #[tokio::test]
    async fn converse_stream_ignores_unknown_event_types() {
        // Forward-compat: AWS may add new event types — don't crash on
        // unknown kinds, just skip.
        let frames = vec![
            converse_event("messageStart", json!({"role": "assistant"})),
            converse_event("futureEventType", json!({"anything": "goes"})),
            converse_event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 0, "delta": {"text": "ok"}}),
            ),
            converse_event("messageStop", json!({"stopReason": "end_turn"})),
            converse_event("metadata", json!({
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
            })),
        ];
        let (url, shutdown) = spawn_fake_converse_stream(frames);
        let cfg = BedrockConfig::new(test_creds(), "us-east-1", "m").with_endpoint(url);
        let c = BedrockConverseChat::new(cfg).unwrap();
        let s = c.stream(vec![Message::user("x")], &ChatOptions::default())
            .await
            .unwrap();
        let events = drain(s).await;
        // Stream still completes successfully despite the unknown event.
        let deltas: Vec<&str> = events.iter().filter_map(|e| match e {
            ChatStreamEvent::Delta { text } => Some(text.as_str()),
            _ => None,
        }).collect();
        assert_eq!(deltas, vec!["ok"]);
        assert!(matches!(events.last().unwrap(), ChatStreamEvent::Done { .. }));
        let _ = shutdown.send(());
    }
}
