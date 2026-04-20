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
mod sigv4;

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
        .splitn(2, "://")
        .nth(1)
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
}
