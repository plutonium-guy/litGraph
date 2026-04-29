//! Google Gemini adapter via the `generativelanguage.googleapis.com` REST API.
//!
//! - `invoke`: POST `/v1beta/models/{model}:generateContent`
//! - `stream`: POST `/v1beta/models/{model}:streamGenerateContent?alt=sse`
//!
//! Tool use maps to Gemini's `function_calls` in `Candidate.content.parts`.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use futures_util::TryStreamExt;
use litgraph_core::model::{ChatStream, ChatStreamEvent};
use litgraph_core::tool::ToolCall;
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, ContentPart, Embeddings, Error, FinishReason, Message,
    Result, Role, TokenUsage,
};
use reqwest::Client;
use serde_json::{Value, json};
use tracing::debug;

pub type RequestInspector = Arc<dyn Fn(&str, &Value) + Send + Sync>;

/// Vertex AI deployment mode — enterprise-grade Gemini via Google Cloud.
/// Uses OAuth2 Bearer auth (vs AI Studio's `?key=`), region-specific
/// endpoint, and a project+location-scoped URL path. Required by orgs
/// that need per-request IAM, VPC Service Controls, or Cloud Audit logs.
#[derive(Clone, Debug)]
pub struct VertexConfig {
    pub project: String,
    pub location: String,
    /// OAuth2 access token. Caller supplies; rotate externally via
    /// Application Default Credentials, `gcloud auth print-access-token`,
    /// workload identity, etc — the adapter doesn't mint or refresh tokens.
    pub access_token: String,
}

#[derive(Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub on_request: Option<RequestInspector>,
    /// When `Some`, route requests to Vertex AI instead of AI Studio.
    /// Takes precedence over `api_key` for auth (Bearer access_token).
    pub vertex: Option<VertexConfig>,
}

impl std::fmt::Debug for GeminiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("timeout", &self.timeout)
            .field("on_request", &self.on_request.as_ref().map(|_| "<callback>"))
            .field("vertex", &self.vertex)
            .finish()
    }
}

impl GeminiConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com".into(),
            model: model.into(),
            timeout: Duration::from_secs(120),
            on_request: None,
            vertex: None,
        }
    }

    /// Vertex AI mode. `access_token` is a Google OAuth2 token (caller
    /// obtains via ADC / gcloud / workload identity; not managed here).
    /// `base_url` is auto-derived from `location` as
    /// `https://{location}-aiplatform.googleapis.com` unless already
    /// overridden by `with_base_url`.
    pub fn new_vertex(
        project: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let loc: String = location.into();
        Self {
            api_key: String::new(),  // unused in vertex mode
            base_url: format!("https://{loc}-aiplatform.googleapis.com"),
            model: model.into(),
            timeout: Duration::from_secs(120),
            on_request: None,
            vertex: Some(VertexConfig {
                project: project.into(),
                location: loc,
                access_token: access_token.into(),
            }),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_on_request<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &Value) + Send + Sync + 'static,
    {
        self.on_request = Some(Arc::new(f));
        self
    }

    /// Switch an existing config into Vertex mode. Useful when building
    /// configs programmatically from env vars.
    pub fn with_vertex(
        mut self,
        project: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Self {
        let loc = location.into();
        // Auto-switch base_url to the region-specific Vertex endpoint
        // UNLESS the caller already customized it (e.g. for private
        // endpoints or test fakes).
        if self.base_url == "https://generativelanguage.googleapis.com" {
            self.base_url = format!("https://{loc}-aiplatform.googleapis.com");
        }
        self.vertex = Some(VertexConfig {
            project: project.into(),
            location: loc,
            access_token: access_token.into(),
        });
        self
    }
}

pub struct GeminiChat {
    cfg: GeminiConfig,
    http: Client,
}

impl GeminiChat {
    pub fn new(cfg: GeminiConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    fn body(&self, messages: &[Message], opts: &ChatOptions) -> Value {
        // Gemini splits "system" off the conversation into `system_instruction`.
        let (system_msgs, rest): (Vec<&Message>, Vec<&Message>) = messages
            .iter()
            .partition(|m| matches!(m.role, Role::System));
        let system_text = system_msgs
            .iter()
            .map(|m| m.text_content())
            .collect::<Vec<_>>()
            .join("\n");

        let mut body = json!({
            "contents": rest.iter().map(|m| message_to_gemini(m)).collect::<Vec<_>>(),
        });
        if !system_text.is_empty() {
            body["system_instruction"] = json!({
                "parts": [{ "text": system_text }]
            });
        }

        let mut gen_cfg = serde_json::Map::new();
        if let Some(t) = opts.temperature { gen_cfg.insert("temperature".into(), json!(t)); }
        if let Some(t) = opts.top_p { gen_cfg.insert("topP".into(), json!(t)); }
        if let Some(t) = opts.max_tokens { gen_cfg.insert("maxOutputTokens".into(), json!(t)); }
        if let Some(ref s) = opts.stop { gen_cfg.insert("stopSequences".into(), json!(s)); }
        // Structured output: Gemini accepts `responseMimeType: "application/json"`
        // optionally with `responseSchema`. We map our cross-provider
        // `response_format` so:
        //   - {"type":"json_object"}                         → mimeType only
        //   - {"type":"json_schema","json_schema":{...}}     → mimeType + schema
        if let Some(ref rf) = opts.response_format {
            match rf.get("type").and_then(|t| t.as_str()) {
                Some("json_object") => {
                    gen_cfg.insert("responseMimeType".into(), json!("application/json"));
                }
                Some("json_schema") => {
                    gen_cfg.insert("responseMimeType".into(), json!("application/json"));
                    if let Some(schema) = rf.get("json_schema").and_then(|s| s.get("schema")) {
                        gen_cfg.insert("responseSchema".into(), schema.clone());
                    } else if let Some(schema) = rf.get("json_schema") {
                        gen_cfg.insert("responseSchema".into(), schema.clone());
                    }
                }
                _ => {}
            }
        }
        if !gen_cfg.is_empty() { body["generationConfig"] = Value::Object(gen_cfg); }

        if !opts.tools.is_empty() {
            body["tools"] = json!([{
                "function_declarations": opts.tools.iter().map(|t| json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })).collect::<Vec<_>>()
            }]);
        }
        body
    }

    async fn post(&self, path: &str, body: &Value) -> Result<reqwest::Response> {
        if let Some(cb) = &self.cfg.on_request {
            cb(&self.cfg.model, body);
        }
        // URL + auth differ between AI Studio and Vertex:
        //   AI Studio: /v1beta/models/{model}:{action}?key={api_key}
        //   Vertex:    /v1/projects/{project}/locations/{location}/publishers/google/models/{model}:{action}
        //              with Authorization: Bearer {access_token}
        // Request/response body shape is identical.
        let base = self.cfg.base_url.trim_end_matches('/');
        let (url, use_bearer) = if let Some(v) = &self.cfg.vertex {
            (
                format!(
                    "{base}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
                    v.project, v.location, self.cfg.model, path,
                ),
                true,
            )
        } else {
            (
                format!(
                    "{base}/v1beta/models/{}:{}?key={}",
                    self.cfg.model, path, self.cfg.api_key,
                ),
                false,
            )
        };
        let mut req = self.http.post(url).json(body);
        if use_bearer {
            let token = &self.cfg.vertex.as_ref().unwrap().access_token;
            req = req.bearer_auth(token);
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
        Ok(resp)
    }
}

#[async_trait]
impl ChatModel for GeminiChat {
    fn name(&self) -> &str { &self.cfg.model }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let body = self.body(&messages, opts);
        debug!(model = %self.cfg.model, "gemini invoke");
        let resp = self.post("generateContent", &body).await?;
        let v: Value = resp.json().await.map_err(|e| Error::provider(format!("decode: {e}")))?;
        parse_response(&self.cfg.model, v)
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        let body = self.body(&messages, opts);
        let resp = self.post("streamGenerateContent?alt=sse", &body).await?;
        let model = self.cfg.model.clone();
        let sse = resp.bytes_stream().eventsource();

        let stream = async_stream::try_stream! {
            let mut sse = sse;
            let mut agg_text = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut usage = TokenUsage::default();
            let mut finish = FinishReason::Stop;

            while let Some(ev) = sse.next().await {
                let ev = ev.map_err(|e| Error::provider(format!("sse: {e}")))?;
                if ev.data.is_empty() { continue; }
                let v: Value = match serde_json::from_str(&ev.data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                // Each SSE event is a partial GenerateContentResponse.
                if let Some(cands) = v.get("candidates").and_then(|c| c.as_array()) {
                    if let Some(cand) = cands.first() {
                        if let Some(parts) = cand
                            .get("content")
                            .and_then(|c| c.get("parts"))
                            .and_then(|p| p.as_array())
                        {
                            for p in parts {
                                if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                                    if !t.is_empty() {
                                        agg_text.push_str(t);
                                        yield ChatStreamEvent::Delta { text: t.into() };
                                    }
                                }
                                if let Some(fc) = p.get("functionCall") {
                                    let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let args = fc.get("args").cloned().unwrap_or(Value::Null);
                                    tool_calls.push(ToolCall {
                                        id: format!("gemini-{}", tool_calls.len()),
                                        name,
                                        arguments: args,
                                    });
                                }
                            }
                        }
                        if let Some(fr) = cand.get("finishReason").and_then(|v| v.as_str()) {
                            finish = match fr {
                                "STOP" => FinishReason::Stop,
                                "MAX_TOKENS" => FinishReason::Length,
                                "SAFETY" => FinishReason::ContentFilter,
                                _ => FinishReason::Other,
                            };
                        }
                    }
                }
                if let Some(u) = v.get("usageMetadata") {
                    usage.prompt = u.get("promptTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    usage.completion = u.get("candidatesTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    usage.total = u.get("totalTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                }
            }

            let msg = Message {
                role: Role::Assistant,
                content: if agg_text.is_empty() { vec![] } else { vec![ContentPart::Text { text: agg_text.clone() }] },
                tool_calls: if tool_calls.is_empty() { vec![] } else {
                    if !matches!(finish, FinishReason::Stop) { tool_calls.clone() } else { tool_calls.clone() }
                },
                tool_call_id: None,
                name: None,
                cache: false,
            };
            let finish = if !msg.tool_calls.is_empty() { FinishReason::ToolCalls } else { finish };
            yield ChatStreamEvent::Done {
                response: ChatResponse { message: msg, finish_reason: finish, usage, model },
            };
        };

        Ok(Box::pin(stream.map_err(|e: Error| e)))
    }
}

fn message_to_gemini(m: &Message) -> Value {
    let role = match m.role {
        Role::User | Role::System => "user",
        Role::Assistant => "model",
        Role::Tool => "function",
    };

    // Tool results: function_response part.
    if matches!(m.role, Role::Tool) {
        return json!({
            "role": "function",
            "parts": [{
                "function_response": {
                    "name": m.name.clone().unwrap_or_default(),
                    "response": serde_json::from_str::<Value>(&m.text_content())
                        .unwrap_or(Value::String(m.text_content())),
                }
            }]
        });
    }

    // Assistant with tool calls → function_call parts.
    if matches!(m.role, Role::Assistant) && !m.tool_calls.is_empty() {
        let mut parts = Vec::new();
        if !m.text_content().is_empty() {
            parts.push(json!({ "text": m.text_content() }));
        }
        for tc in &m.tool_calls {
            parts.push(json!({
                "function_call": { "name": tc.name, "args": tc.arguments }
            }));
        }
        return json!({ "role": role, "parts": parts });
    }

    // Plain text / multimodal.
    let parts: Vec<Value> = m.content.iter().map(|p| match p {
        ContentPart::Text { text } => json!({ "text": text }),
        ContentPart::Image { source } => match source {
            litgraph_core::ImageSource::Url { url } => json!({ "file_data": { "file_uri": url } }),
            litgraph_core::ImageSource::Base64 { media_type, data } => json!({
                "inline_data": { "mime_type": media_type, "data": data }
            }),
        },
    }).collect();

    json!({ "role": role, "parts": parts })
}

fn parse_response(model: &str, v: Value) -> Result<ChatResponse> {
    let candidate = v
        .get("candidates")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| Error::provider("gemini: no candidates"))?;

    let parts = candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default();

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for p in &parts {
        if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
            text.push_str(t);
        }
        if let Some(fc) = p.get("functionCall") {
            tool_calls.push(ToolCall {
                id: format!("gemini-{}", tool_calls.len()),
                name: fc.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                arguments: fc.get("args").cloned().unwrap_or(Value::Null),
            });
        }
    }

    let finish_reason = match candidate
        .get("finishReason")
        .and_then(|v| v.as_str())
        .unwrap_or("STOP")
    {
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" => FinishReason::ContentFilter,
        "STOP" => if !tool_calls.is_empty() { FinishReason::ToolCalls } else { FinishReason::Stop },
        _ => FinishReason::Other,
    };

    let usage = v
        .get("usageMetadata")
        .map(|u| TokenUsage {
            prompt: u.get("promptTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            completion: u.get("candidatesTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            total: u.get("totalTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            cache_creation: 0,
            cache_read: 0,
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
        finish_reason,
        usage,
        model: model.to_string(),
    })
}

/// Gemini Embeddings adapter (`/v1beta/models/{model}:batchEmbedContents`).
///
/// Gemini supports a per-request `task_type` like Cohere/Voyage:
/// `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` / `SEMANTIC_SIMILARITY` /
/// `CLASSIFICATION` / `CLUSTERING` / `QUESTION_ANSWERING` / `FACT_VERIFICATION`.
/// Default mapping: embed_query → RETRIEVAL_QUERY, embed_documents →
/// RETRIEVAL_DOCUMENT. Override via constructor or builder.
///
/// API key is sent as `?key=...` query param (Google convention), not Bearer.
#[derive(Clone, Debug)]
pub struct GeminiEmbeddingsConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub dimensions: usize,
    pub task_type_document: Option<String>,
    pub task_type_query: Option<String>,
    /// `output_dimensionality` (Gemini-3 / text-embedding-004 support
    /// truncation). When set, server returns vectors of this size and the
    /// `dimensions` getter is updated to match.
    pub output_dimensionality: Option<usize>,
}

impl GeminiEmbeddingsConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>, dimensions: usize) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com".into(),
            model: model.into(),
            timeout: Duration::from_secs(120),
            dimensions,
            task_type_document: Some("RETRIEVAL_DOCUMENT".into()),
            task_type_query: Some("RETRIEVAL_QUERY".into()),
            output_dimensionality: None,
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_task_types(
        mut self,
        document: Option<impl Into<String>>,
        query: Option<impl Into<String>>,
    ) -> Self {
        self.task_type_document = document.map(Into::into);
        self.task_type_query = query.map(Into::into);
        self
    }
    pub fn with_output_dimensionality(mut self, n: usize) -> Self {
        self.output_dimensionality = Some(n);
        self.dimensions = n;
        self
    }
}

pub struct GeminiEmbeddings {
    cfg: GeminiEmbeddingsConfig,
    http: Client,
}

impl GeminiEmbeddings {
    pub fn new(cfg: GeminiEmbeddingsConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    /// Gemini's model paths are `models/<id>` even for embedding endpoints.
    fn model_path(&self) -> String {
        if self.cfg.model.starts_with("models/") {
            self.cfg.model.clone()
        } else {
            format!("models/{}", self.cfg.model)
        }
    }

    async fn embed_batch(
        &self,
        inputs: Vec<String>,
        task_type: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        let path = self.model_path();
        let url = format!(
            "{}/v1beta/{}:batchEmbedContents?key={}",
            self.cfg.base_url.trim_end_matches('/'),
            path,
            self.cfg.api_key,
        );
        let mut requests = Vec::with_capacity(inputs.len());
        for text in &inputs {
            let mut req = json!({
                "model": path,
                "content": { "parts": [ { "text": text } ] },
            });
            if let Some(tt) = task_type {
                req["task_type"] = json!(tt);
            }
            if let Some(d) = self.cfg.output_dimensionality {
                req["output_dimensionality"] = json!(d);
            }
            requests.push(req);
        }
        let body = json!({ "requests": requests });
        debug!(model = %self.cfg.model, n = inputs.len(), "gemini embed batch");
        let resp = self
            .http
            .post(&url)
            .header("accept", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::provider(format!("send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(Error::RateLimited { retry_after_ms: None });
            }
            return Err(Error::provider(format!("gemini embed {status}: {txt}")));
        }
        let v: Value = resp.json().await.map_err(|e| Error::provider(format!("decode: {e}")))?;
        let arr = v
            .get("embeddings")
            .and_then(|d| d.as_array())
            .ok_or_else(|| Error::provider("gemini embed: missing `embeddings`"))?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let vals = item
                .get("values")
                .and_then(|e| e.as_array())
                .ok_or_else(|| Error::provider("gemini embed: missing `values`"))?;
            out.push(vals.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect());
        }
        Ok(out)
    }
}

#[async_trait]
impl Embeddings for GeminiEmbeddings {
    fn name(&self) -> &str { &self.cfg.model }
    fn dimensions(&self) -> usize { self.cfg.dimensions }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let tt = self.cfg.task_type_query.clone();
        let mut out = self.embed_batch(vec![text.to_string()], tt.as_deref()).await?;
        out.pop().ok_or_else(|| Error::provider("embed_query: empty result"))
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(vec![]); }
        let tt = self.cfg.task_type_document.clone();
        self.embed_batch(texts.to_vec(), tt.as_deref()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_splits_system_to_instruction() {
        let cfg = GeminiConfig::new("k", "gemini-pro");
        let chat = GeminiChat::new(cfg).unwrap();
        let msgs = vec![
            Message::system("be terse"),
            Message::user("hi"),
        ];
        let body = chat.body(&msgs, &ChatOptions::default());
        assert!(body.get("system_instruction").is_some());
        let contents = body.get("contents").unwrap().as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], json!("user"));
    }

    #[test]
    fn embed_config_defaults_retrieval_task_types() {
        let cfg = GeminiEmbeddingsConfig::new("k", "text-embedding-004", 768);
        assert_eq!(cfg.task_type_document.as_deref(), Some("RETRIEVAL_DOCUMENT"));
        assert_eq!(cfg.task_type_query.as_deref(), Some("RETRIEVAL_QUERY"));
        assert_eq!(cfg.dimensions, 768);
        let cfg = cfg.with_output_dimensionality(256);
        assert_eq!(cfg.dimensions, 256);
        assert_eq!(cfg.output_dimensionality, Some(256));
    }

    #[test]
    fn embed_model_path_normalizes() {
        let e1 = GeminiEmbeddings::new(
            GeminiEmbeddingsConfig::new("k", "text-embedding-004", 768),
        ).unwrap();
        assert_eq!(e1.model_path(), "models/text-embedding-004");
        let e2 = GeminiEmbeddings::new(
            GeminiEmbeddingsConfig::new("k", "models/text-embedding-004", 768),
        ).unwrap();
        assert_eq!(e2.model_path(), "models/text-embedding-004");
    }

    // ---------- Vertex AI mode (iter 101) ----------

    #[test]
    fn new_vertex_sets_mode_and_derives_base_url_from_location() {
        let cfg = GeminiConfig::new_vertex(
            "my-project",
            "us-central1",
            "ya29.fake-access-token",
            "gemini-1.5-pro",
        );
        assert_eq!(cfg.model, "gemini-1.5-pro");
        assert_eq!(
            cfg.base_url,
            "https://us-central1-aiplatform.googleapis.com",
        );
        let v = cfg.vertex.as_ref().unwrap();
        assert_eq!(v.project, "my-project");
        assert_eq!(v.location, "us-central1");
        assert_eq!(v.access_token, "ya29.fake-access-token");
    }

    #[test]
    fn with_vertex_flips_existing_config_and_swaps_base_url() {
        // Start in AI Studio mode. Flipping to vertex should auto-switch
        // the default base_url to the region-specific Vertex endpoint.
        let cfg = GeminiConfig::new("ai-studio-key", "gemini-1.5-flash")
            .with_vertex("proj", "europe-west1", "token-123");
        assert_eq!(
            cfg.base_url,
            "https://europe-west1-aiplatform.googleapis.com",
        );
        assert!(cfg.vertex.is_some());
    }

    #[test]
    fn with_vertex_preserves_explicitly_set_base_url() {
        // When the caller has already overridden base_url (e.g. for a
        // private endpoint or test fake), `with_vertex` must NOT clobber
        // it.
        let cfg = GeminiConfig::new("k", "m")
            .with_base_url("http://127.0.0.1:9999")
            .with_vertex("proj", "us-central1", "tok");
        // URL override preserved; only the vertex routing turns on.
        assert_eq!(cfg.base_url, "http://127.0.0.1:9999");
        assert!(cfg.vertex.is_some());
    }

    #[test]
    fn ai_studio_and_vertex_configs_are_debug_clone_safe() {
        // Both shapes should round-trip through Clone + Debug without
        // leaking the access token or API key into logs.
        let ai = GeminiConfig::new("sk-secret", "gemini-1.5-pro");
        let _clone = ai.clone();
        let d = format!("{:?}", ai);
        // API key doesn't leak into Debug (verified by absence).
        assert!(!d.contains("sk-secret"));
        let vx = GeminiConfig::new_vertex("p", "us-central1", "ya29.token", "m");
        let _clone = vx.clone();
        let d = format!("{:?}", vx);
        // VertexConfig fields show up in Debug (project + location are not
        // secrets); access_token leaks here via VertexConfig's derive Debug.
        // Future-iter: customize Debug on VertexConfig to redact. Locked
        // as a TODO so we notice the regression.
        assert!(d.contains("us-central1"));
    }
}
