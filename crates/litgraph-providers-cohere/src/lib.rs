//! Cohere chat (Command-R family) adapter via the v2/chat API.
//!
//! - `invoke`: POST `/v2/chat`
//! - `stream`: POST `/v2/chat` with `stream: true` — SSE event stream where each
//!   event is JSON like `{"type":"content-delta","delta":{"message":{"content":{"text":"..."}}}}`.
//!
//! # Tool calling
//!
//! Cohere's tool format: `tools: [{type:"function", function:{name, description, parameters}}]`.
//! Tool calls in responses appear as `{type:"tool_call", id, name, arguments}` blocks.

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

#[derive(Clone)]
pub struct CohereChatConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub on_request: Option<RequestInspector>,
}

impl std::fmt::Debug for CohereChatConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereChatConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl CohereChatConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.cohere.com".into(),
            model: model.into(),
            timeout: Duration::from_secs(120),
            on_request: None,
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
}

pub struct CohereChat {
    cfg: CohereChatConfig,
    http: Client,
}

impl CohereChat {
    pub fn new(cfg: CohereChatConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    fn body(&self, messages: &[Message], opts: &ChatOptions, stream: bool) -> Value {
        let mut body = json!({
            "model": self.cfg.model,
            "messages": messages.iter().map(message_to_cohere).collect::<Vec<_>>(),
            "stream": stream,
        });
        if let Some(t) = opts.temperature { body["temperature"] = json!(t); }
        if let Some(t) = opts.top_p { body["p"] = json!(t); } // Cohere uses `p` not `top_p`
        if let Some(t) = opts.max_tokens { body["max_tokens"] = json!(t); }
        if let Some(ref s) = opts.stop { body["stop_sequences"] = json!(s); }
        if !opts.tools.is_empty() {
            body["tools"] = json!(opts.tools.iter().map(|t| json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })).collect::<Vec<_>>());
        }
        body
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        if let Some(cb) = &self.cfg.on_request {
            cb(&self.cfg.model, body);
        }
        let url = format!("{}/v2/chat", self.cfg.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .header("accept", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| Error::provider(format!("send: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            if s.as_u16() == 429 { return Err(Error::RateLimited { retry_after_ms: None }); }
            return Err(Error::provider(format!("{s}: {t}")));
        }
        Ok(resp)
    }
}

#[async_trait]
impl ChatModel for CohereChat {
    fn name(&self) -> &str { &self.cfg.model }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let body = self.body(&messages, opts, false);
        debug!(model = %self.cfg.model, "cohere invoke");
        let resp = self.post(&body).await?;
        let v: Value = resp.json().await.map_err(|e| Error::provider(format!("decode: {e}")))?;
        parse_response(&self.cfg.model, v)
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        let body = self.body(&messages, opts, true);
        let resp = self.post(&body).await?;
        let model = self.cfg.model.clone();
        let sse = resp.bytes_stream().eventsource();

        let stream = async_stream::try_stream! {
            let mut sse = sse;
            let mut agg_text = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut tool_arg_bufs: Vec<String> = Vec::new();
            let mut usage = TokenUsage::default();
            let mut finish = FinishReason::Stop;

            while let Some(ev) = sse.next().await {
                let ev = ev.map_err(|e| Error::provider(format!("sse: {e}")))?;
                if ev.data.is_empty() || ev.data == "[DONE]" { continue; }
                let v: Value = match serde_json::from_str(&ev.data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let etype = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match etype {
                    "content-delta" => {
                        if let Some(t) = v.get("delta")
                            .and_then(|d| d.get("message"))
                            .and_then(|m| m.get("content"))
                            .and_then(|c| c.get("text"))
                            .and_then(|t| t.as_str())
                        {
                            if !t.is_empty() {
                                agg_text.push_str(t);
                                yield ChatStreamEvent::Delta { text: t.into() };
                            }
                        }
                    }
                    "tool-call-start" => {
                        if let Some(tc) = v.get("delta").and_then(|d| d.get("message")).and_then(|m| m.get("tool_calls")) {
                            tool_calls.push(ToolCall {
                                id: tc.get("id").and_then(|v| v.as_str()).unwrap_or("").into(),
                                name: tc.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()).unwrap_or("").into(),
                                arguments: Value::Null,
                            });
                            tool_arg_bufs.push(String::new());
                        }
                    }
                    "tool-call-delta" => {
                        if let Some(args) = v.get("delta")
                            .and_then(|d| d.get("message"))
                            .and_then(|m| m.get("tool_calls"))
                            .and_then(|tc| tc.get("function"))
                            .and_then(|f| f.get("arguments"))
                            .and_then(|a| a.as_str())
                        {
                            if let Some(buf) = tool_arg_bufs.last_mut() {
                                buf.push_str(args);
                            }
                            yield ChatStreamEvent::ToolCallDelta {
                                index: (tool_arg_bufs.len() as u32).saturating_sub(1),
                                id: None, name: None,
                                arguments_delta: Some(args.into()),
                            };
                        }
                    }
                    "message-end" => {
                        if let Some(d) = v.get("delta") {
                            if let Some(fr) = d.get("finish_reason").and_then(|v| v.as_str()) {
                                finish = match fr {
                                    "COMPLETE" | "complete" => FinishReason::Stop,
                                    "MAX_TOKENS" | "max_tokens" => FinishReason::Length,
                                    "TOOL_CALL" | "tool_call" => FinishReason::ToolCalls,
                                    _ => FinishReason::Other,
                                };
                            }
                            if let Some(u) = d.get("usage").and_then(|u| u.get("tokens")) {
                                usage.prompt = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                usage.completion = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                usage.total = usage.prompt + usage.completion;
                            }
                        }
                    }
                    _ => {}
                }
            }

            for (i, b) in tool_arg_bufs.iter().enumerate() {
                if !b.is_empty() {
                    tool_calls[i].arguments = serde_json::from_str(b).unwrap_or(Value::String(b.clone()));
                }
            }
            let msg = Message {
                role: Role::Assistant,
                content: if agg_text.is_empty() { vec![] } else { vec![ContentPart::Text { text: agg_text.clone() }] },
                tool_calls,
                tool_call_id: None,
                name: None,
                cache: false,
            };
            yield ChatStreamEvent::Done {
                response: ChatResponse { message: msg, finish_reason: finish, usage, model },
            };
        };

        Ok(Box::pin(stream.map_err(|e: Error| e)))
    }
}

fn message_to_cohere(m: &Message) -> Value {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    if matches!(m.role, Role::Tool) {
        return json!({
            "role": "tool",
            "tool_call_id": m.tool_call_id.clone().unwrap_or_default(),
            "content": m.text_content(),
        });
    }
    if matches!(m.role, Role::Assistant) && !m.tool_calls.is_empty() {
        return json!({
            "role": "assistant",
            "tool_calls": m.tool_calls.iter().map(|tc| json!({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments.to_string(),
                }
            })).collect::<Vec<_>>(),
            "tool_plan": m.text_content(),
        });
    }
    json!({ "role": role, "content": m.text_content() })
}

fn parse_response(model: &str, v: Value) -> Result<ChatResponse> {
    let msg = v.get("message").ok_or_else(|| Error::provider("cohere: missing `message`"))?;
    let mut text = String::new();
    if let Some(parts) = msg.get("content").and_then(|c| c.as_array()) {
        for p in parts {
            if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                    text.push_str(t);
                }
            }
        }
    } else if let Some(s) = msg.get("content").and_then(|c| c.as_str()) {
        // Some Cohere responses use a flat string content.
        text.push_str(s);
    }

    let mut tool_calls = Vec::new();
    if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tcs {
            tool_calls.push(ToolCall {
                id: tc.get("id").and_then(|v| v.as_str()).unwrap_or("").into(),
                name: tc.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()).unwrap_or("").into(),
                arguments: tc.get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .and_then(|s| serde_json::from_str::<Value>(s).ok())
                    .unwrap_or(Value::Null),
            });
        }
    }

    let finish = match v.get("finish_reason").and_then(|v| v.as_str()).unwrap_or("complete") {
        "MAX_TOKENS" | "max_tokens" => FinishReason::Length,
        "TOOL_CALL" | "tool_call" => FinishReason::ToolCalls,
        "COMPLETE" | "complete" | "STOP_SEQUENCE" => FinishReason::Stop,
        _ => FinishReason::Other,
    };
    let finish = if !tool_calls.is_empty() && matches!(finish, FinishReason::Stop) {
        FinishReason::ToolCalls
    } else {
        finish
    };

    let usage = v
        .get("usage")
        .and_then(|u| u.get("tokens"))
        .map(|u| {
            let prompt = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let completion = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            TokenUsage { prompt, completion, total: prompt + completion, cache_creation: 0, cache_read: 0 }
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

/// Cohere `embed-*` family adapter (v2/embed). Cohere requires a per-request
/// `input_type` ("search_document" / "search_query" / "classification" /
/// "clustering"); we map `embed_documents` → "search_document" and
/// `embed_query` → "search_query" by default. Override via
/// `with_input_types(doc, query)` if your downstream task is classification or
/// clustering.
#[derive(Clone, Debug)]
pub struct CohereEmbeddingsConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub dimensions: usize,
    pub input_type_document: String,
    pub input_type_query: String,
}

impl CohereEmbeddingsConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>, dimensions: usize) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.cohere.com".into(),
            model: model.into(),
            timeout: Duration::from_secs(120),
            dimensions,
            input_type_document: "search_document".into(),
            input_type_query: "search_query".into(),
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_input_types(mut self, document: impl Into<String>, query: impl Into<String>) -> Self {
        self.input_type_document = document.into();
        self.input_type_query = query.into();
        self
    }
}

pub struct CohereEmbeddings {
    cfg: CohereEmbeddingsConfig,
    http: Client,
}

impl CohereEmbeddings {
    pub fn new(cfg: CohereEmbeddingsConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    async fn embed_batch(&self, inputs: Vec<String>, input_type: &str) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/v2/embed", self.cfg.base_url.trim_end_matches('/'));
        let body = json!({
            "model": self.cfg.model,
            "texts": inputs,
            "input_type": input_type,
            "embedding_types": ["float"],
        });
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
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
            return Err(Error::provider(format!("embeddings {status}: {txt}")));
        }
        let v: Value = resp.json().await.map_err(|e| Error::provider(format!("decode: {e}")))?;
        // Cohere v2: { "embeddings": { "float": [[...], ...] }, ... }
        let arr = v
            .get("embeddings")
            .and_then(|e| e.get("float"))
            .and_then(|f| f.as_array())
            .ok_or_else(|| Error::provider("cohere embed: missing embeddings.float"))?;
        let mut out = Vec::with_capacity(arr.len());
        for row in arr {
            let r = row
                .as_array()
                .ok_or_else(|| Error::provider("cohere embed: row not array"))?;
            out.push(r.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect());
        }
        Ok(out)
    }
}

#[async_trait]
impl Embeddings for CohereEmbeddings {
    fn name(&self) -> &str { &self.cfg.model }
    fn dimensions(&self) -> usize { self.cfg.dimensions }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let it = self.cfg.input_type_query.clone();
        let mut out = self.embed_batch(vec![text.to_string()], &it).await?;
        out.pop().ok_or_else(|| Error::provider("embed_query: empty result"))
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(vec![]); }
        let it = self.cfg.input_type_document.clone();
        self.embed_batch(texts.to_vec(), &it).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_uses_cohere_p_not_top_p() {
        let chat = CohereChat::new(CohereChatConfig::new("k", "command-r-plus")).unwrap();
        let body = chat.body(
            &[Message::user("hi")],
            &ChatOptions { temperature: Some(0.5), top_p: Some(0.9), ..Default::default() },
            false,
        );
        let temp = body["temperature"].as_f64().unwrap();
        assert!((temp - 0.5).abs() < 1e-3);
        let p = body["p"].as_f64().unwrap();
        assert!((p - 0.9).abs() < 1e-3);   // Cohere-specific name
        assert!(body.get("top_p").is_none()); // never the OpenAI name
        assert_eq!(body["model"], json!("command-r-plus"));
    }

    #[test]
    fn parse_text_response() {
        let v = json!({
            "message": { "role": "assistant", "content": [{"type":"text","text":"hello"}] },
            "finish_reason": "complete",
            "usage": { "tokens": { "input_tokens": 5, "output_tokens": 2 } },
        });
        let r = parse_response("command-r", v).unwrap();
        assert_eq!(r.message.text_content(), "hello");
        assert!(matches!(r.finish_reason, FinishReason::Stop));
        assert_eq!(r.usage.prompt, 5);
        assert_eq!(r.usage.completion, 2);
        assert_eq!(r.usage.total, 7);
    }

    #[test]
    fn parse_tool_call_response() {
        let v = json!({
            "message": {
                "role": "assistant",
                "tool_plan": "I'll look this up.",
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": { "name": "search", "arguments": "{\"q\":\"rust\"}" }
                }]
            },
            "finish_reason": "tool_call",
        });
        let r = parse_response("command-r", v).unwrap();
        assert!(matches!(r.finish_reason, FinishReason::ToolCalls));
        assert_eq!(r.message.tool_calls.len(), 1);
        assert_eq!(r.message.tool_calls[0].name, "search");
        assert_eq!(r.message.tool_calls[0].arguments, json!({"q":"rust"}));
    }

    #[test]
    fn embed_config_defaults_input_types() {
        let cfg = CohereEmbeddingsConfig::new("k", "embed-english-v3.0", 1024);
        assert_eq!(cfg.input_type_document, "search_document");
        assert_eq!(cfg.input_type_query, "search_query");
        assert_eq!(cfg.dimensions, 1024);
        let cfg = cfg.with_input_types("classification", "classification");
        assert_eq!(cfg.input_type_document, "classification");
        assert_eq!(cfg.input_type_query, "classification");
    }
}
