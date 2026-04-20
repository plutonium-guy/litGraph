//! OpenAI / OpenAI-compatible chat adapter — chat completions, streaming (SSE),
//! native tool calling. Bring-your-own base URL covers Ollama / vLLM / Together / Groq
//! / Fireworks / DeepSeek / LM Studio etc.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use futures_util::TryStreamExt;
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, ContentPart, Embeddings, Error, FinishReason, Message,
    Result, Role, TokenUsage,
    tool::ToolCall,
};
use litgraph_core::model::{ChatStream, ChatStreamEvent};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::debug;

/// Pre-flight inspector: receives the final request body just before it hits
/// the network. Use to log, snapshot, or assert in tests. Closure must be
/// `Send + Sync` since providers are shared across tasks.
pub type RequestInspector = Arc<dyn Fn(&str, &Value) + Send + Sync>;

#[derive(Clone)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub organization: Option<String>,
    /// Optional `(model, body) -> ()` callback fired before each HTTP send.
    /// One of FEATURES.md's "solves 50% of debugging pain" hooks.
    pub on_request: Option<RequestInspector>,
}

impl std::fmt::Debug for OpenAIConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("timeout", &self.timeout)
            .field("organization", &self.organization)
            .field("on_request", &self.on_request.as_ref().map(|_| "<callback>"))
            .finish()
    }
}

impl OpenAIConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".into(),
            model: model.into(),
            timeout: Duration::from_secs(120),
            organization: None,
            on_request: None,
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    /// Install a request inspector. Replaces any previously-set callback.
    pub fn with_on_request<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &Value) + Send + Sync + 'static,
    {
        self.on_request = Some(Arc::new(f));
        self
    }
}

pub struct OpenAIChat {
    cfg: OpenAIConfig,
    http: Client,
}

impl OpenAIChat {
    pub fn new(cfg: OpenAIConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    fn body(&self, messages: &[Message], opts: &ChatOptions, stream: bool) -> Value {
        let mut body = json!({
            "model": self.cfg.model,
            "messages": messages.iter().map(message_to_openai).collect::<Vec<_>>(),
            "stream": stream,
        });
        if let Some(t) = opts.temperature { body["temperature"] = json!(t); }
        if let Some(t) = opts.top_p { body["top_p"] = json!(t); }
        if let Some(t) = opts.max_tokens { body["max_tokens"] = json!(t); }
        if let Some(ref s) = opts.stop { body["stop"] = json!(s); }
        if let Some(s) = opts.seed { body["seed"] = json!(s); }
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
        if let Some(ref rf) = opts.response_format {
            body["response_format"] = rf.clone();
        }
        if let Some(ref tc) = opts.tool_choice {
            body["tool_choice"] = tc.clone();
        }
        body
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        // Fire the inspector before any network I/O. Errors in the inspector
        // would be a user bug — we let them propagate as panics so they're noisy.
        if let Some(cb) = &self.cfg.on_request {
            cb(&self.cfg.model, body);
        }
        let url = format!("{}/chat/completions", self.cfg.base_url);
        let mut req = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .json(body);
        if let Some(ref org) = self.cfg.organization {
            req = req.header("OpenAI-Organization", org);
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
        Ok(resp)
    }
}

#[async_trait]
impl ChatModel for OpenAIChat {
    fn name(&self) -> &str { &self.cfg.model }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let body = self.body(&messages, opts, false);
        debug!(model = %self.cfg.model, "openai invoke");
        let resp = self.post(&body).await?;
        let json: Value = resp.json().await.map_err(|e| Error::provider(format!("decode: {e}")))?;
        parse_response(&self.cfg.model, json)
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        let body = self.body(&messages, opts, true);
        let resp = self.post(&body).await?;
        let model_name = self.cfg.model.clone();
        let sse = resp.bytes_stream().eventsource();

        // Aggregate partial tool-call JSON fragments keyed by index.
        let mut agg_text = String::new();
        let mut agg_tools: Vec<ToolCall> = Vec::new();
        let mut tool_arg_bufs: Vec<String> = Vec::new();
        let mut final_reason = FinishReason::Stop;
        let mut final_usage = TokenUsage::default();

        let stream = async_stream::try_stream! {
            let mut sse = sse;
            while let Some(ev) = sse.next().await {
                let ev = ev.map_err(|e| Error::provider(format!("sse: {e}")))?;
                if ev.data == "[DONE]" { break; }
                let v: Value = serde_json::from_str(&ev.data)
                    .map_err(|e| Error::provider(format!("sse json: {e}")))?;
                if let Some(choice) = v.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first()) {
                    if let Some(delta) = choice.get("delta") {
                        if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
                            if !text.is_empty() {
                                agg_text.push_str(text);
                                yield ChatStreamEvent::Delta { text: text.into() };
                            }
                        }
                        if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                            for tc in tcs {
                                let idx = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                                while agg_tools.len() <= idx {
                                    agg_tools.push(ToolCall { id: String::new(), name: String::new(), arguments: Value::Null });
                                    tool_arg_bufs.push(String::new());
                                }
                                let cur = &mut agg_tools[idx];
                                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                    if !id.is_empty() { cur.id = id.into(); }
                                }
                                if let Some(func) = tc.get("function") {
                                    if let Some(n) = func.get("name").and_then(|v| v.as_str()) {
                                        if !n.is_empty() { cur.name = n.into(); }
                                    }
                                    if let Some(args_chunk) = func.get("arguments").and_then(|v| v.as_str()) {
                                        tool_arg_bufs[idx].push_str(args_chunk);
                                    }
                                }
                                yield ChatStreamEvent::ToolCallDelta {
                                    index: idx as u32,
                                    id: tc.get("id").and_then(|v| v.as_str()).map(String::from),
                                    name: tc.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()).map(String::from),
                                    arguments_delta: tc.get("function").and_then(|f| f.get("arguments")).and_then(|v| v.as_str()).map(String::from),
                                };
                            }
                        }
                    }
                    if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                        final_reason = match fr {
                            "stop" => FinishReason::Stop,
                            "length" => FinishReason::Length,
                            "tool_calls" | "function_call" => FinishReason::ToolCalls,
                            "content_filter" => FinishReason::ContentFilter,
                            _ => FinishReason::Other,
                        };
                    }
                }
                if let Some(u) = v.get("usage") {
                    final_usage.prompt = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    final_usage.completion = u.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    final_usage.total = u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                }
            }

            // Finalize aggregated tool calls.
            for (i, buf) in tool_arg_bufs.iter().enumerate() {
                if !buf.is_empty() {
                    agg_tools[i].arguments = serde_json::from_str(buf).unwrap_or(Value::String(buf.clone()));
                }
            }

            let msg = Message {
                role: Role::Assistant,
                content: if agg_text.is_empty() { vec![] } else { vec![ContentPart::Text { text: agg_text.clone() }] },
                tool_calls: agg_tools,
                tool_call_id: None,
                name: None,
                cache: false,
            };
            yield ChatStreamEvent::Done {
                response: ChatResponse {
                    message: msg,
                    finish_reason: final_reason,
                    usage: final_usage,
                    model: model_name,
                },
            };
        };

        Ok(Box::pin(stream.map_err(|e: Error| e)))
    }
}

fn message_to_openai(m: &Message) -> Value {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };

    // Simple path: all-text content → string. Multimodal → array.
    let content: Value = if m.content.iter().all(|p| matches!(p, ContentPart::Text { .. })) {
        Value::String(m.text_content())
    } else {
        Value::Array(m.content.iter().map(|p| match p {
            ContentPart::Text { text } => json!({"type": "text", "text": text}),
            ContentPart::Image { source } => match source {
                litgraph_core::ImageSource::Url { url } => json!({"type": "image_url", "image_url": {"url": url}}),
                litgraph_core::ImageSource::Base64 { media_type, data } => json!({
                    "type": "image_url",
                    "image_url": { "url": format!("data:{};base64,{}", media_type, data) }
                }),
            },
        }).collect())
    };

    let mut out = json!({ "role": role, "content": content });
    if !m.tool_calls.is_empty() {
        out["tool_calls"] = json!(m.tool_calls.iter().map(|tc| json!({
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": tc.arguments.to_string(),
            }
        })).collect::<Vec<_>>());
    }
    if let Some(ref id) = m.tool_call_id {
        out["tool_call_id"] = json!(id);
    }
    if let Some(ref name) = m.name {
        out["name"] = json!(name);
    }
    out
}

fn parse_response(model: &str, v: Value) -> Result<ChatResponse> {
    let choice = v
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| Error::provider("missing choices"))?;
    let msg = choice.get("message").ok_or_else(|| Error::provider("missing message"))?;
    let text = msg.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
    let tool_calls: Vec<ToolCall> = msg
        .get("tool_calls")
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    Some(ToolCall {
                        id: tc.get("id")?.as_str()?.to_string(),
                        name: tc.get("function")?.get("name")?.as_str()?.to_string(),
                        arguments: serde_json::from_str(
                            tc.get("function")?.get("arguments")?.as_str()?,
                        )
                        .unwrap_or(Value::Null),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let finish = match choice.get("finish_reason").and_then(|v| v.as_str()).unwrap_or("stop") {
        "length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        "stop" => FinishReason::Stop,
        _ => FinishReason::Other,
    };

    let usage = v
        .get("usage")
        .map(|u| TokenUsage {
            prompt: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            completion: u.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            total: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
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
        finish_reason: finish,
        usage,
        model: model.to_string(),
    })
}

// =================== Embeddings ===================

/// OpenAI Embeddings adapter (`/embeddings` endpoint). Default `dimensions`
/// is supplied at construction so downstream vector stores can be sized
/// without an extra API round-trip. `text-embedding-3-*` models accept a
/// `dimensions` field that trims the returned vectors — set
/// `with_override_dimensions(N)` to use it.
#[derive(Clone)]
pub struct OpenAIEmbeddingsConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub dimensions: usize,
    pub override_dimensions: Option<usize>,
}

impl OpenAIEmbeddingsConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>, dimensions: usize) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".into(),
            model: model.into(),
            timeout: Duration::from_secs(120),
            dimensions,
            override_dimensions: None,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_override_dimensions(mut self, n: usize) -> Self {
        self.override_dimensions = Some(n);
        self.dimensions = n;
        self
    }
}

pub struct OpenAIEmbeddings {
    cfg: OpenAIEmbeddingsConfig,
    http: Client,
}

impl OpenAIEmbeddings {
    pub fn new(cfg: OpenAIEmbeddingsConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    async fn embed_batch(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.cfg.base_url);
        let mut body = json!({ "model": self.cfg.model, "input": inputs });
        if let Some(d) = self.cfg.override_dimensions {
            body["dimensions"] = json!(d);
        }
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
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
        let data = v
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| Error::provider("embeddings: missing `data`"))?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let emb = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| Error::provider("embeddings: missing `embedding`"))?;
            out.push(emb.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect());
        }
        Ok(out)
    }
}

#[async_trait]
impl Embeddings for OpenAIEmbeddings {
    fn name(&self) -> &str { &self.cfg.model }
    fn dimensions(&self) -> usize { self.cfg.dimensions }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed_batch(vec![text.to_string()]).await?;
        out.pop().ok_or_else(|| Error::provider("embed_query: empty result"))
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(vec![]); }
        self.embed_batch(texts.to_vec()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[tokio::test]
    async fn on_request_inspector_sees_final_body_with_messages() {
        let captured: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let cfg = OpenAIConfig::new("sk-fake", "gpt-fake")
            .with_base_url("http://127.0.0.1:1") // unreachable; we only want the hook to fire
            .with_on_request(move |model, body| {
                assert_eq!(model, "gpt-fake");
                *cap.lock().unwrap() = Some(body.clone());
            });
        let chat = OpenAIChat::new(cfg).unwrap();
        // Send fails (no server) but the inspector ran first.
        let _ = chat
            .invoke(
                vec![Message::user("hello inspector")],
                &ChatOptions { temperature: Some(0.7), ..Default::default() },
            )
            .await;
        let body = captured.lock().unwrap().take().expect("inspector should have fired");
        assert_eq!(body["model"], json!("gpt-fake"));
        assert_eq!(body["messages"][0]["role"], json!("user"));
        assert_eq!(body["messages"][0]["content"], json!("hello inspector"));
        let t = body["temperature"].as_f64().expect("temperature is a number");
        assert!((t - 0.7).abs() < 1e-3, "temperature ≈ 0.7, got {t}");
        assert_eq!(body["stream"], json!(false));
    }
}
