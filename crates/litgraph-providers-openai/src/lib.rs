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

// ---------- OpenAI Responses API (iter 83) ----------
//
// `/v1/responses` is OpenAI's new agentic endpoint (replaces `/chat/completions`
// for new code). Different request/response shape than chat completions:
// - Request body has `input` (messages or string), `instructions`,
//   `previous_response_id` for stateful conversations, and `tools` (incl.
//   built-in server-side tools like `web_search`, `file_search`,
//   `code_interpreter` that don't require a client-side tool loop).
// - Response body has `output` (array of items: messages, function_call,
//   web_search_call, etc.) and `usage` with `input_tokens` / `output_tokens`
//   instead of `prompt_tokens` / `completion_tokens`.
//
// We expose `OpenAIResponses` as a separate ChatModel impl. Function tools
// flow through identically to chat completions (parsed into
// `ChatResponse.message.tool_calls`); built-in server-side tools execute on
// OpenAI's side and produce no tool_calls — the response is just text.
// `previous_response_id` is exposed via `with_previous_response_id(id)` for
// stateful chains. Stream support deferred (Responses SSE has its own event
// shape and isn't bit-compat with chat completions SSE).

/// Configuration for the OpenAI Responses API. Same auth + base URL knobs as
/// `OpenAIConfig`; adds `previous_response_id` for stateful conversations
/// (the server-side equivalent of replaying message history).
#[derive(Clone)]
pub struct OpenAIResponsesConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub organization: Option<String>,
    pub on_request: Option<RequestInspector>,
    /// Optional `instructions` (system-prompt-like, but server-attached for
    /// stateful chains). Sent on every call.
    pub instructions: Option<String>,
    /// Continuation token from a prior response — when set, OpenAI replays
    /// that conversation's server-side context instead of requiring the
    /// caller to re-send all messages.
    pub previous_response_id: Option<String>,
}

impl std::fmt::Debug for OpenAIResponsesConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIResponsesConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("timeout", &self.timeout)
            .field("organization", &self.organization)
            .field("on_request", &self.on_request.as_ref().map(|_| "<callback>"))
            .field("instructions", &self.instructions)
            .field("previous_response_id", &self.previous_response_id)
            .finish()
    }
}

impl OpenAIResponsesConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".into(),
            model: model.into(),
            timeout: Duration::from_secs(120),
            organization: None,
            on_request: None,
            instructions: None,
            previous_response_id: None,
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self { self.base_url = url.into(); self }
    pub fn with_instructions(mut self, s: impl Into<String>) -> Self { self.instructions = Some(s.into()); self }
    pub fn with_previous_response_id(mut self, id: impl Into<String>) -> Self {
        self.previous_response_id = Some(id.into()); self
    }
    pub fn with_on_request<F>(mut self, f: F) -> Self
    where F: Fn(&str, &Value) + Send + Sync + 'static,
    { self.on_request = Some(Arc::new(f)); self }
}

pub struct OpenAIResponses {
    cfg: OpenAIResponsesConfig,
    http: Client,
}

impl OpenAIResponses {
    pub fn new(cfg: OpenAIResponsesConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    /// Returns the most recent response id (None until invoke is called).
    /// Caller should snapshot this and feed it into
    /// `with_previous_response_id()` for the next turn of a stateful chain.
    pub fn config(&self) -> &OpenAIResponsesConfig { &self.cfg }

    fn build_body(&self, messages: &[Message], opts: &ChatOptions) -> Value {
        // The Responses API `input` accepts either a string or an array of
        // input items. We always send an array so multi-turn message
        // history works without `previous_response_id`.
        let input: Vec<Value> = messages
            .iter()
            .filter_map(message_to_responses_input)
            .collect();
        let mut body = json!({
            "model": self.cfg.model,
            "input": input,
        });
        if let Some(ref ins) = self.cfg.instructions {
            body["instructions"] = json!(ins);
        }
        if let Some(ref prev) = self.cfg.previous_response_id {
            body["previous_response_id"] = json!(prev);
        }
        if let Some(t) = opts.temperature { body["temperature"] = json!(t); }
        if let Some(t) = opts.top_p { body["top_p"] = json!(t); }
        if let Some(t) = opts.max_tokens { body["max_output_tokens"] = json!(t); }
        if !opts.tools.is_empty() {
            // Responses API tool shape: function tools use `type: "function"`
            // with name/description/parameters at the same level (NOT nested
            // under `function`, unlike chat completions).
            body["tools"] = json!(opts.tools.iter().map(|t| json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            })).collect::<Vec<_>>());
        }
        if let Some(ref tc) = opts.tool_choice {
            body["tool_choice"] = tc.clone();
        }
        body
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        if let Some(cb) = &self.cfg.on_request {
            cb(&self.cfg.model, body);
        }
        let url = format!("{}/responses", self.cfg.base_url);
        let mut req = self.http.post(&url).bearer_auth(&self.cfg.api_key).json(body);
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
impl ChatModel for OpenAIResponses {
    fn name(&self) -> &str { &self.cfg.model }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let body = self.build_body(&messages, opts);
        debug!(model = %self.cfg.model, "openai responses invoke");
        let resp = self.post(&body).await?;
        let json: Value = resp.json().await.map_err(|e| Error::provider(format!("decode: {e}")))?;
        parse_responses_response(&self.cfg.model, json)
    }

    async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
        // Responses API SSE has a different event taxonomy than chat
        // completions (response.output_text.delta, response.completed, etc.).
        // Wire up later — for v1, fall back to non-streaming invoke().
        Err(Error::provider(
            "OpenAIResponses streaming not yet implemented; use invoke() or OpenAIChat::stream",
        ))
    }
}

/// Convert a `Message` into a Responses-API input item. Returns `None` for
/// roles that map to non-input contexts (e.g. assistant tool_call markers
/// that are echoed via `previous_response_id` instead of message history).
fn message_to_responses_input(m: &Message) -> Option<Value> {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        // Tool responses in the Responses API have a dedicated input-item
        // shape: `{type:"function_call_output", call_id, output}`.
        Role::Tool => {
            return Some(json!({
                "type": "function_call_output",
                "call_id": m.tool_call_id.clone().unwrap_or_default(),
                "output": m.text_content(),
            }));
        }
    };
    let parts: Vec<Value> = m.content.iter().map(|p| match p {
        ContentPart::Text { text } => {
            // Assistant messages use `output_text`; user/system use `input_text`.
            let kind = if matches!(m.role, Role::Assistant) { "output_text" } else { "input_text" };
            json!({ "type": kind, "text": text })
        }
        ContentPart::Image { source } => match source {
            litgraph_core::ImageSource::Url { url } => {
                json!({ "type": "input_image", "image_url": url })
            }
            litgraph_core::ImageSource::Base64 { media_type, data } => {
                json!({
                    "type": "input_image",
                    "image_url": format!("data:{};base64,{}", media_type, data),
                })
            }
        },
    }).collect();
    Some(json!({ "role": role, "content": parts }))
}

fn parse_responses_response(model: &str, v: Value) -> Result<ChatResponse> {
    // The output array contains one or more items. Aggregate all
    // `output_text` content from `message` items into a single text body and
    // collect `function_call` items as tool_calls.
    let output = v
        .get("output")
        .and_then(|o| o.as_array())
        .ok_or_else(|| Error::provider("responses: missing output"))?;

    let mut text_parts = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for item in output {
        let kind = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match kind {
            "message" => {
                if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
                    for c in content {
                        let ctype = c.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if ctype == "output_text" || ctype == "text" {
                            if let Some(t) = c.get("text").and_then(|t| t.as_str()) {
                                text_parts.push_str(t);
                            }
                        }
                    }
                }
            }
            "function_call" => {
                let id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let args_str = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = serde_json::from_str(args_str).unwrap_or(Value::Null);
                tool_calls.push(ToolCall { id, name, arguments });
            }
            // Other item types (web_search_call, file_search_call, etc.)
            // are server-side tool invocations — we ignore them; the
            // resulting message item carries the model's final text.
            _ => {}
        }
    }

    let finish = if !tool_calls.is_empty() {
        FinishReason::ToolCalls
    } else {
        match v.get("status").and_then(|s| s.as_str()).unwrap_or("completed") {
            "incomplete" => FinishReason::Length,
            _ => FinishReason::Stop,
        }
    };

    let usage = v.get("usage").map(|u| TokenUsage {
        prompt: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        completion: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        total: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        cache_creation: 0,
        cache_read: 0,
    }).unwrap_or_default();

    Ok(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: if text_parts.is_empty() { vec![] } else { vec![ContentPart::Text { text: text_parts }] },
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

    // ---------- Responses API (iter 83) ----------

    use litgraph_core::tool::ToolSchema;

    #[test]
    fn responses_request_body_uses_input_array_and_responses_tool_shape() {
        let cfg = OpenAIResponsesConfig::new("sk-fake", "gpt-4o");
        let model = OpenAIResponses::new(cfg).unwrap();
        let opts = ChatOptions {
            temperature: Some(0.5),
            max_tokens: Some(256),
            tools: vec![ToolSchema {
                name: "lookup".into(),
                description: "Look something up.".into(),
                parameters: json!({"type":"object","properties":{"q":{"type":"string"}}}),
            }],
            ..Default::default()
        };
        let body = model.build_body(
            &[Message::system("be brief"), Message::user("hi")],
            &opts,
        );
        // input is array, not string.
        let input = body["input"].as_array().expect("input must be array");
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["role"], "system");
        // System/user text content uses `input_text` type.
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "be brief");
        assert_eq!(input[1]["role"], "user");
        // max_tokens maps to max_output_tokens (Responses naming).
        assert_eq!(body["max_output_tokens"], json!(256));
        // Tool shape: name/description/parameters at top level (NOT nested
        // under `function` like chat completions).
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "lookup");
        assert_eq!(tools[0]["description"], "Look something up.");
        assert!(tools[0]["parameters"].is_object());
    }

    #[test]
    fn responses_request_body_includes_instructions_and_previous_response_id() {
        let cfg = OpenAIResponsesConfig::new("sk-fake", "gpt-4o")
            .with_instructions("You are a helpful tutor.")
            .with_previous_response_id("resp_abc123");
        let model = OpenAIResponses::new(cfg).unwrap();
        let body = model.build_body(&[Message::user("continue")], &ChatOptions::default());
        assert_eq!(body["instructions"], "You are a helpful tutor.");
        assert_eq!(body["previous_response_id"], "resp_abc123");
    }

    #[test]
    fn responses_request_omits_instructions_and_prev_id_when_unset() {
        let cfg = OpenAIResponsesConfig::new("sk-fake", "gpt-4o");
        let model = OpenAIResponses::new(cfg).unwrap();
        let body = model.build_body(&[Message::user("hi")], &ChatOptions::default());
        assert!(body.get("instructions").is_none());
        assert!(body.get("previous_response_id").is_none());
    }

    #[test]
    fn responses_tool_role_maps_to_function_call_output_input_item() {
        // After a tool runs, the agent appends a Role::Tool message; the
        // Responses API expects this as a `function_call_output` input item
        // with the call_id echoed back.
        let mut tool_msg = Message::tool_response("call_xyz", "42");
        // Sanity: helper sets tool_call_id correctly.
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call_xyz"));
        tool_msg.tool_call_id = Some("call_xyz".into());
        let item = message_to_responses_input(&tool_msg).unwrap();
        assert_eq!(item["type"], "function_call_output");
        assert_eq!(item["call_id"], "call_xyz");
        assert_eq!(item["output"], "42");
    }

    #[test]
    fn parse_responses_extracts_text_message_content() {
        let v = json!({
            "id": "resp_1",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello, world!"}
                ]
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let resp = parse_responses_response("gpt-4o", v).unwrap();
        assert_eq!(resp.message.text_content(), "Hello, world!");
        assert_eq!(resp.usage.prompt, 10);
        assert_eq!(resp.usage.completion, 5);
        assert_eq!(resp.usage.total, 15);
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
        assert!(resp.message.tool_calls.is_empty());
    }

    #[test]
    fn parse_responses_extracts_function_call_into_tool_calls() {
        let v = json!({
            "id": "resp_2",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "call_id": "call_42",
                "name": "add",
                "arguments": "{\"a\": 4, \"b\": 5}"
            }],
            "usage": {"input_tokens": 8, "output_tokens": 12, "total_tokens": 20}
        });
        let resp = parse_responses_response("gpt-4o", v).unwrap();
        assert!(matches!(resp.finish_reason, FinishReason::ToolCalls));
        assert_eq!(resp.message.tool_calls.len(), 1);
        let tc = &resp.message.tool_calls[0];
        assert_eq!(tc.id, "call_42");
        assert_eq!(tc.name, "add");
        assert_eq!(tc.arguments["a"], 4);
        assert_eq!(tc.arguments["b"], 5);
    }

    #[test]
    fn parse_responses_handles_message_and_function_call_in_same_response() {
        // Responses API can interleave a function_call AND a textual message
        // (e.g. "Let me look that up..."). Both must surface — text into
        // content, function_call into tool_calls.
        let v = json!({
            "id": "resp_3",
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Looking up..."}
                ]},
                {"type": "function_call", "call_id": "c1", "name": "search",
                 "arguments": "{\"q\":\"rust async\"}"}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}
        });
        let resp = parse_responses_response("gpt-4o", v).unwrap();
        assert_eq!(resp.message.text_content(), "Looking up...");
        assert_eq!(resp.message.tool_calls.len(), 1);
        assert!(matches!(resp.finish_reason, FinishReason::ToolCalls));
    }

    #[test]
    fn parse_responses_ignores_unknown_output_item_types() {
        // Server-side built-in tools (web_search_call, file_search_call,
        // code_interpreter_call) appear as their own item types — the actual
        // model response is in a separate `message` item. We ignore the
        // server-side call records and just collect message text.
        let v = json!({
            "id": "resp_4",
            "status": "completed",
            "output": [
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Per my web search: 42."}
                ]}
            ],
            "usage": {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10}
        });
        let resp = parse_responses_response("gpt-4o", v).unwrap();
        assert_eq!(resp.message.text_content(), "Per my web search: 42.");
        assert!(resp.message.tool_calls.is_empty());
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
    }

    #[tokio::test]
    async fn responses_inspector_fires_with_responses_endpoint_body() {
        let captured: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let cfg = OpenAIResponsesConfig::new("sk-fake", "gpt-4o")
            .with_base_url("http://127.0.0.1:1")
            .with_on_request(move |model, body| {
                assert_eq!(model, "gpt-4o");
                *cap.lock().unwrap() = Some(body.clone());
            });
        let model = OpenAIResponses::new(cfg).unwrap();
        let _ = model.invoke(vec![Message::user("hi")], &ChatOptions::default()).await;
        let body = captured.lock().unwrap().take().expect("inspector ran");
        // No `messages` key (chat completions); has `input` (responses).
        assert!(body.get("messages").is_none());
        assert!(body.get("input").is_some());
        // No `stream` key (we don't set it for responses; default off).
        assert!(body.get("stream").is_none());
    }
}
