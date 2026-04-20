//! Anthropic Messages API adapter. Supports native tool use + streaming (SSE).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use futures_util::TryStreamExt;
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, ContentPart, Error, FinishReason, Message, Result, Role,
    TokenUsage,
    tool::ToolCall,
};
use litgraph_core::model::{ChatStream, ChatStreamEvent};
use reqwest::Client;
use serde_json::{Value, json};
use tracing::debug;

pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Tool name we synthesize when the caller asks for `response_format =
/// json_schema`. parse_response() unwraps the tool call back into the text
/// content so callers see structured JSON in the standard `text` field.
pub(crate) const SYNTHESIZED_TOOL_NAME: &str = "litgraph__submit_response";

/// Pre-flight inspector — see `litgraph_providers_openai::RequestInspector`.
pub type RequestInspector = Arc<dyn Fn(&str, &Value) + Send + Sync>;

#[derive(Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub anthropic_version: String,
    pub max_tokens: u32,
    pub on_request: Option<RequestInspector>,
}

impl std::fmt::Debug for AnthropicConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("timeout", &self.timeout)
            .field("anthropic_version", &self.anthropic_version)
            .field("max_tokens", &self.max_tokens)
            .field("on_request", &self.on_request.as_ref().map(|_| "<callback>"))
            .finish()
    }
}

impl AnthropicConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".into(),
            model: model.into(),
            timeout: Duration::from_secs(120),
            anthropic_version: ANTHROPIC_VERSION.into(),
            max_tokens: 4096,
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

pub struct AnthropicChat {
    cfg: AnthropicConfig,
    http: Client,
}

impl AnthropicChat {
    pub fn new(cfg: AnthropicConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    fn body(&self, messages: &[Message], opts: &ChatOptions, stream: bool) -> Value {
        // Anthropic requires `system` as a top-level field, not a message.
        let (system, rest): (Vec<&Message>, Vec<&Message>) = messages
            .iter()
            .partition(|m| matches!(m.role, Role::System));
        // System messages: emit as typed-block array so cache_control can be
        // attached. If any system message is marked cached, attach
        // cache_control to its text block (Anthropic's recommended breakpoint
        // pattern is "the end of the cached prefix").
        let max = opts.max_tokens.unwrap_or(self.cfg.max_tokens);
        let mut body = json!({
            "model": self.cfg.model,
            "max_tokens": max,
            "messages": rest.iter().map(|m| message_to_anthropic(m)).collect::<Vec<_>>(),
            "stream": stream,
        });
        if !system.is_empty() {
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

        // Build the tools array. User tools first, then optionally a synthesized
        // `submit_response` tool for structured-output forcing.
        let mut tools: Vec<Value> = opts.tools.iter().map(|t| json!({
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        })).collect();

        // Anthropic structured-output workaround: synthesize a single tool whose
        // input_schema is the user's schema, then force `tool_choice` to that
        // tool. The response's tool_use block carries the structured data.
        // parse_response() unwraps it back to text content so callers see JSON
        // in the standard `text` field — same shape as OpenAI / Gemini.
        if let Some(ref rf) = opts.response_format {
            if rf.get("type").and_then(|t| t.as_str()) == Some("json_schema") {
                let schema = rf.get("json_schema")
                    .and_then(|s| s.get("schema"))
                    .or_else(|| rf.get("json_schema"))
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object"}));
                tools.push(json!({
                    "name": SYNTHESIZED_TOOL_NAME,
                    "description": "Submit your final response. Always call this — never reply with plain text.",
                    "input_schema": schema,
                }));
                body["tool_choice"] = json!({
                    "type": "tool",
                    "name": SYNTHESIZED_TOOL_NAME,
                });
            }
        }

        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        body
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        if let Some(cb) = &self.cfg.on_request {
            cb(&self.cfg.model, body);
        }
        let url = format!("{}/messages", self.cfg.base_url);
        let resp = self
            .http
            .post(&url)
            .header("x-api-key", &self.cfg.api_key)
            .header("anthropic-version", &self.cfg.anthropic_version)
            .json(body)
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
impl ChatModel for AnthropicChat {
    fn name(&self) -> &str { &self.cfg.model }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let body = self.body(&messages, opts, false);
        debug!(model = %self.cfg.model, "anthropic invoke");
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
                                        agg_text.push_str(t);
                                        yield ChatStreamEvent::Delta { text: t.into() };
                                    }
                                }
                                Some("input_json_delta") => {
                                    if let Some(idx) = v.get("index").and_then(|v| v.as_u64()) {
                                        let idx = idx as usize;
                                        if let Some(buf) = tool_arg_bufs.get_mut(idx) {
                                            if let Some(pj) = delta.get("partial_json").and_then(|v| v.as_str()) {
                                                buf.push_str(pj);
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
                                usage.cache_creation = u.get("cache_creation_input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                usage.cache_read = u.get("cache_read_input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            }
                        }
                    }
                    _ => {}
                }
            }

            for (i, buf) in tool_arg_bufs.iter().enumerate() {
                if !buf.is_empty() {
                    tool_calls[i].arguments = serde_json::from_str(buf).unwrap_or(Value::String(buf.clone()));
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
                response: ChatResponse { message: msg, finish_reason: finish, usage, model },
            };
        };

        Ok(Box::pin(stream.map_err(|e: Error| e)))
    }
}

/// Attach `cache_control: {"type":"ephemeral"}` to the last content block in
/// `blocks` — Anthropic's recommended way to mark a cache breakpoint at the
/// end of a prefix.
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

    // Tool result messages need a `tool_result` block.
    if matches!(m.role, Role::Tool) {
        let mut blocks = vec![json!({
            "type": "tool_result",
            "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
            "content": m.text_content(),
        })];
        if m.cache { add_cache_breakpoint(&mut blocks); }
        return json!({"role": "user", "content": blocks});
    }

    // Assistant with tool calls → tool_use blocks alongside any text.
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

    // Plain text / multimodal.
    let mut blocks: Vec<Value> = if m.content.iter().all(|p| matches!(p, ContentPart::Text { .. })) {
        vec![json!({"type": "text", "text": m.text_content()})]
    } else {
        m.content.iter().map(|p| match p {
            ContentPart::Text { text } => json!({"type": "text", "text": text}),
            ContentPart::Image { source } => match source {
                litgraph_core::ImageSource::Url { url } => json!({"type": "image", "source": {"type": "url", "url": url}}),
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
        .ok_or_else(|| Error::provider("missing content blocks"))?;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut synthesized_struct_output: Option<Value> = None;
    for b in blocks {
        match b.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = b.get("text").and_then(|v| v.as_str()) { text.push_str(t); }
            }
            Some("tool_use") => {
                let name = b.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let input = b.get("input").cloned().unwrap_or(Value::Null);
                if name == SYNTHESIZED_TOOL_NAME {
                    // Unwrap structured-output workaround: caller asked for a
                    // schema, model called our synthesized tool — surface the
                    // args as the response text so the cross-provider API
                    // (text contains JSON) is consistent.
                    synthesized_struct_output = Some(input);
                } else {
                    tool_calls.push(ToolCall {
                        id: b.get("id").and_then(|v| v.as_str()).unwrap_or("").into(),
                        name,
                        arguments: input,
                    });
                }
            }
            _ => {}
        }
    }
    if let Some(args) = synthesized_struct_output.take() {
        // Replace text with serialized JSON args.
        text = args.to_string();
    }
    let mut finish = match v.get("stop_reason").and_then(|v| v.as_str()).unwrap_or("end_turn") {
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        _ => FinishReason::Other,
    };
    // If the only "tool call" was our synthesized one, we no longer have any
    // real tool calls — finish_reason must reflect Stop, not ToolCalls, or
    // the agent loop would try to dispatch a non-existent tool.
    if tool_calls.is_empty() && matches!(finish, FinishReason::ToolCalls) {
        finish = FinishReason::Stop;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn anthropic() -> AnthropicChat {
        AnthropicChat::new(AnthropicConfig::new("k", "claude-opus-4-7-1m")).unwrap()
    }

    #[test]
    fn cached_system_message_emits_typed_block_array_with_cache_control() {
        let chat = anthropic();
        let msgs = vec![
            Message::system("you are a helpful assistant. <very long context here>").cached(),
            Message::user("hi"),
        ];
        let body = chat.body(&msgs, &ChatOptions::default(), false);
        // System must be a typed-block array (not a flat string) so cache_control attaches.
        let sys = body["system"].as_array().expect("system must be a typed-block array");
        assert_eq!(sys.len(), 1);
        assert_eq!(sys[0]["type"], json!("text"));
        assert_eq!(sys[0]["cache_control"], json!({"type": "ephemeral"}));
    }

    #[test]
    fn uncached_system_message_omits_cache_control() {
        let chat = anthropic();
        let msgs = vec![Message::system("plain"), Message::user("hi")];
        let body = chat.body(&msgs, &ChatOptions::default(), false);
        let sys = body["system"].as_array().unwrap();
        assert!(sys[0].get("cache_control").is_none());
    }

    #[test]
    fn cached_user_message_attaches_cache_control_to_last_block() {
        let chat = anthropic();
        let msgs = vec![Message::user("big context to cache").cached()];
        let body = chat.body(&msgs, &ChatOptions::default(), false);
        let blocks = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.last().unwrap()["cache_control"], json!({"type": "ephemeral"}));
    }

    #[test]
    fn parse_response_surfaces_cache_creation_and_read_tokens() {
        let v = json!({
            "content": [{"type":"text","text":"hello"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 1500,
                "cache_read_input_tokens": 500,
            }
        });
        let r = parse_response("claude-opus-4-7-1m", v).unwrap();
        assert_eq!(r.usage.cache_creation, 1500);
        assert_eq!(r.usage.cache_read, 500);
        assert_eq!(r.usage.prompt, 10);
        assert_eq!(r.usage.completion, 5);
    }
}
