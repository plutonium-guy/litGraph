use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::{Message, Result, tool::ToolSchema};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub tools: Vec<ToolSchema>,
    /// Provider-native response_format / structured-output JSON Schema.
    pub response_format: Option<serde_json::Value>,
    /// Tool choice: "auto" | "none" | "required" | {"type":"function","function":{"name":"..."}}
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Other,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt: u32,
    pub completion: u32,
    pub total: u32,
    /// Anthropic prompt-cache write tokens (cost ~1.25× normal input).
    /// Zero for providers that don't support caching.
    #[serde(default)]
    pub cache_creation: u32,
    /// Anthropic prompt-cache read tokens (cost ~0.1× normal input).
    /// Zero for providers that don't support caching.
    #[serde(default)]
    pub cache_read: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    pub finish_reason: FinishReason,
    pub usage: TokenUsage,
    pub model: String,
}

/// Streaming event emitted during `stream(..)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatStreamEvent {
    /// Partial assistant text.
    Delta { text: String },
    /// Partial tool-call arguments — provider-specific; aggregate client-side.
    ToolCallDelta { index: u32, id: Option<String>, name: Option<String>, arguments_delta: Option<String> },
    /// Final message + usage.
    Done { response: ChatResponse },
}

pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatStreamEvent>> + Send>>;

#[async_trait]
pub trait ChatModel: Send + Sync {
    fn name(&self) -> &str;

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse>;

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream>;

    async fn batch(&self, inputs: Vec<Vec<Message>>, opts: &ChatOptions) -> Result<Vec<ChatResponse>> {
        // Default naive impl — providers override for true parallel batch.
        let mut out = Vec::with_capacity(inputs.len());
        for msgs in inputs {
            out.push(self.invoke(msgs, opts).await?);
        }
        Ok(out)
    }
}

#[async_trait]
pub trait Embeddings: Send + Sync {
    fn name(&self) -> &str;
    fn dimensions(&self) -> usize;

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}
