use std::time::SystemTime;

use litgraph_core::model::TokenUsage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase { Start, End, Error }

/// Structured event emitted by providers, tools, and graph nodes. Kept flat and
/// cheap to clone — the batched callback bus may clone millions per minute on a
/// busy server, so we avoid heavy containers here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    /// LLM call lifecycle.
    Llm {
        phase: Phase,
        model: String,
        usage: Option<TokenUsage>,
        error: Option<String>,
        ts_ms: u64,
    },
    /// Streaming token delta (use sparingly — can be high volume).
    LlmToken {
        model: String,
        text: String,
        ts_ms: u64,
    },
    /// Tool call lifecycle.
    Tool {
        phase: Phase,
        name: String,
        args: Option<Value>,
        error: Option<String>,
        duration_ms: Option<u64>,
        ts_ms: u64,
    },
    /// Graph node lifecycle — bridges from `litgraph_graph::GraphEvent`.
    Node {
        phase: Phase,
        node: String,
        step: u64,
        error: Option<String>,
        ts_ms: u64,
    },
    /// Free-form user event — anything the app wants to trace.
    Custom { name: String, payload: Value, ts_ms: u64 },
}

impl Event {
    pub fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}
