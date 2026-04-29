use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Events emitted during graph execution. Streaming consumers subscribe to these.
///
/// Stream modes (LangGraph parity):
/// - `values` — full state after each step (derived from `NodeEnd`)
/// - `updates` — the raw partial update per node (`NodeEnd`)
/// - `messages` — provider token deltas (emitted by provider adapters via `Custom`)
/// - `custom` — user-emitted `Custom` payloads
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GraphEvent {
    GraphStart { thread_id: String },
    NodeStart { node: String, step: u64 },
    NodeEnd { node: String, step: u64, update: Value },
    NodeError { node: String, step: u64, error: String },
    Interrupt { node: String, step: u64, payload: Value },
    Custom { node: String, name: String, payload: Value },
    GraphEnd { thread_id: String, steps: u64 },
}
