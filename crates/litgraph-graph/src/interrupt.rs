use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Fan-out command — used in `NodeOutput::sends` to spawn parallel sub-invocations
/// of a target node with different state slices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    pub goto: String,
    pub update: Value,
}

impl Command {
    pub fn to<S: Into<String>>(target: S) -> Self {
        Self { goto: target.into(), update: Value::Null }
    }

    pub fn with<S: serde::Serialize>(mut self, partial: S) -> Self {
        self.update = serde_json::to_value(partial).unwrap_or(Value::Null);
        self
    }
}

/// An interrupt suspends graph execution and surfaces a payload to the caller.
/// Resume by calling `CompiledGraph::resume(thread_id, resume_value)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interrupt {
    pub node: String,
    pub payload: Value,
}
