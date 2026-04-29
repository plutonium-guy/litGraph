use litgraph_core::{ChatOptions, Message};

/// Deterministic cache key for an LLM call.
///
/// Hashes: model name, JSON-serialized messages, JSON-serialized options.
/// Output: hex-encoded blake3 (32 bytes → 64 hex chars).
pub fn cache_key(model: &str, messages: &[Message], opts: &ChatOptions) -> String {
    let msgs_json = serde_json::to_vec(messages).unwrap_or_default();
    let opts_json = serde_json::to_vec(opts).unwrap_or_default();
    let mut h = blake3::Hasher::new();
    h.update(b"litgraph-cache-v1\0");
    h.update(model.as_bytes());
    h.update(b"\0");
    h.update(&(msgs_json.len() as u64).to_le_bytes());
    h.update(&msgs_json);
    h.update(&(opts_json.len() as u64).to_le_bytes());
    h.update(&opts_json);
    let hash = h.finalize();
    hash.to_hex().to_string()
}
