//! Token counting and message-history trimming.
//!
//! # Counting
//!
//! `count_tokens(model, text)` and `count_message_tokens(model, messages)` use
//! `tiktoken-rs` for OpenAI BPE encodings (`cl100k_base`, `o200k_base`, etc).
//! For non-OpenAI models we fall back to a `chars / 4` estimate — wrong by
//! 10-25% on average, but in the right ballpark for budgeting. Pass an
//! Anthropic / Gemini model name and you'll get the fallback automatically.
//!
//! # Trimming
//!
//! `trim_messages` keeps the system messages, then drops the OLDEST non-system
//! messages until the total token count is under `max_tokens`. Use to bound
//! conversation history before it blows the model's context window. This is
//! LangChain's `trim_messages` equivalent.

use std::sync::Arc;

use litgraph_core::{Message, Role};
use parking_lot::Mutex;
use tiktoken_rs::CoreBPE;

/// Resolve a model name to a `tiktoken-rs` encoding. Returns `None` for
/// non-OpenAI models — caller should fall back to `chars / 4`.
fn bpe_for(model: &str) -> Option<Arc<CoreBPE>> {
    use std::collections::HashMap;
    static CACHE: Mutex<Option<HashMap<String, Arc<CoreBPE>>>> = Mutex::new(None);

    let mut g = CACHE.lock();
    let cache = g.get_or_insert_with(HashMap::new);
    if let Some(b) = cache.get(model) {
        return Some(b.clone());
    }
    let bpe = tiktoken_rs::get_bpe_from_model(model).ok()?;
    let arc = Arc::new(bpe);
    cache.insert(model.to_string(), arc.clone());
    Some(arc)
}

pub fn count_tokens(model: &str, text: &str) -> usize {
    match bpe_for(model) {
        Some(bpe) => bpe.encode_with_special_tokens(text).len(),
        None => fallback_estimate(text),
    }
}

/// Approximate per-message token count modeled on OpenAI's accounting:
/// each message has a small overhead for the role + name + delimiters.
/// We use 4 tokens per message as the constant — close enough for budgeting.
const PER_MESSAGE_OVERHEAD: usize = 4;
const PER_REPLY_OVERHEAD: usize = 2;

pub fn count_message_tokens(model: &str, messages: &[Message]) -> usize {
    let mut total = 0;
    for m in messages {
        total += PER_MESSAGE_OVERHEAD;
        total += count_tokens(model, &m.text_content());
        for tc in &m.tool_calls {
            total += count_tokens(model, &tc.name);
            total += count_tokens(model, &tc.arguments.to_string());
        }
    }
    total + PER_REPLY_OVERHEAD
}

fn fallback_estimate(text: &str) -> usize {
    // 4 chars/token is the standard rule-of-thumb across modern BPE models.
    text.len().div_ceil(4)
}

/// Drop the OLDEST non-system messages until the total token count is under
/// `max_tokens`. Always preserves system messages (typically a system prompt
/// you must never lose). Always keeps at least the LAST message (so an empty
/// conversation isn't returned even under a tiny budget).
pub fn trim_messages(model: &str, messages: &[Message], max_tokens: usize) -> Vec<Message> {
    if count_message_tokens(model, messages) <= max_tokens {
        return messages.to_vec();
    }

    let (system, rest): (Vec<&Message>, Vec<&Message>) = messages
        .iter()
        .partition(|m| matches!(m.role, Role::System));

    // Always keep the last message — typically the latest user query.
    let last = rest.last().cloned();
    // Walk from oldest, dropping until we fit. We score by including all kept,
    // so this is O(n^2) over count — fine for typical history sizes.
    let mut kept: Vec<Message> = Vec::with_capacity(rest.len());
    let last_idx = rest.len().saturating_sub(1);
    for (i, m) in rest.iter().enumerate().rev() {
        let mut candidate = kept.clone();
        candidate.insert(0, (*m).clone());
        let mut combined: Vec<Message> = system.iter().map(|m| (*m).clone()).collect();
        combined.extend(candidate.iter().cloned());
        let cost = count_message_tokens(model, &combined);
        if cost <= max_tokens || i == last_idx {
            kept = candidate;
        } else {
            break;
        }
    }
    if kept.is_empty() {
        if let Some(l) = last { kept.push(l.clone()); }
    }

    let mut out: Vec<Message> = system.into_iter().cloned().collect();
    out.extend(kept);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_openai_model_tokens_via_tiktoken() {
        let n = count_tokens("gpt-4o", "hello world");
        assert!(n > 0 && n < 10, "tiktoken should give a small count, got {n}");
    }

    #[test]
    fn falls_back_to_chars_div_4_for_unknown_model() {
        let n = count_tokens("anthropic.claude-opus-4-7", "abcdefghij"); // 10 chars
        assert_eq!(n, 3); // (10+3)/4 = 3
    }

    #[test]
    fn count_message_tokens_includes_overhead() {
        let msgs = vec![Message::user("hi")];
        let n = count_message_tokens("gpt-4o", &msgs);
        // 4 (overhead) + 1 (hi) + 2 (reply overhead) = 7
        assert!(n >= 5 && n <= 12);
    }

    #[test]
    fn trim_drops_oldest_keeps_system_and_last() {
        // Construct enough messages to need trimming.
        let msgs = vec![
            Message::system("be terse"),
            Message::user("a".repeat(2000)),
            Message::assistant("b".repeat(2000)),
            Message::user("c".repeat(2000)),
            Message::user("d".repeat(2000)),
            Message::user("latest question"),
        ];
        let trimmed = trim_messages("anthropic.claude-test", &msgs, 200);
        // System message preserved
        assert!(matches!(trimmed[0].role, Role::System));
        // Last message preserved
        assert_eq!(trimmed.last().unwrap().text_content(), "latest question");
        // Some old messages were dropped
        assert!(trimmed.len() < msgs.len());
    }

    #[test]
    fn trim_returns_unchanged_when_under_budget() {
        let msgs = vec![Message::user("short")];
        let out = trim_messages("gpt-4o", &msgs, 1000);
        assert_eq!(out.len(), 1);
    }
}
