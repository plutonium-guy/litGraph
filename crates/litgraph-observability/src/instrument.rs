//! `InstrumentedChatModel` — wraps any `ChatModel`, emits lifecycle events into
//! a `CallbackHandle`. Users who don't want to modify every provider get
//! observability for free by wrapping once at the top level.
//!
//! Iter 385 wires trace exemplars (prompt + completion excerpts) onto a
//! `tracing::info_span!` opened around every `invoke`. The span fires
//! through whichever subscriber the user has installed — OTel via
//! `litgraph-tracing-otel`, or any stock `tracing-subscriber`. When
//! the span is disabled (e.g. no subscriber, env filter excludes it)
//! the `record` calls are no-ops, so the hot path stays cheap.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Message, Result};
use tracing::{Span, field, info_span};

use crate::callback::CallbackHandle;
use crate::event::{Event, Phase};

/// Maximum bytes per excerpt attached as a span attribute. Mirrors
/// `litgraph_tracing_otel::exemplars::MAX_EXCERPT_BYTES`. Tunable at
/// startup via `LITGRAPH_EXEMPLAR_BYTES`.
const MAX_EXCERPT_BYTES: usize = 512;

pub struct InstrumentedChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub events: CallbackHandle,
}

impl InstrumentedChatModel {
    pub fn new(inner: Arc<dyn ChatModel>, events: CallbackHandle) -> Self {
        Self { inner, events }
    }
}

#[async_trait]
impl ChatModel for InstrumentedChatModel {
    fn name(&self) -> &str { self.inner.name() }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let model = self.inner.name().to_string();
        // Span declared with `Empty` fields so we can `record` them
        // post-hoc with the prompt + completion excerpts. If the
        // subscriber filters this level out the span is disabled and
        // `record` becomes a no-op.
        let span = info_span!(
            "chat.invoke",
            model = %model,
            prompt_excerpt = field::Empty,
            completion_excerpt = field::Empty,
        );
        let _enter = span.enter();
        record_excerpt(&span, "prompt_excerpt", &render_prompt_excerpt(&messages));

        self.events.emit(Event::Llm {
            phase: Phase::Start,
            model: model.clone(),
            usage: None,
            error: None,
            ts_ms: Event::now_ms(),
        });
        match self.inner.invoke(messages, opts).await {
            Ok(resp) => {
                record_excerpt(
                    &span,
                    "completion_excerpt",
                    &resp.message.text_content(),
                );
                self.events.emit(Event::Llm {
                    phase: Phase::End,
                    model,
                    usage: Some(resp.usage),
                    error: None,
                    ts_ms: Event::now_ms(),
                });
                Ok(resp)
            }
            Err(e) => {
                self.events.emit(Event::Llm {
                    phase: Phase::Error,
                    model,
                    usage: None,
                    error: Some(e.to_string()),
                    ts_ms: Event::now_ms(),
                });
                Err(e)
            }
        }
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        // Stream bypass — emit only Start; downstream stream handler should emit LlmToken
        // + the terminal End. Prompt exemplar still useful for tying a
        // streaming call to its prompt; completion lands per-chunk on
        // the stream subscriber, not via this span.
        let model = self.inner.name().to_string();
        let span = info_span!(
            "chat.stream",
            model = %model,
            prompt_excerpt = field::Empty,
        );
        let _enter = span.enter();
        record_excerpt(&span, "prompt_excerpt", &render_prompt_excerpt(&messages));
        self.events.emit(Event::Llm {
            phase: Phase::Start,
            model,
            usage: None,
            error: None,
            ts_ms: Event::now_ms(),
        });
        self.inner.stream(messages, opts).await
    }
}

/// Render an excerpt of the prompt — joins messages with role tags,
/// then defers truncation + control-char sanitization to
/// [`sanitise_excerpt`]. Bias toward the LAST message (the live user
/// query) for the most diagnostic value when a span is short on
/// budget.
fn render_prompt_excerpt(messages: &[Message]) -> String {
    // Take last 2 messages — typical "user + tool result" or "system +
    // user" pair. Skip older history; that's what the full prompt log
    // (CallbackHandle Event::Llm) is for.
    let tail: Vec<&Message> = messages.iter().rev().take(2).collect();
    let mut out = String::new();
    for m in tail.iter().rev() {
        let tag = match m.role {
            litgraph_core::Role::System => "sys",
            litgraph_core::Role::User => "user",
            litgraph_core::Role::Assistant => "asst",
            litgraph_core::Role::Tool => "tool",
        };
        out.push_str(tag);
        out.push_str(": ");
        out.push_str(&m.text_content());
        out.push('\n');
    }
    out
}

fn record_excerpt(span: &Span, field_name: &'static str, value: &str) {
    if span.is_disabled() {
        return;
    }
    let cap = std::env::var("LITGRAPH_EXEMPLAR_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(MAX_EXCERPT_BYTES);
    let excerpt = sanitise_excerpt(value, cap);
    span.record(field_name, excerpt.as_str());
}

/// Truncate to `cap` bytes on a UTF-8 boundary; collapse control
/// chars to spaces; append `…` when truncated.
pub fn sanitise_excerpt(value: &str, cap: usize) -> String {
    let bytes = value.as_bytes();
    let end = if bytes.len() <= cap {
        bytes.len()
    } else {
        let mut e = cap;
        while e > 0 && !value.is_char_boundary(e) {
            e -= 1;
        }
        e
    };
    let mut out = String::with_capacity(end + 1);
    for c in value[..end].chars() {
        if c.is_control() {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    if end < bytes.len() {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitise_excerpt_short_passthrough() {
        assert_eq!(sanitise_excerpt("hello", 100), "hello");
    }

    #[test]
    fn sanitise_excerpt_truncates_with_ellipsis() {
        let s = "a".repeat(1000);
        let out = sanitise_excerpt(&s, 16);
        assert_eq!(out.len(), 16 + "…".len());
        assert!(out.ends_with('…'));
    }

    #[test]
    fn sanitise_excerpt_respects_utf8_boundary() {
        let s = "abc🦀def";
        // Cap at 4 bytes — would land mid-emoji. Helper must walk
        // back to the previous boundary, yielding "abc…".
        let out = sanitise_excerpt(s, 4);
        assert!(!out.contains(char::REPLACEMENT_CHARACTER));
        assert!(out.starts_with("abc"));
    }

    #[test]
    fn sanitise_excerpt_replaces_control_chars_with_space() {
        // `\n` and `\t` are control chars; both collapse to space.
        let out = sanitise_excerpt("hi\nthere\there", 100);
        assert_eq!(out, "hi there here");
    }

    #[test]
    fn render_prompt_excerpt_keeps_last_two_with_role_tags() {
        let msgs = vec![
            Message::system("you are a duck"),
            Message::user("ignored older"),
            Message::user("the live question"),
        ];
        let excerpt = render_prompt_excerpt(&msgs);
        // Tail is last 2: "ignored older" + "the live question"
        assert!(excerpt.contains("user: the live question"));
        assert!(excerpt.contains("user: ignored older"));
        // System prompt at index 0 is older than the 2-message tail —
        // not surfaced in the excerpt because we want the diagnostic
        // bytes spent on the live query.
        assert!(!excerpt.contains("you are a duck"));
    }
}
