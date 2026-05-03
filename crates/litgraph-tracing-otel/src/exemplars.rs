//! Trace exemplars — link an OTel span to its prompt + completion
//! excerpt without dumping the full body.
//!
//! When a litGraph span fires (provider call, tool call, graph node),
//! we want to attach a *small* piece of the prompt + completion as
//! span attributes so that:
//!
//! 1. Trace UIs (Jaeger / Tempo / Honeycomb) show the excerpt next
//!    to latency, model name, token counts.
//! 2. Bulk full-text search over a trace store gets useful hits
//!    without the cost of indexing every full body.
//! 3. PII-scrub policy is per-attribute (truncate / redact at the
//!    boundary), not per-payload.
//!
//! The standard OTel `events` model already supports attaching
//! arbitrary structured data, but most trace UIs only render *attributes*
//! prominently. Exemplars therefore go in attributes, capped at
//! [`MAX_EXCERPT_BYTES`] to keep span payload size bounded.
//!
//! # Usage
//!
//! ```no_run
//! use litgraph_tracing_otel::exemplars::{attach_prompt_excerpt, attach_completion_excerpt};
//! use tracing::info_span;
//!
//! let span = info_span!("provider.call", model = "gpt-5");
//! let _e = span.enter();
//! attach_prompt_excerpt("user: tell me about photosynthesis");
//! // … network call …
//! attach_completion_excerpt("Photosynthesis is the process by which …");
//! ```
//!
//! `attach_*` are no-ops if the current span is `Span::none()`, so
//! call sites are zero-overhead when tracing is disabled.

use tracing::Span;

/// Maximum bytes per excerpt attached as a span attribute. Span sizes
/// flow through the OTLP pipeline and large attribute payloads slow
/// every collector; 512 B is the common-sense cap that fits
/// 1-2 sentences of context (UTF-8). Tune via the env var
/// `LITGRAPH_EXEMPLAR_BYTES` if needed at startup.
pub const MAX_EXCERPT_BYTES: usize = 512;

/// Attach a *prompt excerpt* to the currently-entered span. The
/// excerpt is truncated to [`MAX_EXCERPT_BYTES`] (rounded down to the
/// nearest UTF-8 char boundary) and added as the
/// `litgraph.prompt_excerpt` attribute.
///
/// Newlines + control chars are replaced with spaces so the excerpt
/// is a single-line attribute (most trace UIs collapse newlines
/// anyway, but this makes diffing two spans cleaner).
pub fn attach_prompt_excerpt(prompt: &str) {
    attach_named_excerpt("litgraph.prompt_excerpt", prompt, runtime_cap());
}

/// Attach a *completion excerpt* to the current span. Same shape as
/// [`attach_prompt_excerpt`] but uses the
/// `litgraph.completion_excerpt` attribute.
pub fn attach_completion_excerpt(completion: &str) {
    attach_named_excerpt("litgraph.completion_excerpt", completion, runtime_cap());
}

/// Attach an arbitrary excerpt under a custom attribute name. Use
/// for tool-call inputs / outputs, retriever queries, etc.
pub fn attach_named_excerpt(attr: &'static str, value: &str, cap: usize) {
    let span = Span::current();
    if span.is_disabled() {
        return;
    }
    let excerpt = sanitise(value, cap);
    span.record(attr, excerpt.as_str());
}

fn runtime_cap() -> usize {
    std::env::var("LITGRAPH_EXEMPLAR_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(MAX_EXCERPT_BYTES)
}

/// Truncate `value` to `cap` bytes on a UTF-8 char boundary +
/// collapse control characters to spaces. Public for testing; in
/// production callers should use [`attach_prompt_excerpt`] /
/// [`attach_completion_excerpt`] directly.
pub fn sanitise(value: &str, cap: usize) -> String {
    let bytes = value.as_bytes();
    let end = if bytes.len() <= cap {
        bytes.len()
    } else {
        // Walk back to the previous char boundary so we don't slice
        // through a multibyte rune.
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
    fn sanitise_short_string_passes_through() {
        assert_eq!(sanitise("hello", 100), "hello");
    }

    #[test]
    fn sanitise_truncates_with_ellipsis() {
        let s = "a".repeat(1000);
        let out = sanitise(&s, 16);
        assert_eq!(out.len(), 16 + "…".len());
        assert!(out.ends_with('…'));
    }

    #[test]
    fn sanitise_respects_utf8_boundary() {
        // 4-byte emoji at the cut point — must NOT slice mid-rune.
        let s = "abc🦀def";
        let out = sanitise(s, 5);
        // 5 bytes lands inside the crab emoji; we walk back to byte 3.
        // First 3 bytes = "abc", then we add ellipsis.
        assert!(out.starts_with("abc"));
        assert!(out.ends_with('…'));
    }

    #[test]
    fn sanitise_collapses_newlines_to_space() {
        let s = "line one\nline two\rline three";
        let out = sanitise(s, 100);
        assert!(!out.contains('\n'));
        assert!(!out.contains('\r'));
        assert!(out.contains("line one line two line three"));
    }

    #[test]
    fn sanitise_handles_empty() {
        assert_eq!(sanitise("", 100), "");
    }

    #[test]
    fn sanitise_zero_cap_returns_just_ellipsis() {
        let out = sanitise("anything", 0);
        assert_eq!(out, "…");
    }

    #[test]
    fn attach_excerpts_no_op_outside_span() {
        // Span::current() returns a disabled span when no span is
        // active. attach_* should not panic.
        attach_prompt_excerpt("anything");
        attach_completion_excerpt("anything");
        attach_named_excerpt("litgraph.custom", "anything", 64);
    }
}
