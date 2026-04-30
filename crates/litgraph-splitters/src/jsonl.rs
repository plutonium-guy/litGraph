//! `JsonLinesSplitter` — emit one chunk per JSONL record.
//!
//! # Distinct from `JsonSplitter`
//!
//! - `JsonSplitter` parses one big JSON document and navigates
//!   its structure to produce chunks (suitable for nested
//!   schemas).
//! - `JsonLinesSplitter` (this) treats input as line-delimited
//!   JSON: each line is its own record. Emit one chunk per
//!   record.
//!
//! # Real prod use
//!
//! - **Log analysis**: structured-log streams (Datadog, Loki,
//!   ELK) export to NDJSON. Each line is one event.
//! - **OpenAI fine-tune format**: `{"messages": [...]}` per line.
//! - **Dataset processing**: HuggingFace / streaming datasets
//!   are typically NDJSON.
//! - **Replay traces**: agent conversation logs persisted as
//!   NDJSON for offline analysis.
//!
//! # Args
//!
//! - `pretty: bool` — re-serialize each record as pretty-printed
//!   JSON in the output chunk. Default `false` (preserve the
//!   original line, modulo whitespace trim).
//! - `skip_invalid: bool` — silently drop lines that don't parse
//!   as valid JSON. Default `false` — invalid lines surface as
//!   an Err in `split_text`'s caller, but since `Splitter`
//!   returns `Vec<String>` (no Result), we don't have a clean
//!   place to signal an error. So the actual behavior is: if
//!   `skip_invalid` is `false`, invalid lines are kept verbatim
//!   in the output (caller can see them); if `true`, invalid
//!   lines are dropped.

use crate::Splitter;

#[derive(Debug, Clone)]
pub struct JsonLinesSplitter {
    pub pretty: bool,
    pub skip_invalid: bool,
}

impl Default for JsonLinesSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonLinesSplitter {
    pub fn new() -> Self {
        Self {
            pretty: false,
            skip_invalid: false,
        }
    }

    pub fn with_pretty(mut self, b: bool) -> Self {
        self.pretty = b;
        self
    }

    pub fn with_skip_invalid(mut self, b: bool) -> Self {
        self.skip_invalid = b;
        self
    }
}

impl Splitter for JsonLinesSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for raw in text.lines() {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Try to parse. If valid AND pretty=true, re-serialize.
            // If valid AND pretty=false, just keep the original line.
            // If invalid AND skip_invalid=true, drop. Else, keep.
            match serde_json::from_str::<serde_json::Value>(trimmed) {
                Ok(v) => {
                    if self.pretty {
                        match serde_json::to_string_pretty(&v) {
                            Ok(s) => out.push(s),
                            Err(_) => out.push(trimmed.to_string()),
                        }
                    } else {
                        out.push(trimmed.to_string());
                    }
                }
                Err(_) => {
                    if !self.skip_invalid {
                        out.push(trimmed.to_string());
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_chunk_per_record() {
        let s = JsonLinesSplitter::new();
        let input = r#"{"a": 1}
{"a": 2}
{"a": 3}"#;
        let out = s.split_text(input);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], r#"{"a": 1}"#);
        assert_eq!(out[1], r#"{"a": 2}"#);
        assert_eq!(out[2], r#"{"a": 3}"#);
    }

    #[test]
    fn skips_blank_lines() {
        let s = JsonLinesSplitter::new();
        let input = "{\"a\": 1}\n\n\n{\"a\": 2}\n";
        let out = s.split_text(input);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn pretty_reformats_each_record() {
        let s = JsonLinesSplitter::new().with_pretty(true);
        let input = r#"{"a":1,"b":2}
{"x":[1,2,3]}"#;
        let out = s.split_text(input);
        assert_eq!(out.len(), 2);
        assert!(out[0].contains('\n'), "pretty should multi-line");
        assert!(out[1].contains('\n'));
    }

    #[test]
    fn skip_invalid_drops_bad_lines() {
        let s = JsonLinesSplitter::new().with_skip_invalid(true);
        let input = r#"{"a": 1}
not json at all
{"a": 2}"#;
        let out = s.split_text(input);
        assert_eq!(out.len(), 2);
        assert!(out[0].contains(r#""a": 1"#));
        assert!(out[1].contains(r#""a": 2"#));
    }

    #[test]
    fn keep_invalid_by_default() {
        let s = JsonLinesSplitter::new();
        let input = r#"{"a": 1}
not json
{"a": 2}"#;
        let out = s.split_text(input);
        // Invalid line is kept verbatim, surfacing the issue.
        assert_eq!(out.len(), 3);
        assert_eq!(out[1], "not json");
    }

    #[test]
    fn empty_input_returns_empty() {
        let s = JsonLinesSplitter::new();
        assert!(s.split_text("").is_empty());
        assert!(s.split_text("\n\n\n").is_empty());
    }

    #[test]
    fn handles_crlf_line_endings() {
        let s = JsonLinesSplitter::new();
        let input = "{\"a\":1}\r\n{\"a\":2}\r\n";
        let out = s.split_text(input);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn openai_finetune_format_round_trip() {
        // Realistic OpenAI fine-tune line format.
        let s = JsonLinesSplitter::new();
        let input = r#"{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}
{"messages":[{"role":"user","content":"bye"},{"role":"assistant","content":"goodbye"}]}"#;
        let out = s.split_text(input);
        assert_eq!(out.len(), 2);
        for chunk in &out {
            // Each chunk parses as valid JSON with the expected shape.
            let v: serde_json::Value = serde_json::from_str(chunk).unwrap();
            assert!(v.get("messages").is_some());
        }
    }
}
