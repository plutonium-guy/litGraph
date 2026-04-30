//! `RegexExtractTool` — apply a regex pattern to a string and
//! return matches or captures.
//!
//! Common agent workflow: fetch a webpage / API response via
//! `WebFetchTool` or `HttpRequestTool`, then extract specific
//! fields (prices, dates, IDs, links) without writing a custom
//! parsing tool. Universal — applies to any text input.
//!
//! # Args
//!
//! - `pattern: String` — Rust `regex` crate syntax (PCRE-like
//!   without lookaround / backrefs; same as the rest of litGraph
//!   uses). Compiled per-call (cheap; the regex crate's compile
//!   cost is in the µs range for typical patterns).
//! - `text: String` — the haystack.
//! - `mode: "all" | "first" | "captures"` — how to return matches.
//!   - `all` (default): array of all match strings.
//!   - `first`: just the first match (string), or null if none.
//!   - `captures`: array of arrays — each inner array is the
//!     full match + numbered capture groups for one match.
//!
//! # Returns
//!
//! Mode-dependent JSON. The schema description tells the LLM what
//! to expect.

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use regex::Regex;
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct RegexExtractTool;

impl RegexExtractTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for RegexExtractTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "regex_extract".into(),
            description: "Apply a regex pattern to a string and return matches. \
                Useful for extracting structured data (prices, dates, IDs, links) \
                from unstructured text. Mode `all` (default) returns every match; \
                `first` returns the first match or null; `captures` returns each \
                match along with its numbered capture groups."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern (Rust regex crate syntax — PCRE-like; no lookaround/backrefs)."
                    },
                    "text": {
                        "type": "string",
                        "description": "Haystack to search."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["all", "first", "captures"],
                        "description": "How to return results. Default 'all'."
                    }
                },
                "required": ["pattern", "text"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("regex_extract: missing `pattern`"))?;
        let text = args
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("regex_extract: missing `text`"))?;
        let mode = args
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("all");

        let re = Regex::new(pattern)
            .map_err(|e| Error::invalid(format!("regex_extract: bad pattern: {e}")))?;

        match mode {
            "all" => {
                let matches: Vec<Value> = re
                    .find_iter(text)
                    .map(|m| Value::String(m.as_str().to_string()))
                    .collect();
                Ok(Value::Array(matches))
            }
            "first" => match re.find(text) {
                Some(m) => Ok(Value::String(m.as_str().to_string())),
                None => Ok(Value::Null),
            },
            "captures" => {
                let mut out: Vec<Value> = Vec::new();
                for caps in re.captures_iter(text) {
                    let mut group: Vec<Value> = Vec::with_capacity(caps.len());
                    for i in 0..caps.len() {
                        match caps.get(i) {
                            Some(m) => group.push(Value::String(m.as_str().to_string())),
                            None => group.push(Value::Null),
                        }
                    }
                    out.push(Value::Array(group));
                }
                Ok(Value::Array(out))
            }
            other => Err(Error::invalid(format!(
                "regex_extract: unknown mode `{other}` (use 'all', 'first', or 'captures')",
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mode_all_returns_every_match() {
        let t = RegexExtractTool::new();
        let v = t
            .run(json!({
                "pattern": r"\$\d+(?:\.\d+)?",
                "text": "The price is $9.99 today and $19.50 tomorrow."
            }))
            .await
            .unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0].as_str(), Some("$9.99"));
        assert_eq!(arr[1].as_str(), Some("$19.50"));
    }

    #[tokio::test]
    async fn mode_first_returns_string_or_null() {
        let t = RegexExtractTool::new();
        let v = t
            .run(json!({
                "pattern": r"\d{4}",
                "text": "year 2024 and 2025",
                "mode": "first"
            }))
            .await
            .unwrap();
        assert_eq!(v.as_str(), Some("2024"));

        let none = t
            .run(json!({
                "pattern": r"\d{4}",
                "text": "no digits at all",
                "mode": "first"
            }))
            .await
            .unwrap();
        assert!(none.is_null());
    }

    #[tokio::test]
    async fn mode_captures_returns_groups() {
        let t = RegexExtractTool::new();
        let v = t
            .run(json!({
                "pattern": r"(\d{4})-(\d{2})-(\d{2})",
                "text": "Date: 2024-09-15 was nice. Then 2025-01-01.",
                "mode": "captures"
            }))
            .await
            .unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        let first = arr[0].as_array().unwrap();
        // [full_match, year, month, day]
        assert_eq!(first.len(), 4);
        assert_eq!(first[0].as_str(), Some("2024-09-15"));
        assert_eq!(first[1].as_str(), Some("2024"));
        assert_eq!(first[2].as_str(), Some("09"));
        assert_eq!(first[3].as_str(), Some("15"));
    }

    #[tokio::test]
    async fn mode_default_is_all() {
        let t = RegexExtractTool::new();
        let v = t
            .run(json!({"pattern": r"\d+", "text": "1 2 3"}))
            .await
            .unwrap();
        assert_eq!(v.as_array().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn empty_match_returns_empty_array_in_all_mode() {
        let t = RegexExtractTool::new();
        let v = t
            .run(json!({"pattern": r"xyz\d+", "text": "no match here"}))
            .await
            .unwrap();
        assert!(v.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn bad_pattern_returns_invalid_input_error() {
        let t = RegexExtractTool::new();
        let r = t
            .run(json!({"pattern": "(unclosed", "text": "anything"}))
            .await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn missing_pattern_or_text_errors() {
        let t = RegexExtractTool::new();
        assert!(matches!(
            t.run(json!({"text": "x"})).await,
            Err(Error::InvalidInput(_)),
        ));
        assert!(matches!(
            t.run(json!({"pattern": "x"})).await,
            Err(Error::InvalidInput(_)),
        ));
    }

    #[tokio::test]
    async fn unknown_mode_errors() {
        let t = RegexExtractTool::new();
        let r = t
            .run(json!({
                "pattern": ".",
                "text": "x",
                "mode": "weird"
            }))
            .await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn captures_with_optional_unmatched_group_returns_null() {
        let t = RegexExtractTool::new();
        let v = t
            .run(json!({
                "pattern": r"(\d+)(?:-(\d+))?",
                "text": "12 and 34-56",
                "mode": "captures"
            }))
            .await
            .unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        // First match: "12" with no second group.
        let first = arr[0].as_array().unwrap();
        assert_eq!(first[0].as_str(), Some("12"));
        assert_eq!(first[1].as_str(), Some("12"));
        assert!(first[2].is_null(), "unmatched optional group should be null");
        // Second match: "34-56" with both groups present.
        let second = arr[1].as_array().unwrap();
        assert_eq!(second[1].as_str(), Some("34"));
        assert_eq!(second[2].as_str(), Some("56"));
    }
}
