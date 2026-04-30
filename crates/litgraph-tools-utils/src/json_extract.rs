//! `JsonExtractTool` — extract a value from a JSON document via
//! a small path expression. Sibling to iter-280 `RegexExtractTool`
//! for unstructured text; this one handles JSON.
//!
//! # Path syntax
//!
//! Tiny, deliberate subset of JSONPath — easy for LLMs to
//! generate without confusion:
//!
//! - `$` — root.
//! - `.field` — object key access.
//! - `[N]` — array index (zero-based, can be negative for from-end).
//! - `[*]` — all array elements; the result becomes a JSON array.
//!
//! Examples:
//!
//! - `$.users[0].name` → first user's name.
//! - `$.users[*].name` → array of all user names.
//! - `$.results[-1]` → last result.
//! - `$.data` → the data object.
//!
//! Not supported: filter expressions (`?(@.age > 30)`), recursive
//! descent (`..`), slices (`[1:5]`). Those add LLM-confusion
//! surface and aren't typically needed for agent workflows.
//!
//! # Args
//!
//! - `json: Value` — the JSON document to extract from. Can be
//!   passed as a JSON value OR as a string (auto-parsed if string).
//! - `path: String` — the path expression.
//!
//! # Returns
//!
//! The extracted value (any JSON type), or `null` if the path
//! resolves to nothing. `[*]` yields an array, even if 0 or 1
//! elements match.

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct JsonExtractTool;

impl JsonExtractTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for JsonExtractTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "json_extract".into(),
            description: "Extract a value from a JSON document via a path expression. \
                Path syntax: $ for root, .field for object keys, [N] for array index \
                (negative N counts from end), [*] for all array elements. \
                Examples: '$.users[0].name', '$.users[*].email', '$.results[-1]'. \
                Returns the extracted value or null if the path doesn't match. \
                Does NOT support filter expressions, recursive descent (..), or slices."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "json": {
                        "description": "JSON document. Can be a JSON value or a string (auto-parsed)."
                    },
                    "path": {
                        "type": "string",
                        "description": "Path expression. Example: '$.users[0].name'."
                    }
                },
                "required": ["json", "path"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("json_extract: missing `path`"))?;
        let json_arg = args
            .get("json")
            .ok_or_else(|| Error::invalid("json_extract: missing `json`"))?;

        // Auto-parse if `json` was passed as a string. Otherwise
        // use the value as-is (the LLM may have constructed a
        // structured arg).
        let doc: Value = if let Some(s) = json_arg.as_str() {
            serde_json::from_str(s)
                .map_err(|e| Error::invalid(format!("json_extract: bad JSON: {e}")))?
        } else {
            json_arg.clone()
        };

        let segments = parse_path(path)
            .map_err(|e| Error::invalid(format!("json_extract: bad path: {e}")))?;
        Ok(walk(&doc, &segments))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Segment {
    Field(String),
    Index(i64),
    Wildcard,
}

fn parse_path(path: &str) -> std::result::Result<Vec<Segment>, String> {
    let mut chars = path.chars().peekable();
    // Optional leading `$`.
    if matches!(chars.peek(), Some('$')) {
        chars.next();
    }
    let mut out: Vec<Segment> = Vec::new();
    while let Some(&c) = chars.peek() {
        match c {
            '.' => {
                chars.next();
                let mut field = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch == '.' || ch == '[' {
                        break;
                    }
                    field.push(ch);
                    chars.next();
                }
                if field.is_empty() {
                    return Err("expected field name after '.'".into());
                }
                out.push(Segment::Field(field));
            }
            '[' => {
                chars.next();
                let mut inner = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch == ']' {
                        break;
                    }
                    inner.push(ch);
                    chars.next();
                }
                if !matches!(chars.next(), Some(']')) {
                    return Err("unclosed '['".into());
                }
                let trimmed = inner.trim();
                if trimmed == "*" {
                    out.push(Segment::Wildcard);
                } else {
                    let n: i64 = trimmed
                        .parse()
                        .map_err(|_| format!("bad index: '{trimmed}'"))?;
                    out.push(Segment::Index(n));
                }
            }
            other => return Err(format!("unexpected char '{other}'")),
        }
    }
    Ok(out)
}

fn walk(value: &Value, segments: &[Segment]) -> Value {
    let Some((first, rest)) = segments.split_first() else {
        return value.clone();
    };
    match first {
        Segment::Field(name) => match value.get(name) {
            Some(v) => walk(v, rest),
            None => Value::Null,
        },
        Segment::Index(i) => match value.as_array() {
            Some(arr) => {
                let len = arr.len() as i64;
                let idx = if *i < 0 { len + i } else { *i };
                if idx < 0 || idx >= len {
                    Value::Null
                } else {
                    walk(&arr[idx as usize], rest)
                }
            }
            None => Value::Null,
        },
        Segment::Wildcard => match value.as_array() {
            Some(arr) => {
                let mut out: Vec<Value> = Vec::with_capacity(arr.len());
                for v in arr {
                    let r = walk(v, rest);
                    if !r.is_null() {
                        out.push(r);
                    }
                }
                Value::Array(out)
            }
            None => Value::Array(Vec::new()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        json!({
            "users": [
                {"name": "Alice", "age": 30, "email": "alice@example.com"},
                {"name": "Bob", "age": 25, "email": "bob@example.com"},
                {"name": "Carol", "age": 40}
            ],
            "results": ["a", "b", "c"],
            "meta": {"version": 2, "tags": ["x", "y"]}
        })
    }

    #[tokio::test]
    async fn simple_field_access() {
        let t = JsonExtractTool::new();
        let v = t
            .run(json!({"json": fixture(), "path": "$.meta.version"}))
            .await
            .unwrap();
        assert_eq!(v, json!(2));
    }

    #[tokio::test]
    async fn array_index_access() {
        let t = JsonExtractTool::new();
        let v = t
            .run(json!({"json": fixture(), "path": "$.users[0].name"}))
            .await
            .unwrap();
        assert_eq!(v, json!("Alice"));
    }

    #[tokio::test]
    async fn wildcard_array() {
        let t = JsonExtractTool::new();
        let v = t
            .run(json!({"json": fixture(), "path": "$.users[*].name"}))
            .await
            .unwrap();
        assert_eq!(v, json!(["Alice", "Bob", "Carol"]));
    }

    #[tokio::test]
    async fn negative_index_counts_from_end() {
        let t = JsonExtractTool::new();
        let v = t
            .run(json!({"json": fixture(), "path": "$.results[-1]"}))
            .await
            .unwrap();
        assert_eq!(v, json!("c"));
    }

    #[tokio::test]
    async fn missing_field_returns_null() {
        let t = JsonExtractTool::new();
        let v = t
            .run(json!({"json": fixture(), "path": "$.nonexistent"}))
            .await
            .unwrap();
        assert!(v.is_null());
    }

    #[tokio::test]
    async fn out_of_bounds_index_returns_null() {
        let t = JsonExtractTool::new();
        let v = t
            .run(json!({"json": fixture(), "path": "$.results[99]"}))
            .await
            .unwrap();
        assert!(v.is_null());
    }

    #[tokio::test]
    async fn wildcard_skips_missing_fields() {
        // $.users[*].email — Carol has no email; should return only Alice + Bob's.
        let t = JsonExtractTool::new();
        let v = t
            .run(json!({"json": fixture(), "path": "$.users[*].email"}))
            .await
            .unwrap();
        assert_eq!(v, json!(["alice@example.com", "bob@example.com"]));
    }

    #[tokio::test]
    async fn json_arg_can_be_string() {
        let t = JsonExtractTool::new();
        let v = t
            .run(json!({
                "json": "{\"name\": \"hello\"}",
                "path": "$.name"
            }))
            .await
            .unwrap();
        assert_eq!(v, json!("hello"));
    }

    #[tokio::test]
    async fn path_without_leading_dollar_works() {
        let t = JsonExtractTool::new();
        let v = t
            .run(json!({"json": fixture(), "path": ".meta.version"}))
            .await
            .unwrap();
        assert_eq!(v, json!(2));
    }

    #[tokio::test]
    async fn bad_path_syntax_errors() {
        let t = JsonExtractTool::new();
        let r = t
            .run(json!({"json": fixture(), "path": "$.users["}))
            .await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn bad_json_string_errors() {
        let t = JsonExtractTool::new();
        let r = t
            .run(json!({"json": "{not valid", "path": "$.x"}))
            .await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn empty_path_returns_full_doc() {
        let t = JsonExtractTool::new();
        let v = t
            .run(json!({"json": fixture(), "path": "$"}))
            .await
            .unwrap();
        assert_eq!(v, fixture());
    }

    #[tokio::test]
    async fn nested_arrays_chained() {
        let t = JsonExtractTool::new();
        let v = t
            .run(json!({"json": fixture(), "path": "$.meta.tags[1]"}))
            .await
            .unwrap();
        assert_eq!(v, json!("y"));
    }

    #[tokio::test]
    async fn missing_args_errors() {
        let t = JsonExtractTool::new();
        assert!(matches!(
            t.run(json!({"json": fixture()})).await,
            Err(Error::InvalidInput(_)),
        ));
        assert!(matches!(
            t.run(json!({"path": "$.x"})).await,
            Err(Error::InvalidInput(_)),
        ));
    }
}
