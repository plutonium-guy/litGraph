//! `TextDiffTool` — line-level diff between two strings.
//!
//! Real prod scenarios:
//!
//! - **Code-change agents**: show "what did the refactor change?"
//!   in a structured form before applying or reporting.
//! - **Document review**: agent reading two versions of a doc
//!   summarizes the diffs.
//! - **Audit trails**: agent records what changed in a config /
//!   policy / spec between two snapshots.
//! - **Test failure analysis**: diff actual vs expected output.
//!
//! # Args
//!
//! - `before: String` — original text.
//! - `after: String` — new text.
//! - `format: "unified" | "structured"` — default `unified`.
//!   - `unified`: traditional diff output (`+`/`-` line prefixes,
//!     `@@ ... @@` hunk headers). Compact, human-readable, easy
//!     for the LLM to summarize.
//!   - `structured`: JSON object with `additions: [{line_num,
//!     text}, ...]`, `deletions: [...]`, `summary: {added, removed}`.
//!     Better when the agent needs to programmatically count or
//!     filter the changes.
//! - `context_lines: u32` — for `unified` mode, how many
//!   unchanged lines to show around each hunk. Default 3.

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{json, Value};
use similar::{ChangeTag, TextDiff};

#[derive(Debug, Clone, Default)]
pub struct TextDiffTool;

impl TextDiffTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for TextDiffTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "text_diff".into(),
            description: "Compute a line-level diff between `before` and `after`. \
                Format 'unified' (default) returns a traditional diff with +/- line prefixes \
                and @@ hunk headers — compact, easy to summarize. Format 'structured' \
                returns a JSON object with separate additions/deletions arrays plus a count \
                summary — better for programmatic filtering. `context_lines` controls how \
                many unchanged lines surround each hunk in unified mode (default 3)."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "before": {
                        "type": "string",
                        "description": "Original text."
                    },
                    "after": {
                        "type": "string",
                        "description": "New text."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["unified", "structured"],
                        "description": "Output format. Default 'unified'."
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Unchanged lines around each hunk (unified mode only). Default 3.",
                        "minimum": 0,
                        "maximum": 50
                    }
                },
                "required": ["before", "after"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let before = args
            .get("before")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("text_diff: missing `before`"))?;
        let after = args
            .get("after")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("text_diff: missing `after`"))?;
        let format = args
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("unified");
        let context_lines = args
            .get("context_lines")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;

        let diff = TextDiff::from_lines(before, after);

        match format {
            "unified" => {
                let unified = diff
                    .unified_diff()
                    .context_radius(context_lines)
                    .header("before", "after")
                    .to_string();
                Ok(json!({
                    "format": "unified",
                    "diff": unified,
                }))
            }
            "structured" => {
                let mut additions: Vec<Value> = Vec::new();
                let mut deletions: Vec<Value> = Vec::new();
                let mut new_line_num: usize = 0;
                let mut old_line_num: usize = 0;
                for change in diff.iter_all_changes() {
                    let text = change.value().trim_end_matches('\n').to_string();
                    match change.tag() {
                        ChangeTag::Insert => {
                            new_line_num += 1;
                            additions.push(json!({
                                "line_num": new_line_num,
                                "text": text,
                            }));
                        }
                        ChangeTag::Delete => {
                            old_line_num += 1;
                            deletions.push(json!({
                                "line_num": old_line_num,
                                "text": text,
                            }));
                        }
                        ChangeTag::Equal => {
                            new_line_num += 1;
                            old_line_num += 1;
                        }
                    }
                }
                Ok(json!({
                    "format": "structured",
                    "additions": additions,
                    "deletions": deletions,
                    "summary": {
                        "added": additions.len(),
                        "removed": deletions.len(),
                    },
                }))
            }
            other => Err(Error::invalid(format!(
                "text_diff: unknown format '{other}' (use 'unified' or 'structured')",
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unified_diff_marks_additions_and_deletions() {
        let t = TextDiffTool::new();
        let v = t
            .run(json!({
                "before": "line1\nline2\nline3\n",
                "after": "line1\nLINE2\nline3\nline4\n",
                "format": "unified"
            }))
            .await
            .unwrap();
        let diff = v.get("diff").and_then(|x| x.as_str()).unwrap();
        assert!(diff.contains("-line2"));
        assert!(diff.contains("+LINE2"));
        assert!(diff.contains("+line4"));
    }

    #[tokio::test]
    async fn structured_diff_returns_separate_arrays() {
        let t = TextDiffTool::new();
        let v = t
            .run(json!({
                "before": "a\nb\nc\n",
                "after": "a\nB\nc\nd\n",
                "format": "structured"
            }))
            .await
            .unwrap();
        let adds = v.get("additions").unwrap().as_array().unwrap();
        let dels = v.get("deletions").unwrap().as_array().unwrap();
        // 2 additions: "B" and "d". 1 deletion: "b".
        assert_eq!(adds.len(), 2);
        assert_eq!(dels.len(), 1);
        let summary = v.get("summary").unwrap();
        assert_eq!(
            summary.get("added").and_then(|x| x.as_u64()),
            Some(2),
        );
        assert_eq!(
            summary.get("removed").and_then(|x| x.as_u64()),
            Some(1),
        );
    }

    #[tokio::test]
    async fn structured_diff_includes_line_numbers() {
        let t = TextDiffTool::new();
        let v = t
            .run(json!({
                "before": "a\nb\nc\n",
                "after": "a\nb\nNEW\nc\n",
                "format": "structured"
            }))
            .await
            .unwrap();
        let adds = v.get("additions").unwrap().as_array().unwrap();
        assert_eq!(adds.len(), 1);
        // "NEW" is on line 3 of the new file.
        assert_eq!(
            adds[0].get("line_num").and_then(|x| x.as_u64()),
            Some(3),
        );
        assert_eq!(
            adds[0].get("text").and_then(|x| x.as_str()),
            Some("NEW"),
        );
    }

    #[tokio::test]
    async fn identical_inputs_return_no_changes() {
        let t = TextDiffTool::new();
        let v = t
            .run(json!({
                "before": "same\ntext\n",
                "after": "same\ntext\n",
                "format": "structured"
            }))
            .await
            .unwrap();
        assert!(v.get("additions").unwrap().as_array().unwrap().is_empty());
        assert!(v.get("deletions").unwrap().as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn unified_default_format() {
        let t = TextDiffTool::new();
        let v = t
            .run(json!({
                "before": "x\n",
                "after": "y\n"
            }))
            .await
            .unwrap();
        assert_eq!(v.get("format").and_then(|x| x.as_str()), Some("unified"));
        assert!(v.get("diff").is_some());
    }

    #[tokio::test]
    async fn context_lines_zero_produces_minimal_hunk() {
        let t = TextDiffTool::new();
        let v_zero = t
            .run(json!({
                "before": "a\nb\nc\nd\ne\n",
                "after": "a\nb\nC\nd\ne\n",
                "format": "unified",
                "context_lines": 0
            }))
            .await
            .unwrap();
        let v_three = t
            .run(json!({
                "before": "a\nb\nc\nd\ne\n",
                "after": "a\nb\nC\nd\ne\n",
                "format": "unified",
                "context_lines": 3
            }))
            .await
            .unwrap();
        let zero_diff = v_zero.get("diff").and_then(|x| x.as_str()).unwrap();
        let three_diff = v_three.get("diff").and_then(|x| x.as_str()).unwrap();
        // Zero-context output should be shorter (no surrounding equal lines).
        assert!(
            zero_diff.lines().count() < three_diff.lines().count(),
            "context_lines=0 should produce fewer lines than 3",
        );
    }

    #[tokio::test]
    async fn empty_before_or_after() {
        let t = TextDiffTool::new();
        let v = t
            .run(json!({
                "before": "",
                "after": "new content\n",
                "format": "structured"
            }))
            .await
            .unwrap();
        let adds = v.get("additions").unwrap().as_array().unwrap();
        let dels = v.get("deletions").unwrap().as_array().unwrap();
        assert_eq!(adds.len(), 1);
        assert_eq!(dels.len(), 0);
        assert_eq!(
            adds[0].get("text").and_then(|x| x.as_str()),
            Some("new content"),
        );
    }

    #[tokio::test]
    async fn missing_args_errors() {
        let t = TextDiffTool::new();
        assert!(matches!(
            t.run(json!({"after": "x"})).await,
            Err(Error::InvalidInput(_)),
        ));
        assert!(matches!(
            t.run(json!({"before": "x"})).await,
            Err(Error::InvalidInput(_)),
        ));
    }

    #[tokio::test]
    async fn unknown_format_errors() {
        let t = TextDiffTool::new();
        let r = t
            .run(json!({"before": "a", "after": "b", "format": "ndjson"}))
            .await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }
}
