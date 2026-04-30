//! `UuidTool` — generate UUIDs. Real prod uses: agents minting
//! IDs for new records, idempotency keys for write tools, trace
//! IDs for distributed-tracing context, content addresses.
//!
//! # Args
//!
//! - `version: "v4" | "v7"` — default `v7`. Choice matters:
//!   - **v4**: 122 random bits. Good for opacity (no info leaks
//!     in the ID). Bad for DB primary keys — random ordering
//!     destroys B-tree page locality, making writes slow at
//!     scale.
//!   - **v7**: 48-bit Unix-millis timestamp prefix + 74 random
//!     bits. Sortable, locality-friendly for DB indexes, still
//!     globally unique. The right default for most agent
//!     workflows that mint IDs for storage.
//! - `count: u32` — how many to return. Default 1, max 100
//!   (cap is a sanity guard against the LLM looping).
//! - `format: "hyphenated" | "simple" | "urn"` — default
//!   `hyphenated` (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).
//!   `simple` is no-hyphen 32-char hex; `urn` is `urn:uuid:...`.
//!
//! # Returns
//!
//! `{version, format, uuids: [<id>, ...]}`. Always an array even
//! for `count=1` so callers don't have to special-case length.

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{json, Value};
use uuid::Uuid;

const MAX_COUNT: u32 = 100;

#[derive(Debug, Clone, Default)]
pub struct UuidTool;

impl UuidTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for UuidTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "uuid".into(),
            description: "Generate UUIDs. Version 'v7' (default) is timestamp-ordered \
                — recommended for DB primary keys and trace IDs since it preserves \
                index locality. Version 'v4' is fully random — use for opaque tokens \
                where ID ordering would leak information. Returns an array of UUIDs in \
                the requested format ('hyphenated' default, 'simple' for no-hyphen, \
                'urn' for 'urn:uuid:...')."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "version": {
                        "type": "string",
                        "enum": ["v4", "v7"],
                        "description": "UUID version. Default 'v7'."
                    },
                    "count": {
                        "type": "integer",
                        "description": "How many UUIDs to generate. Default 1, max 100.",
                        "minimum": 1,
                        "maximum": 100
                    },
                    "format": {
                        "type": "string",
                        "enum": ["hyphenated", "simple", "urn"],
                        "description": "Output format. Default 'hyphenated'."
                    }
                }
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let version = args
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("v7");
        let count = args
            .get("count")
            .and_then(|v| v.as_u64())
            .unwrap_or(1);
        if count == 0 {
            return Err(Error::invalid("uuid: count must be >= 1"));
        }
        if count > MAX_COUNT as u64 {
            return Err(Error::invalid(format!(
                "uuid: count {count} exceeds max {MAX_COUNT}",
            )));
        }
        let format = args
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("hyphenated");

        let mut uuids: Vec<Value> = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let id = match version {
                "v4" => Uuid::new_v4(),
                "v7" => Uuid::now_v7(),
                other => {
                    return Err(Error::invalid(format!(
                        "uuid: unknown version '{other}' (use 'v4' or 'v7')",
                    )));
                }
            };
            let s = match format {
                "hyphenated" => id.as_hyphenated().to_string(),
                "simple" => id.as_simple().to_string(),
                "urn" => id.as_urn().to_string(),
                other => {
                    return Err(Error::invalid(format!(
                        "uuid: unknown format '{other}' (use 'hyphenated', 'simple', or 'urn')",
                    )));
                }
            };
            uuids.push(Value::String(s));
        }
        Ok(json!({
            "version": version,
            "format": format,
            "uuids": uuids,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn default_v7_hyphenated() {
        let t = UuidTool::new();
        let v = t.run(json!({})).await.unwrap();
        assert_eq!(v.get("version").and_then(|x| x.as_str()), Some("v7"));
        assert_eq!(v.get("format").and_then(|x| x.as_str()), Some("hyphenated"));
        let arr = v.get("uuids").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 1);
        let s = arr[0].as_str().unwrap();
        // v7 hyphenated form: 8-4-4-4-12 hex with version nibble = 7.
        assert_eq!(s.len(), 36);
        assert_eq!(&s[14..15], "7", "expected v7 marker, got: {s}");
    }

    #[tokio::test]
    async fn v4_explicit() {
        let t = UuidTool::new();
        let v = t.run(json!({"version": "v4"})).await.unwrap();
        let s = v
            .get("uuids")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(&s[14..15], "4", "expected v4 marker, got: {s}");
    }

    #[tokio::test]
    async fn count_returns_n_distinct_ids() {
        let t = UuidTool::new();
        let v = t.run(json!({"count": 5})).await.unwrap();
        let arr = v.get("uuids").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 5);
        // Verify all distinct.
        let mut set = std::collections::HashSet::new();
        for id in arr {
            set.insert(id.as_str().unwrap().to_string());
        }
        assert_eq!(set.len(), 5);
    }

    #[tokio::test]
    async fn simple_format_no_hyphens() {
        let t = UuidTool::new();
        let v = t.run(json!({"format": "simple"})).await.unwrap();
        let s = v
            .get("uuids")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(s.len(), 32);
        assert!(!s.contains('-'));
    }

    #[tokio::test]
    async fn urn_format_has_prefix() {
        let t = UuidTool::new();
        let v = t.run(json!({"format": "urn"})).await.unwrap();
        let s = v
            .get("uuids")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .as_str()
            .unwrap()
            .to_string();
        assert!(s.starts_with("urn:uuid:"));
    }

    #[tokio::test]
    async fn v7_ids_are_sortable_by_creation_time() {
        // Generate 10 IDs serially with sleeps; sorting by string
        // value should put them in creation order.
        let t = UuidTool::new();
        let mut ids: Vec<String> = Vec::new();
        for _ in 0..10 {
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            let v = t.run(json!({"count": 1})).await.unwrap();
            let s = v.get("uuids").unwrap().as_array().unwrap()[0]
                .as_str()
                .unwrap()
                .to_string();
            ids.push(s);
        }
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted, "v7 IDs should sort to creation order");
    }

    #[tokio::test]
    async fn count_zero_errors() {
        let t = UuidTool::new();
        let r = t.run(json!({"count": 0})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn count_exceeds_max_errors() {
        let t = UuidTool::new();
        let r = t.run(json!({"count": 1000})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn unknown_version_errors() {
        let t = UuidTool::new();
        let r = t.run(json!({"version": "v6"})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn unknown_format_errors() {
        let t = UuidTool::new();
        let r = t.run(json!({"format": "lol"})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }
}
