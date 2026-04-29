//! `VirtualFilesystemTool` — in-memory scratch filesystem for an agent
//! session. Mirrors the `deepagents` virtual-FS backend without touching the
//! real filesystem, so the agent has a sandboxed scratch space for notes,
//! intermediate output, and large tool results that would otherwise pollute
//! the chat context.
//!
//! Single tool with `action` discriminator: `read`, `write`, `append`, `list`,
//! `delete`, `exists`. Paths are normalized as forward-slash strings; an
//! optional `prefix` on `list` filters by leading-substring.
//!
//! State lives behind `Arc<Mutex<…>>` so parallel tool calls stay consistent
//! and one tool instance can be cloned across threads. Each agent session
//! should construct its own instance to scope state.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{json, Value};

use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};

#[derive(Debug, Default)]
struct VfsState {
    files: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub struct VirtualFilesystemTool {
    state: Arc<Mutex<VfsState>>,
    /// Hard cap on total bytes stored (sum of all file contents). 0 = unlimited.
    max_total_bytes: usize,
}

impl VirtualFilesystemTool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_total_bytes(max_total_bytes: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(VfsState::default())),
            max_total_bytes,
        }
    }

    pub fn snapshot(&self) -> BTreeMap<String, String> {
        self.state.lock().unwrap().files.clone()
    }

    pub fn total_bytes(&self) -> usize {
        self.state.lock().unwrap().files.values().map(|s| s.len()).sum()
    }
}

fn normalize_path(p: &str) -> Result<String> {
    let trimmed = p.trim();
    if trimmed.is_empty() {
        return Err(Error::invalid("vfs: empty path"));
    }
    let mut out = trimmed.replace('\\', "/");
    while out.starts_with("//") {
        out.remove(0);
    }
    if out.contains("..") {
        return Err(Error::invalid("vfs: `..` not allowed"));
    }
    Ok(out)
}

#[async_trait]
impl Tool for VirtualFilesystemTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "vfs".into(),
            description: "Sandboxed in-memory virtual filesystem. action='read' returns the file \
                          contents (`path`), 'write' overwrites a file (`path`+`content`), \
                          'append' concatenates to an existing or new file, 'list' returns paths \
                          under an optional `prefix`, 'delete' removes a file, 'exists' checks \
                          presence. Paths use forward slashes; `..` is rejected."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "append", "list", "delete", "exists"]
                    },
                    "path": {
                        "type": "string",
                        "description": "Forward-slash path (read/write/append/delete/exists)."
                    },
                    "content": {
                        "type": "string",
                        "description": "File content (write/append)."
                    },
                    "prefix": {
                        "type": "string",
                        "description": "Path prefix filter for action=list."
                    }
                },
                "required": ["action"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("vfs: missing `action`"))?;
        let mut state = self.state.lock().unwrap();
        match action {
            "list" => {
                let prefix = args.get("prefix").and_then(|v| v.as_str()).unwrap_or("");
                let mut paths: Vec<String> = state
                    .files
                    .keys()
                    .filter(|p| p.starts_with(prefix))
                    .cloned()
                    .collect();
                paths.sort();
                Ok(json!({"paths": paths, "count": paths.len()}))
            }
            "read" => {
                let path = normalize_path(
                    args.get("path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::invalid("vfs(read): missing `path`"))?,
                )?;
                let content = state
                    .files
                    .get(&path)
                    .ok_or_else(|| Error::invalid(format!("vfs: file not found: {path}")))?;
                Ok(json!({"path": path, "content": content, "size": content.len()}))
            }
            "exists" => {
                let path = normalize_path(
                    args.get("path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::invalid("vfs(exists): missing `path`"))?,
                )?;
                Ok(json!({"exists": state.files.contains_key(&path), "path": path}))
            }
            "write" | "append" => {
                let path = normalize_path(
                    args.get("path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::invalid("vfs(write): missing `path`"))?,
                )?;
                let content = args
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::invalid("vfs(write): missing `content`"))?;
                let new_value = if action == "write" {
                    content.to_string()
                } else {
                    let mut existing = state.files.get(&path).cloned().unwrap_or_default();
                    existing.push_str(content);
                    existing
                };
                if self.max_total_bytes > 0 {
                    let other_bytes: usize = state
                        .files
                        .iter()
                        .filter(|(k, _)| **k != path)
                        .map(|(_, v)| v.len())
                        .sum();
                    if other_bytes + new_value.len() > self.max_total_bytes {
                        return Err(Error::invalid(format!(
                            "vfs: total size cap exceeded ({} bytes max)",
                            self.max_total_bytes
                        )));
                    }
                }
                let bytes = new_value.len();
                state.files.insert(path.clone(), new_value);
                Ok(json!({"path": path, "bytes": bytes, "action": action}))
            }
            "delete" => {
                let path = normalize_path(
                    args.get("path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::invalid("vfs(delete): missing `path`"))?,
                )?;
                let removed = state.files.remove(&path).is_some();
                Ok(json!({"path": path, "deleted": removed}))
            }
            other => Err(Error::invalid(format!("vfs: unknown action `{other}`"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn list_starts_empty() {
        let t = VirtualFilesystemTool::new();
        let r = t.run(json!({"action": "list"})).await.unwrap();
        assert_eq!(r["count"], 0);
    }

    #[tokio::test]
    async fn write_then_read_roundtrips() {
        let t = VirtualFilesystemTool::new();
        t.run(json!({"action": "write", "path": "a/b.txt", "content": "hello"}))
            .await
            .unwrap();
        let r = t
            .run(json!({"action": "read", "path": "a/b.txt"}))
            .await
            .unwrap();
        assert_eq!(r["content"], "hello");
        assert_eq!(r["size"], 5);
    }

    #[tokio::test]
    async fn append_concatenates_and_creates() {
        let t = VirtualFilesystemTool::new();
        t.run(json!({"action": "append", "path": "log.txt", "content": "a"}))
            .await
            .unwrap();
        t.run(json!({"action": "append", "path": "log.txt", "content": "b"}))
            .await
            .unwrap();
        let r = t
            .run(json!({"action": "read", "path": "log.txt"}))
            .await
            .unwrap();
        assert_eq!(r["content"], "ab");
    }

    #[tokio::test]
    async fn write_overwrites() {
        let t = VirtualFilesystemTool::new();
        t.run(json!({"action": "write", "path": "k", "content": "1"})).await.unwrap();
        t.run(json!({"action": "write", "path": "k", "content": "2"})).await.unwrap();
        let r = t.run(json!({"action": "read", "path": "k"})).await.unwrap();
        assert_eq!(r["content"], "2");
    }

    #[tokio::test]
    async fn list_filters_by_prefix() {
        let t = VirtualFilesystemTool::new();
        for p in ["docs/a", "docs/b", "src/c"] {
            t.run(json!({"action": "write", "path": p, "content": "x"}))
                .await
                .unwrap();
        }
        let r = t
            .run(json!({"action": "list", "prefix": "docs/"}))
            .await
            .unwrap();
        assert_eq!(r["count"], 2);
    }

    #[tokio::test]
    async fn delete_returns_truthy_only_if_existed() {
        let t = VirtualFilesystemTool::new();
        let r = t
            .run(json!({"action": "delete", "path": "missing"}))
            .await
            .unwrap();
        assert_eq!(r["deleted"], false);
        t.run(json!({"action": "write", "path": "x", "content": "v"}))
            .await
            .unwrap();
        let r = t.run(json!({"action": "delete", "path": "x"})).await.unwrap();
        assert_eq!(r["deleted"], true);
    }

    #[tokio::test]
    async fn read_missing_errors() {
        let t = VirtualFilesystemTool::new();
        let err = t
            .run(json!({"action": "read", "path": "missing"}))
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("not found"));
    }

    #[tokio::test]
    async fn exists_does_not_create() {
        let t = VirtualFilesystemTool::new();
        let r = t
            .run(json!({"action": "exists", "path": "nope"}))
            .await
            .unwrap();
        assert_eq!(r["exists"], false);
        assert!(t.snapshot().is_empty());
    }

    #[tokio::test]
    async fn dotdot_rejected() {
        let t = VirtualFilesystemTool::new();
        let err = t
            .run(json!({"action": "write", "path": "../etc/passwd", "content": "x"}))
            .await
            .unwrap_err();
        assert!(format!("{err}").contains(".."));
    }

    #[tokio::test]
    async fn empty_path_rejected() {
        let t = VirtualFilesystemTool::new();
        let err = t
            .run(json!({"action": "write", "path": "  ", "content": "x"}))
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("empty"));
    }

    #[tokio::test]
    async fn size_cap_blocks_oversize_write() {
        let t = VirtualFilesystemTool::with_max_total_bytes(10);
        t.run(json!({"action": "write", "path": "a", "content": "12345"}))
            .await
            .unwrap();
        let err = t
            .run(json!({"action": "write", "path": "b", "content": "1234567"}))
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("cap"));
    }
}
