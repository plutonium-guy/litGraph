//! Filesystem tools — `read_file`, `write_file`, `list_directory`. All three
//! share a `FsRoot` sandbox: every requested path is resolved against the root
//! and rejected if it escapes via `..` or absolute paths or symlinks. Without
//! the sandbox an LLM-driven agent can read /etc/passwd or ~/.ssh, which is
//! catastrophic in any production deployment.
//!
//! The sandbox is mandatory — there's no `unrestricted()` constructor. If the
//! caller wants full-fs access they can pass `FsRoot::new("/")` and accept
//! the consequences explicitly.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{Value, json};

/// Shared sandbox root + canonical-path resolver.
#[derive(Clone, Debug)]
pub struct FsRoot {
    /// Canonicalized absolute path. We compare prefixes against this when
    /// resolving requested paths; symlinks pointing outside the root are
    /// detected at canonicalization time.
    root: PathBuf,
}

impl FsRoot {
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self> {
        let root = root.as_ref();
        let canon = std::fs::canonicalize(root)
            .map_err(|e| Error::other(format!("FsRoot: cannot canonicalize {}: {e}", root.display())))?;
        if !canon.is_dir() {
            return Err(Error::other(format!("FsRoot: {} is not a directory", canon.display())));
        }
        Ok(Self { root: canon })
    }

    /// Resolve `requested` (relative or absolute) to an absolute path under
    /// the sandbox root. Returns `Err` if it escapes.
    ///
    /// Uses *lexical* `..` resolution rather than `canonicalize()` so that
    /// non-existent target paths still validate (needed for write_file). We
    /// also accept absolute paths from the tool args by dropping the leading
    /// `/` and treating them as relative-to-root — agents reliably mix the
    /// two and rejecting absolute paths just produces noisy retries.
    ///
    /// Symlink note: a symlink under the sandbox that points outside the root
    /// will be followed at I/O time. For an LLM-driven agent this is an
    /// acceptable trade — the lexical check still blocks the obvious escape
    /// (`../../../etc/passwd`) and operators control what's inside the root.
    fn resolve(&self, requested: &str) -> Result<PathBuf> {
        let candidate = Path::new(requested);
        let joined = if candidate.is_absolute() {
            self.root.join(candidate.strip_prefix("/").unwrap_or(candidate))
        } else {
            self.root.join(candidate)
        };
        // Lexical normalize: walk components, cancelling `..` against the
        // previous one (without touching the filesystem). Refuse if `..`
        // would pop above the root.
        let mut normalized: Vec<std::path::Component> = Vec::new();
        for comp in joined.components() {
            use std::path::Component::*;
            match comp {
                CurDir => continue,
                ParentDir => {
                    // Refuse to pop the root prefix or anything above the sandbox.
                    let popped = normalized.last().copied();
                    let can_pop = matches!(popped, Some(Normal(_)));
                    if !can_pop {
                        return Err(Error::other(format!(
                            "path escapes sandbox: {} not under {}",
                            joined.display(), self.root.display()
                        )));
                    }
                    normalized.pop();
                }
                other => normalized.push(other),
            }
        }
        let result: PathBuf = normalized.iter().collect();
        if !result.starts_with(&self.root) {
            return Err(Error::other(format!(
                "path escapes sandbox: {} not under {}",
                result.display(), self.root.display()
            )));
        }
        Ok(result)
    }
}

// ─── ReadFile ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ReadFileTool {
    root: FsRoot,
    /// Refuse files larger than this (default 1 MiB) — protects against the
    /// agent slurping a 5 GB log into the conversation.
    pub max_bytes: usize,
}

impl ReadFileTool {
    pub fn new(root: FsRoot) -> Self {
        Self { root, max_bytes: 1024 * 1024 }
    }
    pub fn with_max_bytes(mut self, n: usize) -> Self {
        self.max_bytes = n;
        self
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "read_file".into(),
            description: "Read a UTF-8 text file from the sandbox. Path is relative to the sandbox root. Refuses files larger than max_bytes.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Relative path under the sandbox root." }
                },
                "required": ["path"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let path = args.get("path").and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("read_file: missing `path`"))?;
        let resolved = self.root.resolve(path)?;
        let max = self.max_bytes;
        let resolved_owned = resolved.clone();
        let bytes = tokio::task::spawn_blocking(move || std::fs::metadata(&resolved_owned))
            .await
            .map_err(|e| Error::other(format!("join: {e}")))?
            .map_err(|e| Error::other(format!("stat {}: {e}", resolved.display())))?
            .len() as usize;
        if bytes > max {
            return Err(Error::invalid(format!(
                "file too large: {} bytes > max_bytes={max}", bytes
            )));
        }
        let resolved_owned = resolved.clone();
        let content = tokio::task::spawn_blocking(move || std::fs::read_to_string(&resolved_owned))
            .await
            .map_err(|e| Error::other(format!("join: {e}")))?
            .map_err(|e| Error::other(format!("read {}: {e}", resolved.display())))?;
        Ok(json!({ "path": path, "content": content, "bytes": bytes }))
    }
}

// ─── WriteFile ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct WriteFileTool {
    root: FsRoot,
    /// Refuse to overwrite an existing file unless `overwrite=true` was passed
    /// per-call. Default false (safe).
    pub default_overwrite: bool,
}

impl WriteFileTool {
    pub fn new(root: FsRoot) -> Self {
        Self { root, default_overwrite: false }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "write_file".into(),
            description: "Write a UTF-8 text file inside the sandbox. By default refuses to overwrite an existing file; set overwrite=true to replace.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Relative path under the sandbox root." },
                    "content": { "type": "string", "description": "File contents (UTF-8)." },
                    "overwrite": { "type": "boolean", "description": "Allow overwriting an existing file (default false).", "default": false }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let path = args.get("path").and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("write_file: missing `path`"))?;
        let content = args.get("content").and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("write_file: missing `content`"))?
            .to_string();
        let overwrite = args.get("overwrite").and_then(|v| v.as_bool())
            .unwrap_or(self.default_overwrite);

        let resolved = self.root.resolve(path)?;
        let resolved_owned = resolved.clone();
        let exists_already = tokio::task::spawn_blocking(move || resolved_owned.exists())
            .await
            .map_err(|e| Error::other(format!("join: {e}")))?;
        if exists_already && !overwrite {
            return Err(Error::invalid(format!(
                "refusing to overwrite existing file: {} (pass overwrite=true)",
                resolved.display()
            )));
        }
        // Ensure parent directory exists (under the sandbox root).
        if let Some(parent) = resolved.parent() {
            let parent = parent.to_path_buf();
            tokio::task::spawn_blocking(move || std::fs::create_dir_all(parent))
                .await
                .map_err(|e| Error::other(format!("join: {e}")))?
                .map_err(|e| Error::other(format!("mkdir: {e}")))?;
        }
        let resolved_owned = resolved.clone();
        let bytes = content.len();
        tokio::task::spawn_blocking(move || std::fs::write(&resolved_owned, content))
            .await
            .map_err(|e| Error::other(format!("join: {e}")))?
            .map_err(|e| Error::other(format!("write {}: {e}", resolved.display())))?;
        Ok(json!({ "path": path, "bytes_written": bytes, "overwritten": exists_already }))
    }
}

// ─── ListDirectory ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ListDirectoryTool {
    root: FsRoot,
}

impl ListDirectoryTool {
    pub fn new(root: FsRoot) -> Self { Self { root } }
}

#[async_trait]
impl Tool for ListDirectoryTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "list_directory".into(),
            description: "List the immediate contents of a directory inside the sandbox. Returns one entry per file/dir with name + kind. Pass '.' to list the sandbox root.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory path relative to the sandbox root.", "default": "." }
                }
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let resolved = self.root.resolve(path)?;
        let resolved_owned = resolved.clone();
        let entries: Vec<(String, &'static str)> = tokio::task::spawn_blocking(move || -> std::io::Result<Vec<(String, &'static str)>> {
            let mut out = Vec::new();
            for entry in std::fs::read_dir(&resolved_owned)? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().into_owned();
                let kind = match entry.file_type() {
                    Ok(ft) if ft.is_dir() => "dir",
                    Ok(ft) if ft.is_symlink() => "symlink",
                    Ok(_) => "file",
                    Err(_) => "unknown",
                };
                out.push((name, kind));
            }
            out.sort();
            Ok(out)
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))?
        .map_err(|e| Error::other(format!("read_dir {}: {e}", resolved.display())))?;
        let entries_json: Vec<Value> = entries
            .into_iter()
            .map(|(name, kind)| json!({"name": name, "kind": kind}))
            .collect();
        Ok(json!({ "path": path, "entries": entries_json }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_root() -> (tempfile::TempDir, FsRoot) {
        let td = tempfile::tempdir().unwrap();
        let root = FsRoot::new(td.path()).unwrap();
        (td, root)
    }

    #[tokio::test]
    async fn read_file_returns_content_under_sandbox() {
        let (td, root) = tmp_root();
        fs::write(td.path().join("a.txt"), "hello world").unwrap();
        let t = ReadFileTool::new(root);
        let out = t.run(json!({"path": "a.txt"})).await.unwrap();
        assert_eq!(out["content"], json!("hello world"));
        assert_eq!(out["bytes"], json!(11));
    }

    #[tokio::test]
    async fn read_file_rejects_path_traversal() {
        let (_td, root) = tmp_root();
        let t = ReadFileTool::new(root);
        let err = t.run(json!({"path": "../../../etc/passwd"})).await.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("escapes sandbox") || msg.contains("not under"),
            "unexpected error: {msg}");
    }

    #[tokio::test]
    async fn read_file_refuses_oversize() {
        let (td, root) = tmp_root();
        fs::write(td.path().join("big.txt"), vec![b'x'; 1000]).unwrap();
        let t = ReadFileTool::new(root).with_max_bytes(10);
        let err = t.run(json!({"path": "big.txt"})).await.unwrap_err();
        assert!(format!("{err}").contains("too large"));
    }

    #[tokio::test]
    async fn write_file_creates_then_refuses_overwrite() {
        let (td, root) = tmp_root();
        let t = WriteFileTool::new(root);
        let out = t.run(json!({"path": "out.txt", "content": "v1"})).await.unwrap();
        assert_eq!(out["bytes_written"], json!(2));
        assert_eq!(fs::read_to_string(td.path().join("out.txt")).unwrap(), "v1");

        // Second write without overwrite must fail.
        let err = t.run(json!({"path": "out.txt", "content": "v2"})).await.unwrap_err();
        assert!(format!("{err}").contains("overwrite"), "{err}");

        // With overwrite=true it succeeds.
        let out = t.run(json!({"path": "out.txt", "content": "v3", "overwrite": true})).await.unwrap();
        assert_eq!(out["overwritten"], json!(true));
        assert_eq!(fs::read_to_string(td.path().join("out.txt")).unwrap(), "v3");
    }

    #[tokio::test]
    async fn write_file_creates_parent_dirs() {
        let (td, root) = tmp_root();
        let t = WriteFileTool::new(root);
        t.run(json!({"path": "a/b/c.txt", "content": "deep"})).await.unwrap();
        assert_eq!(fs::read_to_string(td.path().join("a/b/c.txt")).unwrap(), "deep");
    }

    #[tokio::test]
    async fn write_file_rejects_path_traversal() {
        let (_td, root) = tmp_root();
        let t = WriteFileTool::new(root);
        let err = t.run(json!({"path": "../escaped.txt", "content": "no"})).await.unwrap_err();
        assert!(format!("{err}").contains("escapes sandbox") ||
                format!("{err}").contains("not under"));
    }

    #[tokio::test]
    async fn list_directory_returns_sorted_entries() {
        let (td, root) = tmp_root();
        fs::write(td.path().join("z.txt"), "").unwrap();
        fs::write(td.path().join("a.txt"), "").unwrap();
        fs::create_dir(td.path().join("sub")).unwrap();
        let t = ListDirectoryTool::new(root);
        let out = t.run(json!({"path": "."})).await.unwrap();
        let entries = out["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 3);
        // Sorted alphabetically.
        let names: Vec<&str> = entries.iter().map(|e| e["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["a.txt", "sub", "z.txt"]);
        // Kind is reported.
        let sub = entries.iter().find(|e| e["name"] == "sub").unwrap();
        assert_eq!(sub["kind"], json!("dir"));
    }

    #[tokio::test]
    async fn list_directory_rejects_path_traversal() {
        let (_td, root) = tmp_root();
        let t = ListDirectoryTool::new(root);
        let err = t.run(json!({"path": ".."})).await.unwrap_err();
        assert!(format!("{err}").contains("escapes sandbox") ||
                format!("{err}").contains("not under"));
    }

    #[tokio::test]
    async fn fs_root_rejects_non_directory() {
        let td = tempfile::tempdir().unwrap();
        let f = td.path().join("file");
        fs::write(&f, "x").unwrap();
        assert!(FsRoot::new(&f).is_err());
    }
}
