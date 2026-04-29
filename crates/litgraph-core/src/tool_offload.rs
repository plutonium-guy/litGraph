//! Tool-result offload — wrap any [`Tool`] so oversized JSON results get
//! written to a backend (filesystem, in-memory, custom Store) and
//! replaced in the model's context with a small handle.
//!
//! # Why offload
//!
//! LangChain / LangGraph 1.1 added this pattern because models started
//! returning million-token tool results (full file contents, large
//! search dumps, scraped pages). Stuffing those back into the prompt
//! blows the context window AND racks up token cost on every subsequent
//! turn. Offloading keeps the prompt lean while preserving the data:
//! the model sees a handle, can decide to fetch back the full payload
//! via a follow-up tool call, and the user-facing UI can display the
//! preview inline.
//!
//! # Shape of the offloaded marker
//!
//! When a result exceeds [`OffloadingTool::threshold_bytes`], the JSON
//! returned to the agent is replaced with:
//!
//! ```json
//! {
//!   "_offloaded": true,
//!   "handle": "tool-<rand>",
//!   "size_bytes": 12345,
//!   "preview": "first N chars of the stringified result …",
//!   "tool": "search_web"
//! }
//! ```
//!
//! Convention: the `_offloaded` flag lets downstream code distinguish a
//! marker from a normal result that happens to have a `handle` field.
//!
//! # Resolving handles
//!
//! Use [`resolve_handle`] to fetch the full payload back, or expose a
//! companion "fetch_offloaded(handle)" tool to the agent so it can pull
//! the body when it needs to.
//!
//! # Backends
//!
//! - [`InMemoryOffloadBackend`] — `Arc<DashMap>`-style HashMap+Mutex.
//!   Process-local; loses data on restart. Right for tests + short-lived
//!   sessions where the agent fetches back during the same run.
//! - [`FilesystemOffloadBackend`] — writes JSON files under a directory.
//!   Survives restarts. Default file naming: `<handle>.json`.
//! - Custom: implement [`OffloadBackend`] for Postgres / Redis / S3.
//!
//! # Threshold sizing
//!
//! Default [`DEFAULT_THRESHOLD_BYTES`] = 8 KiB — chosen because most
//! "small" tool results (JSON dicts, short search snippets) fit; once
//! you're past 8 KiB you're paying for re-tokenisation on every turn.
//! Tune via [`OffloadingTool::with_threshold_bytes`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Mutex;

use crate::tool::{Tool, ToolSchema};
use crate::{Error, Result};

/// 8 KiB — see module docs for the reasoning.
pub const DEFAULT_THRESHOLD_BYTES: usize = 8 * 1024;

/// First N chars of the stringified payload included in the marker so
/// the model has something to read without having to fetch.
pub const DEFAULT_PREVIEW_BYTES: usize = 256;

#[async_trait]
pub trait OffloadBackend: Send + Sync {
    /// Persist the value under `handle`. Overwrite is allowed.
    async fn put(&self, handle: &str, value: Value) -> Result<()>;
    /// Fetch the value, or `None` if the handle isn't recognised.
    async fn get(&self, handle: &str) -> Result<Option<Value>>;
    /// Best-effort delete. Implementations may no-op if the backend
    /// doesn't support deletion (e.g. immutable object stores).
    async fn delete(&self, handle: &str) -> Result<()>;
}

// ---- In-memory backend -----------------------------------------------------

/// Process-local backend. Cheap, lossy across restarts. Good default
/// for tests + ephemeral sessions.
#[derive(Clone, Default)]
pub struct InMemoryOffloadBackend {
    inner: Arc<Mutex<HashMap<String, Value>>>,
}

impl InMemoryOffloadBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total handles currently stored. Mainly for tests.
    pub fn len(&self) -> usize {
        self.inner.lock().expect("inmem offload poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().expect("inmem offload poisoned").is_empty()
    }
}

#[async_trait]
impl OffloadBackend for InMemoryOffloadBackend {
    async fn put(&self, handle: &str, value: Value) -> Result<()> {
        self.inner
            .lock()
            .expect("inmem offload poisoned")
            .insert(handle.to_string(), value);
        Ok(())
    }
    async fn get(&self, handle: &str) -> Result<Option<Value>> {
        Ok(self
            .inner
            .lock()
            .expect("inmem offload poisoned")
            .get(handle)
            .cloned())
    }
    async fn delete(&self, handle: &str) -> Result<()> {
        self.inner
            .lock()
            .expect("inmem offload poisoned")
            .remove(handle);
        Ok(())
    }
}

// ---- Filesystem backend ----------------------------------------------------

/// JSON-on-disk backend. One file per handle: `<dir>/<handle>.json`.
/// Directory is created lazily on first write.
#[derive(Clone)]
pub struct FilesystemOffloadBackend {
    dir: PathBuf,
}

impl FilesystemOffloadBackend {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn path_for(&self, handle: &str) -> PathBuf {
        self.dir.join(format!("{handle}.json"))
    }
}

#[async_trait]
impl OffloadBackend for FilesystemOffloadBackend {
    async fn put(&self, handle: &str, value: Value) -> Result<()> {
        // Reject handles that would escape the directory. Defence-in-
        // depth: handles are caller-supplied via OffloadingTool which
        // generates them, but an exposed Python binding could pass any
        // string.
        if handle.is_empty() || handle.contains('/') || handle.contains("..") {
            return Err(Error::invalid(format!(
                "offload handle invalid: {handle:?}"
            )));
        }
        let dir = self.dir.clone();
        let path = self.path_for(handle);
        // Sync IO inside spawn_blocking to keep the runtime non-blocking
        // without pulling in tokio's `fs` feature (which would force the
        // feature on every dependent crate).
        tokio::task::spawn_blocking(move || -> Result<()> {
            std::fs::create_dir_all(&dir)
                .map_err(|e| Error::other(format!("offload mkdir: {e}")))?;
            let bytes = serde_json::to_vec(&value)
                .map_err(|e| Error::other(format!("offload encode: {e}")))?;
            std::fs::write(&path, bytes)
                .map_err(|e| Error::other(format!("offload write: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::other(format!("offload join: {e}")))?
    }

    async fn get(&self, handle: &str) -> Result<Option<Value>> {
        let path = self.path_for(handle);
        tokio::task::spawn_blocking(move || -> Result<Option<Value>> {
            match std::fs::read(&path) {
                Ok(bytes) => {
                    let v: Value = serde_json::from_slice(&bytes)
                        .map_err(|e| Error::other(format!("offload decode: {e}")))?;
                    Ok(Some(v))
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(Error::other(format!("offload read: {e}"))),
            }
        })
        .await
        .map_err(|e| Error::other(format!("offload join: {e}")))?
    }

    async fn delete(&self, handle: &str) -> Result<()> {
        let path = self.path_for(handle);
        tokio::task::spawn_blocking(move || -> Result<()> {
            match std::fs::remove_file(&path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(Error::other(format!("offload delete: {e}"))),
            }
        })
        .await
        .map_err(|e| Error::other(format!("offload join: {e}")))?
    }
}

// ---- The wrapping Tool -----------------------------------------------------

/// Wraps a [`Tool`] so oversized JSON results get offloaded automatically.
/// Implements `Tool` itself, so it slots transparently into any agent /
/// graph that holds `Arc<dyn Tool>`.
pub struct OffloadingTool {
    inner: Arc<dyn Tool>,
    backend: Arc<dyn OffloadBackend>,
    threshold_bytes: usize,
    preview_bytes: usize,
    handle_prefix: String,
}

impl OffloadingTool {
    pub fn new(inner: Arc<dyn Tool>, backend: Arc<dyn OffloadBackend>) -> Self {
        let prefix = inner.name();
        Self {
            inner,
            backend,
            threshold_bytes: DEFAULT_THRESHOLD_BYTES,
            preview_bytes: DEFAULT_PREVIEW_BYTES,
            handle_prefix: prefix,
        }
    }

    pub fn with_threshold_bytes(mut self, n: usize) -> Self {
        self.threshold_bytes = n;
        self
    }

    pub fn with_preview_bytes(mut self, n: usize) -> Self {
        self.preview_bytes = n;
        self
    }

    /// Override the prefix used in generated handle names. Default = the
    /// inner tool's name. Useful when wrapping multiple tools that share
    /// a backend and you want to namespace them.
    pub fn with_handle_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.handle_prefix = prefix.into();
        self
    }

    /// Build the offload marker. Public for tests + so callers writing a
    /// custom OffloadingTool variant can reuse the format.
    pub fn build_marker(&self, handle: &str, raw: &Value) -> Value {
        let raw_str = serde_json::to_string(raw).unwrap_or_default();
        let preview = truncate_chars(&raw_str, self.preview_bytes);
        json!({
            "_offloaded": true,
            "handle": handle,
            "size_bytes": raw_str.len(),
            "preview": preview,
            "tool": self.inner.name(),
        })
    }

    fn next_handle(&self) -> String {
        // Process-local atomic + nanos covers parallel calls within the
        // same process; uuid would be nicer but we don't want a uuid dep
        // for one helper.
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("{}-{nanos:x}-{n:x}", sanitize_handle_prefix(&self.handle_prefix))
    }
}

fn sanitize_handle_prefix(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

#[async_trait]
impl Tool for OffloadingTool {
    fn schema(&self) -> ToolSchema {
        // Pass-through — the model still sees the original tool surface.
        // The offloading is invisible to the model except via the marker
        // shape on big results.
        self.inner.schema()
    }

    fn name(&self) -> String {
        self.inner.name()
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let raw = self.inner.run(args).await?;
        let raw_str = serde_json::to_string(&raw)
            .map_err(|e| Error::other(format!("offload measure: {e}")))?;
        if raw_str.len() <= self.threshold_bytes {
            return Ok(raw);
        }
        let handle = self.next_handle();
        self.backend.put(&handle, raw.clone()).await?;
        Ok(self.build_marker(&handle, &raw))
    }
}

// ---- Helpers ---------------------------------------------------------------

/// Resolve an offloaded handle back to its original payload. Returns
/// `Ok(None)` if the handle isn't recognised. Handy for building a
/// companion "fetch_offloaded" tool exposed to the agent.
pub async fn resolve_handle(
    backend: &Arc<dyn OffloadBackend>,
    handle: &str,
) -> Result<Option<Value>> {
    backend.get(handle).await
}

/// Inspect a JSON value and return the offload handle if it looks like
/// a marker (i.e. has `_offloaded == true` AND a string `handle`).
pub fn is_offloaded_marker(v: &Value) -> Option<&str> {
    let obj = v.as_object()?;
    if obj.get("_offloaded").and_then(|x| x.as_bool()) != Some(true) {
        return None;
    }
    obj.get("handle").and_then(|x| x.as_str())
}

/// Truncate at char (not byte) boundary so we don't slice mid-codepoint.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    if max_chars == 0 || s.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(s.len().min(max_chars * 4));
    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

/// Try a directory under `LITGRAPH_OFFLOAD_DIR`, falling back to a
/// platform-appropriate temp dir. Returns `None` if neither is writable.
/// Convenience for callers who want a "just works" filesystem backend.
pub fn default_offload_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("LITGRAPH_OFFLOAD_DIR") {
        let p = PathBuf::from(d);
        if ensure_writable(&p).is_ok() {
            return Some(p);
        }
    }
    let p = std::env::temp_dir().join("litgraph-offload");
    if ensure_writable(&p).is_ok() {
        return Some(p);
    }
    None
}

fn ensure_writable(p: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(p)?;
    let probe = p.join(".litgraph-write-probe");
    std::fs::write(&probe, b"")?;
    let _ = std::fs::remove_file(&probe);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolSchema;

    /// Tool that returns a payload of caller-controllable size — lets
    /// tests probe both above and below the threshold.
    struct PayloadTool {
        size: usize,
    }

    #[async_trait]
    impl Tool for PayloadTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "payload".into(),
                description: "test".into(),
                parameters: json!({"type": "object"}),
            }
        }
        async fn run(&self, _args: Value) -> Result<Value> {
            Ok(json!({ "data": "x".repeat(self.size) }))
        }
    }

    #[tokio::test]
    async fn small_result_passes_through_unchanged() {
        let inner = Arc::new(PayloadTool { size: 100 }) as Arc<dyn Tool>;
        let backend: Arc<dyn OffloadBackend> = Arc::new(InMemoryOffloadBackend::new());
        let tool = OffloadingTool::new(inner, backend.clone()).with_threshold_bytes(1024);
        let out = tool.run(json!({})).await.unwrap();
        assert!(out.get("_offloaded").is_none());
        assert_eq!(out["data"].as_str().unwrap().len(), 100);
        // Backend stayed empty. Inspect via a separately-held in-memory
        // handle (we keep one for the test by constructing with the
        // concrete type and cloning into the trait object).
        let mem = InMemoryOffloadBackend::new();
        let inner = Arc::new(PayloadTool { size: 100 }) as Arc<dyn Tool>;
        let backend2: Arc<dyn OffloadBackend> = Arc::new(mem.clone());
        let tool2 = OffloadingTool::new(inner, backend2).with_threshold_bytes(1024);
        let _ = tool2.run(json!({})).await.unwrap();
        assert_eq!(mem.len(), 0);
    }

    #[tokio::test]
    async fn large_result_offloaded_and_marker_returned() {
        let inner = Arc::new(PayloadTool { size: 50_000 }) as Arc<dyn Tool>;
        let mem = InMemoryOffloadBackend::new();
        let backend: Arc<dyn OffloadBackend> = Arc::new(mem.clone());
        let tool = OffloadingTool::new(inner, backend).with_threshold_bytes(1024);
        let out = tool.run(json!({})).await.unwrap();
        assert_eq!(out["_offloaded"].as_bool(), Some(true));
        let handle = out["handle"].as_str().unwrap().to_string();
        assert!(handle.starts_with("payload-"));
        assert!(out["size_bytes"].as_u64().unwrap() > 1024);
        let preview = out["preview"].as_str().unwrap();
        assert!(!preview.is_empty());
        assert!(preview.len() <= DEFAULT_PREVIEW_BYTES * 4 + 5); // chars→bytes worst case
        assert_eq!(out["tool"].as_str(), Some("payload"));

        // Backend got the full payload.
        assert_eq!(mem.len(), 1);
        let fetched = mem.get(&handle).await.unwrap().unwrap();
        assert_eq!(fetched["data"].as_str().unwrap().len(), 50_000);
    }

    #[tokio::test]
    async fn resolve_handle_round_trips() {
        let inner = Arc::new(PayloadTool { size: 50_000 }) as Arc<dyn Tool>;
        let backend: Arc<dyn OffloadBackend> = Arc::new(InMemoryOffloadBackend::new());
        let tool = OffloadingTool::new(inner, backend.clone()).with_threshold_bytes(1024);
        let out = tool.run(json!({})).await.unwrap();
        let handle = out["handle"].as_str().unwrap();
        let resolved = resolve_handle(&backend, handle).await.unwrap().unwrap();
        assert_eq!(resolved["data"].as_str().unwrap().len(), 50_000);
    }

    #[tokio::test]
    async fn missing_handle_returns_none() {
        let backend: Arc<dyn OffloadBackend> = Arc::new(InMemoryOffloadBackend::new());
        assert!(resolve_handle(&backend, "nonexistent").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_removes_payload() {
        let backend = InMemoryOffloadBackend::new();
        backend.put("h1", json!({"x": 1})).await.unwrap();
        assert!(backend.get("h1").await.unwrap().is_some());
        backend.delete("h1").await.unwrap();
        assert!(backend.get("h1").await.unwrap().is_none());
        // Idempotent.
        backend.delete("h1").await.unwrap();
    }

    #[tokio::test]
    async fn fs_backend_round_trips() {
        let dir = std::env::temp_dir().join(format!(
            "litgraph-fs-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let backend = FilesystemOffloadBackend::new(&dir);
        backend
            .put("h1", json!({"x": "value"}))
            .await
            .unwrap();
        let got = backend.get("h1").await.unwrap().unwrap();
        assert_eq!(got["x"].as_str(), Some("value"));
        backend.delete("h1").await.unwrap();
        assert!(backend.get("h1").await.unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn fs_backend_rejects_path_traversal() {
        let dir = std::env::temp_dir().join("litgraph-fs-traversal");
        let backend = FilesystemOffloadBackend::new(&dir);
        for bad in ["../foo", "a/b", "..", ""] {
            let err = backend.put(bad, json!({})).await.unwrap_err();
            assert!(format!("{err}").contains("invalid"), "bad={bad}, err={err}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn marker_recogniser_distinguishes_offloaded_from_normal() {
        let marker = json!({"_offloaded": true, "handle": "h"});
        assert_eq!(is_offloaded_marker(&marker), Some("h"));

        let not_marker = json!({"handle": "h"}); // no _offloaded flag
        assert!(is_offloaded_marker(&not_marker).is_none());

        let false_flag = json!({"_offloaded": false, "handle": "h"});
        assert!(is_offloaded_marker(&false_flag).is_none());

        let no_handle = json!({"_offloaded": true});
        assert!(is_offloaded_marker(&no_handle).is_none());

        let scalar = json!("just a string");
        assert!(is_offloaded_marker(&scalar).is_none());
    }

    #[tokio::test]
    async fn handle_prefix_sanitised() {
        // Inner tool with name containing characters the FS backend
        // would reject — wrapper must sanitise to keep handles safe.
        struct WeirdName;
        #[async_trait]
        impl Tool for WeirdName {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "weird/name with spaces".into(),
                    description: "x".into(),
                    parameters: json!({"type": "object"}),
                }
            }
            async fn run(&self, _: Value) -> Result<Value> {
                Ok(json!({"x": "x".repeat(50_000)}))
            }
        }
        let backend: Arc<dyn OffloadBackend> = Arc::new(InMemoryOffloadBackend::new());
        let tool = OffloadingTool::new(Arc::new(WeirdName), backend.clone())
            .with_threshold_bytes(1024);
        let out = tool.run(json!({})).await.unwrap();
        let handle = out["handle"].as_str().unwrap();
        assert!(!handle.contains('/'));
        assert!(!handle.contains(' '));
    }

    #[tokio::test]
    async fn preview_clipped_to_configured_byte_budget() {
        let inner = Arc::new(PayloadTool { size: 100_000 }) as Arc<dyn Tool>;
        let backend: Arc<dyn OffloadBackend> = Arc::new(InMemoryOffloadBackend::new());
        let tool = OffloadingTool::new(inner, backend)
            .with_threshold_bytes(1024)
            .with_preview_bytes(64);
        let out = tool.run(json!({})).await.unwrap();
        let preview = out["preview"].as_str().unwrap();
        // 64 chars max, plus the ellipsis sentinel.
        assert!(preview.chars().count() <= 65, "preview={preview}");
    }

    #[tokio::test]
    async fn schema_passthrough_preserves_inner_surface() {
        let inner = Arc::new(PayloadTool { size: 10 }) as Arc<dyn Tool>;
        let backend: Arc<dyn OffloadBackend> = Arc::new(InMemoryOffloadBackend::new());
        let tool = OffloadingTool::new(inner, backend);
        assert_eq!(tool.schema().name, "payload");
        assert_eq!(tool.name(), "payload");
    }

    #[test]
    fn truncate_chars_handles_unicode_boundary() {
        // Multi-byte chars: ensure we don't panic on mid-codepoint slice.
        let s = "日本語テスト";
        let out = truncate_chars(s, 3);
        assert_eq!(out, "日本語…");
    }

    #[test]
    fn truncate_chars_empty_inputs() {
        assert_eq!(truncate_chars("", 100), "");
        assert_eq!(truncate_chars("abc", 0), "");
    }

    #[test]
    fn default_offload_dir_returns_some() {
        assert!(default_offload_dir().is_some());
    }
}
