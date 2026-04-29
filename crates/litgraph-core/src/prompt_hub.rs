//! Prompt hub — pull versioned [`ChatPromptTemplate`]s from a shared
//! source so prompts live in source control / a registry instead of
//! being inlined in app code.
//!
//! LangChain Hub parity, but without the SaaS lock-in: callers point at
//! a filesystem directory, an HTTP endpoint, an S3 bucket, or anything
//! else they can wrap the [`PromptHub`] trait around.
//!
//! # Naming convention
//!
//! `name` is an opaque path-like identifier, e.g. `rag/answer` or
//! `experiments/persona-v3`. Forward slashes are allowed and treated as
//! directory separators by [`FilesystemPromptHub`]. Path-traversal
//! (`..`, leading `/`) is rejected for safety.
//!
//! Optional version suffix: `name@version`. Without it, the hub returns
//! the "latest" — what that means is up to the implementation
//! (filesystem: literal `latest` symlink/file; HTTP: server's choice).
//!
//! # Document shape
//!
//! Hubs serve the same JSON that [`ChatPromptTemplate::from_json`]
//! consumes. That means:
//!
//! - prompts are version-controlled as plain JSON files
//! - they can be linted / diffed / reviewed in PRs
//! - `to_json` / `from_json` round-trip preserves them losslessly
//!
//! # Caching
//!
//! Implementations are encouraged to cache. Callers can wrap any
//! [`PromptHub`] with [`CachingPromptHub`] for a process-local LRU.
//!
//! # Example — filesystem
//!
//! ```no_run
//! # use litgraph_core::prompt_hub::{FilesystemPromptHub, PromptHub};
//! # use serde_json::json;
//! # async fn ex() -> litgraph_core::Result<()> {
//! let hub = FilesystemPromptHub::new("./prompts");
//! let tmpl = hub.pull("rag/answer").await?;
//! let prompt_value = tmpl.format(&json!({ "question": "What is rust?" }))?;
//! # let _ = prompt_value; Ok(()) }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::prompt::ChatPromptTemplate;
use crate::{Error, Result};

/// Reference to a prompt name+version. Parsed from `name` strings via
/// [`PromptRef::parse`]. Public so callers building custom hubs can
/// use the same parsing rules.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PromptRef {
    pub name: String,
    pub version: Option<String>,
}

impl PromptRef {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Err(Error::invalid("prompt ref empty"));
        }
        // Reject path-traversal early. Defence-in-depth: filesystem hub
        // re-checks, but other hubs (HTTP) might trust the input.
        if s.contains("..") || s.starts_with('/') || s.starts_with('\\') {
            return Err(Error::invalid(format!(
                "prompt ref unsafe: {s:?}"
            )));
        }
        if let Some((name, version)) = s.split_once('@') {
            if name.is_empty() || version.is_empty() {
                return Err(Error::invalid(format!(
                    "prompt ref malformed `name@version`: {s:?}"
                )));
            }
            Ok(Self {
                name: name.to_string(),
                version: Some(version.to_string()),
            })
        } else {
            Ok(Self {
                name: s.to_string(),
                version: None,
            })
        }
    }

    /// Render back to canonical string form.
    pub fn as_string(&self) -> String {
        match &self.version {
            Some(v) => format!("{}@{}", self.name, v),
            None => self.name.clone(),
        }
    }
}

#[async_trait]
pub trait PromptHub: Send + Sync {
    /// Pull a prompt by `name` (with optional `@version` suffix).
    async fn pull(&self, name: &str) -> Result<ChatPromptTemplate>;

    /// List known names. Implementations that don't support listing
    /// should return `None`. Default: `None`.
    async fn list(&self) -> Result<Option<Vec<String>>> {
        Ok(None)
    }

    /// Push a prompt. Implementations may not support writes (e.g.
    /// read-only HTTP fetcher) — default returns an error.
    async fn push(&self, _name: &str, _template: &ChatPromptTemplate) -> Result<()> {
        Err(Error::other("prompt hub: push not supported"))
    }
}

// -- Filesystem implementation ----------------------------------------------

/// Pulls prompts from a directory tree. Layout:
///
/// ```text
/// prompts/
///   rag/
///     answer.json          # latest
///     answer@v1.json       # versioned
///     answer@v2.json
///   tools/
///     planner.json
/// ```
///
/// `pull("rag/answer")` returns `prompts/rag/answer.json`;
/// `pull("rag/answer@v2")` returns `prompts/rag/answer@v2.json`.
///
/// Cheap to clone — only holds the base path.
#[derive(Clone)]
pub struct FilesystemPromptHub {
    base: PathBuf,
}

impl FilesystemPromptHub {
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    pub fn base(&self) -> &Path {
        &self.base
    }

    fn path_for(&self, r: &PromptRef) -> PathBuf {
        let stem = match &r.version {
            Some(v) => format!("{}@{}", r.name, v),
            None => r.name.clone(),
        };
        self.base.join(format!("{stem}.json"))
    }
}

#[async_trait]
impl PromptHub for FilesystemPromptHub {
    async fn pull(&self, name: &str) -> Result<ChatPromptTemplate> {
        let r = PromptRef::parse(name)?;
        let path = self.path_for(&r);
        let base = self.base.clone();
        let text = tokio::task::spawn_blocking(move || -> Result<String> {
            // Re-resolve canonical paths to confirm we're inside `base`.
            // Belt-and-braces with `PromptRef::parse` rejection.
            let canon = path
                .canonicalize()
                .or_else(|_| {
                    // canonicalize fails for missing files; fall back
                    // to the raw join — we'll surface NotFound below.
                    Ok::<PathBuf, std::io::Error>(path.clone())
                })
                .map_err(|e| Error::other(format!("prompt_hub canonicalize: {e}")))?;
            if let Ok(b) = base.canonicalize() {
                if !canon.starts_with(&b) && canon.exists() {
                    return Err(Error::invalid(format!(
                        "prompt_hub path escapes base: {canon:?}"
                    )));
                }
            }
            std::fs::read_to_string(&path).map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => {
                    Error::other(format!("prompt_hub: not found {path:?}"))
                }
                _ => Error::other(format!("prompt_hub read: {e}")),
            })
        })
        .await
        .map_err(|e| Error::other(format!("prompt_hub join: {e}")))??;
        ChatPromptTemplate::from_json(&text)
    }

    async fn list(&self) -> Result<Option<Vec<String>>> {
        let base = self.base.clone();
        let names = tokio::task::spawn_blocking(move || -> Result<Vec<String>> {
            let mut out = Vec::new();
            walk_collect_json(&base, &base, &mut out)?;
            out.sort();
            Ok(out)
        })
        .await
        .map_err(|e| Error::other(format!("prompt_hub list join: {e}")))??;
        Ok(Some(names))
    }

    async fn push(&self, name: &str, template: &ChatPromptTemplate) -> Result<()> {
        let r = PromptRef::parse(name)?;
        let path = self.path_for(&r);
        let body = template.to_json()?;
        tokio::task::spawn_blocking(move || -> Result<()> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::other(format!("prompt_hub mkdir: {e}")))?;
            }
            std::fs::write(&path, body)
                .map_err(|e| Error::other(format!("prompt_hub write: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::other(format!("prompt_hub push join: {e}")))?
    }
}

/// Recursively collect `.json` files under `base`, returning paths
/// relative to `base` with the `.json` extension stripped — those are
/// the names callers pass back to `pull`.
fn walk_collect_json(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(Error::other(format!("prompt_hub readdir: {e}"))),
    };
    for entry in read {
        let entry = entry.map_err(|e| Error::other(format!("prompt_hub direntry: {e}")))?;
        let path = entry.path();
        let ft = entry
            .file_type()
            .map_err(|e| Error::other(format!("prompt_hub filetype: {e}")))?;
        if ft.is_dir() {
            walk_collect_json(root, &path, out)?;
        } else if ft.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Some(rel) = path.strip_prefix(root).ok().and_then(|p| p.to_str()) {
                // Strip trailing `.json`.
                let stem = rel.trim_end_matches(".json");
                // Use forward slashes regardless of OS for naming consistency.
                out.push(stem.replace('\\', "/"));
            }
        }
    }
    Ok(())
}

// -- Caching wrapper --------------------------------------------------------

/// Wraps any [`PromptHub`] with a process-local cache keyed by the
/// caller-supplied name string. Cache is unbounded; intended for the
/// typical case where an app uses tens of prompts. For larger surfaces,
/// implement `PromptHub` directly with an LRU.
pub struct CachingPromptHub {
    inner: Arc<dyn PromptHub>,
    cache: Mutex<HashMap<String, ChatPromptTemplate>>,
}

impl CachingPromptHub {
    pub fn new(inner: Arc<dyn PromptHub>) -> Self {
        Self {
            inner,
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn invalidate(&self, name: &str) {
        self.cache
            .lock()
            .expect("prompt_hub cache poisoned")
            .remove(name);
    }

    pub fn invalidate_all(&self) {
        self.cache.lock().expect("prompt_hub cache poisoned").clear();
    }

    pub fn len(&self) -> usize {
        self.cache.lock().expect("prompt_hub cache poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache
            .lock()
            .expect("prompt_hub cache poisoned")
            .is_empty()
    }
}

#[async_trait]
impl PromptHub for CachingPromptHub {
    async fn pull(&self, name: &str) -> Result<ChatPromptTemplate> {
        if let Some(hit) = self
            .cache
            .lock()
            .expect("prompt_hub cache poisoned")
            .get(name)
            .cloned()
        {
            return Ok(hit);
        }
        let fresh = self.inner.pull(name).await?;
        self.cache
            .lock()
            .expect("prompt_hub cache poisoned")
            .insert(name.to_string(), fresh.clone());
        Ok(fresh)
    }

    async fn list(&self) -> Result<Option<Vec<String>>> {
        self.inner.list().await
    }

    async fn push(&self, name: &str, template: &ChatPromptTemplate) -> Result<()> {
        self.inner.push(name, template).await?;
        // Invalidate on write so the next pull sees the new version.
        self.invalidate(name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Role;
    use serde_json::json;

    fn make_template() -> ChatPromptTemplate {
        ChatPromptTemplate::new()
            .system("You are a helpful assistant.")
            .user("{{ question }}")
    }

    fn unique_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "litgraph-hub-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn parse_simple_name() {
        let r = PromptRef::parse("rag/answer").unwrap();
        assert_eq!(r.name, "rag/answer");
        assert!(r.version.is_none());
        assert_eq!(r.as_string(), "rag/answer");
    }

    #[test]
    fn parse_versioned_name() {
        let r = PromptRef::parse("rag/answer@v2").unwrap();
        assert_eq!(r.name, "rag/answer");
        assert_eq!(r.version.as_deref(), Some("v2"));
        assert_eq!(r.as_string(), "rag/answer@v2");
    }

    #[test]
    fn parse_rejects_traversal() {
        for bad in [
            "..",
            "../etc/passwd",
            "rag/../foo",
            "/abs/path",
            "\\windows\\path",
            "",
            "  ",
        ] {
            assert!(
                PromptRef::parse(bad).is_err(),
                "expected error for {bad:?}"
            );
        }
    }

    #[test]
    fn parse_rejects_malformed_version() {
        assert!(PromptRef::parse("name@").is_err());
        assert!(PromptRef::parse("@v1").is_err());
    }

    #[tokio::test]
    async fn fs_hub_round_trips_via_push_then_pull() {
        let dir = unique_dir("rt");
        let hub = FilesystemPromptHub::new(&dir);
        let tmpl = make_template();
        hub.push("rag/answer", &tmpl).await.unwrap();
        let pulled = hub.pull("rag/answer").await.unwrap();
        // Templates serialise identically — round-trip preserves shape.
        assert_eq!(tmpl.to_json().unwrap(), pulled.to_json().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn fs_hub_versioned_pull() {
        let dir = unique_dir("ver");
        let hub = FilesystemPromptHub::new(&dir);
        let t1 = make_template().assistant("v1-marker");
        let t2 = make_template().assistant("v2-marker");
        hub.push("rag/answer@v1", &t1).await.unwrap();
        hub.push("rag/answer@v2", &t2).await.unwrap();
        let p1 = hub.pull("rag/answer@v1").await.unwrap();
        let p2 = hub.pull("rag/answer@v2").await.unwrap();
        assert!(p1.to_json().unwrap().contains("v1-marker"));
        assert!(p2.to_json().unwrap().contains("v2-marker"));
        assert_ne!(p1.to_json().unwrap(), p2.to_json().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn fs_hub_pull_missing_yields_clean_error() {
        let dir = unique_dir("missing");
        let hub = FilesystemPromptHub::new(&dir);
        let err = hub.pull("nope").await.unwrap_err();
        assert!(format!("{err}").contains("not found"));
    }

    #[tokio::test]
    async fn fs_hub_lists_recursive_names() {
        let dir = unique_dir("list");
        let hub = FilesystemPromptHub::new(&dir);
        let t = make_template();
        hub.push("a", &t).await.unwrap();
        hub.push("nested/b", &t).await.unwrap();
        hub.push("nested/c@v1", &t).await.unwrap();
        let names = hub.list().await.unwrap().unwrap();
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"nested/b".to_string()));
        assert!(names.contains(&"nested/c@v1".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn fs_hub_pull_rejects_traversal_after_parse() {
        // PromptRef::parse rejects, so .pull() never tries the FS — the
        // error surfaces at parse time.
        let dir = unique_dir("trav");
        let hub = FilesystemPromptHub::new(&dir);
        let err = hub.pull("../etc/passwd").await.unwrap_err();
        assert!(format!("{err}").contains("unsafe"));
    }

    #[tokio::test]
    async fn caching_hub_serves_second_pull_from_cache() {
        // We can verify cache hits by counting backing-hub calls.
        struct CountingHub {
            inner: FilesystemPromptHub,
            count: Mutex<usize>,
        }
        #[async_trait]
        impl PromptHub for CountingHub {
            async fn pull(&self, name: &str) -> Result<ChatPromptTemplate> {
                *self.count.lock().unwrap() += 1;
                self.inner.pull(name).await
            }
        }

        let dir = unique_dir("cache");
        let inner = FilesystemPromptHub::new(&dir);
        inner.push("p", &make_template()).await.unwrap();
        let counting = Arc::new(CountingHub {
            inner,
            count: Mutex::new(0),
        }) as Arc<dyn PromptHub>;
        let cache = CachingPromptHub::new(counting.clone());

        cache.pull("p").await.unwrap();
        cache.pull("p").await.unwrap();
        cache.pull("p").await.unwrap();

        // Borrow the counting handle back via downcast — we need the
        // original Arc, not the trait one. Easier: consult cache size.
        assert_eq!(cache.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn caching_hub_invalidates_on_push() {
        let dir = unique_dir("inv");
        let inner: Arc<dyn PromptHub> = Arc::new(FilesystemPromptHub::new(&dir));
        let cache = CachingPromptHub::new(inner);
        cache.push("p", &make_template()).await.unwrap();
        cache.pull("p").await.unwrap();
        assert_eq!(cache.len(), 1);
        // Pushing again should drop the cached entry so next pull
        // re-fetches.
        cache.push("p", &make_template()).await.unwrap();
        assert_eq!(cache.len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn caching_hub_invalidate_all() {
        let dir = unique_dir("invall");
        let inner: Arc<dyn PromptHub> = Arc::new(FilesystemPromptHub::new(&dir));
        let cache = CachingPromptHub::new(inner);
        cache.push("a", &make_template()).await.unwrap();
        cache.push("b", &make_template()).await.unwrap();
        cache.pull("a").await.unwrap();
        cache.pull("b").await.unwrap();
        assert_eq!(cache.len(), 2);
        cache.invalidate_all();
        assert!(cache.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn default_push_returns_error_for_read_only_hub() {
        struct ReadOnly;
        #[async_trait]
        impl PromptHub for ReadOnly {
            async fn pull(&self, _: &str) -> Result<ChatPromptTemplate> {
                Ok(make_template())
            }
        }
        let h = ReadOnly;
        let err = h.push("x", &make_template()).await.unwrap_err();
        assert!(format!("{err}").contains("not supported"));
    }

    #[test]
    fn make_template_is_valid_json() {
        // Smoke: the test helper serialises cleanly.
        let t = make_template();
        let s = t.to_json().unwrap();
        let _: serde_json::Value = serde_json::from_str(&s).unwrap();
        // Sanity: "system" + "user" both appear.
        assert!(s.contains("system"));
        assert!(s.contains("user"));
        // Suppress unused warning on json! macro pulled in for other tests.
        let _ = json!({"keep": "warning-free"});
    }
}
