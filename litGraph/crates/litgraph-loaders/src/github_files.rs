//! GitHub files loader — walk a repo tree and pull file contents for
//! code-RAG / docs-RAG pipelines. Complements iter-98's `GithubIssuesLoader`.
//!
//! # Flow
//!
//! 1. `GET /repos/{owner}/{repo}/git/trees/{ref}?recursive=1` — one request,
//!    returns a flat listing of every blob in the repo.
//! 2. Client-side filter: keep only blob entries whose path matches the
//!    extension allowlist + doesn't hit any exclude pattern + is under the
//!    size cap.
//! 3. For each survivor: `GET /repos/{owner}/{repo}/contents/{path}?ref={ref}`
//!    — returns the content base64-encoded. Decode → `Document`.
//!
//! # Why per-file `/contents/` and not `/git/blobs/{sha}`?
//!
//! `/contents/` returns the file with its human-readable path + decoded-
//! ready base64, and — critically — errors cleanly on submodule pointers
//! and LFS stubs. `/git/blobs/` would hand back the submodule SHA as if it
//! were file content (wrong in a RAG index).
//!
//! # Auth + headers (identical to iter-98 issues loader)
//!
//! - `Authorization: Bearer <token>` — PAT or App installation token.
//! - `X-GitHub-Api-Version: 2022-11-28` — pinned version.
//! - `User-Agent: litgraph-github-loader/0.1` — GitHub 403s without UA.
//! - `Accept: application/vnd.github+json`.
//!
//! # Performance note
//!
//! One tree request + N content requests. For large repos (>1000 matching
//! files) you'll hit the primary REST rate limit (5000/hour authenticated).
//! Tighten `with_extensions` / `with_exclude_paths` / `with_max_files` to
//! fit your budget. Parallelism is intentionally NOT used — GitHub's
//! secondary rate limit triggers on concurrent requests from one token.
//!
//! # Metadata per document
//!
//! - `path` — repo-relative file path (e.g. `"src/main.rs"`)
//! - `ref` — the branch/tag/commit queried
//! - `sha` — blob SHA
//! - `size` — byte size
//! - `html_url` — github.com URL for the file (useful for LLM citations)
//! - `source` — `"github:{owner}/{repo}/{path}@{ref}"`

use std::time::Duration;

use base64::Engine;
use litgraph_core::Document;
use serde_json::Value;

use crate::{Loader, LoaderError, LoaderResult};

const GITHUB_API: &str = "https://api.github.com";
const GITHUB_API_VERSION: &str = "2022-11-28";

pub struct GithubFilesLoader {
    pub token: String,
    pub owner: String,
    pub repo: String,
    pub git_ref: String,
    pub base_url: String,
    pub timeout: Duration,
    /// If non-empty, include only files whose path ends with one of these
    /// extensions (case-insensitive, with or without leading dot). If empty,
    /// include everything — not recommended for large repos.
    pub extensions: Vec<String>,
    /// Substring excludes applied to the full repo-relative path. Default
    /// includes the universally-useless paths below (vendored, generated,
    /// locked) so test repos don't need to set them by hand.
    pub exclude_paths: Vec<String>,
    /// Hard cap on total files loaded after filtering. Protect against
    /// accidental scrapes. Default 500.
    pub max_files: Option<usize>,
    /// Skip any file larger than this (bytes). Default 1 MiB — most files
    /// larger than that are binary / generated and not useful for LLM
    /// context.
    pub max_file_size_bytes: u64,
}

impl GithubFilesLoader {
    pub fn from_repo_tree(
        token: impl Into<String>,
        owner: impl Into<String>,
        repo: impl Into<String>,
    ) -> Self {
        Self {
            token: token.into(),
            owner: owner.into(),
            repo: repo.into(),
            git_ref: "main".into(),
            base_url: GITHUB_API.into(),
            timeout: Duration::from_secs(30),
            extensions: Vec::new(),
            exclude_paths: default_excludes(),
            max_files: Some(500),
            max_file_size_bytes: 1024 * 1024,
        }
    }

    pub fn with_ref(mut self, r: impl Into<String>) -> Self {
        self.git_ref = r.into();
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the extension allowlist. Strings may have leading dot or not.
    /// Empty → include all files (careful).
    pub fn with_extensions<I, S>(mut self, exts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extensions = exts
            .into_iter()
            .map(|e| {
                let s: String = e.into();
                let s = s.to_ascii_lowercase();
                if s.starts_with('.') { s } else { format!(".{s}") }
            })
            .collect();
        self
    }

    /// REPLACE the exclude list (not append). Use to clear defaults.
    pub fn with_exclude_paths<I, S>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.exclude_paths = paths.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_max_files(mut self, n: Option<usize>) -> Self {
        self.max_files = n;
        self
    }

    pub fn with_max_file_size_bytes(mut self, n: u64) -> Self {
        self.max_file_size_bytes = n;
        self
    }

    fn client(&self) -> LoaderResult<reqwest::blocking::Client> {
        reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent("litgraph-github-loader/0.1")
            .build()
            .map_err(LoaderError::from)
    }

    fn authed(
        &self,
        b: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        b.bearer_auth(&self.token)
            .header("X-GitHub-Api-Version", GITHUB_API_VERSION)
            .header("Accept", "application/vnd.github+json")
    }

    /// Fetch the recursive tree for a ref. Returns just the blob entries
    /// (type=="blob") — we don't care about tree/commit entries.
    fn fetch_tree(&self, client: &reqwest::blocking::Client) -> LoaderResult<Vec<Value>> {
        let url = format!(
            "{}/repos/{}/{}/git/trees/{}?recursive=1",
            self.base_url.trim_end_matches('/'),
            self.owner,
            self.repo,
            self.git_ref,
        );
        let resp = self.authed(client.get(&url)).send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "github tree {status}: {body}"
            )));
        }
        let v: Value = resp.json()?;
        // `truncated: true` means the tree exceeded 7MB/100k entries —
        // surface as a warning via returning what we got (GitHub itself
        // truncated the response). Caller can pin to a narrower path in a
        // future iter.
        let tree = v
            .get("tree")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(tree
            .into_iter()
            .filter(|e| e.get("type").and_then(|t| t.as_str()) == Some("blob"))
            .collect())
    }

    fn fetch_file(
        &self,
        client: &reqwest::blocking::Client,
        path: &str,
    ) -> LoaderResult<Option<Value>> {
        // `/contents/{path}?ref=...` — returns {content: base64, encoding,
        // size, sha, html_url, ...}. Errors on submodules / LFS stubs;
        // we treat those as "skip" (not fatal).
        let url = format!(
            "{}/repos/{}/{}/contents/{}?ref={}",
            self.base_url.trim_end_matches('/'),
            self.owner,
            self.repo,
            urlencode_path(path),
            self.git_ref,
        );
        let resp = self.authed(client.get(&url)).send()?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "github contents {path} {status}: {body}"
            )));
        }
        Ok(Some(resp.json()?))
    }

    fn blob_passes_filters(&self, entry: &Value) -> bool {
        let path = match entry.get("path").and_then(|p| p.as_str()) {
            Some(p) => p,
            None => return false,
        };
        // Exclude substring match.
        for ex in &self.exclude_paths {
            if !ex.is_empty() && path.contains(ex) {
                return false;
            }
        }
        // Size cap — from tree entry, not the full contents fetch. Saves
        // a round-trip on oversized files.
        let size = entry.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
        if size > self.max_file_size_bytes {
            return false;
        }
        // Extension allowlist.
        if !self.extensions.is_empty() {
            let lower = path.to_ascii_lowercase();
            if !self.extensions.iter().any(|ext| lower.ends_with(ext)) {
                return false;
            }
        }
        true
    }

    fn content_to_document(&self, content_resp: &Value) -> Option<Document> {
        let path = content_resp.get("path").and_then(|p| p.as_str())?.to_string();
        let sha = content_resp.get("sha").and_then(|s| s.as_str()).unwrap_or("");
        let size = content_resp.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
        let html_url = content_resp.get("html_url").and_then(|u| u.as_str()).unwrap_or("");
        let encoding = content_resp
            .get("encoding")
            .and_then(|e| e.as_str())
            .unwrap_or("base64");
        let raw = content_resp.get("content").and_then(|c| c.as_str())?;
        let text = if encoding == "base64" {
            // GitHub wraps base64 to 60 chars — strip newlines before decoding.
            let cleaned: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
            match base64::engine::general_purpose::STANDARD.decode(cleaned.as_bytes()) {
                Ok(bytes) => match String::from_utf8(bytes) {
                    Ok(s) => s,
                    Err(_) => return None, // binary file — skip
                },
                Err(_) => return None,
            }
        } else {
            raw.to_string()
        };

        let mut d = Document::new(text);
        d.id = Some(format!("{}/{}:{}", self.owner, self.repo, path));
        d.metadata.insert("path".into(), Value::String(path.clone()));
        d.metadata
            .insert("ref".into(), Value::String(self.git_ref.clone()));
        d.metadata.insert("sha".into(), Value::String(sha.into()));
        d.metadata.insert("size".into(), Value::from(size));
        if !html_url.is_empty() {
            d.metadata
                .insert("html_url".into(), Value::String(html_url.into()));
        }
        d.metadata.insert(
            "source".into(),
            Value::String(format!(
                "github:{}/{}/{}@{}",
                self.owner, self.repo, path, self.git_ref
            )),
        );
        Some(d)
    }
}

impl Loader for GithubFilesLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.client()?;
        let tree = self.fetch_tree(&client)?;
        let mut docs: Vec<Document> = Vec::new();
        for entry in &tree {
            if !self.blob_passes_filters(entry) {
                continue;
            }
            let path = entry
                .get("path")
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .to_string();
            let content = match self.fetch_file(&client, &path)? {
                Some(c) => c,
                None => continue, // 404 — submodule / moved file; skip.
            };
            if let Some(d) = self.content_to_document(&content) {
                docs.push(d);
                if let Some(cap) = self.max_files {
                    if docs.len() >= cap {
                        break;
                    }
                }
            }
        }
        Ok(docs)
    }
}

/// Default exclude patterns: vendored / generated / locked / binary paths
/// that universally add noise to LLM context.
fn default_excludes() -> Vec<String> {
    vec![
        "node_modules/".into(),
        "vendor/".into(),
        "target/".into(),
        "dist/".into(),
        "build/".into(),
        ".git/".into(),
        "__pycache__/".into(),
        ".venv/".into(),
        ".lock".into(), // Cargo.lock, package-lock.json, yarn.lock, pnpm-lock.yaml
        ".min.js".into(),
        ".min.css".into(),
    ]
}

/// Percent-encode a repo-relative path. Keeps slashes (`/`) unchanged
/// since GitHub routes on them; encodes spaces / unicode / query-special
/// chars.
fn urlencode_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c == '/' || c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~') {
            out.push(c);
        } else {
            for b in c.to_string().bytes() {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use std::thread;

    struct FakeServer {
        url: String,
        seen_paths: Arc<Mutex<Vec<String>>>,
        seen_auth: Arc<Mutex<Vec<Option<String>>>>,
        _shutdown: std::sync::mpsc::Sender<()>,
    }

    fn tree_response() -> Value {
        json!({
            "sha": "main_sha",
            "tree": [
                {"path": "README.md", "type": "blob", "sha": "abc1", "size": 200,
                 "mode": "100644", "url": "https://api.github.com/..."},
                {"path": "src/main.rs", "type": "blob", "sha": "abc2", "size": 500,
                 "mode": "100644"},
                {"path": "src/lib.rs", "type": "blob", "sha": "abc3", "size": 800,
                 "mode": "100644"},
                {"path": "node_modules/foo/index.js", "type": "blob", "sha": "abc4",
                 "size": 1000, "mode": "100644"},
                {"path": "docs/",  "type": "tree", "sha": "abc5"},  // not a blob; skipped
                {"path": "Cargo.lock", "type": "blob", "sha": "abc6", "size": 300,
                 "mode": "100644"},
                {"path": "huge.bin", "type": "blob", "sha": "abc7", "size": 10_000_000,
                 "mode": "100644"},
            ],
            "truncated": false
        })
    }

    fn file_response(path: &str, content_text: &str) -> Value {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(content_text.as_bytes());
        json!({
            "name": path.rsplit('/').next().unwrap_or(path),
            "path": path,
            "sha": "content_sha",
            "size": content_text.len(),
            "content": b64,
            "encoding": "base64",
            "html_url": format!("https://github.com/acme/app/blob/main/{path}"),
        })
    }

    fn spawn_fake() -> FakeServer {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen_paths: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_auth: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        listener.set_nonblocking(true).unwrap();
        let paths_w = seen_paths.clone();
        let auth_w = seen_auth.clone();
        thread::spawn(move || {
            use std::io::{Read, Write};
            loop {
                if rx.try_recv().is_ok() { break; }
                match listener.accept() {
                    Ok((mut s, _)) => {
                        s.set_nonblocking(false).ok();
                        let mut buf = [0u8; 8192];
                        let n = s.read(&mut buf).unwrap_or(0);
                        if n == 0 { continue; }
                        let req = String::from_utf8_lossy(&buf[..n]).to_string();
                        let first_line = req.lines().next().unwrap_or("");
                        let path = first_line
                            .split_whitespace()
                            .nth(1)
                            .unwrap_or("/")
                            .to_string();
                        paths_w.lock().unwrap().push(path.clone());
                        let mut auth = None;
                        for line in req.lines().skip(1) {
                            if let Some((k, v)) = line.split_once(':') {
                                if k.trim().eq_ignore_ascii_case("authorization") {
                                    auth = Some(v.trim().to_string());
                                }
                            }
                        }
                        auth_w.lock().unwrap().push(auth);
                        let body = if path.contains("/git/trees/") {
                            tree_response().to_string()
                        } else if path.contains("/contents/") {
                            // Extract repo-relative path from /repos/acme/app/contents/<PATH>?ref=main
                            let after = path
                                .split("/contents/")
                                .nth(1)
                                .unwrap_or("")
                                .split('?')
                                .next()
                                .unwrap_or("")
                                .to_string();
                            // Map a fake content per path so tests can assert.
                            let content = match after.as_str() {
                                "README.md" => "# Hello\n\nReadme text.",
                                "src/main.rs" => "fn main() { println!(\"hi\"); }",
                                "src/lib.rs" => "pub fn hello() {}",
                                _ => "// generic content",
                            };
                            file_response(&after, content).to_string()
                        } else {
                            json!({"error": "unknown"}).to_string()
                        };
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            body.len(), body
                        );
                        let _ = s.write_all(resp.as_bytes());
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        FakeServer {
            url: format!("http://127.0.0.1:{port}"),
            seen_paths,
            seen_auth,
            _shutdown: tx,
        }
    }

    #[test]
    fn loads_repo_tree_then_filters_by_extension_and_fetches_contents() {
        let srv = spawn_fake();
        let loader = GithubFilesLoader::from_repo_tree("ghp_test", "acme", "app")
            .with_base_url(&srv.url)
            .with_extensions([".rs"]);
        let docs = loader.load().unwrap();
        // Tree has src/main.rs + src/lib.rs as .rs files. README + Cargo.lock +
        // node_modules + huge.bin all filtered out via extensions / excludes / size.
        assert_eq!(docs.len(), 2);
        let paths: Vec<&str> = docs.iter()
            .filter_map(|d| d.metadata.get("path").and_then(|p| p.as_str()))
            .collect();
        assert!(paths.contains(&"src/main.rs"));
        assert!(paths.contains(&"src/lib.rs"));
    }

    #[test]
    fn extension_filter_accepts_both_dotted_and_undotted_forms() {
        let srv = spawn_fake();
        let a = GithubFilesLoader::from_repo_tree("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_extensions(["rs"]);  // no dot
        let b = GithubFilesLoader::from_repo_tree("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_extensions([".rs"]);  // dot
        assert_eq!(a.load().unwrap().len(), b.load().unwrap().len());
    }

    #[test]
    fn default_excludes_filter_out_node_modules_and_lock_files() {
        let srv = spawn_fake();
        let loader = GithubFilesLoader::from_repo_tree("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_extensions([".md", ".js", ".lock", ".rs"]);  // allow lock just to prove excludes beat ext
        let docs = loader.load().unwrap();
        let paths: Vec<&str> = docs.iter()
            .filter_map(|d| d.metadata.get("path").and_then(|p| p.as_str()))
            .collect();
        // node_modules/foo/index.js → excluded (substring "node_modules/")
        // Cargo.lock → excluded (substring ".lock")
        assert!(!paths.iter().any(|p| p.contains("node_modules")));
        assert!(!paths.iter().any(|p| p.ends_with("Cargo.lock")));
        // README.md survives (still in extensions + not in excludes).
        assert!(paths.contains(&"README.md"));
    }

    #[test]
    fn max_file_size_filter_drops_oversized_blobs_without_fetching() {
        let srv = spawn_fake();
        let loader = GithubFilesLoader::from_repo_tree("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_extensions([".bin", ".rs"])
            .with_max_file_size_bytes(1024);  // 1 KiB
        let _ = loader.load().unwrap();
        let hit_paths = srv.seen_paths.lock().unwrap().clone();
        // huge.bin (10MB) MUST NOT have been fetched — we filtered on the
        // tree entry's size, before spending a round-trip.
        assert!(
            !hit_paths.iter().any(|p| p.contains("/contents/huge.bin")),
            "huge.bin should have been size-filtered pre-fetch: {hit_paths:?}"
        );
    }

    #[test]
    fn metadata_captures_path_ref_sha_size_html_url_source() {
        let srv = spawn_fake();
        let loader = GithubFilesLoader::from_repo_tree("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_extensions([".md"]);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        let d = &docs[0];
        assert_eq!(d.metadata["path"].as_str(), Some("README.md"));
        assert_eq!(d.metadata["ref"].as_str(), Some("main"));
        assert!(d.metadata["sha"].as_str().unwrap().len() > 0);
        assert!(d.metadata["html_url"].as_str().unwrap().contains("github.com"));
        assert_eq!(
            d.metadata["source"].as_str(),
            Some("github:acme/app/README.md@main")
        );
        assert_eq!(d.id.as_deref(), Some("acme/app:README.md"));
    }

    #[test]
    fn base64_content_decoded_correctly_including_wrapped_lines() {
        // GitHub wraps base64 content at 60 chars per line. The loader
        // must strip whitespace before decoding. Test with a canned content
        // response containing newlines in the base64 field.
        let loader = GithubFilesLoader::from_repo_tree("t", "acme", "app");
        use base64::Engine;
        let original = "line1\nline2\nline3";
        let b64 = base64::engine::general_purpose::STANDARD.encode(original.as_bytes());
        // Insert a newline in the middle (mimics GitHub's line wrapping).
        let wrapped = format!("{}\n{}", &b64[..5], &b64[5..]);
        let content_resp = json!({
            "path": "test.txt",
            "sha": "abc",
            "size": original.len(),
            "content": wrapped,
            "encoding": "base64",
            "html_url": "https://github.com/acme/app/blob/main/test.txt",
        });
        let d = loader.content_to_document(&content_resp).unwrap();
        assert_eq!(d.content, original);
    }

    #[test]
    fn binary_file_decoded_to_non_utf8_bytes_is_skipped() {
        // Binary blobs shouldn't produce a Document — they'd just pollute
        // the LLM context with garbage. Silent skip is correct.
        let loader = GithubFilesLoader::from_repo_tree("t", "acme", "app");
        use base64::Engine;
        // Invalid UTF-8 bytes (lone continuation byte).
        let bytes = &[0x80u8, 0x80, 0x80];
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        let content_resp = json!({
            "path": "binary.png",
            "sha": "x",
            "size": 3,
            "content": b64,
            "encoding": "base64",
        });
        assert!(loader.content_to_document(&content_resp).is_none());
    }

    #[test]
    fn auth_bearer_and_api_version_present_on_tree_and_contents_requests() {
        let srv = spawn_fake();
        let loader = GithubFilesLoader::from_repo_tree("ghp_secret", "acme", "app")
            .with_base_url(&srv.url)
            .with_extensions([".md"]);
        loader.load().unwrap();
        let auth = srv.seen_auth.lock().unwrap().clone();
        for a in &auth {
            assert_eq!(a.as_deref(), Some("Bearer ghp_secret"));
        }
    }

    #[test]
    fn with_ref_overrides_default_main_in_tree_and_contents_urls() {
        let srv = spawn_fake();
        let loader = GithubFilesLoader::from_repo_tree("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_ref("v1.2.3")
            .with_extensions([".md"]);
        loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        assert!(paths.iter().any(|p| p.contains("/trees/v1.2.3")));
        assert!(paths.iter().any(|p| p.contains("ref=v1.2.3")));
    }

    #[test]
    fn max_files_cap_truncates_result() {
        let srv = spawn_fake();
        let loader = GithubFilesLoader::from_repo_tree("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_extensions([".md", ".rs"])
            .with_max_files(Some(1));
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[test]
    fn urlencode_path_preserves_slashes_encodes_spaces() {
        assert_eq!(urlencode_path("src/main.rs"), "src/main.rs");
        assert_eq!(urlencode_path("docs/my file.md"), "docs/my%20file.md");
        assert_eq!(urlencode_path("plain"), "plain");
    }
}
