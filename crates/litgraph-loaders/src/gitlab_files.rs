//! GitLab files loader. Parallel to `GithubFilesLoader` for code-RAG
//! over self-hosted GitLab repos. Fetches `repository/tree` then
//! `repository/files/{path}` for each file passing filters.
//!
//! # vs GithubFilesLoader
//!
//! - **Auth header**: GitLab uses `PRIVATE-TOKEN` (PAT) by default,
//!   `Authorization: Bearer` for OAuth. Same toggle as `GitLabIssuesLoader`.
//! - **No size in tree entries**: GitLab's tree API doesn't return file
//!   sizes, so we can't pre-filter by size. The size check happens AFTER
//!   the contents fetch (one wasted round-trip per oversized file). For
//!   large repos with many big binaries, prefer the `extensions` allowlist
//!   to skip them at the path level.
//! - **Path encoding**: filepath in URL must be percent-encoded
//!   (slashes → `%2F`). Different from GitHub which keeps slashes raw.
//! - **Pagination**: tree results paginated 100 per page; we fetch up
//!   to 10 pages (1000 entries) then stop with a truncation log. Most
//!   real repos fit; truly massive repos need narrower path filtering.

use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use litgraph_core::Document;
use serde_json::Value;

use crate::{Loader, LoaderError, LoaderResult};

const GITLAB_API: &str = "https://gitlab.com/api/v4";
const MAX_TREE_PAGES: u32 = 10;

pub struct GitLabFilesLoader {
    pub token: String,
    /// Numeric ID (`"12345"`) or URL-encoded path (`"group%2Fmyproject"`).
    pub project: String,
    pub git_ref: String,
    pub base_url: String,
    pub timeout: Duration,
    pub extensions: Vec<String>,
    pub exclude_paths: Vec<String>,
    pub max_files: Option<usize>,
    pub max_file_size_bytes: u64,
    /// `false` → `PRIVATE-TOKEN` header (PAT); `true` → Bearer (OAuth).
    pub oauth: bool,
}

impl GitLabFilesLoader {
    pub fn from_project(token: impl Into<String>, project: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            project: project.into(),
            git_ref: "main".into(),
            base_url: GITLAB_API.into(),
            timeout: Duration::from_secs(30),
            extensions: Vec::new(),
            exclude_paths: default_excludes(),
            max_files: Some(500),
            max_file_size_bytes: 1024 * 1024,
            oauth: false,
        }
    }

    pub fn with_ref(mut self, r: impl Into<String>) -> Self { self.git_ref = r.into(); self }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self { self.base_url = url.into(); self }
    pub fn with_oauth(mut self, b: bool) -> Self { self.oauth = b; self }
    pub fn with_max_files(mut self, n: Option<usize>) -> Self { self.max_files = n; self }
    pub fn with_max_file_size_bytes(mut self, n: u64) -> Self { self.max_file_size_bytes = n; self }

    pub fn with_extensions<I, S>(mut self, exts: I) -> Self
    where I: IntoIterator<Item = S>, S: Into<String> {
        self.extensions = exts.into_iter().map(|e| {
            let s: String = e.into();
            let s = s.to_ascii_lowercase();
            if s.starts_with('.') { s } else { format!(".{s}") }
        }).collect();
        self
    }

    /// REPLACE the exclude list (not append). Use to clear defaults.
    pub fn with_exclude_paths<I, S>(mut self, paths: I) -> Self
    where I: IntoIterator<Item = S>, S: Into<String> {
        self.exclude_paths = paths.into_iter().map(Into::into).collect();
        self
    }

    fn client(&self) -> LoaderResult<reqwest::blocking::Client> {
        reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent("litgraph-gitlab-loader/0.1")
            .build()
            .map_err(LoaderError::from)
    }

    fn authed(&self, b: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
        if self.oauth {
            b.bearer_auth(&self.token)
        } else {
            b.header("PRIVATE-TOKEN", &self.token)
        }
    }

    /// Fetch all blob entries from the recursive tree, paginating until
    /// either page < per_page (last page) or `MAX_TREE_PAGES` cap. Filters
    /// to `type=="blob"` (skip subtrees + commits).
    fn fetch_tree(&self, client: &reqwest::blocking::Client) -> LoaderResult<Vec<Value>> {
        let mut all = Vec::new();
        for page in 1..=MAX_TREE_PAGES {
            let url = format!(
                "{}/projects/{}/repository/tree?ref={}&recursive=true&per_page=100&page={}",
                self.base_url.trim_end_matches('/'),
                self.project,
                self.git_ref,
                page,
            );
            let resp = self.authed(client.get(&url)).send()?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                return Err(LoaderError::Other(format!("gitlab tree {status}: {body}")));
            }
            let entries: Vec<Value> = resp.json()?;
            let n = entries.len();
            for e in entries {
                if e.get("type").and_then(|t| t.as_str()) == Some("blob") {
                    all.push(e);
                }
            }
            if n < 100 {
                break;
            }
        }
        Ok(all)
    }

    fn fetch_file(
        &self,
        client: &reqwest::blocking::Client,
        path: &str,
    ) -> LoaderResult<Option<Value>> {
        // GitLab requires the filepath URL-encoded in the path, slashes too.
        let url = format!(
            "{}/projects/{}/repository/files/{}?ref={}",
            self.base_url.trim_end_matches('/'),
            self.project,
            urlencode_full(path),
            self.git_ref,
        );
        let resp = self.authed(client.get(&url)).send()?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!("gitlab file {path} {status}: {body}")));
        }
        Ok(Some(resp.json()?))
    }

    fn blob_passes_path_filters(&self, entry: &Value) -> bool {
        let path = match entry.get("path").and_then(|p| p.as_str()) {
            Some(p) => p,
            None => return false,
        };
        for ex in &self.exclude_paths {
            if !ex.is_empty() && path.contains(ex) {
                return false;
            }
        }
        if !self.extensions.is_empty() {
            let lower = path.to_ascii_lowercase();
            if !self.extensions.iter().any(|ext| lower.ends_with(ext)) {
                return false;
            }
        }
        true
    }

    fn content_to_document(&self, content_resp: &Value) -> Option<Document> {
        // GitLab's contents response uses `file_path` (not `path`) and
        // `blob_id` (not `sha`). `size` is in bytes; `encoding` is "base64".
        let path = content_resp
            .get("file_path")
            .and_then(|p| p.as_str())?
            .to_string();
        let blob_id = content_resp
            .get("blob_id")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let size = content_resp.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
        // POST-fetch size cap (no size in tree → only checkable now).
        if size > self.max_file_size_bytes {
            return None;
        }
        let encoding = content_resp
            .get("encoding")
            .and_then(|e| e.as_str())
            .unwrap_or("base64");
        let raw = content_resp.get("content").and_then(|c| c.as_str())?;
        let text = if encoding == "base64" {
            let cleaned: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
            match STANDARD.decode(cleaned.as_bytes()) {
                Ok(bytes) => match String::from_utf8(bytes) {
                    Ok(s) => s,
                    Err(_) => return None, // binary — skip
                },
                Err(_) => return None,
            }
        } else {
            raw.to_string()
        };

        let last_commit_id = content_resp
            .get("last_commit_id")
            .and_then(|s| s.as_str())
            .unwrap_or("");

        let mut d = Document::new(text);
        d.id = Some(format!("{}:{}", self.project, path));
        d.metadata.insert("path".into(), Value::String(path.clone()));
        d.metadata.insert("ref".into(), Value::String(self.git_ref.clone()));
        d.metadata.insert("blob_id".into(), Value::String(blob_id.into()));
        d.metadata.insert("size".into(), Value::from(size));
        if !last_commit_id.is_empty() {
            d.metadata
                .insert("last_commit_id".into(), Value::String(last_commit_id.into()));
        }
        d.metadata.insert(
            "source".into(),
            Value::String(format!("gitlab:{}/{}@{}", self.project, path, self.git_ref)),
        );
        Some(d)
    }
}

impl Loader for GitLabFilesLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.client()?;
        let tree = self.fetch_tree(&client)?;
        let mut docs: Vec<Document> = Vec::new();
        for entry in &tree {
            if !self.blob_passes_path_filters(entry) {
                continue;
            }
            let path = entry
                .get("path")
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .to_string();
            let content = match self.fetch_file(&client, &path)? {
                Some(c) => c,
                None => continue,
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
        ".lock".into(),
        ".min.js".into(),
        ".min.css".into(),
    ]
}

/// Percent-encode the FULL string (including slashes — GitLab requires it
/// for the file_path URL segment).
fn urlencode_full(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~') {
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
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    struct FakeGitLab {
        port: u16,
        captured: Arc<Mutex<Vec<String>>>,
    }

    fn start_fake(responses: Vec<(u16, String)>) -> FakeGitLab {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = captured.clone();
        thread::spawn(move || {
            let mut idx = 0;
            for stream in listener.incoming() {
                let mut s = match stream { Ok(s) => s, Err(_) => break };
                let mut buf = [0u8; 8192];
                let mut total = Vec::new();
                loop {
                    let n = match s.read(&mut buf) { Ok(0) => break, Ok(n) => n, Err(_) => break };
                    total.extend_from_slice(&buf[..n]);
                    if total.windows(4).any(|w| w == b"\r\n\r\n") { break; }
                }
                captured_clone.lock().unwrap().push(String::from_utf8_lossy(&total).to_string());
                let (status, body) = responses.get(idx).cloned().unwrap_or((200, "[]".to_string()));
                idx += 1;
                let header = format!(
                    "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    status, body.len()
                );
                let _ = s.write_all(header.as_bytes());
                let _ = s.write_all(body.as_bytes());
                if idx >= responses.len() { break; }
            }
        });
        FakeGitLab { port, captured }
    }

    fn tree_entry(path: &str) -> Value {
        serde_json::json!({
            "id": "abc",
            "name": path.split('/').last().unwrap_or(path),
            "type": "blob",
            "path": path,
            "mode": "100644",
        })
    }

    fn file_response(path: &str, body: &str, size: u64) -> String {
        let encoded = STANDARD.encode(body.as_bytes());
        serde_json::json!({
            "file_name": path.split('/').last().unwrap_or(path),
            "file_path": path,
            "size": size,
            "encoding": "base64",
            "content": encoded,
            "blob_id": format!("blob-{path}"),
            "last_commit_id": "commit-abc",
            "ref": "main",
        }).to_string()
    }

    #[test]
    fn loads_files_from_tree() {
        let tree = serde_json::json!([
            tree_entry("README.md"),
            tree_entry("src/main.rs"),
        ]).to_string();
        let r1 = file_response("README.md", "# Hello", 100);
        let r2 = file_response("src/main.rs", "fn main() {}", 200);
        let fake = start_fake(vec![
            (200, tree),
            (200, r1),
            (200, r2),
        ]);
        let docs = GitLabFilesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs.iter().any(|d| d.content == "# Hello"));
        assert!(docs.iter().any(|d| d.content == "fn main() {}"));
    }

    #[test]
    fn private_token_header_used_by_default() {
        let tree = "[]".to_string();
        let fake = start_fake(vec![(200, tree)]);
        let _ = GitLabFilesLoader::from_project("my-pat", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        let req = &fake.captured.lock().unwrap()[0];
        let lower = req.to_lowercase();
        assert!(lower.contains("private-token: my-pat"));
        assert!(!lower.contains("authorization: bearer"));
    }

    #[test]
    fn oauth_mode_uses_bearer_auth() {
        let fake = start_fake(vec![(200, "[]".to_string())]);
        let _ = GitLabFilesLoader::from_project("oauth-tok", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .with_oauth(true)
            .load()
            .unwrap();
        let req = &fake.captured.lock().unwrap()[0];
        let lower = req.to_lowercase();
        assert!(lower.contains("authorization: bearer oauth-tok"));
        assert!(!lower.contains("private-token"));
    }

    #[test]
    fn extension_filter_applied_to_tree_entries() {
        let tree = serde_json::json!([
            tree_entry("README.md"),
            tree_entry("src/main.rs"),
            tree_entry("src/lib.py"),
        ]).to_string();
        let r1 = file_response("README.md", "rd", 10);
        let fake = start_fake(vec![
            (200, tree),
            (200, r1),
        ]);
        let docs = GitLabFilesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .with_extensions([".md"])
            .load()
            .unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].metadata["path"], "README.md");
    }

    #[test]
    fn default_excludes_skip_node_modules_and_lockfiles() {
        let tree = serde_json::json!([
            tree_entry("src/main.rs"),
            tree_entry("node_modules/foo/index.js"),
            tree_entry("Cargo.lock"),
            tree_entry("README.md"),
        ]).to_string();
        let r1 = file_response("src/main.rs", "main", 4);
        let r2 = file_response("README.md", "rd", 2);
        let fake = start_fake(vec![
            (200, tree),
            (200, r1),
            (200, r2),
        ]);
        let docs = GitLabFilesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        // node_modules + Cargo.lock excluded by default.
        assert_eq!(docs.len(), 2);
        let paths: Vec<&str> = docs.iter().map(|d| d.metadata["path"].as_str().unwrap()).collect();
        assert!(paths.contains(&"src/main.rs"));
        assert!(paths.contains(&"README.md"));
    }

    #[test]
    fn max_files_caps_total_loaded() {
        let tree = serde_json::json!([
            tree_entry("a.md"),
            tree_entry("b.md"),
            tree_entry("c.md"),
        ]).to_string();
        let fake = start_fake(vec![
            (200, tree),
            (200, file_response("a.md", "A", 1)),
            (200, file_response("b.md", "B", 1)),
            (200, file_response("c.md", "C", 1)),
        ]);
        let docs = GitLabFilesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .with_max_files(Some(2))
            .load()
            .unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn max_file_size_skips_oversized_after_fetch() {
        let tree = serde_json::json!([
            tree_entry("small.txt"),
            tree_entry("big.txt"),
        ]).to_string();
        let r1 = file_response("small.txt", "x", 10);
        let r2 = file_response("big.txt", "y".repeat(2000).as_str(), 2_000_000); // 2MB
        let fake = start_fake(vec![
            (200, tree),
            (200, r1),
            (200, r2),
        ]);
        let docs = GitLabFilesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .with_max_file_size_bytes(1024 * 1024) // 1 MiB
            .load()
            .unwrap();
        // Only `small.txt` survives; big.txt fetched then dropped.
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].metadata["path"], "small.txt");
    }

    #[test]
    fn binary_file_skipped_silently() {
        let tree = serde_json::json!([tree_entry("photo.jpg")]).to_string();
        // Base64-encoded invalid UTF-8 bytes.
        let bin = STANDARD.encode([0xff, 0xfe, 0xfd]);
        let resp = serde_json::json!({
            "file_path": "photo.jpg",
            "size": 3,
            "encoding": "base64",
            "content": bin,
            "blob_id": "bin-blob",
            "ref": "main",
        }).to_string();
        let fake = start_fake(vec![
            (200, tree),
            (200, resp),
        ]);
        let docs = GitLabFilesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        assert!(docs.is_empty());
    }

    #[test]
    fn http_error_on_tree_surfaces() {
        let fake = start_fake(vec![(401, r#"{"message":"unauthorized"}"#.to_string())]);
        let err = GitLabFilesLoader::from_project("bad", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap_err();
        assert!(err.to_string().contains("401"));
    }

    #[test]
    fn document_id_includes_project_and_path() {
        let tree = serde_json::json!([tree_entry("docs/intro.md")]).to_string();
        let r1 = file_response("docs/intro.md", "intro", 5);
        let fake = start_fake(vec![(200, tree), (200, r1)]);
        let docs = GitLabFilesLoader::from_project("tk", "group%2Fmyproj")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        assert_eq!(docs[0].id.as_deref(), Some("group%2Fmyproj:docs/intro.md"));
    }

    #[test]
    fn url_encodes_filepath_with_slashes() {
        // The fetch_file URL must percent-encode slashes to %2F.
        let tree = serde_json::json!([tree_entry("src/sub/main.rs")]).to_string();
        let r1 = file_response("src/sub/main.rs", "code", 4);
        let fake = start_fake(vec![(200, tree), (200, r1)]);
        let _ = GitLabFilesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        // Second request (file fetch) must contain encoded path.
        let req2 = &fake.captured.lock().unwrap()[1];
        assert!(req2.contains("src%2Fsub%2Fmain.rs"), "request: {req2}");
    }

    #[test]
    fn metadata_carries_path_blob_id_size_commit() {
        let tree = serde_json::json!([tree_entry("file.txt")]).to_string();
        let r1 = file_response("file.txt", "hello", 5);
        let fake = start_fake(vec![(200, tree), (200, r1)]);
        let docs = GitLabFilesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        let md = &docs[0].metadata;
        assert_eq!(md["path"], "file.txt");
        assert_eq!(md["ref"], "main");
        assert_eq!(md["blob_id"], "blob-file.txt");
        assert_eq!(md["size"], 5);
        assert_eq!(md["last_commit_id"], "commit-abc");
        assert_eq!(md["source"], "gitlab:12345/file.txt@main");
    }
}
