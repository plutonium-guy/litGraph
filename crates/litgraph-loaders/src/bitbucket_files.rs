//! `BitbucketFilesLoader` — recursive source-file fetcher for
//! Bitbucket Cloud repositories. Pairs with iter-264
//! `BitbucketIssuesLoader` to round out the trio's files variant
//! (after GitHub iter 42 and GitLab iter 154).
//!
//! # API surface
//!
//! Bitbucket Cloud exposes repo source under `/2.0/repositories/
//! {workspace}/{repo}/src/{ref}/{path}`. Behavior depends on what
//! `path` points to:
//!
//! - **Directory** → JSON listing `{values: [entry, ...], next}`
//!   where each entry has `type: "commit_directory" | "commit_file"`,
//!   `path`, `size`, and `links.self.href`.
//! - **File** → raw bytes/text in the response body.
//!
//! Walking strategy: BFS from `path` (default repo root); collect
//! `commit_file` entries (skipping symlinks and commits); fetch
//! each file's raw text individually.
//!
//! # Auth
//!
//! Identical to [`crate::BitbucketIssuesLoader`]: app-password
//! Basic auth (default) or OAuth bearer (`with_oauth(true)`).
//!
//! # Filter knobs
//!
//! - `with_extensions([".md", ".rs"])` — allowlist by suffix.
//! - `with_exclude_paths([...])` — drop any file whose path
//!   contains a substring (defaults skip lockfiles, build
//!   artifacts, etc).
//! - `with_max_files(Some(n))` — cap on total file count.
//! - `with_max_file_size_bytes(n)` — skip files larger than this.
//! - `with_max_depth(d)` — directory recursion depth limit.

use std::time::Duration;

use litgraph_core::Document;
use serde_json::{json, Value};

use crate::{Loader, LoaderError, LoaderResult};

const BITBUCKET_API: &str = "https://api.bitbucket.org/2.0";

pub struct BitbucketFilesLoader {
    pub username: String,
    pub token: String,
    pub workspace: String,
    pub repo: String,
    pub git_ref: String,
    pub root_path: String,
    pub base_url: String,
    pub timeout: Duration,
    pub extensions: Vec<String>,
    pub exclude_paths: Vec<String>,
    pub max_files: Option<usize>,
    pub max_file_size_bytes: u64,
    pub max_depth: usize,
    pub oauth: bool,
}

impl BitbucketFilesLoader {
    pub fn new(
        username: impl Into<String>,
        token: impl Into<String>,
        workspace: impl Into<String>,
        repo: impl Into<String>,
    ) -> Self {
        Self {
            username: username.into(),
            token: token.into(),
            workspace: workspace.into(),
            repo: repo.into(),
            git_ref: "main".into(),
            root_path: String::new(),
            base_url: BITBUCKET_API.into(),
            timeout: Duration::from_secs(30),
            extensions: Vec::new(),
            exclude_paths: default_excludes(),
            max_files: Some(500),
            max_file_size_bytes: 1024 * 1024,
            max_depth: 32,
            oauth: false,
        }
    }

    pub fn with_ref(mut self, r: impl Into<String>) -> Self {
        self.git_ref = r.into();
        self
    }
    pub fn with_root_path(mut self, p: impl Into<String>) -> Self {
        self.root_path = p.into();
        self
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_oauth(mut self, b: bool) -> Self {
        self.oauth = b;
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
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

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
                if s.starts_with('.') {
                    s
                } else {
                    format!(".{s}")
                }
            })
            .collect();
        self
    }

    pub fn with_exclude_paths<I, S>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.exclude_paths = paths.into_iter().map(Into::into).collect();
        self
    }

    fn client(&self) -> LoaderResult<reqwest::blocking::Client> {
        reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent("litgraph-bitbucket-loader/0.1")
            .build()
            .map_err(LoaderError::from)
    }

    fn authed(
        &self,
        b: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        if self.oauth {
            b.bearer_auth(&self.token)
        } else {
            b.basic_auth(&self.username, Some(&self.token))
        }
    }

    fn dir_url(&self, path: &str) -> String {
        let trimmed = path.trim_start_matches('/');
        if trimmed.is_empty() {
            format!(
                "{}/repositories/{}/{}/src/{}/?pagelen=100",
                self.base_url.trim_end_matches('/'),
                self.workspace,
                self.repo,
                self.git_ref,
            )
        } else {
            format!(
                "{}/repositories/{}/{}/src/{}/{trimmed}?pagelen=100",
                self.base_url.trim_end_matches('/'),
                self.workspace,
                self.repo,
                self.git_ref,
            )
        }
    }

    fn file_url(&self, path: &str) -> String {
        let trimmed = path.trim_start_matches('/');
        format!(
            "{}/repositories/{}/{}/src/{}/{trimmed}",
            self.base_url.trim_end_matches('/'),
            self.workspace,
            self.repo,
            self.git_ref,
        )
    }

    /// Decide whether a file path passes the extension + exclude
    /// filters. Public for offline tests.
    pub fn passes_filters(&self, path: &str) -> bool {
        for ex in &self.exclude_paths {
            if path.contains(ex.as_str()) {
                return false;
            }
        }
        if self.extensions.is_empty() {
            return true;
        }
        let lower = path.to_ascii_lowercase();
        self.extensions.iter().any(|e| lower.ends_with(e))
    }

    /// Build a [`Document`] from a path + raw content. Public so
    /// unit tests can drive the conversion without HTTP.
    pub fn file_to_document(&self, path: &str, content: String) -> Document {
        let mut doc = Document::new(content).with_id(format!(
            "bitbucket:{}/{}@{}#{path}",
            self.workspace, self.repo, self.git_ref,
        ));
        doc.metadata
            .insert("workspace".into(), json!(self.workspace));
        doc.metadata.insert("repo".into(), json!(self.repo));
        doc.metadata.insert("ref".into(), json!(self.git_ref));
        doc.metadata.insert("path".into(), json!(path));
        doc
    }
}

impl Loader for BitbucketFilesLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.client()?;
        let cap = self.max_files.unwrap_or(usize::MAX);
        let mut docs: Vec<Document> = Vec::new();

        // BFS queue of (path, depth).
        let mut queue: Vec<(String, usize)> = vec![(self.root_path.clone(), 0)];

        while let Some((path, depth)) = queue.pop() {
            if docs.len() >= cap {
                break;
            }
            if depth > self.max_depth {
                continue;
            }
            // List this directory; paginate.
            let mut url = self.dir_url(&path);
            loop {
                let resp = self.authed(client.get(&url)).send()?;
                let status = resp.status();
                if !status.is_success() {
                    let body = resp.text().unwrap_or_default();
                    return Err(LoaderError::Other(format!(
                        "bitbucket dir {status}: {body}",
                    )));
                }
                let v: Value = resp.json()?;
                let entries = v
                    .get("values")
                    .and_then(|x| x.as_array())
                    .cloned()
                    .unwrap_or_default();
                for entry in &entries {
                    if docs.len() >= cap {
                        break;
                    }
                    let kind = entry.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    let p = entry
                        .get("path")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string();
                    if p.is_empty() {
                        continue;
                    }
                    match kind {
                        "commit_directory" => {
                            queue.push((p, depth + 1));
                        }
                        "commit_file" => {
                            if !self.passes_filters(&p) {
                                continue;
                            }
                            let size = entry
                                .get("size")
                                .and_then(|s| s.as_u64())
                                .unwrap_or(0);
                            if size > self.max_file_size_bytes {
                                continue;
                            }
                            // Fetch raw content.
                            let body = self
                                .authed(client.get(self.file_url(&p)))
                                .send()?;
                            if !body.status().is_success() {
                                continue;
                            }
                            let text = body.text().unwrap_or_default();
                            docs.push(self.file_to_document(&p, text));
                        }
                        _ => {} // skip commits / symlinks / etc
                    }
                }
                match v.get("next").and_then(|x| x.as_str()) {
                    Some(next) => url = next.to_string(),
                    None => break,
                }
            }
        }
        Ok(docs)
    }
}

fn default_excludes() -> Vec<String> {
    [
        "node_modules/",
        "target/",
        "dist/",
        "build/",
        ".git/",
        "Cargo.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "uv.lock",
        "poetry.lock",
        ".min.js",
        ".min.css",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_filters_extension_allowlist() {
        let l = BitbucketFilesLoader::new("u", "p", "ws", "repo")
            .with_extensions([".md", "rs"]);
        assert!(l.passes_filters("README.md"));
        assert!(l.passes_filters("src/lib.rs"));
        assert!(!l.passes_filters("src/lib.py"));
    }

    #[test]
    fn passes_filters_no_extensions_means_allow_all() {
        let l = BitbucketFilesLoader::new("u", "p", "ws", "repo")
            .with_extensions::<Vec<&str>, &str>(Vec::new());
        assert!(l.passes_filters("anything.txt"));
        assert!(l.passes_filters("path/to/file.bin"));
    }

    #[test]
    fn passes_filters_excludes_default_lockfiles() {
        let l = BitbucketFilesLoader::new("u", "p", "ws", "repo");
        assert!(!l.passes_filters("Cargo.lock"));
        assert!(!l.passes_filters("frontend/package-lock.json"));
        assert!(!l.passes_filters("node_modules/lib/index.js"));
        assert!(l.passes_filters("src/main.rs"));
    }

    #[test]
    fn dir_url_root_includes_pagelen() {
        let l = BitbucketFilesLoader::new("u", "p", "ws", "repo");
        let url = l.dir_url("");
        assert!(url.ends_with("/repositories/ws/repo/src/main/?pagelen=100"));
    }

    #[test]
    fn dir_url_subpath() {
        let l = BitbucketFilesLoader::new("u", "p", "ws", "repo");
        let url = l.dir_url("docs/");
        assert!(url.contains("/src/main/docs/?pagelen=100"));
    }

    #[test]
    fn file_url_constructs_raw_path() {
        let l = BitbucketFilesLoader::new("u", "p", "ws", "repo")
            .with_ref("dev");
        let url = l.file_url("src/lib.rs");
        assert!(url.ends_with("/repositories/ws/repo/src/dev/src/lib.rs"));
    }

    #[test]
    fn file_to_document_carries_metadata() {
        let l = BitbucketFilesLoader::new("u", "p", "ws", "repo");
        let doc = l.file_to_document("src/main.rs", "fn main(){}".into());
        assert_eq!(doc.content, "fn main(){}");
        assert_eq!(
            doc.id.as_deref(),
            Some("bitbucket:ws/repo@main#src/main.rs"),
        );
        assert_eq!(
            doc.metadata.get("workspace").and_then(|v| v.as_str()),
            Some("ws"),
        );
        assert_eq!(
            doc.metadata.get("ref").and_then(|v| v.as_str()),
            Some("main"),
        );
        assert_eq!(
            doc.metadata.get("path").and_then(|v| v.as_str()),
            Some("src/main.rs"),
        );
    }

    #[test]
    fn with_exclude_paths_replaces_defaults() {
        let l = BitbucketFilesLoader::new("u", "p", "ws", "repo")
            .with_exclude_paths(["secrets/"]);
        // Cargo.lock no longer excluded after replace.
        assert!(l.passes_filters("Cargo.lock"));
        assert!(!l.passes_filters("secrets/api_key.txt"));
    }

    #[test]
    fn extension_normalization_dots_and_case() {
        let l = BitbucketFilesLoader::new("u", "p", "ws", "repo")
            .with_extensions(["MD", ".RS"]);
        assert!(l.passes_filters("readme.md"));
        assert!(l.passes_filters("src/Lib.rs"));
        assert!(!l.passes_filters("src/main.go"));
    }
}
