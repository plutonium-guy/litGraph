//! GitHub loader — pull issues (and optionally comments) from a repo via
//! the GitHub REST API. Direct LangChain `GitHubIssuesLoader` parity.
//!
//! # Auth
//!
//! Personal Access Token (classic or fine-grained) OR GitHub App
//! installation token. Passed as `Authorization: Bearer <token>`. All
//! requests pin `X-GitHub-Api-Version: 2022-11-28` — matches GitHub's
//! stable API version policy (breaking changes get a new version string).
//!
//! # Scope
//!
//! - `from_repo_issues(token, owner, repo)` — list ALL issues (open + closed)
//!   in a repo, newest first, paginated (`per_page=100`, follows `Link: next`).
//! - `.with_state("open" | "closed" | "all")` — default `all`.
//! - `.with_include_comments(bool)` — if true, each issue's comments get
//!   fetched via `GET /repos/{owner}/{repo}/issues/{number}/comments`
//!   and inlined below the issue body.
//! - `.with_labels(&[...])` — filter by label(s) (AND).
//! - `.with_max_issues(cap)` — default 1000.
//!
//! # Pull requests
//!
//! GitHub's `/issues` endpoint returns BOTH issues AND PRs; we leave them in
//! (PRs are a superset of issues — a support bot usually wants both in its
//! RAG index). Callers who only want issues can filter downstream on
//! `metadata["is_pull_request"]` which we set accurately.
//!
//! # Metadata per document
//!
//! - `number` — issue / PR number
//! - `title`
//! - `state` — "open" / "closed"
//! - `user` — author login
//! - `labels` — comma-joined label names
//! - `created_at` / `updated_at` — ISO-8601 timestamps
//! - `is_pull_request` — true if this is actually a PR
//! - `html_url` — the web URL (useful for citations in LLM answers)
//! - `source` — `"github:{owner}/{repo}#{number}"`

use std::time::Duration;

use litgraph_core::Document;
use serde_json::Value;

use crate::{Loader, LoaderError, LoaderResult};

const GITHUB_API: &str = "https://api.github.com";
const GITHUB_API_VERSION: &str = "2022-11-28";

pub struct GithubIssuesLoader {
    pub token: String,
    pub owner: String,
    pub repo: String,
    pub base_url: String,
    pub timeout: Duration,
    pub state: String,
    pub include_comments: bool,
    pub labels: Vec<String>,
    pub max_issues: Option<usize>,
}

impl GithubIssuesLoader {
    pub fn from_repo_issues(
        token: impl Into<String>,
        owner: impl Into<String>,
        repo: impl Into<String>,
    ) -> Self {
        Self {
            token: token.into(),
            owner: owner.into(),
            repo: repo.into(),
            base_url: GITHUB_API.into(),
            timeout: Duration::from_secs(30),
            state: "all".into(),
            include_comments: false,
            labels: Vec::new(),
            max_issues: Some(1000),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_state(mut self, state: impl Into<String>) -> Self {
        self.state = state.into();
        self
    }

    pub fn with_include_comments(mut self, b: bool) -> Self {
        self.include_comments = b;
        self
    }

    pub fn with_labels<I, S>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.labels = labels.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_max_issues(mut self, n: Option<usize>) -> Self {
        self.max_issues = n;
        self
    }

    fn client(&self) -> LoaderResult<reqwest::blocking::Client> {
        reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent("litgraph-github-loader/0.1")
            .build()
            .map_err(LoaderError::from)
    }

    fn authed(&self, b: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
        b.bearer_auth(&self.token)
            .header("X-GitHub-Api-Version", GITHUB_API_VERSION)
            .header("Accept", "application/vnd.github+json")
    }

    fn fetch_issues_page(
        &self,
        client: &reqwest::blocking::Client,
        page: u32,
    ) -> LoaderResult<Vec<Value>> {
        let mut url = format!(
            "{}/repos/{}/{}/issues?state={}&per_page=100&page={}",
            self.base_url.trim_end_matches('/'),
            self.owner,
            self.repo,
            self.state,
            page,
        );
        if !self.labels.is_empty() {
            url.push_str("&labels=");
            url.push_str(&self.labels.join(","));
        }
        let resp = self.authed(client.get(&url)).send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "github issues {status}: {body}"
            )));
        }
        let arr: Vec<Value> = resp.json()?;
        Ok(arr)
    }

    fn fetch_comments(
        &self,
        client: &reqwest::blocking::Client,
        issue_number: i64,
    ) -> LoaderResult<Vec<Value>> {
        // Comments endpoint. Paginated like issues; for simplicity + typical
        // volumes we fetch up to 100 per issue (most issues are below that;
        // very noisy threads can use a future multi-page enhancement).
        let url = format!(
            "{}/repos/{}/{}/issues/{}/comments?per_page=100",
            self.base_url.trim_end_matches('/'),
            self.owner,
            self.repo,
            issue_number,
        );
        let resp = self.authed(client.get(&url)).send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "github comments {status}: {body}"
            )));
        }
        resp.json().map_err(LoaderError::from)
    }

    fn issue_to_document(
        &self,
        issue: &Value,
        comments: Option<&[Value]>,
    ) -> Option<Document> {
        let number = issue.get("number").and_then(|n| n.as_i64())?;
        let title = issue.get("title").and_then(|t| t.as_str()).unwrap_or("");
        // GitHub returns body=null for issues without a description.
        let body = issue
            .get("body")
            .and_then(|b| b.as_str())
            .unwrap_or("");
        let author = issue
            .pointer("/user/login")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let state = issue.get("state").and_then(|s| s.as_str()).unwrap_or("");
        let is_pr = issue.get("pull_request").is_some();

        // Content: title + body + (optional) comments inlined under
        // `--- comments ---` separator. LLMs handle the structure better
        // with an explicit delimiter than with plain-joined text.
        let mut content = String::new();
        if !title.is_empty() {
            content.push_str("# ");
            content.push_str(title);
            content.push('\n');
            content.push('\n');
        }
        if !body.is_empty() {
            content.push_str(body);
        }
        if let Some(cs) = comments {
            if !cs.is_empty() {
                content.push_str("\n\n--- comments ---\n");
                for c in cs {
                    let cuser = c
                        .pointer("/user/login")
                        .and_then(|s| s.as_str())
                        .unwrap_or("");
                    let cbody = c.get("body").and_then(|b| b.as_str()).unwrap_or("");
                    if !cbody.is_empty() {
                        content.push_str(&format!("\n[@{cuser}]: {cbody}\n"));
                    }
                }
            }
        }

        let mut d = Document::new(content);
        d.id = Some(format!("{}/{}#{}", self.owner, self.repo, number));
        d.metadata.insert("number".into(), Value::from(number));
        d.metadata
            .insert("title".into(), Value::String(title.to_string()));
        d.metadata
            .insert("state".into(), Value::String(state.to_string()));
        d.metadata
            .insert("user".into(), Value::String(author.to_string()));
        d.metadata
            .insert("is_pull_request".into(), Value::Bool(is_pr));
        if let Some(labels) = issue.get("labels").and_then(|l| l.as_array()) {
            let names: Vec<String> = labels
                .iter()
                .filter_map(|l| l.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect();
            d.metadata
                .insert("labels".into(), Value::String(names.join(",")));
        }
        if let Some(ts) = issue.get("created_at").and_then(|t| t.as_str()) {
            d.metadata.insert("created_at".into(), Value::String(ts.into()));
        }
        if let Some(ts) = issue.get("updated_at").and_then(|t| t.as_str()) {
            d.metadata.insert("updated_at".into(), Value::String(ts.into()));
        }
        if let Some(url) = issue.get("html_url").and_then(|u| u.as_str()) {
            d.metadata.insert("html_url".into(), Value::String(url.into()));
        }
        d.metadata.insert(
            "source".into(),
            Value::String(format!("github:{}/{}#{}", self.owner, self.repo, number)),
        );
        Some(d)
    }
}

impl Loader for GithubIssuesLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.client()?;
        let mut docs: Vec<Document> = Vec::new();
        let mut page: u32 = 1;
        loop {
            let issues = self.fetch_issues_page(&client, page)?;
            if issues.is_empty() {
                break;
            }
            for issue in &issues {
                let comments = if self.include_comments {
                    let n = issue
                        .get("number")
                        .and_then(|n| n.as_i64())
                        .unwrap_or(0);
                    // Avoid the round-trip when the issue has zero comments.
                    let count = issue
                        .get("comments")
                        .and_then(|c| c.as_i64())
                        .unwrap_or(0);
                    if count > 0 {
                        Some(self.fetch_comments(&client, n)?)
                    } else {
                        Some(Vec::new())
                    }
                } else {
                    None
                };
                if let Some(d) = self.issue_to_document(issue, comments.as_deref()) {
                    docs.push(d);
                    if let Some(cap) = self.max_issues {
                        if docs.len() >= cap {
                            return Ok(docs);
                        }
                    }
                }
            }
            // GitHub pagination: stop when a page returns fewer than per_page
            // items (simpler than parsing the Link header).
            if issues.len() < 100 {
                break;
            }
            page += 1;
        }
        Ok(docs)
    }
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
        seen_version: Arc<Mutex<Vec<Option<String>>>>,
        _shutdown: std::sync::mpsc::Sender<()>,
    }

    fn page_1() -> Value {
        json!([
            {
                "number": 1, "title": "Bug: X crashes", "state": "open",
                "body": "Steps to repro:\n1. Click",
                "user": {"login": "alice"},
                "labels": [{"name": "bug"}, {"name": "p1"}],
                "created_at": "2025-01-01T12:00:00Z",
                "updated_at": "2025-01-02T15:00:00Z",
                "html_url": "https://github.com/acme/app/issues/1",
                "comments": 2
            },
            {
                "number": 2, "title": "Feature: Y", "state": "closed",
                "body": "",
                "user": {"login": "bob"},
                "labels": [{"name": "feature"}],
                "created_at": "2025-01-03T00:00:00Z",
                "updated_at": "2025-01-04T00:00:00Z",
                "html_url": "https://github.com/acme/app/issues/2",
                "comments": 0,
                // GitHub marks PRs with a `pull_request` sub-object.
                "pull_request": {"url": "..."}
            }
        ])
    }

    fn page_2_empty() -> Value {
        // Second page returns < 100 items → loader stops paginating.
        json!([])
    }

    fn comments_for_issue_1() -> Value {
        json!([
            {"body": "confirmed, same here", "user": {"login": "carol"}},
            {"body": "will look this week", "user": {"login": "dave"}}
        ])
    }

    fn spawn_fake() -> FakeServer {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen_paths: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_auth: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_version: Arc<Mutex<Vec<Option<String>>>> =
            Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        listener.set_nonblocking(true).unwrap();
        let paths_w = seen_paths.clone();
        let auth_w = seen_auth.clone();
        let ver_w = seen_version.clone();
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
                        let mut version = None;
                        for line in req.lines().skip(1) {
                            if let Some((k, v)) = line.split_once(':') {
                                let k_lc = k.trim().to_ascii_lowercase();
                                if k_lc == "authorization" {
                                    auth = Some(v.trim().to_string());
                                } else if k_lc == "x-github-api-version" {
                                    version = Some(v.trim().to_string());
                                }
                            }
                        }
                        auth_w.lock().unwrap().push(auth);
                        ver_w.lock().unwrap().push(version);

                        let body = if path.contains("/issues/1/comments") {
                            comments_for_issue_1().to_string()
                        } else if path.contains("/issues?") || path.contains("/issues&") {
                            if path.contains("page=1") || !path.contains("page=") {
                                page_1().to_string()
                            } else {
                                page_2_empty().to_string()
                            }
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
            seen_version,
            _shutdown: tx,
        }
    }

    #[test]
    fn loads_issues_returns_one_doc_per_issue_with_title_and_body() {
        let srv = spawn_fake();
        let loader = GithubIssuesLoader::from_repo_issues("ghp_test", "acme", "app")
            .with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
        // Title formatted as markdown H1 header in content.
        assert!(docs[0].content.starts_with("# Bug: X crashes"));
        assert!(docs[0].content.contains("Steps to repro"));
    }

    #[test]
    fn metadata_captures_issue_attributes_and_source_url() {
        let srv = spawn_fake();
        let loader = GithubIssuesLoader::from_repo_issues("t", "acme", "app")
            .with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        let d0 = &docs[0];
        assert_eq!(d0.metadata["number"].as_i64(), Some(1));
        assert_eq!(d0.metadata["title"].as_str(), Some("Bug: X crashes"));
        assert_eq!(d0.metadata["state"].as_str(), Some("open"));
        assert_eq!(d0.metadata["user"].as_str(), Some("alice"));
        assert_eq!(d0.metadata["labels"].as_str(), Some("bug,p1"));
        assert_eq!(d0.metadata["is_pull_request"].as_bool(), Some(false));
        assert_eq!(
            d0.metadata["html_url"].as_str(),
            Some("https://github.com/acme/app/issues/1")
        );
        assert_eq!(d0.metadata["source"].as_str(), Some("github:acme/app#1"));
        assert_eq!(d0.id.as_deref(), Some("acme/app#1"));
    }

    #[test]
    fn pull_request_flag_set_when_issue_has_pull_request_field() {
        let srv = spawn_fake();
        let loader = GithubIssuesLoader::from_repo_issues("t", "acme", "app")
            .with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        // Issue #2 in the fake data has a `pull_request` sub-object.
        assert_eq!(docs[1].metadata["is_pull_request"].as_bool(), Some(true));
        assert_eq!(docs[0].metadata["is_pull_request"].as_bool(), Some(false));
    }

    #[test]
    fn auth_bearer_and_api_version_headers_set_on_every_request() {
        let srv = spawn_fake();
        let loader = GithubIssuesLoader::from_repo_issues("ghp_secret", "acme", "app")
            .with_base_url(&srv.url);
        loader.load().unwrap();
        let auth = srv.seen_auth.lock().unwrap().clone();
        let ver = srv.seen_version.lock().unwrap().clone();
        for a in &auth {
            assert_eq!(a.as_deref(), Some("Bearer ghp_secret"));
        }
        for v in &ver {
            assert_eq!(v.as_deref(), Some("2022-11-28"));
        }
    }

    #[test]
    fn include_comments_inlines_comment_thread_under_separator() {
        let srv = spawn_fake();
        let loader = GithubIssuesLoader::from_repo_issues("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_include_comments(true);
        let docs = loader.load().unwrap();
        // Issue #1 has 2 comments (from `comments_for_issue_1`).
        let d0 = &docs[0];
        assert!(d0.content.contains("--- comments ---"));
        assert!(d0.content.contains("[@carol]: confirmed, same here"));
        assert!(d0.content.contains("[@dave]: will look this week"));
        // Issue #2 has 0 comments → no comments section (but the separator
        // also shouldn't appear).
        assert!(!docs[1].content.contains("--- comments ---"));
    }

    #[test]
    fn include_comments_skips_http_call_when_comments_count_is_zero() {
        // The loader should NOT issue a comments fetch for an issue with
        // `comments: 0` — otherwise a busy repo with 1000 comment-less PRs
        // burns 1000 pointless round-trips.
        let srv = spawn_fake();
        let loader = GithubIssuesLoader::from_repo_issues("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_include_comments(true);
        loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        // Exactly 1 comments call (for issue #1), not 2.
        let comment_hits: Vec<_> = paths
            .iter()
            .filter(|p| p.contains("/comments"))
            .collect();
        assert_eq!(comment_hits.len(), 1);
        assert!(comment_hits[0].contains("/issues/1/comments"));
    }

    #[test]
    fn state_filter_appears_in_query_string() {
        let srv = spawn_fake();
        let loader = GithubIssuesLoader::from_repo_issues("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_state("open");
        loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        let first = paths.iter().find(|p| p.starts_with("/repos/")).unwrap();
        assert!(first.contains("state=open"), "got: {first}");
    }

    #[test]
    fn labels_filter_appears_in_query_string() {
        let srv = spawn_fake();
        let loader = GithubIssuesLoader::from_repo_issues("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_labels(["bug", "p1"]);
        loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        let first = paths.iter().find(|p| p.starts_with("/repos/")).unwrap();
        assert!(first.contains("labels=bug,p1"), "got: {first}");
    }

    #[test]
    fn max_issues_cap_truncates_result() {
        let srv = spawn_fake();
        let loader = GithubIssuesLoader::from_repo_issues("t", "acme", "app")
            .with_base_url(&srv.url)
            .with_max_issues(Some(1));
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[test]
    fn pagination_stops_when_page_returns_fewer_than_per_page() {
        // Page 1 has 2 items (< 100), so loader should NOT request page 2.
        // (Real GitHub would also use Link headers but we simplify.)
        let srv = spawn_fake();
        let loader = GithubIssuesLoader::from_repo_issues("t", "acme", "app")
            .with_base_url(&srv.url);
        loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        let issue_paths: Vec<_> = paths
            .iter()
            .filter(|p| p.starts_with("/repos/acme/app/issues?"))
            .collect();
        // Exactly one issue-list request (no follow-up page).
        assert_eq!(issue_paths.len(), 1);
    }

    #[test]
    fn body_null_does_not_break_document_construction() {
        // GitHub returns body=null for title-only issues. Ensure we don't
        // panic or produce a "null"-string document.
        let loader = GithubIssuesLoader::from_repo_issues("t", "acme", "app");
        let issue = json!({
            "number": 99,
            "title": "Title only",
            "state": "open",
            "body": null,
            "user": {"login": "u"},
            "labels": [],
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "html_url": "https://github.com/acme/app/issues/99"
        });
        let d = loader.issue_to_document(&issue, None).unwrap();
        assert!(d.content.starts_with("# Title only"));
        assert!(!d.content.contains("null"), "body=null must not render as 'null'");
    }
}
