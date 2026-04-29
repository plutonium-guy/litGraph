//! GitLab issues loader. Parallel to `GithubIssuesLoader` (iter ~50)
//! for the 4th major dev platform (after GitHub, Linear, Jira). Self-hosted
//! GitLab instances are common in enterprise — `with_base_url` points at
//! `https://gitlab.your-corp.com/api/v4` instead of the public SaaS.
//!
//! # Auth
//!
//! GitLab personal-access tokens (or OAuth tokens). Sent in the
//! `PRIVATE-TOKEN` header — GitLab's standard, NOT `Authorization: Bearer`.
//! (OAuth bearer tokens use `Authorization: Bearer ...`; we default to
//! `PRIVATE-TOKEN` since most teams generate PATs. Override with
//! `with_oauth(true)` if needed.)
//!
//! # Project addressing
//!
//! GitLab identifies projects by integer ID OR URL-encoded full path
//! (`group%2Fsubgroup%2Fproject`). We accept both via `from_project()`
//! taking `&str` — the user's responsibility to URL-encode if using
//! a path with slashes.

use std::time::Duration;

use litgraph_core::Document;
use serde_json::Value;

use crate::{Loader, LoaderError, LoaderResult};

const GITLAB_API: &str = "https://gitlab.com/api/v4";

pub struct GitLabIssuesLoader {
    pub token: String,
    /// Project identifier — either numeric ID (`"12345"`) or URL-encoded
    /// path (`"group%2Fsubgroup%2Fmyproject"`).
    pub project: String,
    pub base_url: String,
    pub timeout: Duration,
    /// Filter: `"opened"`, `"closed"`, or `"all"` (default `"all"`).
    /// GitLab spells it "opened" not "open" — we accept either and
    /// translate.
    pub state: String,
    pub include_notes: bool,
    /// Comma-separated label list. Empty → no label filter.
    pub labels: Vec<String>,
    pub max_issues: Option<usize>,
    /// `false` → use `PRIVATE-TOKEN` header (PAT). `true` → `Authorization: Bearer`
    /// (OAuth). Default false.
    pub oauth: bool,
}

impl GitLabIssuesLoader {
    pub fn from_project(token: impl Into<String>, project: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            project: project.into(),
            base_url: GITLAB_API.into(),
            timeout: Duration::from_secs(30),
            state: "all".into(),
            include_notes: false,
            labels: Vec::new(),
            max_issues: Some(1000),
            oauth: false,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self { self.base_url = url.into(); self }
    pub fn with_state(mut self, state: impl Into<String>) -> Self { self.state = state.into(); self }
    pub fn with_include_notes(mut self, b: bool) -> Self { self.include_notes = b; self }
    pub fn with_labels<I, S>(mut self, labels: I) -> Self
    where I: IntoIterator<Item = S>, S: Into<String> {
        self.labels = labels.into_iter().map(Into::into).collect();
        self
    }
    pub fn with_max_issues(mut self, n: Option<usize>) -> Self { self.max_issues = n; self }
    pub fn with_oauth(mut self, b: bool) -> Self { self.oauth = b; self }

    fn translate_state(&self) -> &str {
        // GitLab uses "opened"/"closed"/"all"; accept GitHub-style "open" too.
        match self.state.as_str() {
            "open" => "opened",
            other => other,
        }
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

    fn fetch_issues_page(
        &self,
        client: &reqwest::blocking::Client,
        page: u32,
    ) -> LoaderResult<Vec<Value>> {
        let mut url = format!(
            "{}/projects/{}/issues?state={}&per_page=100&page={}",
            self.base_url.trim_end_matches('/'),
            self.project,
            self.translate_state(),
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
            return Err(LoaderError::Other(format!("gitlab issues {status}: {body}")));
        }
        let arr: Vec<Value> = resp.json()?;
        Ok(arr)
    }

    fn fetch_notes(
        &self,
        client: &reqwest::blocking::Client,
        issue_iid: i64,
    ) -> LoaderResult<Vec<Value>> {
        // Notes endpoint = GitLab's "comments." Up to 100 per issue (most
        // are well under that; future-enhancement: paginate for noisy threads).
        let url = format!(
            "{}/projects/{}/issues/{}/notes?per_page=100",
            self.base_url.trim_end_matches('/'),
            self.project,
            issue_iid,
        );
        let resp = self.authed(client.get(&url)).send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!("gitlab notes {status}: {body}")));
        }
        resp.json().map_err(LoaderError::from)
    }

    fn issue_to_document(&self, issue: &Value, notes: Option<&[Value]>) -> Option<Document> {
        // GitLab uses `iid` as the per-project number (what users see); `id`
        // is the global database id. We surface `iid` in the document id.
        let iid = issue.get("iid").and_then(|n| n.as_i64())?;
        let title = issue.get("title").and_then(|t| t.as_str()).unwrap_or("");
        let body = issue.get("description").and_then(|b| b.as_str()).unwrap_or("");
        let author = issue
            .pointer("/author/username")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let state = issue.get("state").and_then(|s| s.as_str()).unwrap_or("");

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
        if let Some(ns) = notes {
            // GitLab returns SYSTEM notes (status changes, label adds) mixed
            // with user comments. Filter system notes — they're noise for RAG.
            let user_notes: Vec<&Value> = ns
                .iter()
                .filter(|n| !n.get("system").and_then(|b| b.as_bool()).unwrap_or(false))
                .collect();
            if !user_notes.is_empty() {
                content.push_str("\n\n--- notes ---\n");
                for n in user_notes {
                    let nuser = n
                        .pointer("/author/username")
                        .and_then(|s| s.as_str())
                        .unwrap_or("");
                    let nbody = n.get("body").and_then(|b| b.as_str()).unwrap_or("");
                    if !nbody.is_empty() {
                        content.push_str(&format!("\n[@{nuser}]: {nbody}\n"));
                    }
                }
            }
        }

        let mut d = Document::new(content);
        d.id = Some(format!("{}#{}", self.project, iid));
        d.metadata.insert("iid".into(), Value::from(iid));
        d.metadata.insert("title".into(), Value::String(title.into()));
        d.metadata.insert("state".into(), Value::String(state.into()));
        d.metadata.insert("author".into(), Value::String(author.into()));
        if let Some(labels) = issue.get("labels").and_then(|l| l.as_array()) {
            let names: Vec<String> = labels
                .iter()
                .filter_map(|l| l.as_str().map(String::from))
                .collect();
            d.metadata.insert("labels".into(), Value::String(names.join(",")));
        }
        if let Some(ts) = issue.get("created_at").and_then(|t| t.as_str()) {
            d.metadata.insert("created_at".into(), Value::String(ts.into()));
        }
        if let Some(ts) = issue.get("updated_at").and_then(|t| t.as_str()) {
            d.metadata.insert("updated_at".into(), Value::String(ts.into()));
        }
        if let Some(url) = issue.get("web_url").and_then(|u| u.as_str()) {
            d.metadata.insert("web_url".into(), Value::String(url.into()));
        }
        d.metadata.insert(
            "source".into(),
            Value::String(format!("gitlab:{}#{}", self.project, iid)),
        );
        Some(d)
    }
}

impl Loader for GitLabIssuesLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.client()?;
        let mut docs = Vec::new();
        let mut page: u32 = 1;
        loop {
            let issues = self.fetch_issues_page(&client, page)?;
            if issues.is_empty() {
                break;
            }
            for issue in &issues {
                if let Some(max) = self.max_issues {
                    if docs.len() >= max {
                        return Ok(docs);
                    }
                }
                let notes = if self.include_notes {
                    let iid = issue.get("iid").and_then(|n| n.as_i64()).unwrap_or(0);
                    if iid > 0 {
                        Some(self.fetch_notes(&client, iid)?)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(doc) = self.issue_to_document(issue, notes.as_deref()) {
                    docs.push(doc);
                }
            }
            // Standard GitLab pagination: less-than-full page → done.
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
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    /// Tiny one-thread fake server. Serves a sequence of (status, body)
    /// responses in order. Captures each request's path + headers for
    /// assertions.
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
                captured_clone
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&total).to_string());
                let (status, body) = responses
                    .get(idx)
                    .cloned()
                    .unwrap_or((200, "[]".to_string()));
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

    fn issue(iid: i64, title: &str, body: &str, state: &str) -> String {
        serde_json::json!({
            "iid": iid,
            "id": iid * 1000,
            "title": title,
            "description": body,
            "state": state,
            "author": {"username": "alice"},
            "labels": ["bug", "p1"],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "web_url": format!("https://gitlab.com/x/y/-/issues/{iid}"),
        }).to_string()
    }

    #[test]
    fn loads_one_page_of_issues() {
        let body = format!("[{}, {}]", issue(1, "first bug", "body 1", "opened"), issue(2, "second bug", "body 2", "closed"));
        let fake = start_fake(vec![(200, body)]);
        let docs = GitLabIssuesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].metadata["iid"], 1);
        assert_eq!(docs[0].metadata["title"], "first bug");
        assert!(docs[0].content.contains("# first bug"));
        assert!(docs[0].content.contains("body 1"));
        assert_eq!(docs[1].metadata["state"], "closed");
    }

    #[test]
    fn private_token_header_used_by_default() {
        let fake = start_fake(vec![(200, "[]".to_string())]);
        let _ = GitLabIssuesLoader::from_project("my-pat", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        let req = &fake.captured.lock().unwrap()[0];
        // HTTP headers are case-insensitive — reqwest may emit lowercased.
        let lower = req.to_lowercase();
        assert!(lower.contains("private-token: my-pat"), "request: {req}");
        assert!(!lower.contains("authorization: bearer"));
    }

    #[test]
    fn oauth_mode_uses_bearer_auth() {
        let fake = start_fake(vec![(200, "[]".to_string())]);
        let _ = GitLabIssuesLoader::from_project("oauth-tok", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .with_oauth(true)
            .load()
            .unwrap();
        let req = &fake.captured.lock().unwrap()[0];
        let lower = req.to_lowercase();
        assert!(lower.contains("authorization: bearer oauth-tok"), "request: {req}");
        assert!(!lower.contains("private-token"));
    }

    #[test]
    fn state_open_translated_to_opened() {
        let fake = start_fake(vec![(200, "[]".to_string())]);
        let _ = GitLabIssuesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .with_state("open")  // GitHub style
            .load()
            .unwrap();
        let req = &fake.captured.lock().unwrap()[0];
        assert!(req.contains("state=opened"), "request: {req}");
    }

    #[test]
    fn labels_appended_to_query() {
        let fake = start_fake(vec![(200, "[]".to_string())]);
        let _ = GitLabIssuesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .with_labels(["bug", "p1"])
            .load()
            .unwrap();
        let req = &fake.captured.lock().unwrap()[0];
        assert!(req.contains("labels=bug,p1"), "request: {req}");
    }

    #[test]
    fn pagination_stops_when_page_smaller_than_100() {
        let body = format!("[{}, {}]", issue(1, "a", "x", "opened"), issue(2, "b", "y", "opened"));
        // 2 < 100 → loader stops after first page; only 1 fetch.
        let fake = start_fake(vec![(200, body)]);
        let docs = GitLabIssuesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(fake.captured.lock().unwrap().len(), 1);
    }

    #[test]
    fn max_issues_cap_respected() {
        // 3 issues but cap=2.
        let body = format!("[{}, {}, {}]",
            issue(1, "a", "x", "opened"),
            issue(2, "b", "y", "opened"),
            issue(3, "c", "z", "opened"),
        );
        let fake = start_fake(vec![(200, body)]);
        let docs = GitLabIssuesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .with_max_issues(Some(2))
            .load()
            .unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn include_notes_appends_user_comments_skips_system() {
        let issue_body = issue(7, "needs reply", "the question", "opened");
        let notes = serde_json::json!([
            {"author": {"username": "bob"}, "body": "i'll look", "system": false},
            {"author": {"username": "ghost"}, "body": "added label bug", "system": true},
            {"author": {"username": "carol"}, "body": "fixed it", "system": false},
        ]).to_string();
        let fake = start_fake(vec![
            (200, format!("[{issue_body}]")),
            (200, notes),
        ]);
        let docs = GitLabIssuesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .with_include_notes(true)
            .load()
            .unwrap();
        assert_eq!(docs.len(), 1);
        let c = &docs[0].content;
        assert!(c.contains("--- notes ---"));
        assert!(c.contains("[@bob]: i'll look"));
        assert!(c.contains("[@carol]: fixed it"));
        // System note filtered out.
        assert!(!c.contains("added label"));
    }

    #[test]
    fn http_error_surfaces_with_status_and_body() {
        let fake = start_fake(vec![(401, r#"{"message":"401 Unauthorized"}"#.to_string())]);
        let err = GitLabIssuesLoader::from_project("bad", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap_err();
        let s = err.to_string();
        assert!(s.contains("401"));
        assert!(s.contains("Unauthorized"));
    }

    #[test]
    fn metadata_carries_iid_state_labels_url() {
        let body = format!("[{}]", issue(42, "the issue", "body", "opened"));
        let fake = start_fake(vec![(200, body)]);
        let docs = GitLabIssuesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        let md = &docs[0].metadata;
        assert_eq!(md["iid"], 42);
        assert_eq!(md["state"], "opened");
        assert_eq!(md["author"], "alice");
        assert_eq!(md["labels"], "bug,p1");
        assert_eq!(md["web_url"], "https://gitlab.com/x/y/-/issues/42");
        assert_eq!(md["source"], "gitlab:12345#42");
    }

    #[test]
    fn document_id_includes_project_and_iid() {
        let body = format!("[{}]", issue(7, "x", "y", "opened"));
        let fake = start_fake(vec![(200, body)]);
        let docs = GitLabIssuesLoader::from_project("tk", "group%2Fmyproject")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        assert_eq!(docs[0].id.as_deref(), Some("group%2Fmyproject#7"));
    }

    #[test]
    fn empty_response_returns_empty_docs() {
        let fake = start_fake(vec![(200, "[]".to_string())]);
        let docs = GitLabIssuesLoader::from_project("tk", "12345")
            .with_base_url(format!("http://127.0.0.1:{}/api/v4", fake.port))
            .load()
            .unwrap();
        assert!(docs.is_empty());
    }
}
