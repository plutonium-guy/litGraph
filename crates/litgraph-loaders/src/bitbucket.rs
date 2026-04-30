//! `BitbucketIssuesLoader` — Bitbucket Cloud REST API issue
//! tracker loader. Third Git provider in the loader trio after
//! GitHub (iter 41-42) and GitLab (iter 149/154).
//!
//! # API surface
//!
//! Bitbucket Cloud uses the v2 REST API at
//! `https://api.bitbucket.org/2.0/`. The issues endpoint is
//! per-repository:
//!
//! - `GET /repositories/{workspace}/{repo}/issues` — list issues.
//!   Returns `{values: [...], next: "https://…?page=2"}`. Cursor-
//!   based pagination via the `next` URL.
//! - `GET /repositories/{workspace}/{repo}/issues/{id}/comments`
//!   — list comments for one issue.
//!
//! # Auth
//!
//! Bitbucket Cloud accepts:
//! - **App password** (Basic auth: `username:app_password`).
//!   Default — most common for CI/server agents.
//! - **OAuth bearer token** (`Authorization: Bearer ...`).
//!   Set via [`BitbucketIssuesLoader::with_oauth`].
//!
//! # Filter knobs
//!
//! - `with_state("open" | "resolved" | "closed" | "all")`. State
//!   names match Bitbucket; aliases `"opened"` (GitLab-style) and
//!   `"closed"` are also accepted and translated.
//! - `with_kind("bug" | "enhancement" | "task" | "proposal" |
//!   "all")`. Defaults to `"all"`.
//! - `with_max_issues(Some(n))` cap.
//! - `with_include_comments(true)` to fetch each issue's comments
//!   and embed them into the issue document.
//!
//! # Output shape
//!
//! Each issue → one `Document`:
//! - `content`: title + body (and optionally comments).
//! - `id`: `bitbucket:{workspace}/{repo}#{issue_id}`.
//! - `metadata`: `{workspace, repo, issue_id, state, kind,
//!   priority, assignee, reporter, created_on, updated_on,
//!   votes, watches, link}`.

use std::time::Duration;

use litgraph_core::Document;
use serde_json::{json, Value};

use crate::{Loader, LoaderError, LoaderResult};

const BITBUCKET_API: &str = "https://api.bitbucket.org/2.0";

pub struct BitbucketIssuesLoader {
    pub username: String,
    /// App password OR OAuth bearer (depending on `oauth` flag).
    pub token: String,
    pub workspace: String,
    pub repo: String,
    pub base_url: String,
    pub timeout: Duration,
    /// Filter: `"open"`, `"resolved"`, `"closed"`, `"all"`.
    /// Aliases `"opened"` is accepted and translated.
    pub state: String,
    /// Filter: `"bug"`, `"enhancement"`, `"task"`, `"proposal"`,
    /// `"all"`. Default `"all"`.
    pub kind: String,
    pub include_comments: bool,
    pub max_issues: Option<usize>,
    /// `false` → Basic auth (username + app password). `true` →
    /// `Authorization: Bearer <token>`. Default false.
    pub oauth: bool,
}

impl BitbucketIssuesLoader {
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
            base_url: BITBUCKET_API.into(),
            timeout: Duration::from_secs(30),
            state: "all".into(),
            kind: "all".into(),
            include_comments: false,
            max_issues: Some(1000),
            oauth: false,
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
    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = kind.into();
        self
    }
    pub fn with_include_comments(mut self, b: bool) -> Self {
        self.include_comments = b;
        self
    }
    pub fn with_max_issues(mut self, n: Option<usize>) -> Self {
        self.max_issues = n;
        self
    }
    pub fn with_oauth(mut self, b: bool) -> Self {
        self.oauth = b;
        self
    }
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    fn translate_state(&self) -> &str {
        match self.state.as_str() {
            // Accept GitLab/GitHub-style aliases.
            "opened" => "open",
            other => other,
        }
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

    fn issues_url(&self) -> String {
        let mut q = String::new();
        let state = self.translate_state();
        if state != "all" {
            q.push_str(&format!("state=\"{state}\""));
        }
        if self.kind != "all" {
            if !q.is_empty() {
                q.push_str(" AND ");
            }
            q.push_str(&format!("kind=\"{}\"", self.kind));
        }
        // Bitbucket's filter syntax is BBQL via `q=`. URL-encode
        // the q-clause minimally — quotes and spaces are the only
        // commonly needed escapes here.
        let q_encoded = q.replace(' ', "%20").replace('"', "%22");
        let mut url = format!(
            "{}/repositories/{}/{}/issues?pagelen=50",
            self.base_url.trim_end_matches('/'),
            self.workspace,
            self.repo,
        );
        if !q_encoded.is_empty() {
            url.push_str(&format!("&q={q_encoded}"));
        }
        url
    }

    fn fetch_page(
        &self,
        client: &reqwest::blocking::Client,
        url: &str,
    ) -> LoaderResult<Page> {
        let resp = self.authed(client.get(url)).send()?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "bitbucket {status}: {body}",
            )));
        }
        let v: Value = resp.json()?;
        let values = v
            .get("values")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        let next = v
            .get("next")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string());
        Ok(Page { values, next })
    }

    fn fetch_comments(
        &self,
        client: &reqwest::blocking::Client,
        issue_id: i64,
    ) -> LoaderResult<Vec<Value>> {
        let url = format!(
            "{}/repositories/{}/{}/issues/{}/comments?pagelen=100",
            self.base_url.trim_end_matches('/'),
            self.workspace,
            self.repo,
            issue_id,
        );
        let resp = self.authed(client.get(&url)).send()?;
        if !resp.status().is_success() {
            return Ok(Vec::new());
        }
        let v: Value = resp.json()?;
        Ok(v.get("values")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default())
    }

    /// Convert one issue + its (optional) comments into a Document.
    /// Public for offline tests.
    pub fn issue_to_document(
        &self,
        issue: &Value,
        comments: &[Value],
    ) -> Option<Document> {
        let id = issue.get("id")?.as_i64()?;
        let title = issue
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let body = issue
            .get("content")
            .and_then(|v| v.get("raw"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let mut content = if body.is_empty() {
            title.clone()
        } else {
            format!("{title}\n\n{body}")
        };
        if !comments.is_empty() {
            content.push_str("\n\n--- Comments ---\n");
            for c in comments {
                let user = c
                    .get("user")
                    .and_then(|u| u.get("display_name").or(u.get("nickname")))
                    .and_then(|v| v.as_str())
                    .unwrap_or("anon");
                let body = c
                    .get("content")
                    .and_then(|v| v.get("raw"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !body.is_empty() {
                    content.push_str(&format!("\n@{user}: {body}\n"));
                }
            }
        }
        let mut doc = Document::new(content).with_id(format!(
            "bitbucket:{}/{}#{id}",
            self.workspace, self.repo,
        ));
        doc.metadata
            .insert("workspace".into(), json!(self.workspace));
        doc.metadata.insert("repo".into(), json!(self.repo));
        doc.metadata.insert("issue_id".into(), json!(id));
        doc.metadata.insert("title".into(), json!(title));
        for f in [
            "state",
            "kind",
            "priority",
            "votes",
            "watches",
            "created_on",
            "updated_on",
        ] {
            if let Some(v) = issue.get(f).cloned() {
                doc.metadata.insert(f.into(), v);
            }
        }
        for (k, path) in [
            ("assignee", &["assignee", "display_name"][..]),
            ("reporter", &["reporter", "display_name"][..]),
        ] {
            let mut cur = issue;
            let mut got = None;
            for seg in path {
                if let Some(next) = cur.get(*seg) {
                    cur = next;
                    got = Some(cur);
                } else {
                    got = None;
                    break;
                }
            }
            if let Some(v) = got {
                if let Some(s) = v.as_str() {
                    doc.metadata.insert(k.into(), json!(s));
                }
            }
        }
        if let Some(link) = issue
            .get("links")
            .and_then(|l| l.get("html"))
            .and_then(|h| h.get("href"))
            .and_then(|v| v.as_str())
        {
            doc.metadata.insert("link".into(), json!(link));
        }
        Some(doc)
    }
}

struct Page {
    values: Vec<Value>,
    next: Option<String>,
}

impl Loader for BitbucketIssuesLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.client()?;
        let mut docs: Vec<Document> = Vec::new();
        let mut url = self.issues_url();
        let cap = self.max_issues.unwrap_or(usize::MAX);
        loop {
            if docs.len() >= cap {
                break;
            }
            let page = self.fetch_page(&client, &url)?;
            for issue in &page.values {
                if docs.len() >= cap {
                    break;
                }
                let comments = if self.include_comments {
                    let id = issue.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                    self.fetch_comments(&client, id).unwrap_or_default()
                } else {
                    Vec::new()
                };
                if let Some(doc) = self.issue_to_document(issue, &comments) {
                    docs.push(doc);
                }
            }
            match page.next {
                Some(n) => url = n,
                None => break,
            }
        }
        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_issue() -> Value {
        json!({
            "id": 7,
            "title": "Crash on startup",
            "content": {"raw": "Reproduces on macOS 14.0\n```\npanic!\n```"},
            "state": "open",
            "kind": "bug",
            "priority": "major",
            "votes": 3,
            "watches": 5,
            "created_on": "2024-01-15T12:00:00Z",
            "updated_on": "2024-01-16T08:00:00Z",
            "assignee": {"display_name": "Alice Doe"},
            "reporter": {"display_name": "Bob Smith"},
            "links": {"html": {"href": "https://bitbucket.org/ws/repo/issues/7"}}
        })
    }

    #[test]
    fn issue_to_document_with_no_comments() {
        let l = BitbucketIssuesLoader::new("u", "p", "ws", "repo");
        let issue = fixture_issue();
        let doc = l.issue_to_document(&issue, &[]).unwrap();
        assert!(doc.content.contains("Crash on startup"));
        assert!(doc.content.contains("Reproduces on macOS"));
        assert_eq!(doc.id.as_deref(), Some("bitbucket:ws/repo#7"));
        assert_eq!(
            doc.metadata.get("issue_id").and_then(|v| v.as_i64()),
            Some(7),
        );
        assert_eq!(
            doc.metadata.get("kind").and_then(|v| v.as_str()),
            Some("bug"),
        );
        assert_eq!(
            doc.metadata.get("assignee").and_then(|v| v.as_str()),
            Some("Alice Doe"),
        );
        assert_eq!(
            doc.metadata.get("link").and_then(|v| v.as_str()),
            Some("https://bitbucket.org/ws/repo/issues/7"),
        );
    }

    #[test]
    fn issue_to_document_with_comments_appends_thread() {
        let l = BitbucketIssuesLoader::new("u", "p", "ws", "repo");
        let issue = fixture_issue();
        let comments = vec![
            json!({"user": {"display_name": "Eve"}, "content": {"raw": "Same here on 13.6"}}),
            json!({"user": {"display_name": "Frank"}, "content": {"raw": "Workaround: …"}}),
        ];
        let doc = l.issue_to_document(&issue, &comments).unwrap();
        assert!(doc.content.contains("--- Comments ---"));
        assert!(doc.content.contains("@Eve: Same here on 13.6"));
        assert!(doc.content.contains("@Frank: Workaround"));
    }

    #[test]
    fn issue_with_no_id_returns_none() {
        let l = BitbucketIssuesLoader::new("u", "p", "ws", "repo");
        let issue = json!({"title": "no id"});
        assert!(l.issue_to_document(&issue, &[]).is_none());
    }

    #[test]
    fn translate_state_aliases() {
        let l = BitbucketIssuesLoader::new("u", "p", "ws", "repo")
            .with_state("opened");
        assert_eq!(l.translate_state(), "open");
        let l2 = BitbucketIssuesLoader::new("u", "p", "ws", "repo")
            .with_state("resolved");
        assert_eq!(l2.translate_state(), "resolved");
    }

    #[test]
    fn issues_url_contains_state_and_kind_filters() {
        let l = BitbucketIssuesLoader::new("u", "p", "ws", "repo")
            .with_state("open")
            .with_kind("bug");
        let url = l.issues_url();
        assert!(url.contains("/repositories/ws/repo/issues"));
        assert!(url.contains("pagelen=50"));
        assert!(url.contains("q="));
        assert!(url.contains("state=%22open%22"));
        assert!(url.contains("kind=%22bug%22"));
        assert!(url.contains("AND"));
    }

    #[test]
    fn issues_url_drops_q_when_all_all() {
        let l = BitbucketIssuesLoader::new("u", "p", "ws", "repo");
        let url = l.issues_url();
        assert!(!url.contains("q="));
    }

    #[test]
    fn body_only_falls_back_to_title() {
        let l = BitbucketIssuesLoader::new("u", "p", "ws", "repo");
        let issue = json!({
            "id": 9,
            "title": "Title-only issue",
            "content": {"raw": ""},
            "state": "open",
            "kind": "task"
        });
        let doc = l.issue_to_document(&issue, &[]).unwrap();
        assert_eq!(doc.content, "Title-only issue");
    }
}
