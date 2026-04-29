//! Jira issues loader — pull issues from Jira Cloud via the REST v3 API.
//! Ninth SaaS loader. Parallels iter-98 `GithubIssuesLoader` + iter-124
//! `LinearIssuesLoader`.
//!
//! # Auth
//!
//! Jira Cloud: HTTP Basic with `email:api_token`. Tokens are minted at
//! `id.atlassian.com/manage/api-tokens`. Data Center / Server: Bearer
//! PAT — see `with_bearer_token`.
//!
//! # Query
//!
//! JQL (Jira Query Language) — caller-supplied. Common shapes:
//! - `project = ENG AND status = "In Progress"`
//! - `assignee = currentUser() AND resolution = Unresolved`
//! - `created >= -30d AND labels = bug`
//!
//! No default JQL → caller MUST provide one (otherwise you'd scan the
//! whole Jira instance — slow + potentially exposing data).
//!
//! Pagination: `startAt` + `maxResults` (default 50, max 100 per Atlassian).
//!
//! # Description → text (ADF walker)
//!
//! Jira v3 returns the `description` field in Atlassian Document Format
//! (ADF) — a nested JSON tree of typed nodes. We walk it and emit plain
//! text with newlines between block-level containers. Inline marks
//! (bold / italic / code) are dropped — downstream LLM doesn't care.
//!
//! # Metadata per document
//!
//! - `issue_key` (e.g. "ENG-123"), `issue_id`, `summary`
//! - `status`, `priority`, `issuetype`
//! - `assignee`, `reporter` (display names)
//! - `labels` (comma-joined), `components` (comma-joined)
//! - `created`, `updated` (ISO-8601)
//! - `url` — `{base}/browse/{key}` for LLM citations
//! - `source` = `"jira:{key}"`, document id = `"jira:{key}"`

use std::time::Duration;

use base64::Engine;
use litgraph_core::Document;
use serde_json::{json, Value};

use crate::{Loader, LoaderError, LoaderResult};

pub struct JiraIssuesLoader {
    pub base_url: String,
    pub auth_header: String,
    pub jql: String,
    pub max_issues: usize,
    pub timeout: Duration,
}

impl JiraIssuesLoader {
    /// Jira Cloud: email + API token (HTTP Basic).
    pub fn cloud(
        base_url: impl Into<String>,
        email: impl AsRef<str>,
        api_token: impl AsRef<str>,
        jql: impl Into<String>,
    ) -> Self {
        let b64 = base64::engine::general_purpose::STANDARD
            .encode(format!("{}:{}", email.as_ref(), api_token.as_ref()));
        Self {
            base_url: base_url.into(),
            auth_header: format!("Basic {b64}"),
            jql: jql.into(),
            max_issues: 500,
            timeout: Duration::from_secs(30),
        }
    }

    /// Data Center / Server: Personal Access Token via `Bearer`.
    pub fn with_bearer_token(
        base_url: impl Into<String>,
        token: impl AsRef<str>,
        jql: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            auth_header: format!("Bearer {}", token.as_ref()),
            jql: jql.into(),
            max_issues: 500,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_max_issues(mut self, n: usize) -> Self {
        self.max_issues = n;
        self
    }
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
}

impl Loader for JiraIssuesLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(|e| LoaderError::Other(format!("jira build: {e}")))?;

        let mut docs = Vec::new();
        let mut start_at: usize = 0;

        loop {
            let remaining = self.max_issues.saturating_sub(docs.len());
            if remaining == 0 {
                break;
            }
            // Jira's documented default is 50 per page, max 100. We use
            // 50 — larger pages don't speed up typical small result sets
            // and risk hitting per-field-eval timeouts on big JQL.
            let page_size = remaining.min(50);

            let url = format!("{}/rest/api/3/search", self.base_url.trim_end_matches('/'));
            let body = json!({
                "jql": self.jql,
                "startAt": start_at,
                "maxResults": page_size,
                // Limit returned fields to what we actually read — saves
                // bandwidth + avoids overfetching on wide custom-field tenants.
                "fields": [
                    "summary",
                    "description",
                    "status",
                    "priority",
                    "issuetype",
                    "assignee",
                    "reporter",
                    "labels",
                    "components",
                    "created",
                    "updated",
                ],
            });

            let resp = client
                .post(&url)
                .header("Authorization", &self.auth_header)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .map_err(|e| LoaderError::Other(format!("jira send: {e}")))?;

            let status = resp.status();
            let text = resp
                .text()
                .map_err(|e| LoaderError::Other(format!("jira read: {e}")))?;
            if !status.is_success() {
                return Err(LoaderError::Other(format!(
                    "jira {}: {}",
                    status.as_u16(),
                    text
                )));
            }

            let v: Value = serde_json::from_str(&text)
                .map_err(|e| LoaderError::Other(format!("jira parse: {e}")))?;
            let issues = v
                .get("issues")
                .and_then(|i| i.as_array())
                .cloned()
                .unwrap_or_default();

            if issues.is_empty() {
                break;
            }

            for issue in &issues {
                docs.push(issue_to_document(&self.base_url, issue));
                if docs.len() >= self.max_issues {
                    break;
                }
            }

            // Continue paging if the server says there's more.
            let total = v.get("total").and_then(|t| t.as_u64()).unwrap_or(0);
            start_at += issues.len();
            if (start_at as u64) >= total {
                break;
            }
            if docs.len() >= self.max_issues {
                break;
            }
        }

        Ok(docs)
    }
}

fn issue_to_document(base_url: &str, issue: &Value) -> Document {
    let key = issue
        .get("key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let id = issue.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let fields = issue.get("fields").cloned().unwrap_or(Value::Null);

    let summary = fields
        .get("summary")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let status = fields
        .get("status")
        .and_then(|s| s.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let priority = fields
        .get("priority")
        .and_then(|p| p.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let issuetype = fields
        .get("issuetype")
        .and_then(|t| t.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let assignee = display_name_from(&fields, "assignee");
    let reporter = display_name_from(&fields, "reporter");
    let created = fields
        .get("created")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let updated = fields
        .get("updated")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let labels: Vec<String> = fields
        .get("labels")
        .and_then(|l| l.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let components: Vec<String> = fields
        .get("components")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|c| c.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let description_text = fields
        .get("description")
        .map(adf_to_text)
        .unwrap_or_default();

    let content = if description_text.trim().is_empty() {
        format!("# {key} {summary}\n")
    } else {
        format!("# {key} {summary}\n\n{description_text}\n")
    };

    let url = format!("{}/browse/{}", base_url.trim_end_matches('/'), key);
    let source = format!("jira:{key}");

    Document::new(content)
        .with_id(source.clone())
        .with_metadata("issue_id", json!(id))
        .with_metadata("issue_key", json!(key))
        .with_metadata("summary", json!(summary))
        .with_metadata("status", json!(status))
        .with_metadata("priority", json!(priority))
        .with_metadata("issuetype", json!(issuetype))
        .with_metadata("assignee", json!(assignee))
        .with_metadata("reporter", json!(reporter))
        .with_metadata("labels", json!(labels.join(", ")))
        .with_metadata("components", json!(components.join(", ")))
        .with_metadata("created", json!(created))
        .with_metadata("updated", json!(updated))
        .with_metadata("url", json!(url))
        .with_metadata("source", json!(source))
}

fn display_name_from(fields: &Value, key: &str) -> String {
    // Assignee / reporter are either `null` (unassigned) or an object
    // with `displayName`. Account-ID is also present but for privacy-
    // safe defaults we use displayName.
    fields
        .get(key)
        .and_then(|u| u.get("displayName"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Walk Atlassian Document Format (ADF). Return concatenated plain text
/// with double-newline between block-level containers.
///
/// ADF shape:
/// ```json
/// {
///   "type": "doc",
///   "version": 1,
///   "content": [
///     {"type": "paragraph", "content": [
///        {"type": "text", "text": "hello"}]}
///   ]
/// }
/// ```
///
/// Block-level node types that emit newlines between siblings:
/// paragraph, heading, bulletList, orderedList, listItem, codeBlock,
/// blockquote, table, panel, rule. Inline types (text, mention, emoji,
/// hardBreak) emit inline text only.
pub(crate) fn adf_to_text(node: &Value) -> String {
    let mut out = String::new();
    walk_adf(node, &mut out);
    // Collapse any 3+ newlines down to 2 for tidiness.
    let mut collapsed = String::with_capacity(out.len());
    let mut nl_run = 0u8;
    for ch in out.chars() {
        if ch == '\n' {
            nl_run += 1;
            if nl_run <= 2 {
                collapsed.push(ch);
            }
        } else {
            nl_run = 0;
            collapsed.push(ch);
        }
    }
    collapsed.trim().to_string()
}

fn walk_adf(node: &Value, out: &mut String) {
    let ty = node.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match ty {
        "text" => {
            if let Some(s) = node.get("text").and_then(|v| v.as_str()) {
                out.push_str(s);
            }
        }
        "hardBreak" => out.push('\n'),
        "mention" => {
            if let Some(attrs) = node.get("attrs") {
                if let Some(name) = attrs.get("text").and_then(|v| v.as_str()) {
                    out.push_str(name);
                } else if let Some(id) = attrs.get("id").and_then(|v| v.as_str()) {
                    out.push('@');
                    out.push_str(id);
                }
            }
        }
        "emoji" => {
            if let Some(attrs) = node.get("attrs") {
                if let Some(s) = attrs.get("text").and_then(|v| v.as_str()) {
                    out.push_str(s);
                } else if let Some(s) = attrs.get("shortName").and_then(|v| v.as_str()) {
                    out.push_str(s);
                }
            }
        }
        "inlineCard" | "blockCard" => {
            if let Some(url) = node.get("attrs").and_then(|a| a.get("url")).and_then(|u| u.as_str())
            {
                out.push_str(url);
            }
        }
        // Block-level containers — recurse + emit newline between siblings.
        "paragraph" | "heading" | "codeBlock" | "blockquote" | "panel" | "bulletList"
        | "orderedList" | "listItem" | "table" | "tableRow" | "tableHeader" | "tableCell"
        | "doc" | "rule" | "mediaSingle" | "media" => {
            if let Some(children) = node.get("content").and_then(|c| c.as_array()) {
                for child in children {
                    walk_adf(child, out);
                }
            }
            // Block-level separator. Skip for inline-like containers.
            if matches!(
                ty,
                "paragraph"
                    | "heading"
                    | "listItem"
                    | "codeBlock"
                    | "blockquote"
                    | "panel"
                    | "tableRow"
                    | "rule"
            ) {
                out.push('\n');
            }
        }
        _ => {
            // Unknown / custom node — walk its children if any.
            if let Some(children) = node.get("content").and_then(|c| c.as_array()) {
                for child in children {
                    walk_adf(child, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};
    use std::thread;

    struct FakeJira {
        listener: TcpListener,
        pages: Arc<Mutex<Vec<String>>>,
        captured_bodies: Arc<Mutex<Vec<Vec<u8>>>>,
        captured_auth: Arc<Mutex<Vec<String>>>,
    }

    impl FakeJira {
        fn spawn(pages: Vec<String>) -> (String, Arc<Mutex<Vec<Vec<u8>>>>, Arc<Mutex<Vec<String>>>) {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let url = format!("http://127.0.0.1:{port}");
            let pages = Arc::new(Mutex::new(pages));
            let bodies = Arc::new(Mutex::new(Vec::new()));
            let auth = Arc::new(Mutex::new(Vec::new()));
            let srv = FakeJira {
                listener,
                pages: pages.clone(),
                captured_bodies: bodies.clone(),
                captured_auth: auth.clone(),
            };
            thread::spawn(move || srv.run());
            (url, bodies, auth)
        }

        fn run(self) {
            loop {
                match self.listener.accept() {
                    Ok((stream, _)) => {
                        let pages = self.pages.clone();
                        let bodies = self.captured_bodies.clone();
                        let auth = self.captured_auth.clone();
                        thread::spawn(move || {
                            let body = pages.lock().unwrap().pop().unwrap_or_else(empty_page);
                            handle(stream, &bodies, &auth, body);
                        });
                    }
                    Err(_) => return,
                }
            }
        }
    }

    fn handle(
        mut stream: TcpStream,
        captured_bodies: &Mutex<Vec<Vec<u8>>>,
        captured_auth: &Mutex<Vec<String>>,
        response_body: String,
    ) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            if line == "\r\n" {
                break;
            }
            let lc = line.to_ascii_lowercase();
            if lc.starts_with("content-length:") {
                content_length = line[15..].trim().parse().unwrap_or(0);
            }
            if lc.starts_with("authorization:") {
                captured_auth
                    .lock()
                    .unwrap()
                    .push(line[14..].trim().to_string());
            }
        }
        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body).unwrap();
        }
        captured_bodies.lock().unwrap().push(body);

        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            response_body.len()
        );
        stream.write_all(resp.as_bytes()).unwrap();
        stream.write_all(response_body.as_bytes()).unwrap();
    }

    fn empty_page() -> String {
        json!({"issues": [], "total": 0, "startAt": 0, "maxResults": 50}).to_string()
    }

    fn adf_paragraph(text: &str) -> Value {
        json!({
            "type": "doc", "version": 1,
            "content": [{
                "type": "paragraph",
                "content": [{"type": "text", "text": text}]
            }]
        })
    }

    fn issue(key: &str, summary: &str, description_adf: Value) -> Value {
        json!({
            "id": "10001",
            "key": key,
            "fields": {
                "summary": summary,
                "description": description_adf,
                "status": {"name": "In Progress"},
                "priority": {"name": "High"},
                "issuetype": {"name": "Bug"},
                "assignee": {"displayName": "Alice"},
                "reporter": {"displayName": "Bob"},
                "labels": ["backend", "urgent"],
                "components": [{"name": "api"}, {"name": "auth"}],
                "created": "2026-04-01T10:00:00Z",
                "updated": "2026-04-02T12:00:00Z",
            }
        })
    }

    fn page(issues: Vec<Value>, total: u64) -> String {
        json!({
            "issues": issues,
            "total": total,
            "startAt": 0,
            "maxResults": 50,
        })
        .to_string()
    }

    #[test]
    fn cloud_auth_sends_basic_with_email_token_encoded() {
        let (url, _b, auth) = FakeJira::spawn(vec![page(
            vec![issue("ENG-1", "first", adf_paragraph("body"))],
            1,
        )]);
        let loader = JiraIssuesLoader::cloud(&url, "me@co.com", "tok123", "project = ENG");
        let _ = loader.load().unwrap();
        let a = &auth.lock().unwrap()[0];
        assert!(a.starts_with("Basic "));
        let expected = base64::engine::general_purpose::STANDARD.encode("me@co.com:tok123");
        assert_eq!(a, &format!("Basic {expected}"));
    }

    #[test]
    fn bearer_token_for_data_center() {
        let (url, _b, auth) = FakeJira::spawn(vec![page(
            vec![issue("DC-1", "dc test", adf_paragraph("x"))],
            1,
        )]);
        let loader = JiraIssuesLoader::with_bearer_token(&url, "pat_abc", "project = DC");
        let _ = loader.load().unwrap();
        assert_eq!(auth.lock().unwrap()[0], "Bearer pat_abc");
    }

    #[test]
    fn request_body_carries_jql_and_pagination_params() {
        let (url, bodies, _a) = FakeJira::spawn(vec![page(
            vec![issue("ENG-1", "a", adf_paragraph("b"))],
            1,
        )]);
        let loader =
            JiraIssuesLoader::cloud(&url, "e", "t", "project = ENG AND status = Open");
        let _ = loader.load().unwrap();
        let body: Value = serde_json::from_slice(&bodies.lock().unwrap()[0]).unwrap();
        assert_eq!(body["jql"], "project = ENG AND status = Open");
        assert_eq!(body["startAt"], 0);
        assert_eq!(body["maxResults"], 50);
        // Fields list limits to the ones we actually use.
        let fields = body["fields"].as_array().unwrap();
        assert!(fields.iter().any(|f| f == "summary"));
        assert!(fields.iter().any(|f| f == "status"));
    }

    #[test]
    fn content_has_markdown_h1_with_key_and_summary() {
        let (url, _b, _a) = FakeJira::spawn(vec![page(
            vec![issue(
                "ENG-1",
                "fix the auth bug",
                adf_paragraph("Root cause here."),
            )],
            1,
        )]);
        let loader = JiraIssuesLoader::cloud(&url, "e", "t", "project = ENG");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.starts_with("# ENG-1 fix the auth bug"));
        assert!(docs[0].content.contains("Root cause here."));
    }

    #[test]
    fn metadata_fields_populated() {
        let (url, _b, _a) = FakeJira::spawn(vec![page(
            vec![issue("ENG-5", "topic", adf_paragraph("d"))],
            1,
        )]);
        let loader = JiraIssuesLoader::cloud(&url, "e", "t", "project = ENG");
        let docs = loader.load().unwrap();
        let md = &docs[0].metadata;
        assert_eq!(md.get("issue_key").unwrap().as_str().unwrap(), "ENG-5");
        assert_eq!(md.get("status").unwrap().as_str().unwrap(), "In Progress");
        assert_eq!(md.get("priority").unwrap().as_str().unwrap(), "High");
        assert_eq!(md.get("issuetype").unwrap().as_str().unwrap(), "Bug");
        assert_eq!(md.get("assignee").unwrap().as_str().unwrap(), "Alice");
        assert_eq!(md.get("reporter").unwrap().as_str().unwrap(), "Bob");
        assert_eq!(
            md.get("labels").unwrap().as_str().unwrap(),
            "backend, urgent"
        );
        assert_eq!(md.get("components").unwrap().as_str().unwrap(), "api, auth");
        assert_eq!(
            md.get("source").unwrap().as_str().unwrap(),
            "jira:ENG-5"
        );
        assert_eq!(docs[0].id.as_deref(), Some("jira:ENG-5"));
        let url_md = md.get("url").unwrap().as_str().unwrap();
        assert!(url_md.ends_with("/browse/ENG-5"));
    }

    #[test]
    fn paginates_until_total_reached() {
        // Total 3, maxResults 50 per page → could fit in one page. But
        // we test that multi-page logic works: first page has 2 issues,
        // second page has 1, loader continues until start_at >= total.
        //
        // Trick: server returns same total but different issues based on
        // request body's startAt. Our fake server consumes pages LIFO.
        let p1 = json!({
            "issues": [issue("E-1", "t", adf_paragraph("x")), issue("E-2", "t", adf_paragraph("x"))],
            "total": 3, "startAt": 0, "maxResults": 2,
        })
        .to_string();
        let p2 = json!({
            "issues": [issue("E-3", "t", adf_paragraph("x"))],
            "total": 3, "startAt": 2, "maxResults": 2,
        })
        .to_string();
        // PAGES popped LIFO → second page pushed first.
        let (url, bodies, _a) = FakeJira::spawn(vec![p2, p1]);
        let loader = JiraIssuesLoader::cloud(&url, "e", "t", "project = E").with_max_issues(10);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 3);
        let keys: Vec<String> = docs
            .iter()
            .map(|d| {
                d.metadata
                    .get("issue_key")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(keys, vec!["E-1", "E-2", "E-3"]);
        assert_eq!(bodies.lock().unwrap().len(), 2);
    }

    #[test]
    fn max_issues_caps_result() {
        let (url, bodies, _a) = FakeJira::spawn(vec![page(
            vec![
                issue("E-1", "a", adf_paragraph("b")),
                issue("E-2", "a", adf_paragraph("b")),
                issue("E-3", "a", adf_paragraph("b")),
            ],
            100,
        )]);
        let loader =
            JiraIssuesLoader::cloud(&url, "e", "t", "project = E").with_max_issues(2);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
        // Only one page fetched even though total=100 — the max_issues
        // cap halts pagination.
        assert_eq!(bodies.lock().unwrap().len(), 1);
    }

    #[test]
    fn http_error_surfaces() {
        // Fake server won't be nice; hook uses a direct 401-returning server.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                // Read request, ignore.
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let body = r#"{"errorMessages":["Unauthorized"]}"#;
                let resp = format!(
                    "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });
        let url = format!("http://127.0.0.1:{port}");
        let loader = JiraIssuesLoader::cloud(&url, "e", "t", "project = E");
        let err = loader.load().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("401"));
        assert!(msg.contains("Unauthorized"));
    }

    #[test]
    fn empty_issues_returns_empty_vec() {
        let (url, _b, _a) = FakeJira::spawn(vec![empty_page()]);
        let loader = JiraIssuesLoader::cloud(&url, "e", "t", "project = NONE");
        let docs = loader.load().unwrap();
        assert!(docs.is_empty());
    }

    #[test]
    fn empty_description_produces_title_only_document() {
        let (url, _b, _a) = FakeJira::spawn(vec![page(
            vec![issue(
                "ENG-1",
                "just the title",
                json!({"type": "doc", "version": 1, "content": []}),
            )],
            1,
        )]);
        let loader = JiraIssuesLoader::cloud(&url, "e", "t", "project = ENG");
        let docs = loader.load().unwrap();
        assert_eq!(docs[0].content.trim(), "# ENG-1 just the title");
    }

    // ADF walker tests ------------------------------------------------

    #[test]
    fn adf_single_paragraph_extracts_text() {
        let adf = adf_paragraph("hello world");
        assert_eq!(adf_to_text(&adf), "hello world");
    }

    #[test]
    fn adf_multiple_paragraphs_separated_by_blank_line() {
        let adf = json!({
            "type": "doc", "version": 1,
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "first"}]},
                {"type": "paragraph", "content": [{"type": "text", "text": "second"}]},
            ]
        });
        let text = adf_to_text(&adf);
        assert!(text.contains("first"));
        assert!(text.contains("second"));
        assert!(text.contains('\n'));
    }

    #[test]
    fn adf_bullet_list_walks_nested_structure() {
        let adf = json!({
            "type": "doc", "version": 1,
            "content": [{
                "type": "bulletList",
                "content": [
                    {"type": "listItem", "content": [
                        {"type": "paragraph", "content": [
                            {"type": "text", "text": "item one"}]}]},
                    {"type": "listItem", "content": [
                        {"type": "paragraph", "content": [
                            {"type": "text", "text": "item two"}]}]}
                ]
            }]
        });
        let text = adf_to_text(&adf);
        assert!(text.contains("item one"));
        assert!(text.contains("item two"));
    }

    #[test]
    fn adf_mentions_and_emoji_become_text() {
        let adf = json!({
            "type": "doc", "version": 1,
            "content": [{
                "type": "paragraph",
                "content": [
                    {"type": "mention", "attrs": {"id": "u1", "text": "@alice"}},
                    {"type": "text", "text": " is "},
                    {"type": "emoji", "attrs": {"shortName": ":smile:"}}
                ]
            }]
        });
        let text = adf_to_text(&adf);
        assert!(text.contains("@alice"));
        assert!(text.contains(":smile:"));
    }

    #[test]
    fn adf_hard_break_inserts_newline() {
        let adf = json!({
            "type": "doc", "version": 1,
            "content": [{
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "line1"},
                    {"type": "hardBreak"},
                    {"type": "text", "text": "line2"}
                ]
            }]
        });
        let text = adf_to_text(&adf);
        assert!(text.contains("line1\nline2"));
    }

    #[test]
    fn adf_unknown_node_type_walks_children() {
        // Custom / future node types: walker must recurse into `content`
        // to avoid data loss.
        let adf = json!({
            "type": "doc", "version": 1,
            "content": [{
                "type": "customBlock",
                "content": [{
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "salvaged"}]
                }]
            }]
        });
        assert_eq!(adf_to_text(&adf), "salvaged");
    }
}
