//! Linear issues loader — pull issues from Linear.app via their GraphQL
//! API. Completes the SaaS-loader matrix's 8th source. New pattern: this
//! is the first GraphQL loader (all prior loaders are REST).
//!
//! # Auth
//!
//! Linear API keys go in the `Authorization` header with NO `Bearer`
//! prefix (Linear quirk — distinct from every other SaaS in this repo).
//! Keys are scoped workspace-wide; rate limit 1500 req/hr per key.
//!
//! # Query shape
//!
//! One GraphQL query covers listing + per-issue fields — no N+1 fetches.
//! Pagination via Relay-style cursor (`pageInfo.endCursor`). Each page
//! returns up to 50 issues (Linear's default; max 250).
//!
//! # Filters
//!
//! All are OPTIONAL and stack via GraphQL's `filter` AND-semantics:
//! - `team_key` — scope to a single team (e.g. "ENG"). Translates to
//!   `filter: {team: {key: {eq: "..."}}}`.
//! - `state_names` — list of state names. Issues in ANY of the listed
//!   states pass.
//! - `label_names` — issues with ANY of these labels pass.
//! - `max_issues` — hard cap on returned issues (default 500).
//!
//! # Metadata per document
//!
//! - `issue_id`, `identifier` (e.g. "ENG-123"), `title`
//! - `state_name`, `team_name` (if present)
//! - `labels` (comma-joined)
//! - `created_at` / `updated_at` — ISO-8601
//! - `url` — web URL for LLM citations
//! - `source` = `"linear:{identifier}"`, document id = `"linear:{identifier}"`

use std::time::Duration;

use litgraph_core::Document;
use serde_json::{json, Value};

use crate::{Loader, LoaderError, LoaderResult};

const LINEAR_GRAPHQL: &str = "https://api.linear.app/graphql";

pub struct LinearIssuesLoader {
    pub api_key: String,
    pub base_url: String,
    pub timeout: Duration,
    pub team_key: Option<String>,
    pub state_names: Vec<String>,
    pub label_names: Vec<String>,
    pub max_issues: usize,
}

impl LinearIssuesLoader {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: LINEAR_GRAPHQL.into(),
            timeout: Duration::from_secs(30),
            team_key: None,
            state_names: Vec::new(),
            label_names: Vec::new(),
            max_issues: 500,
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
    pub fn with_team(mut self, team_key: impl Into<String>) -> Self {
        self.team_key = Some(team_key.into());
        self
    }
    pub fn with_states<I, S>(mut self, states: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.state_names = states.into_iter().map(Into::into).collect();
        self
    }
    pub fn with_labels<I, S>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.label_names = labels.into_iter().map(Into::into).collect();
        self
    }
    pub fn with_max_issues(mut self, n: usize) -> Self {
        self.max_issues = n;
        self
    }

    /// Build the GraphQL `filter` object from the configured filters.
    /// `None` if no filters set → omit the `filter` arg entirely.
    fn build_filter(&self) -> Option<Value> {
        let mut f = serde_json::Map::new();
        if let Some(ref k) = self.team_key {
            f.insert("team".into(), json!({"key": {"eq": k}}));
        }
        if !self.state_names.is_empty() {
            f.insert("state".into(), json!({"name": {"in": self.state_names}}));
        }
        if !self.label_names.is_empty() {
            // `labels.some.name.in` — issue where any of its labels has
            // a name in the list.
            f.insert(
                "labels".into(),
                json!({"some": {"name": {"in": self.label_names}}}),
            );
        }
        if f.is_empty() {
            None
        } else {
            Some(Value::Object(f))
        }
    }
}

const ISSUES_QUERY: &str = "
query Issues($first: Int!, $after: String, $filter: IssueFilter) {
  issues(first: $first, after: $after, filter: $filter) {
    pageInfo { endCursor hasNextPage }
    nodes {
      id
      identifier
      title
      description
      url
      createdAt
      updatedAt
      state { name }
      team { name key }
      labels { nodes { name } }
    }
  }
}
";

impl Loader for LinearIssuesLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(|e| LoaderError::Other(format!("linear build: {e}")))?;

        let mut docs = Vec::new();
        let mut after: Option<String> = None;
        let filter = self.build_filter();

        loop {
            // Page size: 50 per Linear default. Clamp final page so we
            // don't overfetch past `max_issues`.
            let remaining = self.max_issues.saturating_sub(docs.len());
            if remaining == 0 {
                break;
            }
            let page_size = remaining.min(50);

            let mut vars = serde_json::Map::new();
            vars.insert("first".into(), json!(page_size));
            if let Some(ref cursor) = after {
                vars.insert("after".into(), Value::String(cursor.clone()));
            }
            if let Some(ref f) = filter {
                vars.insert("filter".into(), f.clone());
            }

            let body = json!({
                "query": ISSUES_QUERY,
                "variables": Value::Object(vars),
            });

            let resp = client
                .post(&self.base_url)
                // Linear quirk: raw API key in Authorization header, NO "Bearer".
                .header("Authorization", &self.api_key)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .map_err(|e| LoaderError::Other(format!("linear send: {e}")))?;

            let status = resp.status();
            let text = resp
                .text()
                .map_err(|e| LoaderError::Other(format!("linear read: {e}")))?;
            if !status.is_success() {
                return Err(LoaderError::Other(format!(
                    "linear {}: {}",
                    status.as_u16(),
                    text
                )));
            }

            let v: Value = serde_json::from_str(&text)
                .map_err(|e| LoaderError::Other(format!("linear parse: {e}")))?;
            if let Some(errors) = v.get("errors").and_then(|e| e.as_array()) {
                if !errors.is_empty() {
                    let msg = errors
                        .iter()
                        .filter_map(|e| e.get("message").and_then(|m| m.as_str()))
                        .collect::<Vec<_>>()
                        .join("; ");
                    return Err(LoaderError::Other(format!("linear graphql: {msg}")));
                }
            }

            let issues_obj = v
                .get("data")
                .and_then(|d| d.get("issues"))
                .ok_or_else(|| {
                    LoaderError::Other("linear: missing data.issues in response".into())
                })?;
            let nodes = issues_obj
                .get("nodes")
                .and_then(|n| n.as_array())
                .cloned()
                .unwrap_or_default();

            for node in &nodes {
                docs.push(issue_to_document(node));
                if docs.len() >= self.max_issues {
                    break;
                }
            }

            // Pagination.
            let page_info = issues_obj.get("pageInfo").cloned().unwrap_or_default();
            let has_next = page_info
                .get("hasNextPage")
                .and_then(|b| b.as_bool())
                .unwrap_or(false);
            if !has_next || docs.len() >= self.max_issues {
                break;
            }
            after = page_info
                .get("endCursor")
                .and_then(|c| c.as_str())
                .map(String::from);
            if after.is_none() {
                break;
            }
        }

        Ok(docs)
    }
}

fn issue_to_document(node: &Value) -> Document {
    let id = node.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let identifier = node
        .get("identifier")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let title = node.get("title").and_then(|v| v.as_str()).unwrap_or("");
    let description = node
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let state_name = node
        .get("state")
        .and_then(|s| s.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let team_name = node
        .get("team")
        .and_then(|t| t.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let url = node.get("url").and_then(|v| v.as_str()).unwrap_or("");
    let created = node
        .get("createdAt")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let updated = node
        .get("updatedAt")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let labels: Vec<String> = node
        .get("labels")
        .and_then(|l| l.get("nodes"))
        .and_then(|n| n.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|lbl| {
                    lbl.get("name").and_then(|n| n.as_str()).map(String::from)
                })
                .collect()
        })
        .unwrap_or_default();

    // Content: markdown H1 with identifier + title, then description.
    let content = if description.is_empty() {
        format!("# {identifier} {title}\n")
    } else {
        format!("# {identifier} {title}\n\n{description}\n")
    };

    let source = format!("linear:{identifier}");
    let mut doc = Document::new(content)
        .with_id(source.clone())
        .with_metadata("issue_id", json!(id))
        .with_metadata("identifier", json!(identifier))
        .with_metadata("title", json!(title))
        .with_metadata("state_name", json!(state_name))
        .with_metadata("team_name", json!(team_name))
        .with_metadata("labels", json!(labels.join(", ")))
        .with_metadata("created_at", json!(created))
        .with_metadata("updated_at", json!(updated))
        .with_metadata("url", json!(url))
        .with_metadata("source", json!(source.clone()));
    // expose labels array too for structured filtering downstream
    doc = doc.with_metadata("labels_array", json!(labels));
    doc
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};
    use std::thread;

    /// Configurable fake GraphQL endpoint. Responds with canned page
    /// payloads in order; captures every request body for inspection.
    struct FakeLinear {
        listener: TcpListener,
        pages: Arc<Mutex<Vec<String>>>,
        captured_bodies: Arc<Mutex<Vec<Vec<u8>>>>,
        captured_auth: Arc<Mutex<Vec<String>>>,
    }

    impl FakeLinear {
        fn spawn(pages: Vec<String>) -> (String, Arc<Mutex<Vec<Vec<u8>>>>, Arc<Mutex<Vec<String>>>) {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let url = format!("http://127.0.0.1:{port}/graphql");
            let pages = Arc::new(Mutex::new(pages));
            let captured_bodies = Arc::new(Mutex::new(Vec::new()));
            let captured_auth = Arc::new(Mutex::new(Vec::new()));
            let srv = FakeLinear {
                listener,
                pages: pages.clone(),
                captured_bodies: captured_bodies.clone(),
                captured_auth: captured_auth.clone(),
            };
            thread::spawn(move || srv.run());
            (url, captured_bodies, captured_auth)
        }

        fn run(self) {
            loop {
                match self.listener.accept() {
                    Ok((stream, _)) => {
                        let pages = self.pages.clone();
                        let bodies = self.captured_bodies.clone();
                        let auth = self.captured_auth.clone();
                        thread::spawn(move || {
                            let body = pages
                                .lock()
                                .unwrap()
                                .pop()
                                .unwrap_or_else(|| r#"{"data":{"issues":{"pageInfo":{"hasNextPage":false},"nodes":[]}}}"#.into());
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

    fn page(has_next: bool, cursor: &str, issues: &[(&str, &str, &str)]) -> String {
        let nodes: Vec<Value> = issues
            .iter()
            .enumerate()
            .map(|(i, (ident, title, desc))| {
                json!({
                    "id": format!("id_{i}"),
                    "identifier": ident,
                    "title": title,
                    "description": desc,
                    "url": format!("https://linear.app/team/issue/{ident}"),
                    "createdAt": "2026-04-01T00:00:00Z",
                    "updatedAt": "2026-04-02T00:00:00Z",
                    "state": {"name": "In Progress"},
                    "team": {"name": "Engineering", "key": "ENG"},
                    "labels": {"nodes": [{"name": "bug"}, {"name": "p1"}]},
                })
            })
            .collect();
        json!({
            "data": {
                "issues": {
                    "pageInfo": {"endCursor": cursor, "hasNextPage": has_next},
                    "nodes": nodes,
                }
            }
        })
        .to_string()
    }

    #[test]
    fn loads_single_page_and_builds_documents() {
        let (url, bodies, auth) = FakeLinear::spawn(vec![page(
            false,
            "cursor0",
            &[("ENG-1", "first issue", "body one")],
        )]);
        let loader = LinearIssuesLoader::new("lin_secret").with_base_url(&url);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].metadata.get("identifier").unwrap().as_str().unwrap(),
            "ENG-1"
        );
        assert!(docs[0].content.contains("# ENG-1 first issue"));
        assert!(docs[0].content.contains("body one"));
        assert_eq!(docs[0].id.as_deref(), Some("linear:ENG-1"));

        // Auth header has RAW key (no Bearer prefix).
        let auth_headers = auth.lock().unwrap();
        assert_eq!(auth_headers[0], "lin_secret");
        assert!(!auth_headers[0].starts_with("Bearer"));

        // Exactly one POST (no extra pagination requests).
        assert_eq!(bodies.lock().unwrap().len(), 1);
    }

    #[test]
    fn paginates_across_multiple_pages() {
        // Pages are popped from end; so second-page goes first in the vec.
        let (url, bodies, _auth) = FakeLinear::spawn(vec![
            page(false, "cursor2", &[("ENG-3", "third", "c")]),
            page(true, "cursor1", &[("ENG-2", "second", "b")]),
            page(true, "cursor0", &[("ENG-1", "first", "a")]),
        ]);
        let loader = LinearIssuesLoader::new("k").with_base_url(&url);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 3);
        let idents: Vec<String> = docs
            .iter()
            .map(|d| {
                d.metadata
                    .get("identifier")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        assert_eq!(idents, vec!["ENG-1", "ENG-2", "ENG-3"]);
        // 3 pages fetched.
        assert_eq!(bodies.lock().unwrap().len(), 3);
    }

    #[test]
    fn max_issues_caps_result_and_stops_pagination_early() {
        // 3 pages available, but loader caps at 2 issues.
        let (url, bodies, _auth) = FakeLinear::spawn(vec![
            page(true, "x", &[("ENG-9", "unused", "u")]), // popped last
            page(true, "cursor1", &[("ENG-2", "second", "b")]),
            page(true, "cursor0", &[("ENG-1", "first", "a")]),
        ]);
        let loader = LinearIssuesLoader::new("k")
            .with_base_url(&url)
            .with_max_issues(2);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
        // Third page never fetched.
        assert_eq!(bodies.lock().unwrap().len(), 2);
    }

    #[test]
    fn team_filter_injected_into_variables() {
        let (url, bodies, _auth) =
            FakeLinear::spawn(vec![page(false, "x", &[("ENG-1", "t", "b")])]);
        let loader = LinearIssuesLoader::new("k")
            .with_base_url(&url)
            .with_team("ENG");
        let _ = loader.load().unwrap();
        let body: Value =
            serde_json::from_slice(&bodies.lock().unwrap()[0]).unwrap();
        let filter = &body["variables"]["filter"];
        assert_eq!(filter["team"]["key"]["eq"], "ENG");
    }

    #[test]
    fn state_and_label_filters_stack() {
        let (url, bodies, _auth) =
            FakeLinear::spawn(vec![page(false, "x", &[("ENG-1", "t", "b")])]);
        let loader = LinearIssuesLoader::new("k")
            .with_base_url(&url)
            .with_states(["Todo", "In Progress"])
            .with_labels(["bug"]);
        let _ = loader.load().unwrap();
        let body: Value =
            serde_json::from_slice(&bodies.lock().unwrap()[0]).unwrap();
        let filter = &body["variables"]["filter"];
        assert_eq!(filter["state"]["name"]["in"][0], "Todo");
        assert_eq!(filter["state"]["name"]["in"][1], "In Progress");
        assert_eq!(filter["labels"]["some"]["name"]["in"][0], "bug");
    }

    #[test]
    fn graphql_errors_surface_as_loader_error() {
        let (url, _bodies, _auth) = FakeLinear::spawn(vec![r#"{"errors":[{"message":"Access denied"}]}"#.into()]);
        let loader = LinearIssuesLoader::new("k").with_base_url(&url);
        let err = loader.load().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Access denied"));
    }

    #[test]
    fn labels_metadata_includes_joined_and_array_forms() {
        let (url, _b, _a) =
            FakeLinear::spawn(vec![page(false, "x", &[("ENG-1", "t", "b")])]);
        let loader = LinearIssuesLoader::new("k").with_base_url(&url);
        let docs = loader.load().unwrap();
        assert_eq!(
            docs[0].metadata.get("labels").unwrap().as_str().unwrap(),
            "bug, p1"
        );
        let arr = docs[0]
            .metadata
            .get("labels_array")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn empty_description_still_produces_title_only_document() {
        let (url, _b, _a) = FakeLinear::spawn(vec![page(false, "x", &[("ENG-1", "title", "")])]);
        let loader = LinearIssuesLoader::new("k").with_base_url(&url);
        let docs = loader.load().unwrap();
        assert_eq!(docs[0].content.trim(), "# ENG-1 title");
    }

    #[test]
    fn empty_response_returns_empty_vec_without_error() {
        let (url, _b, _a) =
            FakeLinear::spawn(vec![r#"{"data":{"issues":{"pageInfo":{"hasNextPage":false},"nodes":[]}}}"#.into()]);
        let loader = LinearIssuesLoader::new("k").with_base_url(&url);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 0);
    }
}
