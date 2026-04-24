//! Confluence loader — pull pages from a Confluence space via the REST API.
//!
//! Direct LangChain `ConfluenceLoader` parity. Works with Atlassian Cloud
//! AND self-hosted Server/Data Center — two auth modes are supported
//! (Basic with email+API-token for Cloud; Bearer with Personal Access
//! Token for DC). Pure-Rust via existing `reqwest::blocking`.
//!
//! # Auth modes
//!
//! - **Cloud** (`*.atlassian.net`): Basic auth `email:api_token` —
//!   construct via `from_space_cloud(base_url, email, api_token, space_key)`.
//! - **Server/DC** (self-hosted): Bearer with Personal Access Token —
//!   construct via `from_space_bearer(base_url, token, space_key)`.
//!
//! Both share the same REST surface (`GET /wiki/rest/api/content`) — only
//! the auth header differs.
//!
//! # What we extract
//!
//! Page body comes back as XHTML in the `body.storage.value` field (with
//! `expand=body.storage`). We strip it via the existing `html::strip_html`
//! helper and decode HTML entities. Metadata per document:
//! - `id` — Confluence page id
//! - `title` — page title
//! - `space_key` — the space the page lives in
//! - `version` — Confluence version number (for dedupe / staleness)
//! - `source` — `"confluence:{space_key}:{page_id}"`
//!
//! # Pagination
//!
//! Confluence paginates at `limit` (default 25, max 100). We follow the
//! `_links.next` URL or fall back to `start` + `size` offset arithmetic.

use std::time::Duration;

use litgraph_core::Document;
use serde_json::Value;

use crate::html::{decode_entities, strip_html};
use crate::{Loader, LoaderError, LoaderResult};

enum Auth {
    Basic { email: String, api_token: String },
    Bearer(String),
}

enum Source {
    Space(String),
    Pages(Vec<String>),
}

pub struct ConfluenceLoader {
    auth: Auth,
    base_url: String,
    source: Source,
    pub timeout: Duration,
    /// Cap on total pages loaded. Default 1000.
    pub max_pages: Option<usize>,
}

impl ConfluenceLoader {
    /// Cloud auth — `base_url` like `"https://acme.atlassian.net"`, plus
    /// the user's email + an API token from
    /// <https://id.atlassian.com/manage-profile/security/api-tokens>.
    pub fn from_space_cloud(
        base_url: impl Into<String>,
        email: impl Into<String>,
        api_token: impl Into<String>,
        space_key: impl Into<String>,
    ) -> Self {
        Self {
            auth: Auth::Basic {
                email: email.into(),
                api_token: api_token.into(),
            },
            base_url: base_url.into(),
            source: Source::Space(space_key.into()),
            timeout: Duration::from_secs(30),
            max_pages: Some(1000),
        }
    }

    /// Server/Data Center auth — `base_url` like `"https://wiki.acme.com"`,
    /// plus a Personal Access Token created under the user's profile.
    pub fn from_space_bearer(
        base_url: impl Into<String>,
        token: impl Into<String>,
        space_key: impl Into<String>,
    ) -> Self {
        Self {
            auth: Auth::Bearer(token.into()),
            base_url: base_url.into(),
            source: Source::Space(space_key.into()),
            timeout: Duration::from_secs(30),
            max_pages: Some(1000),
        }
    }

    /// Load specific pages by id (Cloud auth).
    pub fn from_pages_cloud(
        base_url: impl Into<String>,
        email: impl Into<String>,
        api_token: impl Into<String>,
        page_ids: Vec<String>,
    ) -> Self {
        Self {
            auth: Auth::Basic {
                email: email.into(),
                api_token: api_token.into(),
            },
            base_url: base_url.into(),
            source: Source::Pages(page_ids),
            timeout: Duration::from_secs(30),
            max_pages: None,
        }
    }

    /// Load specific pages by id (Bearer auth).
    pub fn from_pages_bearer(
        base_url: impl Into<String>,
        token: impl Into<String>,
        page_ids: Vec<String>,
    ) -> Self {
        Self {
            auth: Auth::Bearer(token.into()),
            base_url: base_url.into(),
            source: Source::Pages(page_ids),
            timeout: Duration::from_secs(30),
            max_pages: None,
        }
    }

    pub fn with_max_pages(mut self, n: Option<usize>) -> Self {
        self.max_pages = n;
        self
    }

    /// Override timeout for slow corporate proxies.
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    fn client(&self) -> LoaderResult<reqwest::blocking::Client> {
        reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(LoaderError::from)
    }

    fn authed(&self, builder: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
        match &self.auth {
            Auth::Basic { email, api_token } => builder.basic_auth(email, Some(api_token)),
            Auth::Bearer(t) => builder.bearer_auth(t),
        }
    }

    /// List pages in a space, following pagination.
    fn list_space_pages(
        &self,
        client: &reqwest::blocking::Client,
        space_key: &str,
    ) -> LoaderResult<Vec<Value>> {
        let mut out: Vec<Value> = Vec::new();
        let mut start: usize = 0;
        let page_size: usize = 100;
        loop {
            let url = format!(
                "{}/wiki/rest/api/content?spaceKey={}&type=page&expand=body.storage,version&limit={}&start={}",
                self.base_url.trim_end_matches('/'),
                urlencode(space_key),
                page_size,
                start,
            );
            let resp = self.authed(client.get(&url)).send()?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                return Err(LoaderError::Other(format!(
                    "confluence content {status}: {body}"
                )));
            }
            let v: Value = resp.json()?;
            let results = v.get("results").and_then(|r| r.as_array()).cloned().unwrap_or_default();
            let got = results.len();
            for page in results {
                out.push(page);
                if let Some(cap) = self.max_pages {
                    if out.len() >= cap {
                        return Ok(out);
                    }
                }
            }
            // Check for next link or "no more pages".
            let size = v.get("size").and_then(|s| s.as_u64()).unwrap_or(got as u64) as usize;
            // `_links.next` present → more pages available.
            let has_next = v
                .pointer("/_links/next")
                .and_then(|x| x.as_str())
                .is_some();
            if !has_next || size == 0 {
                break;
            }
            start += size;
        }
        Ok(out)
    }

    /// Fetch a single page by id (with body + version).
    fn fetch_page(
        &self,
        client: &reqwest::blocking::Client,
        page_id: &str,
    ) -> LoaderResult<Value> {
        let url = format!(
            "{}/wiki/rest/api/content/{}?expand=body.storage,version,space",
            self.base_url.trim_end_matches('/'),
            urlencode(page_id),
        );
        let resp = self.authed(client.get(&url)).send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "confluence page {page_id} {status}: {body}"
            )));
        }
        resp.json().map_err(LoaderError::from)
    }

    /// Convert a Confluence page JSON → Document. Returns None if the page
    /// has no body (archived pages sometimes lack body.storage).
    fn page_to_document(&self, page: &Value, space_key: &str) -> Option<Document> {
        let id = page.get("id").and_then(|s| s.as_str())?.to_string();
        let title = page.get("title").and_then(|s| s.as_str()).unwrap_or("");
        let xhtml = page
            .pointer("/body/storage/value")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if xhtml.is_empty() {
            return None;
        }
        // Reuse the HTML stripper from html.rs — handles paragraph / heading
        // / list block tags and decodes common entities. Confluence XHTML is
        // well-formed HTML5 with some `<ac:*>` macro tags (strip_html
        // silently drops unknown tags).
        let stripped = strip_html(xhtml, true);
        let text = decode_entities(&stripped);

        let mut d = Document::new(text);
        d.id = Some(id.clone());
        d.metadata.insert("page_id".into(), Value::String(id.clone()));
        if !title.is_empty() {
            d.metadata
                .insert("title".into(), Value::String(title.to_string()));
        }
        d.metadata
            .insert("space_key".into(), Value::String(space_key.to_string()));
        if let Some(ver) = page.pointer("/version/number").and_then(|n| n.as_u64()) {
            d.metadata.insert("version".into(), Value::from(ver as i64));
        }
        d.metadata.insert(
            "source".into(),
            Value::String(format!("confluence:{space_key}:{id}")),
        );
        Some(d)
    }
}

impl Loader for ConfluenceLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.client()?;
        let mut docs = Vec::new();
        match &self.source {
            Source::Space(space_key) => {
                let pages = self.list_space_pages(&client, space_key)?;
                for p in &pages {
                    if let Some(d) = self.page_to_document(p, space_key) {
                        docs.push(d);
                    }
                }
            }
            Source::Pages(ids) => {
                for id in ids {
                    if let Some(cap) = self.max_pages {
                        if docs.len() >= cap {
                            break;
                        }
                    }
                    let page = self.fetch_page(&client, id)?;
                    // Pull space_key from the `space.key` field — Confluence
                    // returns it when `expand=space` is set. Fall back to "".
                    let space_key = page
                        .pointer("/space/key")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if let Some(d) = self.page_to_document(&page, &space_key) {
                        docs.push(d);
                    }
                }
            }
        }
        Ok(docs)
    }
}

/// Minimal URL-encoder for space keys / page ids. Copy of notion.rs's helper
/// (kept local to avoid cross-module coupling for one trivial function).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
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
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use std::thread;

    struct FakeServer {
        url: String,
        seen_paths: Arc<Mutex<Vec<String>>>,
        seen_auth: Arc<Mutex<Vec<Option<String>>>>,
        _shutdown: std::sync::mpsc::Sender<()>,
    }

    fn space_page_1() -> Value {
        json!({
            "results": [
                {
                    "id": "100",
                    "title": "First Page",
                    "body": {"storage": {"value": "<p>Body of <strong>first</strong>.</p>"}},
                    "version": {"number": 3}
                },
                {
                    "id": "101",
                    "title": "Second Page",
                    "body": {"storage": {"value": "<h1>Heading</h1><p>Second page body.</p>"}},
                    "version": {"number": 1}
                }
            ],
            "size": 2,
            "_links": {"next": "/wiki/rest/api/content?spaceKey=DEV&start=2"}
        })
    }

    fn space_page_2() -> Value {
        json!({
            "results": [
                {
                    "id": "102",
                    "title": "Third Page",
                    "body": {"storage": {"value": "<p>Third.</p>"}},
                    "version": {"number": 7}
                }
            ],
            "size": 1,
            "_links": {}
        })
    }

    fn single_page(id: &str, space: &str) -> Value {
        json!({
            "id": id,
            "title": format!("Direct Page {id}"),
            "body": {"storage": {"value": format!("<p>Direct body for {id}.</p>")}},
            "version": {"number": 2},
            "space": {"key": space}
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

                        // Dispatch.
                        let body = if path.starts_with("/wiki/rest/api/content/") {
                            // /content/{id}...
                            let id = path
                                .trim_start_matches("/wiki/rest/api/content/")
                                .split('?')
                                .next()
                                .unwrap_or("")
                                .to_string();
                            single_page(&id, "DEV").to_string()
                        } else if path.starts_with("/wiki/rest/api/content") {
                            if path.contains("start=2") {
                                space_page_2().to_string()
                            } else {
                                space_page_1().to_string()
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
            _shutdown: tx,
        }
    }

    #[test]
    fn space_loader_paginates_and_returns_one_doc_per_page() {
        let srv = spawn_fake();
        let loader = ConfluenceLoader::from_space_bearer(
            &srv.url, "pat-token", "DEV",
        );
        let docs = loader.load().unwrap();
        // 2 + 1 = 3 pages.
        assert_eq!(docs.len(), 3);
        // HTML stripped to plain text.
        assert!(docs[0].content.contains("Body of"));
        assert!(docs[0].content.contains("first"));
        assert!(docs[1].content.contains("Heading"));
        assert!(docs[2].content.contains("Third"));
    }

    #[test]
    fn metadata_includes_id_title_space_key_version_source() {
        let srv = spawn_fake();
        let loader = ConfluenceLoader::from_space_bearer(
            &srv.url, "t", "DEV",
        );
        let docs = loader.load().unwrap();
        let d0 = &docs[0];
        assert_eq!(d0.metadata["page_id"].as_str(), Some("100"));
        assert_eq!(d0.metadata["title"].as_str(), Some("First Page"));
        assert_eq!(d0.metadata["space_key"].as_str(), Some("DEV"));
        assert_eq!(d0.metadata["version"].as_i64(), Some(3));
        assert_eq!(d0.metadata["source"].as_str(), Some("confluence:DEV:100"));
        assert_eq!(d0.id.as_deref(), Some("100"));
    }

    #[test]
    fn cloud_auth_uses_basic_auth_header() {
        let srv = spawn_fake();
        let loader = ConfluenceLoader::from_space_cloud(
            &srv.url, "ada@example.com", "atlassian-token", "DEV",
        );
        let _ = loader.load().unwrap();
        let auth = srv.seen_auth.lock().unwrap().clone();
        // base64("ada@example.com:atlassian-token") = YWRhQGV4YW1wbGUuY29tOmF0bGFzc2lhbi10b2tlbg==
        for a in &auth {
            let a = a.as_deref().unwrap_or("");
            assert!(a.starts_with("Basic "), "expected Basic auth, got: {a}");
            assert_eq!(a, "Basic YWRhQGV4YW1wbGUuY29tOmF0bGFzc2lhbi10b2tlbg==");
        }
    }

    #[test]
    fn bearer_auth_uses_bearer_header() {
        let srv = spawn_fake();
        let loader = ConfluenceLoader::from_space_bearer(
            &srv.url, "pat-abc123", "DEV",
        );
        let _ = loader.load().unwrap();
        let auth = srv.seen_auth.lock().unwrap().clone();
        for a in &auth {
            assert_eq!(a.as_deref(), Some("Bearer pat-abc123"));
        }
    }

    #[test]
    fn from_pages_fetches_each_page_by_id() {
        let srv = spawn_fake();
        let loader = ConfluenceLoader::from_pages_bearer(
            &srv.url, "t",
            vec!["200".into(), "201".into()],
        );
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].id.as_deref(), Some("200"));
        assert_eq!(docs[1].id.as_deref(), Some("201"));
        // space_key pulled from the per-page `space.key` expansion.
        assert_eq!(docs[0].metadata["space_key"].as_str(), Some("DEV"));
    }

    #[test]
    fn max_pages_cap_truncates_space_results() {
        let srv = spawn_fake();
        let loader = ConfluenceLoader::from_space_bearer(&srv.url, "t", "DEV")
            .with_max_pages(Some(1));
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[test]
    fn space_key_and_page_id_are_url_encoded_in_path() {
        // Space keys with special chars (rare but allowed) must be encoded.
        let srv = spawn_fake();
        let loader = ConfluenceLoader::from_space_bearer(&srv.url, "t", "DEV TEAM");
        let _ = loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        let first = paths.iter().find(|p| p.starts_with("/wiki/rest/api/content?")).unwrap();
        assert!(first.contains("spaceKey=DEV%20TEAM"), "got: {first}");
    }

    #[test]
    fn pagination_follows_links_next_until_exhausted() {
        let srv = spawn_fake();
        let loader = ConfluenceLoader::from_space_bearer(&srv.url, "t", "DEV");
        let _ = loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        // Both start=0 (implicit) and start=2 pages hit.
        let has_first = paths.iter().any(|p| p.contains("start=0") || (p.starts_with("/wiki/rest/api/content?") && !p.contains("start=2")));
        let has_second = paths.iter().any(|p| p.contains("start=2"));
        assert!(has_first);
        assert!(has_second);
    }

    #[test]
    fn empty_body_page_is_skipped() {
        // Page with no body.storage.value returns None from page_to_document
        // → skipped. Verify by constructing a page object directly.
        let loader = ConfluenceLoader::from_space_bearer("http://x", "t", "DEV");
        let empty_body = json!({
            "id": "999",
            "title": "Archived",
            "body": {"storage": {"value": ""}},
            "version": {"number": 1}
        });
        assert!(loader.page_to_document(&empty_body, "DEV").is_none());

        let missing_body = json!({
            "id": "998",
            "title": "Weird",
            "version": {"number": 1}
        });
        assert!(loader.page_to_document(&missing_body, "DEV").is_none());
    }

    #[test]
    fn xhtml_body_is_stripped_to_plain_text() {
        let loader = ConfluenceLoader::from_space_bearer("http://x", "t", "DEV");
        let page = json!({
            "id": "1",
            "title": "T",
            "body": {"storage": {"value": "<h2>Title</h2><p>Line 1.</p><p>Line &amp; 2.</p>"}},
            "version": {"number": 1}
        });
        let d = loader.page_to_document(&page, "DEV").unwrap();
        assert!(d.content.contains("Title"));
        assert!(d.content.contains("Line 1."));
        assert!(d.content.contains("Line & 2."), "entity should decode: {:?}", d.content);
        // Tags stripped.
        assert!(!d.content.contains("<p>"));
    }
}
