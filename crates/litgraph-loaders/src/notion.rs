//! Notion loader — pull pages from a Notion database OR a list of page ids.
//!
//! Direct LangChain `NotionDirectoryLoader` / `NotionDBLoader` parity in
//! spirit (we don't read filesystem-exported markdown — the API is the
//! source of truth + always fresh).
//!
//! # Auth
//!
//! Pass an "internal integration secret" from a Notion integration the
//! page/database has been shared with. Two headers per request:
//! - `Authorization: Bearer <secret>`
//! - `Notion-Version: 2022-06-28` (pinned; bumping requires testing).
//!
//! # What we extract
//!
//! For each page: `paragraph` / `heading_1..3` / `bulleted_list_item` /
//! `numbered_list_item` / `to_do` / `quote` / `callout` / `code` blocks.
//! Joined with `\n` for paragraph breaks. Other block types (image, video,
//! database refs) are skipped — they carry no extractable text. Page title
//! lands in `metadata["title"]`.
//!
//! # Pagination
//!
//! Notion paginates database queries + block children at 100 per page.
//! We follow `has_more` + `next_cursor` until exhausted.

use std::time::Duration;

use litgraph_core::Document;
use serde_json::{Value, json};

use crate::{Loader, LoaderError, LoaderResult};

const NOTION_API: &str = "https://api.notion.com/v1";
const NOTION_VERSION: &str = "2022-06-28";

#[derive(Debug, Clone)]
enum Source {
    Database(String),
    Pages(Vec<String>),
}

pub struct NotionLoader {
    pub api_key: String,
    pub base_url: String,
    pub timeout: Duration,
    source: Source,
    /// Hard cap on total pages loaded — protect against accidental scrapes
    /// of huge databases. None = no cap. Default 1000.
    pub max_pages: Option<usize>,
}

impl NotionLoader {
    /// Load pages by enumerating a Notion database.
    pub fn from_database(api_key: impl Into<String>, database_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: NOTION_API.into(),
            timeout: Duration::from_secs(30),
            source: Source::Database(database_id.into()),
            max_pages: Some(1000),
        }
    }

    /// Load specific pages by id.
    pub fn from_pages(api_key: impl Into<String>, page_ids: Vec<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: NOTION_API.into(),
            timeout: Duration::from_secs(30),
            source: Source::Pages(page_ids),
            max_pages: None,
        }
    }

    /// Override the API base URL — used by tests pointing at fake servers,
    /// and by self-hosted Notion-compat alternatives.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_max_pages(mut self, n: Option<usize>) -> Self {
        self.max_pages = n;
        self
    }

    fn client(&self) -> LoaderResult<reqwest::blocking::Client> {
        reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(LoaderError::from)
    }

    fn req(
        &self,
        client: &reqwest::blocking::Client,
        method: reqwest::Method,
        path: &str,
    ) -> reqwest::blocking::RequestBuilder {
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), path);
        client
            .request(method, url)
            .bearer_auth(&self.api_key)
            .header("Notion-Version", NOTION_VERSION)
    }

    /// Enumerate all page ids in a database, following pagination.
    fn list_database_pages(
        &self,
        client: &reqwest::blocking::Client,
        db_id: &str,
    ) -> LoaderResult<Vec<(String, Option<String>)>> {
        let mut out: Vec<(String, Option<String>)> = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let mut body = json!({ "page_size": 100 });
            if let Some(c) = &cursor {
                body["start_cursor"] = json!(c);
            }
            let resp = self
                .req(client, reqwest::Method::POST, &format!("/databases/{db_id}/query"))
                .json(&body)
                .send()?;
            if !resp.status().is_success() {
                let status = resp.status();
                let txt = resp.text().unwrap_or_default();
                return Err(LoaderError::Other(format!(
                    "notion db query {status}: {txt}"
                )));
            }
            let v: Value = resp.json()?;
            if let Some(arr) = v.get("results").and_then(|r| r.as_array()) {
                for page in arr {
                    if let Some(id) = page.get("id").and_then(|s| s.as_str()) {
                        let title = extract_page_title(page);
                        out.push((id.to_string(), title));
                        if let Some(cap) = self.max_pages {
                            if out.len() >= cap {
                                return Ok(out);
                            }
                        }
                    }
                }
            }
            let has_more = v.get("has_more").and_then(|b| b.as_bool()).unwrap_or(false);
            if !has_more {
                break;
            }
            cursor = v
                .get("next_cursor")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string());
            if cursor.is_none() {
                break;
            }
        }
        Ok(out)
    }

    /// Fetch a single page's blocks (children), follow pagination, extract
    /// text. Public-but-internal so tests can hit it without an enclosing
    /// database loader.
    pub(crate) fn fetch_page_text(
        &self,
        client: &reqwest::blocking::Client,
        page_id: &str,
    ) -> LoaderResult<String> {
        let mut text = String::new();
        let mut cursor: Option<String> = None;
        loop {
            let mut path = format!("/blocks/{page_id}/children?page_size=100");
            if let Some(c) = &cursor {
                path.push_str(&format!("&start_cursor={}", urlencode(c)));
            }
            let resp = self.req(client, reqwest::Method::GET, &path).send()?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                return Err(LoaderError::Other(format!(
                    "notion blocks {status}: {body}"
                )));
            }
            let v: Value = resp.json()?;
            if let Some(arr) = v.get("results").and_then(|r| r.as_array()) {
                for block in arr {
                    if let Some(t) = extract_block_text(block) {
                        if !text.is_empty() { text.push('\n'); }
                        text.push_str(&t);
                    }
                }
            }
            let has_more = v.get("has_more").and_then(|b| b.as_bool()).unwrap_or(false);
            if !has_more {
                break;
            }
            cursor = v.get("next_cursor").and_then(|c| c.as_str()).map(String::from);
            if cursor.is_none() {
                break;
            }
        }
        Ok(text)
    }

    /// Fetch the page object itself (for title metadata when loading by
    /// page_id directly, since the database query gives us titles for free).
    fn fetch_page_title(
        &self,
        client: &reqwest::blocking::Client,
        page_id: &str,
    ) -> Option<String> {
        let resp = self
            .req(client, reqwest::Method::GET, &format!("/pages/{page_id}"))
            .send()
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let v: Value = resp.json().ok()?;
        extract_page_title(&v)
    }
}

impl Loader for NotionLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.client()?;
        let mut docs = Vec::new();
        let pages: Vec<(String, Option<String>)> = match &self.source {
            Source::Database(db_id) => self.list_database_pages(&client, db_id)?,
            Source::Pages(ids) => {
                let mut out = Vec::new();
                for id in ids {
                    if let Some(cap) = self.max_pages {
                        if out.len() >= cap { break; }
                    }
                    let title = self.fetch_page_title(&client, id);
                    out.push((id.clone(), title));
                }
                out
            }
        };
        for (page_id, title) in pages {
            let text = self.fetch_page_text(&client, &page_id)?;
            let mut d = Document::new(text);
            d.id = Some(page_id.clone());
            d.metadata.insert(
                "source".into(),
                Value::String(format!("notion:{page_id}")),
            );
            d.metadata.insert("page_id".into(), Value::String(page_id));
            if let Some(t) = title {
                d.metadata.insert("title".into(), Value::String(t));
            }
            docs.push(d);
        }
        Ok(docs)
    }
}

/// Extract the page's title from its `properties.{title-prop}.title[].plain_text`
/// (Notion stores titles as rich-text arrays). Returns None if no title.
fn extract_page_title(page: &Value) -> Option<String> {
    let props = page.get("properties")?.as_object()?;
    for (_, prop) in props {
        // Title properties have `type == "title"` and a `title` array of
        // rich-text spans.
        if prop.get("type").and_then(|t| t.as_str()) == Some("title") {
            let spans = prop.get("title")?.as_array()?;
            let combined: String = spans
                .iter()
                .filter_map(|s| s.get("plain_text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("");
            if !combined.is_empty() {
                return Some(combined);
            }
        }
    }
    None
}

/// Pull plain text from a Notion block, if it's a text-bearing type.
/// Returns None for image / divider / non-text blocks.
fn extract_block_text(block: &Value) -> Option<String> {
    let block_type = block.get("type")?.as_str()?;
    let inner = block.get(block_type)?;
    // Text-bearing blocks all have a `rich_text` array of spans.
    let rich_text = inner.get("rich_text")?.as_array()?;
    let mut text: String = rich_text
        .iter()
        .filter_map(|s| s.get("plain_text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("");
    // Lightweight markdown affordances for common block types so the LLM
    // sees structure (matches what LangChain's NotionDBLoader does).
    text = match block_type {
        "heading_1" => format!("# {text}"),
        "heading_2" => format!("## {text}"),
        "heading_3" => format!("### {text}"),
        "bulleted_list_item" => format!("- {text}"),
        "numbered_list_item" => format!("1. {text}"),
        "to_do" => {
            let checked = inner
                .get("checked")
                .and_then(|b| b.as_bool())
                .unwrap_or(false);
            format!("- [{}] {}", if checked { "x" } else { " " }, text)
        }
        "quote" => format!("> {text}"),
        "code" => {
            let lang = inner.get("language").and_then(|l| l.as_str()).unwrap_or("");
            format!("```{lang}\n{text}\n```")
        }
        _ => text,
    };
    if text.is_empty() { None } else { Some(text) }
}

/// Tiny URL-encoder for the cursor query string. Notion cursors are
/// opaque base64-ish but may contain `+` / `/` / `=` — encode them. We
/// avoid pulling `urlencoding` for one helper.
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
    use std::sync::{Arc as StdArc, Mutex};
    use std::thread;

    /// Tiny fake Notion HTTP server. One request, parse path, dispatch, return canned JSON.
    /// Captures the Authorization + Notion-Version headers per request.
    struct FakeServer {
        url: String,
        seen_paths: StdArc<Mutex<Vec<String>>>,
        seen_auth: StdArc<Mutex<Vec<Option<String>>>>,
        seen_version: StdArc<Mutex<Vec<Option<String>>>>,
        _shutdown: std::sync::mpsc::Sender<()>,
    }

    /// Build a minimal /databases/{id}/query reply: 2 pages with titles.
    fn fake_db_query() -> Value {
        json!({
            "results": [
                {
                    "id": "page-1",
                    "properties": {
                        "Name": {
                            "type": "title",
                            "title": [{"plain_text": "First Page"}]
                        }
                    }
                },
                {
                    "id": "page-2",
                    "properties": {
                        "Name": {
                            "type": "title",
                            "title": [{"plain_text": "Second Page"}]
                        }
                    }
                }
            ],
            "has_more": false,
            "next_cursor": null
        })
    }

    /// Build a /blocks/{id}/children reply with a heading + paragraph + bullet.
    fn fake_blocks() -> Value {
        json!({
            "results": [
                {
                    "type": "heading_1",
                    "heading_1": { "rich_text": [{"plain_text": "Hello"}] }
                },
                {
                    "type": "paragraph",
                    "paragraph": { "rich_text": [{"plain_text": "World"}] }
                },
                {
                    "type": "bulleted_list_item",
                    "bulleted_list_item": { "rich_text": [{"plain_text": "item one"}] }
                }
            ],
            "has_more": false,
            "next_cursor": null
        })
    }

    fn fake_page() -> Value {
        json!({
            "id": "page-X",
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": "Direct Page"}]
                }
            }
        })
    }

    fn spawn_fake() -> FakeServer {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen_paths: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
        let seen_auth: StdArc<Mutex<Vec<Option<String>>>> = StdArc::new(Mutex::new(Vec::new()));
        let seen_version: StdArc<Mutex<Vec<Option<String>>>> = StdArc::new(Mutex::new(Vec::new()));
        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel::<()>();
        listener.set_nonblocking(true).unwrap();
        let paths_w = seen_paths.clone();
        let auth_w = seen_auth.clone();
        let ver_w = seen_version.clone();
        thread::spawn(move || {
            use std::io::{Read, Write};
            loop {
                if shutdown_rx.try_recv().is_ok() { break; }
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).ok();
                        let mut buf = [0u8; 8192];
                        let n = stream.read(&mut buf).unwrap_or(0);
                        if n == 0 { continue; }
                        let req = String::from_utf8_lossy(&buf[..n]).to_string();
                        let first_line = req.lines().next().unwrap_or("");
                        let path = first_line
                            .split_whitespace()
                            .nth(1)
                            .unwrap_or("/")
                            .to_string();
                        paths_w.lock().unwrap().push(path.clone());
                        // Capture headers.
                        let mut auth = None;
                        let mut version = None;
                        for line in req.lines().skip(1) {
                            if let Some((k, v)) = line.split_once(':') {
                                let k_lc = k.trim().to_ascii_lowercase();
                                if k_lc == "authorization" {
                                    auth = Some(v.trim().to_string());
                                } else if k_lc == "notion-version" {
                                    version = Some(v.trim().to_string());
                                }
                            }
                        }
                        auth_w.lock().unwrap().push(auth);
                        ver_w.lock().unwrap().push(version);
                        // Dispatch.
                        let body = if path.contains("/databases/") && path.ends_with("/query") {
                            fake_db_query().to_string()
                        } else if path.contains("/blocks/") && path.contains("/children") {
                            fake_blocks().to_string()
                        } else if path.starts_with("/pages/") {
                            fake_page().to_string()
                        } else {
                            json!({"error": "unknown"}).to_string()
                        };
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(resp.as_bytes());
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
            _shutdown: shutdown_tx,
        }
    }

    #[test]
    fn database_loader_returns_one_doc_per_page_with_title_metadata() {
        let srv = spawn_fake();
        let loader = NotionLoader::from_database("secret_test", "db-123")
            .with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
        // First doc: title from db query, content from /blocks reply.
        assert_eq!(docs[0].metadata["title"], "First Page");
        assert_eq!(docs[0].metadata["page_id"], "page-1");
        assert!(docs[0].content.contains("# Hello"));
        assert!(docs[0].content.contains("World"));
        assert!(docs[0].content.contains("- item one"));
    }

    #[test]
    fn auth_and_version_headers_set_correctly() {
        let srv = spawn_fake();
        let loader = NotionLoader::from_database("secret_xyz", "db-x")
            .with_base_url(&srv.url);
        let _ = loader.load().unwrap();
        let auth = srv.seen_auth.lock().unwrap().clone();
        let ver = srv.seen_version.lock().unwrap().clone();
        // Every request carries Bearer + version 2022-06-28.
        for a in &auth {
            assert_eq!(a.as_deref(), Some("Bearer secret_xyz"));
        }
        for v in &ver {
            assert_eq!(v.as_deref(), Some("2022-06-28"));
        }
    }

    #[test]
    fn from_pages_loader_fetches_each_page_individually() {
        let srv = spawn_fake();
        let loader = NotionLoader::from_pages(
            "secret",
            vec!["page-A".into(), "page-B".into()],
        )
        .with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].metadata["page_id"], "page-A");
        assert_eq!(docs[1].metadata["page_id"], "page-B");
        // Title pulled from /pages/{id}.
        assert_eq!(docs[0].metadata["title"], "Direct Page");
    }

    #[test]
    fn extract_block_text_handles_known_block_types() {
        let to_do = json!({
            "type": "to_do",
            "to_do": {
                "rich_text": [{"plain_text": "task one"}],
                "checked": true
            }
        });
        assert_eq!(extract_block_text(&to_do).unwrap(), "- [x] task one");

        let unchecked = json!({
            "type": "to_do",
            "to_do": {
                "rich_text": [{"plain_text": "task two"}],
                "checked": false
            }
        });
        assert_eq!(extract_block_text(&unchecked).unwrap(), "- [ ] task two");

        let code = json!({
            "type": "code",
            "code": {
                "rich_text": [{"plain_text": "let x = 1;"}],
                "language": "rust"
            }
        });
        let s = extract_block_text(&code).unwrap();
        assert!(s.contains("```rust"));
        assert!(s.contains("let x = 1;"));

        let quote = json!({
            "type": "quote",
            "quote": { "rich_text": [{"plain_text": "noted"}] }
        });
        assert_eq!(extract_block_text(&quote).unwrap(), "> noted");
    }

    #[test]
    fn extract_block_text_skips_unknown_or_non_text_block_types() {
        let image = json!({
            "type": "image",
            "image": { "caption": [], "type": "external", "external": {"url": "x.com"} }
        });
        // No rich_text on image → None.
        assert!(extract_block_text(&image).is_none());

        let divider = json!({"type": "divider", "divider": {}});
        assert!(extract_block_text(&divider).is_none());
    }

    #[test]
    fn extract_page_title_returns_none_for_titleless_page() {
        let no_title = json!({ "id": "x", "properties": {} });
        assert!(extract_page_title(&no_title).is_none());

        let empty_title = json!({
            "id": "x",
            "properties": {
                "Name": { "type": "title", "title": [] }
            }
        });
        assert!(extract_page_title(&empty_title).is_none());
    }

    #[test]
    fn max_pages_cap_truncates_database_results() {
        let srv = spawn_fake();
        let loader = NotionLoader::from_database("secret", "db")
            .with_base_url(&srv.url)
            .with_max_pages(Some(1));  // fake server returns 2; cap at 1.
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[test]
    fn urlencode_handles_special_chars_in_cursor() {
        // Notion cursors can contain `+` / `/` / `=` (base64-ish).
        assert_eq!(urlencode("a+b/c=d"), "a%2Bb%2Fc%3Dd");
        assert_eq!(urlencode("plain123"), "plain123");
    }
}
