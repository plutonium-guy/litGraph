//! Gmail loader — pull messages via the Gmail REST API.
//!
//! Direct LangChain `GmailLoader` parity-in-spirit. Pure-Rust via existing
//! `reqwest::blocking`. Auth is OAuth2 Bearer — caller obtains the access
//! token externally (via Google's auth libs, workload identity, gcloud
//! application-default credentials, etc) and passes it in. Adapter does
//! NOT mint or refresh tokens.
//!
//! # Flow
//!
//! 1. `GET /gmail/v1/users/{userId}/messages?q=...&maxResults=100` —
//!    returns a page of message IDs. Follow `nextPageToken` until exhausted
//!    (or `max_messages` cap hit).
//! 2. Per ID: `GET /gmail/v1/users/{userId}/messages/{id}?format=full`
//!    → full message with headers + multipart body.
//! 3. Text extraction: prefer `text/plain` parts (recursively walk
//!    `payload.parts`); fall back to HTML stripped via `html::strip_html`
//!    when no plain part exists. base64url-decode the `body.data` field.
//!
//! # Metadata per document
//!
//! - `id`, `thread_id` — stable message IDs
//! - `from`, `to`, `subject`, `date` — from the RFC-2822 headers
//! - `labels` — comma-joined label IDs (e.g. `INBOX,IMPORTANT`)
//! - `snippet` — Gmail's server-side preview (always present; cheap to
//!    populate even when body extraction fails)
//! - `source` — `"gmail:{user_id}/{message_id}"`

use std::time::Duration;

use base64::Engine;
use litgraph_core::Document;
use serde_json::Value;

use crate::html::{decode_entities, strip_html};
use crate::{Loader, LoaderError, LoaderResult};

const GMAIL_API: &str = "https://gmail.googleapis.com";

pub struct GmailLoader {
    pub access_token: String,
    pub user_id: String,
    pub base_url: String,
    pub timeout: Duration,
    pub query: Option<String>,
    pub max_messages: Option<usize>,
    pub include_body: bool,
}

impl GmailLoader {
    /// `access_token` is a Google OAuth2 token with `gmail.readonly` scope
    /// (or higher). `user_id` defaults to `"me"` — the authenticated user;
    /// admin-scope callers can pass any mailbox.
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            user_id: "me".into(),
            base_url: GMAIL_API.into(),
            timeout: Duration::from_secs(30),
            query: None,
            max_messages: Some(100),
            include_body: true,
        }
    }

    pub fn with_user_id(mut self, u: impl Into<String>) -> Self {
        self.user_id = u.into();
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Gmail search syntax — e.g. `"from:alice label:inbox after:2024/01/01"`.
    /// Forwarded verbatim as `?q=` parameter; Gmail applies its usual ranking.
    pub fn with_query(mut self, q: impl Into<String>) -> Self {
        self.query = Some(q.into());
        self
    }

    pub fn with_max_messages(mut self, n: Option<usize>) -> Self {
        self.max_messages = n;
        self
    }

    /// When `false`, only load headers + snippet. Use for cheap "list
    /// recent threads" indexing; flip to `true` for full-body RAG.
    pub fn with_include_body(mut self, b: bool) -> Self {
        self.include_body = b;
        self
    }

    fn client(&self) -> LoaderResult<reqwest::blocking::Client> {
        reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(LoaderError::from)
    }

    fn authed(&self, b: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
        b.bearer_auth(&self.access_token)
    }

    /// Fetch one page of message IDs. Returns `(ids, next_page_token)`.
    fn fetch_id_page(
        &self,
        client: &reqwest::blocking::Client,
        page_token: Option<&str>,
    ) -> LoaderResult<(Vec<String>, Option<String>)> {
        let mut url = format!(
            "{}/gmail/v1/users/{}/messages?maxResults=100",
            self.base_url.trim_end_matches('/'),
            urlencode(&self.user_id),
        );
        if let Some(q) = &self.query {
            url.push_str(&format!("&q={}", urlencode(q)));
        }
        if let Some(pt) = page_token {
            url.push_str(&format!("&pageToken={}", urlencode(pt)));
        }
        let resp = self.authed(client.get(&url)).send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "gmail messages.list {status}: {body}"
            )));
        }
        let v: Value = resp.json()?;
        let ids: Vec<String> = v
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let next = v
            .get("nextPageToken")
            .and_then(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from);
        Ok((ids, next))
    }

    fn fetch_message(
        &self,
        client: &reqwest::blocking::Client,
        id: &str,
    ) -> LoaderResult<Value> {
        let format_arg = if self.include_body { "full" } else { "metadata" };
        let url = format!(
            "{}/gmail/v1/users/{}/messages/{}?format={}",
            self.base_url.trim_end_matches('/'),
            urlencode(&self.user_id),
            urlencode(id),
            format_arg,
        );
        let resp = self.authed(client.get(&url)).send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "gmail messages.get {id} {status}: {body}"
            )));
        }
        resp.json().map_err(LoaderError::from)
    }

    /// Recursively walk `payload.parts` (and nested multiparts) to find the
    /// first `text/plain` part. Falls back to the first `text/html` part
    /// stripped via `html::strip_html` if no plain part exists. Returns
    /// `("", false)` when no textual content is present at all (e.g.
    /// image/attachment-only messages).
    fn extract_body(payload: &Value) -> (String, bool) {
        fn walk(part: &Value, prefer_plain: bool) -> Option<String> {
            let mime = part
                .get("mimeType")
                .and_then(|s| s.as_str())
                .unwrap_or("");
            // If current part is the desired MIME + has body.data, decode.
            if (prefer_plain && mime == "text/plain")
                || (!prefer_plain && mime == "text/html")
            {
                if let Some(data) = part.pointer("/body/data").and_then(|s| s.as_str()) {
                    if let Some(decoded) = decode_b64url(data) {
                        return Some(decoded);
                    }
                }
            }
            // Recurse into nested parts (multipart/mixed, multipart/alternative).
            if let Some(parts) = part.get("parts").and_then(|p| p.as_array()) {
                for p in parts {
                    if let Some(s) = walk(p, prefer_plain) {
                        return Some(s);
                    }
                }
            }
            None
        }

        // Top-level payload might itself have body.data (small plain-text
        // messages with no multipart wrapping).
        let top_mime = payload
            .get("mimeType")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        if top_mime == "text/plain" {
            if let Some(data) = payload.pointer("/body/data").and_then(|s| s.as_str()) {
                if let Some(decoded) = decode_b64url(data) {
                    return (decoded, false);
                }
            }
        }

        // Prefer plain; fall back to HTML stripped.
        if let Some(plain) = walk(payload, true) {
            return (plain, false);
        }
        if let Some(html) = walk(payload, false) {
            let text = decode_entities(&strip_html(&html, true));
            return (text, true);
        }
        (String::new(), false)
    }

    fn message_to_document(&self, v: &Value) -> Option<Document> {
        let id = v.get("id").and_then(|s| s.as_str())?.to_string();
        let thread_id = v
            .get("threadId")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let snippet = v
            .get("snippet")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let labels: Vec<String> = v
            .get("labelIds")
            .and_then(|l| l.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Pull headers via a small scan — payload.headers is an array of
        // {name, value} dicts.
        let headers = v
            .pointer("/payload/headers")
            .and_then(|h| h.as_array())
            .cloned()
            .unwrap_or_default();
        let hget = |name: &str| -> Option<String> {
            headers.iter().find_map(|h| {
                let n = h.get("name").and_then(|s| s.as_str())?;
                if n.eq_ignore_ascii_case(name) {
                    h.get("value").and_then(|s| s.as_str()).map(String::from)
                } else {
                    None
                }
            })
        };
        let from = hget("From").unwrap_or_default();
        let to = hget("To").unwrap_or_default();
        let subject = hget("Subject").unwrap_or_default();
        let date = hget("Date").unwrap_or_default();

        // Extract body if include_body=true (else fall back to snippet).
        let content = if self.include_body {
            if let Some(payload) = v.get("payload") {
                let (body, _from_html) = Self::extract_body(payload);
                if body.is_empty() { snippet.clone() } else { body }
            } else {
                snippet.clone()
            }
        } else {
            snippet.clone()
        };

        // Prefix subject as markdown H1 so the LLM sees the most-salient
        // field first — matches how we format GitHub issues.
        let final_content = if subject.is_empty() {
            content
        } else {
            format!("# {subject}\n\n{content}")
        };

        let mut d = Document::new(final_content);
        d.id = Some(format!("gmail:{}/{}", self.user_id, id));
        d.metadata.insert("message_id".into(), Value::String(id.clone()));
        if !thread_id.is_empty() {
            d.metadata
                .insert("thread_id".into(), Value::String(thread_id));
        }
        if !from.is_empty() {
            d.metadata.insert("from".into(), Value::String(from));
        }
        if !to.is_empty() {
            d.metadata.insert("to".into(), Value::String(to));
        }
        if !subject.is_empty() {
            d.metadata.insert("subject".into(), Value::String(subject));
        }
        if !date.is_empty() {
            d.metadata.insert("date".into(), Value::String(date));
        }
        if !labels.is_empty() {
            d.metadata
                .insert("labels".into(), Value::String(labels.join(",")));
        }
        d.metadata.insert("snippet".into(), Value::String(snippet));
        d.metadata.insert(
            "source".into(),
            Value::String(format!("gmail:{}/{}", self.user_id, id)),
        );
        Some(d)
    }
}

impl Loader for GmailLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.client()?;
        let mut docs: Vec<Document> = Vec::new();
        let mut page_token: Option<String> = None;
        loop {
            let (ids, next) = self.fetch_id_page(&client, page_token.as_deref())?;
            for id in ids {
                let msg = self.fetch_message(&client, &id)?;
                if let Some(d) = self.message_to_document(&msg) {
                    docs.push(d);
                    if let Some(cap) = self.max_messages {
                        if docs.len() >= cap {
                            return Ok(docs);
                        }
                    }
                }
            }
            match next {
                Some(t) => page_token = Some(t),
                None => break,
            }
        }
        Ok(docs)
    }
}

/// Gmail encodes `body.data` as base64url (RFC 4648 section 5). Decode with
/// URL_SAFE + no padding; silently return None on failure (binary bodies
/// without textual content).
fn decode_b64url(s: &str) -> Option<String> {
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(cleaned.as_bytes())
        .or_else(|_| {
            base64::engine::general_purpose::URL_SAFE
                .decode(cleaned.as_bytes())
        })
        .ok()?;
    String::from_utf8(bytes).ok()
}

/// Tiny URL-encoder for query params. Preserves unreserved chars; encodes
/// everything else.
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

    fn b64url(s: &str) -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(s.as_bytes())
    }

    fn list_page_1() -> Value {
        json!({
            "messages": [
                {"id": "msg1", "threadId": "thr1"},
                {"id": "msg2", "threadId": "thr2"},
            ],
            "nextPageToken": "PAGE2",
            "resultSizeEstimate": 3
        })
    }

    fn list_page_2() -> Value {
        json!({
            "messages": [
                {"id": "msg3", "threadId": "thr3"},
            ],
            "resultSizeEstimate": 3
        })
    }

    fn message_full(id: &str, subject: &str, body: &str) -> Value {
        json!({
            "id": id,
            "threadId": format!("thr_{id}"),
            "labelIds": ["INBOX", "IMPORTANT"],
            "snippet": format!("Preview of {id}"),
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "To", "value": "bob@example.com"},
                    {"name": "Subject", "value": subject},
                    {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                ],
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": b64url(body)}
                    },
                    {
                        "mimeType": "text/html",
                        "body": {"data": b64url(&format!("<p>{body}</p>"))}
                    }
                ]
            }
        })
    }

    fn spawn_fake() -> FakeServer {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen_paths: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_auth: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        listener.set_nonblocking(true).unwrap();
        let pw = seen_paths.clone();
        let aw = seen_auth.clone();
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
                        pw.lock().unwrap().push(path.clone());
                        let mut auth = None;
                        for line in req.lines().skip(1) {
                            if let Some((k, v)) = line.split_once(':') {
                                if k.trim().eq_ignore_ascii_case("authorization") {
                                    auth = Some(v.trim().to_string());
                                }
                            }
                        }
                        aw.lock().unwrap().push(auth);
                        let body = if path.starts_with("/gmail/v1/users/me/messages/") {
                            let id = path
                                .trim_start_matches("/gmail/v1/users/me/messages/")
                                .split('?')
                                .next()
                                .unwrap_or("")
                                .to_string();
                            message_full(&id, &format!("Subject {id}"), &format!("Body of {id}"))
                                .to_string()
                        } else if path.starts_with("/gmail/v1/users/me/messages") {
                            if path.contains("pageToken=PAGE2") {
                                list_page_2().to_string()
                            } else {
                                list_page_1().to_string()
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
    fn loader_paginates_and_returns_one_doc_per_message() {
        let srv = spawn_fake();
        let loader = GmailLoader::new("ya29.test_token").with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 3);
        // Subject formatted as markdown H1.
        assert!(docs[0].content.starts_with("# Subject msg1"));
        assert!(docs[0].content.contains("Body of msg1"));
    }

    #[test]
    fn metadata_captures_headers_labels_snippet_and_ids() {
        let srv = spawn_fake();
        let loader = GmailLoader::new("t").with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        let d = &docs[0];
        assert_eq!(d.metadata["message_id"].as_str(), Some("msg1"));
        assert_eq!(d.metadata["thread_id"].as_str(), Some("thr_msg1"));
        assert_eq!(d.metadata["from"].as_str(), Some("alice@example.com"));
        assert_eq!(d.metadata["to"].as_str(), Some("bob@example.com"));
        assert_eq!(d.metadata["subject"].as_str(), Some("Subject msg1"));
        assert_eq!(d.metadata["labels"].as_str(), Some("INBOX,IMPORTANT"));
        assert!(d.metadata["snippet"].as_str().unwrap().contains("Preview"));
        assert_eq!(d.metadata["source"].as_str(), Some("gmail:me/msg1"));
        assert_eq!(d.id.as_deref(), Some("gmail:me/msg1"));
    }

    #[test]
    fn auth_bearer_set_on_both_list_and_get_requests() {
        let srv = spawn_fake();
        let loader = GmailLoader::new("ya29.SECRET").with_base_url(&srv.url);
        loader.load().unwrap();
        let auth = srv.seen_auth.lock().unwrap().clone();
        for a in &auth {
            assert_eq!(a.as_deref(), Some("Bearer ya29.SECRET"));
        }
    }

    #[test]
    fn page_token_forwarded_for_pagination() {
        let srv = spawn_fake();
        let loader = GmailLoader::new("t").with_base_url(&srv.url);
        loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        // Page 1: no pageToken. Page 2: pageToken=PAGE2.
        assert!(paths.iter().any(|p| p.contains("pageToken=PAGE2")));
    }

    #[test]
    fn query_parameter_forwarded_to_list_endpoint() {
        let srv = spawn_fake();
        let loader = GmailLoader::new("t")
            .with_base_url(&srv.url)
            .with_query("from:alice after:2024/01/01");
        loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        let list_path = paths
            .iter()
            .find(|p| p.starts_with("/gmail/v1/users/me/messages?"))
            .unwrap();
        // `:` and `/` and space URL-encoded.
        assert!(list_path.contains("q="), "got: {list_path}");
        assert!(list_path.contains("alice"), "got: {list_path}");
        assert!(list_path.contains("%20") || list_path.contains("+"), "got: {list_path}");
    }

    #[test]
    fn max_messages_cap_truncates_result() {
        let srv = spawn_fake();
        let loader = GmailLoader::new("t")
            .with_base_url(&srv.url)
            .with_max_messages(Some(1));
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[test]
    fn include_body_false_uses_metadata_format_and_returns_snippet() {
        let srv = spawn_fake();
        let loader = GmailLoader::new("t")
            .with_base_url(&srv.url)
            .with_include_body(false);
        let docs = loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        // Each GET uses format=metadata.
        assert!(paths.iter().any(|p| p.contains("format=metadata")));
        // Content = snippet when body skipped.
        assert!(docs[0].content.contains("Preview of msg1"));
    }

    #[test]
    fn extract_body_prefers_text_plain_over_html() {
        let payload = json!({
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": b64url("plain text here")}},
                {"mimeType": "text/html", "body": {"data": b64url("<p>html version</p>")}},
            ]
        });
        let (body, from_html) = GmailLoader::extract_body(&payload);
        assert_eq!(body, "plain text here");
        assert!(!from_html);
    }

    #[test]
    fn extract_body_falls_back_to_html_stripped_when_no_plain_part() {
        let payload = json!({
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/html",
                 "body": {"data": b64url("<h1>Subject</h1><p>Body &amp; rest.</p>")}}
            ]
        });
        let (body, from_html) = GmailLoader::extract_body(&payload);
        assert!(from_html);
        assert!(body.contains("Subject"));
        assert!(body.contains("Body & rest."));  // entity decoded
        assert!(!body.contains("<h1>"));  // stripped
    }

    #[test]
    fn extract_body_walks_nested_multipart() {
        // multipart/mixed containing a multipart/alternative containing
        // the actual text/plain. Real Gmail often nests like this when
        // attachments are present.
        let payload = json!({
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/plain",
                         "body": {"data": b64url("nested plain")}}
                    ]
                },
                {"mimeType": "application/pdf", "body": {"attachmentId": "att1"}}
            ]
        });
        let (body, _) = GmailLoader::extract_body(&payload);
        assert_eq!(body, "nested plain");
    }

    #[test]
    fn extract_body_handles_top_level_plain_with_no_parts() {
        // Small message, no multipart wrapping — body.data lives on the
        // top-level payload.
        let payload = json!({
            "mimeType": "text/plain",
            "body": {"data": b64url("simple note")}
        });
        let (body, _) = GmailLoader::extract_body(&payload);
        assert_eq!(body, "simple note");
    }

    #[test]
    fn extract_body_returns_empty_for_attachment_only_message() {
        let payload = json!({
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "application/pdf", "body": {"attachmentId": "a1"}},
                {"mimeType": "image/png", "body": {"attachmentId": "a2"}}
            ]
        });
        let (body, _) = GmailLoader::extract_body(&payload);
        assert_eq!(body, "");
    }

    #[test]
    fn decode_b64url_handles_both_padded_and_unpadded_input() {
        // Gmail typically emits unpadded base64url; the decoder must accept
        // both forms (standard library's NO_PAD engine rejects padded input,
        // so we fall back to padded decoder on failure).
        let encoded = b64url("hello world");  // no padding
        assert_eq!(decode_b64url(&encoded), Some("hello world".to_string()));
        let padded = base64::engine::general_purpose::URL_SAFE.encode(b"hello world");
        assert_eq!(decode_b64url(&padded), Some("hello world".to_string()));
    }

    #[test]
    fn decode_b64url_strips_whitespace_before_decoding() {
        // Some mail systems wrap base64 in the API response at 76 chars.
        // Strip \n / \r / spaces before feeding to decoder.
        let encoded = b64url("the quick brown fox");
        let wrapped = format!("{}\n{}", &encoded[..5], &encoded[5..]);
        assert_eq!(decode_b64url(&wrapped), Some("the quick brown fox".to_string()));
    }
}
