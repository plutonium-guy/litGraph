//! Outlook / Microsoft 365 mail loader. Pulls messages from a user's
//! mailbox via Microsoft Graph (`graph.microsoft.com`). Mirrors the
//! Gmail loader's surface so callers can swap providers with minimal
//! changes.
//!
//! # Auth
//!
//! Bearer token only — caller obtains it externally (via MSAL, Azure
//! workload identity, device-code flow, etc.) and passes it in. We
//! don't mint or refresh tokens.
//!
//! Required scopes: `Mail.Read` (read your own mail) or `Mail.Read.All`
//! (read any user's, admin scope only).
//!
//! # Flow
//!
//! 1. `GET /v1.0/me/messages?$top=N&$select=...&$search="..."` (or
//!    `/users/{id}/messages` when targeting another mailbox).
//! 2. Follow the `@odata.nextLink` URL each page until exhausted or
//!    [`max_messages`] is hit. Each page returns up to 1000 messages
//!    per Graph's `$top` cap; we default to 50 because most callers
//!    want a recent slice.
//! 3. Convert each message JSON to a [`Document`].
//!
//! # Body content type
//!
//! By default we send `Prefer: outlook.body-content-type="text"` so
//! the `body.content` field comes back as plain text (Graph would
//! otherwise return HTML). This means downstream embedders see prose,
//! not tags. Set [`with_html_body`] to disable the prefer header.
//!
//! # Document shape
//!
//! - `id` = Outlook message id (long base64url string, stable forever)
//! - `content` = subject + blank line + body text (concatenated so
//!   semantic search hits both header and body)
//! - `metadata`:
//!   - `subject`: subject line
//!   - `from`: sender email (best-effort extracted from the nested
//!     `from.emailAddress.address` field)
//!   - `from_name`: display name of sender
//!   - `to`: comma-joined recipient addresses
//!   - `received_date`: `receivedDateTime` ISO-8601
//!   - `conversation_id`: thread id for grouping
//!   - `is_read`: bool
//!   - `has_attachments`: bool
//!   - `web_link`: deep link into Outlook web
//!   - `source`: `"outlook"`

use std::time::Duration;

use litgraph_core::Document;
use serde_json::{json, Value};

use crate::{Loader, LoaderError, LoaderResult};

const GRAPH_API: &str = "https://graph.microsoft.com/v1.0";
const DEFAULT_MAX_MESSAGES: usize = 50;
const DEFAULT_PAGE_SIZE: usize = 50;

pub struct OutlookMessagesLoader {
    pub access_token: String,
    /// `me` (default) hits the authenticated user. Admin scope callers
    /// can target another mailbox by user id (a UUID) or UPN (the
    /// email-shaped principal name).
    pub user_id: String,
    pub max_messages: usize,
    pub page_size: usize,
    pub folder: Option<String>,
    pub search: Option<String>,
    pub filter: Option<String>,
    pub html_body: bool,
    pub timeout: Duration,
    pub user_agent: String,
}

impl OutlookMessagesLoader {
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            user_id: "me".into(),
            max_messages: DEFAULT_MAX_MESSAGES,
            page_size: DEFAULT_PAGE_SIZE,
            folder: None,
            search: None,
            filter: None,
            html_body: false,
            timeout: Duration::from_secs(30),
            user_agent: format!("litgraph-loaders/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    /// Target a specific user mailbox (admin scope required). Pass
    /// either a UUID or a UPN like `alice@contoso.com`.
    pub fn with_user_id(mut self, id: impl Into<String>) -> Self {
        self.user_id = id.into();
        self
    }

    /// Cap on the total number of messages returned across all pages.
    pub fn with_max_messages(mut self, n: usize) -> Self {
        self.max_messages = n;
        self
    }

    /// Per-page `$top` value (Graph caps at 1000). Smaller pages = more
    /// HTTP round-trips but tighter memory; larger = fewer calls but
    /// risk hitting Graph throttling.
    pub fn with_page_size(mut self, n: usize) -> Self {
        self.page_size = n.min(1000).max(1);
        self
    }

    /// Restrict to a named folder (`Inbox`, `Sent Items`, `Drafts`,
    /// `Deleted Items`, or a folder id). Default = whole mailbox.
    pub fn with_folder(mut self, folder: impl Into<String>) -> Self {
        self.folder = Some(folder.into());
        self
    }

    /// Free-text search query (Graph's `$search` parameter, KQL-flavoured).
    /// Mutually exclusive with `with_filter` per Graph's limits — we
    /// don't enforce that, but if you set both, Graph errors.
    pub fn with_search(mut self, q: impl Into<String>) -> Self {
        self.search = Some(q.into());
        self
    }

    /// OData `$filter` expression (e.g. `"isRead eq false"`). See
    /// Graph's filter docs for the supported operators.
    pub fn with_filter(mut self, f: impl Into<String>) -> Self {
        self.filter = Some(f.into());
        self
    }

    /// Disable the `Prefer: outlook.body-content-type="text"` header,
    /// returning HTML bodies. Use when downstream tooling prefers HTML
    /// (e.g. you'll feed it to a renderer or a richer parser).
    pub fn with_html_body(mut self) -> Self {
        self.html_body = true;
        self
    }

    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    fn endpoint(&self) -> String {
        std::env::var("LITGRAPH_GRAPH_API").unwrap_or_else(|_| GRAPH_API.into())
    }

    /// Build the first-page URL. Subsequent pages come from
    /// `@odata.nextLink` verbatim.
    fn first_page_url(&self) -> String {
        let base = self.endpoint();
        let user_segment = if self.user_id == "me" {
            "/me".to_string()
        } else {
            format!("/users/{}", self.user_id)
        };
        let folder_segment = match &self.folder {
            Some(f) => format!("/mailFolders/{}", encode_segment(f)),
            None => String::new(),
        };
        let mut url = format!("{base}{user_segment}{folder_segment}/messages?$top={}", self.page_size);
        // Limit the columns we pull. Reduces payload + Graph cost.
        let select = "id,subject,bodyPreview,body,from,toRecipients,receivedDateTime,\
                      conversationId,isRead,hasAttachments,webLink";
        url.push_str(&format!("&$select={}", encode_query(select)));
        if let Some(q) = &self.search {
            // `$search` requires double-quoted value per Graph spec.
            url.push_str(&format!("&$search={}", encode_query(&format!("\"{q}\""))));
        }
        if let Some(f) = &self.filter {
            url.push_str(&format!("&$filter={}", encode_query(f)));
        }
        url
    }

    fn build_client(&self) -> LoaderResult<reqwest::blocking::Client> {
        Ok(reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent(&self.user_agent)
            .build()?)
    }

    fn fetch_page(
        &self,
        client: &reqwest::blocking::Client,
        url: &str,
    ) -> LoaderResult<Value> {
        let mut req = client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.access_token));
        if !self.html_body {
            req = req.header("Prefer", r#"outlook.body-content-type="text""#);
        }
        // `$search` requires `ConsistencyLevel: eventual` per Graph spec.
        if self.search.is_some() {
            req = req.header("ConsistencyLevel", "eventual");
        }
        let resp = req.send()?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "outlook {status} {url}: {body}"
            )));
        }
        Ok(resp.json()?)
    }
}

impl Loader for OutlookMessagesLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        if self.max_messages == 0 {
            return Ok(Vec::new());
        }
        let client = self.build_client()?;
        let mut docs: Vec<Document> = Vec::new();
        let mut next_url = Some(self.first_page_url());
        while let Some(url) = next_url.take() {
            if docs.len() >= self.max_messages {
                break;
            }
            let body = self.fetch_page(&client, &url)?;
            let messages = body
                .get("value")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            for m in messages {
                if docs.len() >= self.max_messages {
                    break;
                }
                docs.push(message_to_document(&m));
            }
            next_url = body
                .get("@odata.nextLink")
                .and_then(|v| v.as_str())
                .map(String::from);
            if next_url.is_none() {
                break;
            }
        }
        Ok(docs)
    }
}

// ---- message → Document ----------------------------------------------------

pub(crate) fn message_to_document(msg: &Value) -> Document {
    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let subject = msg
        .get("subject")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let body_content = msg
        .get("body")
        .and_then(|b| b.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let body_preview = msg
        .get("bodyPreview")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    // Prefer full body; fall back to bodyPreview if Graph omitted body.
    let body_text = if !body_content.is_empty() {
        body_content
    } else {
        body_preview
    };

    // Concatenate subject + blank line + body so embeddings cover both.
    let mut content = String::new();
    if !subject.is_empty() {
        content.push_str(&subject);
        if !body_text.is_empty() {
            content.push_str("\n\n");
        }
    }
    content.push_str(body_text);

    let from_email = msg
        .get("from")
        .and_then(|f| f.get("emailAddress"))
        .and_then(|e| e.get("address"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let from_name = msg
        .get("from")
        .and_then(|f| f.get("emailAddress"))
        .and_then(|e| e.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let to_emails: Vec<String> = msg
        .get("toRecipients")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|r| {
                    r.get("emailAddress")
                        .and_then(|e| e.get("address"))
                        .and_then(|a| a.as_str())
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default();
    let to_joined = to_emails.join(", ");

    let received = msg
        .get("receivedDateTime")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let conversation_id = msg
        .get("conversationId")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let is_read = msg.get("isRead").and_then(|v| v.as_bool()).unwrap_or(false);
    let has_attachments = msg
        .get("hasAttachments")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let web_link = msg
        .get("webLink")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let mut d = Document::new(content);
    if !id.is_empty() {
        d = d.with_id(&id);
    }
    let mut put = |k: &str, v: Value| {
        d.metadata.insert(k.into(), v);
    };
    put("subject", Value::String(subject));
    put("from", Value::String(from_email));
    put("from_name", Value::String(from_name));
    put("to", Value::String(to_joined));
    put("received_date", Value::String(received));
    put("conversation_id", Value::String(conversation_id));
    put("is_read", Value::Bool(is_read));
    put("has_attachments", Value::Bool(has_attachments));
    put("web_link", Value::String(web_link));
    put("source", Value::String("outlook".into()));
    let _ = json!({}); // silence unused-import warning for json! in tests-only paths
    d
}

// ---- encoding helpers ------------------------------------------------------

/// URL-encode for path segments (folder names can contain spaces).
fn encode_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// URL-encode for query-string values. Same as `encode_segment` but
/// also escapes `&` and `=` to be safe inside `?key=value` pairs.
fn encode_query(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! Tests use a hand-rolled TCP fake server so we don't pull
    //! `wiremock` for one module — same pattern as discord.rs and
    //! http_prompt_hub.rs.

    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::sync::Arc;

    fn fake_server_sequence(responses: Vec<&'static str>) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel();
        let responses = Arc::new(responses);

        std::thread::spawn(move || {
            for body in responses.iter() {
                let (mut stream, _) = match listener.accept() {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let _ = tx.send(req);
                let response = format!(
                    "HTTP/1.1 200 OK\r\n\
                     Content-Type: application/json\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\
                     \r\n\
                     {}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (format!("http://127.0.0.1:{port}"), rx)
    }

    fn fake_status(status: u16, body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let response = format!(
                    "HTTP/1.1 {status} ERR\r\n\
                     Content-Type: application/json\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\
                     \r\n\
                     {}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    fn lock_env_to(value: &str) -> std::sync::MutexGuard<'static, ()> {
        use std::sync::Mutex;
        static GUARD: Mutex<()> = Mutex::new(());
        let g = GUARD.lock().unwrap();
        std::env::set_var("LITGRAPH_GRAPH_API", value);
        g
    }

    fn unlock_env() {
        std::env::remove_var("LITGRAPH_GRAPH_API");
    }

    fn sample_message(id: &str, subject: &str, body: &str) -> Value {
        json!({
            "id": id,
            "subject": subject,
            "bodyPreview": &body[..body.len().min(50)],
            "body": {"contentType": "text", "content": body},
            "from": {
                "emailAddress": {"name": "Alice", "address": "alice@example.com"}
            },
            "toRecipients": [
                {"emailAddress": {"name": "Bob", "address": "bob@example.com"}},
                {"emailAddress": {"name": "Carol", "address": "carol@example.com"}}
            ],
            "receivedDateTime": "2024-06-01T10:00:00Z",
            "conversationId": "thread-1",
            "isRead": false,
            "hasAttachments": false,
            "webLink": "https://outlook.office.com/mail/x"
        })
    }

    // ---- message_to_document ----

    #[test]
    fn message_to_document_concatenates_subject_and_body() {
        let m = sample_message("abc", "Hello", "Body text here.");
        let d = message_to_document(&m);
        assert_eq!(d.id.as_deref(), Some("abc"));
        assert!(d.content.contains("Hello"));
        assert!(d.content.contains("Body text here."));
        // Blank-line separator between header and body.
        assert!(d.content.contains("\n\n"));
    }

    #[test]
    fn message_to_document_extracts_from_to_metadata() {
        let m = sample_message("abc", "x", "y");
        let d = message_to_document(&m);
        assert_eq!(
            d.metadata.get("from").and_then(|v| v.as_str()),
            Some("alice@example.com")
        );
        assert_eq!(
            d.metadata.get("from_name").and_then(|v| v.as_str()),
            Some("Alice")
        );
        // Recipients comma-joined for cheap downstream filtering.
        assert_eq!(
            d.metadata.get("to").and_then(|v| v.as_str()),
            Some("bob@example.com, carol@example.com")
        );
    }

    #[test]
    fn message_to_document_falls_back_to_body_preview() {
        let m = json!({
            "id": "x",
            "subject": "S",
            "bodyPreview": "preview-only body",
            // body field omitted — Graph sometimes does this for very
            // large messages.
            "from": {"emailAddress": {"address": "a@x.com", "name": "A"}},
            "toRecipients": [],
            "receivedDateTime": "2024-01-01",
            "conversationId": "c",
            "isRead": true,
            "hasAttachments": false,
            "webLink": ""
        });
        let d = message_to_document(&m);
        assert!(d.content.contains("preview-only body"));
    }

    #[test]
    fn message_to_document_handles_missing_subject() {
        let m = json!({
            "id": "x",
            // No subject — system / draft messages can omit it.
            "body": {"contentType": "text", "content": "body only"},
            "from": {"emailAddress": {"address": "a@x.com"}},
            "toRecipients": [],
            "receivedDateTime": "",
            "conversationId": "",
            "isRead": false,
            "hasAttachments": false,
            "webLink": ""
        });
        let d = message_to_document(&m);
        // No leading newline gap when subject was missing.
        assert!(!d.content.starts_with("\n"));
        assert!(d.content.starts_with("body only"));
    }

    #[test]
    fn message_to_document_flags_attachments_and_read_state() {
        let m = json!({
            "id": "x",
            "subject": "S",
            "body": {"contentType": "text", "content": "B"},
            "from": {"emailAddress": {"address": "a@x.com"}},
            "toRecipients": [],
            "receivedDateTime": "2024-01-01",
            "conversationId": "c",
            "isRead": true,
            "hasAttachments": true,
            "webLink": "https://outlook"
        });
        let d = message_to_document(&m);
        assert_eq!(d.metadata.get("is_read").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            d.metadata.get("has_attachments").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(d.metadata.get("source").and_then(|v| v.as_str()), Some("outlook"));
    }

    // ---- URL building ----

    #[test]
    fn first_page_url_uses_me_by_default() {
        let l = OutlookMessagesLoader::new("token");
        let url = l.first_page_url();
        assert!(url.contains("/me/messages"), "{url}");
        assert!(url.contains("$top=50"), "{url}");
        assert!(url.contains("$select="), "{url}");
        assert!(!url.contains("$search="));
        assert!(!url.contains("$filter="));
    }

    #[test]
    fn first_page_url_with_user_id_uses_users_segment() {
        let l = OutlookMessagesLoader::new("token").with_user_id("alice@contoso.com");
        let url = l.first_page_url();
        assert!(url.contains("/users/alice@contoso.com/messages"), "{url}");
    }

    #[test]
    fn first_page_url_includes_folder_segment() {
        let l = OutlookMessagesLoader::new("token").with_folder("Inbox");
        let url = l.first_page_url();
        assert!(url.contains("/mailFolders/Inbox/messages"), "{url}");
    }

    #[test]
    fn first_page_url_url_encodes_folder_with_space() {
        let l = OutlookMessagesLoader::new("token").with_folder("Sent Items");
        let url = l.first_page_url();
        assert!(url.contains("/mailFolders/Sent%20Items/messages"), "{url}");
    }

    #[test]
    fn first_page_url_includes_search_with_quotes() {
        let l = OutlookMessagesLoader::new("token").with_search("project");
        let url = l.first_page_url();
        // $search requires double-quoted value, URL-encoded.
        assert!(url.contains("$search=%22project%22"), "{url}");
    }

    #[test]
    fn first_page_url_includes_filter() {
        let l = OutlookMessagesLoader::new("token").with_filter("isRead eq false");
        let url = l.first_page_url();
        assert!(url.contains("$filter=isRead%20eq%20false"), "{url}");
    }

    #[test]
    fn page_size_clamped_to_graph_max() {
        let l = OutlookMessagesLoader::new("token").with_page_size(99999);
        let url = l.first_page_url();
        assert!(url.contains("$top=1000"), "{url}");
    }

    #[test]
    fn page_size_clamped_to_minimum_one() {
        let l = OutlookMessagesLoader::new("token").with_page_size(0);
        let url = l.first_page_url();
        assert!(url.contains("$top=1"), "{url}");
    }

    // ---- end-to-end ----

    #[tokio::test(flavor = "current_thread")]
    async fn loads_single_page_with_bearer_auth_and_prefer_header() {
        let body = serde_json::to_string(&json!({
            "value": [
                sample_message("m1", "Hi", "Body 1"),
                sample_message("m2", "Hey", "Body 2"),
            ]
        }))
        .unwrap();
        let body: &'static str = Box::leak(body.into_boxed_str());

        let (base, rx) = fake_server_sequence(vec![body]);
        let _g = lock_env_to(&base);

        let docs = tokio::task::spawn_blocking(|| {
            OutlookMessagesLoader::new("FAKE_TOKEN").load()
        })
        .await
        .unwrap()
        .unwrap();

        assert_eq!(docs.len(), 2);
        assert!(docs[0].content.contains("Hi"));
        assert!(docs[1].content.contains("Hey"));

        let req = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(req.contains("/me/messages"), "{req}");
        assert!(
            req.to_lowercase().contains("authorization: bearer fake_token"),
            "{req}"
        );
        // Default body-content-type=text Prefer header attached.
        assert!(
            req.to_lowercase().contains(r#"prefer: outlook.body-content-type="text""#),
            "{req}"
        );
        unlock_env();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pagination_follows_odata_nextlink() {
        // Page 1 includes a nextLink; we point it at the SAME fake
        // server (which serves sequential responses).
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let next_link = format!("http://127.0.0.1:{port}/me/messages?$skip=2");
        let page1 = serde_json::to_string(&json!({
            "value": [
                sample_message("m1", "Subj 1", "B1"),
                sample_message("m2", "Subj 2", "B2"),
            ],
            "@odata.nextLink": next_link,
        }))
        .unwrap();
        let page2 = serde_json::to_string(&json!({
            "value": [sample_message("m3", "Subj 3", "B3")]
        }))
        .unwrap();
        let p1: &'static str = Box::leak(page1.into_boxed_str());
        let p2: &'static str = Box::leak(page2.into_boxed_str());
        let resps = Arc::new(vec![p1, p2]);
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            for body in resps.iter() {
                let (mut stream, _) = match listener.accept() {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let _ = tx.send(req);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\n\
                     Content-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });

        let _g = lock_env_to(&format!("http://127.0.0.1:{port}/v1.0"));

        let docs = tokio::task::spawn_blocking(|| {
            OutlookMessagesLoader::new("X").with_max_messages(100).load()
        })
        .await
        .unwrap()
        .unwrap();

        assert_eq!(docs.len(), 3);
        // First request must be the canonical first-page URL; second
        // must use the nextLink path verbatim.
        let req1 = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let req2 = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(req1.contains("/me/messages"), "{req1}");
        assert!(req2.contains("$skip=2"), "{req2}");
        unlock_env();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn max_messages_caps_pagination() {
        let body = serde_json::to_string(&json!({
            "value": [
                sample_message("m1", "S1", "B1"),
                sample_message("m2", "S2", "B2"),
                sample_message("m3", "S3", "B3"),
            ]
        }))
        .unwrap();
        let body: &'static str = Box::leak(body.into_boxed_str());

        let (base, _rx) = fake_server_sequence(vec![body]);
        let _g = lock_env_to(&base);

        let docs = tokio::task::spawn_blocking(|| {
            OutlookMessagesLoader::new("X").with_max_messages(2).load()
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(docs.len(), 2);
        unlock_env();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn search_attaches_consistency_level_header() {
        let body = serde_json::to_string(&json!({"value": []})).unwrap();
        let body: &'static str = Box::leak(body.into_boxed_str());
        let (base, rx) = fake_server_sequence(vec![body]);
        let _g = lock_env_to(&base);

        let _ = tokio::task::spawn_blocking(|| {
            OutlookMessagesLoader::new("X").with_search("invoice").load()
        })
        .await
        .unwrap()
        .unwrap();

        let req = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            req.to_lowercase().contains("consistencylevel: eventual"),
            "{req}"
        );
        unlock_env();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn html_body_disables_prefer_header() {
        let body = serde_json::to_string(&json!({"value": []})).unwrap();
        let body: &'static str = Box::leak(body.into_boxed_str());
        let (base, rx) = fake_server_sequence(vec![body]);
        let _g = lock_env_to(&base);

        let _ = tokio::task::spawn_blocking(|| {
            OutlookMessagesLoader::new("X").with_html_body().load()
        })
        .await
        .unwrap()
        .unwrap();

        let req = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            !req.to_lowercase().contains("prefer:"),
            "expected no Prefer header, got: {req}"
        );
        unlock_env();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn http_error_surfaces_clean_error() {
        let url = fake_status(401, r#"{"error": {"code": "InvalidAuthenticationToken"}}"#);
        let _g = lock_env_to(&url);

        let res = tokio::task::spawn_blocking(|| {
            OutlookMessagesLoader::new("BAD_TOKEN").load()
        })
        .await
        .unwrap();
        let err = res.unwrap_err();
        assert!(format!("{err}").contains("401"), "{err}");
        unlock_env();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn max_messages_zero_returns_empty_without_network() {
        // No server registered — proves we short-circuit.
        let docs = tokio::task::spawn_blocking(|| {
            OutlookMessagesLoader::new("X").with_max_messages(0).load()
        })
        .await
        .unwrap()
        .unwrap();
        assert!(docs.is_empty());
    }

    // ---- builder ----

    #[test]
    fn builder_setters_persist() {
        let l = OutlookMessagesLoader::new("token")
            .with_user_id("u@d.com")
            .with_folder("Inbox")
            .with_search("foo")
            .with_filter("isRead eq false")
            .with_max_messages(7)
            .with_html_body();
        assert_eq!(l.user_id, "u@d.com");
        assert_eq!(l.folder.as_deref(), Some("Inbox"));
        assert_eq!(l.search.as_deref(), Some("foo"));
        assert_eq!(l.filter.as_deref(), Some("isRead eq false"));
        assert_eq!(l.max_messages, 7);
        assert!(l.html_body);
    }
}
