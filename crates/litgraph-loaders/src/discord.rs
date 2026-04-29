//! Discord channel loader. Pulls messages from one channel via the
//! Discord REST API, paginating with the `before=<msg_id>` cursor.
//!
//! # Auth
//!
//! Two header forms are accepted by Discord, surfaced as builder
//! methods:
//!
//! - **Bot token** (most common): `Authorization: Bot <TOKEN>`. Use
//!   [`DiscordChannelLoader::with_bot_token`]. Required scope:
//!   `View Channel` + `Read Message History` on the target channel.
//! - **Bearer token** (OAuth user flows): use [`with_bearer_token`].
//!   Generally only useful when running on behalf of a user.
//!
//! # Document shape
//!
//! One Document per message. Order: chronological (oldest-first) so a
//! downstream summariser sees the conversation in reading order.
//!
//! - `content` = message `content` text. Rich embeds / attachments are
//!   referenced from metadata, not inlined into content (would dilute
//!   embeddings).
//! - `id` = Discord message id (snowflake — sortable + unique forever).
//! - `metadata`:
//!   - `channel_id`: source channel id
//!   - `author_id`: snowflake
//!   - `author_username`: username (no discriminator — Discord
//!     deprecated those)
//!   - `author_bot`: bool
//!   - `timestamp`: ISO-8601 string from Discord
//!   - `edited_timestamp`: nullable ISO-8601
//!   - `attachments`: array of `{filename, url, size, content_type}`
//!   - `mentions_count`: how many users were @-mentioned (cheap signal
//!     for filtering noise)
//!   - `source`: `"discord"`
//!
//! # Pagination
//!
//! Discord caps each `/messages` call at 100. We fetch in batches with
//! `before=<oldest_id_seen>` until empty or [`max_messages`] is hit.
//! Returned messages are reversed (Discord delivers newest-first) so
//! the final Vec is oldest-first.

use std::sync::Mutex;
use std::time::Duration;

use litgraph_core::Document;
use serde_json::{json, Value};

use crate::{Loader, LoaderError, LoaderResult};

const ENDPOINT: &str = "https://discord.com/api/v10";
const PAGE_SIZE: usize = 100;
const DEFAULT_MAX_MESSAGES: usize = 1000;

pub struct DiscordChannelLoader {
    pub channel_id: String,
    pub max_messages: usize,
    /// Pre-formatted Authorization header value (`"Bot xyz"` or
    /// `"Bearer xyz"`). Stored as the full string so we don't have
    /// to know which kind it is at fetch time.
    auth_header: Mutex<Option<String>>,
    pub timeout: Duration,
    pub user_agent: String,
}

impl DiscordChannelLoader {
    pub fn new(channel_id: impl Into<String>) -> Self {
        Self {
            channel_id: channel_id.into(),
            max_messages: DEFAULT_MAX_MESSAGES,
            auth_header: Mutex::new(None),
            timeout: Duration::from_secs(30),
            user_agent: format!(
                "DiscordBot (litgraph-loaders/{}, {})",
                env!("CARGO_PKG_VERSION"),
                "https://github.com/litgraph/litgraph"
            ),
        }
    }

    /// Set the bot token. Discord requires the `Bot ` prefix in the
    /// header — we handle that, the caller passes the bare token.
    pub fn with_bot_token(self, token: impl AsRef<str>) -> Self {
        *self.auth_header.lock().expect("auth header mutex") =
            Some(format!("Bot {}", token.as_ref()));
        self
    }

    /// Bearer-token auth (OAuth user flows).
    pub fn with_bearer_token(self, token: impl AsRef<str>) -> Self {
        *self.auth_header.lock().expect("auth header mutex") =
            Some(format!("Bearer {}", token.as_ref()));
        self
    }

    pub fn with_max_messages(mut self, n: usize) -> Self {
        self.max_messages = n;
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
        std::env::var("LITGRAPH_DISCORD_API").unwrap_or_else(|_| ENDPOINT.into())
    }

    fn auth_value(&self) -> Option<String> {
        self.auth_header.lock().expect("auth header mutex").clone()
    }

    /// Fetch one page of messages. `before` is the snowflake of the
    /// oldest message we've already seen — pass `None` for the first
    /// page.
    fn fetch_page(
        &self,
        client: &reqwest::blocking::Client,
        before: Option<&str>,
        page_limit: usize,
    ) -> LoaderResult<Vec<Value>> {
        let mut url = format!(
            "{}/channels/{}/messages?limit={}",
            self.endpoint(),
            self.channel_id,
            page_limit.min(PAGE_SIZE)
        );
        if let Some(b) = before {
            url.push_str(&format!("&before={b}"));
        }
        let mut req = client.get(&url);
        if let Some(auth) = self.auth_value() {
            req = req.header("Authorization", auth);
        }
        let resp = req.send()?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "discord {status} {url}: {body}"
            )));
        }
        let v: Value = resp.json()?;
        let arr = v
            .as_array()
            .ok_or_else(|| LoaderError::Other("discord: response was not an array".into()))?
            .clone();
        Ok(arr)
    }
}

impl Loader for DiscordChannelLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        if self.max_messages == 0 {
            return Ok(Vec::new());
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent(&self.user_agent)
            .build()?;
        let mut collected: Vec<Value> = Vec::new();
        let mut before: Option<String> = None;
        loop {
            let remaining = self.max_messages.saturating_sub(collected.len());
            if remaining == 0 {
                break;
            }
            let want = remaining.min(PAGE_SIZE);
            let page = self.fetch_page(&client, before.as_deref(), want)?;
            if page.is_empty() {
                break;
            }
            // Discord delivers newest-first within a page. The OLDEST
            // message in the page is the last element. Use its id as
            // the next `before` cursor.
            let oldest_id = page
                .last()
                .and_then(|m| m.get("id"))
                .and_then(|v| v.as_str())
                .map(String::from);
            collected.extend(page);
            match oldest_id {
                Some(id) => before = Some(id),
                None => break, // defensive: malformed page → stop
            }
        }
        // Reverse to oldest-first.
        collected.reverse();
        // Honour cap exactly (the loop above can overshoot by < PAGE_SIZE).
        collected.truncate(self.max_messages);
        Ok(collected
            .iter()
            .map(|m| message_to_document(m, &self.channel_id))
            .collect())
    }
}

// ---- message → Document ----------------------------------------------------

pub(crate) fn message_to_document(msg: &Value, channel_id: &str) -> Document {
    let id = msg
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let content = msg
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let author = msg.get("author");
    let author_id = author
        .and_then(|a| a.get("id"))
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_default();
    let author_username = author
        .and_then(|a| a.get("username"))
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_default();
    let author_bot = author
        .and_then(|a| a.get("bot"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let timestamp = msg
        .get("timestamp")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_default();
    let edited_timestamp = msg
        .get("edited_timestamp")
        .and_then(|v| v.as_str())
        .map(String::from);

    let attachments: Vec<Value> = msg
        .get("attachments")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|a| {
                    json!({
                        "filename": a.get("filename").and_then(|v| v.as_str()).unwrap_or(""),
                        "url": a.get("url").and_then(|v| v.as_str()).unwrap_or(""),
                        "size": a.get("size").and_then(|v| v.as_u64()).unwrap_or(0),
                        "content_type": a.get("content_type").and_then(|v| v.as_str()).unwrap_or(""),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let mentions_count = msg
        .get("mentions")
        .and_then(|v| v.as_array())
        .map(|a| a.len() as u64)
        .unwrap_or(0);

    let mut d = Document::new(content);
    if !id.is_empty() {
        d = d.with_id(&id);
    }
    let mut put = |k: &str, v: Value| {
        d.metadata.insert(k.into(), v);
    };
    put("channel_id", Value::String(channel_id.to_string()));
    put("message_id", Value::String(id));
    put("author_id", Value::String(author_id));
    put("author_username", Value::String(author_username));
    put("author_bot", Value::Bool(author_bot));
    if !timestamp.is_empty() {
        put("timestamp", Value::String(timestamp));
    }
    put(
        "edited_timestamp",
        edited_timestamp.map(Value::String).unwrap_or(Value::Null),
    );
    put("attachments", Value::Array(attachments));
    put("mentions_count", Value::Number(mentions_count.into()));
    put("source", Value::String("discord".into()));
    d
}

#[cfg(test)]
mod tests {
    //! Tests use a hand-rolled TCP server so we don't pull in
    //! `wiremock` for one module. The server can serve sequential
    //! responses to test pagination.

    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::sync::Arc;

    /// Fake server that serves N pre-canned responses sequentially.
    /// Returns the base URL to point the loader at, and a Receiver
    /// that yields the request lines for each call (one per
    /// expected request).
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

    fn fake_server_status(status: u16, body: &'static str) -> String {
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

    /// Set the env override under a serial guard so parallel tests
    /// don't race. Caller drops the guard when done.
    fn lock_env_to(value: &str) -> std::sync::MutexGuard<'static, ()> {
        use std::sync::Mutex;
        static GUARD: Mutex<()> = Mutex::new(());
        let g = GUARD.lock().unwrap();
        std::env::set_var("LITGRAPH_DISCORD_API", value);
        g
    }

    fn unlock_env() {
        std::env::remove_var("LITGRAPH_DISCORD_API");
    }

    fn make_msg(id: &str, content: &str, author: &str) -> Value {
        json!({
            "id": id,
            "content": content,
            "author": {
                "id": "100",
                "username": author,
                "bot": false
            },
            "timestamp": "2024-01-15T10:30:00.000000+00:00",
            "edited_timestamp": null,
            "attachments": [],
            "mentions": []
        })
    }

    // ---- message_to_document ----

    #[test]
    fn message_to_document_extracts_core_fields() {
        let msg = json!({
            "id": "999",
            "content": "Hello",
            "author": {"id": "10", "username": "alice", "bot": false},
            "timestamp": "2024-01-01T00:00:00+00:00",
            "edited_timestamp": "2024-01-01T00:01:00+00:00",
            "attachments": [],
            "mentions": [{"id": "x"}, {"id": "y"}]
        });
        let d = message_to_document(&msg, "chan-1");
        assert_eq!(d.id.as_deref(), Some("999"));
        assert_eq!(d.content, "Hello");
        assert_eq!(
            d.metadata.get("channel_id").and_then(|v| v.as_str()),
            Some("chan-1")
        );
        assert_eq!(
            d.metadata.get("author_username").and_then(|v| v.as_str()),
            Some("alice")
        );
        assert_eq!(
            d.metadata.get("edited_timestamp").and_then(|v| v.as_str()),
            Some("2024-01-01T00:01:00+00:00")
        );
        assert_eq!(
            d.metadata.get("mentions_count").and_then(|v| v.as_u64()),
            Some(2)
        );
    }

    #[test]
    fn message_to_document_attachment_shape() {
        let msg = json!({
            "id": "1",
            "content": "see file",
            "author": {"id": "10", "username": "alice"},
            "timestamp": "2024-01-01T00:00:00+00:00",
            "attachments": [{
                "filename": "report.pdf",
                "url": "https://cdn.discordapp.com/x/report.pdf",
                "size": 12345,
                "content_type": "application/pdf"
            }]
        });
        let d = message_to_document(&msg, "chan-1");
        let atts = d.metadata.get("attachments").and_then(|v| v.as_array()).unwrap();
        assert_eq!(atts.len(), 1);
        assert_eq!(atts[0]["filename"], "report.pdf");
        assert_eq!(atts[0]["size"], 12345);
        assert_eq!(atts[0]["content_type"], "application/pdf");
    }

    #[test]
    fn message_to_document_handles_missing_author_block() {
        // System messages can have no author block.
        let msg = json!({
            "id": "1",
            "content": "system",
            "timestamp": "2024-01-01T00:00:00+00:00"
        });
        let d = message_to_document(&msg, "chan-1");
        assert_eq!(d.metadata.get("author_id").and_then(|v| v.as_str()), Some(""));
        assert_eq!(d.metadata.get("author_bot").and_then(|v| v.as_bool()), Some(false));
        // Attachments default to empty array, not null.
        assert_eq!(
            d.metadata.get("attachments").and_then(|v| v.as_array()).unwrap().len(),
            0
        );
    }

    // ---- end-to-end pagination ----

    #[tokio::test(flavor = "current_thread")]
    async fn loads_single_page() {
        let body = serde_json::to_string(&vec![
            make_msg("3", "third", "alice"),
            make_msg("2", "second", "alice"),
            make_msg("1", "first", "alice"),
        ])
        .unwrap();
        // Leak so the &'static str lifetime works for the test server.
        let body: &'static str = Box::leak(body.into_boxed_str());

        let (base, rx) = fake_server_sequence(vec![body, "[]"]);
        let _g = lock_env_to(&base);

        let docs = tokio::task::spawn_blocking(|| {
            DiscordChannelLoader::new("chan-1")
                .with_bot_token("FAKE")
                .with_max_messages(10)
                .load()
        })
        .await
        .unwrap()
        .unwrap();

        // Oldest-first.
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].content, "first");
        assert_eq!(docs[1].content, "second");
        assert_eq!(docs[2].content, "third");

        // First request must include the bot auth header + channel id.
        let req = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(req.contains("/channels/chan-1/messages"), "{req}");
        assert!(
            req.to_lowercase().contains("authorization: bot fake"),
            "{req}"
        );
        unlock_env();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pagination_uses_before_cursor_and_stops_on_empty_page() {
        let page1 = serde_json::to_string(&vec![
            make_msg("5", "msg5", "a"),
            make_msg("4", "msg4", "a"),
        ])
        .unwrap();
        let page2 = serde_json::to_string(&vec![make_msg("3", "msg3", "a")]).unwrap();
        let page3 = "[]"; // empty → stop
        let p1: &'static str = Box::leak(page1.into_boxed_str());
        let p2: &'static str = Box::leak(page2.into_boxed_str());

        let (base, rx) = fake_server_sequence(vec![p1, p2, page3]);
        let _g = lock_env_to(&base);

        let docs = tokio::task::spawn_blocking(|| {
            DiscordChannelLoader::new("chan-1")
                .with_bot_token("X")
                .with_max_messages(100)
                .load()
        })
        .await
        .unwrap()
        .unwrap();

        assert_eq!(docs.len(), 3);
        // Oldest-first.
        assert_eq!(docs[0].content, "msg3");
        assert_eq!(docs[2].content, "msg5");

        // Three requests fired. Verify pagination cursors.
        let req1 = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let req2 = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let req3 = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!req1.contains("before="), "first request shouldn't paginate: {req1}");
        // After page 1, oldest seen is msg id "4" → before=4.
        assert!(req2.contains("before=4"), "{req2}");
        // After page 2, oldest seen is msg id "3" → before=3.
        assert!(req3.contains("before=3"), "{req3}");
        unlock_env();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn max_messages_caps_pagination() {
        let page = serde_json::to_string(&vec![
            make_msg("3", "c", "a"),
            make_msg("2", "b", "a"),
            make_msg("1", "a", "a"),
        ])
        .unwrap();
        let p: &'static str = Box::leak(page.into_boxed_str());
        // Even though the server has more, we cap the loader at 2.
        let (base, _rx) = fake_server_sequence(vec![p]);
        let _g = lock_env_to(&base);

        let docs = tokio::task::spawn_blocking(|| {
            DiscordChannelLoader::new("chan-1")
                .with_bot_token("X")
                .with_max_messages(2)
                .load()
        })
        .await
        .unwrap()
        .unwrap();

        assert_eq!(docs.len(), 2);
        unlock_env();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn max_messages_zero_returns_empty() {
        // No fake server needed — should short-circuit before any
        // network call.
        let docs = tokio::task::spawn_blocking(|| {
            DiscordChannelLoader::new("chan-1")
                .with_bot_token("X")
                .with_max_messages(0)
                .load()
        })
        .await
        .unwrap()
        .unwrap();
        assert!(docs.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn auth_uses_bearer_form_when_set() {
        let body = "[]"; // empty page → loader stops after first call
        let (base, rx) = fake_server_sequence(vec![body]);
        let _g = lock_env_to(&base);

        let _ = tokio::task::spawn_blocking(|| {
            DiscordChannelLoader::new("chan-1")
                .with_bearer_token("USERTOKEN")
                .load()
        })
        .await
        .unwrap()
        .unwrap();

        let req = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            req.to_lowercase().contains("authorization: bearer usertoken"),
            "{req}"
        );
        unlock_env();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn http_error_surfaces_clean_error() {
        let url = fake_server_status(403, r#"{"code": 50001, "message": "Missing Access"}"#);
        let _g = lock_env_to(&url);

        let res = tokio::task::spawn_blocking(|| {
            DiscordChannelLoader::new("chan-1")
                .with_bot_token("X")
                .load()
        })
        .await
        .unwrap();
        let err = res.unwrap_err();
        assert!(format!("{err}").contains("403"), "{err}");
        unlock_env();
    }

    // ---- builder tests (no network) ----

    #[test]
    fn endpoint_default_when_env_unset() {
        std::env::remove_var("LITGRAPH_DISCORD_API");
        let l = DiscordChannelLoader::new("chan-1");
        assert_eq!(l.endpoint(), ENDPOINT);
    }

    #[test]
    fn with_bot_token_prefixes_correctly() {
        let l = DiscordChannelLoader::new("chan-1").with_bot_token("ABC");
        assert_eq!(l.auth_value().as_deref(), Some("Bot ABC"));
    }

    #[test]
    fn with_bearer_token_prefixes_correctly() {
        let l = DiscordChannelLoader::new("chan-1").with_bearer_token("XYZ");
        assert_eq!(l.auth_value().as_deref(), Some("Bearer XYZ"));
    }

    #[test]
    fn auth_unset_by_default() {
        let l = DiscordChannelLoader::new("chan-1");
        assert!(l.auth_value().is_none());
    }
}
