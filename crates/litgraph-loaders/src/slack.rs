//! Slack loader — pull channel message history via the Slack Web API.
//!
//! Direct LangChain `SlackDirectoryLoader` parity-in-spirit, but reads the
//! live API (not a local export archive). Each message becomes one
//! `Document`; thread replies optionally flattened inline.
//!
//! # Auth
//!
//! Bearer token from a Slack app. Two scopes needed:
//! - `channels:history` (public channels) or `groups:history` (private).
//! - `channels:read` / `groups:read` for metadata (optional; we don't hit it).
//!
//! Token pattern: `xoxb-...` (bot token) or `xoxp-...` (user token).
//!
//! # What we extract
//!
//! Primary field is `message.text`. Metadata per document:
//! - `channel` — the channel id passed at construction
//! - `user` — Slack user id of the author (None for bot_message / other)
//! - `ts` — Slack's ordering timestamp (opaque string; use for deduping)
//! - `type` — Slack message type (typically "message")
//! - `thread_ts` — parent thread ts if this is a reply; None for top-level
//!
//! # Threads
//!
//! `with_include_threads(true)` follows every thread-parent's
//! `conversations.replies` and appends all replies (in order, after the
//! parent). `reply_count > 0` is the only reliable signal for "has thread";
//! we check it before issuing the replies call so channels with no threads
//! cost exactly one `conversations.history` round-trip.
//!
//! # Pagination
//!
//! Slack paginates at `limit` (default 100, max 1000). We follow
//! `response_metadata.next_cursor` until exhausted OR `max_messages` cap hit.

use std::time::Duration;

use litgraph_core::Document;
use serde_json::Value;

use crate::{Loader, LoaderError, LoaderResult};

const SLACK_API: &str = "https://slack.com/api";

pub struct SlackLoader {
    pub api_key: String,
    pub channel_id: String,
    pub base_url: String,
    pub timeout: Duration,
    /// Cap on total messages loaded. Protects against accidental scrapes of
    /// #general over 10 years. Default 1000.
    pub max_messages: Option<usize>,
    /// If set, pull thread replies for every top-level message that has
    /// `reply_count > 0`. Default false — threads add 1 extra round-trip
    /// per parent, so opt-in.
    pub include_threads: bool,
    /// Unix timestamp (seconds, as string per Slack's convention) lower bound.
    pub oldest_ts: Option<String>,
    pub latest_ts: Option<String>,
}

impl SlackLoader {
    pub fn new(api_key: impl Into<String>, channel_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            channel_id: channel_id.into(),
            base_url: SLACK_API.into(),
            timeout: Duration::from_secs(30),
            max_messages: Some(1000),
            include_threads: false,
            oldest_ts: None,
            latest_ts: None,
        }
    }

    /// Override the API base URL — used by tests pointing at fake servers.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_max_messages(mut self, n: Option<usize>) -> Self {
        self.max_messages = n;
        self
    }

    pub fn with_include_threads(mut self, b: bool) -> Self {
        self.include_threads = b;
        self
    }

    pub fn with_oldest(mut self, ts: impl Into<String>) -> Self {
        self.oldest_ts = Some(ts.into());
        self
    }

    pub fn with_latest(mut self, ts: impl Into<String>) -> Self {
        self.latest_ts = Some(ts.into());
        self
    }

    fn client(&self) -> LoaderResult<reqwest::blocking::Client> {
        reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(LoaderError::from)
    }

    /// Single `conversations.history` page. Returns (messages, next_cursor).
    fn fetch_history_page(
        &self,
        client: &reqwest::blocking::Client,
        cursor: Option<&str>,
        limit: usize,
    ) -> LoaderResult<(Vec<Value>, Option<String>)> {
        let url = format!("{}/conversations.history", self.base_url.trim_end_matches('/'));
        let mut req = client
            .get(&url)
            .bearer_auth(&self.api_key)
            .query(&[("channel", self.channel_id.as_str())])
            .query(&[("limit", limit.to_string().as_str())]);
        if let Some(c) = cursor {
            req = req.query(&[("cursor", c)]);
        }
        if let Some(ref o) = self.oldest_ts {
            req = req.query(&[("oldest", o.as_str())]);
        }
        if let Some(ref l) = self.latest_ts {
            req = req.query(&[("latest", l.as_str())]);
        }
        let resp = req.send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "slack conversations.history {status}: {body}"
            )));
        }
        let v: Value = resp.json()?;
        // Slack's envelope: { ok: bool, error?: string, messages: [...], response_metadata: { next_cursor } }
        if !v.get("ok").and_then(|b| b.as_bool()).unwrap_or(false) {
            let err = v.get("error").and_then(|e| e.as_str()).unwrap_or("unknown");
            return Err(LoaderError::Other(format!(
                "slack conversations.history api: {err}"
            )));
        }
        let messages = v
            .get("messages")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();
        let next_cursor = v
            .pointer("/response_metadata/next_cursor")
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from);
        Ok((messages, next_cursor))
    }

    /// `conversations.replies` for a thread parent. Returns all replies
    /// (EXCLUDING the parent, which we already have from history).
    fn fetch_thread_replies(
        &self,
        client: &reqwest::blocking::Client,
        thread_ts: &str,
    ) -> LoaderResult<Vec<Value>> {
        let url = format!("{}/conversations.replies", self.base_url.trim_end_matches('/'));
        let resp = client
            .get(&url)
            .bearer_auth(&self.api_key)
            .query(&[
                ("channel", self.channel_id.as_str()),
                ("ts", thread_ts),
                ("limit", "1000"),
            ])
            .send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "slack conversations.replies {status}: {body}"
            )));
        }
        let v: Value = resp.json()?;
        if !v.get("ok").and_then(|b| b.as_bool()).unwrap_or(false) {
            let err = v.get("error").and_then(|e| e.as_str()).unwrap_or("unknown");
            return Err(LoaderError::Other(format!(
                "slack conversations.replies api: {err}"
            )));
        }
        // Slack returns the parent as replies[0]; drop it so we don't dup.
        let mut messages = v
            .get("messages")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();
        if !messages.is_empty() {
            messages.remove(0);
        }
        Ok(messages)
    }

    fn message_to_document(&self, m: &Value) -> Option<Document> {
        let text = m.get("text").and_then(|t| t.as_str())?;
        if text.is_empty() {
            return None;
        }
        let mut d = Document::new(text);
        d.metadata
            .insert("channel".into(), Value::String(self.channel_id.clone()));
        if let Some(user) = m.get("user").and_then(|u| u.as_str()) {
            d.metadata.insert("user".into(), Value::String(user.into()));
        }
        if let Some(ts) = m.get("ts").and_then(|t| t.as_str()) {
            d.metadata.insert("ts".into(), Value::String(ts.into()));
            d.id = Some(format!("{}:{}", self.channel_id, ts));
        }
        if let Some(mt) = m.get("type").and_then(|t| t.as_str()) {
            d.metadata.insert("type".into(), Value::String(mt.into()));
        }
        // thread_ts present on replies AND on parents that have replies.
        // For parents, thread_ts == ts — we only tag actual replies (where
        // thread_ts != ts) so downstream consumers can filter easily.
        if let Some(tts) = m.get("thread_ts").and_then(|t| t.as_str()) {
            let is_reply = m
                .get("ts")
                .and_then(|t| t.as_str())
                .map(|s| s != tts)
                .unwrap_or(false);
            if is_reply {
                d.metadata
                    .insert("thread_ts".into(), Value::String(tts.into()));
            }
        }
        d.metadata.insert(
            "source".into(),
            Value::String(format!("slack:{}", self.channel_id)),
        );
        Some(d)
    }
}

impl Loader for SlackLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.client()?;
        let mut docs: Vec<Document> = Vec::new();
        let mut cursor: Option<String> = None;
        let cap = self.max_messages;

        'pages: loop {
            let (messages, next_cursor) = self.fetch_history_page(
                &client,
                cursor.as_deref(),
                100,
            )?;
            for m in &messages {
                if let Some(d) = self.message_to_document(m) {
                    docs.push(d);
                    if let Some(c) = cap {
                        if docs.len() >= c {
                            break 'pages;
                        }
                    }
                }
                // Follow thread replies if opted in + parent has replies.
                if self.include_threads {
                    let has_replies = m
                        .get("reply_count")
                        .and_then(|n| n.as_u64())
                        .unwrap_or(0)
                        > 0;
                    if has_replies {
                        if let Some(ts) = m.get("ts").and_then(|t| t.as_str()) {
                            let replies = self.fetch_thread_replies(&client, ts)?;
                            for r in &replies {
                                if let Some(d) = self.message_to_document(r) {
                                    docs.push(d);
                                    if let Some(c) = cap {
                                        if docs.len() >= c {
                                            break 'pages;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            match next_cursor {
                Some(c) => cursor = Some(c),
                None => break,
            }
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
        _shutdown: std::sync::mpsc::Sender<()>,
    }

    fn history_page_1() -> Value {
        json!({
            "ok": true,
            "messages": [
                {"type": "message", "text": "first", "user": "U1", "ts": "100.0"},
                {"type": "message", "text": "second with replies", "user": "U2",
                 "ts": "200.0", "thread_ts": "200.0", "reply_count": 2},
            ],
            "response_metadata": {"next_cursor": "CURSOR1"}
        })
    }

    fn history_page_2() -> Value {
        json!({
            "ok": true,
            "messages": [
                {"type": "message", "text": "third", "user": "U3", "ts": "300.0"},
            ],
            "response_metadata": {"next_cursor": ""}
        })
    }

    fn replies() -> Value {
        json!({
            "ok": true,
            "messages": [
                // Slack always returns the parent first; we drop it.
                {"type": "message", "text": "second with replies", "user": "U2",
                 "ts": "200.0", "thread_ts": "200.0"},
                {"type": "message", "text": "reply one", "user": "U4",
                 "ts": "201.0", "thread_ts": "200.0"},
                {"type": "message", "text": "reply two", "user": "U5",
                 "ts": "202.0", "thread_ts": "200.0"},
            ]
        })
    }

    fn spawn_fake() -> FakeServer {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen_paths: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_auth: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel::<()>();
        listener.set_nonblocking(true).unwrap();
        let paths_w = seen_paths.clone();
        let auth_w = seen_auth.clone();
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
                        let mut auth = None;
                        for line in req.lines().skip(1) {
                            if let Some((k, v)) = line.split_once(':') {
                                if k.trim().eq_ignore_ascii_case("authorization") {
                                    auth = Some(v.trim().to_string());
                                }
                            }
                        }
                        auth_w.lock().unwrap().push(auth);
                        let body = if path.starts_with("/conversations.history") {
                            if path.contains("cursor=CURSOR1") {
                                history_page_2().to_string()
                            } else {
                                history_page_1().to_string()
                            }
                        } else if path.starts_with("/conversations.replies") {
                            replies().to_string()
                        } else {
                            json!({"ok": false, "error": "unknown"}).to_string()
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
            _shutdown: shutdown_tx,
        }
    }

    #[test]
    fn basic_history_loader_paginates_and_returns_one_doc_per_message() {
        let srv = spawn_fake();
        let loader = SlackLoader::new("xoxb-test", "C123").with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        // 2 messages on page 1 + 1 on page 2 = 3. No threads included.
        assert_eq!(docs.len(), 3);
        let texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
        assert_eq!(texts, vec!["first", "second with replies", "third"]);
    }

    #[test]
    fn metadata_includes_channel_user_ts_type() {
        let srv = spawn_fake();
        let loader = SlackLoader::new("xoxb", "C999").with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        let d0 = &docs[0];
        assert_eq!(d0.metadata["channel"].as_str(), Some("C999"));
        assert_eq!(d0.metadata["user"].as_str(), Some("U1"));
        assert_eq!(d0.metadata["ts"].as_str(), Some("100.0"));
        assert_eq!(d0.metadata["type"].as_str(), Some("message"));
        assert_eq!(d0.metadata["source"].as_str(), Some("slack:C999"));
        // Document id = channel:ts.
        assert_eq!(d0.id.as_deref(), Some("C999:100.0"));
    }

    #[test]
    fn top_level_messages_do_not_get_thread_ts_metadata() {
        // Parent of a thread has thread_ts == ts; we should NOT tag it as
        // a reply. Only actual replies (ts != thread_ts) get the field.
        let srv = spawn_fake();
        let loader = SlackLoader::new("x", "C1").with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        // docs[1] is the parent with thread_ts == "200.0" == ts. No tag.
        assert!(docs[1].metadata.get("thread_ts").is_none());
    }

    #[test]
    fn auth_bearer_token_on_every_request() {
        let srv = spawn_fake();
        let loader = SlackLoader::new("xoxb-secret", "C1").with_base_url(&srv.url);
        let _ = loader.load().unwrap();
        let auth = srv.seen_auth.lock().unwrap().clone();
        assert!(!auth.is_empty());
        for a in &auth {
            assert_eq!(a.as_deref(), Some("Bearer xoxb-secret"));
        }
    }

    #[test]
    fn cursor_pagination_follows_next_cursor() {
        let srv = spawn_fake();
        let loader = SlackLoader::new("x", "C1").with_base_url(&srv.url);
        let _ = loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        // Page 1 has no cursor; page 2 has cursor=CURSOR1.
        assert!(paths.iter().any(|p| p.starts_with("/conversations.history") && !p.contains("cursor=")));
        assert!(paths.iter().any(|p| p.contains("cursor=CURSOR1")));
    }

    #[test]
    fn include_threads_follows_replies_and_drops_parent_duplicate() {
        let srv = spawn_fake();
        let loader = SlackLoader::new("x", "C1")
            .with_base_url(&srv.url)
            .with_include_threads(true);
        let docs = loader.load().unwrap();
        // 3 top-level + 2 replies (parent dropped from replies response) = 5.
        assert_eq!(docs.len(), 5);
        let reply_texts: Vec<&str> = docs
            .iter()
            .filter(|d| d.metadata.get("thread_ts").is_some())
            .map(|d| d.content.as_str())
            .collect();
        assert_eq!(reply_texts, vec!["reply one", "reply two"]);
    }

    #[test]
    fn include_threads_replies_carry_thread_ts_metadata() {
        let srv = spawn_fake();
        let loader = SlackLoader::new("x", "C1")
            .with_base_url(&srv.url)
            .with_include_threads(true);
        let docs = loader.load().unwrap();
        let replies: Vec<&Document> = docs
            .iter()
            .filter(|d| d.metadata.get("thread_ts").is_some())
            .collect();
        for r in &replies {
            assert_eq!(r.metadata["thread_ts"].as_str(), Some("200.0"));
        }
    }

    #[test]
    fn max_messages_cap_truncates_result() {
        let srv = spawn_fake();
        let loader = SlackLoader::new("x", "C1")
            .with_base_url(&srv.url)
            .with_max_messages(Some(2));
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn oldest_and_latest_ts_appear_in_query_string() {
        let srv = spawn_fake();
        let loader = SlackLoader::new("x", "C1")
            .with_base_url(&srv.url)
            .with_oldest("100.0")
            .with_latest("500.0");
        let _ = loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        let hist = paths
            .iter()
            .find(|p| p.starts_with("/conversations.history"))
            .unwrap();
        assert!(hist.contains("oldest=100.0"), "oldest missing: {hist}");
        assert!(hist.contains("latest=500.0"), "latest missing: {hist}");
    }

    #[test]
    fn slack_api_error_response_surfaces_as_loader_error() {
        // `ok: false` envelope → LoaderError, NOT silent success.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel::<()>();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            loop {
                if shutdown_rx.try_recv().is_ok() { break; }
                match listener.accept() {
                    Ok((mut s, _)) => {
                        s.set_nonblocking(false).ok();
                        let mut buf = [0u8; 4096];
                        let _ = s.read(&mut buf);
                        let body = json!({"ok": false, "error": "channel_not_found"}).to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            body.len(), body
                        );
                        let _ = s.write_all(resp.as_bytes());
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        let loader = SlackLoader::new("x", "bad-channel")
            .with_base_url(format!("http://127.0.0.1:{port}"));
        let err = loader.load().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("channel_not_found"), "got: {msg}");
        drop(shutdown_tx);
    }
}
