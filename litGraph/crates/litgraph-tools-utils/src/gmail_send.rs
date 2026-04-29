//! Gmail send tool. Pairs with `GmailLoader` (read) to complete the
//! read/write loop. Uses Google's REST API: POST to
//! `users/me/messages/send` with a base64url-encoded RFC 2822 message body.
//!
//! # Auth
//!
//! Bearer token (Google OAuth2). Required scope:
//! `https://www.googleapis.com/auth/gmail.send`. Same caller-supplied
//! token pattern as `GmailLoader` — agents don't roll their own OAuth
//! flow; the host app provides a refreshed access token.
//!
//! # Why not full SMTP
//!
//! SMTP needs username + app-password + port + TLS config. The Gmail REST
//! API needs only a bearer token + JSON. For agents, the REST path is
//! one less moving part. SMTP is intentionally NOT supported here — too
//! many footguns (port 587 vs 465, STARTTLS, app-password rotation).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use litgraph_core::{Error, Result, Tool, ToolSchema};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

const GMAIL_SEND_URL: &str = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send";

#[derive(Debug, Clone)]
pub struct GmailSendConfig {
    pub access_token: String,
    pub timeout: Duration,
    /// Optional override for the base URL (test fakes; never useful in prod).
    pub base_url: Option<String>,
    /// Optional fixed `From:` address. Gmail picks one based on the token's
    /// account if omitted, but for service-account / delegated-domain
    /// setups you may want to pin it.
    pub from: Option<String>,
}

impl GmailSendConfig {
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            timeout: Duration::from_secs(30),
            base_url: None,
            from: None,
        }
    }
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }
    pub fn with_from(mut self, addr: impl Into<String>) -> Self {
        self.from = Some(addr.into());
        self
    }
}

pub struct GmailSendTool {
    cfg: Arc<GmailSendConfig>,
    http: Client,
}

impl GmailSendTool {
    pub fn new(cfg: GmailSendConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("gmail_send build: {e}")))?;
        Ok(Self {
            cfg: Arc::new(cfg),
            http,
        })
    }

    fn endpoint(&self) -> String {
        match &self.cfg.base_url {
            Some(b) => format!("{}/gmail/v1/users/me/messages/send", b.trim_end_matches('/')),
            None => GMAIL_SEND_URL.to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SendArgs {
    /// Comma-separated list of addresses (or single address). Required.
    to: String,
    subject: String,
    body: String,
    #[serde(default)]
    cc: Option<String>,
    #[serde(default)]
    bcc: Option<String>,
    #[serde(default)]
    reply_to: Option<String>,
    /// Override the configured `from`. Most callers don't supply this.
    #[serde(default)]
    from: Option<String>,
}

#[async_trait]
impl Tool for GmailSendTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "gmail_send".into(),
            description: "Send an email via Gmail. Provide `to`, `subject`, and `body`. \
                          Optional `cc`, `bcc`, `reply_to`. Returns the sent message id."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient address(es), comma-separated."},
                    "subject": {"type": "string"},
                    "body": {"type": "string", "description": "Plain-text body."},
                    "cc": {"type": "string", "description": "Optional CC address(es)."},
                    "bcc": {"type": "string", "description": "Optional BCC address(es)."},
                    "reply_to": {"type": "string", "description": "Optional Reply-To address."},
                    "from": {"type": "string", "description": "Optional override for the From address."}
                },
                "required": ["to", "subject", "body"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let parsed: SendArgs = serde_json::from_value(args)
            .map_err(|e| Error::invalid(format!("gmail_send args: {e}")))?;
        if parsed.to.trim().is_empty() {
            return Err(Error::invalid("gmail_send: `to` cannot be empty"));
        }
        let from = parsed
            .from
            .or_else(|| self.cfg.from.clone())
            .unwrap_or_else(|| "me".to_string());

        let raw = build_rfc2822(
            &from,
            &parsed.to,
            parsed.cc.as_deref(),
            parsed.bcc.as_deref(),
            parsed.reply_to.as_deref(),
            &parsed.subject,
            &parsed.body,
        );
        let encoded = URL_SAFE_NO_PAD.encode(raw.as_bytes());

        let resp = self
            .http
            .post(self.endpoint())
            .bearer_auth(&self.cfg.access_token)
            .json(&json!({"raw": encoded}))
            .send()
            .await
            .map_err(|e| Error::other(format!("gmail_send send: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("gmail_send {status}: {body}")));
        }
        let v: Value = resp
            .json()
            .await
            .map_err(|e| Error::other(format!("gmail_send decode: {e}")))?;
        Ok(json!({
            "id": v.get("id").and_then(|x| x.as_str()).unwrap_or(""),
            "thread_id": v.get("threadId").and_then(|x| x.as_str()).unwrap_or(""),
        }))
    }
}

/// Build a minimal RFC 2822 message. Plain-text only (no attachments,
/// no MIME multipart). Headers manually concat'd — sub-100-line scope
/// vs pulling in a full email crate.
///
/// Newlines: CRLF as required by RFC 2822 (Gmail tolerates LF but
/// strict mail servers along the path do not — defensive default).
fn build_rfc2822(
    from: &str,
    to: &str,
    cc: Option<&str>,
    bcc: Option<&str>,
    reply_to: Option<&str>,
    subject: &str,
    body: &str,
) -> String {
    let mut s = String::new();
    s.push_str(&format!("From: {from}\r\n"));
    s.push_str(&format!("To: {to}\r\n"));
    if let Some(c) = cc {
        s.push_str(&format!("Cc: {c}\r\n"));
    }
    if let Some(b) = bcc {
        s.push_str(&format!("Bcc: {b}\r\n"));
    }
    if let Some(r) = reply_to {
        s.push_str(&format!("Reply-To: {r}\r\n"));
    }
    s.push_str(&format!("Subject: {subject}\r\n"));
    s.push_str("MIME-Version: 1.0\r\n");
    s.push_str("Content-Type: text/plain; charset=UTF-8\r\n");
    s.push_str("Content-Transfer-Encoding: 7bit\r\n");
    s.push_str("\r\n");
    s.push_str(body);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn start_fake(response_status: u16, response_body: &'static str) -> (u16, std::sync::mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                let mut buf = [0u8; 16384];
                let mut total = Vec::new();
                let mut header_end_pos: Option<usize> = None;
                let mut content_length: usize = 0;
                while header_end_pos.is_none() {
                    let n = match s.read(&mut buf) { Ok(0) => break, Ok(n) => n, Err(_) => break };
                    total.extend_from_slice(&buf[..n]);
                    if let Some(p) = total.windows(4).position(|w| w == b"\r\n\r\n") {
                        header_end_pos = Some(p + 4);
                        let header_str = String::from_utf8_lossy(&total[..p]).to_lowercase();
                        for line in header_str.split("\r\n") {
                            if let Some(v) = line.strip_prefix("content-length:") {
                                content_length = v.trim().parse().unwrap_or(0);
                            }
                        }
                    }
                }
                let header_end = header_end_pos.unwrap_or(total.len());
                while total.len() < header_end + content_length {
                    let n = match s.read(&mut buf) { Ok(0) => break, Ok(n) => n, Err(_) => break };
                    total.extend_from_slice(&buf[..n]);
                }
                tx.send(String::from_utf8_lossy(&total).to_string()).unwrap();

                let header = format!(
                    "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response_status,
                    response_body.len()
                );
                let _ = s.write_all(header.as_bytes());
                let _ = s.write_all(response_body.as_bytes());
            }
        });
        (port, rx)
    }

    fn decode_raw(captured_request: &str) -> String {
        // Find the JSON body, parse, decode the `raw` field.
        let body_start = captured_request.find("\r\n\r\n").map(|p| p + 4).unwrap_or(0);
        let body_str = &captured_request[body_start..];
        let v: Value = serde_json::from_str(body_str.trim_end_matches('\0').trim())
            .unwrap_or_else(|_| panic!("body not JSON: {body_str:?}"));
        let raw = v["raw"].as_str().unwrap();
        let bytes = URL_SAFE_NO_PAD.decode(raw).unwrap();
        String::from_utf8(bytes).unwrap()
    }

    #[tokio::test]
    async fn send_returns_message_id() {
        let (port, rx) = start_fake(200, r#"{"id": "msg-abc", "threadId": "thr-xyz"}"#);
        let cfg = GmailSendConfig::new("token").with_base_url(format!("http://127.0.0.1:{port}"));
        let tool = GmailSendTool::new(cfg).unwrap();
        let resp = tool
            .run(json!({"to": "alice@example.com", "subject": "hi", "body": "hello"}))
            .await
            .unwrap();
        assert_eq!(resp["id"], "msg-abc");
        assert_eq!(resp["thread_id"], "thr-xyz");
        let captured = rx.recv().unwrap();
        assert!(captured.contains("authorization: Bearer token"));
    }

    #[tokio::test]
    async fn send_includes_required_headers_in_rfc2822() {
        let (port, rx) = start_fake(200, r#"{"id": "x", "threadId": "y"}"#);
        let cfg = GmailSendConfig::new("token").with_base_url(format!("http://127.0.0.1:{port}"));
        let tool = GmailSendTool::new(cfg).unwrap();
        tool.run(json!({"to": "alice@example.com", "subject": "Hello!", "body": "the body"}))
            .await
            .unwrap();
        let captured = rx.recv().unwrap();
        let raw = decode_raw(&captured);
        assert!(raw.contains("From: me\r\n"));
        assert!(raw.contains("To: alice@example.com\r\n"));
        assert!(raw.contains("Subject: Hello!\r\n"));
        assert!(raw.contains("MIME-Version: 1.0\r\n"));
        assert!(raw.contains("Content-Type: text/plain; charset=UTF-8\r\n"));
        // Body separated from headers by blank line.
        assert!(raw.ends_with("\r\nthe body"));
    }

    #[tokio::test]
    async fn cc_bcc_reply_to_appear_in_message() {
        let (port, rx) = start_fake(200, r#"{"id": "x", "threadId": "y"}"#);
        let cfg = GmailSendConfig::new("token").with_base_url(format!("http://127.0.0.1:{port}"));
        let tool = GmailSendTool::new(cfg).unwrap();
        tool.run(json!({
            "to": "primary@x", "subject": "s", "body": "b",
            "cc": "cc@x", "bcc": "bcc@x", "reply_to": "reply@x"
        })).await.unwrap();
        let raw = decode_raw(&rx.recv().unwrap());
        assert!(raw.contains("Cc: cc@x\r\n"));
        assert!(raw.contains("Bcc: bcc@x\r\n"));
        assert!(raw.contains("Reply-To: reply@x\r\n"));
    }

    #[tokio::test]
    async fn empty_to_returns_invalid_input() {
        let cfg = GmailSendConfig::new("token");
        let tool = GmailSendTool::new(cfg).unwrap();
        let err = tool.run(json!({"to": "  ", "subject": "s", "body": "b"})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn missing_required_field_returns_invalid_input() {
        let cfg = GmailSendConfig::new("token");
        let tool = GmailSendTool::new(cfg).unwrap();
        let err = tool.run(json!({"to": "x@y", "body": "b"})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn http_error_surfaces_with_status_and_body() {
        let (port, _rx) = start_fake(403, r#"{"error": {"message": "insufficient scope"}}"#);
        let cfg = GmailSendConfig::new("bad-token").with_base_url(format!("http://127.0.0.1:{port}"));
        let tool = GmailSendTool::new(cfg).unwrap();
        let err = tool.run(json!({"to": "x@y", "subject": "s", "body": "b"})).await.unwrap_err();
        let s = err.to_string();
        assert!(s.contains("403"));
        assert!(s.contains("insufficient scope"));
    }

    #[tokio::test]
    async fn configured_from_used_when_args_omit_it() {
        let (port, rx) = start_fake(200, r#"{"id": "x", "threadId": "y"}"#);
        let cfg = GmailSendConfig::new("token")
            .with_base_url(format!("http://127.0.0.1:{port}"))
            .with_from("agent@my-domain.com");
        let tool = GmailSendTool::new(cfg).unwrap();
        tool.run(json!({"to": "alice@x", "subject": "s", "body": "b"})).await.unwrap();
        let raw = decode_raw(&rx.recv().unwrap());
        assert!(raw.contains("From: agent@my-domain.com\r\n"));
    }

    #[tokio::test]
    async fn args_from_overrides_configured_from() {
        let (port, rx) = start_fake(200, r#"{"id": "x", "threadId": "y"}"#);
        let cfg = GmailSendConfig::new("token")
            .with_base_url(format!("http://127.0.0.1:{port}"))
            .with_from("agent@my-domain.com");
        let tool = GmailSendTool::new(cfg).unwrap();
        tool.run(json!({
            "to": "alice@x", "subject": "s", "body": "b", "from": "override@elsewhere.com"
        })).await.unwrap();
        let raw = decode_raw(&rx.recv().unwrap());
        assert!(raw.contains("From: override@elsewhere.com\r\n"));
        assert!(!raw.contains("From: agent@my-domain.com"));
    }

    #[tokio::test]
    async fn schema_has_required_fields() {
        let cfg = GmailSendConfig::new("token");
        let tool = GmailSendTool::new(cfg).unwrap();
        let s = tool.schema();
        assert_eq!(s.name, "gmail_send");
        let required = s.parameters["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "to"));
        assert!(required.iter().any(|v| v == "subject"));
        assert!(required.iter().any(|v| v == "body"));
    }
}
