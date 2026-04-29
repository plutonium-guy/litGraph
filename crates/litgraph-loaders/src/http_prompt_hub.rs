//! HTTP-backed [`PromptHub`] — fetch versioned prompts from a server.
//!
//! Lives in `litgraph-loaders` because it needs `reqwest`, which we
//! deliberately keep out of `litgraph-core`. Drop-in for the in-core
//! [`litgraph_core::FilesystemPromptHub`].
//!
//! # Wire format
//!
//! `pull("rag/answer")` issues `GET {base_url}/rag/answer.json`.
//! `pull("rag/answer@v2")` issues `GET {base_url}/rag/answer@v2.json`.
//!
//! Response body is fed to [`ChatPromptTemplate::from_json`] — same
//! schema as the filesystem hub, so the same prompt files can be
//! served unchanged from a static-file host (S3, GitHub raw, nginx,
//! …) without any server-side logic.
//!
//! # Auth
//!
//! Pass headers via [`HttpPromptHub::with_header`] / [`with_bearer`].
//! For a private GitHub repo: bearer auth + `base_url =
//! "https://raw.githubusercontent.com/owner/repo/main/prompts"`.
//!
//! # Concurrency model
//!
//! Uses `reqwest::blocking` inside `tokio::task::spawn_blocking`.
//! Reasoning: `litgraph-loaders` already pins blocking reqwest for the
//! Loader trait (sync), and pulling in async-reqwest just for the hub
//! would double the dep tree. The cost is one blocking pool slot per
//! concurrent pull, which is fine — prompt pulls happen at startup or
//! on rare cache misses.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::prompt::ChatPromptTemplate;
use litgraph_core::prompt_hub::{PromptHub, PromptRef};
use litgraph_core::{Error, Result};

/// HTTP-backed prompt hub. Cheap to clone (`Arc` inside).
#[derive(Clone)]
pub struct HttpPromptHub {
    base_url: String,
    timeout: Duration,
    user_agent: String,
    headers: Arc<Mutex<HashMap<String, String>>>,
}

impl HttpPromptHub {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(30),
            user_agent: format!("litgraph-loaders/{}", env!("CARGO_PKG_VERSION")),
            headers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// Add an HTTP header to every request. Common cases: `Authorization`,
    /// `Accept-Encoding`. Header names are case-insensitive on the wire
    /// but we store them verbatim — pass canonical names if your server
    /// is picky.
    pub fn with_header(self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers
            .lock()
            .expect("http_prompt_hub headers poisoned")
            .insert(key.into(), value.into());
        self
    }

    /// Convenience for `Authorization: Bearer <token>`.
    pub fn with_bearer(self, token: impl AsRef<str>) -> Self {
        self.with_header("Authorization", format!("Bearer {}", token.as_ref()))
    }

    fn url_for(&self, r: &PromptRef) -> String {
        let stem = match &r.version {
            Some(v) => format!("{}@{}", r.name, v),
            None => r.name.clone(),
        };
        format!("{}/{stem}.json", self.base_url)
    }

    fn snapshot_headers(&self) -> HashMap<String, String> {
        self.headers
            .lock()
            .expect("http_prompt_hub headers poisoned")
            .clone()
    }
}

#[async_trait]
impl PromptHub for HttpPromptHub {
    async fn pull(&self, name: &str) -> Result<ChatPromptTemplate> {
        let r = PromptRef::parse(name)?;
        let url = self.url_for(&r);
        let timeout = self.timeout;
        let ua = self.user_agent.clone();
        let headers = self.snapshot_headers();
        let body = tokio::task::spawn_blocking(move || -> Result<String> {
            let mut builder = reqwest::blocking::Client::builder()
                .timeout(timeout)
                .user_agent(&ua);
            // reqwest 0.12 default-allows up to N redirects. Keep that.
            let _ = &mut builder;
            let client = builder
                .build()
                .map_err(|e| Error::other(format!("http_prompt_hub client: {e}")))?;
            let mut req = client.get(&url);
            for (k, v) in &headers {
                req = req.header(k, v);
            }
            let resp = req
                .send()
                .map_err(|e| Error::other(format!("http_prompt_hub send: {e}")))?;
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().unwrap_or_default();
                return Err(Error::other(format!(
                    "http_prompt_hub {status} {url}: {body}"
                )));
            }
            resp.text()
                .map_err(|e| Error::other(format!("http_prompt_hub body: {e}")))
        })
        .await
        .map_err(|e| Error::other(format!("http_prompt_hub join: {e}")))??;
        ChatPromptTemplate::from_json(&body)
    }
}

#[cfg(test)]
mod tests {
    //! Tests use a tiny TCP listener so we don't pull in `httptest` /
    //! `wiremock` for a single set of integration tests. The server
    //! handles one connection per test.

    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    /// Stand up a fake HTTP server on a random port. Spawns a thread
    /// that handles one request, sending `response_body` as the body of
    /// a 200 response (or `status_code` if non-200). Returns the
    /// `(base_url, captured_request_lines)` so tests can both interact
    /// and assert on what the client sent.
    fn fake_server(
        status_code: u16,
        response_body: &'static str,
    ) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            if let Ok((mut stream, _addr)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let _ = tx.send(req);

                let status_text = if status_code == 200 { "OK" } else { "ERR" };
                let response = format!(
                    "HTTP/1.1 {status_code} {status_text}\r\n\
                     Content-Type: application/json\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\
                     \r\n\
                     {}",
                    response_body.len(),
                    response_body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });

        (format!("http://127.0.0.1:{port}"), rx)
    }

    const SAMPLE_PROMPT_JSON: &str = r#"{
        "messages": [
            {"role": "system", "template": "You are helpful."},
            {"role": "user", "template": "{{ question }}"}
        ]
    }"#;

    #[tokio::test]
    async fn pull_round_trips_json_to_template() {
        let (base, _rx) = fake_server(200, SAMPLE_PROMPT_JSON);
        let hub = HttpPromptHub::new(base);
        let tmpl = hub.pull("rag/answer").await.unwrap();
        assert_eq!(tmpl.placeholder_names().len(), 0);
        // Round-tripping back to JSON should preserve the system message.
        let s = tmpl.to_json().unwrap();
        assert!(s.contains("You are helpful"));
    }

    #[tokio::test]
    async fn url_uses_versioned_path() {
        let (base, rx) = fake_server(200, SAMPLE_PROMPT_JSON);
        let hub = HttpPromptHub::new(base);
        let _ = hub.pull("rag/answer@v2").await.unwrap();
        let req = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let first_line = req.lines().next().unwrap_or("");
        assert!(first_line.contains("/rag/answer@v2.json"), "{first_line}");
    }

    #[tokio::test]
    async fn url_uses_unversioned_path_when_no_at() {
        let (base, rx) = fake_server(200, SAMPLE_PROMPT_JSON);
        let hub = HttpPromptHub::new(base);
        let _ = hub.pull("p").await.unwrap();
        let req = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let first_line = req.lines().next().unwrap_or("");
        assert!(first_line.contains("/p.json"), "{first_line}");
    }

    #[tokio::test]
    async fn non_200_surfaces_error() {
        let (base, _rx) = fake_server(404, "not found");
        let hub = HttpPromptHub::new(base);
        let err = hub.pull("missing").await.unwrap_err();
        assert!(format!("{err}").contains("404"), "{err}");
    }

    #[tokio::test]
    async fn bearer_token_attached() {
        let (base, rx) = fake_server(200, SAMPLE_PROMPT_JSON);
        let hub = HttpPromptHub::new(base).with_bearer("supersecret");
        let _ = hub.pull("p").await.unwrap();
        let req = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            req.to_lowercase().contains("authorization: bearer supersecret"),
            "request: {req}"
        );
    }

    #[tokio::test]
    async fn custom_header_attached() {
        let (base, rx) = fake_server(200, SAMPLE_PROMPT_JSON);
        let hub = HttpPromptHub::new(base).with_header("X-Workspace", "team-42");
        let _ = hub.pull("p").await.unwrap();
        let req = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            req.to_lowercase().contains("x-workspace: team-42"),
            "request: {req}"
        );
    }

    #[tokio::test]
    async fn malformed_json_surfaces_clean_error() {
        let (base, _rx) = fake_server(200, "{not even close to json");
        let hub = HttpPromptHub::new(base);
        let err = hub.pull("p").await.unwrap_err();
        // ChatPromptTemplate::from_json error gets wrapped — just confirm
        // we got a clean non-panic error.
        let s = format!("{err}");
        assert!(s.to_lowercase().contains("from_json") || s.contains("missing"));
    }

    #[tokio::test]
    async fn traversal_rejected_before_network_call() {
        // No fake_server set up — if this hits the network, the test
        // hangs or errors with a connection refused; we expect the
        // PromptRef parse to short-circuit first.
        let hub = HttpPromptHub::new("http://localhost:1");
        let err = hub.pull("../etc/passwd").await.unwrap_err();
        assert!(format!("{err}").contains("unsafe"));
    }

    #[test]
    fn base_url_trailing_slash_normalised() {
        let hub = HttpPromptHub::new("https://example.com/prompts/");
        assert_eq!(hub.base_url, "https://example.com/prompts");
    }
}
