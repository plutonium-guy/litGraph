//! Free URL→clean-text Tool. Fetches a URL via reqwest, strips HTML
//! tags / scripts / styles / boilerplate via the same `strip_html`
//! routine that powers `HtmlLoader`. No API key, no third-party
//! service — just an `Authorization`-free HTTP GET.
//!
//! # vs `TavilyExtract` (iter 141)
//!
//! - **Tavily**: server-side readability extraction (better for JS-heavy
//!   pages; needs API key; costs credits).
//! - **WebFetchTool**: pure-Rust regex strip (works for static HTML; no
//!   key required; free).
//!
//! Use Tavily when extraction quality matters more than cost. Use
//! WebFetchTool for known-good static sites (docs, README pages,
//! Wikipedia). They share the same purpose; choose by source quality.
//!
//! # vs `HttpRequestTool` (iter ~30)
//!
//! `HttpRequestTool` returns RAW response bodies (HTML soup). WebFetchTool
//! adds the strip step so the agent gets readable text without burning
//! tokens on `<div class="...">` markup.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::{Error, Result, Tool, ToolSchema};
use litgraph_loaders::html::strip_html;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct WebFetchConfig {
    pub timeout: Duration,
    /// Cap on returned text length. 0 = no cap. Default 16384 (~4K tokens).
    /// Agents should pass `max_chars` per call when they want different limits.
    pub default_max_chars: usize,
    /// User-Agent header value. Some sites 403 on the default reqwest UA.
    pub user_agent: String,
    /// Strip <nav>, <header>, <footer>, <aside> blocks before text extraction.
    /// Default true — those are usually navigation noise.
    pub strip_boilerplate: bool,
}

impl Default for WebFetchConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            default_max_chars: 16384,
            user_agent: "litgraph-web-fetch/0.1".into(),
            strip_boilerplate: true,
        }
    }
}

impl WebFetchConfig {
    pub fn new() -> Self { Self::default() }
    pub fn with_timeout(mut self, t: Duration) -> Self { self.timeout = t; self }
    pub fn with_default_max_chars(mut self, n: usize) -> Self { self.default_max_chars = n; self }
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self { self.user_agent = ua.into(); self }
    pub fn keep_boilerplate(mut self) -> Self { self.strip_boilerplate = false; self }
}

pub struct WebFetchTool {
    cfg: Arc<WebFetchConfig>,
    http: Client,
}

impl WebFetchTool {
    pub fn new(cfg: WebFetchConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .user_agent(&cfg.user_agent)
            .build()
            .map_err(|e| Error::other(format!("web_fetch build: {e}")))?;
        Ok(Self { cfg: Arc::new(cfg), http })
    }
}

#[derive(Debug, Deserialize)]
struct FetchArgs {
    url: String,
    /// Override the config's default_max_chars (None → use default).
    /// 0 means no cap.
    #[serde(default)]
    max_chars: Option<usize>,
}

#[async_trait]
impl Tool for WebFetchTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "web_fetch".into(),
            description: "Fetch a URL and return its clean text content (HTML tags / scripts / \
                          styles / nav-boilerplate stripped). Use for known-good static pages \
                          (docs, README, Wikipedia). For JS-heavy sites, prefer `web_extract` \
                          (Tavily) if available.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute URL (http/https)."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional cap on returned text length. 0 = no cap. Default ~16K chars."
                    }
                },
                "required": ["url"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let parsed: FetchArgs = serde_json::from_value(args)
            .map_err(|e| Error::invalid(format!("web_fetch args: {e}")))?;
        if !(parsed.url.starts_with("http://") || parsed.url.starts_with("https://")) {
            return Err(Error::invalid(
                "web_fetch: `url` must start with http:// or https://",
            ));
        }
        let resp = self
            .http
            .get(&parsed.url)
            .send()
            .await
            .map_err(|e| Error::other(format!("web_fetch send: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            // Read body for the error message but cap at 1KB so a 1MB error
            // page doesn't blow up the agent's context.
            let mut body = resp.text().await.unwrap_or_default();
            if body.len() > 1024 {
                body.truncate(1024);
            }
            return Err(Error::other(format!("web_fetch {status}: {body}")));
        }
        let raw = resp
            .text()
            .await
            .map_err(|e| Error::other(format!("web_fetch decode: {e}")))?;
        let mut text = strip_html(&raw, self.cfg.strip_boilerplate);

        let cap = parsed.max_chars.unwrap_or(self.cfg.default_max_chars);
        let truncated = cap > 0 && text.len() > cap;
        if truncated {
            text.truncate(cap);
        }
        Ok(json!({
            "url": parsed.url,
            "text": text,
            "truncated": truncated,
            "char_count": text.len(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn start_fake(status: u16, content_type: &'static str, body: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = s.read(&mut buf);  // drain headers; ignore
                let header = format!(
                    "HTTP/1.1 {} OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    status, content_type, body.len()
                );
                let _ = s.write_all(header.as_bytes());
                let _ = s.write_all(body.as_bytes());
            }
        });
        port
    }

    #[tokio::test]
    async fn fetches_url_and_returns_clean_text() {
        let html = "<html><body><p>Hello <b>world</b>!</p><p>Second paragraph.</p></body></html>";
        let port = start_fake(200, "text/html", html);
        let tool = WebFetchTool::new(WebFetchConfig::default()).unwrap();
        let resp = tool.run(json!({"url": format!("http://127.0.0.1:{port}/")})).await.unwrap();
        let text = resp["text"].as_str().unwrap();
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(text.contains("Second paragraph"));
        // No HTML tags in output.
        assert!(!text.contains("<p>"));
        assert!(!text.contains("<body>"));
        assert_eq!(resp["truncated"], false);
    }

    #[tokio::test]
    async fn strips_script_and_style() {
        let html = r#"<html><head><style>body { color: red }</style></head><body><script>alert('x')</script><p>visible</p></body></html>"#;
        let port = start_fake(200, "text/html", html);
        let tool = WebFetchTool::new(WebFetchConfig::default()).unwrap();
        let text = tool.run(json!({"url": format!("http://127.0.0.1:{port}/")})).await.unwrap()["text"].as_str().unwrap().to_string();
        assert!(text.contains("visible"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("color: red"));
    }

    #[tokio::test]
    async fn strips_boilerplate_by_default() {
        let html = r#"<html><body><nav>navigation</nav><header>top bar</header><main>real content</main><footer>copyright</footer></body></html>"#;
        let port = start_fake(200, "text/html", html);
        let tool = WebFetchTool::new(WebFetchConfig::default()).unwrap();
        let text = tool.run(json!({"url": format!("http://127.0.0.1:{port}/")})).await.unwrap()["text"].as_str().unwrap().to_string();
        assert!(text.contains("real content"));
        assert!(!text.contains("navigation"));
        assert!(!text.contains("copyright"));
    }

    #[tokio::test]
    async fn keep_boilerplate_includes_nav() {
        let html = r#"<html><body><nav>navigation</nav><main>real content</main></body></html>"#;
        let port = start_fake(200, "text/html", html);
        let tool = WebFetchTool::new(WebFetchConfig::default().keep_boilerplate()).unwrap();
        let text = tool.run(json!({"url": format!("http://127.0.0.1:{port}/")})).await.unwrap()["text"].as_str().unwrap().to_string();
        assert!(text.contains("navigation"));
        assert!(text.contains("real content"));
    }

    #[tokio::test]
    async fn max_chars_truncates_and_flags() {
        let html = "<html><body><p>".to_string() + &"x".repeat(1000) + "</p></body></html>";
        let port = start_fake(200, "text/html", Box::leak(html.into_boxed_str()));
        let tool = WebFetchTool::new(WebFetchConfig::default()).unwrap();
        let resp = tool.run(json!({"url": format!("http://127.0.0.1:{port}/"), "max_chars": 50})).await.unwrap();
        assert_eq!(resp["truncated"], true);
        assert!(resp["text"].as_str().unwrap().len() <= 50);
    }

    #[tokio::test]
    async fn max_chars_zero_means_no_cap() {
        let html = "<html><body><p>".to_string() + &"x".repeat(500) + "</p></body></html>";
        let port = start_fake(200, "text/html", Box::leak(html.into_boxed_str()));
        let tool = WebFetchTool::new(WebFetchConfig::default()).unwrap();
        let resp = tool.run(json!({"url": format!("http://127.0.0.1:{port}/"), "max_chars": 0})).await.unwrap();
        assert_eq!(resp["truncated"], false);
        assert!(resp["text"].as_str().unwrap().len() >= 500);
    }

    #[tokio::test]
    async fn non_http_url_returns_invalid_input() {
        let tool = WebFetchTool::new(WebFetchConfig::default()).unwrap();
        let err = tool.run(json!({"url": "ftp://example.com/file"})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn missing_url_returns_invalid_input() {
        let tool = WebFetchTool::new(WebFetchConfig::default()).unwrap();
        let err = tool.run(json!({})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn http_error_surfaces_with_status() {
        let port = start_fake(404, "text/html", "<html><body>not found</body></html>");
        let tool = WebFetchTool::new(WebFetchConfig::default()).unwrap();
        let err = tool.run(json!({"url": format!("http://127.0.0.1:{port}/")})).await.unwrap_err();
        let s = err.to_string();
        assert!(s.contains("404"));
    }

    #[tokio::test]
    async fn very_long_error_body_truncated_in_message() {
        let huge = "X".repeat(50_000);
        let port = start_fake(500, "text/plain", Box::leak(huge.into_boxed_str()));
        let tool = WebFetchTool::new(WebFetchConfig::default()).unwrap();
        let err = tool.run(json!({"url": format!("http://127.0.0.1:{port}/")})).await.unwrap_err();
        // The error message itself should be bounded — agent context guard.
        assert!(err.to_string().len() < 2000);
    }

    #[tokio::test]
    async fn schema_has_url_required() {
        let tool = WebFetchTool::new(WebFetchConfig::default()).unwrap();
        let s = tool.schema();
        assert_eq!(s.name, "web_fetch");
        assert!(s.parameters["required"].as_array().unwrap().iter().any(|v| v == "url"));
    }

    #[tokio::test]
    async fn returns_url_and_char_count_in_response() {
        let html = "<html><body><p>hi there</p></body></html>";
        let port = start_fake(200, "text/html", html);
        let url = format!("http://127.0.0.1:{port}/article");
        let tool = WebFetchTool::new(WebFetchConfig::default()).unwrap();
        let resp = tool.run(json!({"url": url.clone()})).await.unwrap();
        assert_eq!(resp["url"], url);
        let cc = resp["char_count"].as_u64().unwrap();
        assert!(cc > 0 && cc < 100);
    }
}
