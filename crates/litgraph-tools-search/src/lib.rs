//! Built-in web search tools for ReactAgent. Each implements `litgraph_core::tool::Tool`.
//!
//! - `BraveSearch` — Brave Search API (`X-Subscription-Token`)
//! - `TavilySearch` — Tavily API (POST with `api_key` in body)
//!
//! Both share the same args schema: `{"query": "...", "max_results": int?}`.
//! Both return `{"results": [{"title", "url", "snippet"}, ...]}` so the agent
//! sees a uniform format regardless of which engine is wired up.

use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use reqwest::Client;
use serde_json::{Value, json};

const SHARED_PARAMS: &str = r#"{
    "type": "object",
    "properties": {
        "query": { "type": "string", "description": "Search query." },
        "max_results": {
            "type": "integer",
            "description": "Max number of results to return (default 5, max 20).",
            "default": 5
        }
    },
    "required": ["query"]
}"#;

fn shared_params() -> Value {
    serde_json::from_str(SHARED_PARAMS).unwrap()
}

#[derive(Debug, Clone)]
pub struct BraveSearchConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout: Duration,
}

impl BraveSearchConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.search.brave.com".into(),
            timeout: Duration::from_secs(20),
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

pub struct BraveSearch {
    cfg: BraveSearchConfig,
    http: Client,
}

impl BraveSearch {
    pub fn new(cfg: BraveSearchConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }
}

#[async_trait]
impl Tool for BraveSearch {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "web_search".into(),
            description: "Search the public web via Brave Search. Returns title, url, and snippet for the top results.".into(),
            parameters: shared_params(),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let q = args.get("query").and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("brave_search: missing `query`"))?;
        let n = args.get("max_results").and_then(|v| v.as_u64()).unwrap_or(5).clamp(1, 20) as u32;
        let url = format!("{}/res/v1/web/search?q={}&count={}",
                          self.cfg.base_url.trim_end_matches('/'),
                          urlencoded(q), n);
        let resp = self
            .http
            .get(&url)
            .header("Accept", "application/json")
            .header("X-Subscription-Token", &self.cfg.api_key)
            .send()
            .await
            .map_err(|e| Error::other(format!("brave send: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            if s.as_u16() == 429 { return Err(Error::RateLimited { retry_after_ms: None }); }
            return Err(Error::other(format!("brave {s}: {t}")));
        }
        let v: Value = resp.json().await.map_err(|e| Error::other(format!("brave decode: {e}")))?;
        let mut out = Vec::new();
        if let Some(results) = v.get("web").and_then(|w| w.get("results")).and_then(|r| r.as_array()) {
            for r in results.iter().take(n as usize) {
                out.push(json!({
                    "title": r.get("title").and_then(|t| t.as_str()).unwrap_or(""),
                    "url": r.get("url").and_then(|t| t.as_str()).unwrap_or(""),
                    "snippet": r.get("description").and_then(|t| t.as_str()).unwrap_or(""),
                }));
            }
        }
        Ok(json!({ "results": out }))
    }
}

#[derive(Debug, Clone)]
pub struct TavilyConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout: Duration,
    /// Tavily-specific search depth: "basic" or "advanced". advanced costs more credits.
    pub search_depth: String,
}

impl TavilyConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.tavily.com".into(),
            timeout: Duration::from_secs(30),
            search_depth: "basic".into(),
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_search_depth(mut self, depth: impl Into<String>) -> Self {
        self.search_depth = depth.into();
        self
    }
}

pub struct TavilySearch {
    cfg: TavilyConfig,
    http: Client,
}

impl TavilySearch {
    pub fn new(cfg: TavilyConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }
}

#[async_trait]
impl Tool for TavilySearch {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "web_search".into(),
            description: "Search the public web via Tavily. Returns title, url, and snippet for the top results.".into(),
            parameters: shared_params(),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let q = args.get("query").and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("tavily_search: missing `query`"))?;
        let n = args.get("max_results").and_then(|v| v.as_u64()).unwrap_or(5).clamp(1, 20) as u32;
        let url = format!("{}/search", self.cfg.base_url.trim_end_matches('/'));
        let body = json!({
            "api_key": self.cfg.api_key,
            "query": q,
            "max_results": n,
            "search_depth": self.cfg.search_depth,
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("tavily send: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("tavily {s}: {t}")));
        }
        let v: Value = resp.json().await.map_err(|e| Error::other(format!("tavily decode: {e}")))?;
        let mut out = Vec::new();
        if let Some(results) = v.get("results").and_then(|r| r.as_array()) {
            for r in results.iter().take(n as usize) {
                out.push(json!({
                    "title": r.get("title").and_then(|t| t.as_str()).unwrap_or(""),
                    "url": r.get("url").and_then(|t| t.as_str()).unwrap_or(""),
                    "snippet": r.get("content").and_then(|t| t.as_str()).unwrap_or(""),
                }));
            }
        }
        Ok(json!({ "results": out }))
    }
}

/// Tavily /extract — fetch full article text for a list of URLs. Pairs
/// with `TavilySearch` to complete the web-research loop: search returns
/// snippets, extract returns full bodies. Much better than hand-rolled
/// HTML scraping — Tavily does readability extraction server-side.
///
/// Input: `{"urls": [url1, url2, ...], "extract_depth"?: "basic"|"advanced"}`.
/// The tool silently caps at 20 URLs per call (Tavily's own limit).
///
/// Returns: `{"results": [{"url", "content"}, ...], "failed_results"?: [...]}`
/// — failures (unreachable / robots-blocked / paywall) are surfaced in a
/// separate array so the caller can decide whether to retry or skip.
pub struct TavilyExtract {
    cfg: TavilyConfig,
    http: Client,
}

impl TavilyExtract {
    pub fn new(cfg: TavilyConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }
}

#[async_trait]
impl Tool for TavilyExtract {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "web_extract".into(),
            description: "Fetch full article text from a list of URLs via Tavily \
                          /extract. Use after `web_search` to read the top results \
                          rather than relying on snippets."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs to extract (1-20). Supply the URLs from web_search results."
                    },
                    "extract_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "`basic` is faster/cheaper; `advanced` pulls more complete text for JS-heavy pages. Default: basic."
                    }
                },
                "required": ["urls"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        // Accept either a JSON array OR a single string (agents sometimes
        // pass one URL as a bare string; don't punish them for it).
        let urls: Vec<String> = match args.get("urls") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            Some(Value::String(s)) => vec![s.clone()],
            _ => {
                return Err(Error::invalid(
                    "tavily_extract: missing `urls` (expected array of strings)",
                ));
            }
        };
        if urls.is_empty() {
            return Err(Error::invalid("tavily_extract: `urls` is empty"));
        }
        // Tavily caps at 20 per call; truncate silently.
        let urls: Vec<String> = urls.into_iter().take(20).collect();

        let extract_depth = args
            .get("extract_depth")
            .and_then(|v| v.as_str())
            .unwrap_or("basic");

        let url = format!("{}/extract", self.cfg.base_url.trim_end_matches('/'));
        let body = json!({
            "api_key": self.cfg.api_key,
            "urls": urls,
            "extract_depth": extract_depth,
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("tavily extract send: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("tavily extract {s}: {t}")));
        }
        let v: Value = resp
            .json()
            .await
            .map_err(|e| Error::other(format!("tavily extract decode: {e}")))?;

        // Normalize the response: Tavily returns `raw_content` but we
        // expose it as `content` for consistency with other loaders.
        let mut out = Vec::new();
        if let Some(results) = v.get("results").and_then(|r| r.as_array()) {
            for r in results {
                out.push(json!({
                    "url": r.get("url").and_then(|t| t.as_str()).unwrap_or(""),
                    "content": r
                        .get("raw_content")
                        .or_else(|| r.get("content"))
                        .and_then(|t| t.as_str())
                        .unwrap_or(""),
                }));
            }
        }
        let mut failed = Vec::new();
        if let Some(fails) = v.get("failed_results").and_then(|r| r.as_array()) {
            for f in fails {
                failed.push(json!({
                    "url": f.get("url").and_then(|t| t.as_str()).unwrap_or(""),
                    "error": f.get("error").and_then(|t| t.as_str()).unwrap_or("unknown"),
                }));
            }
        }
        Ok(json!({"results": out, "failed_results": failed}))
    }
}

/// DuckDuckGo via the Instant-Answer JSON endpoint — the only stable
/// no-API-key escape hatch. Returns the AbstractText (Wikipedia-style summary)
/// and any RelatedTopics that have URLs. Limited vs Brave/Tavily — for
/// "sky is blue why" works great; for "GitHub repo for X" mostly returns the
/// disambiguation list. Document the trade-off; don't pretend it's a Brave
/// replacement.
#[derive(Debug, Clone)]
pub struct DuckDuckGoConfig {
    pub base_url: String,
    pub timeout: Duration,
}

impl Default for DuckDuckGoConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.duckduckgo.com".into(),
            timeout: Duration::from_secs(20),
        }
    }
}

impl DuckDuckGoConfig {
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

pub struct DuckDuckGoSearch {
    cfg: DuckDuckGoConfig,
    http: Client,
}

impl DuckDuckGoSearch {
    pub fn new(cfg: DuckDuckGoConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }
}

#[async_trait]
impl Tool for DuckDuckGoSearch {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "web_search".into(),
            description: "Search the public web via DuckDuckGo's Instant Answer API (no API key required). \
                          Best for definitions, encyclopedic queries, and disambiguation. \
                          For deep web search, prefer BraveSearch or TavilySearch."
                .into(),
            parameters: shared_params(),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let q = args.get("query").and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("duckduckgo_search: missing `query`"))?;
        let n = args.get("max_results").and_then(|v| v.as_u64()).unwrap_or(5).clamp(1, 20) as usize;
        // Instant Answer endpoint params: format=json, no_html=1 (strip HTML
        // out of summaries), no_redirect=1 (don't bounce !-bang queries),
        // skip_disambig=1 (return RelatedTopics directly instead of "did you
        // mean" prompts).
        let url = format!(
            "{}/?q={}&format=json&no_html=1&no_redirect=1&skip_disambig=1",
            self.cfg.base_url.trim_end_matches('/'),
            urlencoded(q),
        );
        let resp = self
            .http
            .get(&url)
            .header("user-agent", "litgraph/0.1")
            .send()
            .await
            .map_err(|e| Error::other(format!("ddg send: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("ddg {s}: {t}")));
        }
        // DDG sometimes returns content-type=application/x-javascript even for
        // JSON bodies; pull the text and parse manually.
        let text = resp.text().await.map_err(|e| Error::other(format!("ddg body: {e}")))?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| Error::other(format!("ddg decode: {e}")))?;

        let mut out = Vec::with_capacity(n);

        // 1. Top-level Abstract (Wikipedia summary). Best signal when present.
        let abstract_text = v.get("AbstractText").and_then(|s| s.as_str()).unwrap_or("");
        let abstract_url = v.get("AbstractURL").and_then(|s| s.as_str()).unwrap_or("");
        if !abstract_text.is_empty() && !abstract_url.is_empty() {
            out.push(json!({
                "title": v.get("Heading").and_then(|s| s.as_str()).unwrap_or("Summary"),
                "url": abstract_url,
                "snippet": abstract_text,
            }));
        }

        // 2. RelatedTopics — flatten one level (DDG nests by category sometimes).
        if out.len() < n {
            if let Some(topics) = v.get("RelatedTopics").and_then(|t| t.as_array()) {
                for topic in topics {
                    if out.len() >= n { break; }
                    // Nested form: {Name: "Category", Topics: [...]}
                    if let Some(nested) = topic.get("Topics").and_then(|t| t.as_array()) {
                        for t in nested {
                            if out.len() >= n { break; }
                            if let Some(item) = topic_to_result(t) { out.push(item); }
                        }
                    } else if let Some(item) = topic_to_result(topic) {
                        out.push(item);
                    }
                }
            }
        }

        Ok(json!({ "results": out }))
    }
}

fn topic_to_result(t: &Value) -> Option<Value> {
    let url = t.get("FirstURL").and_then(|s| s.as_str()).filter(|s| !s.is_empty())?;
    let text = t.get("Text").and_then(|s| s.as_str()).unwrap_or("");
    // The Text often starts with "Title - description"; split on " - " to
    // separate title from snippet when possible.
    let (title, snippet) = match text.split_once(" - ") {
        Some((t, s)) => (t.to_string(), s.to_string()),
        None => (text.to_string(), text.to_string()),
    };
    Some(json!({ "title": title, "url": url, "snippet": snippet }))
}

fn urlencoded(s: &str) -> String {
    s.chars().map(|c| match c {
        'a'..='z'|'A'..='Z'|'0'..='9'|'-'|'_'|'.'|'~' => c.to_string(),
        _ => format!("%{:02X}", c as u8),
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn start_fake(method: &'static str, response: &str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let body = response.to_string();
        let m = method.to_string();
        thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let mut total = Vec::new();
                loop {
                    let n = match s.read(&mut buf) {
                        Ok(0) => break, Ok(n) => n, Err(_) => break,
                    };
                    total.extend_from_slice(&buf[..n]);
                    if total.windows(4).any(|w| w == b"\r\n\r\n") {
                        let s_total = String::from_utf8_lossy(&total).to_string();
                        let cl = s_total
                            .split("\r\n")
                            .find_map(|l| l.strip_prefix("content-length: ").or_else(|| l.strip_prefix("Content-Length: ")))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        let header_end = total.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                        // Confirm method matches (sanity)
                        assert!(s_total.starts_with(&format!("{} ", m)),
                                "expected method {m}, got: {}", &s_total[..20]);
                        while total.len() < header_end + cl {
                            let n = match s.read(&mut buf) {
                                Ok(0) => break, Ok(n) => n, Err(_) => break,
                            };
                            total.extend_from_slice(&buf[..n]);
                        }
                        break;
                    }
                }
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(header.as_bytes());
                let _ = s.write_all(body.as_bytes());
            }
        });
        port
    }

    #[tokio::test]
    async fn brave_returns_normalized_results() {
        let port = start_fake("GET", r#"{"web":{"results":[
            {"title":"Rust","url":"https://rust-lang.org","description":"systems lang"},
            {"title":"Cargo","url":"https://doc.rust-lang.org/cargo","description":"package mgr"}
        ]}}"#);
        let cfg = BraveSearchConfig::new("brave-key")
            .with_base_url(format!("http://127.0.0.1:{port}"));
        let t = BraveSearch::new(cfg).unwrap();
        let out = t.run(json!({"query":"rust","max_results":2})).await.unwrap();
        let results = out["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["title"], json!("Rust"));
        assert_eq!(results[0]["url"], json!("https://rust-lang.org"));
        assert_eq!(results[0]["snippet"], json!("systems lang"));
    }

    #[tokio::test]
    async fn tavily_returns_normalized_results() {
        let port = start_fake("POST", r#"{"results":[
            {"title":"Tokio","url":"https://tokio.rs","content":"async runtime"},
            {"title":"Axum","url":"https://docs.rs/axum","content":"web fwk"}
        ]}"#);
        let cfg = TavilyConfig::new("tvly-key")
            .with_base_url(format!("http://127.0.0.1:{port}"));
        let t = TavilySearch::new(cfg).unwrap();
        let out = t.run(json!({"query":"async rust"})).await.unwrap();
        let results = out["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["title"], json!("Tokio"));
        assert_eq!(results[1]["snippet"], json!("web fwk"));
    }

    #[tokio::test]
    async fn missing_query_returns_invalid_input() {
        let cfg = BraveSearchConfig::new("k");
        let t = BraveSearch::new(cfg).unwrap();
        let err = t.run(json!({})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn schema_shape_is_consistent() {
        let cfg = BraveSearchConfig::new("k");
        let t = BraveSearch::new(cfg).unwrap();
        let s = t.schema();
        assert_eq!(s.name, "web_search");
        assert_eq!(s.parameters["properties"]["query"]["type"], json!("string"));
        assert!(s.parameters["required"].as_array().unwrap().contains(&json!("query")));
    }

    #[tokio::test]
    async fn ddg_extracts_abstract_then_related_topics() {
        let body = r#"{
            "Heading": "Rust (programming language)",
            "AbstractText": "Rust is a multi-paradigm systems programming language.",
            "AbstractURL": "https://en.wikipedia.org/wiki/Rust_(programming_language)",
            "RelatedTopics": [
                {"Text": "Cargo - Rust's package manager", "FirstURL": "https://example.com/cargo"},
                {"Name": "Languages", "Topics": [
                    {"Text": "Mojo - newer language", "FirstURL": "https://example.com/mojo"}
                ]},
                {"Text": "no url here", "FirstURL": ""}
            ]
        }"#;
        let port = start_fake("GET", body);
        let cfg = DuckDuckGoConfig::default()
            .with_base_url(format!("http://127.0.0.1:{port}"));
        let t = DuckDuckGoSearch::new(cfg).unwrap();
        let out = t.run(json!({"query": "rust language", "max_results": 5})).await.unwrap();
        let results = out["results"].as_array().unwrap();
        assert_eq!(results.len(), 3, "got: {results:?}");
        // 1. Abstract first.
        assert_eq!(results[0]["title"], json!("Rust (programming language)"));
        assert_eq!(results[0]["url"], json!("https://en.wikipedia.org/wiki/Rust_(programming_language)"));
        // 2. Related topic with " - " split.
        assert_eq!(results[1]["title"], json!("Cargo"));
        assert_eq!(results[1]["snippet"], json!("Rust's package manager"));
        // 3. Nested topic flattened.
        assert_eq!(results[2]["title"], json!("Mojo"));
        assert_eq!(results[2]["url"], json!("https://example.com/mojo"));
    }

    #[tokio::test]
    async fn ddg_max_results_caps_output() {
        let body = r#"{
            "AbstractText": "abstract here",
            "AbstractURL": "https://x.example",
            "Heading": "X",
            "RelatedTopics": [
                {"Text": "A - desc A", "FirstURL": "https://a"},
                {"Text": "B - desc B", "FirstURL": "https://b"},
                {"Text": "C - desc C", "FirstURL": "https://c"}
            ]
        }"#;
        let port = start_fake("GET", body);
        let cfg = DuckDuckGoConfig::default().with_base_url(format!("http://127.0.0.1:{port}"));
        let t = DuckDuckGoSearch::new(cfg).unwrap();
        let out = t.run(json!({"query": "x", "max_results": 2})).await.unwrap();
        // max_results=2 → abstract + 1 related, total 2.
        assert_eq!(out["results"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn ddg_query_url_encodes_special_chars() {
        // The fake server doesn't validate the URL — but verify the helper
        // doesn't panic on spaces/punctuation.
        let port = start_fake("GET", r#"{"AbstractText":"","AbstractURL":"","RelatedTopics":[]}"#);
        let cfg = DuckDuckGoConfig::default().with_base_url(format!("http://127.0.0.1:{port}"));
        let t = DuckDuckGoSearch::new(cfg).unwrap();
        let out = t.run(json!({"query": "what is C++?"})).await.unwrap();
        assert_eq!(out["results"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn ddg_schema_well_formed() {
        let t = DuckDuckGoSearch::new(DuckDuckGoConfig::default()).unwrap();
        let s = t.schema();
        assert_eq!(s.name, "web_search");
        assert!(s.description.contains("DuckDuckGo"));
    }

    // ----- TavilyExtract tests -----

    #[tokio::test]
    async fn tavily_extract_returns_normalized_results() {
        let body = r#"{
            "results": [
                {"url": "https://a.example", "raw_content": "full body of A"},
                {"url": "https://b.example", "raw_content": "full body of B"}
            ],
            "failed_results": []
        }"#;
        let port = start_fake("POST", body);
        let cfg = TavilyConfig::new("tvly-key")
            .with_base_url(format!("http://127.0.0.1:{port}"));
        let t = TavilyExtract::new(cfg).unwrap();
        let out = t
            .run(json!({"urls": ["https://a.example", "https://b.example"]}))
            .await
            .unwrap();
        let results = out["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["url"], json!("https://a.example"));
        assert_eq!(results[0]["content"], json!("full body of A"));
        assert_eq!(results[1]["content"], json!("full body of B"));
        assert_eq!(out["failed_results"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn tavily_extract_surfaces_failed_results_array() {
        let body = r#"{
            "results": [{"url": "https://ok", "raw_content": "ok content"}],
            "failed_results": [
                {"url": "https://paywall", "error": "403 Forbidden"},
                {"url": "https://dead", "error": "connection refused"}
            ]
        }"#;
        let port = start_fake("POST", body);
        let cfg = TavilyConfig::new("tvly-key")
            .with_base_url(format!("http://127.0.0.1:{port}"));
        let t = TavilyExtract::new(cfg).unwrap();
        let out = t
            .run(json!({"urls": ["https://ok", "https://paywall", "https://dead"]}))
            .await
            .unwrap();
        assert_eq!(out["results"].as_array().unwrap().len(), 1);
        let failed = out["failed_results"].as_array().unwrap();
        assert_eq!(failed.len(), 2);
        assert_eq!(failed[0]["url"], json!("https://paywall"));
        assert_eq!(failed[0]["error"], json!("403 Forbidden"));
    }

    #[tokio::test]
    async fn tavily_extract_accepts_single_url_string() {
        // Agents sometimes pass a bare string; don't punish them.
        let body = r#"{"results":[{"url":"https://x","raw_content":"body"}]}"#;
        let port = start_fake("POST", body);
        let cfg = TavilyConfig::new("k").with_base_url(format!("http://127.0.0.1:{port}"));
        let t = TavilyExtract::new(cfg).unwrap();
        let out = t.run(json!({"urls": "https://x"})).await.unwrap();
        assert_eq!(out["results"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn tavily_extract_missing_urls_returns_invalid_input() {
        let cfg = TavilyConfig::new("k");
        let t = TavilyExtract::new(cfg).unwrap();
        let err = t.run(json!({})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn tavily_extract_empty_urls_returns_invalid_input() {
        let cfg = TavilyConfig::new("k");
        let t = TavilyExtract::new(cfg).unwrap();
        let err = t.run(json!({"urls": []})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn tavily_extract_schema_has_urls_array_required() {
        let cfg = TavilyConfig::new("k");
        let t = TavilyExtract::new(cfg).unwrap();
        let s = t.schema();
        assert_eq!(s.name, "web_extract");
        assert_eq!(s.parameters["properties"]["urls"]["type"], json!("array"));
        assert!(s.parameters["required"].as_array().unwrap().contains(&json!("urls")));
    }

    #[tokio::test]
    async fn tavily_extract_falls_back_to_content_if_no_raw_content() {
        // Some Tavily responses use `content` instead of `raw_content`
        // for short pages. Both should surface.
        let body = r#"{"results":[{"url":"https://x","content":"short body"}]}"#;
        let port = start_fake("POST", body);
        let cfg = TavilyConfig::new("k").with_base_url(format!("http://127.0.0.1:{port}"));
        let t = TavilyExtract::new(cfg).unwrap();
        let out = t.run(json!({"urls": ["https://x"]})).await.unwrap();
        assert_eq!(out["results"][0]["content"], json!("short body"));
    }
}
