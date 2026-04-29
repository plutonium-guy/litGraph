//! Generic HTTP request tool. Lets an agent call a user-controlled HTTP
//! endpoint without you wiring a custom `Tool` for every API. Restrict the
//! allowed methods / hosts at construction so the model can't be talked into
//! sending arbitrary traffic.

use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use reqwest::Client;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub struct HttpRequestConfig {
    pub timeout: Duration,
    /// Allowed methods (uppercase). Defaults to GET only — explicit opt-in for
    /// POST/PUT/DELETE since side-effecting calls deserve scrutiny.
    pub allowed_methods: HashSet<String>,
    /// If non-empty, requests are restricted to these host substrings.
    /// Defense-in-depth against the model exfiltrating internal hosts.
    pub allowed_hosts: Vec<String>,
}

impl Default for HttpRequestConfig {
    fn default() -> Self {
        let mut allowed = HashSet::new();
        allowed.insert("GET".to_string());
        Self {
            timeout: Duration::from_secs(20),
            allowed_methods: allowed,
            allowed_hosts: Vec::new(),
        }
    }
}

impl HttpRequestConfig {
    pub fn with_methods<I, S>(mut self, methods: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_methods = methods.into_iter().map(|m| m.into().to_uppercase()).collect();
        self
    }

    pub fn with_allowed_hosts<I, S>(mut self, hosts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_hosts = hosts.into_iter().map(|s| s.into()).collect();
        self
    }
}

pub struct HttpRequestTool {
    cfg: HttpRequestConfig,
    http: Client,
}

impl HttpRequestTool {
    pub fn new(cfg: HttpRequestConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    fn check_host(&self, url: &str) -> Result<()> {
        if self.cfg.allowed_hosts.is_empty() { return Ok(()); }
        if self.cfg.allowed_hosts.iter().any(|h| url.contains(h)) {
            return Ok(());
        }
        Err(Error::invalid(format!(
            "http_request: host not in allowlist (url={url})"
        )))
    }
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn schema(&self) -> ToolSchema {
        let methods: Vec<&str> = self.cfg.allowed_methods.iter().map(|s| s.as_str()).collect();
        let methods_desc = methods.join(", ");
        ToolSchema {
            name: "http_request".into(),
            description: format!(
                "Make an HTTP request. Allowed methods: {methods_desc}. \
                 Returns {{status, headers, body}}. Body is parsed as JSON when possible."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "method": { "type": "string", "description": "HTTP method (GET/POST/etc)." },
                    "url": { "type": "string", "description": "Full URL." },
                    "headers": {
                        "type": "object",
                        "description": "Optional header map.",
                        "additionalProperties": {"type": "string"}
                    },
                    "body": {
                        "description": "Optional JSON request body (for POST/PUT/PATCH)."
                    }
                },
                "required": ["method", "url"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let method = args.get("method").and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("http_request: missing `method`"))?
            .to_uppercase();
        if !self.cfg.allowed_methods.contains(&method) {
            return Err(Error::invalid(format!(
                "http_request: method `{method}` not in allowlist"
            )));
        }
        let url = args.get("url").and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("http_request: missing `url`"))?;
        self.check_host(url)?;

        let m = reqwest::Method::from_bytes(method.as_bytes())
            .map_err(|e| Error::invalid(format!("http_request: bad method: {e}")))?;
        let mut req = self.http.request(m, url);
        if let Some(headers) = args.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in headers {
                if let Some(s) = v.as_str() {
                    req = req.header(k, s);
                }
            }
        }
        if let Some(body) = args.get("body") {
            if !body.is_null() {
                req = req.json(body);
            }
        }
        let resp = req.send().await.map_err(|e| Error::other(format!("http_request send: {e}")))?;
        let status = resp.status().as_u16();
        let mut headers_map = serde_json::Map::new();
        for (k, v) in resp.headers() {
            if let Ok(s) = v.to_str() {
                headers_map.insert(k.as_str().to_string(), Value::String(s.to_string()));
            }
        }
        let bytes = resp.bytes().await.map_err(|e| Error::other(format!("http_request body: {e}")))?;
        let body_text = String::from_utf8_lossy(&bytes).to_string();
        let body_value: Value = match serde_json::from_str(&body_text) {
            Ok(v) => v,
            Err(_) => Value::String(body_text),
        };
        Ok(json!({
            "status": status,
            "headers": headers_map,
            "body": body_value,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn start_fake() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let _ = s.read(&mut buf);
                let body = b"{\"hello\":\"world\"}";
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(header.as_bytes());
                let _ = s.write_all(body);
            }
        });
        port
    }

    #[tokio::test]
    async fn get_returns_status_headers_body() {
        let port = start_fake();
        let t = HttpRequestTool::new(HttpRequestConfig::default()).unwrap();
        let out = t.run(json!({
            "method": "GET",
            "url": format!("http://127.0.0.1:{port}/"),
        })).await.unwrap();
        assert_eq!(out["status"], json!(200));
        assert_eq!(out["body"], json!({"hello": "world"}));
        assert!(out["headers"]["content-type"].as_str().unwrap().contains("json"));
    }

    #[tokio::test]
    async fn rejects_disallowed_method() {
        let t = HttpRequestTool::new(HttpRequestConfig::default()).unwrap();
        let err = t.run(json!({"method":"POST","url":"http://x/"})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn allowed_methods_extension_works() {
        let cfg = HttpRequestConfig::default().with_methods(["GET", "POST"]);
        let t = HttpRequestTool::new(cfg).unwrap();
        // POST to an unreachable host fails with `other` (network), NOT InvalidInput.
        let err = t.run(json!({"method":"POST","url":"http://127.0.0.1:1/"})).await.unwrap_err();
        assert!(!matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn host_allowlist_blocks_unauthorized_urls() {
        let cfg = HttpRequestConfig::default()
            .with_allowed_hosts(["api.allowed.com"]);
        let t = HttpRequestTool::new(cfg).unwrap();
        let err = t.run(json!({"method":"GET","url":"http://internal.bad/"})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }
}
