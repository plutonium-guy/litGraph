//! Voyage AI rerank API adapter — wraps any base `Retriever` with cross-encoder
//! reranking via Voyage's `/v1/rerank` endpoint.
//!
//! Voyage offers `rerank-2` (best quality) and `rerank-2-lite` (cheaper, lower
//! latency). Pairs naturally with `litgraph-providers-voyage::VoyageEmbeddings`.
//!
//! # Wire format
//!
//! POST `https://api.voyageai.com/v1/rerank`
//!
//! Body:
//! ```json
//! {
//!   "query": "the user's query",
//!   "documents": ["text1", "text2", ...],
//!   "model": "rerank-2",
//!   "top_k": 5,
//!   "truncation": true,
//!   "return_documents": false
//! }
//! ```
//!
//! Response:
//! ```json
//! {
//!   "object": "list",
//!   "data": [
//!     { "index": 3, "relevance_score": 0.95 },
//!     { "index": 0, "relevance_score": 0.87 }
//!   ],
//!   "model": "rerank-2",
//!   "usage": { "total_tokens": 123 }
//! }
//! ```
//!
//! Result indices map back to the input documents and the relevance score is
//! written onto `Document::score`.

use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::{Document, Error, Result};
use litgraph_retrieval::Reranker;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Clone)]
pub struct VoyageRerankConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    /// If true, ask Voyage to truncate inputs that exceed the model's context.
    /// Default true (matches Voyage's own default and avoids surprise 400s on
    /// long docs).
    pub truncation: bool,
}

impl VoyageRerankConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.voyageai.com/v1".into(),
            model: model.into(),
            timeout: Duration::from_secs(60),
            truncation: true,
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_truncation(mut self, t: bool) -> Self {
        self.truncation = t;
        self
    }
}

pub struct VoyageReranker {
    cfg: VoyageRerankConfig,
    http: Client,
}

impl VoyageReranker {
    pub fn new(cfg: VoyageRerankConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }
}

#[derive(Deserialize)]
struct RerankResponse {
    data: Vec<RerankHit>,
}

#[derive(Deserialize)]
struct RerankHit {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl Reranker for VoyageReranker {
    async fn rerank(&self, query: &str, docs: Vec<Document>, top_k: usize) -> Result<Vec<Document>> {
        if docs.is_empty() { return Ok(vec![]); }

        let texts: Vec<String> = docs.iter().map(|d| d.content.clone()).collect();
        let body = json!({
            "model": self.cfg.model,
            "query": query,
            "documents": texts,
            "top_k": top_k.min(docs.len()),
            "truncation": self.cfg.truncation,
            "return_documents": false,
        });

        let url = format!("{}/rerank", self.cfg.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .header("accept", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("voyage send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(Error::RateLimited { retry_after_ms: None });
            }
            return Err(Error::other(format!("voyage rerank {status}: {txt}")));
        }
        let parsed: RerankResponse = resp
            .json()
            .await
            .map_err(|e| Error::other(format!("voyage decode: {e}")))?;

        let mut out = Vec::with_capacity(parsed.data.len());
        for hit in parsed.data {
            if let Some(doc) = docs.get(hit.index) {
                let mut d = doc.clone();
                d.score = Some(hit.relevance_score);
                out.push(d);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn start_fake(response: &str, capture: std::sync::Arc<std::sync::Mutex<Option<String>>>) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let body = response.to_string();
        thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let mut total = Vec::new();
                loop {
                    let n = match s.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => n,
                        Err(_) => break,
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
                        while total.len() < header_end + cl {
                            let n = match s.read(&mut buf) {
                                Ok(0) => break,
                                Ok(n) => n,
                                Err(_) => break,
                            };
                            total.extend_from_slice(&buf[..n]);
                        }
                        if total.len() >= header_end + cl {
                            *capture.lock().unwrap() =
                                Some(String::from_utf8_lossy(&total[header_end..header_end + cl]).to_string());
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
    async fn reranks_by_relevance_score_writing_to_doc_score() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
        let port = start_fake(
            r#"{"object":"list","data":[
                {"index":2,"relevance_score":0.95},
                {"index":0,"relevance_score":0.71},
                {"index":1,"relevance_score":0.30}
            ],"model":"rerank-2","usage":{"total_tokens":42}}"#,
            captured.clone(),
        );

        let cfg = VoyageRerankConfig::new("voy-fake", "rerank-2")
            .with_base_url(format!("http://127.0.0.1:{port}"));
        let r = VoyageReranker::new(cfg).unwrap();

        let docs = vec![
            Document::new("first").with_id("a"),
            Document::new("second").with_id("b"),
            Document::new("third").with_id("c"),
        ];
        let out = r.rerank("question", docs, 3).await.unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].id.as_deref(), Some("c"));
        assert_eq!(out[1].id.as_deref(), Some("a"));
        assert_eq!(out[2].id.as_deref(), Some("b"));
        assert!((out[0].score.unwrap() - 0.95).abs() < 1e-3);

        // Verify wire-body shape (Voyage uses `top_k` + `truncation`, NOT
        // Cohere's `top_n`).
        let body: serde_json::Value =
            serde_json::from_str(captured.lock().unwrap().as_deref().unwrap()).unwrap();
        assert_eq!(body["model"], "rerank-2");
        assert_eq!(body["query"], "question");
        assert_eq!(body["top_k"], 3);
        assert_eq!(body["truncation"], true);
        assert!(body.get("top_n").is_none(), "must not send the Cohere field");
    }

    #[tokio::test]
    async fn empty_docs_returns_empty_no_http() {
        let cfg = VoyageRerankConfig::new("k", "rerank-2").with_base_url("http://127.0.0.1:1");
        let r = VoyageReranker::new(cfg).unwrap();
        let out = r.rerank("q", vec![], 5).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn truncation_can_be_disabled() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
        let port = start_fake(r#"{"data":[],"model":"rerank-2-lite"}"#, captured.clone());
        let cfg = VoyageRerankConfig::new("k", "rerank-2-lite")
            .with_base_url(format!("http://127.0.0.1:{port}"))
            .with_truncation(false);
        let r = VoyageReranker::new(cfg).unwrap();
        let docs = vec![Document::new("hi").with_id("x")];
        let _ = r.rerank("q", docs, 1).await.unwrap();
        let body: serde_json::Value =
            serde_json::from_str(captured.lock().unwrap().as_deref().unwrap()).unwrap();
        assert_eq!(body["truncation"], false);
    }
}
