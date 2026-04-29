//! Cohere rerank API adapter — wraps any base `Retriever` with cross-encoder
//! reranking via Cohere's `/v2/rerank` endpoint.
//!
//! # Wire format
//!
//! POST `https://api.cohere.com/v2/rerank`
//!
//! Body:
//! ```json
//! {
//!   "model": "rerank-english-v3.0",
//!   "query": "the user's query",
//!   "documents": ["text1", "text2", ...],
//!   "top_n": 5,
//!   "return_documents": false
//! }
//! ```
//!
//! Response:
//! ```json
//! {
//!   "results": [
//!     { "index": 3, "relevance_score": 0.95 },
//!     { "index": 0, "relevance_score": 0.87 },
//!     ...
//!   ]
//! }
//! ```
//!
//! We map result indices back to the input documents and write the
//! relevance score onto `Document::score`.

use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::{Document, Error, Result};
use litgraph_retrieval::Reranker;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Clone)]
pub struct CohereConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
}

impl CohereConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.cohere.com".into(),
            model: model.into(),
            timeout: Duration::from_secs(60),
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

pub struct CohereReranker {
    cfg: CohereConfig,
    http: Client,
}

impl CohereReranker {
    pub fn new(cfg: CohereConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankHit>,
}

#[derive(Deserialize)]
struct RerankHit {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl Reranker for CohereReranker {
    async fn rerank(&self, query: &str, docs: Vec<Document>, top_k: usize) -> Result<Vec<Document>> {
        if docs.is_empty() { return Ok(vec![]); }

        let texts: Vec<String> = docs.iter().map(|d| d.content.clone()).collect();
        let body = json!({
            "model": self.cfg.model,
            "query": query,
            "documents": texts,
            "top_n": top_k.min(docs.len()),
            "return_documents": false,
        });

        let url = format!("{}/v2/rerank", self.cfg.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .header("accept", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("cohere send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(Error::RateLimited { retry_after_ms: None });
            }
            return Err(Error::other(format!("cohere {status}: {txt}")));
        }
        let parsed: RerankResponse = resp
            .json()
            .await
            .map_err(|e| Error::other(format!("cohere decode: {e}")))?;

        let mut out = Vec::with_capacity(parsed.results.len());
        for hit in parsed.results {
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

    fn start_fake(response: &str) -> u16 {
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
                        // Drain rest if Content-Length present.
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
    async fn reranks_by_relevance_score() {
        // Cohere reorders 3 docs: index 2 wins, then 0, then 1.
        let port = start_fake(r#"{"results":[
            {"index":2,"relevance_score":0.95},
            {"index":0,"relevance_score":0.71},
            {"index":1,"relevance_score":0.30}
        ]}"#);

        let cfg = CohereConfig::new("co-fake", "rerank-english-v3.0")
            .with_base_url(format!("http://127.0.0.1:{port}"));
        let r = CohereReranker::new(cfg).unwrap();

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
    }

    #[tokio::test]
    async fn empty_docs_returns_empty() {
        let cfg = CohereConfig::new("k", "rerank-english-v3.0");
        let r = CohereReranker::new(cfg).unwrap();
        let out = r.rerank("q", vec![], 5).await.unwrap();
        assert!(out.is_empty());
    }
}
