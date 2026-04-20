//! Jina AI reranker (`/v1/rerank`).
//!
//! `jina-reranker-v2-base-multilingual` and family. Wire shape mirrors
//! Cohere's `/v2/rerank` (`top_n`, `results: [{index, relevance_score}]`).

use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::{Document, Error, Result};
use litgraph_retrieval::Reranker;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Clone)]
pub struct JinaRerankConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
}

impl JinaRerankConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.jina.ai/v1".into(),
            model: model.into(),
            timeout: Duration::from_secs(60),
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

pub struct JinaReranker {
    cfg: JinaRerankConfig,
    http: Client,
}

impl JinaReranker {
    pub fn new(cfg: JinaRerankConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }
}

#[derive(Deserialize)]
struct RerankResponse { results: Vec<RerankHit> }

#[derive(Deserialize)]
struct RerankHit { index: usize, relevance_score: f32 }

#[async_trait]
impl Reranker for JinaReranker {
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

        let url = format!("{}/rerank", self.cfg.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .header("accept", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("jina send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(Error::RateLimited { retry_after_ms: None });
            }
            return Err(Error::other(format!("jina rerank {status}: {txt}")));
        }
        let parsed: RerankResponse = resp
            .json()
            .await
            .map_err(|e| Error::other(format!("jina decode: {e}")))?;
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
    async fn reranks_by_score_writing_to_doc_score() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
        let port = start_fake(
            r#"{"results":[
                {"index":2,"relevance_score":0.9},
                {"index":0,"relevance_score":0.7},
                {"index":1,"relevance_score":0.3}
            ]}"#,
            captured.clone(),
        );
        let cfg = JinaRerankConfig::new("k", "jina-reranker-v2-base-multilingual")
            .with_base_url(format!("http://127.0.0.1:{port}"));
        let r = JinaReranker::new(cfg).unwrap();
        let docs = vec![
            Document::new("first").with_id("a"),
            Document::new("second").with_id("b"),
            Document::new("third").with_id("c"),
        ];
        let out = r.rerank("q", docs, 3).await.unwrap();
        assert_eq!(out[0].id.as_deref(), Some("c"));
        assert_eq!(out[1].id.as_deref(), Some("a"));
        assert_eq!(out[2].id.as_deref(), Some("b"));
        assert!((out[0].score.unwrap() - 0.9).abs() < 1e-3);

        let body: serde_json::Value =
            serde_json::from_str(captured.lock().unwrap().as_deref().unwrap()).unwrap();
        // Jina uses Cohere-style `top_n` (not Voyage's `top_k`).
        assert_eq!(body["top_n"], 3);
        assert!(body.get("top_k").is_none());
    }

    #[tokio::test]
    async fn empty_docs_returns_empty_no_http() {
        let cfg = JinaRerankConfig::new("k", "jina-reranker-v2-base-multilingual")
            .with_base_url("http://127.0.0.1:1");
        let r = JinaReranker::new(cfg).unwrap();
        assert!(r.rerank("q", vec![], 5).await.unwrap().is_empty());
    }
}
