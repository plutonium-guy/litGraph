//! Voyage AI embeddings adapter (`/v1/embeddings` endpoint).
//!
//! Voyage's RAG-tuned models (voyage-3, voyage-3-large, voyage-code-3, etc.)
//! consistently outperform OpenAI/Cohere on MTEB-RAG. Anthropic recommends
//! Voyage as the default embedder for Claude-powered RAG.
//!
//! # Wire format
//!
//! POST `/v1/embeddings`:
//! ```json
//! { "input": ["text", ...], "model": "voyage-3", "input_type": "query"|"document"|null }
//! ```
//! Response shape mirrors OpenAI:
//! ```json
//! { "object": "list", "data": [{"object":"embedding","embedding":[...],"index":0}], "usage": {...} }
//! ```
//!
//! Voyage also supports `truncation` (default true) and `output_dimension` for
//! voyage-3-large / voyage-code-3, but those defaults are correct for typical
//! retrieval pipelines and are not exposed here.

use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::{Embeddings, Error, Result};
use reqwest::Client;
use serde_json::{Value, json};
use tracing::debug;

#[derive(Clone, Debug)]
pub struct VoyageEmbeddingsConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub dimensions: usize,
    /// `Some("query")` for queries, `Some("document")` for documents,
    /// `None` to omit the field (Voyage default = generic).
    pub input_type_document: Option<String>,
    pub input_type_query: Option<String>,
}

impl VoyageEmbeddingsConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>, dimensions: usize) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.voyageai.com/v1".into(),
            model: model.into(),
            timeout: Duration::from_secs(120),
            dimensions,
            input_type_document: Some("document".into()),
            input_type_query: Some("query".into()),
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_input_types(
        mut self,
        document: Option<impl Into<String>>,
        query: Option<impl Into<String>>,
    ) -> Self {
        self.input_type_document = document.map(Into::into);
        self.input_type_query = query.map(Into::into);
        self
    }
}

pub struct VoyageEmbeddings {
    cfg: VoyageEmbeddingsConfig,
    http: Client,
}

impl VoyageEmbeddings {
    pub fn new(cfg: VoyageEmbeddingsConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    async fn embed_batch(
        &self,
        inputs: Vec<String>,
        input_type: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.cfg.base_url.trim_end_matches('/'));
        let mut body = json!({ "model": self.cfg.model, "input": inputs });
        if let Some(it) = input_type {
            body["input_type"] = json!(it);
        }
        debug!(model = %self.cfg.model, n = inputs.len(), "voyage embed");
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .header("accept", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::provider(format!("send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(Error::RateLimited { retry_after_ms: None });
            }
            return Err(Error::provider(format!("voyage embed {status}: {txt}")));
        }
        let v: Value = resp.json().await.map_err(|e| Error::provider(format!("decode: {e}")))?;
        let data = v
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| Error::provider("voyage embed: missing `data`"))?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let emb = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| Error::provider("voyage embed: missing `embedding`"))?;
            out.push(emb.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect());
        }
        Ok(out)
    }
}

#[async_trait]
impl Embeddings for VoyageEmbeddings {
    fn name(&self) -> &str { &self.cfg.model }
    fn dimensions(&self) -> usize { self.cfg.dimensions }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let it = self.cfg.input_type_query.clone();
        let mut out = self.embed_batch(vec![text.to_string()], it.as_deref()).await?;
        out.pop().ok_or_else(|| Error::provider("embed_query: empty result"))
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(vec![]); }
        let it = self.cfg.input_type_document.clone();
        self.embed_batch(texts.to_vec(), it.as_deref()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_query_and_document_input_types() {
        let cfg = VoyageEmbeddingsConfig::new("k", "voyage-3", 1024);
        assert_eq!(cfg.input_type_document.as_deref(), Some("document"));
        assert_eq!(cfg.input_type_query.as_deref(), Some("query"));
        assert_eq!(cfg.dimensions, 1024);
        assert_eq!(cfg.base_url, "https://api.voyageai.com/v1");
    }

    #[test]
    fn with_input_types_can_disable() {
        let cfg = VoyageEmbeddingsConfig::new("k", "voyage-3", 1024)
            .with_input_types(None::<String>, None::<String>);
        assert!(cfg.input_type_document.is_none());
        assert!(cfg.input_type_query.is_none());
    }

    #[tokio::test]
    async fn embed_documents_empty_short_circuits_no_http() {
        // base_url unreachable; if we hit the network, we panic. Empty list must skip.
        let cfg = VoyageEmbeddingsConfig::new("k", "voyage-3", 1024).with_base_url("http://127.0.0.1:1");
        let e = VoyageEmbeddings::new(cfg).unwrap();
        let out = e.embed_documents(&[]).await.unwrap();
        assert!(out.is_empty());
    }
}
