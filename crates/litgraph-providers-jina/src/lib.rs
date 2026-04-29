//! Jina AI embeddings adapter (`/v1/embeddings`).
//!
//! `jina-embeddings-v3` and family. Wire format is OpenAI-compatible at the
//! request/response shape level (`{input: [...], model}` → `{data: [{embedding: [...]}]}`)
//! BUT Jina supports task-aware retrieval via the `task` field —
//! `retrieval.passage` / `retrieval.query` / `text-matching` / `classification`
//! / `separation`. Default mapping: `embed_documents` → `retrieval.passage`,
//! `embed_query` → `retrieval.query` (matches Cohere/Voyage idiom).
//!
//! Optional `dimensions` truncates server-side (Matryoshka representation).

use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::{Embeddings, Error, Result};
use reqwest::Client;
use serde_json::{Value, json};
use tracing::debug;

#[derive(Clone, Debug)]
pub struct JinaEmbeddingsConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub dimensions: usize,
    pub task_document: Option<String>,
    pub task_query: Option<String>,
    /// Optional Matryoshka truncation. When set, server returns vectors of
    /// this length and `dimensions` is updated to match.
    pub output_dimensions: Option<usize>,
}

impl JinaEmbeddingsConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>, dimensions: usize) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.jina.ai/v1".into(),
            model: model.into(),
            timeout: Duration::from_secs(120),
            dimensions,
            task_document: Some("retrieval.passage".into()),
            task_query: Some("retrieval.query".into()),
            output_dimensions: None,
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_tasks(
        mut self,
        document: Option<impl Into<String>>,
        query: Option<impl Into<String>>,
    ) -> Self {
        self.task_document = document.map(Into::into);
        self.task_query = query.map(Into::into);
        self
    }
    pub fn with_output_dimensions(mut self, n: usize) -> Self {
        self.output_dimensions = Some(n);
        self.dimensions = n;
        self
    }
}

pub struct JinaEmbeddings {
    cfg: JinaEmbeddingsConfig,
    http: Client,
}

impl JinaEmbeddings {
    pub fn new(cfg: JinaEmbeddingsConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::provider(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    async fn embed_batch(&self, inputs: Vec<String>, task: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.cfg.base_url.trim_end_matches('/'));
        let mut body = json!({ "model": self.cfg.model, "input": inputs });
        if let Some(t) = task {
            body["task"] = json!(t);
        }
        if let Some(d) = self.cfg.output_dimensions {
            body["dimensions"] = json!(d);
        }
        debug!(model = %self.cfg.model, n = inputs.len(), "jina embed");
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
            return Err(Error::provider(format!("jina embed {status}: {txt}")));
        }
        let v: Value = resp.json().await.map_err(|e| Error::provider(format!("decode: {e}")))?;
        let data = v
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| Error::provider("jina embed: missing `data`"))?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let emb = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| Error::provider("jina embed: missing `embedding`"))?;
            out.push(emb.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect());
        }
        Ok(out)
    }
}

#[async_trait]
impl Embeddings for JinaEmbeddings {
    fn name(&self) -> &str { &self.cfg.model }
    fn dimensions(&self) -> usize { self.cfg.dimensions }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let t = self.cfg.task_query.clone();
        let mut out = self.embed_batch(vec![text.to_string()], t.as_deref()).await?;
        out.pop().ok_or_else(|| Error::provider("embed_query: empty result"))
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(vec![]); }
        let t = self.cfg.task_document.clone();
        self.embed_batch(texts.to_vec(), t.as_deref()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_retrieval_tasks() {
        let cfg = JinaEmbeddingsConfig::new("k", "jina-embeddings-v3", 1024);
        assert_eq!(cfg.task_document.as_deref(), Some("retrieval.passage"));
        assert_eq!(cfg.task_query.as_deref(), Some("retrieval.query"));
        assert_eq!(cfg.dimensions, 1024);
    }

    #[test]
    fn output_dimensions_truncates_and_updates_dims() {
        let cfg = JinaEmbeddingsConfig::new("k", "jina-embeddings-v3", 1024)
            .with_output_dimensions(256);
        assert_eq!(cfg.dimensions, 256);
        assert_eq!(cfg.output_dimensions, Some(256));
    }

    #[tokio::test]
    async fn empty_documents_short_circuits_no_http() {
        let cfg = JinaEmbeddingsConfig::new("k", "jina-embeddings-v3", 1024)
            .with_base_url("http://127.0.0.1:1");
        let e = JinaEmbeddings::new(cfg).unwrap();
        assert!(e.embed_documents(&[]).await.unwrap().is_empty());
    }
}
