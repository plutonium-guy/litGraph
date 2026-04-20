use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{Document, Result};

use crate::retriever::Retriever;

/// Rerank a candidate set by cross-encoder / model scoring.
#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, docs: Vec<Document>, top_k: usize) -> Result<Vec<Document>>;
}

/// Wraps a base `Retriever` with a `Reranker` — the standard 2-stage
/// retrieval pattern. Pulls `over_fetch_k` candidates from the base, then
/// rerank-narrows to `k`. `over_fetch_k` defaults to `4 * k` to give the
/// reranker enough headroom to surface the truly relevant docs that the
/// first-stage retriever may have ranked lower.
pub struct RerankingRetriever {
    pub base: Arc<dyn Retriever>,
    pub reranker: Arc<dyn Reranker>,
    pub over_fetch_k: Option<usize>,
}

impl RerankingRetriever {
    pub fn new(base: Arc<dyn Retriever>, reranker: Arc<dyn Reranker>) -> Self {
        Self { base, reranker, over_fetch_k: None }
    }

    pub fn with_over_fetch_k(mut self, n: usize) -> Self {
        self.over_fetch_k = Some(n);
        self
    }
}

#[async_trait]
impl Retriever for RerankingRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let pull = self.over_fetch_k.unwrap_or_else(|| (k * 4).max(20));
        let candidates = self.base.retrieve(query, pull).await?;
        if candidates.is_empty() { return Ok(vec![]); }
        self.reranker.rerank(query, candidates, k).await
    }
}
