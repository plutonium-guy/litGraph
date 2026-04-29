//! Hybrid retriever — fan out across N child retrievers concurrently, fuse results
//! via Reciprocal Rank Fusion (RRF). Concurrent fan-out uses `tokio::join_all`; this
//! is a direct win over LangChain's sequential ensemble, which waits for each branch.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;
use litgraph_core::{Document, Result};

use crate::retriever::Retriever;

pub struct HybridRetriever {
    pub children: Vec<Arc<dyn Retriever>>,
    pub rrf_k: f32,
    /// Pull ≥ `per_child_k * k` candidates from each child to avoid truncating too early.
    pub per_child_k: Option<usize>,
}

impl HybridRetriever {
    pub fn new(children: Vec<Arc<dyn Retriever>>) -> Self {
        Self { children, rrf_k: 60.0, per_child_k: None }
    }

    pub fn with_rrf_k(mut self, k: f32) -> Self { self.rrf_k = k; self }
    pub fn with_per_child_k(mut self, k: usize) -> Self { self.per_child_k = Some(k); self }
}

#[async_trait]
impl Retriever for HybridRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let child_k = self.per_child_k.unwrap_or(k * 2);
        let futures = self.children.iter().map(|c| {
            let c = c.clone();
            let q = query.to_string();
            async move { c.retrieve(&q, child_k).await }
        });
        let results = join_all(futures).await;

        let mut branches: Vec<Vec<Document>> = Vec::with_capacity(results.len());
        for r in results {
            branches.push(r?);
        }
        Ok(rrf_fuse(&branches, self.rrf_k, k))
    }
}

/// Reciprocal Rank Fusion — canonical hybrid-search aggregator. Score for a doc `d`:
///
/// ```text
///   rrf(d) = Σ_b  1 / (k_rrf + rank_b(d))
/// ```
///
/// Documents are keyed by `id` when present, else by `content` hash (String).
pub fn rrf_fuse(branches: &[Vec<Document>], k_rrf: f32, top_k: usize) -> Vec<Document> {
    let mut scores: HashMap<String, (f32, Document)> = HashMap::new();
    for branch in branches {
        for (rank, doc) in branch.iter().enumerate() {
            let key = doc.id.clone().unwrap_or_else(|| doc.content.clone());
            let contrib = 1.0 / (k_rrf + (rank as f32 + 1.0));
            scores
                .entry(key)
                .and_modify(|e| e.0 += contrib)
                .or_insert_with(|| (contrib, doc.clone()));
        }
    }
    let mut out: Vec<(f32, Document)> = scores.into_values().collect();
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(top_k);
    out.into_iter()
        .map(|(s, mut d)| { d.score = Some(s); d })
        .collect()
}
