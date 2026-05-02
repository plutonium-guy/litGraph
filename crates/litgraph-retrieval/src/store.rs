use std::collections::HashMap;

use async_trait::async_trait;
use litgraph_core::{Document, Result};
use serde_json::Value;

pub type Filter = HashMap<String, Value>;

/// Storage for documents + their embedding vectors, with optional metadata-filtered
/// similarity search.
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add(&self, docs: Vec<Document>, embeddings: Vec<Vec<f32>>) -> Result<Vec<String>>;

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Document>>;

    async fn delete(&self, ids: &[String]) -> Result<()>;

    async fn len(&self) -> usize;

    /// Convenience: `self.len().await == 0`. Override for stores where
    /// emptiness is cheaper than counting (e.g., remote stores that
    /// otherwise need a full count query).
    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}
