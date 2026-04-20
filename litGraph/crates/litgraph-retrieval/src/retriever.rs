use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{Document, Embeddings, Result};

use crate::store::{Filter, VectorStore};

/// Abstract retrieval — given a natural-language query, return top-k documents.
#[async_trait]
pub trait Retriever: Send + Sync {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>>;
}

/// Dense-embedding retriever — embeds the query, then delegates to a `VectorStore`.
pub struct VectorRetriever {
    pub embeddings: Arc<dyn Embeddings>,
    pub store: Arc<dyn VectorStore>,
    pub filter: Option<Filter>,
}

impl VectorRetriever {
    pub fn new(embeddings: Arc<dyn Embeddings>, store: Arc<dyn VectorStore>) -> Self {
        Self { embeddings, store, filter: None }
    }

    pub fn with_filter(mut self, f: Filter) -> Self {
        self.filter = Some(f);
        self
    }
}

#[async_trait]
impl Retriever for VectorRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let emb = self.embeddings.embed_query(query).await?;
        self.store.similarity_search(&emb, k, self.filter.as_ref()).await
    }
}
