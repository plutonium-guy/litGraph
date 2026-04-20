//! Text splitters. All expose a common `Splitter` trait; batch operations parallelize
//! across documents with Rayon (LangChain's `RecursiveCharacterTextSplitter` is
//! single-threaded Python — we get trivial speedup for free).

pub mod recursive;
pub mod markdown;
pub mod semantic;

pub use recursive::{Language, RecursiveCharacterSplitter};
pub use markdown::MarkdownHeaderSplitter;
pub use semantic::SemanticChunker;

use litgraph_core::Document;
use rayon::prelude::*;

pub trait Splitter: Send + Sync {
    /// Split a single text into chunks.
    fn split_text(&self, text: &str) -> Vec<String>;

    /// Split a Document, carrying metadata onto each chunk.
    fn split_document(&self, doc: &Document) -> Vec<Document> {
        self.split_text(&doc.content)
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mut d = Document::new(chunk);
                d.metadata = doc.metadata.clone();
                d.metadata.insert("chunk_index".into(), serde_json::json!(i));
                if let Some(id) = &doc.id {
                    d.id = Some(format!("{id}#{i}"));
                    d.metadata.insert("source_id".into(), serde_json::json!(id));
                }
                d
            })
            .collect()
    }

    /// Batch-split documents in parallel across Rayon's thread pool.
    fn split_documents(&self, docs: &[Document]) -> Vec<Document>
    where
        Self: Sync,
    {
        docs.par_iter()
            .flat_map_iter(|d| self.split_document(d))
            .collect()
    }
}
