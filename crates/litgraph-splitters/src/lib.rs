//! Text splitters. All expose a common `Splitter` trait; batch operations parallelize
//! across documents with Rayon (LangChain's `RecursiveCharacterTextSplitter` is
//! single-threaded Python — we get trivial speedup for free).

pub mod recursive;
pub mod markdown;
pub mod markdown_table;
pub mod semantic;
pub mod json;
pub mod jsonl;
pub mod html_header;
pub mod html_section;
pub mod code;
pub mod token;
pub mod csv_row;
pub mod sentence;

pub use recursive::{Language, RecursiveCharacterSplitter};
pub use markdown::MarkdownHeaderSplitter;
pub use markdown_table::MarkdownTableSplitter;
pub use csv_row::CsvRowSplitter;
pub use sentence::SentenceSplitter;
pub use jsonl::JsonLinesSplitter;
pub use semantic::SemanticChunker;
pub use json::JsonSplitter;
pub use html_header::HtmlHeaderSplitter;
pub use html_section::{HtmlSectionSplitter, DEFAULT_SECTION_TAGS};
pub use code::CodeSplitter;
pub use token::TokenTextSplitter;

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
