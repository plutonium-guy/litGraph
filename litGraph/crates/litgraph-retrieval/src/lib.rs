//! Retrieval primitives — traits for `VectorStore`, `Retriever`, and `Reranker`,
//! plus pure-Rust BM25 and hybrid fusion (RRF).

pub mod store;
pub mod retriever;
pub mod bm25;
pub mod hybrid;
pub mod rerank;
pub mod eval;
pub mod eval_gen;
pub mod parent;
pub mod multi_query;
pub mod compression;
pub mod self_query;
pub mod time_weighted;
pub mod transformers;

pub use store::{VectorStore, Filter};
pub use retriever::{Retriever, VectorRetriever};
pub use bm25::Bm25Index;
pub use hybrid::{HybridRetriever, rrf_fuse};
pub use rerank::{Reranker, RerankingRetriever};
pub use parent::{
    ChildSplitter, DocStore, MemoryDocStore, ParentDocumentRetriever, PARENT_ID_META_KEY,
};
pub use multi_query::MultiQueryRetriever;
pub use compression::{
    Compressor, ContextualCompressionRetriever, EmbeddingsFilterCompressor,
    LlmExtractCompressor, PipelineCompressor,
};
pub use self_query::{AttributeInfo, SelfQueryRetriever};
pub use time_weighted::TimeWeightedRetriever;
pub use eval::{
    evaluate_retrieval, mrr_at_k, ndcg_at_k, recall_at_k, EvalCase, EvalConfig, EvalReport,
    PerQueryMetrics,
};
pub use eval_gen::{
    evaluate_generation, GenEvalConfig, GenReport, GenerationCase, PerCaseGenMetrics,
};
pub use transformers::{embedding_redundant_filter, long_context_reorder, mmr_select};
