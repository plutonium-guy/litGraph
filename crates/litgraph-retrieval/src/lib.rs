//! Retrieval primitives — traits for `VectorStore`, `Retriever`, and `Reranker`,
//! plus pure-Rust BM25 and hybrid fusion (RRF).

pub mod store;
pub mod retriever;
pub mod bm25;
pub mod hybrid;
pub mod ensemble;
pub mod ensemble_rerank;
pub mod multi_vector;
pub mod concurrent;
pub mod race;
pub mod rerank_concurrent;
pub mod rerank;
pub mod eval;
pub mod eval_gen;
pub mod parent;
pub mod multi_query;
pub mod rag_fusion;
pub mod hyde;
pub mod compression;
pub mod self_query;
pub mod time_weighted;
pub mod transformers;
pub mod mmr_retriever;

pub use store::{VectorStore, Filter};
pub use retriever::{Retriever, VectorRetriever};
pub use bm25::Bm25Index;
pub use hybrid::{HybridRetriever, rrf_fuse};
pub use ensemble::{EnsembleRetriever, weighted_rrf_fuse};
pub use ensemble_rerank::{EnsembleReranker, weighted_rrf_fuse_rerank};
pub use multi_vector::{MultiVectorItem, MultiVectorRetriever};
pub use concurrent::{
    retrieve_concurrent, retrieve_concurrent_fail_fast, retrieve_concurrent_stream,
    retrieve_concurrent_stream_with_progress, retrieve_concurrent_stream_with_shutdown,
    retrieve_concurrent_with_progress, retrieve_concurrent_with_shutdown, RetrieveProgress,
    RetrieveStreamItem,
};
pub use race::RaceRetriever;
pub use rerank_concurrent::{
    rerank_concurrent, rerank_concurrent_fail_fast, rerank_concurrent_stream,
    rerank_concurrent_stream_with_progress, rerank_concurrent_stream_with_shutdown,
    rerank_concurrent_with_progress, rerank_concurrent_with_shutdown, RerankProgress,
    RerankStreamItem,
};
pub use rerank::{Reranker, RerankingRetriever};
pub use parent::{
    ChildSplitter, DocStore, MemoryDocStore, ParentDocumentRetriever, PARENT_ID_META_KEY,
};
pub use multi_query::MultiQueryRetriever;
pub use rag_fusion::RagFusionRetriever;
pub use hyde::HydeRetriever;
pub use compression::{
    Compressor, ContextualCompressionRetriever, EmbeddingsFilterCompressor,
    LlmExtractCompressor, PipelineCompressor,
};
pub use self_query::{AttributeInfo, SelfQueryRetriever};
pub use time_weighted::TimeWeightedRetriever;
pub use mmr_retriever::MaxMarginalRelevanceRetriever;
pub use eval::{
    evaluate_retrieval, mrr_at_k, ndcg_at_k, recall_at_k, EvalCase, EvalConfig, EvalReport,
    PerQueryMetrics,
};
pub use eval_gen::{
    evaluate_generation, GenEvalConfig, GenReport, GenerationCase, PerCaseGenMetrics,
};
pub use transformers::{embedding_redundant_filter, long_context_reorder, mmr_select};
