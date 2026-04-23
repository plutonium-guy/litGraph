//! Retrieval primitives — traits for `VectorStore`, `Retriever`, and `Reranker`,
//! plus pure-Rust BM25 and hybrid fusion (RRF).

pub mod store;
pub mod retriever;
pub mod bm25;
pub mod hybrid;
pub mod rerank;
pub mod eval;
pub mod eval_gen;

pub use store::{VectorStore, Filter};
pub use retriever::{Retriever, VectorRetriever};
pub use bm25::Bm25Index;
pub use hybrid::{HybridRetriever, rrf_fuse};
pub use rerank::{Reranker, RerankingRetriever};
pub use eval::{
    evaluate_retrieval, mrr_at_k, ndcg_at_k, recall_at_k, EvalCase, EvalConfig, EvalReport,
    PerQueryMetrics,
};
pub use eval_gen::{
    evaluate_generation, GenEvalConfig, GenReport, GenerationCase, PerCaseGenMetrics,
};
