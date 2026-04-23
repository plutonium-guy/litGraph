//! RAG evaluation harness — compute `recall@k`, `mrr@k`, and `ndcg@k` over a
//! labeled dataset of (query → relevant_doc_ids) pairs. Async-parallel: each
//! query runs concurrently via a JoinSet bounded by `max_concurrency` so a
//! 10K-query suite doesn't stampede a hosted retriever.
//!
//! Choice of metrics: these three cover the common LangChain/Ragas surface
//! without requiring an LLM judge (which adds cost + variance). For
//! generation-quality eval (faithfulness, answer relevance), use a dedicated
//! eval framework — this crate is scoped to the retriever stage where the
//! ground truth is unambiguous.

use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::Retriever;
use litgraph_core::{Error, Result};

/// One labeled retrieval question. `relevant_ids` is the set of document IDs
/// that *should* appear in the top-k for this query — order doesn't matter
/// (rank-aware metrics use the order returned by the retriever, not the
/// dataset's listing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCase {
    pub query: String,
    pub relevant_ids: Vec<String>,
}

/// Per-query metric breakdown — useful for spotting bad cases without
/// re-running the whole suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerQueryMetrics {
    pub query: String,
    pub recall: f64,
    pub mrr: f64,
    pub ndcg: f64,
    /// Document IDs the retriever returned, in rank order.
    pub returned_ids: Vec<String>,
}

/// Aggregated report. `*_macro` averages each per-query metric uniformly
/// (simple mean over queries) — what production teams actually want when
/// the goal is "does this change help on average."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub k: usize,
    pub n_queries: usize,
    pub recall_macro: f64,
    pub mrr_macro: f64,
    pub ndcg_macro: f64,
    pub per_query: Vec<PerQueryMetrics>,
}

#[derive(Debug, Clone, Copy)]
pub struct EvalConfig {
    pub k: usize,
    pub max_concurrency: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self { k: 10, max_concurrency: 8 }
    }
}

/// Run the full eval suite. Each query is dispatched concurrently (bounded
/// by `max_concurrency`); per-query metrics are then averaged into a report.
pub async fn evaluate_retrieval(
    retriever: Arc<dyn Retriever>,
    dataset: &[EvalCase],
    cfg: EvalConfig,
) -> Result<EvalReport> {
    if dataset.is_empty() {
        return Err(Error::other("evaluate_retrieval: empty dataset"));
    }
    let sem = Arc::new(Semaphore::new(cfg.max_concurrency.max(1)));
    let mut tasks: JoinSet<Result<PerQueryMetrics>> = JoinSet::new();

    for case in dataset {
        let r = retriever.clone();
        let sem = sem.clone();
        let q = case.query.clone();
        let relevant: HashSet<String> = case.relevant_ids.iter().cloned().collect();
        let k = cfg.k;
        tasks.spawn(async move {
            let _permit = sem.acquire().await.map_err(|e| Error::other(format!("sem: {e}")))?;
            let docs = r.retrieve(&q, k).await?;
            let returned_ids: Vec<String> = docs
                .iter()
                .map(|d| d.id.clone().unwrap_or_default())
                .collect();
            Ok(PerQueryMetrics {
                query: q,
                recall: recall_at_k(&returned_ids, &relevant, k),
                mrr: mrr_at_k(&returned_ids, &relevant, k),
                ndcg: ndcg_at_k(&returned_ids, &relevant, k),
                returned_ids,
            })
        });
    }

    let mut per_query = Vec::with_capacity(dataset.len());
    while let Some(res) = tasks.join_next().await {
        let m = res
            .map_err(|e| Error::other(format!("eval task join: {e}")))?
            .map_err(|e| Error::other(format!("eval task: {e}")))?;
        per_query.push(m);
    }
    // Stable order for reporting — sort by original query position.
    let order: std::collections::HashMap<&str, usize> = dataset
        .iter()
        .enumerate()
        .map(|(i, c)| (c.query.as_str(), i))
        .collect();
    per_query.sort_by_key(|m| *order.get(m.query.as_str()).unwrap_or(&usize::MAX));

    let n = per_query.len() as f64;
    let recall_macro: f64 = per_query.iter().map(|m| m.recall).sum::<f64>() / n;
    let mrr_macro: f64 = per_query.iter().map(|m| m.mrr).sum::<f64>() / n;
    let ndcg_macro: f64 = per_query.iter().map(|m| m.ndcg).sum::<f64>() / n;

    Ok(EvalReport {
        k: cfg.k,
        n_queries: per_query.len(),
        recall_macro,
        mrr_macro,
        ndcg_macro,
        per_query,
    })
}

/// recall@k = |returned ∩ relevant| / |relevant|. Returns 1.0 if `relevant`
/// is empty (vacuously perfect — a query with no expected docs can only be
/// "wrong" if you set up the dataset wrong).
pub fn recall_at_k(returned: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() { return 1.0; }
    let topk: HashSet<&str> = returned.iter().take(k).map(|s| s.as_str()).collect();
    let hits = relevant.iter().filter(|id| topk.contains(id.as_str())).count();
    hits as f64 / relevant.len() as f64
}

/// MRR@k = 1 / rank_of_first_relevant (1-indexed). Returns 0.0 if no
/// relevant doc appears in the top-k.
pub fn mrr_at_k(returned: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    for (i, id) in returned.iter().take(k).enumerate() {
        if relevant.contains(id) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

/// nDCG@k with binary relevance gains (1 if relevant else 0). Standard
/// formula: DCG = sum_i (rel_i / log2(i + 2)); IDCG = same for the ideal
/// ranking; nDCG = DCG / IDCG. Returns 1.0 if IDCG is 0 (no relevant docs;
/// trivially "perfect").
pub fn ndcg_at_k(returned: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() { return 1.0; }
    let mut dcg = 0.0;
    for (i, id) in returned.iter().take(k).enumerate() {
        if relevant.contains(id) {
            dcg += 1.0 / ((i + 2) as f64).log2();
        }
    }
    let ideal_hits = relevant.len().min(k);
    let mut idcg = 0.0;
    for i in 0..ideal_hits {
        idcg += 1.0 / ((i + 2) as f64).log2();
    }
    if idcg == 0.0 { 1.0 } else { dcg / idcg }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::Document;

    /// Stub retriever: returns a fixed list of doc IDs for any query.
    /// Useful for hand-crafting metric ground truth.
    struct FixedRetriever {
        ids: Vec<String>,
    }

    #[async_trait]
    impl Retriever for FixedRetriever {
        async fn retrieve(&self, _query: &str, k: usize) -> Result<Vec<Document>> {
            Ok(self.ids.iter().take(k)
                .map(|id| Document::new("").with_id(id))
                .collect())
        }
    }

    fn rel(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }
    fn ret(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn recall_perfect_top1_when_relevant_in_position_0() {
        // 1 relevant doc, returned at rank 1 → recall = 1.0
        assert_eq!(recall_at_k(&ret(&["a", "b", "c"]), &rel(&["a"]), 3), 1.0);
    }

    #[test]
    fn recall_partial_when_some_relevant_missed() {
        // 2 relevant, 1 hit → 0.5
        assert_eq!(recall_at_k(&ret(&["a", "x", "y"]), &rel(&["a", "b"]), 3), 0.5);
    }

    #[test]
    fn recall_zero_when_top_k_misses_all() {
        assert_eq!(recall_at_k(&ret(&["x", "y"]), &rel(&["a"]), 2), 0.0);
    }

    #[test]
    fn recall_vacuously_perfect_when_no_relevant_docs() {
        assert_eq!(recall_at_k(&ret(&["x"]), &rel(&[]), 1), 1.0);
    }

    #[test]
    fn mrr_first_position_returns_1() {
        assert_eq!(mrr_at_k(&ret(&["a", "b"]), &rel(&["a"]), 2), 1.0);
    }

    #[test]
    fn mrr_third_position_returns_one_third() {
        let m = mrr_at_k(&ret(&["x", "y", "a", "b"]), &rel(&["a"]), 4);
        assert!((m - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn mrr_zero_when_no_relevant_in_topk() {
        assert_eq!(mrr_at_k(&ret(&["x", "y"]), &rel(&["a"]), 2), 0.0);
    }

    #[test]
    fn ndcg_perfect_ranking_returns_1() {
        // All relevant docs ranked at the top → DCG == IDCG → 1.0
        let n = ndcg_at_k(&ret(&["a", "b", "x"]), &rel(&["a", "b"]), 3);
        assert!((n - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ndcg_imperfect_ranking_below_1() {
        // Relevant doc at rank 3 instead of rank 1 → discounted.
        let n = ndcg_at_k(&ret(&["x", "y", "a"]), &rel(&["a"]), 3);
        assert!(n < 1.0 && n > 0.0, "expected 0 < n < 1, got {n}");
    }

    #[test]
    fn ndcg_zero_relevant_returns_1() {
        assert_eq!(ndcg_at_k(&ret(&["x"]), &rel(&[]), 1), 1.0);
    }

    #[tokio::test]
    async fn evaluate_aggregates_per_query_into_macro_average() {
        // 2 queries against a fixed retriever. Q1 gets a perfect hit; Q2 misses.
        let retriever: Arc<dyn Retriever> = Arc::new(FixedRetriever {
            ids: vec!["a".into(), "b".into(), "c".into()],
        });
        let dataset = vec![
            EvalCase { query: "q1".into(), relevant_ids: vec!["a".into()] },     // recall=1, mrr=1, ndcg=1
            EvalCase { query: "q2".into(), relevant_ids: vec!["zzz".into()] },   // recall=0, mrr=0, ndcg=0 (no relevant in top-3)
        ];
        let report = evaluate_retrieval(retriever, &dataset, EvalConfig { k: 3, max_concurrency: 4 })
            .await.unwrap();
        assert_eq!(report.n_queries, 2);
        assert_eq!(report.k, 3);
        // Macro averages: (1+0)/2 = 0.5 for each.
        assert!((report.recall_macro - 0.5).abs() < 1e-9);
        assert!((report.mrr_macro - 0.5).abs() < 1e-9);
        assert!((report.ndcg_macro - 0.5).abs() < 1e-9);
        // Per-query order matches dataset input order.
        assert_eq!(report.per_query[0].query, "q1");
        assert_eq!(report.per_query[1].query, "q2");
        assert_eq!(report.per_query[0].returned_ids, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn evaluate_empty_dataset_errors() {
        let retriever: Arc<dyn Retriever> = Arc::new(FixedRetriever { ids: vec![] });
        let err = evaluate_retrieval(retriever, &[], EvalConfig::default()).await.unwrap_err();
        assert!(format!("{err}").contains("empty dataset"));
    }
}
