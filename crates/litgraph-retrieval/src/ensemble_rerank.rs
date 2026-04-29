//! `EnsembleReranker` — fan out N rerankers concurrently, fuse their
//! orderings via weighted Reciprocal Rank Fusion.
//!
//! Where [`EnsembleRetriever`](crate::EnsembleRetriever) combines
//! retrievers (each pulling its own candidates), `EnsembleReranker`
//! combines *rerankers* — every child sees the **same** candidate set
//! and produces its own ranking. Useful when two rerankers disagree
//! systematically (e.g. a fast cross-encoder vs an LLM judge); the
//! ensemble smooths out per-model bias and tends to dominate either
//! one alone in offline evals.
//!
//! # Why RRF rather than score-averaging
//!
//! Rerankers report scores in wildly different scales — Cohere's are
//! `[0, 1]` probabilities, FastEmbed cross-encoder logits are
//! unbounded, Voyage's are unnormalised. Averaging raw scores
//! requires a per-reranker calibration step that gets stale as soon
//! as a model is updated. RRF on ranks is scale-free: it only cares
//! about **ordering**, so adding a new reranker doesn't break the
//! fusion math.
//!
//! # Parallelism
//!
//! All children run concurrently via `tokio::join_all`. Each one
//! issues its own provider request (cohere/voyage/jina) or runs its
//! ONNX cross-encoder forward pass; the ensemble's wall-clock
//! latency is `max(t_i)` rather than `Σ t_i`. LangChain's
//! `EnsembleRetriever`-with-rerankers pattern runs sequentially
//! under the GIL.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;
use litgraph_core::{Document, Error, Result};

use crate::rerank::Reranker;

const DEFAULT_RRF_K: f32 = 60.0;

pub struct EnsembleReranker {
    /// Children + their weights (un-normalised; only ratios matter).
    pub children: Vec<(Arc<dyn Reranker>, f32)>,
    pub rrf_k: f32,
}

impl EnsembleReranker {
    /// Equal-weight ensemble. Equivalent to averaging child orderings
    /// over RRF.
    pub fn new(children: Vec<Arc<dyn Reranker>>) -> Self {
        let weighted = children.into_iter().map(|c| (c, 1.0)).collect();
        Self {
            children: weighted,
            rrf_k: DEFAULT_RRF_K,
        }
    }

    /// Build with explicit per-child weights. Length-mismatch and empty
    /// children are rejected via `Result`; surface as `Error::invalid`.
    pub fn with_weights(
        children: Vec<Arc<dyn Reranker>>,
        weights: Vec<f32>,
    ) -> Result<Self> {
        if children.len() != weights.len() {
            return Err(Error::invalid(format!(
                "EnsembleReranker: children ({}) and weights ({}) length mismatch",
                children.len(),
                weights.len(),
            )));
        }
        if children.is_empty() {
            return Err(Error::invalid(
                "EnsembleReranker: need at least one child",
            ));
        }
        Ok(Self {
            children: children.into_iter().zip(weights).collect(),
            rrf_k: DEFAULT_RRF_K,
        })
    }

    pub fn with_rrf_k(mut self, k: f32) -> Self {
        self.rrf_k = k;
        self
    }
}

#[async_trait]
impl Reranker for EnsembleReranker {
    async fn rerank(
        &self,
        query: &str,
        docs: Vec<Document>,
        top_k: usize,
    ) -> Result<Vec<Document>> {
        if docs.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        // Ask each child for the FULL ordering — RRF needs every
        // doc's rank, not just the top-K.
        let n = docs.len();
        let docs_arc: Arc<Vec<Document>> = Arc::new(docs);
        let futures = self.children.iter().map(|(r, _)| {
            let r = r.clone();
            let q = query.to_string();
            let d = docs_arc.as_ref().clone();
            async move { r.rerank(&q, d, n).await }
        });
        let results = join_all(futures).await;

        let mut branches: Vec<(Vec<Document>, f32)> = Vec::with_capacity(self.children.len());
        for (r, (_, w)) in results.into_iter().zip(self.children.iter()) {
            branches.push((r?, *w));
        }
        Ok(weighted_rrf_fuse_rerank(&branches, self.rrf_k, top_k))
    }
}

/// Like [`crate::ensemble::weighted_rrf_fuse`] but tailored for the
/// reranker case: input is `(child_ranking, weight)` tuples, output is
/// the top-`top_k` docs by fused rank score.
///
/// A weight of `0.0` drops that branch entirely (kill-switch for
/// A/B). Negative weights are allowed and act as down-votes — useful
/// when one reranker is known to be miscalibrated for a query class
/// but you still want to lift the ones it ranks low.
pub fn weighted_rrf_fuse_rerank(
    branches: &[(Vec<Document>, f32)],
    k_rrf: f32,
    top_k: usize,
) -> Vec<Document> {
    let mut scores: HashMap<String, (f32, Document)> = HashMap::new();
    for (docs, w) in branches {
        if *w == 0.0 {
            continue;
        }
        for (rank, doc) in docs.iter().enumerate() {
            let key = doc.id.clone().unwrap_or_else(|| doc.content.clone());
            let contrib = w / (k_rrf + (rank as f32 + 1.0));
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
        .map(|(s, mut d)| {
            d.score = Some(s);
            d
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::collections::HashMap as Map;

    /// Reranker that returns docs in a fixed order matching `priority`
    /// (id → ascending position). Docs whose ids aren't in the priority
    /// map are placed at the end in their original order.
    struct ScriptedReranker {
        priority: Map<&'static str, usize>,
    }

    impl ScriptedReranker {
        fn from(order: &[&'static str]) -> Arc<dyn Reranker> {
            let mut priority: Map<&'static str, usize> = Map::new();
            for (i, id) in order.iter().enumerate() {
                priority.insert(*id, i);
            }
            Arc::new(Self { priority })
        }
    }

    #[async_trait]
    impl Reranker for ScriptedReranker {
        async fn rerank(
            &self,
            _query: &str,
            mut docs: Vec<Document>,
            top_k: usize,
        ) -> Result<Vec<Document>> {
            docs.sort_by_key(|d| {
                let id = d.id.as_deref().unwrap_or("");
                *self.priority.get(id).unwrap_or(&usize::MAX)
            });
            docs.truncate(top_k);
            Ok(docs)
        }
    }

    fn doc(id: &str) -> Document {
        Document {
            id: Some(id.into()),
            content: format!("doc {id}"),
            metadata: std::collections::HashMap::new(),
            score: None,
        }
    }

    fn ids(docs: &[Document]) -> Vec<String> {
        docs.iter().map(|d| d.id.clone().unwrap()).collect()
    }

    #[tokio::test]
    async fn empty_docs_returns_empty() {
        let a = ScriptedReranker::from(&["x"]);
        let b = ScriptedReranker::from(&["y"]);
        let r = EnsembleReranker::new(vec![a, b]);
        let out = r.rerank("q", vec![], 5).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn top_k_zero_returns_empty() {
        let a = ScriptedReranker::from(&["x"]);
        let b = ScriptedReranker::from(&["y"]);
        let r = EnsembleReranker::new(vec![a, b]);
        let out = r.rerank("q", vec![doc("x")], 0).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn equal_weights_fuse_all_three_branches() {
        // A ranks: x > y > z. B ranks: y > x > z. y appears at
        // rank 1 in B and rank 2 in A — it should be the fused
        // top under equal weights. (RRF rewards "high in any branch.")
        let a = ScriptedReranker::from(&["x", "y", "z"]);
        let b = ScriptedReranker::from(&["y", "x", "z"]);
        let r = EnsembleReranker::new(vec![a, b]);
        let out = r
            .rerank("q", vec![doc("x"), doc("y"), doc("z")], 3)
            .await
            .unwrap();
        assert_eq!(out.len(), 3);
        // Assemble the rank scores explicitly: y is rank 2+1, x is
        // rank 1+2, but with k_rrf=60 the score difference is sign-only.
        // What matters: y must outrank z (z is bottom in both branches).
        let positions: std::collections::HashMap<String, usize> = out
            .iter()
            .enumerate()
            .map(|(i, d)| (d.id.clone().unwrap(), i))
            .collect();
        assert!(positions["y"] < positions["z"]);
        assert!(positions["x"] < positions["z"]);
    }

    #[tokio::test]
    async fn higher_weight_branch_wins_top_slot() {
        // A: x > y > z (low weight). B: z > y > x (high weight).
        // B's ordering should dominate → z wins top.
        let a = ScriptedReranker::from(&["x", "y", "z"]);
        let b = ScriptedReranker::from(&["z", "y", "x"]);
        let r = EnsembleReranker::with_weights(vec![a, b], vec![0.1, 10.0]).unwrap();
        let out = r
            .rerank("q", vec![doc("x"), doc("y"), doc("z")], 1)
            .await
            .unwrap();
        assert_eq!(out[0].id.as_deref(), Some("z"));
    }

    #[tokio::test]
    async fn weight_zero_branch_is_silenced() {
        let a = ScriptedReranker::from(&["x", "y", "z"]);
        let b = ScriptedReranker::from(&["z", "y", "x"]);
        let r = EnsembleReranker::with_weights(vec![a, b], vec![1.0, 0.0]).unwrap();
        let out = r
            .rerank("q", vec![doc("x"), doc("y"), doc("z")], 3)
            .await
            .unwrap();
        // Only A's order should matter.
        assert_eq!(ids(&out), vec!["x", "y", "z"]);
    }

    #[tokio::test]
    async fn weights_length_mismatch_errors() {
        let a = ScriptedReranker::from(&["x"]);
        match EnsembleReranker::with_weights(vec![a], vec![1.0, 2.0]) {
            Err(e) => assert!(format!("{e}").contains("length mismatch")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[tokio::test]
    async fn empty_children_errors() {
        match EnsembleReranker::with_weights(vec![], vec![]) {
            Err(e) => assert!(format!("{e}").contains("at least one")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[tokio::test]
    async fn child_error_propagates() {
        struct Fail;
        #[async_trait]
        impl Reranker for Fail {
            async fn rerank(
                &self,
                _q: &str,
                _d: Vec<Document>,
                _k: usize,
            ) -> Result<Vec<Document>> {
                Err(Error::other("boom"))
            }
        }
        let a = ScriptedReranker::from(&["x"]);
        let r = EnsembleReranker::new(vec![a, Arc::new(Fail) as Arc<dyn Reranker>]);
        let err = r.rerank("q", vec![doc("x")], 1).await.unwrap_err();
        assert!(format!("{err}").contains("boom"));
    }

    #[tokio::test]
    async fn missing_doc_in_one_branch_still_scored_from_other() {
        // A drops `x` entirely (returns y, z only). B ranks x at #1.
        // RRF should still score x via B's contribution.
        struct Drops;
        #[async_trait]
        impl Reranker for Drops {
            async fn rerank(
                &self,
                _q: &str,
                docs: Vec<Document>,
                top_k: usize,
            ) -> Result<Vec<Document>> {
                let kept: Vec<Document> = docs
                    .into_iter()
                    .filter(|d| d.id.as_deref() != Some("x"))
                    .collect();
                Ok(kept.into_iter().take(top_k).collect())
            }
        }
        let a = Arc::new(Drops) as Arc<dyn Reranker>;
        let b = ScriptedReranker::from(&["x", "y", "z"]);
        let r = EnsembleReranker::with_weights(vec![a, b], vec![1.0, 10.0]).unwrap();
        let out = r
            .rerank("q", vec![doc("x"), doc("y"), doc("z")], 3)
            .await
            .unwrap();
        // x must still appear in the output.
        assert!(ids(&out).iter().any(|i| i == "x"));
    }

    #[test]
    fn fuse_helper_attaches_score() {
        let a = vec![doc("x"), doc("y")];
        let b = vec![doc("y"), doc("x")];
        let out = weighted_rrf_fuse_rerank(&[(a, 1.0), (b, 1.0)], 60.0, 2);
        assert!(out[0].score.is_some());
    }
}
