//! `EnsembleRetriever` — weighted Reciprocal Rank Fusion across N child
//! retrievers, fanned out concurrently.
//!
//! Unlike [`HybridRetriever`](crate::HybridRetriever) (which uses *equal*
//! weights), `EnsembleRetriever` lets the caller bias the fusion towards
//! higher-precision children. The canonical use-case is BM25 + dense
//! vector retrieval where the dense side is more reliable; rather than
//! letting noisy BM25 hits dilute the fused list, the user passes
//! `weights = [0.3, 0.7]` and the dense retriever's ranks dominate.
//!
//! Direct LangChain `EnsembleRetriever` parity, with the parallelism win
//! `HybridRetriever` already gives.
//!
//! # Scoring
//!
//! ```text
//!   score(d) = Σ_i  w_i / (k_rrf + rank_i(d))
//! ```
//!
//! where `rank_i` is the 1-indexed position of `d` in the *i*-th child's
//! result list, and `w_i` is that child's weight (un-normalised — only
//! the ratios matter, since RRF is scale-invariant).
//!
//! # Per-child over-fetch
//!
//! Like the hybrid retriever, each child is asked for `per_child_k * k`
//! candidates (default `k * 2`) so docs that fall just outside the
//! top-`k` of one branch can still get fused in via another branch.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;
use litgraph_core::{Document, Error, Result};

use crate::retriever::Retriever;

/// Default RRF damping. Matches Cormack/Clarke/Buettcher 2009 and the
/// LangChain default.
const DEFAULT_RRF_K: f32 = 60.0;

pub struct EnsembleRetriever {
    /// Children + their weights. Weights are un-normalised; only ratios matter.
    pub children: Vec<(Arc<dyn Retriever>, f32)>,
    pub rrf_k: f32,
    /// Pull `per_child_k.unwrap_or(k * 2)` candidates from each child.
    pub per_child_k: Option<usize>,
}

impl EnsembleRetriever {
    /// Equal-weight ensemble — equivalent to a `HybridRetriever`, but kept
    /// as a separate constructor for API symmetry with `with_weights`.
    pub fn new(children: Vec<Arc<dyn Retriever>>) -> Self {
        let weighted = children.into_iter().map(|c| (c, 1.0)).collect();
        Self {
            children: weighted,
            rrf_k: DEFAULT_RRF_K,
            per_child_k: None,
        }
    }

    /// Build with explicit per-child weights. `weights.len()` must equal
    /// `children.len()`; otherwise the `retrieve` call returns an
    /// `Error::invalid`.
    pub fn with_weights(children: Vec<Arc<dyn Retriever>>, weights: Vec<f32>) -> Result<Self> {
        if children.len() != weights.len() {
            return Err(Error::invalid(format!(
                "EnsembleRetriever: children ({}) and weights ({}) length mismatch",
                children.len(),
                weights.len(),
            )));
        }
        if children.is_empty() {
            return Err(Error::invalid("EnsembleRetriever: need at least one child"));
        }
        Ok(Self {
            children: children.into_iter().zip(weights).collect(),
            rrf_k: DEFAULT_RRF_K,
            per_child_k: None,
        })
    }

    pub fn with_rrf_k(mut self, k: f32) -> Self {
        self.rrf_k = k;
        self
    }

    pub fn with_per_child_k(mut self, k: usize) -> Self {
        self.per_child_k = Some(k);
        self
    }
}

#[async_trait]
impl Retriever for EnsembleRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let child_k = self.per_child_k.unwrap_or(k * 2).max(1);
        let futures = self.children.iter().map(|(c, _)| {
            let c = c.clone();
            let q = query.to_string();
            async move { c.retrieve(&q, child_k).await }
        });
        let results = join_all(futures).await;

        let mut branches: Vec<(Vec<Document>, f32)> = Vec::with_capacity(results.len());
        for (r, (_, w)) in results.into_iter().zip(self.children.iter()) {
            branches.push((r?, *w));
        }
        Ok(weighted_rrf_fuse(&branches, self.rrf_k, k))
    }
}

/// Weighted RRF — `branches` is `(docs, weight)` pairs. A `weight` of `0.0`
/// drops that branch from the fusion entirely (cheap kill-switch in
/// A/B-testing setups). Negative weights are allowed and behave as
/// "down-vote" — useful for paranoid setups where one retriever is
/// known to be adversarial-trapped, but the more typical pattern is to
/// just exclude it.
pub fn weighted_rrf_fuse(
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

    /// Returns a fixed list of `Document`s every call — fine for ranking
    /// tests where the query is irrelevant.
    struct FixedRetriever {
        docs: Vec<Document>,
    }

    impl FixedRetriever {
        fn from_ids(ids: &[&str]) -> Arc<dyn Retriever> {
            let docs = ids
                .iter()
                .map(|id| Document {
                    id: Some((*id).to_string()),
                    content: format!("doc {id}"),
                    metadata: std::collections::HashMap::new(),
                    score: None,
                })
                .collect();
            Arc::new(Self { docs })
        }
    }

    #[async_trait]
    impl Retriever for FixedRetriever {
        async fn retrieve(&self, _query: &str, k: usize) -> Result<Vec<Document>> {
            Ok(self.docs.iter().take(k).cloned().collect())
        }
    }

    fn ids(docs: &[Document]) -> Vec<String> {
        docs.iter().map(|d| d.id.clone().unwrap()).collect()
    }

    #[tokio::test]
    async fn equal_weights_match_unweighted_rrf_order() {
        // Two retrievers with disjoint top results — equal-weight RRF
        // should produce a known order.
        let a = FixedRetriever::from_ids(&["a1", "a2", "a3"]);
        let b = FixedRetriever::from_ids(&["b1", "b2", "b3"]);
        let r = EnsembleRetriever::new(vec![a, b]);
        let out = r.retrieve("q", 6).await.unwrap();
        // a1 and b1 tie at rank 1; subsequent ranks tie too. The
        // `HashMap` iteration order isn't stable, but the *scores* must
        // be equal in pairs and the set must be {a1,a2,a3,b1,b2,b3}.
        let mut got: Vec<String> = ids(&out);
        got.sort();
        assert_eq!(got, vec!["a1", "a2", "a3", "b1", "b2", "b3"]);
    }

    #[tokio::test]
    async fn higher_weight_branch_wins_top_slot() {
        // Same disjoint top-1's, but branch B gets 10× the weight.
        let a = FixedRetriever::from_ids(&["a1", "a2", "a3"]);
        let b = FixedRetriever::from_ids(&["b1", "b2", "b3"]);
        let r = EnsembleRetriever::with_weights(vec![a, b], vec![1.0, 10.0]).unwrap();
        let out = r.retrieve("q", 1).await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id.as_deref(), Some("b1"));
    }

    #[tokio::test]
    async fn weight_zero_branch_drops_out() {
        // Branch B is muted entirely — only A's docs should appear.
        let a = FixedRetriever::from_ids(&["a1", "a2", "a3"]);
        let b = FixedRetriever::from_ids(&["b1", "b2", "b3"]);
        let r = EnsembleRetriever::with_weights(vec![a, b], vec![1.0, 0.0]).unwrap();
        let out = r.retrieve("q", 10).await.unwrap();
        let got: Vec<String> = ids(&out);
        assert_eq!(got, vec!["a1", "a2", "a3"]);
    }

    #[tokio::test]
    async fn shared_docs_sum_contributions() {
        // Doc "x" appears at rank 1 in A and rank 1 in B — its score
        // should beat any doc that only shows up once.
        let a = FixedRetriever::from_ids(&["x", "a2", "a3"]);
        let b = FixedRetriever::from_ids(&["x", "b2", "b3"]);
        let r = EnsembleRetriever::with_weights(vec![a, b], vec![1.0, 1.0]).unwrap();
        let out = r.retrieve("q", 1).await.unwrap();
        assert_eq!(out[0].id.as_deref(), Some("x"));
        // Score = 1/61 + 1/61 = 2/61 ≈ 0.0328.
        let score = out[0].score.unwrap();
        assert!((score - (2.0 / 61.0)).abs() < 1e-6, "got {score}");
    }

    #[tokio::test]
    async fn weights_length_mismatch_errors() {
        let a = FixedRetriever::from_ids(&["a1"]);
        match EnsembleRetriever::with_weights(vec![a], vec![1.0, 2.0]) {
            Err(e) => assert!(format!("{e}").contains("length mismatch")),
            Ok(_) => panic!("expected length-mismatch error"),
        }
    }

    #[tokio::test]
    async fn empty_children_errors() {
        match EnsembleRetriever::with_weights(vec![], vec![]) {
            Err(e) => assert!(format!("{e}").contains("at least one")),
            Ok(_) => panic!("expected empty-children error"),
        }
    }

    #[tokio::test]
    async fn child_error_propagates() {
        struct Fail;
        #[async_trait]
        impl Retriever for Fail {
            async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
                Err(Error::other("boom"))
            }
        }
        let a = FixedRetriever::from_ids(&["a1"]);
        let r = EnsembleRetriever::new(vec![a, Arc::new(Fail) as Arc<dyn Retriever>]);
        let err = r.retrieve("q", 5).await.unwrap_err();
        assert!(format!("{err}").contains("boom"));
    }

    #[tokio::test]
    async fn per_child_k_overfetch_respected() {
        // Construct a retriever that records the `k` it was asked for.
        struct Spy {
            k_seen: std::sync::Mutex<Option<usize>>,
        }
        #[async_trait]
        impl Retriever for Spy {
            async fn retrieve(&self, _q: &str, k: usize) -> Result<Vec<Document>> {
                *self.k_seen.lock().unwrap() = Some(k);
                Ok(vec![])
            }
        }
        let spy = Arc::new(Spy {
            k_seen: std::sync::Mutex::new(None),
        });
        let r =
            EnsembleRetriever::new(vec![spy.clone() as Arc<dyn Retriever>]).with_per_child_k(20);
        let _ = r.retrieve("q", 3).await.unwrap();
        assert_eq!(*spy.k_seen.lock().unwrap(), Some(20));
    }

    #[test]
    fn weighted_rrf_fuse_is_deterministic_for_distinct_scores() {
        let mk = |id: &str| Document {
            id: Some(id.into()),
            content: id.into(),
            metadata: std::collections::HashMap::new(),
            score: None,
        };
        // A: a, b, c — weight 1.0
        // B: c, b, a — weight 2.0
        // Doc c appears at rank 1 in B (high contrib) and rank 3 in A.
        let a = vec![mk("a"), mk("b"), mk("c")];
        let b = vec![mk("c"), mk("b"), mk("a")];
        let out = weighted_rrf_fuse(&[(a, 1.0), (b, 2.0)], 60.0, 3);
        // c should be #1 because B's weight dominates.
        assert_eq!(out[0].id.as_deref(), Some("c"));
    }
}
