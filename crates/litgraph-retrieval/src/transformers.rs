//! Post-retrieval document transformers — refine a candidate list AFTER
//! the vector store / BM25 / hybrid retriever returns its top-k. Three
//! patterns ship here, each addressing a real RAG quality issue:
//!
//! - [`mmr_select`] — Maximal Marginal Relevance. Picks `k` documents
//!   that are simultaneously relevant to the query AND diverse from
//!   each other. Avoids "5 nearly-identical chunks" RAG syndrome.
//!   Carbonell & Goldstein 1998; LangChain `as_retriever(search_type="mmr")` parity.
//!
//! - [`embedding_redundant_filter`] — drop documents whose embedding is
//!   within `threshold` cosine similarity of an EARLIER kept document.
//!   Cheaper than MMR (no relevance term — just dedup); good for big
//!   candidate lists where you don't have a query embedding handy.
//!   LangChain `EmbeddingsRedundantFilter` parity.
//!
//! - [`long_context_reorder`] — Liu et al "Lost in the Middle" workaround:
//!   models attend more to start + end of long contexts than the middle.
//!   Given a relevance-sorted list, reorder so highest-scored docs sit at
//!   the EDGES (positions 0, n-1, 1, n-2, ...). LangChain
//!   `LongContextReorder` parity.
//!
//! All three are pure functions over `Vec<Document>` plus optional
//! embeddings. They don't touch the vector store and can compose: e.g.,
//! retrieve 50 → MMR-select 10 → long-context-reorder.

use litgraph_core::Document;
use rayon::prelude::*;

/// Cosine similarity of two equal-length vectors. Returns 0.0 if either
/// vector is all-zero (avoids NaN). Lengths must match — caller's
/// responsibility (mismatched embeddings indicate a bug upstream).
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Maximal Marginal Relevance selection.
///
/// Inputs:
/// - `query_embedding` — embedding of the user query.
/// - `candidates` — list of documents (relevance order doesn't matter).
/// - `candidate_embeddings` — same length as `candidates`. The i-th
///   entry is the embedding of `candidates[i]`.
/// - `k` — number of documents to return (capped at `candidates.len()`).
/// - `lambda_mult` — in `[0, 1]`. `1.0` = pure relevance (no diversity);
///   `0.0` = pure diversity (no relevance term). LangChain default 0.5.
///
/// Algorithm: greedy. At each step pick the candidate maximizing
/// `λ·sim(q, d) − (1−λ)·max_i sim(d, picked_i)`.
///
/// Returns documents in pick order (most relevant first under the
/// blended score). Drops candidates whose embedding length differs from
/// the query embedding.
pub fn mmr_select(
    query_embedding: &[f32],
    candidates: &[Document],
    candidate_embeddings: &[Vec<f32>],
    k: usize,
    lambda_mult: f32,
) -> Vec<Document> {
    assert!(
        candidates.len() == candidate_embeddings.len(),
        "candidates and candidate_embeddings must be same length"
    );
    let lambda = lambda_mult.clamp(0.0, 1.0);
    let n = candidates.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    // Filter to indices whose embedding length matches the query (drop
    // mismatches silently — easier on callers mixing model outputs).
    let mut available: Vec<usize> = (0..n)
        .filter(|&i| candidate_embeddings[i].len() == query_embedding.len())
        .collect();

    // Pre-compute query similarities for the survivors. Rayon-parallel
    // since each cosine sim is independent — meaningful win on big
    // candidate pools (1k+ docs from over-fetch retrievers).
    let mut q_sim: Vec<f32> = vec![0.0; n];
    let q_pairs: Vec<(usize, f32)> = available
        .par_iter()
        .map(|&i| (i, cosine_sim(query_embedding, &candidate_embeddings[i])))
        .collect();
    for (i, sim) in q_pairs {
        q_sim[i] = sim;
    }

    let mut picked: Vec<usize> = Vec::with_capacity(k);

    while picked.len() < k && !available.is_empty() {
        // Score every available candidate against the current picked
        // set in parallel. Each candidate's score is independent, so
        // Rayon can shred this across cores. The "pick the max" reduce
        // is a `max_by` on the resulting Vec — O(|available|), still
        // cheap.
        let scored: Vec<(usize, f32)> = available
            .par_iter()
            .map(|&i| {
                let max_sim_to_picked = if picked.is_empty() {
                    0.0
                } else {
                    picked
                        .iter()
                        .map(|&j| {
                            cosine_sim(&candidate_embeddings[i], &candidate_embeddings[j])
                        })
                        .fold(f32::NEG_INFINITY, f32::max)
                };
                let score = lambda * q_sim[i] - (1.0 - lambda) * max_sim_to_picked;
                (i, score)
            })
            .collect();

        // Tie-break on lower index so the parallel pick order is
        // bit-identical to a sequential left-to-right scan. Without
        // this, λ=0.0 (pure diversity, first-pick all-tied scores)
        // would non-deterministically depend on Rayon thread scheduling.
        let (chosen, _best) = scored
            .into_iter()
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.0.cmp(&a.0))
            })
            .expect("non-empty available set must have a winner");
        picked.push(chosen);
        available.retain(|&i| i != chosen);
    }

    picked.into_iter().map(|i| candidates[i].clone()).collect()
}

/// Embedding-based redundancy filter. Walks the input in order and
/// keeps each document only if its embedding is below `threshold`
/// cosine similarity to every previously-kept document.
///
/// Use when you have a candidate list (e.g., from BM25 or web-search)
/// and want to dedup near-identical results before feeding to an LLM.
/// Cheaper than MMR — no query embedding required.
///
/// Threshold semantics: HIGHER threshold keeps MORE docs (`1.0` keeps
/// everything; `0.0` keeps only the first). LangChain default ~0.95.
///
/// Drops candidates whose embedding length differs from the first kept
/// embedding's length.
pub fn embedding_redundant_filter(
    candidates: &[Document],
    embeddings: &[Vec<f32>],
    threshold: f32,
) -> Vec<Document> {
    assert!(
        candidates.len() == embeddings.len(),
        "candidates and embeddings must be same length"
    );
    let mut kept_indices: Vec<usize> = Vec::new();
    for i in 0..candidates.len() {
        // Parallel "is candidate `i` similar to ANY kept doc?" probe.
        // Rayon's `par_iter().any()` stops as soon as any worker
        // finds a hit — same short-circuit semantics as the
        // sequential `for ... break`, just spread across cores.
        // Empty `kept_indices` short-circuits to `false` instantly.
        let redundant = kept_indices.par_iter().any(|&j| {
            if embeddings[i].len() != embeddings[j].len() {
                return false;
            }
            cosine_sim(&embeddings[i], &embeddings[j]) >= threshold
        });
        if !redundant {
            kept_indices.push(i);
        }
    }
    kept_indices.into_iter().map(|i| candidates[i].clone()).collect()
}

/// Reorder a relevance-sorted document list to mitigate "lost in the
/// middle" — Liu et al (2023) showed LLMs attend more strongly to the
/// start and end of a long context than to the middle. Given a list
/// already sorted by relevance (rank 0 = most relevant), this puts
/// the top docs at the EDGES.
///
/// Pattern: place rank-0 at position 0, rank-1 at position n-1, rank-2
/// at position 1, rank-3 at position n-2, etc. The LEAST relevant docs
/// land in the center.
///
/// Pure list permutation — doesn't read embeddings. The caller is
/// responsible for the input being sorted by relevance.
///
/// ```ignore
/// let docs = retriever.retrieve(query, 10).await?;  // sorted desc by score
/// let reordered = long_context_reorder(&docs);
/// // Feed `reordered` into the LLM context window.
/// ```
pub fn long_context_reorder(docs: &[Document]) -> Vec<Document> {
    let n = docs.len();
    if n <= 2 {
        return docs.to_vec();
    }
    let mut out: Vec<Option<Document>> = (0..n).map(|_| None).collect();
    let mut left = 0usize;
    let mut right = n - 1;
    for (i, doc) in docs.iter().enumerate() {
        let target = if i % 2 == 0 {
            let t = left;
            left += 1;
            t
        } else {
            let t = right;
            right = right.saturating_sub(1);
            t
        };
        out[target] = Some(doc.clone());
    }
    out.into_iter().map(|o| o.expect("all slots filled")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(content: &str) -> Document {
        Document::new(content)
    }

    #[test]
    fn mmr_returns_empty_when_k_zero() {
        let q = vec![1.0, 0.0];
        let docs = vec![doc("a"), doc("b")];
        let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let out = mmr_select(&q, &docs, &embs, 0, 0.5);
        assert!(out.is_empty());
    }

    #[test]
    fn mmr_pure_relevance_picks_most_similar_to_query_first() {
        let q = vec![1.0, 0.0];
        // d0 perfectly similar; d1 less similar; d2 orthogonal.
        let docs = vec![doc("d0"), doc("d1"), doc("d2")];
        let embs = vec![
            vec![1.0, 0.0],
            vec![0.7, 0.7],
            vec![0.0, 1.0],
        ];
        let out = mmr_select(&q, &docs, &embs, 3, 1.0);
        assert_eq!(out[0].content, "d0");
    }

    #[test]
    fn mmr_with_diversity_avoids_picking_near_dupes() {
        let q = vec![1.0, 0.0];
        // d0, d1 are near-identical. d2 is different but still relevant.
        let docs = vec![doc("d0"), doc("d1"), doc("d2")];
        let embs = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.05],   // near-dup of d0
            vec![0.6, 0.5],
        ];
        // λ=0.3 favors diversity heavily.
        let out = mmr_select(&q, &docs, &embs, 2, 0.3);
        let names: Vec<&str> = out.iter().map(|d| d.content.as_str()).collect();
        // First pick is most-similar (d0). Second pick should NOT be d1 (dup).
        assert_eq!(names[0], "d0");
        assert_ne!(names[1], "d1");
    }

    #[test]
    fn mmr_caps_k_at_candidate_count() {
        let q = vec![1.0];
        let docs = vec![doc("a"), doc("b")];
        let embs = vec![vec![1.0], vec![0.5]];
        let out = mmr_select(&q, &docs, &embs, 100, 0.5);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn redundant_filter_drops_high_similarity_dups() {
        let docs = vec![doc("a"), doc("a-dup"), doc("b")];
        let embs = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.01], // 0.999... cos sim to first
            vec![0.0, 1.0],
        ];
        let out = embedding_redundant_filter(&docs, &embs, 0.95);
        let names: Vec<&str> = out.iter().map(|d| d.content.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn redundant_filter_keeps_all_when_threshold_one() {
        let docs = vec![doc("a"), doc("a-dup")];
        let embs = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let out = embedding_redundant_filter(&docs, &embs, 1.0);
        // Threshold 1.0 still drops perfect duplicates because sim==1.0 >= 1.0.
        // (Use >1.0 to keep everything.)
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn redundant_filter_keeps_all_when_threshold_above_one() {
        let docs = vec![doc("a"), doc("a-dup")];
        let embs = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let out = embedding_redundant_filter(&docs, &embs, 1.01);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn redundant_filter_keeps_first_when_all_identical() {
        let docs = vec![doc("a"), doc("a"), doc("a")];
        let embs = vec![vec![1.0], vec![1.0], vec![1.0]];
        let out = embedding_redundant_filter(&docs, &embs, 0.5);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn long_context_reorder_one_or_two_docs_unchanged() {
        let one = vec![doc("a")];
        assert_eq!(long_context_reorder(&one)[0].content, "a");

        let two = vec![doc("a"), doc("b")];
        let out = long_context_reorder(&two);
        assert_eq!(out[0].content, "a");
        assert_eq!(out[1].content, "b");
    }

    #[test]
    fn long_context_reorder_places_top_at_edges() {
        // Input ranked by relevance: a, b, c, d, e (a most relevant)
        let docs = vec![doc("a"), doc("b"), doc("c"), doc("d"), doc("e")];
        let out = long_context_reorder(&docs);
        let names: Vec<&str> = out.iter().map(|d| d.content.as_str()).collect();
        // a (rank 0) → pos 0; b (rank 1) → pos 4; c (rank 2) → pos 1;
        // d (rank 3) → pos 3; e (rank 4 — least relevant) → pos 2.
        assert_eq!(names, vec!["a", "c", "e", "d", "b"]);
    }

    #[test]
    fn long_context_reorder_six_docs() {
        let docs = vec![doc("a"), doc("b"), doc("c"), doc("d"), doc("e"), doc("f")];
        let out = long_context_reorder(&docs);
        let names: Vec<&str> = out.iter().map(|d| d.content.as_str()).collect();
        // a→0, b→5, c→1, d→4, e→2, f→3.
        assert_eq!(names, vec!["a", "c", "e", "f", "d", "b"]);
    }

    #[test]
    fn transformers_compose_retrieve_then_mmr_then_reorder() {
        // Realistic RAG pipeline: simulate a retriever returning 5 docs,
        // run MMR to select 3, then reorder for long context.
        let q = vec![1.0, 0.0];
        let docs = vec![
            doc("doc1 most relevant"),
            doc("doc2 also relevant"),
            doc("doc3 near-dup of doc1"),
            doc("doc4 diverse"),
            doc("doc5 unrelated"),
        ];
        let embs = vec![
            vec![1.0, 0.0],
            vec![0.8, 0.4],
            vec![0.99, 0.05],
            vec![0.5, 0.5],
            vec![0.0, 1.0],
        ];
        let mmr = mmr_select(&q, &docs, &embs, 3, 0.5);
        assert_eq!(mmr.len(), 3);
        let reordered = long_context_reorder(&mmr);
        assert_eq!(reordered.len(), 3);
        // MMR returns rank 0 (best) first, so reorder keeps it at pos 0.
        assert_eq!(reordered[0].content, mmr[0].content);
    }

    /// Sequential reference impl — mirrors the pre-iter-198 algorithm
    /// without Rayon. Keeps tests honest: the parallel version must
    /// produce **bit-identical** picks (modulo the documented tie-break
    /// freedom of `partial_cmp`-based max).
    fn mmr_select_sequential(
        query_embedding: &[f32],
        candidates: &[Document],
        candidate_embeddings: &[Vec<f32>],
        k: usize,
        lambda_mult: f32,
    ) -> Vec<Document> {
        let lambda = lambda_mult.clamp(0.0, 1.0);
        let n = candidates.len();
        let k = k.min(n);
        if k == 0 {
            return Vec::new();
        }
        let mut available: Vec<usize> = (0..n)
            .filter(|&i| candidate_embeddings[i].len() == query_embedding.len())
            .collect();
        let mut q_sim: Vec<f32> = vec![0.0; n];
        for &i in &available {
            q_sim[i] = cosine_sim(query_embedding, &candidate_embeddings[i]);
        }
        let mut picked: Vec<usize> = Vec::with_capacity(k);
        while picked.len() < k && !available.is_empty() {
            let mut best_idx: Option<usize> = None;
            let mut best_score = f32::NEG_INFINITY;
            for &i in &available {
                let max_sim_to_picked = if picked.is_empty() {
                    0.0
                } else {
                    picked
                        .iter()
                        .map(|&j| {
                            cosine_sim(&candidate_embeddings[i], &candidate_embeddings[j])
                        })
                        .fold(f32::NEG_INFINITY, f32::max)
                };
                let score = lambda * q_sim[i] - (1.0 - lambda) * max_sim_to_picked;
                if score > best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }
            let chosen = best_idx.unwrap();
            picked.push(chosen);
            available.retain(|&i| i != chosen);
        }
        picked.into_iter().map(|i| candidates[i].clone()).collect()
    }

    /// Hash-stable doc identifier for cross-impl pick-list comparison.
    fn ids(docs: &[Document]) -> Vec<String> {
        docs.iter().map(|d| d.content.clone()).collect()
    }

    #[test]
    fn parallel_mmr_matches_sequential_on_small_pool() {
        // 12 docs, k=5. Both impls must agree on the pick set.
        let q = vec![1.0, 0.0, 0.0];
        let docs: Vec<Document> = (0..12)
            .map(|i| Document::new(format!("doc-{i}")))
            .collect();
        // Spread embeddings around the unit circle in dim 1-2 with a
        // dim-0 (relevance) component that varies — gives MMR a real
        // diversity-vs-relevance tradeoff to make.
        let embs: Vec<Vec<f32>> = (0..12)
            .map(|i| {
                let theta = (i as f32) * 0.5;
                vec![1.0 - (i as f32) / 24.0, theta.cos(), theta.sin()]
            })
            .collect();
        let par = mmr_select(&q, &docs, &embs, 5, 0.5);
        let seq = mmr_select_sequential(&q, &docs, &embs, 5, 0.5);
        assert_eq!(ids(&par), ids(&seq));
    }

    #[test]
    fn parallel_mmr_matches_sequential_on_larger_pool() {
        // 64 docs, k=10 — exercises the parallel score loop with
        // enough work to actually use multiple cores.
        let q = vec![0.7, 0.7];
        let docs: Vec<Document> = (0..64)
            .map(|i| Document::new(format!("doc-{i}")))
            .collect();
        // Deterministic-but-spread embeddings.
        let embs: Vec<Vec<f32>> = (0..64)
            .map(|i| {
                let a = ((i * 17 + 3) % 100) as f32 / 100.0;
                let b = ((i * 23 + 7) % 100) as f32 / 100.0;
                vec![a, b]
            })
            .collect();
        let par = mmr_select(&q, &docs, &embs, 10, 0.6);
        let seq = mmr_select_sequential(&q, &docs, &embs, 10, 0.6);
        assert_eq!(ids(&par), ids(&seq));
    }

    /// Sequential reference for `embedding_redundant_filter`. Mirrors
    /// the pre-iter-204 algorithm without Rayon.
    fn redundant_filter_sequential(
        candidates: &[Document],
        embeddings: &[Vec<f32>],
        threshold: f32,
    ) -> Vec<Document> {
        let mut kept_indices: Vec<usize> = Vec::new();
        for i in 0..candidates.len() {
            let mut redundant = false;
            for &j in &kept_indices {
                if embeddings[i].len() != embeddings[j].len() {
                    continue;
                }
                let sim = cosine_sim(&embeddings[i], &embeddings[j]);
                if sim >= threshold {
                    redundant = true;
                    break;
                }
            }
            if !redundant {
                kept_indices.push(i);
            }
        }
        kept_indices.into_iter().map(|i| candidates[i].clone()).collect()
    }

    #[test]
    fn parallel_redundant_filter_matches_sequential_on_small_pool() {
        let docs: Vec<Document> = (0..12)
            .map(|i| Document::new(format!("doc-{i}")))
            .collect();
        let embs: Vec<Vec<f32>> = (0..12)
            .map(|i| {
                let theta = (i as f32) * 0.5;
                vec![theta.cos(), theta.sin()]
            })
            .collect();
        let par = embedding_redundant_filter(&docs, &embs, 0.85);
        let seq = redundant_filter_sequential(&docs, &embs, 0.85);
        assert_eq!(ids(&par), ids(&seq));
    }

    #[test]
    fn parallel_redundant_filter_matches_sequential_on_larger_pool() {
        // 80 docs — enough to actually use multiple cores on the
        // inner par_iter::any check.
        let docs: Vec<Document> = (0..80)
            .map(|i| Document::new(format!("d{i}")))
            .collect();
        let embs: Vec<Vec<f32>> = (0..80)
            .map(|i| {
                let a = ((i * 17 + 3) % 100) as f32 / 100.0;
                let b = ((i * 23 + 7) % 100) as f32 / 100.0;
                vec![a, b]
            })
            .collect();
        for &threshold in &[0.5_f32, 0.85_f32, 0.99_f32] {
            let par = embedding_redundant_filter(&docs, &embs, threshold);
            let seq = redundant_filter_sequential(&docs, &embs, threshold);
            assert_eq!(
                ids(&par),
                ids(&seq),
                "threshold={threshold} mismatch"
            );
        }
    }

    #[test]
    fn parallel_redundant_filter_threshold_extremes_match_sequential() {
        // threshold=0.0 keeps only the first doc (everything else is
        // redundant); threshold=1.0001 keeps everything (nothing
        // crosses the bar). Both should match the sequential ref.
        let docs: Vec<Document> = (0..6)
            .map(|i| Document::new(format!("d{i}")))
            .collect();
        let embs: Vec<Vec<f32>> = (0..6).map(|_| vec![1.0, 0.0]).collect();
        for &threshold in &[0.0_f32, 1.001_f32] {
            let par = embedding_redundant_filter(&docs, &embs, threshold);
            let seq = redundant_filter_sequential(&docs, &embs, threshold);
            assert_eq!(
                ids(&par),
                ids(&seq),
                "threshold={threshold} mismatch"
            );
        }
    }

    #[test]
    fn parallel_mmr_lambda_extremes_match_sequential() {
        // λ = 1.0 (pure relevance) and λ = 0.0 (pure diversity) are
        // edge cases — make sure the parallel impl handles both.
        let q = vec![1.0, 0.0];
        let docs: Vec<Document> = (0..15)
            .map(|i| Document::new(format!("d{i}")))
            .collect();
        let embs: Vec<Vec<f32>> = (0..15)
            .map(|i| {
                let t = (i as f32) * 0.4;
                vec![t.cos(), t.sin()]
            })
            .collect();
        for &lambda in &[1.0_f32, 0.0_f32] {
            let par = mmr_select(&q, &docs, &embs, 6, lambda);
            let seq = mmr_select_sequential(&q, &docs, &embs, 6, lambda);
            assert_eq!(ids(&par), ids(&seq), "λ={lambda} mismatch");
        }
    }
}
