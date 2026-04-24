//! ContextualCompressionRetriever — wraps a base Retriever with a
//! `Compressor` that filters / extracts relevant content from retrieved
//! documents BEFORE handing them to the LLM. Direct LangChain parity.
//!
//! # Why
//!
//! Naive RAG dumps full retrieved chunks into the prompt. With k=10 and
//! 1k-token chunks, that's 10k tokens of context — most of which is
//! irrelevant to the user's actual question. Compression reduces that by:
//! 1. **Filtering** — drop docs that aren't actually relevant
//!    (`EmbeddingsFilterCompressor`).
//! 2. **Extracting** — keep docs but replace their content with only the
//!    sentences that answer the query (`LlmExtractCompressor`).
//!
//! Both compressors are independent `Compressor` impls so callers can stack
//! them: filter first (cheap, drops obvious misses), then extract (LLM cost
//! scales with surviving doc count).

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{ChatModel, ChatOptions, Document, Embeddings, Error, Message, Result};
use tokio::task::JoinSet;

use crate::retriever::Retriever;

/// A document compressor — takes the query and the retrieved docs, returns
/// a (possibly smaller) set of (possibly-rewritten) docs.
#[async_trait]
pub trait Compressor: Send + Sync {
    async fn compress(&self, query: &str, docs: Vec<Document>) -> Result<Vec<Document>>;
}

const DEFAULT_EXTRACT_PROMPT: &str = "Given the following query and document, extract ONLY the verbatim sentences \
from the document that are directly relevant to answering the query. \
Preserve the original wording. If NO sentences are relevant, output exactly: NO_OUTPUT";

const NO_OUTPUT_SENTINEL: &str = "NO_OUTPUT";

/// LLM-driven extractive compressor. For each retrieved doc, asks the LLM
/// to keep only the sentences relevant to the query. Docs whose LLM
/// response is `NO_OUTPUT` (or empty) are dropped entirely.
///
/// One LLM call PER doc, fanned out in parallel via `JoinSet`. For k=10
/// retrieved docs that's 10 concurrent calls — usually faster than one
/// long-context-call asking the LLM to do all docs at once, and degrades
/// independently if any single doc errors.
pub struct LlmExtractCompressor {
    pub llm: Arc<dyn ChatModel>,
    pub prompt: String,
    pub max_concurrency: usize,
}

impl LlmExtractCompressor {
    pub fn new(llm: Arc<dyn ChatModel>) -> Self {
        Self { llm, prompt: DEFAULT_EXTRACT_PROMPT.into(), max_concurrency: 8 }
    }

    pub fn with_prompt(mut self, p: impl Into<String>) -> Self { self.prompt = p.into(); self }
    pub fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1); self
    }
}

#[async_trait]
impl Compressor for LlmExtractCompressor {
    async fn compress(&self, query: &str, docs: Vec<Document>) -> Result<Vec<Document>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let sem = Arc::new(tokio::sync::Semaphore::new(self.max_concurrency));
        // Spawn one LLM call per doc, tagging with original index so we can
        // restore order after the JoinSet returns out-of-completion-order.
        type Out = (usize, std::result::Result<String, String>);
        let mut set: JoinSet<Out> = JoinSet::new();
        for (idx, doc) in docs.iter().enumerate() {
            let llm = self.llm.clone();
            let sem = sem.clone();
            let prompt = self.prompt.clone();
            let q = query.to_string();
            let content = doc.content.clone();
            set.spawn(async move {
                let _permit = match sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => return (idx, Err("semaphore closed".into())),
                };
                let user_msg = format!("Query: {q}\n\nDocument:\n{content}");
                let messages = vec![Message::system(prompt), Message::user(user_msg)];
                let opts = ChatOptions {
                    temperature: Some(0.0),
                    max_tokens: Some(1024),
                    ..Default::default()
                };
                match llm.invoke(messages, &opts).await {
                    Ok(r) => (idx, Ok(r.message.text_content())),
                    Err(e) => (idx, Err(e.to_string())),
                }
            });
        }
        // Restore order: collect (idx, result) then sort by idx.
        let mut by_idx: Vec<Option<std::result::Result<String, String>>> =
            (0..docs.len()).map(|_| None).collect();
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok((idx, res)) => by_idx[idx] = Some(res),
                Err(je) => return Err(Error::other(format!("compress join: {je}"))),
            }
        }

        // Build the compressed result. Skip docs whose LLM returned NO_OUTPUT
        // / empty / errored (treat error as drop — better to lose one doc
        // than fail the whole retrieval).
        let mut out = Vec::with_capacity(docs.len());
        for (i, mut doc) in docs.into_iter().enumerate() {
            let extracted = match by_idx[i].take() {
                Some(Ok(s)) => s.trim().to_string(),
                Some(Err(_)) | None => continue,
            };
            if extracted.is_empty() || extracted.eq_ignore_ascii_case(NO_OUTPUT_SENTINEL) {
                continue;
            }
            doc.content = extracted;
            out.push(doc);
        }
        Ok(out)
    }
}

/// Embeddings-based filter compressor. Computes cosine similarity between
/// the query embedding and each doc's embedding (re-embedded inline — most
/// vector stores already did this once, but the doc that came back doesn't
/// carry its embedding). Drops docs below `similarity_threshold`.
///
/// Cheap (one batch embedding call) but coarse — works well as a pre-filter
/// before the more expensive `LlmExtractCompressor`.
pub struct EmbeddingsFilterCompressor {
    pub embeddings: Arc<dyn Embeddings>,
    pub similarity_threshold: f32,
}

impl EmbeddingsFilterCompressor {
    pub fn new(embeddings: Arc<dyn Embeddings>, similarity_threshold: f32) -> Self {
        Self { embeddings, similarity_threshold }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na.sqrt()) * (nb.sqrt());
    if denom == 0.0 { 0.0 } else { dot / denom }
}

#[async_trait]
impl Compressor for EmbeddingsFilterCompressor {
    async fn compress(&self, query: &str, docs: Vec<Document>) -> Result<Vec<Document>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let q_emb = self.embeddings.embed_query(query).await?;
        let texts: Vec<String> = docs.iter().map(|d| d.content.clone()).collect();
        let doc_embs = self.embeddings.embed_documents(&texts).await?;
        let mut out = Vec::new();
        for (i, mut doc) in docs.into_iter().enumerate() {
            let sim = cosine_similarity(&q_emb, &doc_embs[i]);
            if sim >= self.similarity_threshold {
                // Update score so downstream consumers can sort/filter.
                doc.score = Some(sim);
                out.push(doc);
            }
        }
        // Sort by similarity descending — best-relevance-first ordering.
        out.sort_by(|a, b| {
            b.score.unwrap_or(0.0).partial_cmp(&a.score.unwrap_or(0.0)).unwrap()
        });
        Ok(out)
    }
}

/// Stack of compressors applied in order. Use this when you want
/// "filter → extract" pipelines without writing a wrapper.
pub struct PipelineCompressor {
    pub steps: Vec<Arc<dyn Compressor>>,
}

impl PipelineCompressor {
    pub fn new(steps: Vec<Arc<dyn Compressor>>) -> Self { Self { steps } }
}

#[async_trait]
impl Compressor for PipelineCompressor {
    async fn compress(&self, query: &str, mut docs: Vec<Document>) -> Result<Vec<Document>> {
        for step in &self.steps {
            docs = step.compress(query, docs).await?;
            if docs.is_empty() {
                break;
            }
        }
        Ok(docs)
    }
}

/// Retriever wrapper: base.retrieve(query, k * over_fetch) → compressor.compress(query, docs).
/// Returns up to `k` compressed docs.
pub struct ContextualCompressionRetriever {
    pub base: Arc<dyn Retriever>,
    pub compressor: Arc<dyn Compressor>,
    /// Multiplier on `k` when calling base — over-fetch so the compressor
    /// has room to drop irrelevant docs without underrunning the requested k.
    /// Default 2.
    pub over_fetch_factor: usize,
}

impl ContextualCompressionRetriever {
    pub fn new(base: Arc<dyn Retriever>, compressor: Arc<dyn Compressor>) -> Self {
        Self { base, compressor, over_fetch_factor: 2 }
    }

    pub fn with_over_fetch_factor(mut self, n: usize) -> Self {
        self.over_fetch_factor = n.max(1); self
    }
}

#[async_trait]
impl Retriever for ContextualCompressionRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let raw = self.base.retrieve(query, k * self.over_fetch_factor).await?;
        let mut compressed = self.compressor.compress(query, raw).await?;
        compressed.truncate(k);
        Ok(compressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use litgraph_core::model::{ChatStream, FinishReason};
    use litgraph_core::{ChatResponse, ContentPart, Result as LgResult, Role, TokenUsage};
    use std::sync::Mutex;

    /// Echo-extractor LLM: returns "NO_OUTPUT" when content lacks the
    /// query keyword, otherwise returns the relevant sentence verbatim.
    /// Lets tests assert deterministic compression behavior.
    struct ExtractLlm;
    #[async_trait]
    impl ChatModel for ExtractLlm {
        fn name(&self) -> &str { "extract" }
        async fn invoke(&self, messages: Vec<Message>, _o: &ChatOptions) -> LgResult<ChatResponse> {
            // The user message is "Query: <q>\n\nDocument:\n<doc>".
            let user = messages.last().unwrap().text_content();
            let parts: Vec<&str> = user.splitn(2, "\n\nDocument:\n").collect();
            let query = parts[0].trim_start_matches("Query: ").to_string();
            let doc = parts.get(1).copied().unwrap_or("").to_string();
            // Pull the first sentence containing any query word, else NO_OUTPUT.
            let q_words: Vec<&str> = query.split_whitespace().collect();
            let pick = doc
                .split('.')
                .find(|sentence| q_words.iter().any(|qw| sentence.to_lowercase().contains(&qw.to_lowercase())))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            let text = if pick.is_empty() {
                "NO_OUTPUT".to_string()
            } else {
                format!("{pick}.")
            };
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text }],
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "extract".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> LgResult<ChatStream> {
            unimplemented!()
        }
    }

    /// Length-based deterministic embedder for test stability.
    struct LenEmb;
    #[async_trait]
    impl Embeddings for LenEmb {
        fn name(&self) -> &str { "len" }
        fn dimensions(&self) -> usize { 4 }
        async fn embed_query(&self, q: &str) -> LgResult<Vec<f32>> {
            // Embed the query as the SAME vector as a doc with the same length
            // so we can deterministically tune similarity.
            Ok(vec![q.len() as f32, 1.0, 0.0, 0.0])
        }
        async fn embed_documents(&self, ts: &[String]) -> LgResult<Vec<Vec<f32>>> {
            Ok(ts.iter().map(|t| vec![t.len() as f32, 1.0, 0.0, 0.0]).collect())
        }
    }

    /// Pre-canned base retriever.
    struct CannedBase {
        docs: Vec<Document>,
        last_k: Mutex<usize>,
    }
    #[async_trait]
    impl Retriever for CannedBase {
        async fn retrieve(&self, _q: &str, k: usize) -> LgResult<Vec<Document>> {
            *self.last_k.lock().unwrap() = k;
            Ok(self.docs.clone().into_iter().take(k).collect())
        }
    }

    fn doc(content: &str) -> Document { Document::new(content) }

    #[tokio::test]
    async fn llm_extract_keeps_relevant_drops_no_output() {
        let llm = Arc::new(ExtractLlm);
        let comp = LlmExtractCompressor::new(llm);
        let docs = vec![
            doc("Rust prevents data races. Other unrelated content here."),
            doc("Python uses a GIL. Threads are limited."),
            doc("Pure boilerplate with no relation to anything."),
        ];
        let out = comp.compress("data races", docs).await.unwrap();
        // Doc 1 has a sentence matching "data races" → kept (just that sentence).
        // Docs 2 & 3 don't → dropped.
        assert_eq!(out.len(), 1);
        assert!(out[0].content.contains("data races"));
        assert!(!out[0].content.contains("Other unrelated"), "extract should narrow content");
    }

    #[tokio::test]
    async fn llm_extract_preserves_input_order_across_parallel_calls() {
        // Spawn 5 concurrent extractions; ensure the surviving docs come back
        // in the same order they went in, regardless of completion order.
        let llm = Arc::new(ExtractLlm);
        let comp = LlmExtractCompressor::new(llm);
        let docs = (0..5)
            .map(|i| doc(&format!("doc {i} mentions rust here")))
            .collect();
        let out = comp.compress("rust", docs).await.unwrap();
        // All 5 contain "rust" → all kept; order preserved.
        assert_eq!(out.len(), 5);
        for (i, d) in out.iter().enumerate() {
            assert!(d.content.contains(&format!("doc {i}")), "got order: {}", d.content);
        }
    }

    #[tokio::test]
    async fn embeddings_filter_drops_docs_below_threshold() {
        // Query "abc" (len 3). Doc with len 3 → cosine sim ≈ 1.0 (kept).
        // Doc with len 100 → very different vector → cosine sim much less.
        let comp = EmbeddingsFilterCompressor::new(Arc::new(LenEmb), 0.999);
        let docs = vec![
            doc("xyz"),                  // len 3 → matches
            doc("a".repeat(100).as_str()), // len 100 → drops
        ];
        let out = comp.compress("abc", docs).await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, "xyz");
        assert!(out[0].score.unwrap() > 0.999);
    }

    #[tokio::test]
    async fn embeddings_filter_sorts_by_descending_similarity() {
        let comp = EmbeddingsFilterCompressor::new(Arc::new(LenEmb), 0.0);
        let docs = vec![
            doc("a".repeat(50).as_str()),
            doc("xyz"),  // len 3
            doc("a".repeat(20).as_str()),
        ];
        let out = comp.compress("abc", docs).await.unwrap();  // query len 3
        // xyz (len 3) ranks first because vector matches query best.
        assert_eq!(out[0].content, "xyz");
    }

    #[tokio::test]
    async fn pipeline_compressor_runs_steps_in_order_and_short_circuits_on_empty() {
        // First step: filter that drops everything (threshold = 2.0, impossible).
        // Second step: would crash if it ran (no llm), so verify short-circuit.
        struct ExplodeIfRun;
        #[async_trait]
        impl Compressor for ExplodeIfRun {
            async fn compress(&self, _q: &str, _docs: Vec<Document>) -> LgResult<Vec<Document>> {
                Err(Error::other("should never run"))
            }
        }
        let pipe = PipelineCompressor::new(vec![
            Arc::new(EmbeddingsFilterCompressor::new(Arc::new(LenEmb), 2.0)),
            Arc::new(ExplodeIfRun),
        ]);
        let out = pipe.compress("query", vec![doc("anything")]).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn ccr_over_fetches_then_truncates_to_k() {
        let docs: Vec<Document> = (0..10)
            .map(|i| doc(&format!("doc {i} talks about rust")))
            .collect();
        let base = Arc::new(CannedBase { docs, last_k: Mutex::new(0) });
        let comp = LlmExtractCompressor::new(Arc::new(ExtractLlm));
        let r = ContextualCompressionRetriever::new(base.clone(), Arc::new(comp));
        let out = r.retrieve("rust", 3).await.unwrap();
        assert_eq!(out.len(), 3);
        // base saw k * 2 = 6, NOT k = 3.
        assert_eq!(*base.last_k.lock().unwrap(), 6);
    }

    #[tokio::test]
    async fn ccr_returns_empty_when_compressor_drops_everything() {
        let docs = vec![doc("nothing related"), doc("also unrelated")];
        let base = Arc::new(CannedBase { docs, last_k: Mutex::new(0) });
        let comp = LlmExtractCompressor::new(Arc::new(ExtractLlm));
        let r = ContextualCompressionRetriever::new(base, Arc::new(comp));
        let out = r.retrieve("rust", 5).await.unwrap();
        assert!(out.is_empty());
    }
}
