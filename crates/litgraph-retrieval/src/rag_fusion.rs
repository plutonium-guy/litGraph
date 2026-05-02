//! `RagFusionRetriever` — LLM query expansion + reciprocal-rank
//! fusion over the per-query result lists.
//!
//! # Distinct from MultiQueryRetriever
//!
//! `MultiQueryRetriever` (iter 132-era) generates N paraphrases,
//! fans out, then **dedups by id with first-occurrence-wins**.
//! That preserves the original query's ranking but loses the
//! signal from how often / how highly a doc appeared across all
//! N paraphrases.
//!
//! `RagFusionRetriever` (this iter) uses the same paraphrase
//! generation but **fuses via reciprocal-rank fusion** (Cormack
//! et al. 2009) — a doc that ranks #2 in query A and #1 in
//! query B beats a doc that ranks #1 in query A but isn't
//! found in B. The technique was popularized as "RAG-Fusion"
//! by Raudaschl 2023; same idea with branding for retrieval-
//! augmented generation.
//!
//! # When to use which
//!
//! - **MultiQueryRetriever**: when you want to recall docs that
//!   use different vocabulary, but trust the underlying
//!   retriever's per-query ranking.
//! - **RagFusionRetriever**: when you want robustness across
//!   paraphrases — docs that consistently rank well across
//!   multiple phrasings get amplified, while flukes from any
//!   single phrasing get dampened.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{ChatModel, ChatOptions, Document, Error, Message, Result};
use tokio::task::JoinSet;

use crate::hybrid::rrf_fuse;
use crate::retriever::Retriever;

const DEFAULT_PROMPT: &str = "You are an assistant that rewrites a search query into N alternative phrasings. \
Output ONE phrasing per line, no numbering, no commentary, no blank lines. \
The phrasings should preserve the user's intent but use different words and sentence structure to \
maximize the chance of matching documents that use varied vocabulary.";

const DEFAULT_NUM_QUERIES: usize = 4;
/// Standard RRF dampening constant (Cormack 2009 §3). Higher
/// values flatten the rank-position weight curve.
const DEFAULT_RRF_K: f32 = 60.0;

pub struct RagFusionRetriever {
    pub base: Arc<dyn Retriever>,
    pub llm: Arc<dyn ChatModel>,
    pub num_variations: usize,
    pub system_prompt: String,
    pub include_original: bool,
    /// RRF dampening constant. Cormack 2009 picked 60 empirically.
    pub rrf_k: f32,
    /// Per-query retrieval depth. Each branch fetches `branch_k`
    /// docs, then RRF picks the top `k` from the fused union.
    /// Larger values give RRF more candidates to score across.
    pub branch_k: usize,
}

impl RagFusionRetriever {
    pub fn new(base: Arc<dyn Retriever>, llm: Arc<dyn ChatModel>) -> Self {
        Self {
            base,
            llm,
            num_variations: DEFAULT_NUM_QUERIES,
            system_prompt: DEFAULT_PROMPT.into(),
            include_original: true,
            rrf_k: DEFAULT_RRF_K,
            branch_k: 10,
        }
    }

    pub fn with_num_variations(mut self, n: usize) -> Self {
        self.num_variations = n.max(1);
        self
    }
    pub fn with_system_prompt(mut self, p: impl Into<String>) -> Self {
        self.system_prompt = p.into();
        self
    }
    pub fn with_include_original(mut self, b: bool) -> Self {
        self.include_original = b;
        self
    }
    pub fn with_rrf_k(mut self, k: f32) -> Self {
        self.rrf_k = k.max(1.0);
        self
    }
    pub fn with_branch_k(mut self, n: usize) -> Self {
        self.branch_k = n.max(1);
        self
    }

    /// Generate paraphrases via the LLM. Public so callers can
    /// preview / cache outside the retrieval path.
    pub async fn generate_queries(&self, query: &str) -> Result<Vec<String>> {
        let user_msg = format!(
            "Rewrite the following query into {} alternative phrasings.\n\nQuery: {}",
            self.num_variations, query,
        );
        let messages = vec![
            Message::system(self.system_prompt.clone()),
            Message::user(user_msg),
        ];
        let opts = ChatOptions {
            temperature: Some(0.0),
            max_tokens: Some(512),
            ..Default::default()
        };
        let resp = self
            .llm
            .invoke(messages, &opts)
            .await
            .map_err(|e| Error::other(format!("rag_fusion llm: {e}")))?;
        Ok(parse_queries(&resp.message.text_content()))
    }
}

#[async_trait]
impl Retriever for RagFusionRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let mut queries = self.generate_queries(query).await.unwrap_or_default();
        if self.include_original {
            let original = query.to_string();
            queries.retain(|q| q != &original);
            queries.insert(0, original);
        }
        if queries.is_empty() {
            queries.push(query.to_string());
        }

        // Per-query retrieval, parallel.
        let mut set: JoinSet<Result<Vec<Document>>> = JoinSet::new();
        let branch_k = self.branch_k.max(k);
        for q in queries {
            let base = self.base.clone();
            set.spawn(async move { base.retrieve(&q, branch_k).await });
        }

        let mut branches: Vec<Vec<Document>> = Vec::new();
        let mut errors = Vec::new();
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(docs)) => branches.push(docs),
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(je) => errors.push(format!("join: {je}")),
            }
        }
        if branches.is_empty() && !errors.is_empty() {
            return Err(Error::other(format!(
                "rag_fusion: all {} branches failed (first: {})",
                errors.len(),
                errors[0],
            )));
        }
        Ok(rrf_fuse(&branches, self.rrf_k, k))
    }
}

/// Strip numbering prefixes ("1. ", "1) ", "- ", "* "), drop
/// blanks, dedup. Mirrors the helper in `multi_query.rs` —
/// duplicated to keep the modules independent.
fn parse_queries(raw: &str) -> Vec<String> {
    use std::collections::HashSet;
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for line in raw.lines() {
        let mut s = line.trim().to_string();
        if s.is_empty() {
            continue;
        }
        for prefix in ["- ", "* ", "• "] {
            if let Some(rest) = s.strip_prefix(prefix) {
                s = rest.trim().to_string();
                break;
            }
        }
        let digits = s.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 && digits < s.len() {
            let rest = &s[digits..];
            if let Some(stripped) = rest
                .strip_prefix(". ")
                .or_else(|| rest.strip_prefix(") "))
            {
                s = stripped.trim().to_string();
            }
        }
        if !s.is_empty() && seen.insert(s.clone()) {
            out.push(s);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason};
    use litgraph_core::{ChatResponse, ContentPart, Role, TokenUsage};
    use std::sync::Mutex;

    fn doc(id: &str, content: &str) -> Document {
        Document::new(content.to_string()).with_id(id.to_string())
    }

    /// Scripted retriever: returns a fixed Vec for each query.
    struct ScriptedRetriever {
        responses: Mutex<std::collections::HashMap<String, Vec<Document>>>,
    }

    #[async_trait]
    impl Retriever for ScriptedRetriever {
        async fn retrieve(&self, query: &str, _k: usize) -> Result<Vec<Document>> {
            Ok(self
                .responses
                .lock()
                .unwrap()
                .get(query)
                .cloned()
                .unwrap_or_default())
        }
    }

    /// Scripted LLM: returns a fixed text on first invoke.
    struct ScriptedLlm {
        text: String,
    }

    #[async_trait]
    impl ChatModel for ScriptedLlm {
        fn name(&self) -> &str {
            "scripted"
        }
        async fn invoke(
            &self,
            _msgs: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text {
                        text: self.text.clone(),
                    }],
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "scripted".into(),
            })
        }
        async fn stream(
            &self,
            _msgs: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[test]
    fn parse_queries_strips_numeric_and_dash_prefixes() {
        let input = "1. how to fix a leaky faucet\n2) plumber tips\n- handle a drip\n* repair guide";
        let qs = parse_queries(input);
        assert_eq!(qs.len(), 4);
        assert_eq!(qs[0], "how to fix a leaky faucet");
        assert_eq!(qs[1], "plumber tips");
        assert_eq!(qs[2], "handle a drip");
        assert_eq!(qs[3], "repair guide");
    }

    #[tokio::test]
    async fn fuses_results_via_rrf_consistent_doc_wins() {
        // c is rank-1 in all three branches; a and b have mixed ranks.
        // RRF score(d) = sum of 1/(k+rank); k=60 default. c sums to
        // 3/61 ≈ 0.04918; the others sum to ≈0.04839 each. c wins
        // unambiguously.
        let mut map = std::collections::HashMap::new();
        map.insert(
            "original q".into(),
            vec![doc("c", "gamma"), doc("b", "beta"), doc("a", "alpha")],
        );
        map.insert(
            "para 1".into(),
            vec![doc("c", "gamma"), doc("b", "beta"), doc("a", "alpha")],
        );
        map.insert(
            "para 2".into(),
            vec![doc("c", "gamma"), doc("a", "alpha"), doc("b", "beta")],
        );
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm {
            text: "para 1\npara 2".into(),
        });
        let r = RagFusionRetriever::new(base, llm)
            .with_num_variations(2)
            .with_branch_k(5);
        let docs = r.retrieve("original q", 3).await.unwrap();
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].id.as_deref(), Some("c"));
    }

    #[tokio::test]
    async fn falls_back_to_literal_query_when_llm_returns_empty() {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "original q".into(),
            vec![doc("a", "alpha"), doc("b", "beta")],
        );
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        // LLM returns empty text (no paraphrases).
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm { text: "".into() });
        let r = RagFusionRetriever::new(base, llm)
            .with_include_original(false);
        let docs = r.retrieve("original q", 5).await.unwrap();
        // Even with no paraphrases AND include_original=false, we fall
        // back to the literal query so we degrade gracefully.
        assert_eq!(docs.len(), 2);
    }

    #[tokio::test]
    async fn include_original_keeps_user_phrasing() {
        // Only the original query is in the scripted retriever; the
        // paraphrase returns nothing. include_original=true ensures the
        // original is queried.
        let mut map = std::collections::HashMap::new();
        map.insert(
            "original q".into(),
            vec![doc("a", "alpha"), doc("b", "beta")],
        );
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm {
            text: "unknown paraphrase".into(),
        });
        let r = RagFusionRetriever::new(base, llm)
            .with_num_variations(1)
            .with_include_original(true);
        let docs = r.retrieve("original q", 5).await.unwrap();
        assert_eq!(docs.len(), 2);
        let ids: Vec<_> = docs.iter().filter_map(|d| d.id.as_deref()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[tokio::test]
    async fn truncates_to_top_k() {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "original q".into(),
            (0..10)
                .map(|i| doc(&format!("d{i}"), &format!("doc {i}")))
                .collect(),
        );
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm { text: "".into() });
        let r = RagFusionRetriever::new(base, llm).with_branch_k(20);
        let docs = r.retrieve("original q", 3).await.unwrap();
        assert_eq!(docs.len(), 3);
    }
}
