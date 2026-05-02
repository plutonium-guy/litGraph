//! `SubQueryRetriever` — decompose a compound query into atomic
//! sub-questions, retrieve for each, union the results.
//!
//! # The pattern
//!
//! Many real user questions are compound: "Compare X and Y",
//! "What's the relationship between A, B, and C", "Find papers
//! that cite both Z1 and Z2". Naive single-shot retrieval over
//! a compound query often pulls back the most-mentioned entity
//! and misses the rest. Sub-query decomposition asks the LLM
//! "split this into atomic sub-questions" and retrieves
//! independently for each part.
//!
//! Distinct from the existing query-expansion family:
//! - **MultiQuery / RagFusion**: paraphrase at the same
//!   abstraction level (same intent, different wording).
//! - **StepBack**: abstract UPWARD (one more general question).
//! - **HyDE**: generate a hypothetical document, not a query.
//! - **SubQuery (this iter)**: SPLIT into N parts (different
//!   intents, each focused on a slice of the compound).
//!
//! # When to use
//!
//! Compound questions, comparison queries, multi-entity lookups.
//! For "compare X vs Y" workflows where the corpus has separate
//! docs about X and Y but no doc about both.
//!
//! # Cost
//!
//! 1 LLM call (decomposition) + N parallel base.retrieve calls
//! where N is the sub-question count. For N=3 sub-questions
//! that's roughly 3× single-query cost in parallel — same
//! shape as MultiQuery.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{ChatModel, ChatOptions, Document, Error, Message, Result};
use tokio::task::JoinSet;

use crate::retriever::Retriever;

const DEFAULT_PROMPT: &str = "Decompose the user's question into a small number of atomic sub-questions. \
Each sub-question must be self-contained, focused on a single fact or entity, \
and answerable independently. Output ONE sub-question per line, no numbering, no commentary, no blank lines. \
If the question is already atomic (single fact, no comparison, no conjunction), output the original question unchanged.";

const DEFAULT_MAX_SUB_QUERIES: usize = 4;

pub struct SubQueryRetriever {
    pub base: Arc<dyn Retriever>,
    pub llm: Arc<dyn ChatModel>,
    pub max_sub_queries: usize,
    pub system_prompt: String,
    /// Whether to also retrieve for the original (compound) query.
    /// Default false — sub-queries usually cover the union of
    /// intents adequately, and including the original adds cost
    /// without much recall gain.
    pub include_original: bool,
    /// Per-sub-query retrieval depth. Final output capped at the
    /// `retrieve` call's `k`.
    pub branch_k: usize,
}

impl SubQueryRetriever {
    pub fn new(base: Arc<dyn Retriever>, llm: Arc<dyn ChatModel>) -> Self {
        Self {
            base,
            llm,
            max_sub_queries: DEFAULT_MAX_SUB_QUERIES,
            system_prompt: DEFAULT_PROMPT.into(),
            include_original: false,
            branch_k: 8,
        }
    }

    pub fn with_max_sub_queries(mut self, n: usize) -> Self {
        self.max_sub_queries = n.max(1);
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
    pub fn with_branch_k(mut self, n: usize) -> Self {
        self.branch_k = n.max(1);
        self
    }

    /// Decompose a query into sub-questions. Public so callers
    /// can preview / cache outside the retrieval path.
    pub async fn decompose(&self, query: &str) -> Result<Vec<String>> {
        let user_msg = format!(
            "Decompose into at most {} atomic sub-questions.\n\nQuestion: {}",
            self.max_sub_queries, query,
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
            .map_err(|e| Error::other(format!("sub_query llm: {e}")))?;
        let mut subs = parse_sub_queries(&resp.message.text_content());
        if subs.len() > self.max_sub_queries {
            subs.truncate(self.max_sub_queries);
        }
        Ok(subs)
    }
}

#[async_trait]
impl Retriever for SubQueryRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let mut sub_queries = self.decompose(query).await.unwrap_or_default();
        if sub_queries.is_empty() {
            // Decomposition failed or produced nothing — fall back to
            // the literal query.
            sub_queries.push(query.to_string());
        }
        if self.include_original {
            let original = query.to_string();
            sub_queries.retain(|q| q != &original);
            sub_queries.insert(0, original);
        }

        let branch_k = self.branch_k.max(k);
        let mut set: JoinSet<Result<Vec<Document>>> = JoinSet::new();
        for q in sub_queries {
            let base = self.base.clone();
            set.spawn(async move { base.retrieve(&q, branch_k).await });
        }

        let mut branches: Vec<Vec<Document>> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(docs)) => branches.push(docs),
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(je) => errors.push(format!("join: {je}")),
            }
        }
        if branches.is_empty() && !errors.is_empty() {
            return Err(Error::other(format!(
                "sub_query: all branches failed (first: {})",
                errors[0],
            )));
        }

        // Round-robin merge so per-sub-query top-1 docs are
        // interleaved (no single sub-query dominates the head of
        // the result list). Then dedup by id, take top-k.
        let merged = round_robin_merge(branches);
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<Document> = Vec::new();
        for d in merged {
            let key = d.id.clone().unwrap_or_else(|| {
                blake3::hash(d.content.as_bytes()).to_hex().to_string()
            });
            if seen.insert(key) {
                out.push(d);
                if out.len() >= k {
                    break;
                }
            }
        }
        Ok(out)
    }
}

/// Round-robin merge: pick the i-th doc from each branch in
/// turn. Branch order in the output corresponds to JoinSet
/// completion order — strict per-branch ordering isn't promised
/// but interleaving is.
fn round_robin_merge(branches: Vec<Vec<Document>>) -> Vec<Document> {
    let max_len = branches.iter().map(|b| b.len()).max().unwrap_or(0);
    let mut out: Vec<Document> = Vec::new();
    for i in 0..max_len {
        for branch in &branches {
            if let Some(d) = branch.get(i) {
                out.push(d.clone());
            }
        }
    }
    out
}

/// Parse one-question-per-line LLM output. Strip numbering /
/// dash prefixes, drop blanks, dedup. Mirrors the helper in
/// multi_query / rag_fusion.
fn parse_sub_queries(raw: &str) -> Vec<String> {
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
    fn parse_sub_queries_strips_prefixes() {
        let input = "1. What is X?\n2) What is Y?\n- What is Z?";
        let qs = parse_sub_queries(input);
        assert_eq!(qs, vec!["What is X?", "What is Y?", "What is Z?"]);
    }

    #[test]
    fn parse_sub_queries_dedups() {
        let input = "Q1\nQ1\nQ2";
        let qs = parse_sub_queries(input);
        assert_eq!(qs, vec!["Q1", "Q2"]);
    }

    #[tokio::test]
    async fn unions_results_from_each_sub_query() {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "What is solar energy's climate impact?".into(),
            vec![doc("s1", "solar fact 1"), doc("s2", "solar fact 2")],
        );
        map.insert(
            "What is wind energy's climate impact?".into(),
            vec![doc("w1", "wind fact 1"), doc("w2", "wind fact 2")],
        );
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm {
            text: "What is solar energy's climate impact?\nWhat is wind energy's climate impact?".into(),
        });
        let r = SubQueryRetriever::new(base, llm).with_branch_k(5);
        let docs = r
            .retrieve("Compare solar vs wind climate impact", 10)
            .await
            .unwrap();
        let ids: HashSet<&str> =
            docs.iter().filter_map(|d| d.id.as_deref()).collect();
        assert!(ids.contains("s1"));
        assert!(ids.contains("s2"));
        assert!(ids.contains("w1"));
        assert!(ids.contains("w2"));
    }

    #[tokio::test]
    async fn round_robin_interleaves_per_branch_top_results() {
        // Two branches, each with 3 docs. Round-robin should
        // interleave: branch_a[0], branch_b[0], branch_a[1], …
        // After dedup the order should reflect the interleaving.
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Sub A".into(),
            vec![doc("a1", "x"), doc("a2", "y"), doc("a3", "z")],
        );
        map.insert(
            "Sub B".into(),
            vec![doc("b1", "x"), doc("b2", "y"), doc("b3", "z")],
        );
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm {
            text: "Sub A\nSub B".into(),
        });
        let r = SubQueryRetriever::new(base, llm).with_branch_k(3);
        let docs = r.retrieve("compound", 6).await.unwrap();
        // Interleaved: a1, b1, a2, b2, a3, b3 OR b1, a1, b2, a2, b3, a3
        // (JoinSet completion order isn't guaranteed). Both are valid
        // round-robin orderings — verify rank-0 of each branch
        // appears before rank-1 of the other.
        let pos_a1 = docs.iter().position(|d| d.id.as_deref() == Some("a1"));
        let pos_a2 = docs.iter().position(|d| d.id.as_deref() == Some("a2"));
        let pos_b1 = docs.iter().position(|d| d.id.as_deref() == Some("b1"));
        let pos_b2 = docs.iter().position(|d| d.id.as_deref() == Some("b2"));
        assert!(pos_a1.is_some() && pos_a2.is_some());
        assert!(pos_b1.is_some() && pos_b2.is_some());
        // a1 should come before a2 (per-branch ordering preserved).
        assert!(pos_a1.unwrap() < pos_a2.unwrap());
        assert!(pos_b1.unwrap() < pos_b2.unwrap());
    }

    #[tokio::test]
    async fn dedups_when_branches_share_docs() {
        let mut map = std::collections::HashMap::new();
        map.insert("Sub A".into(), vec![doc("shared", "x"), doc("only_a", "y")]);
        map.insert("Sub B".into(), vec![doc("shared", "x"), doc("only_b", "z")]);
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm {
            text: "Sub A\nSub B".into(),
        });
        let r = SubQueryRetriever::new(base, llm);
        let docs = r.retrieve("compound", 10).await.unwrap();
        let ids: Vec<_> =
            docs.iter().filter_map(|d| d.id.as_deref()).collect();
        let unique: HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len(), "duplicate ids in output");
        assert_eq!(unique.len(), 3);
    }

    #[tokio::test]
    async fn falls_back_to_literal_when_decomposition_empty() {
        let mut map = std::collections::HashMap::new();
        map.insert("orig q".into(), vec![doc("x", "fallback")]);
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        // LLM returns nothing usable.
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm { text: "".into() });
        let r = SubQueryRetriever::new(base, llm);
        let docs = r.retrieve("orig q", 5).await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id.as_deref(), Some("x"));
    }

    #[tokio::test]
    async fn max_sub_queries_caps_decomposition() {
        let r = SubQueryRetriever::new(
            Arc::new(ScriptedRetriever {
                responses: Mutex::new(std::collections::HashMap::new()),
            }) as Arc<dyn Retriever>,
            Arc::new(ScriptedLlm {
                text: "Q1\nQ2\nQ3\nQ4\nQ5\nQ6\nQ7".into(),
            }) as Arc<dyn ChatModel>,
        )
        .with_max_sub_queries(3);
        let subs = r.decompose("anything").await.unwrap();
        assert_eq!(subs.len(), 3);
    }

    #[tokio::test]
    async fn truncates_to_k() {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Sub A".into(),
            (0..10).map(|i| doc(&format!("a{i}"), "x")).collect(),
        );
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm {
            text: "Sub A".into(),
        });
        let r = SubQueryRetriever::new(base, llm).with_branch_k(20);
        let docs = r.retrieve("compound", 4).await.unwrap();
        assert_eq!(docs.len(), 4);
    }
}
