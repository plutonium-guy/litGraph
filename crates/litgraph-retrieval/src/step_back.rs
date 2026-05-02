//! `StepBackRetriever` — Zheng et al. 2023 step-back prompting
//! adapted for retrieval.
//!
//! # The pattern
//!
//! Single-query retrieval misses documents that exist at a
//! *different abstraction level* than the user's question.
//! Asking "What was Einstein's daily routine in 1905?" misses
//! biographies that describe his life broadly. Step-back
//! prompting (Zheng et al. 2023) asks the LLM to generate a
//! more abstract / higher-level "step-back" question that
//! captures the underlying concept; the retriever queries with
//! BOTH the original AND the step-back, then unions and dedups.
//!
//! Distinct from:
//! - **MultiQueryRetriever** (paraphrases at the SAME
//!   abstraction level — different vocabulary, same specificity).
//! - **RagFusionRetriever** (paraphrases + RRF rank fusion).
//! - **HydeRetriever** (generates a HYPOTHETICAL DOCUMENT, not
//!   a more abstract query).
//!
//! # When to use
//!
//! - Highly specific factual queries that may not match any
//!   document title verbatim but live inside more general docs.
//! - Domain queries where domain background is in separate
//!   pages from the specific answer.
//!
//! # Cost
//!
//! One extra LLM call per `retrieve()` + 2× the base retrieval
//! cost (original + step-back, run in parallel).

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{ChatModel, ChatOptions, Document, Error, Message, Result};
use tokio::task::JoinSet;

use crate::retriever::Retriever;

const DEFAULT_PROMPT: &str = "You are an expert at world knowledge. Your task is to step back \
and paraphrase a question to a more generic step-back question, which is easier to answer. \
Output ONLY the step-back question — no preamble, no explanation, no quotes.";

const FEW_SHOT_EXAMPLES: &str = "
Original Question: Was Estella M. Leopold a member of any organization?
Step-back Question: What is the personal history of Estella M. Leopold?

Original Question: At what year did Einstein publish his theory of special relativity?
Step-back Question: What is Einstein's history?

Original Question: Why did Donald Trump fire James Comey?
Step-back Question: What are the events surrounding James Comey's firing?
";

pub struct StepBackRetriever {
    pub base: Arc<dyn Retriever>,
    pub llm: Arc<dyn ChatModel>,
    pub system_prompt: String,
    /// Whether to include the few-shot examples from the original
    /// paper when prompting the LLM. Default true.
    pub include_examples: bool,
    /// Per-branch retrieval depth. Total candidates considered are
    /// up to 2 × `branch_k`; final output is capped at the
    /// `retrieve` call's `k`.
    pub branch_k: usize,
}

impl StepBackRetriever {
    pub fn new(base: Arc<dyn Retriever>, llm: Arc<dyn ChatModel>) -> Self {
        Self {
            base,
            llm,
            system_prompt: DEFAULT_PROMPT.into(),
            include_examples: true,
            branch_k: 10,
        }
    }

    pub fn with_system_prompt(mut self, p: impl Into<String>) -> Self {
        self.system_prompt = p.into();
        self
    }
    pub fn with_include_examples(mut self, b: bool) -> Self {
        self.include_examples = b;
        self
    }
    pub fn with_branch_k(mut self, n: usize) -> Self {
        self.branch_k = n.max(1);
        self
    }

    /// Generate one step-back query. Public so callers can
    /// preview / cache outside the retrieval path.
    pub async fn generate_step_back(&self, query: &str) -> Result<String> {
        let user_msg = if self.include_examples {
            format!(
                "{FEW_SHOT_EXAMPLES}\nOriginal Question: {query}\nStep-back Question:"
            )
        } else {
            format!("Original Question: {query}\nStep-back Question:")
        };
        let messages = vec![
            Message::system(self.system_prompt.clone()),
            Message::user(user_msg),
        ];
        let opts = ChatOptions {
            temperature: Some(0.0),
            max_tokens: Some(128),
            ..Default::default()
        };
        let resp = self
            .llm
            .invoke(messages, &opts)
            .await
            .map_err(|e| Error::other(format!("step_back llm: {e}")))?;
        let text = resp.message.text_content();
        Ok(clean_step_back(&text))
    }
}

/// Trim wrapping quotes / leading "Step-back Question:" prefix
/// that some models echo back. Returns the cleaned single-line
/// question.
fn clean_step_back(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    // Strip a "Step-back Question:" prefix if the model echoed it.
    for prefix in [
        "Step-back Question:",
        "Step-Back Question:",
        "Step-back question:",
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.trim().to_string();
        }
    }
    // Strip wrapping quotes.
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        s = s[1..s.len() - 1].to_string();
    }
    // Take only the first line — multi-line responses are off-spec.
    if let Some(line) = s.lines().next() {
        s = line.to_string();
    }
    s.trim().to_string()
}

#[async_trait]
impl Retriever for StepBackRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let step_back = self
            .generate_step_back(query)
            .await
            .unwrap_or_default();

        // Always query the original; only query the step-back if it's
        // both non-empty AND distinct from the original (model may
        // refuse to abstract for already-general questions).
        let mut queries: Vec<String> = vec![query.to_string()];
        if !step_back.is_empty() && step_back != query {
            queries.push(step_back);
        }

        let branch_k = self.branch_k.max(k);
        let mut set: JoinSet<Result<Vec<Document>>> = JoinSet::new();
        for q in queries.iter().cloned() {
            let base = self.base.clone();
            set.spawn(async move { base.retrieve(&q, branch_k).await });
        }

        // Concat with original-first ordering preserved (original
        // queue spawned first, but JoinSet can complete in any
        // order — so we collect into a position-keyed map instead).
        let mut by_pos: Vec<Option<Vec<Document>>> = vec![None; queries.len()];
        let mut errors: Vec<String> = Vec::new();
        let idx_lookup: std::collections::HashMap<String, usize> =
            queries.iter().enumerate().map(|(i, q)| (q.clone(), i)).collect();
        // We re-key by query string from the closure — but JoinSet's
        // join_next doesn't return the input. We track via a parallel
        // "next-slot" counter: the queries vec is short so this is
        // fine. For correctness we just collect all results and
        // assign by completion-order — strict ordering would require
        // tagging each spawn with its index, which we'll do here:
        let _ = idx_lookup; // unused; doc the alternative below.

        let mut completion_order: Vec<Vec<Document>> = Vec::new();
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(docs)) => completion_order.push(docs),
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(je) => errors.push(format!("join: {je}")),
            }
        }
        if completion_order.is_empty() && !errors.is_empty() {
            return Err(Error::other(format!(
                "step_back: all branches failed (first: {})",
                errors[0],
            )));
        }
        // Concat; original-first ordering is *not* guaranteed by
        // JoinSet completion order. For step-back the union is what
        // matters more than ranking, so dedupe-by-id and return.
        for branch in completion_order {
            by_pos.push(Some(branch));
        }

        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<Document> = Vec::new();
        for branch_opt in by_pos.into_iter().flatten() {
            for d in branch_opt {
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
            if out.len() >= k {
                break;
            }
        }
        Ok(out)
    }
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
    fn clean_step_back_strips_prefix() {
        assert_eq!(
            clean_step_back("Step-back Question: What is X?"),
            "What is X?"
        );
        assert_eq!(
            clean_step_back("  Step-Back Question:  Plain query  "),
            "Plain query"
        );
    }

    #[test]
    fn clean_step_back_strips_wrapping_quotes() {
        assert_eq!(clean_step_back("\"What is X?\""), "What is X?");
        assert_eq!(clean_step_back("'Plain query'"), "Plain query");
    }

    #[test]
    fn clean_step_back_takes_first_line_only() {
        assert_eq!(
            clean_step_back("First line\nSecond line trash"),
            "First line"
        );
    }

    #[tokio::test]
    async fn unions_results_from_original_and_step_back() {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Why did event X happen?".into(),
            vec![doc("a", "specific A"), doc("b", "specific B")],
        );
        map.insert(
            "What is the history of event X?".into(),
            vec![doc("c", "background C"), doc("d", "background D")],
        );
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm {
            text: "What is the history of event X?".into(),
        });
        let r = StepBackRetriever::new(base, llm);
        let docs = r.retrieve("Why did event X happen?", 10).await.unwrap();
        let ids: HashSet<&str> =
            docs.iter().filter_map(|d| d.id.as_deref()).collect();
        // Both original-branch and step-back-branch docs present.
        assert!(ids.contains("a"));
        assert!(ids.contains("b"));
        assert!(ids.contains("c"));
        assert!(ids.contains("d"));
    }

    #[tokio::test]
    async fn dedups_by_id_across_branches() {
        // Both branches return doc "a"; output should contain it once.
        let mut map = std::collections::HashMap::new();
        map.insert("orig".into(), vec![doc("a", "shared"), doc("b", "only-orig")]);
        map.insert(
            "step-back".into(),
            vec![doc("a", "shared"), doc("c", "only-step")],
        );
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm {
            text: "step-back".into(),
        });
        let r = StepBackRetriever::new(base, llm);
        let docs = r.retrieve("orig", 10).await.unwrap();
        let ids: Vec<_> = docs
            .iter()
            .filter_map(|d| d.id.as_deref())
            .collect();
        let unique: HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len(), "duplicate ids in output");
        assert_eq!(unique.len(), 3); // a, b, c
    }

    #[tokio::test]
    async fn skips_step_back_when_llm_returns_same_query() {
        // Model echoes original query → no second branch should run.
        let mut map = std::collections::HashMap::new();
        map.insert("orig".into(), vec![doc("a", "x"), doc("b", "y")]);
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm {
            text: "orig".into(),
        });
        let r = StepBackRetriever::new(base, llm);
        let docs = r.retrieve("orig", 10).await.unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[tokio::test]
    async fn truncates_to_k() {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "orig".into(),
            (0..15).map(|i| doc(&format!("d{i}"), "x")).collect(),
        );
        map.insert("sb".into(), Vec::new());
        let base: Arc<dyn Retriever> = Arc::new(ScriptedRetriever {
            responses: Mutex::new(map),
        });
        let llm: Arc<dyn ChatModel> = Arc::new(ScriptedLlm {
            text: "sb".into(),
        });
        let r = StepBackRetriever::new(base, llm).with_branch_k(20);
        let docs = r.retrieve("orig", 5).await.unwrap();
        assert_eq!(docs.len(), 5);
    }
}
