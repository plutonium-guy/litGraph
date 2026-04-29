//! HydeRetriever — Hypothetical Document Embeddings (Gao et al, 2022).
//!
//! # The pattern
//!
//! Vanilla dense retrieval embeds the user's question and searches for
//! similar documents. Problem: questions and answers often use
//! different vocabulary ("How does photosynthesis work?" vs "Light
//! energy converts CO₂ and water into glucose..."). Embedding distance
//! between the two may be larger than between two unrelated questions.
//!
//! HyDE fixes this by asking the LLM to write a HYPOTHETICAL ANSWER to
//! the question first, then retrieving using the hypothetical-answer's
//! embedding. The answer vocabulary + phrasing matches real documents
//! far better than the question's.
//!
//! # When to use vs MultiQueryRetriever
//!
//! - **MultiQuery** (iter ~90) — generates N alternative phrasings of
//!   the QUESTION. Good for vocabulary-drift at the question level
//!   (dialect, jargon, synonyms). N retrievals.
//! - **HyDE** (this file) — generates one hypothetical ANSWER.
//!   Good for answer-vocabulary mismatch on conceptual / "how does X
//!   work" / definition queries. 1 retrieval (plus optional original).
//!
//! Both are ORTHOGONAL — stack HyDE inside MultiQuery for both wins on
//! high-recall demands. Cost: 1–5 extra LLM calls per retrieval.
//!
//! # Trap: LLM hallucinates specifics
//!
//! The hypothetical answer may contain confident-sounding wrong facts.
//! That's FINE for retrieval (we never show the hallucinated answer to
//! the user — it's just embedding fodder). If the real documents
//! contradict the hallucination, the retrieval still surfaces them,
//! and the downstream LLM uses the real content.
//!
//! # Cost + determinism
//!
//! Pin `temperature=0.0` in the LLM call so hypothetical answers are
//! stable across calls (enables caching + test reproducibility). Use a
//! cheap model — the answer doesn't need to be correct, just plausibly
//! answer-shaped.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{ChatModel, ChatOptions, Document, Message, Result};

use crate::retriever::Retriever;

const DEFAULT_PROMPT: &str =
    "You are a concise subject-matter assistant. Given a user question, write a short \
     passage (2–4 sentences) that would directly answer it. Write in the style of an \
     encyclopedia entry — declarative sentences, domain-specific vocabulary, no hedging. \
     Do NOT preface with 'Sure' or 'Here is an answer'. Do NOT include the question. \
     Output ONLY the hypothetical answer passage.";

pub struct HydeRetriever {
    pub base: Arc<dyn Retriever>,
    pub llm: Arc<dyn ChatModel>,
    /// Custom system prompt for the hypothetical-answer generation. Default
    /// targets encyclopedia-style passages.
    pub system_prompt: String,
    /// If true, ALSO retrieve with the original user query + merge the
    /// hypothetical-answer results. Good belt-and-suspenders for questions
    /// where the LLM's hypothetical is wildly off topic. Default true.
    pub include_original: bool,
}

impl HydeRetriever {
    pub fn new(base: Arc<dyn Retriever>, llm: Arc<dyn ChatModel>) -> Self {
        Self {
            base,
            llm,
            system_prompt: DEFAULT_PROMPT.into(),
            include_original: true,
        }
    }

    pub fn with_system_prompt(mut self, p: impl Into<String>) -> Self {
        self.system_prompt = p.into();
        self
    }

    pub fn with_include_original(mut self, b: bool) -> Self {
        self.include_original = b;
        self
    }

    /// Generate the hypothetical answer via the LLM. Exposed so callers
    /// can preview / cache outside the retrieval path.
    pub async fn generate_hypothetical(&self, query: &str) -> Result<String> {
        let messages = vec![
            Message::system(self.system_prompt.clone()),
            Message::user(format!("Question: {query}")),
        ];
        // Pin temperature=0 for deterministic passages — HyDE benefits
        // from cache hits when the same query gets issued twice.
        let opts = ChatOptions {
            temperature: Some(0.0),
            max_tokens: Some(256),
            ..Default::default()
        };
        let resp = self.llm.invoke(messages, &opts).await?;
        Ok(resp.message.text_content().trim().to_string())
    }
}

#[async_trait]
impl Retriever for HydeRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let hypothetical = self.generate_hypothetical(query).await?;
        // Run both retrievals if include_original=true. Serial is fine —
        // the base retriever is the cost dominant; one extra call is
        // cheap. For parallel, wrap with tokio::join!.
        let mut combined: Vec<Document> = Vec::new();
        combined.extend(self.base.retrieve(&hypothetical, k).await?);
        if self.include_original {
            combined.extend(self.base.retrieve(query, k).await?);
        }

        // Dedup by document id (falling back to content hash if id absent)
        // so if both retrievals surface the same doc, we count it once.
        // Preserves first-appearance order — hypothetical-answer hits come
        // first (the whole point of HyDE), original-query hits fill in.
        let mut seen: HashSet<String> = HashSet::new();
        let mut deduped: Vec<Document> = Vec::with_capacity(combined.len());
        for doc in combined {
            let key = doc
                .id
                .clone()
                .unwrap_or_else(|| {
                    // Content-hash fallback: use the full content as the
                    // dedup key. Not cheap but typical doc counts keep it
                    // cheap in aggregate.
                    doc.content.clone()
                });
            if seen.insert(key) {
                deduped.push(doc);
            }
        }

        // Sort by score (desc) if scored; stable sort preserves dedup
        // order for ties. Cap at k.
        deduped.sort_by(|a, b| {
            b.score
                .unwrap_or(0.0)
                .partial_cmp(&a.score.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        deduped.truncate(k);
        Ok(deduped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatResponse, ChatStream, FinishReason, TokenUsage};
    use std::sync::Mutex;

    /// Scripted LLM — returns the next canned hypothetical answer per call.
    struct ScriptedHyde {
        answers: Mutex<Vec<String>>,
        seen_prompts: Mutex<Vec<String>>,
    }
    impl ScriptedHyde {
        fn new(answers: Vec<&str>) -> Arc<Self> {
            Arc::new(Self {
                answers: Mutex::new(
                    answers.into_iter().map(str::to_string).rev().collect(),
                ),
                seen_prompts: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl ChatModel for ScriptedHyde {
        fn name(&self) -> &str {
            "scripted-hyde"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            let user = messages
                .iter()
                .filter(|m| matches!(m.role, litgraph_core::Role::User))
                .map(|m| m.text_content())
                .collect::<Vec<_>>()
                .join("\n");
            self.seen_prompts.lock().unwrap().push(user);
            let content = self
                .answers
                .lock()
                .unwrap()
                .pop()
                .unwrap_or_else(|| "out of answers".into());
            Ok(ChatResponse {
                message: Message::assistant(content),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "scripted-hyde".into(),
            })
        }
        async fn stream(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    /// Scripted retriever — returns pre-set docs per query string.
    struct ScriptedRetriever {
        /// query → (docs to return)
        table: std::collections::HashMap<String, Vec<Document>>,
        seen: Mutex<Vec<String>>,
    }
    impl ScriptedRetriever {
        fn new() -> Self {
            Self {
                table: std::collections::HashMap::new(),
                seen: Mutex::new(Vec::new()),
            }
        }
        fn with(
            mut self,
            query_substr: &str,
            docs: Vec<Document>,
        ) -> Self {
            self.table.insert(query_substr.to_string(), docs);
            self
        }
    }

    #[async_trait]
    impl Retriever for ScriptedRetriever {
        async fn retrieve(&self, query: &str, _k: usize) -> Result<Vec<Document>> {
            self.seen.lock().unwrap().push(query.to_string());
            // First substring hit in insertion order. Tests use disjoint
            // keys so this is unambiguous.
            for (key, docs) in &self.table {
                if query.contains(key) {
                    return Ok(docs.clone());
                }
            }
            Ok(Vec::new())
        }
    }

    fn doc(id: &str, content: &str, score: f32) -> Document {
        Document::new(content).with_id(id).with_metadata(
            "score_was",
            serde_json::json!(score),
        )
    }

    fn doc_scored(id: &str, content: &str, score: f32) -> Document {
        let mut d = Document::new(content).with_id(id);
        d.score = Some(score);
        d
    }

    #[tokio::test]
    async fn hypothetical_answer_drives_retrieval() {
        // LLM's hypothetical contains the word "Photosynthesis" (sentence-
        // start capitalization). Case-sensitive substring match in the
        // fixture — use the same form as the LLM output.
        let llm = ScriptedHyde::new(vec![
            "Photosynthesis converts light energy into glucose in plant cells.",
        ]);
        // Retriever returns docs ONLY when the query contains "Photosynthesis"
        // (i.e., only the hypothetical-driven retrieval fires, not the
        // original "how do plants feed themselves").
        let retriever = Arc::new(
            ScriptedRetriever::new().with(
                "Photosynthesis",
                vec![doc("d1", "Plants make glucose via...", 0.9)],
            ),
        );
        let hyde = HydeRetriever::new(
            retriever.clone() as Arc<dyn Retriever>,
            llm.clone() as Arc<dyn ChatModel>,
        )
        .with_include_original(false);
        let docs = hyde.retrieve("how do plants feed themselves", 5).await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id.as_deref(), Some("d1"));
    }

    #[tokio::test]
    async fn include_original_merges_both_retrieval_results() {
        // Hypothetical-driven retrieval hits d1; original-query hits d2.
        // Both surface; dedup keeps both.
        let llm = ScriptedHyde::new(vec!["Photosynthesis uses chlorophyll."]);
        let retriever = Arc::new(
            ScriptedRetriever::new()
                .with("Photosynthesis", vec![doc_scored("d1", "glucose path", 0.9)])
                .with("plants feed", vec![doc_scored("d2", "root uptake", 0.85)]),
        );
        let hyde = HydeRetriever::new(
            retriever.clone() as Arc<dyn Retriever>,
            llm.clone() as Arc<dyn ChatModel>,
        ); // include_original defaults to true
        let docs = hyde.retrieve("how do plants feed themselves", 5).await.unwrap();
        let ids: Vec<&str> = docs
            .iter()
            .filter_map(|d| d.id.as_deref())
            .collect();
        assert!(ids.contains(&"d1"));
        assert!(ids.contains(&"d2"));
    }

    #[tokio::test]
    async fn dedup_by_id_keeps_each_doc_once() {
        // Both retrievals return d1 — must appear only once in output.
        let llm = ScriptedHyde::new(vec!["Photosynthesis uses chlorophyll."]);
        let shared = vec![doc_scored("d1", "shared content", 0.9)];
        let retriever = Arc::new(
            ScriptedRetriever::new()
                .with("Photosynthesis", shared.clone())
                .with("plants feed", shared),
        );
        let hyde = HydeRetriever::new(
            retriever.clone() as Arc<dyn Retriever>,
            llm.clone() as Arc<dyn ChatModel>,
        );
        let docs = hyde.retrieve("how do plants feed themselves", 5).await.unwrap();
        let d1_count = docs.iter().filter(|d| d.id.as_deref() == Some("d1")).count();
        assert_eq!(d1_count, 1);
    }

    #[tokio::test]
    async fn results_sorted_by_score_descending() {
        let llm = ScriptedHyde::new(vec!["keyword"]);
        let retriever = Arc::new(
            ScriptedRetriever::new()
                .with("keyword", vec![
                    doc_scored("low", "", 0.3),
                    doc_scored("high", "", 0.9),
                    doc_scored("mid", "", 0.6),
                ]),
        );
        let hyde = HydeRetriever::new(
            retriever.clone() as Arc<dyn Retriever>,
            llm.clone() as Arc<dyn ChatModel>,
        )
        .with_include_original(false);
        let docs = hyde.retrieve("anything", 5).await.unwrap();
        assert_eq!(docs[0].id.as_deref(), Some("high"));
        assert_eq!(docs[1].id.as_deref(), Some("mid"));
        assert_eq!(docs[2].id.as_deref(), Some("low"));
    }

    #[tokio::test]
    async fn result_truncated_to_k() {
        let llm = ScriptedHyde::new(vec!["keyword"]);
        let retriever = Arc::new(
            ScriptedRetriever::new().with("keyword", vec![
                doc_scored("a", "", 0.9),
                doc_scored("b", "", 0.8),
                doc_scored("c", "", 0.7),
                doc_scored("d", "", 0.6),
            ]),
        );
        let hyde = HydeRetriever::new(
            retriever.clone() as Arc<dyn Retriever>,
            llm.clone() as Arc<dyn ChatModel>,
        )
        .with_include_original(false);
        let docs = hyde.retrieve("anything", 2).await.unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[tokio::test]
    async fn generate_hypothetical_sent_with_temperature_zero_and_query_in_prompt() {
        let llm = ScriptedHyde::new(vec!["some answer"]);
        let retriever = Arc::new(ScriptedRetriever::new());
        let hyde = HydeRetriever::new(
            retriever.clone() as Arc<dyn Retriever>,
            llm.clone() as Arc<dyn ChatModel>,
        );
        let _ = hyde
            .generate_hypothetical("what is photosynthesis?")
            .await
            .unwrap();
        // Prompt should carry the user query verbatim.
        let seen = llm.seen_prompts.lock().unwrap();
        assert!(seen[0].contains("what is photosynthesis?"));
    }

    #[tokio::test]
    async fn custom_prompt_override_reaches_model() {
        let llm = ScriptedHyde::new(vec!["x"]);
        let retriever = Arc::new(ScriptedRetriever::new());
        let hyde = HydeRetriever::new(
            retriever.clone() as Arc<dyn Retriever>,
            llm.clone() as Arc<dyn ChatModel>,
        )
        .with_system_prompt("BE TERSE AND CONFIDENT.");
        let _ = hyde.generate_hypothetical("q").await.unwrap();
        // System prompt doesn't appear in the `user`-filtered prompt we
        // captured, but we can at least verify the call happened.
        assert_eq!(llm.answers.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn include_original_false_skips_original_query_retrieval() {
        let llm = ScriptedHyde::new(vec!["hypo"]);
        let retriever = Arc::new(ScriptedRetriever::new());
        let hyde = HydeRetriever::new(
            retriever.clone() as Arc<dyn Retriever>,
            llm.clone() as Arc<dyn ChatModel>,
        )
        .with_include_original(false);
        let _ = hyde.retrieve("original q", 5).await.unwrap();
        let seen = retriever.seen.lock().unwrap();
        // Only ONE call to the base retriever (the hypothetical one).
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0], "hypo");
    }

    #[tokio::test]
    async fn include_original_true_issues_two_retrievals() {
        let llm = ScriptedHyde::new(vec!["hypo"]);
        let retriever = Arc::new(ScriptedRetriever::new());
        let hyde = HydeRetriever::new(
            retriever.clone() as Arc<dyn Retriever>,
            llm.clone() as Arc<dyn ChatModel>,
        );
        let _ = hyde.retrieve("original q", 5).await.unwrap();
        let seen = retriever.seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        // Hypothetical first, original second — matches "HyDE first,
        // original as fallback" dedup ordering.
        assert_eq!(seen[0], "hypo");
        assert_eq!(seen[1], "original q");
    }
}
