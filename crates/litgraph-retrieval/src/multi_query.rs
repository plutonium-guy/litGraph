//! MultiQueryRetriever — LLM-driven query expansion for recall improvement.
//!
//! # The pattern
//!
//! Single-query retrieval misses documents that use different vocabulary
//! than the user's question. MultiQueryRetriever asks the LLM to generate N
//! paraphrases of the query, runs the base retriever for each, deduplicates
//! the union by content hash, and returns the top-k.
//!
//! Direct LangChain `MultiQueryRetriever` parity. The base can be ANY
//! `Retriever` impl — VectorRetriever, BM25, Hybrid, ParentDocumentRetriever,
//! etc — so this composes cleanly with the rest of the retriever stack.
//!
//! # Cost
//!
//! One extra LLM call per `retrieve()` (cheap on flash/mini models) +
//! N×base_cost retrieval calls in parallel. For embedding retrieval where
//! N=4 is typical, that's 4 embedding calls in parallel — fast enough that
//! end-to-end latency stays within p95 SLO for most apps.
//!
//! # Determinism for tests
//!
//! The LLM call goes through `ChatModel::invoke()`; pin temperature=0.0 in
//! `ChatOptions` for deterministic paraphrases (test fixtures will drift
//! otherwise).

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{ChatModel, ChatOptions, Document, Error, Message, Result};
use tokio::task::JoinSet;

use crate::retriever::Retriever;

const DEFAULT_PROMPT: &str = "You are an assistant that rewrites a search query into N alternative phrasings. \
Output ONE phrasing per line, no numbering, no commentary, no blank lines. \
The phrasings should preserve the user's intent but use different words and sentence structure to \
maximize the chance of matching documents that use varied vocabulary.";

const DEFAULT_NUM_QUERIES: usize = 4;

pub struct MultiQueryRetriever {
    pub base: Arc<dyn Retriever>,
    pub llm: Arc<dyn ChatModel>,
    /// How many query variations to generate (in addition to the original).
    /// The original query is ALWAYS included so the user's literal phrasing
    /// still gets a shot — paraphrases supplement, not replace.
    pub num_variations: usize,
    pub system_prompt: String,
    pub include_original: bool,
}

impl MultiQueryRetriever {
    pub fn new(base: Arc<dyn Retriever>, llm: Arc<dyn ChatModel>) -> Self {
        Self {
            base,
            llm,
            num_variations: DEFAULT_NUM_QUERIES,
            system_prompt: DEFAULT_PROMPT.into(),
            include_original: true,
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

    /// Generate paraphrases via the LLM. Returns the cleaned list (no blank
    /// lines, no numbering prefixes like "1." or "- "). Public so callers can
    /// preview / cache outside the retrieval path.
    pub async fn generate_queries(&self, query: &str) -> Result<Vec<String>> {
        let user_msg = format!(
            "Rewrite the following query into {} alternative phrasings.\n\nQuery: {}",
            self.num_variations, query
        );
        let messages = vec![
            Message::system(self.system_prompt.clone()),
            Message::user(user_msg),
        ];
        // Pin temperature=0 here — paraphrases need to be deterministic-ish
        // for cache hits + test reproducibility.
        let opts = ChatOptions {
            temperature: Some(0.0),
            max_tokens: Some(512),
            ..Default::default()
        };
        let resp = self
            .llm
            .invoke(messages, &opts)
            .await
            .map_err(|e| Error::other(format!("multi_query llm: {e}")))?;
        let text = resp.message.text_content();
        Ok(parse_queries(&text))
    }
}

/// Strip numbering prefixes ("1. ", "1) ", "- ", "* "), drop blanks, dedup.
fn parse_queries(raw: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for line in raw.lines() {
        let mut s = line.trim().to_string();
        if s.is_empty() {
            continue;
        }
        // Strip leading list markers.
        for prefix in ["- ", "* ", "• "] {
            if let Some(rest) = s.strip_prefix(prefix) {
                s = rest.trim().to_string();
                break;
            }
        }
        // Numeric prefix "1. " / "12) ".
        if let Some(idx) = s.chars().take_while(|c| c.is_ascii_digit()).count().checked_sub(0) {
            if idx > 0 && idx < s.len() {
                let rest = &s[idx..];
                if let Some(stripped) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
                    s = stripped.trim().to_string();
                }
            }
        }
        if !s.is_empty() && seen.insert(s.clone()) {
            out.push(s);
        }
    }
    out
}

#[async_trait]
impl Retriever for MultiQueryRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        // Generate paraphrases.
        let mut queries = self.generate_queries(query).await.unwrap_or_default();
        if self.include_original {
            // Original goes first so it ranks first when ties happen.
            let original = query.to_string();
            queries.retain(|q| q != &original);
            queries.insert(0, original);
        }
        if queries.is_empty() {
            // LLM returned nothing usable — fall back to literal query so
            // we degrade gracefully instead of returning [].
            queries.push(query.to_string());
        }

        // Fan out base.retrieve(q, k) for each variation in parallel.
        let mut set: JoinSet<Result<Vec<Document>>> = JoinSet::new();
        for q in queries {
            let base = self.base.clone();
            set.spawn(async move { base.retrieve(&q, k).await });
        }

        // Collect + deduplicate by id (fallback content hash). Order:
        // first occurrence wins (the first paraphrase to surface a doc
        // ranks it higher than later paraphrases that re-find it).
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<Document> = Vec::new();
        let mut errors = Vec::new();
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(docs)) => {
                    for d in docs {
                        let key = d.id.clone().unwrap_or_else(|| {
                            // Cheap content hash: blake3 of the content bytes.
                            blake3::hash(d.content.as_bytes()).to_hex().to_string()
                        });
                        if seen.insert(key) {
                            out.push(d);
                        }
                    }
                }
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(je) => errors.push(format!("join: {je}")),
            }
        }
        // If EVERY branch failed AND we got nothing, surface the first error
        // — better than silently returning [].
        if out.is_empty() && !errors.is_empty() {
            return Err(Error::other(format!(
                "multi_query: all {} branches failed (first: {})",
                errors.len(),
                errors[0]
            )));
        }
        out.truncate(k);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use litgraph_core::model::{ChatStream, FinishReason};
    use litgraph_core::{ChatResponse, ContentPart, Role, TokenUsage};
    use std::sync::Mutex;

    /// Scripted LLM: the next invoke returns the canned text.
    struct ScriptedLlm {
        responses: Mutex<Vec<String>>,
        seen_user_messages: Mutex<Vec<String>>,
    }
    impl ScriptedLlm {
        fn new(rs: Vec<&str>) -> Self {
            Self {
                responses: Mutex::new(rs.into_iter().map(String::from).collect()),
                seen_user_messages: Mutex::new(Vec::new()),
            }
        }
    }
    #[async_trait]
    impl ChatModel for ScriptedLlm {
        fn name(&self) -> &str { "scripted" }
        async fn invoke(&self, messages: Vec<Message>, _opts: &ChatOptions) -> Result<ChatResponse> {
            for m in &messages {
                if matches!(m.role, Role::User) {
                    self.seen_user_messages.lock().unwrap().push(m.text_content());
                }
            }
            let mut r = self.responses.lock().unwrap();
            let text = r.remove(0);
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
                model: "scripted".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    /// Retriever that returns canned docs for specific queries; used to
    /// verify all paraphrases get fanned out + that results dedupe correctly.
    struct ByQueryRetriever {
        responses: std::collections::HashMap<String, Vec<Document>>,
        call_count: std::sync::atomic::AtomicUsize,
        seen: Mutex<Vec<String>>,
    }
    impl ByQueryRetriever {
        fn new(map: std::collections::HashMap<String, Vec<Document>>) -> Self {
            Self {
                responses: map,
                call_count: std::sync::atomic::AtomicUsize::new(0),
                seen: Mutex::new(Vec::new()),
            }
        }
    }
    #[async_trait]
    impl Retriever for ByQueryRetriever {
        async fn retrieve(&self, query: &str, _k: usize) -> Result<Vec<Document>> {
            self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.seen.lock().unwrap().push(query.into());
            Ok(self.responses.get(query).cloned().unwrap_or_default())
        }
    }

    fn doc_with_id(id: &str, content: &str) -> Document {
        let mut d = Document::new(content);
        d.id = Some(id.into());
        d
    }

    #[test]
    fn parse_queries_strips_numbered_and_bulleted_prefixes() {
        let raw = "1. foo bar\n- baz\n* qux\n2) hello\nworld\n\n3. foo bar\n";
        let out = parse_queries(raw);
        // foo bar dedups; the rest survive.
        assert_eq!(out, vec!["foo bar", "baz", "qux", "hello", "world"]);
    }

    #[test]
    fn parse_queries_drops_blank_lines() {
        let out = parse_queries("\n\n  \n  hello\n  world\n");
        assert_eq!(out, vec!["hello", "world"]);
    }

    #[tokio::test]
    async fn generate_queries_uses_llm_and_returns_cleaned_lines() {
        let llm = Arc::new(ScriptedLlm::new(vec![
            "1. how does borrow checking work\n2. lifetime semantics in rust\n3. rust ownership"
        ]));
        let base = Arc::new(ByQueryRetriever::new(Default::default()));
        let mqr = MultiQueryRetriever::new(base, llm.clone());
        let queries = mqr.generate_queries("borrow checker").await.unwrap();
        assert_eq!(queries.len(), 3);
        assert_eq!(queries[0], "how does borrow checking work");
        // The user message includes the original query.
        let seen = llm.seen_user_messages.lock().unwrap();
        assert!(seen[0].contains("borrow checker"));
    }

    #[tokio::test]
    async fn retrieve_fans_out_to_all_paraphrases_and_dedupes_by_id() {
        // LLM gives 2 paraphrases; with `include_original=true` (default),
        // we run 3 base.retrieve calls. Same doc surfaces from multiple
        // queries — result must dedupe by id.
        let llm = Arc::new(ScriptedLlm::new(vec!["alt phrasing one\nalt phrasing two"]));

        let mut map = std::collections::HashMap::new();
        map.insert("query".into(), vec![
            doc_with_id("d1", "doc one"),
            doc_with_id("d2", "doc two"),
        ]);
        map.insert("alt phrasing one".into(), vec![
            doc_with_id("d2", "doc two"),  // dup of above
            doc_with_id("d3", "doc three"),
        ]);
        map.insert("alt phrasing two".into(), vec![
            doc_with_id("d4", "doc four"),
        ]);
        let base = Arc::new(ByQueryRetriever::new(map));

        let mqr = MultiQueryRetriever::new(base.clone(), llm).with_num_variations(2);
        let docs = mqr.retrieve("query", 10).await.unwrap();

        // 3 base.retrieve calls (1 original + 2 paraphrases).
        assert_eq!(base.call_count.load(std::sync::atomic::Ordering::SeqCst), 3);
        // Dedup → 4 unique docs (d1, d2, d3, d4). d2 only appears once.
        assert_eq!(docs.len(), 4);
        let ids: Vec<&str> = docs.iter().filter_map(|d| d.id.as_deref()).collect();
        let unique: HashSet<&&str> = ids.iter().collect();
        assert_eq!(unique.len(), 4);
    }

    #[tokio::test]
    async fn retrieve_truncates_to_k() {
        let llm = Arc::new(ScriptedLlm::new(vec!["alt"]));
        let mut map = std::collections::HashMap::new();
        map.insert("query".into(), (0..5).map(|i| doc_with_id(&format!("o{i}"), "")).collect());
        map.insert("alt".into(), (0..5).map(|i| doc_with_id(&format!("a{i}"), "")).collect());
        let base = Arc::new(ByQueryRetriever::new(map));
        let mqr = MultiQueryRetriever::new(base, llm).with_num_variations(1);
        let docs = mqr.retrieve("query", 3).await.unwrap();
        assert_eq!(docs.len(), 3);
    }

    #[tokio::test]
    async fn retrieve_falls_back_to_literal_query_when_llm_returns_nothing() {
        let llm = Arc::new(ScriptedLlm::new(vec![""]));
        let mut map = std::collections::HashMap::new();
        map.insert("query".into(), vec![doc_with_id("d1", "literal hit")]);
        let base = Arc::new(ByQueryRetriever::new(map));
        let mqr = MultiQueryRetriever::new(base, llm).with_include_original(false);
        let docs = mqr.retrieve("query", 5).await.unwrap();
        // Literal query still ran even though LLM gave nothing.
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "literal hit");
    }

    #[tokio::test]
    async fn retrieve_dedupes_by_content_hash_when_id_missing() {
        let llm = Arc::new(ScriptedLlm::new(vec!["alt"]));
        // No ids on either result — must dedupe by content hash.
        let mut map = std::collections::HashMap::new();
        map.insert("query".into(), vec![Document::new("same content")]);
        map.insert("alt".into(), vec![Document::new("same content")]);
        let base = Arc::new(ByQueryRetriever::new(map));
        let mqr = MultiQueryRetriever::new(base, llm).with_num_variations(1);
        let docs = mqr.retrieve("query", 5).await.unwrap();
        assert_eq!(docs.len(), 1, "content-hash dedup should collapse to 1");
    }

    #[tokio::test]
    async fn retrieve_surfaces_error_when_all_branches_fail() {
        struct AlwaysErr;
        #[async_trait]
        impl Retriever for AlwaysErr {
            async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
                Err(Error::other("boom"))
            }
        }
        let llm = Arc::new(ScriptedLlm::new(vec!["alt"]));
        let base = Arc::new(AlwaysErr);
        let mqr = MultiQueryRetriever::new(base, llm).with_num_variations(1);
        let err = mqr.retrieve("query", 5).await.unwrap_err();
        assert!(format!("{err}").contains("boom"), "got: {err}");
    }
}
