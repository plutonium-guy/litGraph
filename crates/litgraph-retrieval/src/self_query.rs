//! SelfQueryRetriever — LLM extracts a metadata filter from the user's
//! natural-language query, then runs a filtered vector search. Direct
//! LangChain `SelfQueryRetriever` parity.
//!
//! # Why
//!
//! Naive vector retrieval embeds the entire query string and trusts cosine
//! similarity to find the right docs. That breaks when the user's intent is
//! "find posts about X **filtered by attribute Y**" — embeddings smear
//! across both the topic and the filter terms, and you end up with low
//! precision (lots of off-attribute hits) or low recall (the filter words
//! crowd out topic similarity).
//!
//! SelfQueryRetriever splits these:
//! 1. LLM call extracts `{query: str, filter: {attr: value, ...}}` from the
//!    raw user input, given a schema describing each indexed attribute.
//! 2. Vector store runs `similarity_search(embed(query), k, filter)`.
//!
//! # Example
//!
//! Given attributes `[year (int), language (string), stars (int)]` and a
//! query "rust crates with >1000 stars from 2024", the LLM might return
//! `{query: "rust crates", filter: {language: "rust", year: 2024, stars: 1000}}`.
//! The vector search then runs ONLY over docs matching the filter.
//!
//! # Filter capability per store
//!
//! Each VectorStore impl decides how to interpret the `Filter`. Most do
//! exact-match equality (Memory/HNSW/Qdrant/Weaviate/Chroma/Pgvector all
//! support this). For range / regex / set-membership filters, the user
//! needs to swap in an explicit metadata pre-filter pipeline (out of scope
//! for v1).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{
    ChatModel, ChatOptions, Document, Embeddings, Error, Message, Result, StructuredChatModel,
};
use serde_json::{Value, json};

use crate::retriever::Retriever;
use crate::store::{Filter, VectorStore};

/// Describes one filterable attribute on the indexed documents. Used to
/// build the LLM's extraction prompt + JSON schema.
#[derive(Debug, Clone)]
pub struct AttributeInfo {
    pub name: String,
    /// Natural-language description; goes verbatim into the LLM prompt.
    /// Be specific — the LLM uses this to decide when to populate the
    /// attribute in the filter.
    pub description: String,
    /// `"string" | "integer" | "number" | "boolean"`. Maps to JSON Schema
    /// types in the extraction output.
    pub field_type: String,
}

impl AttributeInfo {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        field_type: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            field_type: field_type.into(),
        }
    }
}

const DEFAULT_SYSTEM_PROMPT: &str = "You analyze a user's question against a set of attributes describing a corpus of \
documents. Output a JSON object with two keys: `query` (the rewritten search query — \
the topic words ONLY, with all attribute references stripped out) and `filter` (a \
flat object mapping attribute names to the literal values mentioned in the question, \
or an empty object if no attributes are referenced). Use only the attributes listed; \
do not invent new ones. Match values to the documented types.";

pub struct SelfQueryRetriever {
    pub embeddings: Arc<dyn Embeddings>,
    pub store: Arc<dyn VectorStore>,
    pub llm: Arc<dyn ChatModel>,
    /// One-line description of what the documents contain. Goes into the
    /// LLM prompt so it knows what kind of query to expect.
    pub document_contents: String,
    pub attributes: Vec<AttributeInfo>,
    pub system_prompt: String,
}

impl SelfQueryRetriever {
    pub fn new(
        embeddings: Arc<dyn Embeddings>,
        store: Arc<dyn VectorStore>,
        llm: Arc<dyn ChatModel>,
        document_contents: impl Into<String>,
        attributes: Vec<AttributeInfo>,
    ) -> Self {
        Self {
            embeddings,
            store,
            llm,
            document_contents: document_contents.into(),
            attributes,
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
        }
    }

    pub fn with_system_prompt(mut self, p: impl Into<String>) -> Self {
        self.system_prompt = p.into(); self
    }

    /// Build the JSON schema for the LLM extraction step.
    fn build_extraction_schema(&self) -> Value {
        let mut filter_props = serde_json::Map::new();
        for a in &self.attributes {
            filter_props.insert(
                a.name.clone(),
                json!({ "type": a.field_type, "description": a.description }),
            );
        }
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The semantic search query with all attribute references stripped out. \
                                    If the user's question is purely an attribute lookup, this can be empty.",
                },
                "filter": {
                    "type": "object",
                    "description": "Attribute filters extracted from the user's question. \
                                    Empty object if no attributes referenced.",
                    "properties": filter_props,
                    "additionalProperties": false,
                },
            },
            "required": ["query", "filter"],
            "additionalProperties": false,
        })
    }

    /// Build the user prompt: docs description + attributes table + the user's raw query.
    fn build_user_prompt(&self, raw_query: &str) -> String {
        let mut s = String::new();
        s.push_str(&format!("Document contents: {}\n\nAttributes:\n", self.document_contents));
        for a in &self.attributes {
            s.push_str(&format!("  - {} ({}): {}\n", a.name, a.field_type, a.description));
        }
        s.push_str(&format!("\nUser question: {raw_query}"));
        s
    }

    /// Run JUST the extraction step — useful for previewing what the LLM
    /// inferred without firing a search. Returns (query, filter) where
    /// query is the rewritten semantic query and filter is the metadata
    /// equality map.
    pub async fn extract_query_and_filter(
        &self,
        raw_query: &str,
    ) -> Result<(String, Filter)> {
        let schema = self.build_extraction_schema();
        let structured = StructuredChatModel::new(
            self.llm.clone(),
            schema,
            "SelfQueryExtraction",
        );
        let messages = vec![
            Message::system(self.system_prompt.clone()),
            Message::user(self.build_user_prompt(raw_query)),
        ];
        let opts = ChatOptions {
            temperature: Some(0.0),
            max_tokens: Some(512),
            ..Default::default()
        };
        let v = structured.invoke_structured(messages, &opts).await?;
        let query = v
            .get("query")
            .and_then(|q| q.as_str())
            .unwrap_or("")
            .to_string();
        let mut filter: Filter = HashMap::new();
        if let Some(obj) = v.get("filter").and_then(|f| f.as_object()) {
            for (k, v) in obj {
                // Skip nulls — LLM occasionally emits `null` for an
                // unreferenced attribute when strict mode allows it.
                if !v.is_null() {
                    filter.insert(k.clone(), v.clone());
                }
            }
        }
        Ok((query, filter))
    }
}

#[async_trait]
impl Retriever for SelfQueryRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let (extracted_query, filter) = self.extract_query_and_filter(query).await?;

        // If the LLM left query blank (pure attribute lookup), fall back
        // to the original raw input for the embedding — better than embedding
        // an empty string.
        let q_for_embed = if extracted_query.trim().is_empty() {
            query
        } else {
            &extracted_query
        };
        let q_emb = self.embeddings.embed_query(q_for_embed).await?;
        let filter_arg = if filter.is_empty() { None } else { Some(&filter) };
        self.store.similarity_search(&q_emb, k, filter_arg).await
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use litgraph_core::model::{ChatStream, FinishReason};
    use litgraph_core::{ChatResponse, ContentPart, Result as LgResult, Role, TokenUsage};
    use std::sync::Mutex;

    /// LLM that returns a canned JSON extraction. Captures the user prompt
    /// + injected response_format so tests can assert wire shape.
    struct ScriptedLlm {
        canned: Mutex<String>,
        last_user_prompt: Mutex<Option<String>>,
        last_response_format: Mutex<Option<Value>>,
    }
    impl ScriptedLlm {
        fn new(c: &str) -> Self {
            Self {
                canned: Mutex::new(c.into()),
                last_user_prompt: Mutex::new(None),
                last_response_format: Mutex::new(None),
            }
        }
    }
    #[async_trait]
    impl ChatModel for ScriptedLlm {
        fn name(&self) -> &str { "scripted" }
        async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> LgResult<ChatResponse> {
            for m in &messages {
                if matches!(m.role, Role::User) {
                    *self.last_user_prompt.lock().unwrap() = Some(m.text_content());
                }
            }
            *self.last_response_format.lock().unwrap() = opts.response_format.clone();
            let text = self.canned.lock().unwrap().clone();
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
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> LgResult<ChatStream> {
            unimplemented!()
        }
    }

    /// Embeddings: doc-len vector. Captures last query for assertion.
    struct LenEmb {
        last_query: Mutex<Option<String>>,
    }
    impl LenEmb {
        fn new() -> Self { Self { last_query: Mutex::new(None) } }
    }
    #[async_trait]
    impl Embeddings for LenEmb {
        fn name(&self) -> &str { "len" }
        fn dimensions(&self) -> usize { 4 }
        async fn embed_query(&self, q: &str) -> LgResult<Vec<f32>> {
            *self.last_query.lock().unwrap() = Some(q.into());
            Ok(vec![q.len() as f32, 1.0, 0.0, 0.0])
        }
        async fn embed_documents(&self, ts: &[String]) -> LgResult<Vec<Vec<f32>>> {
            Ok(ts.iter().map(|t| vec![t.len() as f32, 1.0, 0.0, 0.0]).collect())
        }
    }

    /// Store that captures the filter passed to similarity_search and
    /// returns canned docs.
    struct CapturingStore {
        docs: Vec<Document>,
        last_filter: Mutex<Option<Filter>>,
    }
    impl CapturingStore {
        fn new(docs: Vec<Document>) -> Self {
            Self { docs, last_filter: Mutex::new(None) }
        }
    }
    #[async_trait]
    impl VectorStore for CapturingStore {
        async fn add(&self, _: Vec<Document>, _: Vec<Vec<f32>>) -> LgResult<Vec<String>> {
            Ok(vec![])
        }
        async fn similarity_search(
            &self,
            _q: &[f32],
            k: usize,
            f: Option<&Filter>,
        ) -> LgResult<Vec<Document>> {
            *self.last_filter.lock().unwrap() = f.cloned();
            Ok(self.docs.clone().into_iter().take(k).collect())
        }
        async fn delete(&self, _ids: &[String]) -> LgResult<()> { Ok(()) }
        async fn len(&self) -> usize { self.docs.len() }
    }

    fn doc(content: &str) -> Document { Document::new(content) }

    fn standard_attributes() -> Vec<AttributeInfo> {
        vec![
            AttributeInfo::new("language", "Programming language", "string"),
            AttributeInfo::new("year", "Publication year", "integer"),
            AttributeInfo::new("stars", "GitHub star count", "integer"),
        ]
    }

    #[tokio::test]
    async fn extract_query_and_filter_returns_parsed_pair() {
        let llm = Arc::new(ScriptedLlm::new(
            r#"{"query": "memory safety crates", "filter": {"language": "rust", "year": 2024}}"#,
        ));
        let r = SelfQueryRetriever::new(
            Arc::new(LenEmb::new()),
            Arc::new(CapturingStore::new(vec![])),
            llm,
            "rust crate descriptions",
            standard_attributes(),
        );
        let (query, filter) = r
            .extract_query_and_filter("memory safety crates in rust from 2024")
            .await
            .unwrap();
        assert_eq!(query, "memory safety crates");
        assert_eq!(filter.get("language").unwrap(), &json!("rust"));
        assert_eq!(filter.get("year").unwrap(), &json!(2024));
        assert!(!filter.contains_key("stars"));
    }

    #[tokio::test]
    async fn retrieve_passes_extracted_filter_to_vector_store() {
        let llm = Arc::new(ScriptedLlm::new(
            r#"{"query": "memory safety", "filter": {"language": "rust"}}"#,
        ));
        let store = Arc::new(CapturingStore::new(vec![doc("a"), doc("b")]));
        let store_for_assert = store.clone();
        let r = SelfQueryRetriever::new(
            Arc::new(LenEmb::new()),
            store,
            llm,
            "rust crates",
            standard_attributes(),
        );
        let docs = r.retrieve("memory safety in rust", 5).await.unwrap();
        assert_eq!(docs.len(), 2);
        let filter = store_for_assert.last_filter.lock().unwrap().clone().unwrap();
        assert_eq!(filter.get("language").unwrap(), &json!("rust"));
    }

    #[tokio::test]
    async fn retrieve_uses_extracted_query_for_embedding_not_raw_input() {
        let llm = Arc::new(ScriptedLlm::new(
            r#"{"query": "memory safety", "filter": {"language": "rust"}}"#,
        ));
        let emb = Arc::new(LenEmb::new());
        let emb_for_assert = emb.clone();
        let r = SelfQueryRetriever::new(
            emb,
            Arc::new(CapturingStore::new(vec![])),
            llm,
            "rust crates",
            standard_attributes(),
        );
        let _ = r.retrieve("memory safety in rust from 2024", 5).await.unwrap();
        // Embedded the EXTRACTED query, not the raw input — the filter
        // terms should NOT pollute the embedding vector.
        let q = emb_for_assert.last_query.lock().unwrap().clone().unwrap();
        assert_eq!(q, "memory safety");
    }

    #[tokio::test]
    async fn empty_extracted_query_falls_back_to_raw_input_for_embedding() {
        // Pure attribute lookup ("show me everything in rust") → LLM might
        // return query = "" + filter = {language: rust}. Don't embed empty
        // string — fall back to raw input so we still get *some* signal.
        let llm = Arc::new(ScriptedLlm::new(
            r#"{"query": "", "filter": {"language": "rust"}}"#,
        ));
        let emb = Arc::new(LenEmb::new());
        let emb_for_assert = emb.clone();
        let r = SelfQueryRetriever::new(
            emb,
            Arc::new(CapturingStore::new(vec![])),
            llm,
            "rust crates",
            standard_attributes(),
        );
        let _ = r.retrieve("everything in rust", 5).await.unwrap();
        let q = emb_for_assert.last_query.lock().unwrap().clone().unwrap();
        assert_eq!(q, "everything in rust");
    }

    #[tokio::test]
    async fn empty_filter_means_no_filter_passed_to_store() {
        // LLM returns no filter (no attributes in question). Store should
        // see `None`, not `Some({})` — semantically distinct for some
        // store impls (Weaviate's empty `where` clause is actually invalid).
        let llm = Arc::new(ScriptedLlm::new(
            r#"{"query": "rust", "filter": {}}"#,
        ));
        let store = Arc::new(CapturingStore::new(vec![doc("a")]));
        let store_for_assert = store.clone();
        let r = SelfQueryRetriever::new(
            Arc::new(LenEmb::new()),
            store,
            llm,
            "rust crates",
            standard_attributes(),
        );
        let _ = r.retrieve("rust", 5).await.unwrap();
        assert!(store_for_assert.last_filter.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn filter_null_values_are_dropped() {
        // LLM emits null for unreferenced attribute. Drop it — the store
        // doesn't want a literal None in the filter map.
        let llm = Arc::new(ScriptedLlm::new(
            r#"{"query": "x", "filter": {"language": "rust", "year": null}}"#,
        ));
        let r = SelfQueryRetriever::new(
            Arc::new(LenEmb::new()),
            Arc::new(CapturingStore::new(vec![])),
            llm,
            "rust crates",
            standard_attributes(),
        );
        let (_, filter) = r.extract_query_and_filter("x").await.unwrap();
        assert!(filter.contains_key("language"));
        assert!(!filter.contains_key("year"), "null values should drop");
    }

    #[tokio::test]
    async fn extraction_prompt_includes_attribute_table_and_doc_description() {
        let llm = Arc::new(ScriptedLlm::new(r#"{"query":"x","filter":{}}"#));
        let llm_for_assert = llm.clone();
        let r = SelfQueryRetriever::new(
            Arc::new(LenEmb::new()),
            Arc::new(CapturingStore::new(vec![])),
            llm,
            "rust crate descriptions",
            standard_attributes(),
        );
        let _ = r.extract_query_and_filter("any").await.unwrap();
        let prompt = llm_for_assert.last_user_prompt.lock().unwrap().clone().unwrap();
        assert!(prompt.contains("rust crate descriptions"));
        assert!(prompt.contains("language"));
        assert!(prompt.contains("year"));
        assert!(prompt.contains("Programming language"));
        assert!(prompt.contains("any"));  // raw query echoed
    }

    #[tokio::test]
    async fn extraction_uses_structured_chat_with_correct_schema() {
        let llm = Arc::new(ScriptedLlm::new(r#"{"query":"x","filter":{}}"#));
        let llm_for_assert = llm.clone();
        let r = SelfQueryRetriever::new(
            Arc::new(LenEmb::new()),
            Arc::new(CapturingStore::new(vec![])),
            llm,
            "rust crates",
            standard_attributes(),
        );
        let _ = r.extract_query_and_filter("x").await.unwrap();
        let rf = llm_for_assert.last_response_format.lock().unwrap().clone().unwrap();
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["name"], "SelfQueryExtraction");
        // Schema declares `query` + `filter` + each attribute as filter property.
        let schema = &rf["json_schema"]["schema"];
        assert!(schema["properties"]["query"].is_object());
        assert!(schema["properties"]["filter"]["properties"]["language"].is_object());
        assert_eq!(
            schema["properties"]["filter"]["properties"]["year"]["type"],
            "integer"
        );
    }
}
