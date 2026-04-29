//! StructuredChatModel — wraps any `ChatModel` with a JSON Schema and
//! returns parsed-JSON responses. Direct LangChain `with_structured_output`
//! parity.
//!
//! # Why a wrapper, not just `ChatOptions.response_format`?
//!
//! The provider-native `response_format` field already lets you ask for
//! valid JSON, but it doesn't:
//! - parse the response (callers do `serde_json::from_str` themselves);
//! - validate against a schema (the model can return `{}` and you find out
//!   in production);
//! - surface a clear error when the LLM returns malformed JSON
//!   (bare `serde_json` errors don't tell you which field is missing).
//!
//! `StructuredChatModel` does all three: injects `response_format`, parses
//! the response text, and returns a typed `serde_json::Value` (or errors).
//!
//! Implements `ChatModel` itself so it composes through the standard
//! provider polymorphism — pass to `ReactAgent`, `MultiQueryRetriever`,
//! etc. without changes.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::model::ChatStream;
use crate::{ChatModel, ChatOptions, ChatResponse, Error, Message, Result};

/// Wraps a `ChatModel` with a JSON Schema. `invoke()` injects
/// `response_format = {type: "json_schema", json_schema: {name, schema, strict: true}}`,
/// then parses the LLM's text response as JSON. `invoke_structured()`
/// returns the parsed `Value` directly for callers that don't want to
/// re-parse out of `ChatResponse.message.text_content()`.
pub struct StructuredChatModel {
    inner: Arc<dyn ChatModel>,
    schema: Value,
    schema_name: String,
    /// When true (default), set `strict: true` in the response_format —
    /// providers that support strict mode (OpenAI gpt-4o+) refuse to emit
    /// JSON that doesn't match the schema. False is the OpenAI-compat
    /// default for older models / OSS providers.
    strict: bool,
}

impl StructuredChatModel {
    pub fn new(
        inner: Arc<dyn ChatModel>,
        schema: Value,
        schema_name: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            schema,
            schema_name: schema_name.into(),
            strict: true,
        }
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Invoke the underlying model with response_format injected, parse
    /// the response as JSON, return the parsed `Value` directly. Errors
    /// when the response isn't valid JSON.
    pub async fn invoke_structured(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<Value> {
        let resp = self.invoke(messages, opts).await?;
        let text = resp.message.text_content();
        if text.is_empty() {
            return Err(Error::other(format!(
                "structured output: model returned empty content for schema `{}`",
                self.schema_name
            )));
        }
        serde_json::from_str::<Value>(&text).map_err(|e| {
            Error::other(format!(
                "structured output: failed to parse JSON for schema `{}`: {e}\n--- raw response ---\n{text}",
                self.schema_name
            ))
        })
    }

    fn injected_opts(&self, base: &ChatOptions) -> ChatOptions {
        let mut opts = base.clone();
        // OpenAI-style response_format. Providers that support a different
        // shape (Anthropic uses tool-call mode for structured output) can
        // override in their own ChatModel impls — but this shape is the
        // most widely-supported.
        opts.response_format = Some(json!({
            "type": "json_schema",
            "json_schema": {
                "name": self.schema_name,
                "schema": self.schema,
                "strict": self.strict,
            }
        }));
        opts
    }
}

#[async_trait]
impl ChatModel for StructuredChatModel {
    fn name(&self) -> &str { self.inner.name() }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let opts = self.injected_opts(opts);
        let resp = self.inner.invoke(messages, &opts).await?;
        // Validate text parses as JSON before returning — fail fast at the
        // wrapper, not in user code that assumed the wrapper already checked.
        let text = resp.message.text_content();
        if !text.is_empty() {
            serde_json::from_str::<Value>(&text).map_err(|e| {
                Error::other(format!(
                    "structured output: model returned non-JSON for schema `{}`: {e}\n--- raw ---\n{text}",
                    self.schema_name
                ))
            })?;
        }
        Ok(resp)
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        // For streaming, we can't validate JSON until Done — so just inject
        // the response_format and pass through. Caller is responsible for
        // collecting + parsing the final assembled text.
        let opts = self.injected_opts(opts);
        self.inner.stream(messages, &opts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FinishReason;
    use crate::{ContentPart, Role, TokenUsage};
    use std::sync::Mutex;

    /// Scripted LLM that captures the response_format opts + returns canned text.
    struct ScriptedJsonLlm {
        next_text: Mutex<String>,
        last_response_format: Mutex<Option<Value>>,
    }
    impl ScriptedJsonLlm {
        fn new(text: &str) -> Self {
            Self {
                next_text: Mutex::new(text.into()),
                last_response_format: Mutex::new(None),
            }
        }
    }
    #[async_trait]
    impl ChatModel for ScriptedJsonLlm {
        fn name(&self) -> &str { "scripted-json" }
        async fn invoke(&self, _m: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
            *self.last_response_format.lock().unwrap() = opts.response_format.clone();
            let text = self.next_text.lock().unwrap().clone();
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
                model: "scripted-json".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn person_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        })
    }

    #[tokio::test]
    async fn invoke_structured_returns_parsed_json_dict() {
        let inner = Arc::new(ScriptedJsonLlm::new(r#"{"name": "Ada", "age": 36}"#));
        let m = StructuredChatModel::new(inner, person_schema(), "Person");
        let v = m.invoke_structured(vec![Message::user("who?")], &ChatOptions::default()).await.unwrap();
        assert_eq!(v["name"], "Ada");
        assert_eq!(v["age"], 36);
    }

    #[tokio::test]
    async fn invoke_injects_response_format_with_json_schema() {
        let inner = Arc::new(ScriptedJsonLlm::new(r#"{"name":"a","age":1}"#));
        let inner2 = inner.clone();
        let m = StructuredChatModel::new(inner, person_schema(), "Person");
        let _ = m.invoke(vec![Message::user("x")], &ChatOptions::default()).await.unwrap();
        let rf = inner2.last_response_format.lock().unwrap().clone().unwrap();
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["name"], "Person");
        assert_eq!(rf["json_schema"]["strict"], true);
        assert_eq!(rf["json_schema"]["schema"]["type"], "object");
    }

    #[tokio::test]
    async fn malformed_json_response_errors_at_invoke_with_useful_message() {
        let inner = Arc::new(ScriptedJsonLlm::new("not valid json at all"));
        let m = StructuredChatModel::new(inner, person_schema(), "Person");
        let err = m.invoke(vec![Message::user("x")], &ChatOptions::default()).await.unwrap_err();
        let msg = format!("{err}");
        // Error includes schema name + raw response for debugging.
        assert!(msg.contains("Person"), "got: {msg}");
        assert!(msg.contains("not valid json"), "raw response should be in error, got: {msg}");
    }

    #[tokio::test]
    async fn empty_response_errors_at_invoke_structured() {
        let inner = Arc::new(ScriptedJsonLlm::new(""));
        let m = StructuredChatModel::new(inner, person_schema(), "Person");
        let err = m.invoke_structured(vec![Message::user("x")], &ChatOptions::default()).await.unwrap_err();
        assert!(format!("{err}").contains("empty"));
    }

    #[tokio::test]
    async fn with_strict_false_propagates_to_response_format() {
        let inner = Arc::new(ScriptedJsonLlm::new(r#"{"name":"a","age":1}"#));
        let inner2 = inner.clone();
        let m = StructuredChatModel::new(inner, person_schema(), "Person").with_strict(false);
        let _ = m.invoke(vec![Message::user("x")], &ChatOptions::default()).await.unwrap();
        let rf = inner2.last_response_format.lock().unwrap().clone().unwrap();
        assert_eq!(rf["json_schema"]["strict"], false);
    }

    #[tokio::test]
    async fn name_and_invoke_pass_through_to_inner() {
        let inner = Arc::new(ScriptedJsonLlm::new(r#"{"name":"a","age":1}"#));
        let m = StructuredChatModel::new(inner, person_schema(), "Person");
        // Wrapper exposes inner model name (so logs / observability still
        // identify the underlying provider).
        assert_eq!(m.name(), "scripted-json");
    }

    #[tokio::test]
    async fn caller_supplied_response_format_is_overridden_by_wrapper() {
        // User passes a custom response_format; the wrapper MUST overwrite
        // it (otherwise the schema isn't enforced and silent regression).
        let inner = Arc::new(ScriptedJsonLlm::new(r#"{"name":"a","age":1}"#));
        let inner2 = inner.clone();
        let m = StructuredChatModel::new(inner, person_schema(), "Person");
        let mut opts = ChatOptions::default();
        opts.response_format = Some(json!({"type": "text"}));
        let _ = m.invoke(vec![Message::user("x")], &opts).await.unwrap();
        let rf = inner2.last_response_format.lock().unwrap().clone().unwrap();
        // Wrapper's json_schema, NOT the caller's "text" override.
        assert_eq!(rf["type"], "json_schema");
    }
}
