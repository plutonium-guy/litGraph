use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, ContentPart, Message, PiiScrubber, Result,
};

/// Chat-model wrapper that redacts PII before sending prompts to the
/// inner provider and (optionally) redacts PII from the response text.
///
/// # Why
///
/// - **GDPR / CCPA** — sending raw user emails / phones / SSNs to a
///   third-party LLM vendor without a DPA is an audit finding.
/// - **Prompt-injection hygiene** — AWS keys and JWTs inadvertently
///   pasted by users get scrubbed before the model sees them; reduces
///   exfiltration blast radius.
/// - **Observability safety** — if this wrapper sits between the user
///   and the provider, downstream logging / tracing sees redacted
///   prompts, not raw PII.
///
/// # Default behavior
///
/// - `scrub_inputs = true` — redact outgoing user + system messages.
///   Assistant / tool messages are NOT touched (they came from the
///   model or tool, and re-scrubbing would corrupt agent traces).
/// - `scrub_outputs = false` — leave LLM responses as-is. Real-world
///   LLMs rarely leak PII (they didn't see it); output-scrubbing is
///   off by default to avoid mangling code blocks that contain
///   email-like or IP-like strings. Opt in with `.with_output_scrubbing()`
///   for strict environments.
/// - `scrub_system = false` — system prompts are operator-written and
///   usually contain no real PII. Off by default; opt in if you inject
///   user data into system messages.
///
/// # Note on streaming
///
/// `stream()` currently scrubs inputs but does NOT scrub token deltas.
/// Token-by-token scrubbing would require a streaming PII parser that
/// handles span boundaries — out of scope here. Full-string output
/// scrubbing still works at the final `Done` event if the consumer
/// reassembles the response and passes it through `PiiScrubber::scrub`.
///
/// # Composition
///
/// Stack with other wrappers freely: `Retry(Budget(Scrub(inner)))` is
/// typical — scrub innermost so retries don't re-scrub (CPU waste) and
/// the budget + retry counts apply to the scrubbed payload.
pub struct PiiScrubbingChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub scrubber: Arc<PiiScrubber>,
    pub scrub_inputs: bool,
    pub scrub_system: bool,
    pub scrub_outputs: bool,
}

impl PiiScrubbingChatModel {
    /// Build with the default PiiScrubber (all iter-129 detectors).
    pub fn new(inner: Arc<dyn ChatModel>) -> Self {
        Self {
            inner,
            scrubber: Arc::new(PiiScrubber::new()),
            scrub_inputs: true,
            scrub_system: false,
            scrub_outputs: false,
        }
    }

    /// Build with a caller-provided scrubber (e.g. with custom patterns
    /// or `.without_luhn()` for test environments).
    pub fn with_scrubber(mut self, scrubber: Arc<PiiScrubber>) -> Self {
        self.scrubber = scrubber;
        self
    }

    pub fn with_output_scrubbing(mut self) -> Self {
        self.scrub_outputs = true;
        self
    }

    pub fn with_system_scrubbing(mut self) -> Self {
        self.scrub_system = true;
        self
    }

    pub fn scrub_inputs(mut self, on: bool) -> Self {
        self.scrub_inputs = on;
        self
    }

    /// Scrub a Message's text content IN PLACE. Non-text ContentParts
    /// (images, audio) are untouched — we only mask string PII.
    /// Returns the new Message.
    fn scrub_message(&self, m: &Message) -> Message {
        use litgraph_core::Role;
        // Skip roles where scrubbing is off or doesn't make sense.
        let should_scrub = match m.role {
            Role::User => self.scrub_inputs,
            Role::System => self.scrub_inputs && self.scrub_system,
            // Assistant / Tool messages preserve the model's / tool's output.
            Role::Assistant | Role::Tool => false,
        };
        if !should_scrub {
            return m.clone();
        }
        let new_parts: Vec<ContentPart> = m
            .content
            .iter()
            .map(|p| match p {
                ContentPart::Text { text } => {
                    let scrubbed = self.scrubber.scrub(text).scrubbed;
                    ContentPart::Text { text: scrubbed }
                }
                other => other.clone(),
            })
            .collect();
        Message {
            role: m.role,
            content: new_parts,
            tool_calls: m.tool_calls.clone(),
            tool_call_id: m.tool_call_id.clone(),
            name: m.name.clone(),
            cache: m.cache,
        }
    }

    fn scrub_all(&self, messages: Vec<Message>) -> Vec<Message> {
        if !self.scrub_inputs {
            return messages;
        }
        messages.iter().map(|m| self.scrub_message(m)).collect()
    }

    fn scrub_response_text(&self, mut resp: ChatResponse) -> ChatResponse {
        if !self.scrub_outputs {
            return resp;
        }
        let new_parts: Vec<ContentPart> = resp
            .message
            .content
            .into_iter()
            .map(|p| match p {
                ContentPart::Text { text } => {
                    let scrubbed = self.scrubber.scrub(&text).scrubbed;
                    ContentPart::Text { text: scrubbed }
                }
                other => other,
            })
            .collect();
        resp.message.content = new_parts;
        resp
    }
}

#[async_trait]
impl ChatModel for PiiScrubbingChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let scrubbed = self.scrub_all(messages);
        let resp = self.inner.invoke(scrubbed, opts).await?;
        Ok(self.scrub_response_text(resp))
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // Scrub inputs but pass stream through as-is (token-delta
        // scrubbing is out of scope — see the module doc).
        let scrubbed = self.scrub_all(messages);
        self.inner.stream(scrubbed, opts).await
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    #[allow(unused_imports)]
    use litgraph_core::tool::Tool as _;
    #[allow(unused_imports)]
    use litgraph_core::{ContentPart, Message, Role};
    #[allow(unused_imports)]
    use std::sync::atomic::{AtomicU32, Ordering};
    #[allow(unused_imports)]
    use litgraph_core::Error;
    #[allow(unused_imports)]
    use std::time::Duration;

    // ---- PiiScrubbingChatModel tests -----------------------------------

    /// Chat model that captures the messages it was called with, returning
    /// a canned response.
    struct CapturingChatPii {
        seen: std::sync::Mutex<Vec<Vec<Message>>>,
        canned_response: String,
    }

    impl CapturingChatPii {
        fn new(canned: &str) -> Arc<Self> {
            Arc::new(Self {
                seen: std::sync::Mutex::new(Vec::new()),
                canned_response: canned.to_string(),
            })
        }
    }

    #[async_trait]
    impl ChatModel for CapturingChatPii {
        fn name(&self) -> &str {
            "capturing-pii"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen.lock().unwrap().push(messages);
            Ok(ChatResponse {
                message: Message::assistant(self.canned_response.clone()),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "capturing-pii".into(),
            })
        }
        async fn stream(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn pii_scrub_redacts_user_message_before_invoke() {
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>);
        let msgs = vec![
            Message::user("email me at alice@example.com for details"),
        ];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        let user_text = seen[0].text_content();
        assert!(user_text.contains("<EMAIL>"));
        assert!(!user_text.contains("alice@example.com"));
    }

    #[tokio::test]
    async fn pii_scrub_leaves_assistant_messages_untouched() {
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>);
        // Assistant messages in the history (prior turn) should NOT be
        // re-scrubbed — they came from the model, agent-trace integrity.
        let msgs = vec![
            Message::user("tell me about bob@example.com"),
            Message::assistant("Here's what I know about bob@example.com — he ..."),
            Message::user("continue"),
        ];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        // User messages scrubbed.
        assert!(seen[0].text_content().contains("<EMAIL>"));
        assert!(!seen[0].text_content().contains("bob@example.com"));
        // Assistant message UNTOUCHED.
        assert!(seen[1].text_content().contains("bob@example.com"));
        // Last user message scrubbed (no PII, passes through).
        assert_eq!(seen[2].text_content(), "continue");
    }

    #[tokio::test]
    async fn pii_scrub_system_off_by_default() {
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>);
        let msgs = vec![
            Message::system("operator@corp.com is the admin"),
            Message::user("hi"),
        ];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        // System message unchanged — operator prompts are trusted.
        assert!(seen[0].text_content().contains("operator@corp.com"));
    }

    #[tokio::test]
    async fn pii_scrub_system_opt_in_scrubs_operator_prompt_too() {
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>)
            .with_system_scrubbing();
        let msgs = vec![
            Message::system("admin is operator@corp.com"),
            Message::user("hi"),
        ];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        assert!(seen[0].text_content().contains("<EMAIL>"));
    }

    #[tokio::test]
    async fn pii_scrub_output_off_by_default() {
        // Model returns response containing what looks like PII. Default
        // behavior: don't mangle the LLM's output.
        let inner = CapturingChatPii::new("Contact alice@example.com for support.");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>);
        let msgs = vec![Message::user("who to contact")];
        let resp = scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        assert!(resp.message.text_content().contains("alice@example.com"));
    }

    #[tokio::test]
    async fn pii_scrub_output_opt_in_scrubs_response_text() {
        let inner = CapturingChatPii::new("Contact alice@example.com for support.");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>)
            .with_output_scrubbing();
        let msgs = vec![Message::user("who to contact")];
        let resp = scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let text = resp.message.text_content();
        assert!(text.contains("<EMAIL>"));
        assert!(!text.contains("alice@example.com"));
    }

    #[tokio::test]
    async fn pii_scrub_inputs_false_passes_everything_through() {
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>)
            .scrub_inputs(false);
        let msgs = vec![Message::user("email alice@example.com now")];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        // Scrubbing off → email preserved.
        assert!(seen[0].text_content().contains("alice@example.com"));
    }

    #[tokio::test]
    async fn pii_scrub_with_custom_scrubber_respects_custom_patterns() {
        use regex::Regex;
        // Operator adds an internal INTERNAL_ID pattern on top of defaults.
        let custom = PiiScrubber::new().with_patterns(vec![(
            "INTERNAL_ID".to_string(),
            Regex::new(r"\bINT-\d{4}\b").unwrap(),
        )]);
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>)
            .with_scrubber(Arc::new(custom));
        let msgs = vec![Message::user("issue INT-1234 filed by alice@example.com")];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        let text = seen[0].text_content();
        assert!(text.contains("<INTERNAL_ID>"));
        assert!(text.contains("<EMAIL>"));
    }

    #[tokio::test]
    async fn pii_scrub_name_proxy_from_inner() {
        let inner = CapturingChatPii::new("x");
        let scrub = PiiScrubbingChatModel::new(inner as Arc<dyn ChatModel>);
        assert_eq!(scrub.name(), "capturing-pii");
    }

    #[tokio::test]
    async fn pii_scrub_preserves_tool_calls_and_metadata_fields() {
        use litgraph_core::tool::ToolCall;
        let inner = CapturingChatPii::new("x");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>);
        // User message with tool_calls attached (rare in user role, but
        // test that the shape is preserved on assistant messages too).
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentPart::Text {
                text: "bob@example.com assistant response".into(),
            }],
            tool_calls: vec![ToolCall {
                id: "c1".into(),
                name: "look_up".into(),
                arguments: serde_json::json!({"email": "bob@example.com"}),
            }],
            tool_call_id: Some("prior".into()),
            name: Some("asst".into()),
            cache: true,
        };
        scrub
            .invoke(vec![msg.clone()], &ChatOptions::default())
            .await
            .unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        let kept = &seen[0];
        // Assistant-role → not scrubbed, so email in text is preserved.
        assert!(kept.text_content().contains("bob@example.com"));
        // Tool-calls, tool_call_id, name, cache all round-tripped.
        assert_eq!(kept.tool_calls.len(), 1);
        assert_eq!(kept.tool_calls[0].id, "c1");
        assert_eq!(kept.tool_call_id.as_deref(), Some("prior"));
        assert_eq!(kept.name.as_deref(), Some("asst"));
        assert!(kept.cache);
    }

}
