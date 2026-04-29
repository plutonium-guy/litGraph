//! Agent middleware — Express-style hooks around `ChatModel::invoke`.
//!
//! LangChain 1.0 reframes the agent loop around middleware. We mirror that
//! shape but stay deliberately small: a single trait with default-noop hooks,
//! a `MiddlewareChain` that runs hooks in registration order on the way in
//! and reverse order on the way out (onion model), and a `MiddlewareChatModel`
//! adapter that wraps any `ChatModel` with a chain.
//!
//! Built-in middlewares:
//! * `LoggingMiddleware` — emits `tracing` events on every model call.
//! * `MessageWindowMiddleware` — keeps only the last N non-system messages.
//! * `SystemPromptMiddleware` — prepends a system message if absent.
//!
//! The trait surface is sync-only on purpose — middleware mutations should be
//! cheap. For async work (e.g. fetching a personalised system prompt from a
//! Store), build a custom `ChatModel` wrapper around the chain.

use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info};

use crate::error::Result;
use crate::message::{Message, Role};
use crate::model::{ChatModel, ChatOptions, ChatResponse, ChatStream};

/// Hook trait. All methods default to no-op so callers only override what they
/// care about. Mutations are in-place to keep allocations tight.
pub trait AgentMiddleware: Send + Sync {
    fn name(&self) -> &str;

    /// Called before every model invocation. May edit messages and options.
    fn before_model(&self, _messages: &mut Vec<Message>, _opts: &mut ChatOptions) {}

    /// Called after every model invocation, before the response is returned
    /// to the caller. May edit the response (e.g. redact PII).
    fn after_model(&self, _messages: &[Message], _response: &mut ChatResponse) {}
}

/// Ordered chain of middlewares. Cheap to clone (`Arc` inside).
#[derive(Clone, Default)]
pub struct MiddlewareChain {
    inner: Vec<Arc<dyn AgentMiddleware>>,
}

impl std::fmt::Debug for MiddlewareChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiddlewareChain")
            .field(
                "names",
                &self.inner.iter().map(|m| m.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl MiddlewareChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, middleware: Arc<dyn AgentMiddleware>) -> Self {
        self.inner.push(middleware);
        self
    }

    pub fn push(&mut self, middleware: Arc<dyn AgentMiddleware>) {
        self.inner.push(middleware);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Names in registration order — exposed for debugging / Python repr.
    pub fn names(&self) -> Vec<String> {
        self.inner.iter().map(|m| m.name().to_string()).collect()
    }

    /// Run the `before_model` hooks in order.
    pub fn run_before(&self, messages: &mut Vec<Message>, opts: &mut ChatOptions) {
        for mw in &self.inner {
            mw.before_model(messages, opts);
        }
    }

    /// Run the `after_model` hooks in reverse order (onion unwind).
    pub fn run_after(&self, messages: &[Message], response: &mut ChatResponse) {
        for mw in self.inner.iter().rev() {
            mw.after_model(messages, response);
        }
    }
}

/// Wraps a `ChatModel` with a `MiddlewareChain`. Stream is passed through
/// unchanged today — the chain only runs around `invoke`. We deliberately
/// don't intercept stream chunks because middlewares are sync and per-chunk
/// dispatch would defeat their cheapness.
pub struct MiddlewareChatModel {
    inner: Arc<dyn ChatModel>,
    chain: MiddlewareChain,
}

impl MiddlewareChatModel {
    pub fn new(inner: Arc<dyn ChatModel>, chain: MiddlewareChain) -> Self {
        Self { inner, chain }
    }
}

#[async_trait]
impl ChatModel for MiddlewareChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let mut msgs = messages;
        let mut opts_owned = opts.clone();
        self.chain.run_before(&mut msgs, &mut opts_owned);
        let original_for_after = msgs.clone();
        let mut resp = self.inner.invoke(msgs, &opts_owned).await?;
        self.chain.run_after(&original_for_after, &mut resp);
        Ok(resp)
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        let mut msgs = messages;
        let mut opts_owned = opts.clone();
        self.chain.run_before(&mut msgs, &mut opts_owned);
        self.inner.stream(msgs, &opts_owned).await
    }
}

// ────────────────────────── built-in middlewares ──────────────────────────

/// Emits a `tracing` event before and after every call. Free in release if the
/// subscriber filters out the level.
#[derive(Debug, Clone, Default)]
pub struct LoggingMiddleware;

impl LoggingMiddleware {
    pub fn new() -> Self {
        Self
    }
}

impl AgentMiddleware for LoggingMiddleware {
    fn name(&self) -> &str {
        "logging"
    }

    fn before_model(&self, messages: &mut Vec<Message>, opts: &mut ChatOptions) {
        debug!(
            count = messages.len(),
            tools = opts.tools.len(),
            "litgraph.middleware.before_model",
        );
    }

    fn after_model(&self, _messages: &[Message], response: &mut ChatResponse) {
        info!(
            model = %response.model,
            prompt_tokens = response.usage.prompt,
            completion_tokens = response.usage.completion,
            finish = ?response.finish_reason,
            "litgraph.middleware.after_model",
        );
    }
}

/// Keep only the most recent `keep_last` non-system messages. System messages
/// (positionally pinned at the top) are always retained. Useful for clamping
/// very long histories before they hit the model.
#[derive(Debug, Clone)]
pub struct MessageWindowMiddleware {
    pub keep_last: usize,
}

impl MessageWindowMiddleware {
    pub fn new(keep_last: usize) -> Self {
        Self {
            keep_last: keep_last.max(1),
        }
    }
}

impl AgentMiddleware for MessageWindowMiddleware {
    fn name(&self) -> &str {
        "message_window"
    }

    fn before_model(&self, messages: &mut Vec<Message>, _opts: &mut ChatOptions) {
        let (system, rest): (Vec<_>, Vec<_>) =
            messages.drain(..).partition(|m| matches!(m.role, Role::System));
        let dropped = rest.len().saturating_sub(self.keep_last);
        let kept: Vec<Message> = rest.into_iter().skip(dropped).collect();
        messages.extend(system);
        messages.extend(kept);
    }
}

/// Prepend a system message if none is present. Idempotent — running this
/// middleware twice has the same effect as once.
#[derive(Debug, Clone)]
pub struct SystemPromptMiddleware {
    pub prompt: String,
}

impl SystemPromptMiddleware {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
        }
    }
}

impl AgentMiddleware for SystemPromptMiddleware {
    fn name(&self) -> &str {
        "system_prompt"
    }

    fn before_model(&self, messages: &mut Vec<Message>, _opts: &mut ChatOptions) {
        if messages.iter().any(|m| matches!(m.role, Role::System)) {
            return;
        }
        messages.insert(0, Message::system(self.prompt.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FinishReason, TokenUsage};
    use parking_lot::Mutex;

    struct EchoModel {
        calls: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    #[async_trait]
    impl ChatModel for EchoModel {
        fn name(&self) -> &str {
            "echo"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.calls.lock().push(messages.clone());
            let last_text = messages
                .last()
                .map(|m| m.text_content())
                .unwrap_or_default();
            Ok(ChatResponse {
                message: Message::assistant(last_text),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "echo".into(),
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

    fn echo() -> (Arc<dyn ChatModel>, Arc<Mutex<Vec<Vec<Message>>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let model = EchoModel {
            calls: calls.clone(),
        };
        (Arc::new(model), calls)
    }

    #[tokio::test]
    async fn empty_chain_is_passthrough() {
        let (model, calls) = echo();
        let chain = MiddlewareChain::new();
        let wrapped = MiddlewareChatModel::new(model, chain);
        let resp = wrapped
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(resp.message.text_content(), "hi");
        assert_eq!(calls.lock().len(), 1);
    }

    #[tokio::test]
    async fn system_prompt_prepended_when_absent() {
        let (model, calls) = echo();
        let chain = MiddlewareChain::new()
            .with(Arc::new(SystemPromptMiddleware::new("you are a duck")));
        let wrapped = MiddlewareChatModel::new(model, chain);
        wrapped
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        let seen = calls.lock();
        assert_eq!(seen[0].len(), 2);
        assert_eq!(seen[0][0].role, Role::System);
        assert_eq!(seen[0][0].text_content(), "you are a duck");
    }

    #[tokio::test]
    async fn system_prompt_idempotent() {
        let (model, calls) = echo();
        let chain = MiddlewareChain::new()
            .with(Arc::new(SystemPromptMiddleware::new("a")))
            .with(Arc::new(SystemPromptMiddleware::new("b")));
        let wrapped = MiddlewareChatModel::new(model, chain);
        wrapped
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        let seen = calls.lock();
        let systems: Vec<_> = seen[0].iter().filter(|m| m.role == Role::System).collect();
        assert_eq!(systems.len(), 1, "second middleware must not double-add");
        assert_eq!(systems[0].text_content(), "a");
    }

    #[tokio::test]
    async fn message_window_trims_non_system() {
        let (model, calls) = echo();
        let chain = MiddlewareChain::new().with(Arc::new(MessageWindowMiddleware::new(2)));
        let wrapped = MiddlewareChatModel::new(model, chain);
        let mut msgs = vec![Message::system("sys")];
        for i in 0..5 {
            msgs.push(Message::user(format!("u{i}")));
        }
        wrapped.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = calls.lock();
        // 1 system + 2 most-recent users = 3
        assert_eq!(seen[0].len(), 3);
        assert_eq!(seen[0][0].role, Role::System);
        assert_eq!(seen[0][1].text_content(), "u3");
        assert_eq!(seen[0][2].text_content(), "u4");
    }

    #[tokio::test]
    async fn middleware_runs_before_in_order_after_in_reverse() {
        // Two SystemPromptMiddlewares; order matters because second checks
        // for "any system message" and bails if the first added one.
        let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));

        struct OrderedMw {
            tag: &'static str,
            order: Arc<Mutex<Vec<&'static str>>>,
        }
        impl AgentMiddleware for OrderedMw {
            fn name(&self) -> &str {
                self.tag
            }
            fn before_model(&self, _messages: &mut Vec<Message>, _opts: &mut ChatOptions) {
                self.order.lock().push(self.tag);
            }
            fn after_model(&self, _messages: &[Message], _response: &mut ChatResponse) {
                let mut g = self.order.lock();
                g.push(self.tag);
            }
        }

        let (model, _) = echo();
        let chain = MiddlewareChain::new()
            .with(Arc::new(OrderedMw {
                tag: "A",
                order: order.clone(),
            }))
            .with(Arc::new(OrderedMw {
                tag: "B",
                order: order.clone(),
            }));
        let wrapped = MiddlewareChatModel::new(model, chain);
        wrapped
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        // before: A,B  after: B,A
        assert_eq!(*order.lock(), vec!["A", "B", "B", "A"]);
    }

    #[tokio::test]
    async fn names_returns_registered_in_order() {
        let chain = MiddlewareChain::new()
            .with(Arc::new(LoggingMiddleware::new()))
            .with(Arc::new(MessageWindowMiddleware::new(5)));
        assert_eq!(chain.names(), vec!["logging", "message_window"]);
    }
}
