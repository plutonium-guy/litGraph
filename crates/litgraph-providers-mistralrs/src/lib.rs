//! Local in-process chat backend for litGraph (Tier-1 roadmap #8).
//!
//! Closes the "air-gapped agent" gap: every other provider in the
//! workspace talks to a remote endpoint. `litgraph-providers-mistralrs`
//! runs inference *inside the same Rust process* via `mistralrs-core`
//! (or any backend implementing [`ModelBackend`] below) — no network,
//! no IPC, no Python subprocess.
//!
//! # Layout
//!
//! ```text
//!   ChatModel  ←  MistralRsChat<B: ModelBackend>  ←  ModelBackend trait
//!                                                       ├── MockModelBackend (tests)
//!                                                       └── MistralRsEngineBackend (iter 381, feature = "engine")
//! ```
//!
//! [`MistralRsChat`] is the public adapter — it implements the
//! framework's [`ChatModel`] trait by delegating to a generic
//! [`ModelBackend`] for the actual text generation. The backend split
//! keeps two concerns separate:
//!
//! 1. **Glue** (message-flattening, tool-schema rendering, stream
//!    framing, usage accounting) — owned by the adapter and shared
//!    across every backend implementation. Lives in this crate.
//! 2. **Inference** (forward pass, KV cache, tokenisation, sampling
//!    loop) — owned by the backend. The mock backend returns canned
//!    responses for CI determinism; the real engine wraps
//!    `mistralrs::Engine` behind the `engine` feature flag (lands
//!    iter 381 so this iter ships green tests without pulling ~30
//!    transitive crates into the default workspace build).
//!
//! # Why this exists
//!
//! - First-party offline backend. Every hosted-model provider in the
//!   workspace adds latency, cost, and an outbound network dep.
//! - In-process, GIL-free inference (Rust → no Python interpreter on
//!   the hot path). Token generation runs on the shared tokio runtime
//!   alongside vector search and embeddings.
//! - Shared KV cache + tokenizer instance across calls (per backend
//!   instance), removing the per-call HTTP handshake / TLS / serialize
//!   overhead that's the floor for hosted providers.
//! - Same `ChatModel` trait surface as every other provider so
//!   `ReactAgent`, `StateGraph`, `MiddlewareChain` etc. compose without
//!   special-casing.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, Error, FinishReason, Message, Result, TokenUsage,
};
use litgraph_core::model::{ChatStream, ChatStreamEvent};

/// Backend-side contract for `MistralRsChat`. Implement once per
/// backend (mock for tests, `mistralrs::Engine` for prod) and feed
/// the resulting type into [`MistralRsChat::with_backend`].
///
/// All methods take `&self` so the backend can be wrapped in an `Arc`
/// and shared across concurrent calls — same pattern as `Tool` /
/// `Embeddings` traits.
///
/// # Why we don't expose `&[Message]` directly
///
/// The backend works on flat tokenisable text, not message objects.
/// Conversation rendering (system prompt placement, role prefixes,
/// tool-call serialisation) is the *adapter's* job and is identical
/// across backends — pulling it into the trait would force every
/// future backend to re-implement it. The adapter flattens via
/// [`flatten_messages`] before calling the backend.
#[async_trait]
pub trait ModelBackend: Send + Sync {
    /// Identifier surfaced via `ChatModel::name`. Defaults to
    /// `"mistralrs-local"` if a backend declines to override.
    fn name(&self) -> &str {
        "mistralrs-local"
    }

    /// Run a single forward pass to completion. Return the full
    /// assistant text plus the prompt/completion token counts the
    /// backend observed. The adapter wraps this in a `ChatResponse`.
    async fn generate(&self, prompt: &str, opts: &GenOptions) -> Result<GenOutput>;
}

/// Backend-facing options. Reduced surface vs. the framework-wide
/// `ChatOptions` because most fields there (tools, response format,
/// model selection) are adapter-level concerns; the backend only
/// sees what it can act on inside the generation loop.
#[derive(Debug, Clone, Default)]
pub struct GenOptions {
    /// Sampling temperature. `None` defers to the backend default.
    pub temperature: Option<f32>,
    /// Hard cap on tokens generated. `None` defers to the backend.
    pub max_tokens: Option<u32>,
    /// Stop strings — generation halts as soon as one appears.
    pub stop: Vec<String>,
}

impl GenOptions {
    /// Project the framework-wide `ChatOptions` onto the backend's
    /// reduced surface. Returns the projection; the original is
    /// untouched so the adapter can still read fields like `tools`
    /// for tool-schema rendering.
    pub fn from_chat_options(opts: &ChatOptions) -> Self {
        Self {
            temperature: opts.temperature,
            max_tokens: opts.max_tokens.map(|n| n as u32),
            stop: opts.stop.clone().unwrap_or_default(),
        }
    }
}

/// What a backend returns from one generation pass.
#[derive(Debug, Clone)]
pub struct GenOutput {
    /// Full assistant text. Adapter wraps in a `Message::assistant`.
    pub text: String,
    /// Prompt tokens consumed (best effort — backends may estimate).
    pub prompt_tokens: u32,
    /// Completion tokens produced.
    pub completion_tokens: u32,
    /// `true` if generation halted because a stop string matched.
    /// `false` if the backend ran out naturally (EOS or `max_tokens`).
    pub stopped_by_stop_string: bool,
}

/// The chat adapter — implements `ChatModel` against a generic backend.
///
/// Use [`Self::with_backend`] to wire up a custom backend (mock,
/// mistralrs engine, llama.cpp shim, etc.).
pub struct MistralRsChat {
    backend: Arc<dyn ModelBackend>,
}

impl MistralRsChat {
    pub fn with_backend(backend: Arc<dyn ModelBackend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl ChatModel for MistralRsChat {
    fn name(&self) -> &str {
        self.backend.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let prompt = flatten_messages(&messages);
        let gen_opts = GenOptions::from_chat_options(opts);
        let out = self.backend.generate(&prompt, &gen_opts).await?;
        let finish_reason = if out.stopped_by_stop_string {
            FinishReason::Stop
        } else if let Some(max) = gen_opts.max_tokens {
            if out.completion_tokens >= max {
                FinishReason::Length
            } else {
                FinishReason::Stop
            }
        } else {
            FinishReason::Stop
        };
        Ok(ChatResponse {
            message: Message::assistant(out.text),
            finish_reason,
            usage: TokenUsage {
                prompt: out.prompt_tokens,
                completion: out.completion_tokens,
                total: out.prompt_tokens + out.completion_tokens,
                cache_creation: 0,
                cache_read: 0,
            },
            model: self.backend.name().to_string(),
        })
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // Token-level streaming lands iter 381 once the real engine
        // exposes its per-token callback. For now, the streaming
        // contract is honoured by buffering the full invoke result
        // and emitting it as one `Delta` + one `Done` — same shape
        // every downstream consumer (ReactAgent, AgentEvent::TokenDelta,
        // multiplex_chat_streams) sees from a hosted provider that
        // doesn't support streaming.
        let resp = self.invoke(messages, opts).await?;
        let text = resp.message.text_content();
        let resp_for_done = resp.clone();
        let s = async_stream::try_stream! {
            if !text.is_empty() {
                yield ChatStreamEvent::Delta { text };
            }
            yield ChatStreamEvent::Done { response: resp_for_done };
        };
        Ok(Box::pin(s) as ChatStream)
    }
}

/// Render a message list as a flat prompt the backend can tokenise.
///
/// Uses a generic role-tagged transcript format rather than a model-
/// specific chat template — backends that want their model's native
/// template should wrap the prompt themselves before generation.
///
/// Format:
///
/// ```text
/// <system>
/// you are a duck
/// </system>
/// <user>
/// hi
/// </user>
/// <assistant>
/// hello
/// </assistant>
/// <assistant>
/// ```
///
/// The trailing `<assistant>` opens the response slot — the model is
/// expected to produce text and (optionally) a `</assistant>` close.
/// Stop strings in `GenOptions.stop` typically include `</assistant>`.
pub fn flatten_messages(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        let tag = match m.role {
            litgraph_core::Role::System => "system",
            litgraph_core::Role::User => "user",
            litgraph_core::Role::Assistant => "assistant",
            litgraph_core::Role::Tool => "tool",
        };
        out.push('<');
        out.push_str(tag);
        out.push('>');
        out.push('\n');
        out.push_str(&m.text_content());
        out.push('\n');
        out.push('<');
        out.push('/');
        out.push_str(tag);
        out.push('>');
        out.push('\n');
    }
    out.push_str("<assistant>\n");
    out
}

// ─────────────────────── mock backend ───────────────────────────

/// A canned-response backend for tests + the framework's smoke tests.
/// Returns the same configured string for every call so unit tests
/// stay deterministic. Production code uses `MistralRsEngineBackend`
/// (iter 381, feature-gated).
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use litgraph_providers_mistralrs::{MistralRsChat, MockModelBackend};
///
/// let backend = MockModelBackend::new("hello from mock");
/// let chat = MistralRsChat::with_backend(Arc::new(backend));
/// ```
pub struct MockModelBackend {
    pub canned_response: String,
    pub fake_prompt_tokens: u32,
    pub fake_completion_tokens: u32,
    pub identifier: String,
}

impl MockModelBackend {
    pub fn new(canned: impl Into<String>) -> Self {
        let text = canned.into();
        let completion = text.split_whitespace().count() as u32;
        Self {
            fake_prompt_tokens: 0,
            fake_completion_tokens: completion,
            canned_response: text,
            identifier: "mistralrs-mock".to_string(),
        }
    }

    pub fn with_identifier(mut self, id: impl Into<String>) -> Self {
        self.identifier = id.into();
        self
    }
}

#[async_trait]
impl ModelBackend for MockModelBackend {
    fn name(&self) -> &str {
        &self.identifier
    }

    async fn generate(&self, prompt: &str, opts: &GenOptions) -> Result<GenOutput> {
        // Honour the stop-strings contract — if any stop string appears
        // inside the canned response, truncate at the first match and
        // report `stopped_by_stop_string`. Tests rely on this to verify
        // the adapter wires `opts.stop` through correctly.
        let mut text = self.canned_response.clone();
        let mut stopped = false;
        for s in &opts.stop {
            if let Some(idx) = text.find(s.as_str()) {
                text.truncate(idx);
                stopped = true;
                break;
            }
        }
        // Treat the rendered prompt as roughly the prompt-token count
        // (one token per whitespace-separated chunk). Tests that care
        // about exact numbers configure them via the struct fields.
        let prompt_tokens = if self.fake_prompt_tokens > 0 {
            self.fake_prompt_tokens
        } else {
            prompt.split_whitespace().count() as u32
        };
        Ok(GenOutput {
            text,
            prompt_tokens,
            completion_tokens: self.fake_completion_tokens,
            stopped_by_stop_string: stopped,
        })
    }
}

// ─────────────────────── real engine (iter 381) ─────────────────────

/// Placeholder for the real `mistralrs::Engine`-backed implementation.
///
/// Wired up in iter 381 once we add `mistralrs-core` as an optional
/// dep behind the `engine` feature flag. Today this module exists so
/// `engine = []` resolves cleanly and downstream feature plumbing
/// doesn't change shape between iters.
#[cfg(feature = "engine")]
pub mod engine {
    use super::*;

    /// Stub — replaced by the real impl in iter 381.
    pub struct MistralRsEngineBackend {
        _placeholder: (),
    }

    #[async_trait]
    impl ModelBackend for MistralRsEngineBackend {
        async fn generate(&self, _prompt: &str, _opts: &GenOptions) -> Result<GenOutput> {
            Err(Error::other(
                "MistralRsEngineBackend stub — real engine plumbing lands iter 381",
            ))
        }
    }
}
