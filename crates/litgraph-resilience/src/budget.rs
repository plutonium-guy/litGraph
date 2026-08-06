use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Error, Message, Result};

/// Chat-model wrapper that enforces a token-count budget per invocation.
/// Two modes:
///
/// - **Strict (default)**: messages exceeding the budget → `Error::InvalidInput`.
///   Caller must trim / summarize upstream. Fails fast, predictable cost.
/// - **Auto-trim**: call `.auto_trim()` to enable — the wrapper uses
///   [`litgraph_tokenizers::trim_messages`] to drop oldest non-system
///   messages until under budget, then forwards. System messages are
///   ALWAYS preserved (they carry the persona / task). Last message
///   always preserved too (the user's actual query).
///
/// # Why
///
/// Long conversations silently balloon token cost. Without a budget,
/// a support chatbot that never forgets can rack up $100 LLM bills on
/// a single session. This wrapper makes the cap explicit.
///
/// # Model-specific token counts
///
/// The tokenizer used by [`trim_messages`] is picked by the inner
/// model's `name()`. Providers whose name contains "gpt" use tiktoken's
/// cl100k/o200k; others fall back to `tiktoken-rs` cl100k estimate.
/// Non-GPT-named models may be off by 5–15% — this is a best-effort
/// estimator, not a billing oracle.
///
/// # Composition
///
/// Safe to stack with other wrappers: `Retry(Budget(inner))` is fine,
/// as is `Budget(Retry(inner))`. Typical order: budget innermost
/// (trim ONCE per logical invocation, then retry with the same trimmed
/// message set on transient errors).
///
/// ```ignore
/// use litgraph_resilience::{RetryingChatModel, RetryConfig, TokenBudgetChatModel};
/// let budgeted = TokenBudgetChatModel::new(inner, 4096).auto_trim();
/// let retrying = RetryingChatModel::new(Arc::new(budgeted), RetryConfig::default());
/// ```
pub struct TokenBudgetChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub max_tokens: usize,
    pub auto_trim: bool,
}

impl TokenBudgetChatModel {
    /// Build in strict mode. Use [`.auto_trim()`] to switch to auto-trimming.
    pub fn new(inner: Arc<dyn ChatModel>, max_tokens: usize) -> Self {
        Self {
            inner,
            max_tokens,
            auto_trim: false,
        }
    }

    /// Enable auto-trimming mode.
    pub fn auto_trim(mut self) -> Self {
        self.auto_trim = true;
        self
    }
}

#[async_trait]
impl ChatModel for TokenBudgetChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let cost = litgraph_tokenizers::count_message_tokens(self.inner.name(), &messages);
        if cost <= self.max_tokens {
            return self.inner.invoke(messages, opts).await;
        }
        if !self.auto_trim {
            return Err(Error::invalid(format!(
                "TokenBudgetChatModel: messages use ~{} tokens, budget is {}. \
                 Trim upstream or enable .auto_trim().",
                cost, self.max_tokens
            )));
        }
        // Auto-trim mode: drop oldest non-system until under budget.
        let trimmed = litgraph_tokenizers::trim_messages(
            self.inner.name(),
            &messages,
            self.max_tokens,
        );
        tracing::debug!(
            model = self.inner.name(),
            input = messages.len(),
            kept = trimmed.len(),
            budget = self.max_tokens,
            "TokenBudgetChatModel auto-trimmed history"
        );
        self.inner.invoke(trimmed, opts).await
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // Same budget logic on the streaming path.
        let cost = litgraph_tokenizers::count_message_tokens(self.inner.name(), &messages);
        if cost <= self.max_tokens {
            return self.inner.stream(messages, opts).await;
        }
        if !self.auto_trim {
            return Err(Error::invalid(format!(
                "TokenBudgetChatModel: messages use ~{} tokens, budget is {}.",
                cost, self.max_tokens
            )));
        }
        let trimmed = litgraph_tokenizers::trim_messages(
            self.inner.name(),
            &messages,
            self.max_tokens,
        );
        self.inner.stream(trimmed, opts).await
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

    // ---- TokenBudgetChatModel tests -----------------------------------

    /// Captures `messages.len()` + `system message count` of each invoke call.
    struct CapturingChat {
        last_msg_count: std::sync::atomic::AtomicUsize,
        last_sys_count: std::sync::atomic::AtomicUsize,
    }

    impl CapturingChat {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                last_msg_count: std::sync::atomic::AtomicUsize::new(0),
                last_sys_count: std::sync::atomic::AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl ChatModel for CapturingChat {
        fn name(&self) -> &str {
            "gpt-4o-mini"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.last_msg_count
                .store(messages.len(), Ordering::SeqCst);
            self.last_sys_count.store(
                messages
                    .iter()
                    .filter(|m| matches!(m.role, Role::System))
                    .count(),
                Ordering::SeqCst,
            );
            Ok(ChatResponse {
                message: Message::assistant("ok"),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "gpt-4o-mini".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn long_history(n: usize) -> Vec<Message> {
        let mut out = vec![Message::system("You are helpful.")];
        for i in 0..n {
            out.push(Message::user(format!(
                "message {i} with some filler text to inflate token count"
            )));
            out.push(Message::assistant(format!(
                "response {i} with more filler to inflate tokens"
            )));
        }
        out
    }

    #[tokio::test]
    async fn budget_auto_trim_reduces_history_when_over() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner.clone() as Arc<dyn ChatModel>, 50)
            .auto_trim();
        let msgs = long_history(20); // ~41 messages total
        budget.invoke(msgs.clone(), &ChatOptions::default()).await.unwrap();
        let sent = inner.last_msg_count.load(Ordering::SeqCst);
        assert!(sent < msgs.len(), "expected trim: sent={sent}, input={}", msgs.len());
        // System message preserved.
        assert_eq!(inner.last_sys_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn budget_under_cap_passes_through_unchanged() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner.clone() as Arc<dyn ChatModel>, 10_000)
            .auto_trim();
        let msgs = vec![
            Message::system("be brief"),
            Message::user("hi"),
        ];
        budget.invoke(msgs.clone(), &ChatOptions::default()).await.unwrap();
        assert_eq!(inner.last_msg_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn budget_strict_mode_errors_on_overflow() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner.clone() as Arc<dyn ChatModel>, 50);
        // strict mode (default — auto_trim NOT called)
        let msgs = long_history(20);
        let err = budget
            .invoke(msgs, &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("budget"));
        // Inner model never called.
        assert_eq!(inner.last_msg_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn budget_strict_mode_under_cap_succeeds() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner.clone() as Arc<dyn ChatModel>, 10_000);
        let msgs = vec![Message::user("hi")];
        budget.invoke(msgs, &ChatOptions::default()).await.unwrap();
        assert_eq!(inner.last_msg_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn budget_preserves_system_message_even_under_tight_cap() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner.clone() as Arc<dyn ChatModel>, 20)
            .auto_trim();
        let msgs = long_history(30);
        budget.invoke(msgs, &ChatOptions::default()).await.unwrap();
        // System message always retained by trim_messages.
        assert_eq!(inner.last_sys_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn budget_proxies_name_from_inner() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner as Arc<dyn ChatModel>, 100);
        assert_eq!(budget.name(), "gpt-4o-mini");
    }

}


use litgraph_core::TokenUsage;
use litgraph_observability::cost::{ModelPrice, PriceSheet};
use parking_lot::Mutex as PlMutex;

/// Wrap any `ChatModel` with a hard USD cap on cumulative spend. Once the
/// running total crosses `max_usd`, subsequent `invoke`/`stream` calls fail
/// with `Error::InvalidInput` — before any request reaches the provider.
/// The failing call doesn't burn tokens.
///
/// # Why
///
/// Token budget (iter 130) caps the SIZE of any one call. Rate limit (iter 94)
/// caps the RATE of calls. Neither bounds cumulative $ — an agent stuck in a
/// tool-call loop can burn through a month's budget in minutes. CostCap is
/// the floor-level safety on dollar spend: declare a ceiling, get an error
/// instead of a bill.
///
/// # How the math works
///
/// On each successful invoke, cost is computed from `ChatResponse.usage` +
/// `PriceSheet::lookup(response.model)`:
/// - `prompt_tokens × prompt_per_mtok / 1M`
/// - `completion_tokens × completion_per_mtok / 1M`
/// - `cache_creation_tokens × prompt_per_mtok × 1.25 / 1M`  (Anthropic write)
/// - `cache_read_tokens × prompt_per_mtok × 0.10 / 1M`     (Anthropic read)
///
/// If the model isn't in the price sheet, the call adds 0 to the total — the
/// cap silently doesn't enforce for unpriced models. This is a deliberate
/// fail-open: an unrecognized custom model shouldn't halt the caller's pipeline.
/// The caller can pass a custom `PriceSheet` that includes their model to opt in.
///
/// Streams: cost is tallied from the `ChatStreamEvent::Done` final usage (which
/// the wrapper lets flow through verbatim — no reordering). Error variants on
/// the stream path are NOT charged.
///
/// # Thread safety
///
/// Running total guarded by a `parking_lot::Mutex<f64>`. Two concurrent
/// invokes against the same CostCap might both observe the total below cap
/// and both succeed — there's a small race window between the pre-check and
/// post-update. This is acceptable: over-shoot is bounded by (N_concurrent ×
/// cost_per_call), which for typical cap budgets is a rounding error. Tighter
/// pre-reservation would require estimating cost pre-call (impossible without
/// tokenizing + guessing completion length), which would silently over-reject.
///
/// ```rust,ignore
/// use litgraph_resilience::CostCappedChatModel;
/// use litgraph_observability::cost::default_prices;
/// let guarded = CostCappedChatModel::new(inner, default_prices(), 5.00);  // $5 cap
/// match guarded.invoke(msgs, &opts).await {
///     Ok(r) => { /* normal path */ }
///     Err(e) if e.to_string().contains("cost cap") => { /* over budget */ }
///     Err(e) => { /* other provider error */ }
/// }
/// ```
pub struct CostCappedChatModel {
    pub inner: Arc<dyn ChatModel>,
    prices: PriceSheet,
    max_usd: f64,
    total_usd: Arc<PlMutex<f64>>,
}

impl CostCappedChatModel {
    pub fn new(inner: Arc<dyn ChatModel>, prices: PriceSheet, max_usd: f64) -> Self {
        Self {
            inner,
            prices,
            max_usd: max_usd.max(0.0),
            total_usd: Arc::new(PlMutex::new(0.0)),
        }
    }

    /// Current cumulative spend in USD.
    pub fn total_usd(&self) -> f64 {
        *self.total_usd.lock()
    }

    /// Remaining budget (max_usd − total_usd, clamped at 0).
    pub fn remaining_usd(&self) -> f64 {
        (self.max_usd - self.total_usd()).max(0.0)
    }

    /// Reset the running counter (e.g. at midnight UTC for daily budgets).
    pub fn reset(&self) {
        *self.total_usd.lock() = 0.0;
    }

    /// Calculate the USD cost of a single response given its usage + model.
    /// Public for callers who want to replay the accounting manually.
    pub fn cost_of(&self, usage: &TokenUsage, model: &str) -> f64 {
        let Some(ModelPrice { prompt_per_mtok, completion_per_mtok }) = self.prices.lookup(model)
        else {
            return 0.0;
        };
        let mtok = 1_000_000.0;
        let prompt_cost = usage.prompt as f64 * prompt_per_mtok / mtok;
        let completion_cost = usage.completion as f64 * completion_per_mtok / mtok;
        // Anthropic cache pricing: creation = 1.25× prompt, read = 0.10× prompt.
        let cache_write_cost = usage.cache_creation as f64 * prompt_per_mtok * 1.25 / mtok;
        let cache_read_cost = usage.cache_read as f64 * prompt_per_mtok * 0.10 / mtok;
        prompt_cost + completion_cost + cache_write_cost + cache_read_cost
    }
}

#[async_trait]
impl ChatModel for CostCappedChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        // Pre-check: already over cap? reject before hitting the provider.
        {
            let total = *self.total_usd.lock();
            if total >= self.max_usd {
                return Err(Error::invalid(format!(
                    "CostCappedChatModel: cost cap exceeded (${:.4} used, ${:.4} limit)",
                    total, self.max_usd
                )));
            }
        }
        let resp = self.inner.invoke(messages, opts).await?;
        let cost = self.cost_of(&resp.usage, &resp.model);
        *self.total_usd.lock() += cost;
        tracing::debug!(
            model = %resp.model,
            call_usd = cost,
            total_usd = *self.total_usd.lock(),
            cap_usd = self.max_usd,
            "CostCappedChatModel charged"
        );
        Ok(resp)
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        {
            let total = *self.total_usd.lock();
            if total >= self.max_usd {
                return Err(Error::invalid(format!(
                    "CostCappedChatModel: cost cap exceeded (${:.4} used, ${:.4} limit)",
                    total, self.max_usd
                )));
            }
        }
        // Wrap the inner stream so the terminal Done event updates the total.
        // Don't attempt mid-stream termination if the user crosses the cap
        // during a single long stream — they're already mid-response and the
        // bill is already committed; just let it land and charge.
        let inner_stream = self.inner.stream(messages, opts).await?;
        let total = self.total_usd.clone();
        let prices = self.prices.clone();
        use futures_util::StreamExt;
        let mapped = inner_stream.map(move |event| {
            if let Ok(litgraph_core::model::ChatStreamEvent::Done { response }) = &event {
                if let Some(ModelPrice { prompt_per_mtok, completion_per_mtok }) =
                    prices.lookup(&response.model)
                {
                    let usage = &response.usage;
                    let mtok = 1_000_000.0;
                    let cost = usage.prompt as f64 * prompt_per_mtok / mtok
                        + usage.completion as f64 * completion_per_mtok / mtok
                        + usage.cache_creation as f64 * prompt_per_mtok * 1.25 / mtok
                        + usage.cache_read as f64 * prompt_per_mtok * 0.10 / mtok;
                    *total.lock() += cost;
                }
            }
            event
        });
        Ok(Box::pin(mapped))
    }
}

#[cfg(test)]
mod cost_cap_tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ChatResponse, Message};
    use litgraph_observability::cost::{ModelPrice, PriceSheet};

    struct FixedCostModel {
        usage: TokenUsage,
        model: String,
    }

    #[async_trait]
    impl ChatModel for FixedCostModel {
        fn name(&self) -> &str { "fixed" }
        async fn invoke(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("ok"),
                finish_reason: FinishReason::Stop,
                usage: self.usage,
                model: self.model.clone(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn prices_gpt4o() -> PriceSheet {
        let mut s = PriceSheet::new();
        // gpt-4o: $2.50/Mtok prompt, $10.00/Mtok completion.
        s.set("gpt-4o", ModelPrice { prompt_per_mtok: 2.50, completion_per_mtok: 10.00 });
        s
    }

    #[tokio::test]
    async fn passes_through_under_cap() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 1_000, completion: 500, total: 1_500, cache_creation: 0, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        // $0.0025 + $0.005 = $0.0075 per call
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 1.00);
        chat.invoke(vec![Message::user("hi")], &ChatOptions::default()).await.unwrap();
        let t = chat.total_usd();
        assert!((t - 0.0075).abs() < 1e-9, "expected ~$0.0075, got {t}");
    }

    #[tokio::test]
    async fn rejects_after_cumulative_over_cap() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 100_000, completion: 50_000, total: 150_000, cache_creation: 0, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        // $0.25 + $0.50 = $0.75 per call; cap $1.00 → call 1 ok, call 2 reject.
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 1.00);
        let r1 = chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await;
        assert!(r1.is_ok());
        // At $0.75 < $1.00 so next call is ALLOWED (pre-check only guards >=).
        let r2 = chat.invoke(vec![Message::user("b")], &ChatOptions::default()).await;
        assert!(r2.is_ok(), "second call pre-checks at $0.75 < $1.00");
        // Now at $1.50 > cap. Third call must be rejected.
        let r3 = chat.invoke(vec![Message::user("c")], &ChatOptions::default()).await;
        let err = r3.unwrap_err();
        assert!(err.to_string().contains("cost cap exceeded"),
                "unexpected: {err}");
    }

    #[tokio::test]
    async fn unpriced_model_charges_zero() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 1_000_000, completion: 1_000_000, total: 2_000_000, cache_creation: 0, cache_read: 0 },
            model: "custom-internal-model".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 0.01);
        // Call succeeds — no price for "custom-internal-model" in sheet.
        let r = chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await;
        assert!(r.is_ok());
        assert_eq!(chat.total_usd(), 0.0);
    }

    #[tokio::test]
    async fn cache_creation_charged_at_1_25x_prompt() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 0, completion: 0, total: 0, cache_creation: 1_000_000, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 10.0);
        chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await.unwrap();
        // 1M cache_creation × $2.50 × 1.25 = $3.125
        assert!((chat.total_usd() - 3.125).abs() < 1e-9);
    }

    #[tokio::test]
    async fn cache_read_charged_at_0_10x_prompt() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 0, completion: 0, total: 0, cache_creation: 0, cache_read: 1_000_000 },
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 10.0);
        chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await.unwrap();
        // 1M cache_read × $2.50 × 0.10 = $0.25
        assert!((chat.total_usd() - 0.25).abs() < 1e-9);
    }

    #[tokio::test]
    async fn remaining_usd_decreases_with_spend() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 100_000, completion: 0, total: 100_000, cache_creation: 0, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 1.00);
        assert!((chat.remaining_usd() - 1.00).abs() < 1e-9);
        chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await.unwrap();
        // $0.25 spent, $0.75 remaining.
        assert!((chat.remaining_usd() - 0.75).abs() < 1e-9);
    }

    #[tokio::test]
    async fn reset_returns_total_to_zero() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 100_000, completion: 50_000, total: 150_000, cache_creation: 0, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 10.00);
        chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await.unwrap();
        assert!(chat.total_usd() > 0.0);
        chat.reset();
        assert_eq!(chat.total_usd(), 0.0);
        assert!((chat.remaining_usd() - 10.00).abs() < 1e-9);
    }

    #[tokio::test]
    async fn zero_cap_rejects_all_requests() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage::default(),
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 0.0);
        // total = $0, cap = $0, pre-check `0 >= 0` → reject.
        let r = chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await;
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("cost cap exceeded"));
    }

    #[tokio::test]
    async fn negative_cap_clamps_to_zero() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage::default(),
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), -5.0);
        // Negative cap is clamped to $0 at construction.
        let r = chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await;
        assert!(r.is_err(), "negative cap treated as $0 — all calls rejected");
    }

    #[tokio::test]
    async fn cost_of_helper_matches_invoke_accounting() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 500_000, completion: 200_000, total: 700_000, cache_creation: 0, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 10.0);
        let expected = chat.cost_of(
            &TokenUsage { prompt: 500_000, completion: 200_000, total: 700_000, cache_creation: 0, cache_read: 0 },
            "gpt-4o",
        );
        chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await.unwrap();
        assert!((chat.total_usd() - expected).abs() < 1e-9);
    }

    #[tokio::test]
    async fn name_delegates_to_inner() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage::default(),
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 1.0);
        assert_eq!(chat.name(), "fixed");
    }
}

