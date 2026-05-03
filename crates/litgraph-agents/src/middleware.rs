//! Tool-call middleware traits for agent loops.
//!
//! Declarative trait surface that future agent implementations
//! (`react`, `supervisor`, `plan_execute`) can call to give users
//! cross-cutting hooks around every tool invocation. This iter
//! ships the *trait + types*; integration into the existing agent
//! loops follows in a separate change.
//!
//! ## Why a trait, not a closure
//!
//! Closures are fine for a single hook. Production stacks chain
//! several (audit → PII scrub → budget check → instrumentation), and
//! a trait + a `Vec<Arc<dyn ToolMiddleware>>` chain composes cleanly
//! across crates without Box<dyn Fn> juggling.
//!
//! ## Contract
//!
//! - `before_tool` runs *before* every tool dispatch. May mutate
//!   `args` (return `Some(new)` to replace) or short-circuit
//!   (return `Err(_)`).
//! - `after_tool` runs *after* every tool dispatch with the result.
//!   May replace the result or pass it through.
//! - Hooks run in registration order before; reverse order after
//!   (matches tower / express middleware semantics).
//!
//! Pure-Python users wire equivalent behaviour via
//! `litgraph.tool_hooks`; Rust users compose `ToolMiddleware`
//! impls directly.

use std::sync::Arc;

use serde_json::Value;

/// Cross-cutting hook around every tool call. Implementors can:
///
/// - log / audit / instrument (`before_tool` + `after_tool` no-ops);
/// - mutate args (`before_tool` returns `Some(replacement)`);
/// - replace results (`after_tool` returns a new `Value`);
/// - abort (`before_tool` returns an `Err`).
///
/// Implementations must be `Send + Sync` so the agent loop can hold
/// them in an `Arc<dyn ToolMiddleware>`.
pub trait ToolMiddleware: Send + Sync {
    /// Run before the tool dispatcher invokes the tool. The default
    /// impl is a no-op (returns `None` to mean "args unchanged").
    /// Return `Some(replacement)` to mutate the args. Return
    /// `Err(_)` to abort the whole tool call.
    fn before_tool(
        &self,
        _tool_name: &str,
        _args: &Value,
    ) -> Result<Option<Value>, MiddlewareError> {
        Ok(None)
    }

    /// Run after the tool dispatcher returns a result. The default
    /// impl is a no-op (returns `None` for "pass-through"). Return
    /// `Some(replacement)` to alter the result.
    fn after_tool(
        &self,
        _tool_name: &str,
        _args: &Value,
        _result: &Value,
    ) -> Result<Option<Value>, MiddlewareError> {
        Ok(None)
    }

    /// A short identifier for tracing / log output. Defaults to
    /// the trait impl's type name via Rust's [`std::any::type_name`].
    fn name(&self) -> &str {
        "ToolMiddleware"
    }
}

/// Error type for middleware short-circuits. Carries a string so the
/// agent surface can print it in error messages without forcing
/// `thiserror` on every middleware impl.
#[derive(Debug, Clone)]
pub struct MiddlewareError(pub String);

impl MiddlewareError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl std::fmt::Display for MiddlewareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tool middleware error: {}", self.0)
    }
}

impl std::error::Error for MiddlewareError {}

/// Registered chain of middleware. Cheap to clone (Arc inside).
/// Agents own one of these and run `dispatch_before` / `dispatch_after`
/// at the right loop boundaries.
#[derive(Clone, Default)]
pub struct ToolMiddlewareChain {
    inner: Vec<Arc<dyn ToolMiddleware>>,
}

impl ToolMiddlewareChain {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn push(mut self, mw: Arc<dyn ToolMiddleware>) -> Self {
        self.inner.push(mw);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Run every `before_tool` in registration order, threading
    /// args through each. Returns the final args (possibly mutated)
    /// or an `Err` from the first middleware that aborts.
    pub fn dispatch_before(
        &self,
        tool_name: &str,
        args: &Value,
    ) -> Result<Value, MiddlewareError> {
        let mut current = args.clone();
        for mw in &self.inner {
            if let Some(replacement) = mw.before_tool(tool_name, &current)? {
                current = replacement;
            }
        }
        Ok(current)
    }

    /// Run every `after_tool` in *reverse* registration order
    /// (tower / express semantics), threading the result through.
    pub fn dispatch_after(
        &self,
        tool_name: &str,
        args: &Value,
        result: &Value,
    ) -> Result<Value, MiddlewareError> {
        let mut current = result.clone();
        for mw in self.inner.iter().rev() {
            if let Some(replacement) = mw.after_tool(tool_name, args, &current)? {
                current = replacement;
            }
        }
        Ok(current)
    }
}

// ---- Built-ins ----

/// Hard cap on tool invocations across one agent turn. Counter
/// resets via [`Self::reset`]. Mirrors `litgraph.tool_hooks.ToolBudget`
/// on the Python side.
pub struct ToolBudgetMiddleware {
    max: usize,
    count: parking_lot::Mutex<usize>,
}

impl ToolBudgetMiddleware {
    pub fn new(max_calls_per_turn: usize) -> Self {
        Self {
            max: max_calls_per_turn,
            count: parking_lot::Mutex::new(0),
        }
    }

    pub fn reset(&self) {
        *self.count.lock() = 0;
    }

    pub fn calls(&self) -> usize {
        *self.count.lock()
    }
}

impl ToolMiddleware for ToolBudgetMiddleware {
    fn before_tool(
        &self,
        tool_name: &str,
        _args: &Value,
    ) -> Result<Option<Value>, MiddlewareError> {
        let mut c = self.count.lock();
        if *c >= self.max {
            return Err(MiddlewareError::new(format!(
                "tool budget exceeded: {}/{} (tried to call '{}')",
                *c, self.max, tool_name
            )));
        }
        *c += 1;
        Ok(None)
    }

    fn name(&self) -> &str {
        "ToolBudgetMiddleware"
    }
}

/// Wraps a child middleware and retries it up to `max_retries`
/// times on `MiddlewareError`. Useful for upstream hooks whose
/// failures are transient (token-refresh, rate-limit hint). Wraps
/// both `before_tool` and `after_tool`.
///
/// Note: this only retries the *middleware*, not the underlying
/// `tool.run()` call. The latter has its own retry primitives
/// (`RetryingChatModel`, etc.) for chat-side retries.
pub struct RetryOnMiddlewareErrorMiddleware<M: ToolMiddleware> {
    inner: M,
    max_retries: usize,
}

impl<M: ToolMiddleware> RetryOnMiddlewareErrorMiddleware<M> {
    pub fn new(inner: M, max_retries: usize) -> Self {
        Self { inner, max_retries }
    }
}

impl<M: ToolMiddleware> ToolMiddleware for RetryOnMiddlewareErrorMiddleware<M> {
    fn before_tool(
        &self,
        tool_name: &str,
        args: &Value,
    ) -> Result<Option<Value>, MiddlewareError> {
        let mut last_err: Option<MiddlewareError> = None;
        for _ in 0..=self.max_retries {
            match self.inner.before_tool(tool_name, args) {
                Ok(out) => return Ok(out),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| MiddlewareError::new("retry exhausted")))
    }

    fn after_tool(
        &self,
        tool_name: &str,
        args: &Value,
        result: &Value,
    ) -> Result<Option<Value>, MiddlewareError> {
        let mut last_err: Option<MiddlewareError> = None;
        for _ in 0..=self.max_retries {
            match self.inner.after_tool(tool_name, args, result) {
                Ok(out) => return Ok(out),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| MiddlewareError::new("retry exhausted")))
    }

    fn name(&self) -> &str {
        "RetryOnMiddlewareErrorMiddleware"
    }
}

/// Scrubs PII patterns from tool args + results. Default patterns
/// catch email addresses, US SSNs, and credit-card-shape digits.
/// Pass extra regexes via [`PiiScrubMiddleware::with_extra_patterns`]
/// for app-specific identifiers (employee IDs, internal account
/// numbers, etc.).
///
/// Walks JSON values: replaces every matching substring inside
/// `String` leaves with `[REDACTED]`. Numbers / bools / arrays /
/// nested objects are traversed.
///
/// Mirrors `PiiScrubbingChatModel` (the chat-side scrubber) so
/// agents using both get end-to-end PII protection without holes
/// at the tool boundary.
pub struct PiiScrubMiddleware {
    patterns: Vec<regex::Regex>,
    /// When true, scrub args before the tool runs (good default —
    /// keeps PII out of upstream APIs). When false, args pass
    /// through unchanged.
    scrub_args: bool,
    /// When true, scrub results after the tool runs (good default).
    scrub_results: bool,
}

impl PiiScrubMiddleware {
    /// New scrubber with the standard pattern set.
    pub fn new() -> Self {
        let patterns = vec![
            // Email
            regex::Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").unwrap(),
            // US SSN
            regex::Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            // Credit-card-shape digits (13–16 digits, optional spaces / dashes)
            regex::Regex::new(r"\b(?:\d[ -]?){13,16}\b").unwrap(),
        ];
        Self {
            patterns,
            scrub_args: true,
            scrub_results: true,
        }
    }

    /// Append app-specific patterns to the scrubber. Each input is
    /// compiled at construction; invalid regex panics with a clear
    /// message (use [`with_compiled_patterns`] for fallible input).
    pub fn with_extra_patterns<I>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        for src in patterns {
            self.patterns.push(
                regex::Regex::new(&src)
                    .unwrap_or_else(|e| panic!("PiiScrubMiddleware: bad regex {src:?}: {e}")),
            );
        }
        self
    }

    /// Append already-compiled patterns. Use when the regex source
    /// is dynamic / user-provided and you want to handle `regex::Error`
    /// upstream.
    pub fn with_compiled_patterns<I>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = regex::Regex>,
    {
        self.patterns.extend(patterns);
        self
    }

    /// Disable args scrubbing (results-only). Useful when the
    /// upstream API needs the unredacted PII to do its job
    /// (e.g., a user-lookup tool by email).
    pub fn args_unscrubbed(mut self) -> Self {
        self.scrub_args = false;
        self
    }

    /// Disable results scrubbing (args-only). Pairs with the above
    /// for the inverse: scrub the prompt-bound args but pass the
    /// tool's response through untouched.
    pub fn results_unscrubbed(mut self) -> Self {
        self.scrub_results = false;
        self
    }

    fn scrub_value(&self, v: &Value) -> Value {
        match v {
            Value::String(s) => {
                let mut out = s.clone();
                for re in &self.patterns {
                    out = re.replace_all(&out, "[REDACTED]").into_owned();
                }
                Value::String(out)
            }
            Value::Array(arr) => Value::Array(arr.iter().map(|x| self.scrub_value(x)).collect()),
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, x)| (k.clone(), self.scrub_value(x)))
                    .collect(),
            ),
            _ => v.clone(),
        }
    }
}

impl Default for PiiScrubMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolMiddleware for PiiScrubMiddleware {
    fn before_tool(
        &self,
        _tool_name: &str,
        args: &Value,
    ) -> Result<Option<Value>, MiddlewareError> {
        if self.scrub_args {
            Ok(Some(self.scrub_value(args)))
        } else {
            Ok(None)
        }
    }

    fn after_tool(
        &self,
        _tool_name: &str,
        _args: &Value,
        result: &Value,
    ) -> Result<Option<Value>, MiddlewareError> {
        if self.scrub_results {
            Ok(Some(self.scrub_value(result)))
        } else {
            Ok(None)
        }
    }

    fn name(&self) -> &str {
        "PiiScrubMiddleware"
    }
}

/// Logs every tool call's name + arg shape via `tracing::info!`.
/// Useful as the bottom of a middleware stack (runs first, sees
/// the unmodified args).
pub struct LogToolCallsMiddleware;

impl ToolMiddleware for LogToolCallsMiddleware {
    fn before_tool(
        &self,
        tool_name: &str,
        args: &Value,
    ) -> Result<Option<Value>, MiddlewareError> {
        tracing::info!(
            tool = tool_name,
            args = %args,
            "tool.call.before"
        );
        Ok(None)
    }

    fn after_tool(
        &self,
        tool_name: &str,
        _args: &Value,
        result: &Value,
    ) -> Result<Option<Value>, MiddlewareError> {
        tracing::info!(
            tool = tool_name,
            result = %result,
            "tool.call.after"
        );
        Ok(None)
    }

    fn name(&self) -> &str {
        "LogToolCallsMiddleware"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_chain_passes_through() {
        let chain = ToolMiddlewareChain::new();
        let args = json!({"a": 1});
        let result = json!({"out": 2});
        assert_eq!(chain.dispatch_before("x", &args).unwrap(), args);
        assert_eq!(chain.dispatch_after("x", &args, &result).unwrap(), result);
        assert!(chain.is_empty());
    }

    #[test]
    fn budget_caps_calls() {
        let budget = Arc::new(ToolBudgetMiddleware::new(2));
        let chain = ToolMiddlewareChain::new().push(budget.clone());
        chain.dispatch_before("a", &json!({})).unwrap();
        chain.dispatch_before("b", &json!({})).unwrap();
        let err = chain.dispatch_before("c", &json!({})).unwrap_err();
        assert!(err.0.contains("tool budget exceeded"));
        assert_eq!(budget.calls(), 2);
    }

    #[test]
    fn budget_reset() {
        let budget = Arc::new(ToolBudgetMiddleware::new(1));
        let chain = ToolMiddlewareChain::new().push(budget.clone());
        chain.dispatch_before("x", &json!({})).unwrap();
        assert!(chain.dispatch_before("x", &json!({})).is_err());
        budget.reset();
        chain.dispatch_before("x", &json!({})).unwrap();
    }

    #[test]
    fn middleware_can_mutate_args() {
        struct AddZ;
        impl ToolMiddleware for AddZ {
            fn before_tool(
                &self,
                _tn: &str,
                args: &Value,
            ) -> Result<Option<Value>, MiddlewareError> {
                let mut new = args.clone();
                if let Some(obj) = new.as_object_mut() {
                    obj.insert("z".into(), Value::from(9));
                }
                Ok(Some(new))
            }
        }
        let chain = ToolMiddlewareChain::new().push(Arc::new(AddZ));
        let out = chain.dispatch_before("any", &json!({"a": 1})).unwrap();
        assert_eq!(out["z"], 9);
        assert_eq!(out["a"], 1);
    }

    #[test]
    fn after_runs_in_reverse_order() {
        struct AppendName(String);
        impl ToolMiddleware for AppendName {
            fn after_tool(
                &self,
                _tn: &str,
                _args: &Value,
                result: &Value,
            ) -> Result<Option<Value>, MiddlewareError> {
                let prev = result.get("trace").and_then(|v| v.as_str()).unwrap_or("");
                let next = if prev.is_empty() {
                    self.0.clone()
                } else {
                    format!("{prev},{}", self.0)
                };
                let mut new = result.clone();
                if let Some(obj) = new.as_object_mut() {
                    obj.insert("trace".into(), Value::from(next));
                }
                Ok(Some(new))
            }
        }
        let chain = ToolMiddlewareChain::new()
            .push(Arc::new(AppendName("first".into())))
            .push(Arc::new(AppendName("second".into())));
        // Reverse order in dispatch_after → "second" runs first,
        // then "first" appends.
        let out = chain
            .dispatch_after("x", &json!({}), &json!({}))
            .unwrap();
        assert_eq!(out["trace"], "second,first");
    }

    #[test]
    fn middleware_can_short_circuit() {
        struct Reject;
        impl ToolMiddleware for Reject {
            fn before_tool(
                &self,
                _tn: &str,
                _args: &Value,
            ) -> Result<Option<Value>, MiddlewareError> {
                Err(MiddlewareError::new("nope"))
            }
        }
        let chain = ToolMiddlewareChain::new().push(Arc::new(Reject));
        let err = chain.dispatch_before("x", &json!({})).unwrap_err();
        assert_eq!(err.0, "nope");
    }

    #[test]
    fn retry_middleware_succeeds_when_inner_succeeds() {
        struct OkMw;
        impl ToolMiddleware for OkMw {
            fn before_tool(&self, _tn: &str, _args: &Value)
                -> Result<Option<Value>, MiddlewareError> { Ok(None) }
        }
        let mw = RetryOnMiddlewareErrorMiddleware::new(OkMw, 3);
        assert!(mw.before_tool("x", &json!({})).is_ok());
    }

    #[test]
    fn retry_middleware_eventually_succeeds_after_transient_error() {
        struct Flaky {
            count: parking_lot::Mutex<usize>,
            ok_after: usize,
        }
        impl ToolMiddleware for Flaky {
            fn before_tool(&self, _tn: &str, _args: &Value)
                -> Result<Option<Value>, MiddlewareError> {
                let mut c = self.count.lock();
                *c += 1;
                if *c <= self.ok_after {
                    Err(MiddlewareError::new("transient"))
                } else {
                    Ok(None)
                }
            }
        }
        let inner = Flaky {
            count: parking_lot::Mutex::new(0),
            ok_after: 2,
        };
        let mw = RetryOnMiddlewareErrorMiddleware::new(inner, 3);
        assert!(mw.before_tool("x", &json!({})).is_ok());
    }

    #[test]
    fn retry_middleware_exhausts_and_propagates_last_error() {
        struct Always;
        impl ToolMiddleware for Always {
            fn before_tool(&self, _tn: &str, _args: &Value)
                -> Result<Option<Value>, MiddlewareError> {
                Err(MiddlewareError::new("perm"))
            }
        }
        let mw = RetryOnMiddlewareErrorMiddleware::new(Always, 2);
        let err = mw.before_tool("x", &json!({})).unwrap_err();
        assert_eq!(err.0, "perm");
    }

    // ---- PiiScrubMiddleware ----

    #[test]
    fn pii_scrubs_email_in_string_arg() {
        let mw = PiiScrubMiddleware::new();
        let args = json!({"q": "contact alice@example.com please"});
        let out = mw.before_tool("any", &args).unwrap().unwrap();
        let q = out["q"].as_str().unwrap();
        assert!(!q.contains("alice@example.com"));
        assert!(q.contains("[REDACTED]"));
    }

    #[test]
    fn pii_scrubs_ssn() {
        let mw = PiiScrubMiddleware::new();
        let args = json!({"text": "SSN 123-45-6789 here"});
        let out = mw.before_tool("any", &args).unwrap().unwrap();
        assert!(out["text"].as_str().unwrap().contains("[REDACTED]"));
    }

    #[test]
    fn pii_traverses_nested_objects() {
        let mw = PiiScrubMiddleware::new();
        let args = json!({
            "user": {
                "name": "alice",
                "email": "alice@example.com",
            },
            "tags": ["greet", "send-to bob@example.com"]
        });
        let out = mw.before_tool("any", &args).unwrap().unwrap();
        assert!(out["user"]["email"].as_str().unwrap().contains("[REDACTED]"));
        assert!(out["tags"][1].as_str().unwrap().contains("[REDACTED]"));
        // Non-PII text untouched.
        assert_eq!(out["user"]["name"], "alice");
    }

    #[test]
    fn pii_scrubs_results_too_by_default() {
        let mw = PiiScrubMiddleware::new();
        let args = json!({});
        let result = json!({"answer": "Their email is bob@x.com"});
        let out = mw.after_tool("any", &args, &result).unwrap().unwrap();
        assert!(out["answer"].as_str().unwrap().contains("[REDACTED]"));
    }

    #[test]
    fn pii_args_unscrubbed_passes_through() {
        let mw = PiiScrubMiddleware::new().args_unscrubbed();
        let args = json!({"email": "alice@example.com"});
        // before returns None ⇒ args_unscrubbed
        assert!(mw.before_tool("any", &args).unwrap().is_none());
    }

    #[test]
    fn pii_results_unscrubbed_passes_through() {
        let mw = PiiScrubMiddleware::new().results_unscrubbed();
        let args = json!({});
        let result = json!({"email": "bob@example.com"});
        assert!(mw.after_tool("any", &args, &result).unwrap().is_none());
    }

    #[test]
    fn pii_extra_pattern_catches_app_specific_id() {
        let mw = PiiScrubMiddleware::new()
            .with_extra_patterns(vec![r"EMP-\d{6}".to_string()]);
        let args = json!({"text": "ID EMP-123456 here"});
        let out = mw.before_tool("any", &args).unwrap().unwrap();
        assert!(out["text"].as_str().unwrap().contains("[REDACTED]"));
    }

    // ---- Builder API ----

    #[test]
    fn react_agent_config_with_middleware_replaces_chain() {
        use crate::ReactAgentConfig;
        let chain = ToolMiddlewareChain::new()
            .push(Arc::new(LogToolCallsMiddleware));
        let cfg = ReactAgentConfig::default().with_middleware(chain);
        assert_eq!(cfg.tool_middleware.len(), 1);
    }

    #[test]
    fn react_agent_config_add_middleware_appends() {
        use crate::ReactAgentConfig;
        let cfg = ReactAgentConfig::default()
            .add_middleware(Arc::new(LogToolCallsMiddleware))
            .add_middleware(Arc::new(ToolBudgetMiddleware::new(5)));
        assert_eq!(cfg.tool_middleware.len(), 2);
    }
}
