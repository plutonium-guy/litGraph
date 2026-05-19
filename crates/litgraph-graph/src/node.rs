use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::interrupt::Command;

/// What a node returns after executing.
#[derive(Debug, Clone)]
pub struct NodeOutput {
    /// Partial state update (must serialize to JSON object).
    pub update: Value,
    /// Optional explicit routing — overrides static edges for this step.
    pub goto: Option<Vec<String>>,
    /// Fan-out: emit N parallel sub-invocations (LangGraph `Send` API).
    pub sends: Vec<Command>,
}

impl NodeOutput {
    pub fn update<S: serde::Serialize>(partial: S) -> Self {
        Self {
            update: serde_json::to_value(partial).unwrap_or(Value::Null),
            goto: None,
            sends: vec![],
        }
    }

    pub fn goto(mut self, target: impl Into<String>) -> Self {
        self.goto.get_or_insert_with(Vec::new).push(target.into());
        self
    }

    pub fn send(mut self, cmd: Command) -> Self {
        self.sends.push(cmd);
        self
    }

    pub fn empty() -> Self {
        Self { update: Value::Object(Default::default()), goto: None, sends: vec![] }
    }
}

/// Erased async node function: `State -> Future<Output = Result<NodeOutput>>`.
pub type NodeFn<S> = Arc<
    dyn Fn(S) -> Pin<Box<dyn Future<Output = crate::Result<NodeOutput>> + Send>>
        + Send
        + Sync,
>;

pub(crate) fn wrap_node<S, F, Fut>(f: F) -> NodeFn<S>
where
    S: Send + 'static,
    F: Fn(S) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = NodeOutput> + Send + 'static,
{
    Arc::new(move |s: S| {
        let fut = f(s);
        Box::pin(async move { Ok(fut.await) })
    })
}

pub(crate) fn wrap_fallible_node<S, F, Fut>(f: F) -> NodeFn<S>
where
    S: Send + 'static,
    F: Fn(S) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = crate::Result<NodeOutput>> + Send + 'static,
{
    Arc::new(move |s: S| {
        let fut = f(s);
        Box::pin(fut)
    })
}

/// Wrap a synchronous CPU-bound closure as a `NodeFn`. The closure runs on
/// `tokio::task::spawn_blocking` so it doesn't stall the async runtime.
///
/// Use for nodes that do CPU-heavy work — local model inference inline,
/// tokenization over a big buffer, JSON-walking over megabytes of payload,
/// PDF rasterization, etc. If the closure is short or async-friendly,
/// `add_node` is the right call.
///
/// Cancellation: `spawn_blocking` runs on a dedicated thread pool and
/// cannot be cancelled mid-call by tokio. The scheduler's `cancel` token
/// only stops awaiting the JoinHandle; the closure still runs to
/// completion on its worker thread.
pub(crate) fn wrap_blocking_node<S, F>(f: F) -> NodeFn<S>
where
    S: Send + 'static,
    F: Fn(S) -> NodeOutput + Send + Sync + 'static,
{
    let f = Arc::new(f);
    Arc::new(move |s: S| {
        let f = f.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || f(s))
                .await
                .map_err(|je| crate::GraphError::Panic(je.to_string()))
        })
    })
}

/// Fallible variant of [`wrap_blocking_node`] — closure may return an error.
pub(crate) fn wrap_fallible_blocking_node<S, F>(f: F) -> NodeFn<S>
where
    S: Send + 'static,
    F: Fn(S) -> crate::Result<NodeOutput> + Send + Sync + 'static,
{
    let f = Arc::new(f);
    Arc::new(move |s: S| {
        let f = f.clone();
        Box::pin(async move {
            match tokio::task::spawn_blocking(move || f(s)).await {
                Ok(r) => r,
                Err(je) => Err(crate::GraphError::Panic(je.to_string())),
            }
        })
    })
}
