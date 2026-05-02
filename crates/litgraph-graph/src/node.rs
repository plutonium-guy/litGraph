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
