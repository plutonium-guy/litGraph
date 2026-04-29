use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Serialize, de::DeserializeOwned};

use crate::checkpoint::{Checkpointer, MemoryCheckpointer};
use crate::node::{NodeFn, NodeOutput, wrap_fallible_node, wrap_node};
use crate::scheduler::Scheduler;
use crate::state::merge_append;
use crate::{GraphError, Result};

pub const START: &str = "__start__";
pub const END: &str = "__end__";

type CondFn<S> = Arc<dyn Fn(&S) -> Vec<String> + Send + Sync>;

pub(crate) enum EdgeKind<S> {
    Static(String),
    Conditional(CondFn<S>),
}

pub(crate) struct NodeEntry<S> {
    pub(crate) func: NodeFn<S>,
}

pub struct StateGraph<S>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    pub(crate) nodes: HashMap<String, NodeEntry<S>>,
    pub(crate) edges: HashMap<String, Vec<EdgeKind<S>>>,
    pub(crate) reducer: Arc<dyn Fn(S, serde_json::Value) -> Result<S> + Send + Sync>,
    pub(crate) interrupt_before: HashSet<String>,
    pub(crate) interrupt_after: HashSet<String>,
    pub(crate) max_parallel: usize,
    pub(crate) recursion_limit: u64,
}

impl<S> StateGraph<S>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            reducer: Arc::new(|s, u| merge_append(s, u)),
            interrupt_before: HashSet::new(),
            interrupt_after: HashSet::new(),
            max_parallel: 16,
            recursion_limit: 25,
        }
    }

    /// Supply a custom reducer. Default is `merge_append` (arrays concat, scalars replace).
    pub fn with_reducer<F>(mut self, f: F) -> Self
    where
        F: Fn(S, serde_json::Value) -> Result<S> + Send + Sync + 'static,
    {
        self.reducer = Arc::new(f);
        self
    }

    pub fn with_max_parallel(mut self, n: usize) -> Self { self.max_parallel = n.max(1); self }
    pub fn with_recursion_limit(mut self, n: u64) -> Self { self.recursion_limit = n; self }

    pub fn add_node<F, Fut>(&mut self, name: impl Into<String>, func: F) -> &mut Self
    where
        F: Fn(S) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = NodeOutput> + Send + 'static,
    {
        self.nodes.insert(name.into(), NodeEntry { func: wrap_node(func) });
        self
    }

    pub fn add_fallible_node<F, Fut>(&mut self, name: impl Into<String>, func: F) -> &mut Self
    where
        F: Fn(S) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<NodeOutput>> + Send + 'static,
    {
        self.nodes.insert(name.into(), NodeEntry { func: wrap_fallible_node(func) });
        self
    }

    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) -> &mut Self {
        self.edges.entry(from.into()).or_default().push(EdgeKind::Static(to.into()));
        self
    }

    /// Embed a compiled subgraph as a single node. When the parent reaches
    /// this node, the subgraph runs to completion using the **current parent
    /// state** as its initial state; the subgraph's final state is then
    /// emitted as a `NodeOutput::update` and reduced back into the parent
    /// via the parent's reducer.
    ///
    /// Same-shape composition: the subgraph's state type must match the
    /// parent's. For different-shape composition, write a regular node that
    /// projects/expands state and calls `compiled.invoke(...)` itself.
    ///
    /// Useful for hierarchical / multi-agent workflows: each "team" is a
    /// CompiledGraph; the top-level coordinator embeds them as nodes.
    pub fn add_subgraph(
        &mut self,
        name: impl Into<String>,
        sub: Arc<CompiledGraph<S>>,
    ) -> &mut Self {
        self.add_fallible_node(name, move |state: S| {
            let sub = sub.clone();
            Box::pin(async move {
                let final_state = sub.invoke(state, None).await
                    .map_err(|e| GraphError::Other(format!("subgraph: {e}")))?;
                let v = serde_json::to_value(&final_state)
                    .map_err(|e| GraphError::Other(format!("subgraph state serialize: {e}")))?;
                Ok(crate::NodeOutput::update(v))
            })
        });
        self
    }

    pub fn add_conditional_edges<F>(&mut self, from: impl Into<String>, router: F) -> &mut Self
    where
        F: Fn(&S) -> Vec<String> + Send + Sync + 'static,
    {
        self.edges
            .entry(from.into())
            .or_default()
            .push(EdgeKind::Conditional(Arc::new(router)));
        self
    }

    pub fn interrupt_before(&mut self, node: impl Into<String>) -> &mut Self {
        self.interrupt_before.insert(node.into());
        self
    }

    pub fn interrupt_after(&mut self, node: impl Into<String>) -> &mut Self {
        self.interrupt_after.insert(node.into());
        self
    }

    /// Set the graph entry by adding a `START -> node` edge.
    pub fn set_entry(&mut self, node: impl Into<String>) -> &mut Self {
        self.add_edge(START, node)
    }

    pub fn compile(self) -> Result<CompiledGraph<S>> {
        if !self.edges.contains_key(START) {
            return Err(GraphError::NoEntry);
        }
        for (from, edges) in &self.edges {
            if from != START && from != END && !self.nodes.contains_key(from) {
                return Err(GraphError::UnknownNode(from.clone()));
            }
            for e in edges {
                if let EdgeKind::Static(to) = e {
                    if to != END && to != START && !self.nodes.contains_key(to) {
                        return Err(GraphError::UnknownNode(to.clone()));
                    }
                }
            }
        }
        Ok(CompiledGraph {
            checkpointer: Arc::new(MemoryCheckpointer::new()),
            inner: Arc::new(self),
        })
    }
}

impl<S> Default for StateGraph<S>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    fn default() -> Self { Self::new() }
}

pub struct CompiledGraph<S>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    pub(crate) inner: Arc<StateGraph<S>>,
    pub(crate) checkpointer: Arc<dyn Checkpointer>,
}

impl<S> CompiledGraph<S>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    pub fn with_checkpointer(mut self, cp: Arc<dyn Checkpointer>) -> Self {
        self.checkpointer = cp;
        self
    }

    /// Borrow the underlying checkpointer — use for state-history +
    /// fork_at + rewind_to from outside the scheduler.
    pub fn checkpointer(&self) -> &Arc<dyn Checkpointer> {
        &self.checkpointer
    }

    pub async fn invoke(&self, initial: S, thread_id: Option<String>) -> Result<S> {
        let mut sched = Scheduler::new(self.inner.clone(), self.checkpointer.clone(), thread_id);
        sched.run(initial).await
    }

    /// Stream graph events as they happen. Returns a receiver; the graph runs to completion
    /// (or error) in a spawned tokio task.
    pub fn stream(
        &self,
        initial: S,
        thread_id: Option<String>,
    ) -> tokio::sync::mpsc::Receiver<crate::GraphEvent> {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let graph = self.inner.clone();
        let cp = self.checkpointer.clone();
        tokio::spawn(async move {
            let mut sched = Scheduler::new(graph, cp, thread_id).with_events(tx);
            let _ = sched.run(initial).await;
        });
        rx
    }

    /// Resume an interrupted graph with a value. Returns the final state.
    pub async fn resume(&self, thread_id: String, resume_value: serde_json::Value) -> Result<S> {
        let cp = self
            .checkpointer
            .latest(&thread_id)
            .await?
            .ok_or_else(|| GraphError::Checkpoint(format!("no checkpoint for {thread_id}")))?;
        let state: S = rmp_serde::from_slice(&cp.state)?;
        // Apply resume value as a state update via reducer, then continue.
        let state = (self.inner.reducer)(state, resume_value)?;
        let mut sched = Scheduler::new(self.inner.clone(), self.checkpointer.clone(), Some(thread_id));
        sched.resume_from_with_sends(state, cp.next_nodes, cp.next_sends).await
    }
}
