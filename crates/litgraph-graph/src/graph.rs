use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Serialize, de::DeserializeOwned};

use crate::checkpoint::{Checkpointer, MemoryCheckpointer};
use crate::interrupt::Command;
use crate::node::{
    NodeFn, NodeOutput, wrap_blocking_node, wrap_fallible_blocking_node,
    wrap_fallible_node, wrap_node,
};
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

    /// Add a CPU-bound node whose body runs on `tokio::task::spawn_blocking`,
    /// off the main async runtime. Use for inline local-model inference,
    /// big-buffer tokenization, PDF rasterization, or any synchronous work
    /// that would otherwise block sibling I/O nodes in the same superstep.
    ///
    /// The closure signature is sync (`Fn(S) -> NodeOutput`); for fallible
    /// CPU work use [`Self::add_fallible_blocking_node`].
    ///
    /// Cancellation: tokio cannot cancel a `spawn_blocking` thread mid-call.
    /// The graph's cancel token stops awaiting the result but the closure
    /// finishes on its worker thread.
    pub fn add_blocking_node<F>(&mut self, name: impl Into<String>, func: F) -> &mut Self
    where
        F: Fn(S) -> NodeOutput + Send + Sync + 'static,
    {
        self.nodes.insert(name.into(), NodeEntry { func: wrap_blocking_node(func) });
        self
    }

    /// Fallible CPU-bound node — same dispatch as [`Self::add_blocking_node`]
    /// but the closure may return `Result<NodeOutput>`.
    pub fn add_fallible_blocking_node<F>(
        &mut self,
        name: impl Into<String>,
        func: F,
    ) -> &mut Self
    where
        F: Fn(S) -> Result<NodeOutput> + Send + Sync + 'static,
    {
        self.nodes
            .insert(name.into(), NodeEntry { func: wrap_fallible_blocking_node(func) });
        self
    }

    /// Fan out `count` parallel branches of the same worker node. Each
    /// branch receives the same shared state plus a unique per-branch
    /// payload `{"_fanout_idx": <0..count>}` merged in via the
    /// reducer, so the worker can see which branch it's running.
    ///
    /// Concretely this generates two nodes:
    ///
    /// * `{name}_fanout` — emits `count` LangGraph-style `Send`
    ///   commands targeting `{name}_worker`, each with a distinct
    ///   `_fanout_idx`. Itself returns no state update.
    /// * `{name}_worker` — runs the user-supplied closure once per
    ///   branch in parallel (classic super-step fan-out via
    ///   `JoinSet`). Output update from each branch reduces back
    ///   into shared state.
    ///
    /// The caller wires the fan-out start with `add_edge(START,
    /// "<name>_fanout")` and the join with `add_edge("<name>_worker",
    /// END)` (or any downstream node).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use litgraph_graph::{NodeOutput, StateGraph, START, END};
    ///
    /// #[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
    /// struct State { results: Vec<u32> }
    ///
    /// let mut g = StateGraph::<State>::new();
    /// g.add_parallel_for("embed", 4, |s: State| async move {
    ///     // Each branch sees `_fanout_idx` in its state if the
    ///     // reducer surfaces it. Common pattern: read it from
    ///     // a custom field on State, or rely on a custom reducer.
    ///     let _ = s;
    ///     NodeOutput::update(State { results: vec![1] })
    /// });
    /// g.add_edge(START, "embed_fanout");
    /// g.add_edge("embed_worker", END);
    /// ```
    ///
    /// # When NOT to use
    ///
    /// - Heterogeneous fan-out (different worker per branch) — use
    ///   explicit `Send` commands from a custom node instead.
    /// - The branch count depends on runtime state — same path:
    ///   emit `Send` commands from a custom node. `add_parallel_for`
    ///   is for compile-time-fixed counts.
    pub fn add_parallel_for<F, Fut>(
        &mut self,
        name: impl Into<String>,
        count: usize,
        worker: F,
    ) -> &mut Self
    where
        F: Fn(S) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = NodeOutput> + Send + 'static,
    {
        let base = name.into();
        let fanout_name = format!("{base}_fanout");
        let worker_name = format!("{base}_worker");
        let worker_target = worker_name.clone();
        self.add_node(fanout_name, move |_s: S| {
            let worker_target = worker_target.clone();
            async move {
                let mut out = NodeOutput::empty();
                for i in 0..count {
                    out = out.send(crate::interrupt::Command {
                        goto: worker_target.clone(),
                        update: serde_json::json!({ "_fanout_idx": i }),
                    });
                }
                out
            }
        });
        self.add_node(worker_name, worker);
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

    /// Fan-out-N-copies shorthand over the `NodeOutput::send` / `Forked`
    /// frontier primitive. Registers two nodes: `name` — the dispatcher,
    /// which runs once, calls `select` to draw a collection out of state,
    /// and emits one `Send` per item, each carrying `{item_key: item}` as
    /// its `override_update` — and `worker`, which runs `body` once per
    /// item.
    ///
    /// Each `worker` invocation only ever sees its own item merged into a
    /// state clone (the reducer runs per-fork, same as hand-built sends);
    /// siblings never observe each other's item. Results flow back through
    /// the normal reduce path exactly like any other node's `update`.
    ///
    /// Wire the entry edge into `name` (the dispatcher); wire the
    /// continuation edge out of `worker` (it runs after every fan-out copy
    /// completes, deduplicated by the scheduler's frontier normalization) —
    /// edges added from `name` itself fire once, right after dispatch, not
    /// after the fan-out drains.
    pub fn parallel_for<Sel, F, Fut>(
        &mut self,
        name: impl Into<String>,
        worker: impl Into<String>,
        item_key: impl Into<String>,
        select: Sel,
        body: F,
    ) -> &mut Self
    where
        Sel: Fn(&S) -> Vec<serde_json::Value> + Send + Sync + 'static,
        F: Fn(S) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = NodeOutput> + Send + 'static,
    {
        let worker_name = worker.into();
        let item_key = item_key.into();
        let target = worker_name.clone();
        self.add_node(name, move |state: S| {
            let items = select(&state);
            let target = target.clone();
            let item_key = item_key.clone();
            async move {
                let mut out = NodeOutput::empty();
                for item in items {
                    let mut payload = serde_json::Map::new();
                    payload.insert(item_key.clone(), item);
                    out = out.send(Command::to(target.clone()).with(serde_json::Value::Object(payload)));
                }
                out
            }
        });
        self.add_node(worker_name, body);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Default, Debug, Serialize, serde::Deserialize)]
    struct Pool {
        #[serde(default)]
        pool: Vec<i64>,
        #[serde(default)]
        item: Option<i64>,
        #[serde(default)]
        results: Vec<i64>,
    }

    fn select_pool(s: &Pool) -> Vec<serde_json::Value> {
        s.pool.iter().map(|n| serde_json::Value::from(*n)).collect()
    }

    #[tokio::test]
    async fn parallel_for_fans_out_n_copies_and_reduces_results() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let mut g = StateGraph::<Pool>::new();
        let invs = invocations.clone();
        g.parallel_for("scatter", "work", "item", select_pool, move |s: Pool| {
            let invs = invs.clone();
            async move {
                invs.fetch_add(1, Ordering::SeqCst);
                let item = s.item.expect("worker must see its own item");
                NodeOutput::update(serde_json::json!({ "results": [item * 2] }))
            }
        });
        g.add_edge(START, "scatter");
        g.add_edge("work", END);

        let final_state = g
            .compile()
            .unwrap()
            .invoke(Pool { pool: vec![1, 2, 3, 4], ..Default::default() }, None)
            .await
            .unwrap();

        assert_eq!(invocations.load(Ordering::SeqCst), 4);
        let mut results = final_state.results;
        results.sort();
        assert_eq!(results, vec![2, 4, 6, 8]);
    }

    #[tokio::test]
    async fn parallel_for_isolates_sibling_invocations() {
        // Two forks carry distinct items into the same worker field. If the
        // scheduler merged overrides into shared state instead of per-task
        // clones, both invocations would observe whichever item landed last.
        let seen: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let mut g = StateGraph::<Pool>::new();
        let seen2 = seen.clone();
        g.parallel_for(
            "scatter",
            "work",
            "item",
            |_: &Pool| vec![serde_json::Value::from(10i64), serde_json::Value::from(20i64)],
            move |s: Pool| {
                let seen = seen2.clone();
                async move {
                    let item = s.item.expect("worker must see its own item");
                    seen.lock().unwrap().push(item);
                    NodeOutput::empty()
                }
            },
        );
        g.add_edge(START, "scatter");
        g.add_edge("work", END);

        g.compile().unwrap().invoke(Pool::default(), None).await.unwrap();

        let mut items = seen.lock().unwrap().clone();
        items.sort();
        assert_eq!(items, vec![10, 20]);
    }

    #[tokio::test]
    async fn parallel_for_empty_collection_does_not_hang() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let mut g = StateGraph::<Pool>::new();
        let invs = invocations.clone();
        g.parallel_for("scatter", "work", "item", select_pool, move |s: Pool| {
            let invs = invs.clone();
            async move {
                invs.fetch_add(1, Ordering::SeqCst);
                NodeOutput::update(serde_json::json!({ "results": [s.item.unwrap_or_default()] }))
            }
        });
        g.add_edge(START, "scatter");
        g.add_edge("work", END);

        let final_state = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            g.compile().unwrap().invoke(Pool::default(), None),
        )
        .await
        .expect("parallel_for over an empty collection must not hang")
        .unwrap();

        assert_eq!(invocations.load(Ordering::SeqCst), 0);
        assert!(final_state.results.is_empty());
    }

    #[tokio::test]
    async fn parallel_for_exceeds_max_parallel_still_completes() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let mut g = StateGraph::<Pool>::new();
        let invs = invocations.clone();
        g.parallel_for("scatter", "work", "item", select_pool, move |s: Pool| {
            let invs = invs.clone();
            async move {
                invs.fetch_add(1, Ordering::SeqCst);
                let item = s.item.expect("worker must see its own item");
                NodeOutput::update(serde_json::json!({ "results": [item] }))
            }
        });
        g.add_edge(START, "scatter");
        g.add_edge("work", END);
        let g = g.with_max_parallel(4);

        let pool: Vec<i64> = (0..20).collect();
        let final_state = g
            .compile()
            .unwrap()
            .invoke(Pool { pool: pool.clone(), ..Default::default() }, None)
            .await
            .unwrap();

        assert_eq!(invocations.load(Ordering::SeqCst), 20);
        let mut results = final_state.results;
        results.sort();
        assert_eq!(results, pool);
    }
}
