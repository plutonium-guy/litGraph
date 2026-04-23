//! Parallel Kahn scheduler — executes the graph one "superstep" at a time.
//!
//! Each superstep collects the set of currently-ready nodes, spawns them on the tokio
//! runtime (bounded by a semaphore), merges their outputs into state via the reducer,
//! then computes the next frontier from static + conditional edges + dynamic `goto`
//! and `sends` from `NodeOutput`.
//!
//! This mirrors LangGraph's superstep model and gives trivial per-superstep parallelism:
//! N fan-out branches run concurrently without GIL contention.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::{Semaphore, mpsc};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info_span, warn};

use crate::checkpoint::{Checkpoint, Checkpointer};
use crate::event::GraphEvent;
use crate::graph::{EdgeKind, END, START, StateGraph};
use crate::interrupt::Command;
use crate::node::NodeOutput;
use crate::{GraphError, Result};

/// One entry on the ready frontier. A `Normal` entry runs the target node
/// with the current shared state; a `Forked` entry (from a LangGraph-style
/// `Send` command) runs the target with a per-invocation state override
/// merged in via the reducer, so parallel fan-outs don't see each other's
/// updates until the reduce phase.
#[derive(Debug, Clone)]
enum FrontierEntry {
    Normal(String),
    Forked { node: String, override_update: serde_json::Value },
}

impl FrontierEntry {
    fn node(&self) -> &str {
        match self {
            FrontierEntry::Normal(n) => n.as_str(),
            FrontierEntry::Forked { node, .. } => node.as_str(),
        }
    }
}

pub(crate) struct Scheduler<S>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    graph: Arc<StateGraph<S>>,
    cp: Arc<dyn Checkpointer>,
    thread_id: String,
    events: Option<mpsc::Sender<GraphEvent>>,
    step: u64,
    cancel: CancellationToken,
    /// Skip interrupt_before check for one superstep — used on resume so we don't
    /// re-fire the interrupt that suspended us.
    skip_interrupt_before_once: bool,
}

impl<S> Scheduler<S>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    pub fn new(graph: Arc<StateGraph<S>>, cp: Arc<dyn Checkpointer>, thread_id: Option<String>) -> Self {
        Self {
            graph,
            cp,
            thread_id: thread_id.unwrap_or_else(|| format!("t{}", now_ms())),
            events: None,
            step: 0,
            cancel: CancellationToken::new(),
            skip_interrupt_before_once: false,
        }
    }

    pub fn with_events(mut self, tx: mpsc::Sender<GraphEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    async fn emit(&self, ev: GraphEvent) {
        if let Some(tx) = &self.events {
            let _ = tx.send(ev).await;
        }
    }

    pub async fn run(&mut self, initial: S) -> Result<S> {
        self.emit(GraphEvent::GraphStart { thread_id: self.thread_id.clone() }).await;
        let frontier: Vec<FrontierEntry> = successors(&self.graph, START, &initial)
            .into_iter()
            .map(FrontierEntry::Normal)
            .collect();
        let out = self.execute(initial, frontier).await?;
        self.emit(GraphEvent::GraphEnd { thread_id: self.thread_id.clone(), steps: self.step }).await;
        Ok(out)
    }

    pub async fn resume_from_with_sends(
        &mut self,
        state: S,
        next_nodes: Vec<String>,
        next_sends: Vec<Command>,
    ) -> Result<S> {
        self.skip_interrupt_before_once = true;
        self.emit(GraphEvent::GraphStart { thread_id: self.thread_id.clone() }).await;
        let mut frontier: Vec<FrontierEntry> = next_nodes
            .into_iter()
            .map(FrontierEntry::Normal)
            .collect();
        for cmd in next_sends {
            frontier.push(FrontierEntry::Forked {
                node: cmd.goto,
                override_update: cmd.update,
            });
        }
        let out = self.execute(state, frontier).await?;
        self.emit(GraphEvent::GraphEnd { thread_id: self.thread_id.clone(), steps: self.step }).await;
        Ok(out)
    }

    async fn execute(&mut self, mut state: S, mut frontier: Vec<FrontierEntry>) -> Result<S> {
        let sem = Arc::new(Semaphore::new(self.graph.max_parallel));

        loop {
            if self.step >= self.graph.recursion_limit {
                return Err(GraphError::Other(format!(
                    "recursion limit ({}) reached", self.graph.recursion_limit
                )));
            }
            frontier.retain(|e| e.node() != END);
            // Dedup ONLY Normal entries (same-node, no override) while preserving
            // order — forked invocations with the same target node but different
            // overrides are intentionally distinct and must all run.
            let mut seen_normal: HashSet<String> = HashSet::new();
            let mut deduped: Vec<FrontierEntry> = Vec::with_capacity(frontier.len());
            for e in frontier.drain(..) {
                match &e {
                    FrontierEntry::Normal(n) => {
                        if seen_normal.insert(n.clone()) {
                            deduped.push(e);
                        }
                    }
                    FrontierEntry::Forked { .. } => deduped.push(e),
                }
            }
            frontier = deduped;
            if frontier.is_empty() { break; }

            // Check interrupt_before — if any queued node is in the set, stop and checkpoint.
            // On resume, skip this check for the first superstep (otherwise we'd re-fire
            // the same interrupt that suspended us).
            let skip = std::mem::replace(&mut self.skip_interrupt_before_once, false);
            let hit = if skip { None } else {
                frontier
                    .iter()
                    .find(|e| self.graph.interrupt_before.contains(e.node()))
                    .map(|e| e.node().to_string())
            };
            if let Some(hit) = hit {
                let (checkpoint_nodes, checkpoint_sends) = split_frontier(&frontier);
                let cp = Checkpoint {
                    thread_id: self.thread_id.clone(),
                    step: self.step,
                    state: bincode::serialize(&state)?,
                    next_nodes: checkpoint_nodes,
                    next_sends: checkpoint_sends,
                    pending_interrupt: Some(crate::interrupt::Interrupt {
                        node: hit.clone(),
                        payload: serde_json::Value::Null,
                    }),
                    ts_ms: now_ms(),
                };
                self.cp.put(cp).await?;
                self.emit(GraphEvent::Interrupt {
                    node: hit.clone(),
                    step: self.step,
                    payload: serde_json::Value::Null,
                }).await;
                return Err(GraphError::Interrupted(hit));
            }

            self.step += 1;
            let step = self.step;

            // Spawn all ready nodes in parallel — classic Kahn superstep.
            // Forked entries get a per-task state: reducer(state.clone(), override).
            // This is the key semantic for LangGraph-style `Send` fan-out:
            // sibling forks do NOT see each other's overrides at invocation time.
            let mut set: JoinSet<(String, Result<NodeOutput>)> = JoinSet::new();
            for fe in &frontier {
                let node_name = fe.node().to_string();
                if node_name == END { continue; }
                let Some(entry) = self.graph.nodes.get(&node_name) else {
                    return Err(GraphError::UnknownNode(node_name));
                };
                let func = entry.func.clone();
                let per_task_state = match fe {
                    FrontierEntry::Normal(_) => state.clone(),
                    FrontierEntry::Forked { override_update, .. } => {
                        if matches!(override_update, serde_json::Value::Null) {
                            state.clone()
                        } else {
                            (self.graph.reducer)(state.clone(), override_update.clone())?
                        }
                    }
                };
                let sem = sem.clone();
                let cancel = self.cancel.child_token();
                let name = node_name.clone();
                let tx = self.events.clone();
                set.spawn(async move {
                    let span = info_span!("node", name = %name, step);
                    let _g = span.enter();
                    let permit = match sem.acquire_owned().await {
                        Ok(p) => p,
                        Err(_) => return (name, Err(GraphError::Cancelled)),
                    };
                    if let Some(tx) = &tx {
                        let _ = tx.send(GraphEvent::NodeStart { node: name.clone(), step }).await;
                    }
                    let res = tokio::select! {
                        biased;
                        _ = cancel.cancelled() => Err(GraphError::Cancelled),
                        out = func(per_task_state) => out,
                    };
                    drop(permit);
                    (name, res)
                });
            }

            let mut next_frontier: Vec<FrontierEntry> = Vec::new();
            let mut any_err: Option<GraphError> = None;

            while let Some(joined) = set.join_next().await {
                let (name, res) = match joined {
                    Ok(pair) => pair,
                    Err(je) => {
                        warn!("node join error: {je}");
                        any_err.get_or_insert(GraphError::Panic(je.to_string()));
                        continue;
                    }
                };
                match res {
                    Ok(out) => {
                        self.emit(GraphEvent::NodeEnd {
                            node: name.clone(),
                            step,
                            update: out.update.clone(),
                        }).await;

                        // Reduce the node's output update into the shared
                        // state. For forked tasks this is the "reduce" half
                        // of map-reduce; all siblings' updates land here in
                        // completion order.
                        if !matches!(out.update, serde_json::Value::Null) {
                            state = (self.graph.reducer)(state, out.update)?;
                        }

                        if let Some(goto) = out.goto {
                            next_frontier.extend(goto.into_iter().map(FrontierEntry::Normal));
                        } else {
                            next_frontier.extend(
                                successors(&self.graph, &name, &state)
                                    .into_iter()
                                    .map(FrontierEntry::Normal),
                            );
                        }

                        // LangGraph-style `Send`: each cmd becomes a
                        // FORKED frontier entry with its own payload.
                        // Critically, we do NOT merge the payload into
                        // shared state here — that would make siblings see
                        // each other's inputs. Instead, the next superstep
                        // reduces each payload into a per-task state clone.
                        for cmd in out.sends {
                            next_frontier.push(FrontierEntry::Forked {
                                node: cmd.goto,
                                override_update: cmd.update,
                            });
                        }

                        if self.graph.interrupt_after.contains(&name) {
                            let (checkpoint_nodes, checkpoint_sends) = split_frontier(&next_frontier);
                            let cp = Checkpoint {
                                thread_id: self.thread_id.clone(),
                                step,
                                state: bincode::serialize(&state)?,
                                next_nodes: checkpoint_nodes,
                                next_sends: checkpoint_sends,
                                pending_interrupt: Some(crate::interrupt::Interrupt {
                                    node: name.clone(),
                                    payload: serde_json::Value::Null,
                                }),
                                ts_ms: now_ms(),
                            };
                            self.cp.put(cp).await?;
                            self.emit(GraphEvent::Interrupt {
                                node: name.clone(),
                                step,
                                payload: serde_json::Value::Null,
                            }).await;
                            // Cancel any still-running tasks in this superstep.
                            self.cancel.cancel();
                            return Err(GraphError::Interrupted(name));
                        }
                    }
                    Err(e) => {
                        self.emit(GraphEvent::NodeError {
                            node: name,
                            step,
                            error: e.to_string(),
                        }).await;
                        any_err.get_or_insert(e);
                        self.cancel.cancel();
                    }
                }
            }

            if let Some(e) = any_err { return Err(e); }

            // Persist post-superstep checkpoint.
            let (checkpoint_nodes, checkpoint_sends) = split_frontier(&next_frontier);
            let cp = Checkpoint {
                thread_id: self.thread_id.clone(),
                step,
                state: bincode::serialize(&state)?,
                next_nodes: checkpoint_nodes,
                next_sends: checkpoint_sends,
                pending_interrupt: None,
                ts_ms: now_ms(),
            };
            self.cp.put(cp).await?;

            debug!(step, next = ?next_frontier, "superstep done");
            frontier = next_frontier;
        }

        Ok(state)
    }
}

/// Split a frontier into the (Vec<String>, Vec<Command>) representation that
/// the `Checkpoint` carries on disk. Normal entries → next_nodes; Forked
/// entries → next_sends. Resume reverses this in `resume_from_with_sends`.
fn split_frontier(frontier: &[FrontierEntry]) -> (Vec<String>, Vec<Command>) {
    let mut nodes = Vec::new();
    let mut sends = Vec::new();
    for fe in frontier {
        match fe {
            FrontierEntry::Normal(n) => nodes.push(n.clone()),
            FrontierEntry::Forked { node, override_update } => {
                sends.push(Command {
                    goto: node.clone(),
                    update: override_update.clone(),
                });
            }
        }
    }
    (nodes, sends)
}

fn successors<S>(graph: &StateGraph<S>, from: &str, state: &S) -> Vec<String>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    let mut out = Vec::new();
    if let Some(edges) = graph.edges.get(from) {
        for e in edges {
            match e {
                EdgeKind::Static(to) => out.push(to.clone()),
                EdgeKind::Conditional(f) => out.extend(f(state)),
            }
        }
    }
    out
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}
