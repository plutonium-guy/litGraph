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
use crate::node::NodeOutput;
use crate::{GraphError, Result};

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
        let frontier = successors(&self.graph, START, &initial);
        let out = self.execute(initial, frontier).await?;
        self.emit(GraphEvent::GraphEnd { thread_id: self.thread_id.clone(), steps: self.step }).await;
        Ok(out)
    }

    pub async fn resume_from(&mut self, state: S, next_nodes: Vec<String>) -> Result<S> {
        self.skip_interrupt_before_once = true;
        self.emit(GraphEvent::GraphStart { thread_id: self.thread_id.clone() }).await;
        let out = self.execute(state, next_nodes).await?;
        self.emit(GraphEvent::GraphEnd { thread_id: self.thread_id.clone(), steps: self.step }).await;
        Ok(out)
    }

    async fn execute(&mut self, mut state: S, mut frontier: Vec<String>) -> Result<S> {
        let sem = Arc::new(Semaphore::new(self.graph.max_parallel));

        loop {
            if self.step >= self.graph.recursion_limit {
                return Err(GraphError::Other(format!(
                    "recursion limit ({}) reached", self.graph.recursion_limit
                )));
            }
            frontier.retain(|n| n != END);
            // Dedup while preserving order.
            let mut seen = HashSet::new();
            frontier.retain(|n| seen.insert(n.clone()));
            if frontier.is_empty() { break; }

            // Check interrupt_before — if any queued node is in the set, stop and checkpoint.
            // On resume, skip this check for the first superstep (otherwise we'd re-fire
            // the same interrupt that suspended us).
            let skip = std::mem::replace(&mut self.skip_interrupt_before_once, false);
            let hit = if skip { None } else {
                frontier.iter().find(|n| self.graph.interrupt_before.contains(*n)).cloned()
            };
            if let Some(hit) = hit {
                let cp = Checkpoint {
                    thread_id: self.thread_id.clone(),
                    step: self.step,
                    state: bincode::serialize(&state)?,
                    next_nodes: frontier.clone(),
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
            let mut set: JoinSet<(String, Result<NodeOutput>)> = JoinSet::new();
            for node_name in &frontier {
                if node_name == END { continue; }
                let Some(entry) = self.graph.nodes.get(node_name) else {
                    return Err(GraphError::UnknownNode(node_name.clone()));
                };
                let func = entry.func.clone();
                let st_clone = state.clone();
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
                        out = func(st_clone) => out,
                    };
                    drop(permit);
                    (name, res)
                });
            }

            let mut next_frontier: Vec<String> = Vec::new();
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

                        if !matches!(out.update, serde_json::Value::Null) {
                            state = (self.graph.reducer)(state, out.update)?;
                        }

                        if let Some(goto) = out.goto {
                            next_frontier.extend(goto);
                        } else {
                            next_frontier.extend(successors(&self.graph, &name, &state));
                        }

                        for cmd in out.sends {
                            // Apply fan-out command update then queue target.
                            if !matches!(cmd.update, serde_json::Value::Null) {
                                state = (self.graph.reducer)(state, cmd.update)?;
                            }
                            next_frontier.push(cmd.goto);
                        }

                        if self.graph.interrupt_after.contains(&name) {
                            let cp = Checkpoint {
                                thread_id: self.thread_id.clone(),
                                step,
                                state: bincode::serialize(&state)?,
                                next_nodes: next_frontier.clone(),
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
            let cp = Checkpoint {
                thread_id: self.thread_id.clone(),
                step,
                state: bincode::serialize(&state)?,
                next_nodes: next_frontier.clone(),
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
