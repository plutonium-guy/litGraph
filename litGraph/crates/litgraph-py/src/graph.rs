//! Python-facing StateGraph. State is a JSON object (Python dict); nodes are
//! Python callables that receive a dict and return a dict.
//!
//! # GIL discipline
//!
//! - The Kahn scheduler runs on a background tokio runtime with the GIL **released**.
//! - To call a Python node, the node adapter briefly re-acquires the GIL
//!   (`Python::with_gil`), converts the JSON state to a Python dict, invokes the
//!   callable, and converts the return back to JSON. GIL is dropped for the rest
//!   of the superstep.
//!
//! This means: N parallel branches all compete for the GIL when they run Python
//! code (same as any Python threading), but the scheduler itself + edge evaluation
//! + reducers + I/O all run GIL-free.

use std::sync::{Arc, Mutex};

use litgraph_graph::{
    Command, CompiledGraph, END as GEND, GraphEvent, NodeOutput, START as GSTART, StateGraph,
};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::runtime::{block_on_compat, rt};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("START", GSTART)?;
    m.add("END", GEND)?;
    m.add_class::<PyStateGraph>()?;
    m.add_class::<PyCompiledGraph>()?;
    m.add_class::<PyGraphStream>()?;
    m.add_class::<PySend>()?;
    Ok(())
}

/// LangGraph-style `Send` — fan-out command that runs `goto` with a
/// per-invocation state override. Returned in a node's `__sends__` list to
/// trigger map-reduce-style parallel sub-invocations.
///
/// ```python
/// def split(state):
///     return {
///         "__update__": {"items": [1, 2, 3]},
///         "__sends__": [Send("worker", {"item": i}) for i in [1, 2, 3]],
///     }
/// ```
#[pyclass(name = "Send", module = "litgraph.graph")]
pub struct PySend {
    #[pyo3(get)]
    goto: String,
    #[pyo3(get)]
    update: Py<PyAny>,
}

#[pymethods]
impl PySend {
    #[new]
    fn new(goto: String, update: Py<PyAny>) -> Self {
        Self { goto, update }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let upd = self.update.bind(py).str().map(|s| s.to_string()).unwrap_or_else(|_| "?".into());
        format!("Send(goto='{}', update={})", self.goto, upd)
    }
}

/// Python StateGraph builder. State is always a JSON object (`dict`).
/// Sendable across threads — `inner` only holds Send+Sync types and the
/// Python callables stored in nodes are `Py<PyAny>` (ref-counted, Send+Sync
/// in PyO3 0.22+).
#[pyclass(name = "StateGraph", module = "litgraph.graph")]
pub struct PyStateGraph {
    /// Present until `compile()` consumes it.
    inner: Option<StateGraph<Value>>,
}

#[pymethods]
impl PyStateGraph {
    #[new]
    #[pyo3(signature = (max_parallel=16, recursion_limit=25))]
    fn new(max_parallel: usize, recursion_limit: u64) -> Self {
        let g = StateGraph::<Value>::new()
            .with_max_parallel(max_parallel)
            .with_recursion_limit(recursion_limit);
        Self { inner: Some(g) }
    }

    /// Register a node. `func` is any Python callable `dict -> dict` (or None
    /// for an empty update). Sync only for now — async Python support lands
    /// with pyo3-async-runtimes bridging in a later iteration.
    ///
    /// **Return value contract** (in priority order):
    /// - `None` → empty update, no routing change
    /// - `dict` containing any of `__sends__` / `__goto__` / `__update__`
    ///   → parsed as a structured NodeOutput. `__sends__` is a list of
    ///   `Send(goto, update)` instances (or `{"goto": ..., "update": ...}`
    ///   dicts) that fan out parallel sub-invocations with per-item state.
    ///   `__goto__` is a list[str] of explicit successors. `__update__` is
    ///   the state delta dict.
    /// - any other `dict` → treated as the state delta (legacy behavior).
    fn add_node(&mut self, name: String, func: Py<PyAny>) -> PyResult<()> {
        let g = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("graph already compiled"))?;

        let fn_ref = Arc::new(func);
        let node_name = name.clone();
        g.add_fallible_node(name, move |state: Value| {
            let fn_ref = fn_ref.clone();
            let node_name = node_name.clone();
            Box::pin(async move {
                let raw = call_py_callable_returning_raw(&fn_ref, state).await;
                match raw {
                    Ok(node_output) => Ok(node_output),
                    Err(e) => Err(litgraph_graph::GraphError::Other(
                        format!("node `{node_name}` failed: {e}"),
                    )),
                }
            })
        });
        Ok(())
    }

    fn add_edge(&mut self, from: String, to: String) -> PyResult<()> {
        let g = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("graph already compiled"))?;
        g.add_edge(from, to);
        Ok(())
    }

    /// Embed a `CompiledGraph` as a single node in this graph. When the
    /// parent reaches `name`, the subgraph runs to completion on the current
    /// parent state; its final state becomes the node's update.
    ///
    /// ```python
    /// team = sub_graph.compile()
    /// parent.add_subgraph("team", team)
    /// parent.add_edge("coordinator", "team")
    /// ```
    fn add_subgraph(&mut self, name: String, sub: PyRef<'_, PyCompiledGraph>) -> PyResult<()> {
        let g = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("graph already compiled"))?;
        g.add_subgraph(name, sub.inner.clone());
        Ok(())
    }

    /// Conditional edges: `router` is a Python callable `dict -> str | list[str]`.
    fn add_conditional_edges(&mut self, from: String, router: Py<PyAny>) -> PyResult<()> {
        let g = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("graph already compiled"))?;
        let r = Arc::new(router);
        g.add_conditional_edges(from, move |state: &Value| {
            let r = r.clone();
            let state = state.clone();
            Python::with_gil(|py| {
                let py_state = match json_to_py(py, &state) {
                    Ok(v) => v,
                    Err(_) => return vec![],
                };
                let ret = match r.call1(py, (py_state,)) {
                    Ok(v) => v,
                    Err(_) => return vec![],
                };
                let bound = ret.bind(py);
                if let Ok(s) = bound.extract::<String>() {
                    return vec![s];
                }
                if let Ok(list) = bound.downcast::<PyList>() {
                    return list
                        .iter()
                        .filter_map(|x| x.extract::<String>().ok())
                        .collect();
                }
                vec![]
            })
        });
        Ok(())
    }

    fn interrupt_before(&mut self, node: String) -> PyResult<()> {
        let g = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("graph already compiled"))?;
        g.interrupt_before(node);
        Ok(())
    }

    fn interrupt_after(&mut self, node: String) -> PyResult<()> {
        let g = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("graph already compiled"))?;
        g.interrupt_after(node);
        Ok(())
    }

    fn set_entry(&mut self, node: String) -> PyResult<()> { self.add_edge(GSTART.into(), node) }

    /// Compile the graph. Moves ownership — subsequent `add_*` calls raise.
    fn compile(&mut self) -> PyResult<PyCompiledGraph> {
        let g = self.inner.take()
            .ok_or_else(|| PyRuntimeError::new_err("graph already compiled"))?;
        let compiled = g.compile()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCompiledGraph { inner: Arc::new(compiled) })
    }
}

#[pyclass(name = "CompiledGraph", module = "litgraph.graph")]
pub struct PyCompiledGraph {
    inner: Arc<CompiledGraph<Value>>,
}

#[pymethods]
impl PyCompiledGraph {
    /// Synchronously run the graph. Releases the GIL for the duration.
    #[pyo3(signature = (state, thread_id=None))]
    fn invoke<'py>(
        &self,
        py: Python<'py>,
        state: Bound<'py, PyDict>,
        thread_id: Option<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let initial = py_dict_to_json(&state)?;
        let inner = self.inner.clone();
        let final_state: Value = py.allow_threads(|| {
            block_on_compat(async move { inner.invoke(initial, thread_id).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        json_to_py_dict(py, &final_state)
    }

    /// Resume an interrupted graph. `update` is merged into the checkpointed state
    /// via the reducer before execution continues.
    #[pyo3(signature = (thread_id, update=None))]
    fn resume<'py>(
        &self,
        py: Python<'py>,
        thread_id: String,
        update: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let upd = match update {
            Some(d) => py_dict_to_json(&d)?,
            None => Value::Object(Default::default()),
        };
        let inner = self.inner.clone();
        let final_state = py.allow_threads(|| {
            block_on_compat(async move { inner.resume(thread_id, upd).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        json_to_py_dict(py, &final_state)
    }

    /// Start streaming graph execution. Returns a `GraphStream` that is both an
    /// iterator (`for ev in compiled.stream(state)`) and iterable. Each yielded
    /// item is a `dict` describing one `GraphEvent`.
    #[pyo3(signature = (state, thread_id=None))]
    fn stream<'py>(
        &self,
        state: Bound<'py, PyDict>,
        thread_id: Option<String>,
    ) -> PyResult<PyGraphStream> {
        let initial = py_dict_to_json(&state)?;
        let inner = self.inner.clone();
        // `CompiledGraph::stream` calls `tokio::spawn` internally, which requires a
        // runtime context. We enter our shared runtime's handle before the call.
        let _guard = rt().enter();
        let rx = inner.stream(initial, thread_id);
        Ok(PyGraphStream { rx: Arc::new(Mutex::new(Some(rx))) })
    }

    // ─── time-travel API (iter 120) ───────────────────────────────────────

    /// List every checkpoint for `thread_id` in step order. Each entry is
    /// a dict: `{step, state, next_nodes, pending_interrupt, ts_ms}`.
    /// Useful for UIs ("show me the trajectory"), debugging ("what was
    /// state at step 3?"), and fork-point selection.
    fn state_history<'py>(
        &self,
        py: Python<'py>,
        thread_id: String,
    ) -> PyResult<Bound<'py, PyList>> {
        let cp = self.inner.checkpointer().clone();
        let entries = py.allow_threads(|| {
            block_on_compat(async move {
                litgraph_graph::state_history::<serde_json::Value>(&*cp, &thread_id).await
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let out = PyList::empty_bound(py);
        for e in entries {
            let d = PyDict::new_bound(py);
            d.set_item("thread_id", e.thread_id)?;
            d.set_item("step", e.step)?;
            d.set_item("state", json_to_py(py, &e.state)?)?;
            d.set_item("next_nodes", e.next_nodes)?;
            d.set_item(
                "pending_interrupt",
                match e.pending_interrupt {
                    Some(i) => json_to_py(py, &serde_json::to_value(i).unwrap_or_default())?,
                    None => py.None().into_bound(py),
                },
            )?;
            d.set_item("ts_ms", e.ts_ms)?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// Drop checkpoints with `step > target_step`. The retained checkpoint
    /// at `target_step` becomes the new latest; next `resume()` picks up
    /// from there. Returns the number of checkpoints dropped. Raises
    /// `ValueError` if `thread_id` has no checkpoint at `target_step`.
    fn rewind_to(
        &self,
        py: Python<'_>,
        thread_id: String,
        target_step: u64,
    ) -> PyResult<usize> {
        let cp = self.inner.checkpointer().clone();
        py.allow_threads(|| {
            block_on_compat(async move { cp.rewind_to(&thread_id, target_step).await })
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        })
    }

    /// Copy `thread_id` checkpoints with `step <= source_step` into
    /// `new_thread_id`. The new thread becomes an independent timeline
    /// branching off the original at `source_step`. Returns the number
    /// of checkpoints copied. Raises `ValueError` if `thread_id` has no
    /// checkpoint at `source_step` OR `new_thread_id` already has checkpoints.
    fn fork_at(
        &self,
        py: Python<'_>,
        thread_id: String,
        source_step: u64,
        new_thread_id: String,
    ) -> PyResult<usize> {
        let cp = self.inner.checkpointer().clone();
        py.allow_threads(|| {
            block_on_compat(async move {
                cp.fork_at(&thread_id, source_step, &new_thread_id).await
            })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        })
    }

    /// Drop ALL checkpoints for `thread_id`. Use for GDPR "delete this
    /// session" or resetting a stuck thread.
    fn clear_thread(&self, py: Python<'_>, thread_id: String) -> PyResult<()> {
        let cp = self.inner.checkpointer().clone();
        py.allow_threads(|| {
            block_on_compat(async move { cp.clear_thread(&thread_id).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
}

/// Iterator wrapper over a `Receiver<GraphEvent>`. Each `__next__` blocks on
/// `rx.recv()` with the GIL released, then converts the event into a Python dict.
#[pyclass(name = "GraphStream", module = "litgraph.graph")]
pub struct PyGraphStream {
    rx: Arc<Mutex<Option<mpsc::Receiver<GraphEvent>>>>,
}

#[pymethods]
impl PyGraphStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let slot = self.rx.clone();
        let ev = py.allow_threads(|| {
            // Take the Receiver out so we can await on it; put it back unless closed.
            let mut guard = slot.lock().expect("poisoned");
            let mut rx = match guard.take() {
                Some(r) => r,
                None => return Ok::<Option<GraphEvent>, PyErr>(None),
            };
            let got = block_on_compat(async { rx.recv().await });
            *guard = Some(rx);
            Ok(got)
        })?;
        match ev {
            Some(e) => event_to_py_dict(py, &e),
            None => {
                // Drop receiver and raise StopIteration.
                *self.rx.lock().expect("poisoned") = None;
                Err(PyStopIteration::new_err("graph stream exhausted"))
            }
        }
    }
}

fn event_to_py_dict<'py>(py: Python<'py>, ev: &GraphEvent) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    match ev {
        GraphEvent::GraphStart { thread_id } => {
            d.set_item("type", "graph_start")?;
            d.set_item("thread_id", thread_id)?;
        }
        GraphEvent::NodeStart { node, step } => {
            d.set_item("type", "node_start")?;
            d.set_item("node", node)?;
            d.set_item("step", *step)?;
        }
        GraphEvent::NodeEnd { node, step, update } => {
            d.set_item("type", "node_end")?;
            d.set_item("node", node)?;
            d.set_item("step", *step)?;
            d.set_item("update", json_to_py(py, update)?)?;
        }
        GraphEvent::NodeError { node, step, error } => {
            d.set_item("type", "node_error")?;
            d.set_item("node", node)?;
            d.set_item("step", *step)?;
            d.set_item("error", error)?;
        }
        GraphEvent::Interrupt { node, step, payload } => {
            d.set_item("type", "interrupt")?;
            d.set_item("node", node)?;
            d.set_item("step", *step)?;
            d.set_item("payload", json_to_py(py, payload)?)?;
        }
        GraphEvent::Custom { node, name, payload } => {
            d.set_item("type", "custom")?;
            d.set_item("node", node)?;
            d.set_item("name", name)?;
            d.set_item("payload", json_to_py(py, payload)?)?;
        }
        GraphEvent::GraphEnd { thread_id, steps } => {
            d.set_item("type", "graph_end")?;
            d.set_item("thread_id", thread_id)?;
            d.set_item("steps", *steps)?;
        }
    }
    Ok(d)
}

// ---------- JSON ↔ Python dict conversion --------------------------------

pub(crate) fn py_dict_to_json(d: &Bound<'_, PyDict>) -> PyResult<Value> {
    let mut m = serde_json::Map::with_capacity(d.len());
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        m.insert(key, py_to_json(d.py(), &v)?);
    }
    Ok(Value::Object(m))
}

pub(crate) fn py_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() { return Ok(Value::Null); }
    if let Ok(b) = obj.extract::<bool>() { return Ok(Value::Bool(b)); }
    if let Ok(i) = obj.extract::<i64>() { return Ok(serde_json::json!(i)); }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null));
    }
    if let Ok(s) = obj.extract::<String>() { return Ok(Value::String(s)); }
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for it in list.iter() {
            out.push(py_to_json(py, &it)?);
        }
        return Ok(Value::Array(out));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        return py_dict_to_json(dict);
    }
    // Fallback: repr as string so we never silently drop content.
    Ok(Value::String(obj.str()?.extract::<String>()?))
}

pub(crate) fn json_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    match v {
        Value::Null => Ok(py.None().into_bound(py)),
        Value::Bool(b) => Ok(b.into_py(py).into_bound(py)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() { Ok(i.into_py(py).into_bound(py)) }
            else if let Some(u) = n.as_u64() { Ok(u.into_py(py).into_bound(py)) }
            else if let Some(f) = n.as_f64() { Ok(f.into_py(py).into_bound(py)) }
            else { Err(PyValueError::new_err("bad number")) }
        }
        Value::String(s) => Ok(s.clone().into_py(py).into_bound(py)),
        Value::Array(a) => {
            let list = PyList::empty_bound(py);
            for item in a { list.append(json_to_py(py, item)?)?; }
            Ok(list.into_any())
        }
        Value::Object(m) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in m { dict.set_item(k, json_to_py(py, v)?)?; }
            Ok(dict.into_any())
        }
    }
}

/// Call a Python callable; transparently support `async def` nodes by detecting
/// the returned coroutine and driving it via `asyncio.run()`. Interprets the
/// dict's special `__sends__` / `__goto__` / `__update__` keys (if any) and
/// returns a structured `NodeOutput`. Plain-dict returns (no special keys)
/// become a state-update-only `NodeOutput` — preserves the legacy node
/// contract. The GIL is held for the duration — that's fine because the graph
/// executor already serializes Python calls (rayon would deadlock on
/// contended Python state otherwise).
pub(crate) async fn call_py_callable_returning_raw(
    func: &Arc<Py<PyAny>>,
    state: Value,
) -> Result<NodeOutput, String> {
    Python::with_gil(|py| {
        let py_state = json_to_py(py, &state).map_err(|e| e.to_string())?;
        let ret = func.call1(py, (py_state,)).map_err(|e| e.to_string())?;
        let mut bound = ret.bind(py).clone();
        if bound.hasattr("__await__").map_err(|e| e.to_string())? {
            let asyncio = py.import_bound("asyncio").map_err(|e| e.to_string())?;
            bound = asyncio
                .call_method1("run", (bound,))
                .map_err(|e| e.to_string())?;
        }

        // None → empty NodeOutput.
        if bound.is_none() {
            return Ok(NodeOutput::empty());
        }

        // Dict path: look for special keys. If absent, treat the whole dict
        // as the state update (legacy contract).
        if let Ok(d) = bound.downcast::<PyDict>() {
            let has_sends = d.contains("__sends__").unwrap_or(false);
            let has_goto = d.contains("__goto__").unwrap_or(false);
            let has_update = d.contains("__update__").unwrap_or(false);
            if !(has_sends || has_goto || has_update) {
                let v = py_to_json(py, &bound).map_err(|e| e.to_string())?;
                return Ok(NodeOutput::update(v));
            }

            let update_value = if has_update {
                let u = d.get_item("__update__").map_err(|e| e.to_string())?;
                if let Some(u) = u {
                    py_to_json(py, &u).map_err(|e| e.to_string())?
                } else {
                    Value::Null
                }
            } else {
                Value::Null
            };
            let mut out = if matches!(update_value, Value::Null) {
                NodeOutput::empty()
            } else {
                NodeOutput::update(update_value)
            };

            if has_goto {
                let g = d.get_item("__goto__").map_err(|e| e.to_string())?;
                if let Some(g) = g {
                    if let Ok(s) = g.extract::<String>() {
                        out = out.goto(s);
                    } else if let Ok(list) = g.downcast::<PyList>() {
                        for item in list.iter() {
                            if let Ok(s) = item.extract::<String>() {
                                out = out.goto(s);
                            }
                        }
                    } else {
                        return Err("__goto__ must be a str or list[str]".into());
                    }
                }
            }

            if has_sends {
                let s = d.get_item("__sends__").map_err(|e| e.to_string())?;
                if let Some(s) = s {
                    let list = s.downcast::<PyList>()
                        .map_err(|_| "__sends__ must be a list".to_string())?;
                    for item in list.iter() {
                        let cmd = if let Ok(send) = item.extract::<PyRef<PySend>>() {
                            let upd = py_to_json(py, send.update.bind(py))
                                .map_err(|e| e.to_string())?;
                            Command { goto: send.goto.clone(), update: upd }
                        } else if let Ok(d) = item.downcast::<PyDict>() {
                            let goto: String = d
                                .get_item("goto").map_err(|e| e.to_string())?
                                .ok_or_else(|| "__sends__ dict missing 'goto'".to_string())?
                                .extract().map_err(|e: PyErr| e.to_string())?;
                            let upd = match d.get_item("update").map_err(|e| e.to_string())? {
                                Some(u) => py_to_json(py, &u).map_err(|e| e.to_string())?,
                                None => Value::Null,
                            };
                            Command { goto, update: upd }
                        } else {
                            return Err("__sends__ items must be Send or dict".into());
                        };
                        out = out.send(cmd);
                    }
                }
            }
            return Ok(out);
        }

        // Non-dict, non-None — wrap under "value" key as a state update so
        // weird returns don't crash the graph.
        let v = py_to_json(py, &bound).map_err(|e| e.to_string())?;
        Ok(NodeOutput::update(serde_json::json!({"value": v})))
    })
}

pub(crate) fn json_to_py_dict<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyDict>> {
    match v {
        Value::Object(_) => {
            let any = json_to_py(py, v)?;
            Ok(any.downcast_into::<PyDict>()
                .map_err(|_| PyRuntimeError::new_err("expected dict"))?)
        }
        _ => {
            // Wrap non-object state under a `value` key so Python always sees a dict.
            let d = PyDict::new_bound(py);
            d.set_item("value", json_to_py(py, v)?)?;
            Ok(d)
        }
    }
}
