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

use litgraph_graph::{CompiledGraph, END as GEND, GraphEvent, NodeOutput, START as GSTART, StateGraph};
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
    Ok(())
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
    fn add_node(&mut self, name: String, func: Py<PyAny>) -> PyResult<()> {
        let g = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("graph already compiled"))?;

        let fn_ref = Arc::new(func);
        let node_name = name.clone();
        g.add_fallible_node(name, move |state: Value| {
            let fn_ref = fn_ref.clone();
            let node_name = node_name.clone();
            Box::pin(async move {
                let result = call_py_callable_returning_json(&fn_ref, state).await;
                match result {
                    Ok(v) => Ok(NodeOutput::update(v)),
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
/// the returned coroutine and driving it via `asyncio.run()`. The GIL is held
/// for the duration — that's fine because the graph executor already serializes
/// Python calls (rayon would deadlock on contended Python state otherwise).
pub(crate) async fn call_py_callable_returning_json(
    func: &Arc<Py<PyAny>>,
    state: Value,
) -> Result<Value, String> {
    Python::with_gil(|py| {
        let py_state = json_to_py(py, &state).map_err(|e| e.to_string())?;
        let ret = func.call1(py, (py_state,)).map_err(|e| e.to_string())?;
        let mut bound = ret.bind(py).clone();
        if bound.hasattr("__await__").map_err(|e| e.to_string())? {
            // `asyncio.run(coro)` creates a fresh event loop, drives the
            // coroutine to completion, then tears the loop down. Simpler than
            // bridging tokio↔asyncio and avoids the "no running event loop"
            // error path you hit when no loop exists on the calling thread.
            let asyncio = py.import_bound("asyncio").map_err(|e| e.to_string())?;
            let result = asyncio
                .call_method1("run", (bound,))
                .map_err(|e| e.to_string())?;
            bound = result;
        }
        py_to_json(py, &bound).map_err(|e| e.to_string())
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
