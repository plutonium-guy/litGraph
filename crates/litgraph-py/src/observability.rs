//! Python bindings for observability — CostTracker + PriceSheet.
//!
//! Usage:
//! ```python
//! from litgraph.observability import CostTracker
//! t = CostTracker({"gpt-5": (2.5, 10.0)})   # prompt_per_mtok, completion_per_mtok
//! # t is wired into providers via `provider.instrument(t)` (bound on provider classes).
//! t.usd()
//! t.snapshot()
//! ```

use std::sync::Arc;

use litgraph_observability::{default_prices as core_default_prices, CostTracker, ModelPrice, PriceSheet};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCostTracker>()?;
    m.add_function(wrap_pyfunction!(default_prices, m)?)?;
    m.add_class::<PyProgress>()?;
    m.add_class::<PyProgressObserver>()?;
    Ok(())
}

/// Latest-value progress observable, backed by `tokio::sync::watch`.
/// Multiple `Observer`s can read the **current** state on demand;
/// rapid intermediate writes collapse — observers always see the
/// latest snapshot, never a queue of past values.
///
/// Distinct from a multiplexed event stream (which keeps every
/// event); use this for "what's the current state?" UIs (progress
/// bars, health probes, agent dashboards).
///
/// State is any JSON-serializable Python value.
///
/// ```python
/// from litgraph.observability import Progress
/// p = Progress({"loaders_done": 0, "chunks_embedded": 0})
/// obs = p.observer()
/// # ... in another thread / task:
/// p.set({"loaders_done": 5, "chunks_embedded": 240})
/// snap = obs.snapshot()  # → {"loaders_done": 5, "chunks_embedded": 240}
/// ```
#[pyclass(name = "Progress", module = "litgraph.observability")]
pub struct PyProgress {
    inner: litgraph_core::Progress<serde_json::Value>,
}

#[pymethods]
impl PyProgress {
    #[new]
    fn new(initial: Bound<'_, PyAny>) -> PyResult<Self> {
        let v = crate::graph::py_to_json(initial.py(), &initial)?;
        Ok(Self {
            inner: litgraph_core::Progress::new(v),
        })
    }

    /// Overwrite the current value.
    fn set(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let v = crate::graph::py_to_json(value.py(), &value)?;
        // Ignore the SendError; we don't want set() to raise just
        // because no observer is currently registered (the value is
        // still stored).
        let _ = self.inner.set(v);
        Ok(())
    }

    /// Snapshot the current value.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.snapshot();
        crate::graph::json_to_py(py, &v)
    }

    /// Number of currently active observers.
    fn observer_count(&self) -> usize {
        self.inner.observer_count()
    }

    /// Create a new observer. Snapshots the current value on demand;
    /// `wait_changed` blocks until the next write or returns False on
    /// channel close (all senders dropped).
    fn observer(&self) -> PyProgressObserver {
        PyProgressObserver {
            inner: std::sync::Mutex::new(Some(self.inner.observer())),
        }
    }

    fn __repr__(&self) -> String {
        format!("Progress(observers={})", self.inner.observer_count())
    }
}

#[pyclass(name = "ProgressObserver", module = "litgraph.observability")]
pub struct PyProgressObserver {
    inner: std::sync::Mutex<Option<litgraph_core::ProgressObserver<serde_json::Value>>>,
}

#[pymethods]
impl PyProgressObserver {
    /// Snapshot the current value without blocking.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let guard = self.inner.lock().expect("poisoned");
        let obs = guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("observer closed"))?;
        let v = obs.snapshot();
        crate::graph::json_to_py(py, &v)
    }

    /// Block until the next change to the observed value, then
    /// return True. Returns False if the producer side has dropped
    /// (channel closed) — subsequent calls also return False.
    fn wait_changed(&self, py: Python<'_>) -> PyResult<bool> {
        // Take the observer out of its slot, run the await on a
        // worker thread, then put it back. We can't await with the
        // GIL held, and we can't move out of `&self`, so the Mutex
        // round-trip is the simplest pattern.
        let mut guard = self.inner.lock().expect("poisoned");
        let Some(mut obs) = guard.take() else {
            return Ok(false);
        };
        drop(guard);

        let (changed, returned) = py.allow_threads(|| {
            let changed = crate::runtime::block_on_compat(async {
                Ok::<bool, litgraph_core::Error>(obs.changed().await)
            })
            .unwrap_or(false);
            (changed, obs)
        });

        // Restore the observer for future wait_changed calls (and
        // for snapshot() — which still works after a closed channel
        // because `borrow()` just reads the last value).
        *self.inner.lock().expect("poisoned") = Some(returned);
        Ok(changed)
    }

    fn __repr__(&self) -> String {
        "ProgressObserver()".into()
    }
}

/// Built-in price sheet for the major hosted-LLM endpoints (OpenAI, Anthropic,
/// Gemini, Cohere, Voyage, Jina, Groq, Mistral, DeepSeek, xAI, Bedrock Titan).
/// Per-million-token USD. Use as
///
/// ```python
/// CostTracker(default_prices())
/// ```
///
/// Override individual entries by mutating the returned dict before passing
/// it to `CostTracker`. Versioned model IDs (e.g. `gpt-4o-2024-11-20`) match
/// the short keys via case-insensitive longest-substring lookup at runtime.
#[pyfunction]
fn default_prices<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let sheet = core_default_prices();
    let d = PyDict::new_bound(py);
    for (k, v) in sheet.iter() {
        d.set_item(k, (v.prompt_per_mtok, v.completion_per_mtok))?;
    }
    Ok(d)
}

/// Accumulates token + USD usage across LLM calls. Subscribe to a provider via
/// the provider's `instrument()` method (coming in the same iteration); until
/// then, users can construct it and feed events manually.
#[pyclass(name = "CostTracker", module = "litgraph.observability")]
pub struct PyCostTracker {
    pub(crate) inner: Arc<CostTracker>,
}

#[pymethods]
impl PyCostTracker {
    /// `prices` maps model name → (prompt_usd_per_mtok, completion_usd_per_mtok).
    #[new]
    fn new(prices: Bound<'_, PyDict>) -> PyResult<Self> {
        let mut sheet = PriceSheet::new();
        for (k, v) in prices.iter() {
            let model: String = k.extract()?;
            let tup: (f64, f64) = v.extract()?;
            sheet.set(model, ModelPrice { prompt_per_mtok: tup.0, completion_per_mtok: tup.1 });
        }
        Ok(Self { inner: Arc::new(CostTracker::new(sheet)) })
    }

    /// Running total in USD.
    fn usd(&self) -> f64 { self.inner.usd() }

    /// Full snapshot as a dict.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snap = self.inner.snapshot();
        let d = PyDict::new_bound(py);
        d.set_item("calls", snap.calls)?;
        d.set_item("prompt_tokens", snap.prompt_tokens)?;
        d.set_item("completion_tokens", snap.completion_tokens)?;
        d.set_item("usd", snap.usd)?;
        let per = PyDict::new_bound(py);
        for (k, v) in snap.per_model {
            let m = PyDict::new_bound(py);
            m.set_item("calls", v.calls)?;
            m.set_item("prompt_tokens", v.prompt_tokens)?;
            m.set_item("completion_tokens", v.completion_tokens)?;
            m.set_item("usd", v.usd)?;
            per.set_item(k, m)?;
        }
        d.set_item("per_model", per)?;
        Ok(d)
    }

    fn reset(&self) { self.inner.reset() }

    fn __repr__(&self) -> String {
        let s = self.inner.snapshot();
        format!("CostTracker(calls={}, usd=${:.4})", s.calls, s.usd)
    }
}
