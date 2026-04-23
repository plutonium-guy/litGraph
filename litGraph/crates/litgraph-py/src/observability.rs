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
    Ok(())
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
