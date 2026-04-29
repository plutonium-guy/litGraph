//! PyO3 bindings for litGraph. Thin shim — all heavy lifting lives in the workspace
//! crates. Python module layout:
//!
//! ```text
//! litgraph
//!   ├── __version__
//!   ├── providers.OpenAIChat
//!   ├── retrieval.Bm25Index
//!   ├── splitters.{RecursiveCharacterSplitter,MarkdownHeaderSplitter}
//!   ├── loaders.{TextLoader,JsonLinesLoader,MarkdownLoader,DirectoryLoader}
//!   ├── graph.{StateGraph, CompiledGraph, START, END}
//!   ├── observability.CostTracker
//!   └── cache.{MemoryCache, SqliteCache}
//! ```

use pyo3::prelude::*;

mod providers;
mod retrieval;
mod splitters;
mod loaders;
mod graph;
mod tools;
mod agents;
mod embeddings;
mod observability;
mod cache;
mod tokenizers;
mod memory;
mod mcp;
mod prompts;
mod parsers;
mod evaluators;
mod tracing_otel;
mod runtime;
mod store;
mod middleware;
mod deep_agent;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Register a submodule and make it importable as `litgraph.<name>` by inserting
/// into `sys.modules`. Without this, `from litgraph.graph import X` fails even
/// though `litgraph.graph.X` (attribute access) works.
fn add_sub(
    py: Python<'_>,
    parent: &Bound<'_, PyModule>,
    name: &str,
    register_fn: impl FnOnce(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let sub = PyModule::new_bound(py, name)?;
    register_fn(&sub)?;
    parent.add_submodule(&sub)?;
    let full = format!("litgraph.{name}");
    py.import_bound("sys")?
        .getattr("modules")?
        .set_item(full, &sub)?;
    Ok(())
}

#[pymodule]
fn litgraph(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    add_sub(py, m, "providers", providers::register)?;
    add_sub(py, m, "retrieval", retrieval::register)?;
    add_sub(py, m, "splitters", splitters::register)?;
    add_sub(py, m, "loaders", loaders::register)?;
    add_sub(py, m, "graph", graph::register)?;
    add_sub(py, m, "tools", tools::register)?;
    add_sub(py, m, "agents", agents::register)?;
    add_sub(py, m, "embeddings", embeddings::register)?;
    add_sub(py, m, "observability", observability::register)?;
    add_sub(py, m, "cache", cache::register)?;
    add_sub(py, m, "tokenizers", tokenizers::register)?;
    add_sub(py, m, "memory", memory::register)?;
    add_sub(py, m, "mcp", mcp::register)?;
    add_sub(py, m, "prompts", prompts::register)?;
    add_sub(py, m, "parsers", parsers::register)?;
    add_sub(py, m, "evaluators", evaluators::register)?;
    add_sub(py, m, "tracing", tracing_otel::register)?;
    add_sub(py, m, "store", store::register)?;
    add_sub(py, m, "middleware", middleware::register)?;
    add_sub(py, m, "deep_agent", deep_agent::register)?;

    Ok(())
}
