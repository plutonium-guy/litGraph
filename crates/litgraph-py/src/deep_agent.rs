//! `create_deep_agent` — one-call factory that composes the deep-agents
//! primitives we already ship (PlanningTool, VirtualFilesystemTool,
//! SubagentTool, AGENTS.md/skills loaders, SystemPromptBuilder) into a
//! ready-to-run `ReactAgent`.
//!
//! Mirrors the LangChain `deepagents.create_deep_agent` ergonomics: the
//! caller hands over a model + tools, optionally a path to AGENTS.md and a
//! skills directory, and gets back a `ReactAgent` with PlanningTool +
//! VirtualFilesystemTool pre-injected and a system prompt assembled from the
//! markdown files on disk.
//!
//! Everything underneath is the same primitive set the user could wire by
//! hand; this exists purely to cut the boilerplate of a 12-line setup down
//! to one call.

use std::sync::Arc;

use litgraph_agents::{ReactAgent, ReactAgentConfig};
use litgraph_core::{
    load_agents_md as core_load_agents_md, load_skills_dir as core_load_skills_dir,
    ChatModel, ChatOptions, SystemPromptBuilder,
};
use litgraph_tools_utils::{PlanningTool, VirtualFilesystemTool};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::agents::{extract_chat_model, PyReactAgent};
use crate::tools::extract_tool_arc;

const DEFAULT_DEEP_AGENT_PROMPT: &str =
    "You are a capable autonomous agent. For multi-step work, use the `planning` \
     tool to track your todo list. For scratch state that should not pollute the \
     chat history (large file dumps, intermediate notes, working scratchpads), use \
     the `vfs` virtual-filesystem tool. Reach for tools when they shorten the path \
     to the answer; do not call them for trivial work the model can do directly.";

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(create_deep_agent, m)?)?;
    Ok(())
}

/// One-call factory for a deep-agents-style `ReactAgent`.
///
/// * `model` — any litgraph chat model (OpenAI, Anthropic, etc.) or a
///   wrapped one (`MiddlewareChat`, `RetryingChat`, …).
/// * `tools` — extra tools to expose to the agent, on top of the auto-injected
///   `PlanningTool` and `VirtualFilesystemTool`. Optional.
/// * `system_prompt` — base system prompt. Falls back to a sensible default
///   that tells the agent how to use planning + vfs.
/// * `agents_md_path` — path to a project-level AGENTS.md (or CLAUDE.md, or
///   any plain markdown file). Loaded and stitched into the system prompt
///   under a `## Memory (AGENTS.md)` section. Missing files are non-fatal
///   (treated as "no memory").
/// * `skills_dir` — directory of `*.md` skill files. Each becomes a
///   `### <name>` block under a `## Skills` section in the system prompt.
/// * `max_iterations` — passed through to `ReactAgentConfig`.
/// * `with_planning` / `with_vfs` — set False to skip the auto-inject of
///   either tool (e.g. if you want to provide your own).
#[pyfunction]
#[pyo3(signature = (
    model,
    tools=None,
    system_prompt=None,
    agents_md_path=None,
    skills_dir=None,
    max_iterations=15,
    with_planning=true,
    with_vfs=true,
))]
#[allow(clippy::too_many_arguments)]
pub fn create_deep_agent(
    model: Py<PyAny>,
    tools: Option<Bound<'_, PyList>>,
    system_prompt: Option<String>,
    agents_md_path: Option<String>,
    skills_dir: Option<String>,
    max_iterations: u32,
    with_planning: bool,
    with_vfs: bool,
) -> PyResult<PyReactAgent> {
    let chat_model: Arc<dyn ChatModel> =
        Python::with_gil(|py| extract_chat_model(model.bind(py)))?;

    let base = system_prompt.unwrap_or_else(|| DEFAULT_DEEP_AGENT_PROMPT.to_string());
    let mut builder = SystemPromptBuilder::new(base);

    if let Some(path) = agents_md_path {
        let body = core_load_agents_md(&path)
            .map_err(|e| PyIOError::new_err(format!("agents_md: {e}")))?;
        if let Some(b) = body {
            builder = builder.with_agents_md(b);
        }
    }

    if let Some(dir) = skills_dir {
        let skills = core_load_skills_dir(&dir)
            .map_err(|e| PyIOError::new_err(format!("skills_dir: {e}")))?;
        builder = builder.with_skills(skills);
    }

    let assembled_prompt = builder.build();

    let mut tool_vec: Vec<Arc<dyn litgraph_core::tool::Tool>> = Vec::new();
    if let Some(list) = tools {
        for item in list.iter() {
            tool_vec.push(extract_tool_arc(&item)?);
        }
    }
    if with_planning {
        tool_vec.push(Arc::new(PlanningTool::new()));
    }
    if with_vfs {
        tool_vec.push(Arc::new(VirtualFilesystemTool::new()));
    }

    let cfg = ReactAgentConfig {
        system_prompt: if assembled_prompt.is_empty() {
            None
        } else {
            Some(assembled_prompt)
        },
        max_iterations,
        chat_options: ChatOptions::default(),
        max_parallel_tools: 16,
    };
    let agent = ReactAgent::new(chat_model, tool_vec, cfg)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyReactAgent {
        inner: Arc::new(agent),
    })
}
