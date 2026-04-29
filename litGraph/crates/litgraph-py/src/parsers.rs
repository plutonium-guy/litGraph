//! Python bindings for output parsers — currently XML extraction helpers
//! (LangChain `XMLOutputParser` parity). More parsers (regex, list,
//! comma-separated-list) land here over time.

use std::sync::Arc;

use litgraph_core::{
    boolean_format_instructions, comma_list_format_instructions, fix_with_llm as core_fix,
    markdown_list_format_instructions, numbered_list_format_instructions, parse_boolean,
    parse_comma_list, parse_markdown_list, parse_nested_xml, parse_numbered_list,
    parse_partial_json as core_parse_partial, parse_react_step,
    parse_with_retry as core_parse_retry, parse_xml_tags, react_format_instructions,
    repair_partial_json as core_repair_partial, xml_format_instructions, ChatModel, ChatOptions,
    Error, ReactStep,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::json_to_py;
use crate::runtime::block_on_compat;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_xml_tags_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_nested_xml_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_comma_list_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_numbered_list_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_markdown_list_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_boolean_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_react_step_py, m)?)?;
    m.add_function(wrap_pyfunction!(comma_list_format_instructions_py, m)?)?;
    m.add_function(wrap_pyfunction!(numbered_list_format_instructions_py, m)?)?;
    m.add_function(wrap_pyfunction!(markdown_list_format_instructions_py, m)?)?;
    m.add_function(wrap_pyfunction!(boolean_format_instructions_py, m)?)?;
    m.add_function(wrap_pyfunction!(xml_format_instructions_py, m)?)?;
    m.add_function(wrap_pyfunction!(react_format_instructions_py, m)?)?;
    m.add_function(wrap_pyfunction!(fix_with_llm_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_json_with_retry_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_partial_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(repair_partial_json_py, m)?)?;
    Ok(())
}

/// Flat XML parser — extract the FIRST occurrence of each named tag's
/// inner content. Missing tags are absent from the returned dict (NOT an
/// error — LLMs often forget tags and the caller decides whether to retry).
///
/// ```python
/// from litgraph.parsers import parse_xml_tags
/// text = "<thinking>Let me work this out</thinking>\n<answer>42</answer>"
/// m = parse_xml_tags(text, ["thinking", "answer"])
/// # {"thinking": "Let me work this out", "answer": "42"}
/// ```
#[pyfunction(name = "parse_xml_tags")]
fn parse_xml_tags_py<'py>(
    py: Python<'py>,
    text: String,
    tags: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
    let map = parse_xml_tags(&text, &tag_refs);
    let out = PyDict::new_bound(py);
    for (k, v) in map {
        out.set_item(k, v)?;
    }
    Ok(out)
}

/// Tree XML parser — walk the text and build a nested dict/list of every
/// tag. Leaf tags → string; container tags → dict of their children;
/// repeated same-name children → list.
///
/// ```python
/// from litgraph.parsers import parse_nested_xml
/// text = "<root><item>a</item><item>b</item><name>foo</name></root>"
/// v = parse_nested_xml(text)
/// # {"root": {"item": ["a", "b"], "name": "foo"}}
/// ```
#[pyfunction(name = "parse_nested_xml")]
fn parse_nested_xml_py<'py>(py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
    let v = parse_nested_xml(&text).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_to_py(py, &v)
}

/// Parse a comma-separated list. Strips surrounding prose, trailing
/// punctuation, and quotes around items.
///
/// ```python
/// from litgraph.parsers import parse_comma_list
/// parse_comma_list("Here you go: apple, banana, cherry.")
/// # ["apple", "banana", "cherry"]
/// ```
#[pyfunction(name = "parse_comma_list")]
fn parse_comma_list_py(text: String) -> Vec<String> {
    parse_comma_list(&text)
}

/// Parse a numbered list (`1.`, `1)`, `1:`, or `1 -` delimiters).
///
/// ```python
/// from litgraph.parsers import parse_numbered_list
/// parse_numbered_list("1. apple\n2. banana")
/// # ["apple", "banana"]
/// ```
#[pyfunction(name = "parse_numbered_list")]
fn parse_numbered_list_py(text: String) -> Vec<String> {
    parse_numbered_list(&text)
}

/// Parse a markdown-style bulleted list (`-`, `*`, `+`, `•` bullets).
///
/// ```python
/// from litgraph.parsers import parse_markdown_list
/// parse_markdown_list("- apple\n* banana")
/// # ["apple", "banana"]
/// ```
#[pyfunction(name = "parse_markdown_list")]
fn parse_markdown_list_py(text: String) -> Vec<String> {
    parse_markdown_list(&text)
}

/// Parse a yes/no answer. Raises `ValueError` if ambiguous or missing.
///
/// Accepted affirmative forms: yes / y / true / 1 / affirmative / correct.
/// Accepted negative forms: no / n / false / 0 / negative / incorrect.
///
/// ```python
/// from litgraph.parsers import parse_boolean
/// parse_boolean("Yes, that's right.")  # True
/// parse_boolean("no way")              # False
/// ```
#[pyfunction(name = "parse_boolean")]
fn parse_boolean_py(text: String) -> PyResult<bool> {
    parse_boolean(&text).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Parse a ReAct-format LLM response. Returns a dict:
/// `{"kind": "action", "thought": str|None, "tool": str, "input": Any}`
/// or `{"kind": "final", "thought": str|None, "answer": str}`.
///
/// Raises `ValueError` if neither an `Action:` + `Action Input:` pair
/// nor a `Final Answer:` is found.
///
/// Use this for non-tool-calling models (Ollama local, older Gemini,
/// base-completion) that return reasoning as prose. Tool-calling models
/// (OpenAI, Anthropic, Bedrock Converse) should use the native tool API
/// exposed by the provider adapters instead.
///
/// ```python
/// from litgraph.parsers import parse_react_step
/// text = '''Thought: I need the weather.
/// Action: get_weather
/// Action Input: {"city": "Paris"}'''
/// step = parse_react_step(text)
/// # {"kind": "action", "thought": "I need the weather.",
/// #  "tool": "get_weather", "input": {"city": "Paris"}}
/// ```
#[pyfunction(name = "parse_react_step")]
fn parse_react_step_py<'py>(py: Python<'py>, text: String) -> PyResult<Bound<'py, PyDict>> {
    let step = parse_react_step(&text).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = PyDict::new_bound(py);
    match step {
        ReactStep::Action {
            thought,
            tool,
            input,
        } => {
            out.set_item("kind", "action")?;
            out.set_item("thought", thought)?;
            out.set_item("tool", tool)?;
            out.set_item("input", json_to_py(py, &input)?)?;
        }
        ReactStep::Final { thought, answer } => {
            out.set_item("kind", "final")?;
            out.set_item("thought", thought)?;
            out.set_item("answer", answer)?;
        }
    }
    Ok(out)
}

/// Prompt snippet telling the LLM to emit a comma-separated list.
///
/// ```python
/// from litgraph.parsers import comma_list_format_instructions, parse_comma_list
/// prompt = f"List 5 fruits.\n\n{comma_list_format_instructions()}"
/// response = chat.invoke(prompt)
/// items = parse_comma_list(response["content"])
/// ```
#[pyfunction(name = "comma_list_format_instructions")]
fn comma_list_format_instructions_py() -> String {
    comma_list_format_instructions()
}

/// Prompt snippet telling the LLM to emit a numbered list (`1. foo`).
#[pyfunction(name = "numbered_list_format_instructions")]
fn numbered_list_format_instructions_py() -> String {
    numbered_list_format_instructions()
}

/// Prompt snippet telling the LLM to emit a markdown bulleted list (`- foo`).
#[pyfunction(name = "markdown_list_format_instructions")]
fn markdown_list_format_instructions_py() -> String {
    markdown_list_format_instructions()
}

/// Prompt snippet telling the LLM to answer with yes/no.
#[pyfunction(name = "boolean_format_instructions")]
fn boolean_format_instructions_py() -> String {
    boolean_format_instructions()
}

/// Prompt snippet telling the LLM to wrap each named field in XML tags.
///
/// ```python
/// from litgraph.parsers import xml_format_instructions, parse_xml_tags
/// prompt = xml_format_instructions(["thinking", "answer"])
/// # => tells model to emit <thinking>...</thinking><answer>...</answer>
/// ```
#[pyfunction(name = "xml_format_instructions")]
fn xml_format_instructions_py(tags: Vec<String>) -> String {
    let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
    xml_format_instructions(&tag_refs)
}

/// Prompt snippet for ReAct text-mode agents. Lists tools and the
/// Thought/Action/Action Input/Final Answer grammar. Each entry in
/// `tools` should be a string like `"name: description"`.
///
/// ```python
/// from litgraph.parsers import react_format_instructions
/// react_format_instructions([
///     "get_weather: fetch the weather for a city",
///     "web_search: search the web for a query",
/// ])
/// ```
#[pyfunction(name = "react_format_instructions")]
fn react_format_instructions_py(tools: Vec<String>) -> String {
    let tool_refs: Vec<&str> = tools.iter().map(|s| s.as_str()).collect();
    react_format_instructions(&tool_refs)
}

/// Send raw output + parse error + format hint to a fixer LLM. Returns
/// the model's correction. One call, no parse loop.
///
/// `model` accepts any chat model (OpenAIChat, AnthropicChat, etc).
/// `format_instructions` may be empty.
///
/// ```python
/// from litgraph.parsers import fix_with_llm
/// from litgraph.providers import OpenAIChat
/// fixer = OpenAIChat(api_key=..., model="gpt-4o-mini")
/// corrected = fix_with_llm(
///     raw='{"x": 1,}',
///     error="trailing comma",
///     instructions="valid JSON object",
///     model=fixer,
/// )
/// ```
#[pyfunction(name = "fix_with_llm")]
#[pyo3(signature = (raw, error, instructions, model))]
fn fix_with_llm_py(
    py: Python<'_>,
    raw: String,
    error: String,
    instructions: String,
    model: Py<PyAny>,
) -> PyResult<String> {
    let chat_model: Arc<dyn ChatModel> = crate::agents::extract_chat_model(model.bind(py))?;
    let opts = ChatOptions::default();
    py.allow_threads(|| {
        block_on_compat(async move {
            core_fix(&raw, &error, &instructions, chat_model, &opts).await
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

/// JSON-specialized parse-with-retry. Try `json.loads(raw)`; on fail,
/// ask the fixer model to repair, re-parse, repeat. Returns the parsed
/// dict / list / scalar, or raises on final failure.
///
/// ```python
/// from litgraph.parsers import parse_json_with_retry
/// from litgraph.providers import OpenAIChat
/// fixer = OpenAIChat(api_key=..., model="gpt-4o-mini")
/// data = parse_json_with_retry(
///     raw='{"x": 1,}',  # malformed
///     model=fixer,
///     schema_hint="{'x': int}",
///     max_retries=2,
/// )
/// ```
#[pyfunction(name = "parse_json_with_retry")]
#[pyo3(signature = (raw, model, schema_hint="", max_retries=1))]
fn parse_json_with_retry_py<'py>(
    py: Python<'py>,
    raw: String,
    model: Py<PyAny>,
    schema_hint: &str,
    max_retries: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let chat_model: Arc<dyn ChatModel> = crate::agents::extract_chat_model(model.bind(py))?;
    let opts = ChatOptions::default();
    let instructions = if schema_hint.is_empty() {
        "valid JSON (object, array, or scalar)".to_string()
    } else {
        format!("valid JSON matching this schema: {schema_hint}")
    };
    let parse =
        |s: &str| serde_json::from_str::<serde_json::Value>(s).map_err(Error::from);
    let value = py.allow_threads(|| {
        block_on_compat(async move {
            core_parse_retry(raw, parse, chat_model, &opts, &instructions, max_retries).await
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    json_to_py(py, &value)
}

/// Best-effort parse of a partial JSON string. Auto-closes unclosed
/// strings / braces / brackets at EOF so progressive UIs can render
/// the accumulating object as tokens arrive. Returns `None` if the
/// buffer has no recognizable root or repair still fails.
///
/// ```python
/// from litgraph.parsers import parse_partial_json
/// acc = ""
/// for token in stream_tokens:
///     acc += token
///     partial = parse_partial_json(acc)
///     if partial is not None:
///         render_ui(partial)
/// ```
///
/// Monotonic-growth invariant: as the buffer grows, the parse result
/// keeps accumulating keys (no UI flicker from disappearing fields).
#[pyfunction(name = "parse_partial_json")]
fn parse_partial_json_py<'py>(
    py: Python<'py>,
    text: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match core_parse_partial(text) {
        Some(v) => Ok(Some(json_to_py(py, &v)?)),
        None => Ok(None),
    }
}

/// Like `parse_partial_json` but returns the REPAIRED JSON string
/// instead of the parsed value. Useful for inspecting what the repair
/// produced or forwarding to a different JSON library.
#[pyfunction(name = "repair_partial_json")]
fn repair_partial_json_py(text: &str) -> Option<String> {
    core_repair_partial(text)
}
