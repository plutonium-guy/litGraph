//! Python bindings for token counting + message-history trimming.

use litgraph_core::{Message, Role};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(count_message_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(trim_messages, m)?)?;
    Ok(())
}

/// Count tokens for `text` under `model`. Uses tiktoken-rs for OpenAI models;
/// falls back to `len(text) // 4` for other model names.
#[pyfunction]
fn count_tokens(model: &str, text: &str) -> usize {
    litgraph_tokenizers::count_tokens(model, text)
}

/// Count tokens across a list of message dicts (`{role, content}`). Includes
/// per-message overhead modeled on OpenAI's accounting.
#[pyfunction]
fn count_message_tokens(model: &str, messages: Bound<'_, PyList>) -> PyResult<usize> {
    let msgs = parse_simple_messages(&messages)?;
    Ok(litgraph_tokenizers::count_message_tokens(model, &msgs))
}

/// Drop oldest non-system messages until under `max_tokens`. Always keeps
/// system messages + the last message. Returns the trimmed list of dicts.
#[pyfunction]
fn trim_messages<'py>(
    py: Python<'py>,
    model: &str,
    messages: Bound<'py, PyList>,
    max_tokens: usize,
) -> PyResult<Bound<'py, PyList>> {
    let msgs = parse_simple_messages(&messages)?;
    let trimmed = litgraph_tokenizers::trim_messages(model, &msgs, max_tokens);
    let out = PyList::empty_bound(py);
    for m in trimmed {
        let d = PyDict::new_bound(py);
        d.set_item("role", role_str(m.role))?;
        d.set_item("content", m.text_content())?;
        out.append(d)?;
    }
    Ok(out)
}

fn role_str(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn parse_simple_messages(py_msgs: &Bound<'_, PyList>) -> PyResult<Vec<Message>> {
    let mut out = Vec::with_capacity(py_msgs.len());
    for item in py_msgs.iter() {
        let d: Bound<PyDict> = item.downcast_into()
            .map_err(|_| PyValueError::new_err("message must be dict"))?;
        let role: String = d
            .get_item("role")?
            .ok_or_else(|| PyValueError::new_err("missing 'role'"))?
            .extract()?;
        let content: String = d
            .get_item("content")?
            .ok_or_else(|| PyValueError::new_err("missing 'content'"))?
            .extract()?;
        let role = match role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => return Err(PyValueError::new_err(format!("bad role: {other}"))),
        };
        out.push(Message::text(role, content));
    }
    Ok(out)
}
