//! Python bindings for `litgraph.prompts.ChatPromptTemplate` — Jinja-style
//! `{{ var }}` interpolation, role-tagged parts, and `MessagesPlaceholder`
//! slots for splicing in pre-built message histories.
//!
//! Direct LangChain `ChatPromptTemplate` parity, including the placeholder
//! pattern: declare a slot at template-build time, fill it with messages
//! at format time.

use std::sync::Mutex;

use litgraph_core::{ChatPromptTemplate, FewShotChatPromptTemplate, Message, Role};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::graph::py_dict_to_json;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChatPromptTemplate>()?;
    m.add_class::<PyPromptValue>()?;
    m.add_class::<PyFewShotChatPromptTemplate>()?;
    Ok(())
}

fn role_from_str(s: &str) -> PyResult<Role> {
    match s.to_ascii_lowercase().as_str() {
        "system" => Ok(Role::System),
        "user" | "human" => Ok(Role::User),
        "assistant" | "ai" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        other => Err(PyValueError::new_err(format!(
            "unknown role: '{other}' (expected system / user / assistant / tool)"
        ))),
    }
}

fn role_to_str(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn parse_message_dict(d: &Bound<'_, PyDict>) -> PyResult<Message> {
    let role: String = d
        .get_item("role")?
        .ok_or_else(|| PyValueError::new_err("message dict missing 'role'"))?
        .extract()?;
    let content: String = d
        .get_item("content")?
        .ok_or_else(|| PyValueError::new_err("message dict missing 'content'"))?
        .extract()?;
    Ok(Message::text(role_from_str(&role)?, content))
}

/// `ChatPromptTemplate` — declare role-tagged Jinja templates + named
/// `MessagesPlaceholder` slots; format with a vars dict + per-slot message
/// lists at runtime.
///
/// ```python
/// from litgraph.prompts import ChatPromptTemplate
/// tmpl = ChatPromptTemplate.from_messages([
///     ("system", "You are {{ persona }}."),
///     ("placeholder", "history"),
///     ("user", "{{ input }}"),
/// ])
/// pv = tmpl.format({"persona": "terse", "input": "hi"})
/// pv = pv.with_placeholder("history", [
///     {"role": "user", "content": "previous turn"},
///     {"role": "assistant", "content": "response"},
/// ])
/// messages = pv.to_messages()  # list[dict]
/// ```
#[pyclass(name = "ChatPromptTemplate", module = "litgraph.prompts")]
#[derive(Clone)]
pub struct PyChatPromptTemplate {
    inner: ChatPromptTemplate,
}

#[pymethods]
impl PyChatPromptTemplate {
    /// Build from a list of (role, content) tuples. Roles: `system` /
    /// `user` (alias `human`) / `assistant` (alias `ai`) / `placeholder` /
    /// `optional_placeholder`. For placeholder rows, the second item is the
    /// slot name (NOT a Jinja template).
    #[staticmethod]
    fn from_messages(messages: Bound<'_, PyList>) -> PyResult<Self> {
        let mut tmpl = ChatPromptTemplate::new();
        for item in messages.iter() {
            let pair: Bound<'_, PyTuple> = item.downcast_into::<PyTuple>().map_err(|_| {
                PyValueError::new_err(
                    "each message must be a (role, content) tuple",
                )
            })?;
            if pair.len() != 2 {
                return Err(PyValueError::new_err(
                    "each message must be a (role, content) 2-tuple",
                ));
            }
            let role: String = pair.get_item(0)?.extract()?;
            let content: String = pair.get_item(1)?.extract()?;
            match role.to_ascii_lowercase().as_str() {
                "placeholder" | "messages_placeholder" => {
                    tmpl = tmpl.placeholder(content);
                }
                "optional_placeholder" => {
                    tmpl = tmpl.optional_placeholder(content);
                }
                _ => {
                    let r = role_from_str(&role)?;
                    tmpl = tmpl.push(r, content);
                }
            }
        }
        Ok(Self { inner: tmpl })
    }

    /// Empty constructor — chain `.system()` / `.user()` / `.assistant()` /
    /// `.placeholder()` to build up.
    #[new]
    fn new() -> Self {
        Self { inner: ChatPromptTemplate::new() }
    }

    fn system(&self, template: String) -> Self {
        Self { inner: self.inner.clone().system(template) }
    }
    fn user(&self, template: String) -> Self {
        Self { inner: self.inner.clone().user(template) }
    }
    fn assistant(&self, template: String) -> Self {
        Self { inner: self.inner.clone().assistant(template) }
    }
    fn placeholder(&self, name: String) -> Self {
        Self { inner: self.inner.clone().placeholder(name) }
    }
    fn optional_placeholder(&self, name: String) -> Self {
        Self { inner: self.inner.clone().optional_placeholder(name) }
    }

    /// Names of declared placeholder slots, in order.
    fn placeholder_names(&self) -> Vec<String> {
        self.inner.placeholder_names()
    }

    /// Format the template. `vars` is a dict of variables for Jinja
    /// interpolation. Returns a `PromptValue` whose placeholder slots are
    /// still empty — fill via `with_placeholder()` then `to_messages()`.
    #[pyo3(signature = (vars=None))]
    fn format(&self, vars: Option<Bound<'_, PyDict>>) -> PyResult<PyPromptValue> {
        let v = match vars {
            Some(d) => py_dict_to_json(&d)?,
            None => serde_json::Value::Object(Default::default()),
        };
        let pv = self.inner.format(&v).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPromptValue { inner: Mutex::new(Some(pv)) })
    }

    fn __repr__(&self) -> String {
        format!(
            "ChatPromptTemplate(placeholders={:?})",
            self.inner.placeholder_names()
        )
    }
}

/// A formatted prompt — has its templated parts rendered, but placeholder
/// slots may still be empty. Call `with_placeholder(name, msgs)` once per
/// declared slot, then `to_messages()` to get the final `list[dict]`.
#[pyclass(name = "PromptValue", module = "litgraph.prompts")]
pub struct PyPromptValue {
    /// `Mutex<Option>` because `with_placeholder` consumes-and-returns
    /// the inner `PromptValue` (Rust API), but PyO3 methods take `&self`.
    inner: Mutex<Option<litgraph_core::PromptValue>>,
}

#[pymethods]
impl PyPromptValue {
    /// Splice `messages` into the named placeholder slot. `messages` is a
    /// list of `{"role": ..., "content": ...}` dicts. Errors if the slot
    /// doesn't exist (silent drops are the #1 prompt template footgun).
    fn with_placeholder<'py>(
        slf: Py<Self>,
        py: Python<'py>,
        name: String,
        messages: Bound<'py, PyList>,
    ) -> PyResult<Py<Self>> {
        let msgs: Vec<Message> = messages
            .iter()
            .map(|item| {
                let d = item.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("each message must be a dict")
                })?;
                parse_message_dict(d)
            })
            .collect::<PyResult<_>>()?;
        let bound = slf.bind(py);
        let cell = bound.borrow_mut();
        {
            let mut guard = cell.inner.lock().unwrap();
            let pv = guard.take().ok_or_else(|| {
                PyRuntimeError::new_err("PromptValue already consumed")
            })?;
            let new_pv = pv
                .with_placeholder(&name, msgs)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            *guard = Some(new_pv);
        }
        drop(cell);
        Ok(slf)
    }

    /// Return the formatted messages as a list of `{role, content}` dicts.
    /// Errors if any required placeholder was never filled.
    fn to_messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let guard = self.inner.lock().unwrap();
        let pv = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("PromptValue already consumed"))?
            .clone();
        let msgs = pv
            .into_messages()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let out = PyList::empty_bound(py);
        for m in msgs {
            let d = PyDict::new_bound(py);
            d.set_item("role", role_to_str(m.role))?;
            d.set_item("content", m.text_content())?;
            out.append(d)?;
        }
        Ok(out)
    }
}

// ---------- FewShotChatPromptTemplate (iter 92) ----------

/// Few-shot prompting — render N example (input, output) pairs ahead of the
/// user's actual question to teach the model a format/style. Direct LangChain
/// `FewShotChatMessagePromptTemplate` parity.
///
/// Layout (rendered output):
/// ```
/// [system_prefix?]
/// [example 0 rendered via example_prompt]
/// ...
/// [example N-1 rendered via example_prompt]
/// [input_prompt rendered with vars]
/// ```
///
/// ```python
/// from litgraph.prompts import ChatPromptTemplate, FewShotChatPromptTemplate
/// example_prompt = ChatPromptTemplate.from_messages([
///     ("user", "{{ q }}"),
///     ("assistant", "{{ a }}"),
/// ])
/// input_prompt = ChatPromptTemplate.from_messages([("user", "{{ input }}")])
/// few_shot = FewShotChatPromptTemplate(
///     example_prompt=example_prompt,
///     examples=[
///         {"q": "hello", "a": "bonjour"},
///         {"q": "thank you", "a": "merci"},
///     ],
///     input_prompt=input_prompt,
///     system_prefix="You translate English to French.",
/// )
/// messages = few_shot.format({"input": "good night"})
/// ```
#[pyclass(name = "FewShotChatPromptTemplate", module = "litgraph.prompts")]
pub struct PyFewShotChatPromptTemplate {
    inner: FewShotChatPromptTemplate,
}

#[pymethods]
impl PyFewShotChatPromptTemplate {
    #[new]
    #[pyo3(signature = (example_prompt, examples, input_prompt, system_prefix=None))]
    fn new(
        example_prompt: PyRef<'_, PyChatPromptTemplate>,
        examples: Bound<'_, PyList>,
        input_prompt: PyRef<'_, PyChatPromptTemplate>,
        system_prefix: Option<String>,
    ) -> PyResult<Self> {
        let mut ex_vec: Vec<serde_json::Value> = Vec::with_capacity(examples.len());
        for item in examples.iter() {
            let d = item.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("each example must be a dict")
            })?;
            ex_vec.push(py_dict_to_json(d)?);
        }
        let mut tmpl = FewShotChatPromptTemplate::new()
            .with_example_prompt(example_prompt.inner.clone())
            .with_examples(ex_vec)
            .with_input_prompt(input_prompt.inner.clone());
        if let Some(p) = system_prefix {
            tmpl = tmpl.with_system_prefix(p);
        }
        Ok(Self { inner: tmpl })
    }

    /// Render to `list[{role, content}]`. `vars` is the dict for the
    /// final input prompt; each example carries its own vars (set at
    /// construction).
    #[pyo3(signature = (vars=None))]
    fn format<'py>(
        &self,
        py: Python<'py>,
        vars: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let v = match vars {
            Some(d) => py_dict_to_json(&d)?,
            None => serde_json::Value::Object(Default::default()),
        };
        let msgs = self.inner.format(&v).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let out = PyList::empty_bound(py);
        for m in msgs {
            let d = PyDict::new_bound(py);
            d.set_item("role", role_to_str(m.role))?;
            d.set_item("content", m.text_content())?;
            out.append(d)?;
        }
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!("FewShotChatPromptTemplate(examples={})", self.inner.examples.len())
    }
}
