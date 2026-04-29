//! Python bindings for `litgraph.prompts.ChatPromptTemplate` — Jinja-style
//! `{{ var }}` interpolation, role-tagged parts, and `MessagesPlaceholder`
//! slots for splicing in pre-built message histories.
//!
//! Direct LangChain `ChatPromptTemplate` parity, including the placeholder
//! pattern: declare a slot at template-build time, fill it with messages
//! at format time.

use std::sync::Mutex;

use std::sync::Arc;

use litgraph_core::{
    ChatPromptTemplate, FewShotChatPromptTemplate, LengthBasedExampleSelector, Message, Role,
    SemanticSimilarityExampleSelector, Skill, SystemPromptBuilder, load_agents_md,
    load_skills_dir,
};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::graph::py_dict_to_json;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChatPromptTemplate>()?;
    m.add_class::<PyPromptValue>()?;
    m.add_class::<PyFewShotChatPromptTemplate>()?;
    m.add_class::<PySemanticSimilarityExampleSelector>()?;
    m.add_class::<PyLengthBasedExampleSelector>()?;
    m.add_class::<PySkill>()?;
    m.add_class::<PySystemPromptBuilder>()?;
    m.add_function(pyo3::wrap_pyfunction!(load_agents_md_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(load_skills_dir_py, m)?)?;
    Ok(())
}

/// Read a single AGENTS.md (or CLAUDE.md / system-prompt) file. Returns the
/// file contents as a string, or `None` if the file does not exist.
#[pyfunction(name = "load_agents_md")]
#[pyo3(signature = (path))]
fn load_agents_md_py(path: String) -> PyResult<Option<String>> {
    load_agents_md(&path).map_err(|e| PyIOError::new_err(e.to_string()))
}

/// Read every `*.md` file in `dir` as a `Skill`. Sorted by filename. Returns
/// an empty list if the directory does not exist.
#[pyfunction(name = "load_skills_dir")]
#[pyo3(signature = (dir))]
fn load_skills_dir_py(dir: String) -> PyResult<Vec<PySkill>> {
    let skills = load_skills_dir(&dir).map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(skills.into_iter().map(|s| PySkill { inner: s }).collect())
}

/// One named skill loaded from disk. Construct directly or via
/// `litgraph.prompts.load_skills_dir(...)`.
#[pyclass(name = "Skill", module = "litgraph.prompts")]
#[derive(Clone)]
pub struct PySkill {
    pub(crate) inner: Skill,
}

#[pymethods]
impl PySkill {
    #[new]
    #[pyo3(signature = (name, description, content, source=None))]
    fn new(name: String, description: String, content: String, source: Option<String>) -> Self {
        Self {
            inner: Skill {
                name,
                description,
                content,
                source: source.map(std::path::PathBuf::from),
            },
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn description(&self) -> String {
        self.inner.description.clone()
    }
    #[getter]
    fn content(&self) -> String {
        self.inner.content.clone()
    }
    #[getter]
    fn source(&self) -> Option<String> {
        self.inner
            .source
            .as_ref()
            .and_then(|p| p.to_str().map(|s| s.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("Skill(name={:?}, description={:?})", self.inner.name, self.inner.description)
    }
}

/// Assemble a structured system prompt from a base instruction, an optional
/// AGENTS.md memory body, a list of `Skill`s, and arbitrary extra named
/// sections. Output is plain Markdown (`## Section` headers).
#[pyclass(name = "SystemPromptBuilder", module = "litgraph.prompts")]
pub struct PySystemPromptBuilder {
    inner: SystemPromptBuilder,
}

#[pymethods]
impl PySystemPromptBuilder {
    #[new]
    fn new(base: String) -> Self {
        Self {
            inner: SystemPromptBuilder::new(base),
        }
    }

    fn with_agents_md<'py>(mut slf: PyRefMut<'py, Self>, body: String) -> PyRefMut<'py, Self> {
        let owned = std::mem::take(&mut slf.inner);
        slf.inner = owned.with_agents_md(body);
        slf
    }

    fn with_skill<'py>(
        mut slf: PyRefMut<'py, Self>,
        skill: PyRef<'_, PySkill>,
    ) -> PyRefMut<'py, Self> {
        let owned = std::mem::take(&mut slf.inner);
        slf.inner = owned.with_skill(skill.inner.clone());
        slf
    }

    fn with_skills<'py>(
        mut slf: PyRefMut<'py, Self>,
        skills: Bound<'_, PyList>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut collected: Vec<Skill> = Vec::with_capacity(skills.len());
        for item in skills.iter() {
            let s = item.extract::<PyRef<PySkill>>()?;
            collected.push(s.inner.clone());
        }
        let owned = std::mem::take(&mut slf.inner);
        slf.inner = owned.with_skills(collected);
        Ok(slf)
    }

    fn with_section<'py>(
        mut slf: PyRefMut<'py, Self>,
        heading: String,
        body: String,
    ) -> PyRefMut<'py, Self> {
        let owned = std::mem::take(&mut slf.inner);
        slf.inner = owned.with_section(heading, body);
        slf
    }

    fn build(&self) -> String {
        self.inner.build()
    }

    fn __repr__(&self) -> String {
        let s = self.inner.build();
        let preview: String = s.chars().take(60).collect();
        format!("SystemPromptBuilder(preview={:?})", preview)
    }
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

    /// Parse a template from a JSON string. Schema:
    /// ```json
    /// {
    ///   "messages": [
    ///     {"role": "system", "template": "You are {{ persona }}."},
    ///     {"role": "user", "template": "{{ question }}"},
    ///     {"role": "placeholder", "name": "history", "optional": true}
    ///   ],
    ///   "partials": {"persona": "terse"}
    /// }
    /// ```
    /// Roles: `system`, `user`, `assistant`, `placeholder`.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let inner = ChatPromptTemplate::from_json(text)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Parse from a Python dict (e.g. loaded from a YAML file via PyYAML).
    /// Avoids forcing litgraph to depend on a YAML parser — callers stay
    /// free to use whatever YAML library they prefer.
    ///
    /// ```python
    /// import yaml
    /// from litgraph.prompts import ChatPromptTemplate
    /// with open("my_prompt.yaml") as f:
    ///     tmpl = ChatPromptTemplate.from_dict(yaml.safe_load(f))
    /// ```
    #[staticmethod]
    fn from_dict(spec: Bound<'_, PyDict>) -> PyResult<Self> {
        let v = py_dict_to_json(&spec)?;
        let inner = ChatPromptTemplate::from_value(&v)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Serialize the template to a pretty-printed JSON string. Round-trips
    /// with `from_json`.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// As `to_json` but returns a Python dict (useful for then dumping
    /// to YAML via PyYAML).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.to_value();
        crate::graph::json_to_py(py, &v)
    }

    /// Append `other`'s parts onto this template; returns a new template.
    /// Partials merge with `other` overriding `self` on key collision.
    /// Common pattern: layer prompts loaded from separate files —
    /// `base + role_specific + task_specific`.
    fn extend(&self, other: PyRef<'_, PyChatPromptTemplate>) -> Self {
        Self { inner: self.inner.clone().extend(other.inner.clone()) }
    }

    /// Same as `extend` but mirrors Python's `+` operator semantics
    /// (returns a new combined template; doesn't mutate either input).
    fn __add__(&self, other: PyRef<'_, PyChatPromptTemplate>) -> Self {
        Self { inner: self.inner.concat(&other.inner) }
    }

    /// Number of parts (templated messages + placeholders).
    fn __len__(&self) -> usize { self.inner.len() }
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

/// `SemanticSimilarityExampleSelector` — embedding-based selector for FewShot.
/// Embed the candidate pool once (lazy on first call), embed the query
/// per call, return top-K examples by cosine similarity.
///
/// ```python
/// from litgraph.prompts import SemanticSimilarityExampleSelector, FewShotChatPromptTemplate
/// from litgraph.embeddings import OpenAIEmbeddings
///
/// pool = [
///     {"input": "fix borrow checker", "output": "use clone or rework lifetimes"},
///     {"input": "css flexbox alignment", "output": "use justify-content"},
///     {"input": "python list comprehension", "output": "use [x for x in ...]"},
///     # ... 50 more ...
/// ]
/// embedder = OpenAIEmbeddings(api_key=..., model="text-embedding-3-small")
/// selector = SemanticSimilarityExampleSelector(pool, embedder, key_field="input")
///
/// # Per query: pick the 3 most-relevant examples + render.
/// picked = selector.select("how do I fix a borrow error?", k=3)
/// few_shot = FewShotChatPromptTemplate(...).with_examples(picked)
/// msgs = few_shot.format({"question": "..."})
/// ```
#[pyclass(name = "SemanticSimilarityExampleSelector", module = "litgraph.prompts")]
pub struct PySemanticSimilarityExampleSelector {
    inner: Arc<SemanticSimilarityExampleSelector>,
}

#[pymethods]
impl PySemanticSimilarityExampleSelector {
    /// `pool` is a list of dicts (each MUST contain a string field named
    /// `key_field`). `embeddings` is any litGraph Embeddings instance.
    #[new]
    #[pyo3(signature = (pool, embeddings, key_field="input"))]
    fn new(
        pool: Bound<'_, PyList>,
        embeddings: Bound<'_, PyAny>,
        key_field: &str,
    ) -> PyResult<Self> {
        let mut pool_vec = Vec::with_capacity(pool.len());
        for item in pool.iter() {
            let d: Bound<'_, PyDict> = item.downcast_into().map_err(|_| {
                PyValueError::new_err("each pool entry must be a dict")
            })?;
            let json_val = py_dict_to_json(&d)?;
            pool_vec.push(json_val);
        }
        let embedder = crate::embeddings::extract_embeddings(&embeddings)?;
        let inner = SemanticSimilarityExampleSelector::new(pool_vec, embedder, key_field);
        Ok(Self { inner: Arc::new(inner) })
    }

    /// Top-K most-similar pool examples to `query`. Returns a Python list
    /// of dicts (the original pool entries verbatim). Empty list if pool
    /// is empty or k=0.
    fn select<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let inner = self.inner.clone();
        let q = query.to_string();
        let picked = py.allow_threads(|| {
            crate::runtime::block_on_compat(async move { inner.select(&q, k).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let out = PyList::empty_bound(py);
        for v in picked {
            let d = json_to_py_dict(py, &v)?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// Pre-compute pool embeddings. Idempotent; eliminates first-`select`
    /// latency.
    fn warmup(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            crate::runtime::block_on_compat(async move { inner.warmup().await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn pool_size(&self) -> usize {
        self.inner.pool_size()
    }

    fn __repr__(&self) -> String {
        format!("SemanticSimilarityExampleSelector(pool_size={})", self.inner.pool_size())
    }
}

/// serde_json::Value → Python dict (only top-level Object case used here).
fn json_to_py_dict<'py>(
    py: Python<'py>,
    v: &serde_json::Value,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    if let serde_json::Value::Object(map) = v {
        for (k, val) in map {
            d.set_item(k, json_to_pyobj(py, val)?)?;
        }
    }
    Ok(d)
}

fn json_to_pyobj<'py>(py: Python<'py>, v: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use serde_json::Value as J;
    Ok(match v {
        J::Null => py.None(),
        J::Bool(b) => b.into_py(py),
        J::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py(py)
            } else if let Some(f) = n.as_f64() {
                f.into_py(py)
            } else {
                py.None()
            }
        }
        J::String(s) => s.clone().into_py(py),
        J::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_pyobj(py, item)?)?;
            }
            list.into_py(py)
        }
        J::Object(_) => json_to_py_dict(py, v)?.into_py(py),
    })
}

/// `LengthBasedExampleSelector` — token-budget-greedy example picker.
/// Walks the pool in order and includes each example IFF the running
/// total stays under `max_tokens`. Stops at the first overflow (does NOT
/// skip-and-continue — preserves ordering).
///
/// ```python
/// from litgraph.prompts import LengthBasedExampleSelector
///
/// pool = [{"input": "...", "output": "..."}, ...]  # ordered by priority
/// # Counter is any callable: text -> int. For OpenAI, wrap tiktoken.
/// def count(text):
///     return len(text) // 4   # rough estimate
///
/// sel = LengthBasedExampleSelector(
///     pool, max_tokens=2048,
///     fields=["input", "output"], counter=count,
/// )
/// picked = sel.select()                         # default budget
/// picked = sel.select_with_budget(max_tokens=512)  # override per call
/// ```
#[pyclass(name = "LengthBasedExampleSelector", module = "litgraph.prompts")]
pub struct PyLengthBasedExampleSelector {
    inner: LengthBasedExampleSelector,
}

#[pymethods]
impl PyLengthBasedExampleSelector {
    #[new]
    #[pyo3(signature = (pool, max_tokens, fields=None, counter=None))]
    fn new(
        pool: Bound<'_, PyList>,
        max_tokens: usize,
        fields: Option<Vec<String>>,
        counter: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut pool_vec = Vec::with_capacity(pool.len());
        for item in pool.iter() {
            let d: Bound<'_, PyDict> = item.downcast_into().map_err(|_| {
                PyValueError::new_err("each pool entry must be a dict")
            })?;
            pool_vec.push(py_dict_to_json(&d)?);
        }
        let fields = fields.unwrap_or_else(|| vec!["input".into(), "output".into()]);
        let count_fn: Arc<dyn Fn(&str) -> usize + Send + Sync> = match counter {
            Some(cb) => {
                let cb = Arc::new(cb);
                Arc::new(move |text: &str| {
                    Python::with_gil(|py| {
                        match cb.call1(py, (text,)) {
                            Ok(v) => v.extract::<i64>(py).map(|n| n.max(0) as usize).unwrap_or(0),
                            // Fall back to char/4 estimate if callback errors.
                            Err(_) => text.len() / 4,
                        }
                    })
                })
            }
            None => Arc::new(|t: &str| t.len() / 4),
        };
        Ok(Self {
            inner: LengthBasedExampleSelector::new(pool_vec, max_tokens, fields, count_fn),
        })
    }

    fn select<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let picked = self.inner.select();
        let out = PyList::empty_bound(py);
        for v in picked {
            out.append(json_to_py_dict(py, &v)?)?;
        }
        Ok(out)
    }

    fn select_with_budget<'py>(
        &self,
        py: Python<'py>,
        max_tokens: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let picked = self.inner.select_with_budget(max_tokens);
        let out = PyList::empty_bound(py);
        for v in picked {
            out.append(json_to_py_dict(py, &v)?)?;
        }
        Ok(out)
    }

    fn pool_size(&self) -> usize {
        self.inner.pool_size()
    }

    fn __repr__(&self) -> String {
        format!(
            "LengthBasedExampleSelector(pool_size={}, max_tokens={})",
            self.inner.pool_size(),
            self.inner.max_tokens
        )
    }
}
