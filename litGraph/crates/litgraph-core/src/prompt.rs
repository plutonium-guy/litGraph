//! ChatPromptTemplate — Jinja-style `{{ var }}` interpolation, role-tagged
//! parts, and `MessagesPlaceholder` slots for injecting pre-built message
//! histories (chat memory, few-shot examples, retrieved context).
//!
//! Direct LangChain `ChatPromptTemplate` parity. The canonical pattern:
//!
//! ```ignore
//! let prompt = ChatPromptTemplate::new()
//!     .system("You are {{ persona }}.")
//!     .placeholder("history")
//!     .user("{{ input }}");
//! let messages = prompt.format(&json!({
//!     "persona": "a meticulous engineer",
//!     "input": "Add error handling.",
//! })).unwrap()
//!     .with_placeholder("history", vec![Message::user("Hi"), Message::assistant("Hello")])
//!     .into_messages();
//! ```

use std::collections::HashMap;

use minijinja::{Environment, UndefinedBehavior};
use serde_json::Value;

use crate::{Error, Message, Result, Role};

/// One part of a `ChatPromptTemplate`. Either a Jinja-rendered message with
/// a fixed role, OR a placeholder slot to be filled with caller-supplied
/// pre-built messages at render time.
#[derive(Debug, Clone)]
enum ChatPromptPart {
    Templated { role: Role, template: String },
    Placeholder { name: String, optional: bool },
}

/// A rendered prompt — ready to send to a `ChatModel`. Placeholder slots
/// are still empty until you call `with_placeholder()` to fill them.
#[derive(Debug, Clone)]
pub struct PromptValue {
    /// `Some(Message)` for rendered templated parts; `None` for unfilled
    /// placeholder slots (their position is preserved so a later
    /// `with_placeholder` call can splice messages into the right spot).
    slots: Vec<PromptSlot>,
}

#[derive(Debug, Clone)]
enum PromptSlot {
    Filled(Message),
    Empty { name: String, optional: bool },
}

impl PromptValue {
    /// Splice `messages` into the placeholder named `name`. Returns Err
    /// if no such placeholder exists. Calling on a non-existent name is
    /// noisy by design — silent drops are the #1 prompt-template footgun.
    pub fn with_placeholder(
        mut self,
        name: &str,
        messages: Vec<Message>,
    ) -> Result<Self> {
        let mut found = false;
        let mut new_slots: Vec<PromptSlot> = Vec::with_capacity(self.slots.len() + messages.len());
        for slot in self.slots.drain(..) {
            match slot {
                PromptSlot::Empty { name: n, optional: _ } if n == name => {
                    found = true;
                    for m in &messages {
                        new_slots.push(PromptSlot::Filled(m.clone()));
                    }
                    // Drop the empty slot — it's been filled.
                }
                other => new_slots.push(other),
            }
        }
        if !found {
            return Err(Error::other(format!(
                "ChatPromptTemplate placeholder `{name}` not found (or already filled)"
            )));
        }
        self.slots = new_slots;
        Ok(self)
    }

    /// Convert to `Vec<Message>`. Empty optional placeholders are dropped
    /// silently; empty REQUIRED placeholders error so callers can't ship
    /// half-formed prompts to providers.
    pub fn into_messages(self) -> Result<Vec<Message>> {
        let mut out = Vec::with_capacity(self.slots.len());
        for slot in self.slots {
            match slot {
                PromptSlot::Filled(m) => out.push(m),
                PromptSlot::Empty { name, optional: true } => {
                    // Optional: silently skip.
                    let _ = name;
                }
                PromptSlot::Empty { name, optional: false } => {
                    return Err(Error::other(format!(
                        "ChatPromptTemplate required placeholder `{name}` was never filled"
                    )));
                }
            }
        }
        Ok(out)
    }

    /// Backwards-compat accessor for code that doesn't use placeholders —
    /// equivalent to `into_messages().unwrap()`. Panics if any required
    /// placeholder is unfilled.
    pub fn messages(&self) -> Vec<Message> {
        self.clone()
            .into_messages()
            .expect("PromptValue::messages() called with unfilled required placeholder")
    }
}

/// Role-tagged template parts with `{{ var }}` Jinja-style interpolation
/// and named `MessagesPlaceholder` slots.
#[derive(Debug, Clone)]
pub struct ChatPromptTemplate {
    parts: Vec<ChatPromptPart>,
    partials: HashMap<String, Value>,
}

impl ChatPromptTemplate {
    pub fn new() -> Self {
        Self { parts: Vec::new(), partials: HashMap::new() }
    }

    pub fn from_messages<I, S>(messages: I) -> Self
    where
        I: IntoIterator<Item = (Role, S)>,
        S: Into<String>,
    {
        Self {
            parts: messages
                .into_iter()
                .map(|(r, t)| ChatPromptPart::Templated { role: r, template: t.into() })
                .collect(),
            partials: HashMap::new(),
        }
    }

    pub fn push(mut self, role: Role, template: impl Into<String>) -> Self {
        self.parts.push(ChatPromptPart::Templated { role, template: template.into() });
        self
    }

    /// Convenience builders — system/user/assistant template parts.
    pub fn system(self, template: impl Into<String>) -> Self {
        self.push(Role::System, template)
    }
    pub fn user(self, template: impl Into<String>) -> Self {
        self.push(Role::User, template)
    }
    pub fn assistant(self, template: impl Into<String>) -> Self {
        self.push(Role::Assistant, template)
    }

    /// Reserve a placeholder slot. Caller fills it via
    /// `PromptValue::with_placeholder(name, msgs)`. Required by default;
    /// use `optional_placeholder` to skip silently when unfilled.
    pub fn placeholder(mut self, name: impl Into<String>) -> Self {
        self.parts.push(ChatPromptPart::Placeholder { name: name.into(), optional: false });
        self
    }

    pub fn optional_placeholder(mut self, name: impl Into<String>) -> Self {
        self.parts.push(ChatPromptPart::Placeholder { name: name.into(), optional: true });
        self
    }

    /// Bind a subset of variables and return a new template.
    pub fn partial<K: Into<String>>(mut self, key: K, value: Value) -> Self {
        self.partials.insert(key.into(), value);
        self
    }

    /// Names of declared placeholder slots, in order.
    pub fn placeholder_names(&self) -> Vec<String> {
        self.parts
            .iter()
            .filter_map(|p| match p {
                ChatPromptPart::Placeholder { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn format(&self, vars: &Value) -> Result<PromptValue> {
        // Strict undefined: rendering `{{ missing }}` errors instead of
        // emitting an empty string. Silent empty interpolation is the #1
        // prompt-corruption bug — agent sees "translate '' to French" and
        // returns garbage. Fail fast, fix the input dict.
        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Strict);
        let mut merged = self.partials.clone();
        if let Value::Object(m) = vars {
            for (k, v) in m {
                merged.insert(k.clone(), v.clone());
            }
        }
        let ctx = Value::Object(merged.into_iter().collect());

        let mut slots = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            match part {
                ChatPromptPart::Templated { role, template } => {
                    let t = env.template_from_str(template).map_err(Error::from)?;
                    let text = t.render(&ctx).map_err(Error::from)?;
                    slots.push(PromptSlot::Filled(Message::text(*role, text)));
                }
                ChatPromptPart::Placeholder { name, optional } => {
                    slots.push(PromptSlot::Empty { name: name.clone(), optional: *optional });
                }
            }
        }
        Ok(PromptValue { slots })
    }
}

impl Default for ChatPromptTemplate {
    fn default() -> Self { Self::new() }
}

/// Few-shot prompting — render N example (input, output) pairs ahead of the
/// user's actual question to teach the model a format/style. Direct LangChain
/// `FewShotChatMessagePromptTemplate` parity.
///
/// # Layout (rendered output)
///
/// ```text
/// [system_prefix?]
/// [examples[0] rendered via example_prompt]
/// [examples[1] rendered via example_prompt]
/// ...
/// [examples[N-1] rendered via example_prompt]
/// [input_prompt rendered with vars]
/// ```
///
/// # Example
///
/// ```ignore
/// let example_prompt = ChatPromptTemplate::new()
///     .user("{{ q }}")
///     .assistant("{{ a }}");
/// let few_shot = FewShotChatPromptTemplate::new()
///     .with_system_prefix("You translate English to French.")
///     .with_example_prompt(example_prompt)
///     .with_examples(vec![
///         json!({"q": "hello", "a": "bonjour"}),
///         json!({"q": "thank you", "a": "merci"}),
///     ])
///     .with_input_prompt(ChatPromptTemplate::new().user("{{ input }}"));
/// let messages = few_shot.format(&json!({"input": "good night"})).unwrap();
/// // → [system, user "hello", assistant "bonjour", user "thank you",
/// //    assistant "merci", user "good night"]
/// ```
#[derive(Debug, Clone, Default)]
pub struct FewShotChatPromptTemplate {
    pub system_prefix: Option<String>,
    pub example_prompt: ChatPromptTemplate,
    pub examples: Vec<Value>,
    pub input_prompt: ChatPromptTemplate,
}

impl FewShotChatPromptTemplate {
    pub fn new() -> Self { Self::default() }

    pub fn with_system_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.system_prefix = Some(prefix.into()); self
    }

    pub fn with_example_prompt(mut self, p: ChatPromptTemplate) -> Self {
        self.example_prompt = p; self
    }

    pub fn with_examples(mut self, examples: Vec<Value>) -> Self {
        self.examples = examples; self
    }

    pub fn with_input_prompt(mut self, p: ChatPromptTemplate) -> Self {
        self.input_prompt = p; self
    }

    /// Render to a flat `Vec<Message>`. Each example is rendered through
    /// `example_prompt` with its own vars; results are concatenated; the
    /// input prompt is appended last with the caller's `vars`.
    ///
    /// Returns Err if any example is missing a variable referenced by
    /// `example_prompt`, or if the input prompt declares a placeholder
    /// (placeholders aren't supported in few-shot context — they'd silently
    /// drop the input slot in unintuitive ways; use the standard
    /// `ChatPromptTemplate.placeholder` for chat memory + history instead).
    pub fn format(&self, vars: &Value) -> Result<Vec<Message>> {
        let mut out: Vec<Message> = Vec::new();
        if let Some(prefix) = &self.system_prefix {
            out.push(Message::system(prefix.clone()));
        }
        for (i, ex) in self.examples.iter().enumerate() {
            let pv = self.example_prompt.format(ex).map_err(|e| {
                Error::other(format!(
                    "few-shot: rendering example {i} failed: {e}"
                ))
            })?;
            // example_prompt cannot use placeholders — would always be unfilled.
            let msgs = pv.into_messages().map_err(|e| {
                Error::other(format!(
                    "few-shot: example {i} declared a placeholder slot — \
                     not supported in example_prompt (use plain templated parts only): {e}"
                ))
            })?;
            out.extend(msgs);
        }
        let input_pv = self.input_prompt.format(vars)?;
        out.extend(input_pv.into_messages().map_err(|e| {
            Error::other(format!(
                "few-shot: input_prompt declared a placeholder slot — \
                 not supported (placeholders fall through unfilled): {e}"
            ))
        })?);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_template() {
        let tmpl = ChatPromptTemplate::from_messages([
            (Role::System, "You are {{ persona }}."),
            (Role::User, "Say hi to {{ name }}."),
        ]);
        let pv = tmpl.format(&json!({ "persona": "terse", "name": "Amiya" })).unwrap();
        let msgs = pv.into_messages().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text_content(), "You are terse.");
        assert_eq!(msgs[1].text_content(), "Say hi to Amiya.");
    }

    #[test]
    fn partial_binds_var() {
        let tmpl = ChatPromptTemplate::from_messages([(Role::User, "{{ a }}+{{ b }}")])
            .partial("a", json!("hello"));
        let pv = tmpl.format(&json!({ "b": "world" })).unwrap();
        assert_eq!(pv.into_messages().unwrap()[0].text_content(), "hello+world");
    }

    #[test]
    fn placeholder_splices_messages_in_correct_position() {
        let tmpl = ChatPromptTemplate::new()
            .system("You are {{ persona }}.")
            .placeholder("history")
            .user("{{ input }}");
        let pv = tmpl.format(&json!({ "persona": "terse", "input": "what's up?" })).unwrap();
        let msgs = pv
            .with_placeholder("history", vec![
                Message::user("hi"),
                Message::assistant("hello"),
            ])
            .unwrap()
            .into_messages()
            .unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[0].text_content(), "You are terse.");
        // History spliced between system and user.
        assert_eq!(msgs[1].role, Role::User);
        assert_eq!(msgs[1].text_content(), "hi");
        assert_eq!(msgs[2].role, Role::Assistant);
        assert_eq!(msgs[2].text_content(), "hello");
        assert_eq!(msgs[3].role, Role::User);
        assert_eq!(msgs[3].text_content(), "what's up?");
    }

    #[test]
    fn required_placeholder_unfilled_errors_at_into_messages() {
        let tmpl = ChatPromptTemplate::new()
            .system("hi")
            .placeholder("history")
            .user("end");
        let pv = tmpl.format(&json!({})).unwrap();
        let res = pv.into_messages();
        assert!(res.is_err());
        let msg = format!("{}", res.unwrap_err());
        assert!(msg.contains("history"), "got: {msg}");
    }

    #[test]
    fn optional_placeholder_unfilled_silently_dropped() {
        let tmpl = ChatPromptTemplate::new()
            .system("sys")
            .optional_placeholder("examples")
            .user("end");
        let msgs = tmpl
            .format(&json!({})).unwrap()
            .into_messages().unwrap();
        // Only system + user; optional placeholder dropped silently.
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text_content(), "sys");
        assert_eq!(msgs[1].text_content(), "end");
    }

    #[test]
    fn with_placeholder_unknown_name_errors() {
        let tmpl = ChatPromptTemplate::new().system("hi").placeholder("history").user("end");
        let pv = tmpl.format(&json!({})).unwrap();
        let res = pv.with_placeholder("not_a_slot", vec![]);
        assert!(res.is_err());
        let msg = format!("{}", res.unwrap_err());
        assert!(msg.contains("not_a_slot"), "got: {msg}");
    }

    #[test]
    fn placeholder_names_returns_declared_slots_in_order() {
        let tmpl = ChatPromptTemplate::new()
            .system("sys")
            .placeholder("history")
            .user("u")
            .optional_placeholder("examples")
            .user("end");
        assert_eq!(tmpl.placeholder_names(), vec!["history", "examples"]);
    }

    #[test]
    fn empty_messages_passed_to_required_placeholder_succeeds() {
        // Filling with [] is valid — the slot was filled, it just doesn't
        // splice anything. (Different from "never filled" which errors.)
        let tmpl = ChatPromptTemplate::new()
            .system("s")
            .placeholder("history")
            .user("u");
        let msgs = tmpl
            .format(&json!({})).unwrap()
            .with_placeholder("history", vec![]).unwrap()
            .into_messages().unwrap();
        assert_eq!(msgs.len(), 2);
    }

    // ---------- FewShotChatPromptTemplate (iter 92) ----------

    #[test]
    fn few_shot_renders_examples_then_input() {
        let example_prompt = ChatPromptTemplate::new()
            .user("{{ q }}")
            .assistant("{{ a }}");
        let few_shot = FewShotChatPromptTemplate::new()
            .with_example_prompt(example_prompt)
            .with_examples(vec![
                json!({"q": "hello", "a": "bonjour"}),
                json!({"q": "thank you", "a": "merci"}),
            ])
            .with_input_prompt(ChatPromptTemplate::new().user("{{ input }}"));
        let msgs = few_shot.format(&json!({"input": "good night"})).unwrap();
        // 2 examples × 2 messages each + 1 input = 5 messages.
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[0].text_content(), "hello");
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[1].text_content(), "bonjour");
        assert_eq!(msgs[2].text_content(), "thank you");
        assert_eq!(msgs[3].text_content(), "merci");
        assert_eq!(msgs[4].role, Role::User);
        assert_eq!(msgs[4].text_content(), "good night");
    }

    #[test]
    fn few_shot_system_prefix_prepended() {
        let example_prompt = ChatPromptTemplate::new().user("{{ q }}").assistant("{{ a }}");
        let few_shot = FewShotChatPromptTemplate::new()
            .with_system_prefix("You translate English to French.")
            .with_example_prompt(example_prompt)
            .with_examples(vec![json!({"q": "hi", "a": "salut"})])
            .with_input_prompt(ChatPromptTemplate::new().user("{{ x }}"));
        let msgs = few_shot.format(&json!({"x": "bye"})).unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[0].text_content(), "You translate English to French.");
    }

    #[test]
    fn few_shot_zero_examples_renders_only_input() {
        let few_shot = FewShotChatPromptTemplate::new()
            .with_example_prompt(ChatPromptTemplate::new().user("{{ q }}"))
            .with_examples(vec![])  // empty
            .with_input_prompt(ChatPromptTemplate::new().user("{{ input }}"));
        let msgs = few_shot.format(&json!({"input": "hi"})).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text_content(), "hi");
    }

    #[test]
    fn few_shot_example_missing_var_errors_with_index() {
        // Example 1 (zero-indexed) is missing the "a" var → error
        // mentions which example failed.
        let example_prompt = ChatPromptTemplate::new().user("{{ q }}").assistant("{{ a }}");
        let few_shot = FewShotChatPromptTemplate::new()
            .with_example_prompt(example_prompt)
            .with_examples(vec![
                json!({"q": "hello", "a": "bonjour"}),
                json!({"q": "thanks"}),  // missing "a"
            ])
            .with_input_prompt(ChatPromptTemplate::new().user("{{ input }}"));
        let res = few_shot.format(&json!({"input": "x"}));
        assert!(res.is_err());
        let msg = format!("{}", res.unwrap_err());
        assert!(msg.contains("example 1"), "error should name the failing example index, got: {msg}");
    }

    #[test]
    fn few_shot_input_prompt_with_placeholder_errors() {
        // Placeholders in the input prompt would always be unfilled — surface
        // that as a clear error rather than silently dropping.
        let few_shot = FewShotChatPromptTemplate::new()
            .with_example_prompt(ChatPromptTemplate::new().user("{{ q }}"))
            .with_examples(vec![json!({"q": "x"})])
            .with_input_prompt(
                ChatPromptTemplate::new()
                    .user("{{ input }}")
                    .placeholder("history"),
            );
        let res = few_shot.format(&json!({"input": "y"}));
        assert!(res.is_err());
        let msg = format!("{}", res.unwrap_err());
        assert!(msg.contains("placeholder"));
    }
}
