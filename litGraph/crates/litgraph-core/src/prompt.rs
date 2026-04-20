use std::collections::HashMap;

use minijinja::Environment;
use serde_json::Value;

use crate::{Error, Message, Result, Role};

/// A rendered prompt — ready to send to a `ChatModel`.
#[derive(Debug, Clone)]
pub struct PromptValue {
    pub messages: Vec<Message>,
}

/// Role-tagged template parts with `{{ var }}` Jinja-style interpolation.
#[derive(Debug, Clone)]
pub struct ChatPromptTemplate {
    parts: Vec<(Role, String)>,
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
            parts: messages.into_iter().map(|(r, t)| (r, t.into())).collect(),
            partials: HashMap::new(),
        }
    }

    pub fn push(mut self, role: Role, template: impl Into<String>) -> Self {
        self.parts.push((role, template.into()));
        self
    }

    /// Bind a subset of variables and return a new template.
    pub fn partial<K: Into<String>>(mut self, key: K, value: Value) -> Self {
        self.partials.insert(key.into(), value);
        self
    }

    pub fn format(&self, vars: &Value) -> Result<PromptValue> {
        let env = Environment::new();
        let mut merged = self.partials.clone();
        if let Value::Object(m) = vars {
            for (k, v) in m {
                merged.insert(k.clone(), v.clone());
            }
        }
        let ctx = Value::Object(merged.into_iter().collect());

        let mut rendered = Vec::with_capacity(self.parts.len());
        for (role, tmpl) in &self.parts {
            let t = env.template_from_str(tmpl).map_err(Error::from)?;
            let text = t.render(&ctx).map_err(Error::from)?;
            rendered.push(Message::text(*role, text));
        }
        Ok(PromptValue { messages: rendered })
    }
}

impl Default for ChatPromptTemplate {
    fn default() -> Self { Self::new() }
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
        assert_eq!(pv.messages.len(), 2);
        assert_eq!(pv.messages[0].text_content(), "You are terse.");
        assert_eq!(pv.messages[1].text_content(), "Say hi to Amiya.");
    }

    #[test]
    fn partial_binds_var() {
        let tmpl = ChatPromptTemplate::from_messages([(Role::User, "{{ a }}+{{ b }}")])
            .partial("a", json!("hello"));
        let pv = tmpl.format(&json!({ "b": "world" })).unwrap();
        assert_eq!(pv.messages[0].text_content(), "hello+world");
    }
}
