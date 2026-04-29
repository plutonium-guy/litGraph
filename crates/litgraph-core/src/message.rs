use serde::{Deserialize, Serialize};

use crate::tool::ToolCall;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image { source: ImageSource },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Anthropic prompt-cache breakpoint marker. When `true`, providers that
    /// support prompt caching (Anthropic, Bedrock-on-Anthropic) attach
    /// `cache_control: {"type":"ephemeral"}` to this message's last content
    /// block. Other providers ignore the flag. Cache TTL is 5 minutes; cache
    /// reads cost ~0.1× input, writes ~1.25× input.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cache: bool,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self::text(Role::System, text)
    }
    pub fn user(text: impl Into<String>) -> Self {
        Self::text(Role::User, text)
    }
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(Role::Assistant, text)
    }
    pub fn tool_response(tool_call_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentPart::Text { text: text.into() }],
            tool_calls: vec![],
            tool_call_id: Some(tool_call_id.into()),
            name: None,
            cache: false,
        }
    }

    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text { text: text.into() }],
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            cache: false,
        }
    }

    /// Mark this message as a prompt-cache breakpoint (Anthropic + Bedrock).
    /// Providers that don't support caching ignore the flag.
    pub fn cached(mut self) -> Self {
        self.cache = true;
        self
    }

    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}
