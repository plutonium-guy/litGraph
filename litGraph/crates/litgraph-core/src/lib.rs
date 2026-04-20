//! litgraph-core — shared types, traits, errors for litGraph.
//!
//! No PyO3. Usable as a pure Rust crate. Python bindings live in `litgraph-py`.

pub mod error;
pub mod message;
pub mod prompt;
pub mod model;
pub mod tool;
pub mod document;
pub mod memory;

pub use error::{Error, Result};
pub use memory::{
    BufferMemory, ConversationMemory, MemorySnapshot, TokenBufferMemory, TokenCounter,
    summarize_conversation,
};
pub use message::{Message, Role, ContentPart, ImageSource};
pub use model::{ChatModel, Embeddings, ChatOptions, ChatResponse, TokenUsage, FinishReason};
pub use prompt::{ChatPromptTemplate, PromptValue};
pub use tool::{Tool, ToolCall, ToolResult, ToolSchema};
pub use document::Document;
