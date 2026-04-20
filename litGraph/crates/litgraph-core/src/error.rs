use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("provider error: {0}")]
    Provider(String),

    #[error("rate limited (retry after {retry_after_ms:?}ms)")]
    RateLimited { retry_after_ms: Option<u64> },

    #[error("request timed out")]
    Timeout,

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("template error: {0}")]
    Template(#[from] minijinja::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("tool `{name}` not found")]
    ToolNotFound { name: String },

    #[error("tool `{name}` failed: {source}")]
    ToolFailed { name: String, source: Box<dyn std::error::Error + Send + Sync> },

    #[error("parse error: {0}")]
    Parse(String),

    #[error("cancelled")]
    Cancelled,

    #[error("other: {0}")]
    Other(String),
}

impl Error {
    pub fn provider(msg: impl Into<String>) -> Self { Self::Provider(msg.into()) }
    pub fn invalid(msg: impl Into<String>) -> Self { Self::InvalidInput(msg.into()) }
    pub fn parse(msg: impl Into<String>) -> Self { Self::Parse(msg.into()) }
    pub fn other(msg: impl Into<String>) -> Self { Self::Other(msg.into()) }
}
