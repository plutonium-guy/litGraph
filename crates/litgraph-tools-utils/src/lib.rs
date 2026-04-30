//! Built-in utility tools for ReactAgent. No network credentials required:
//! these are the everyday primitives an agent needs to do basic computation
//! or hit a user-controlled HTTP endpoint without you wiring a custom Tool
//! every time.
//!
//! - `CalculatorTool` — `evalexpr` arithmetic & boolean expressions
//! - `HttpRequestTool` — generic GET/POST with optional headers + JSON body

mod calculator;
mod http_request;
mod filesystem;
mod shell;
mod sqlite_query;
mod whisper;
mod dalle;
mod tts;
mod cached;
mod python_repl;
mod webhook;
mod gmail_send;
mod web_fetch;
mod tool_resilience;
mod planning;
mod virtual_fs;
mod current_time;
mod regex_extract;
mod json_extract;
mod url_parse;
mod hash;

pub use calculator::CalculatorTool;
pub use http_request::{HttpRequestTool, HttpRequestConfig};
pub use filesystem::{FsRoot, ListDirectoryTool, ReadFileTool, WriteFileTool};
pub use shell::ShellTool;
pub use sqlite_query::SqliteQueryTool;
pub use whisper::{WhisperConfig, WhisperTranscribeTool};
pub use dalle::{DalleConfig, DalleImageTool};
pub use tts::{TtsAudioTool, TtsConfig};
pub use cached::CachedTool;
pub use python_repl::{PythonReplConfig, PythonReplTool};
pub use webhook::{WebhookConfig, WebhookPreset, WebhookTool};
pub use gmail_send::{GmailSendConfig, GmailSendTool};
pub use web_fetch::{WebFetchConfig, WebFetchTool};
pub use tool_resilience::{RetryConfig, RetryTool, TimeoutTool};
pub use planning::{PlanningTool, TodoItem, TodoStatus};
pub use virtual_fs::VirtualFilesystemTool;
pub use current_time::CurrentTimeTool;
pub use regex_extract::RegexExtractTool;
pub use json_extract::JsonExtractTool;
pub use url_parse::UrlParseTool;
pub use hash::HashTool;
