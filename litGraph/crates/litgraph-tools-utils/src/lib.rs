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

pub use calculator::CalculatorTool;
pub use http_request::{HttpRequestTool, HttpRequestConfig};
pub use filesystem::{FsRoot, ListDirectoryTool, ReadFileTool, WriteFileTool};
pub use shell::ShellTool;
pub use sqlite_query::SqliteQueryTool;
