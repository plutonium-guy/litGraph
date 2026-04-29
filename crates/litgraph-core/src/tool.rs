use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the args object.
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON arguments as produced by the model.
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn schema(&self) -> ToolSchema;
    fn name(&self) -> String { self.schema().name }
    async fn run(&self, args: Value) -> Result<Value>;
}

type BoxFuture<O> = Pin<Box<dyn Future<Output = Result<O>> + Send>>;

/// Build a `Tool` from an async function plus an explicit JSON Schema.
///
/// The passed closure must produce a boxed, `Send` future — the easiest shape is
/// `|args: MyArgs| Box::pin(async move { ... }) as BoxFuture<MyOut>`.
///
/// Until a proc-macro `#[tool]` lands (which derives the schema via `schemars`),
/// callers pass the schema explicitly.
///
/// ```no_run
/// use litgraph_core::tool::{FnTool, Tool};
/// use serde::{Deserialize, Serialize};
/// use serde_json::json;
///
/// #[derive(Deserialize)]
/// struct AddArgs { a: i64, b: i64 }
/// #[derive(Serialize)]
/// struct AddOut { sum: i64 }
///
/// let tool = FnTool::new(
///     "add",
///     "Add two integers.",
///     json!({
///         "type": "object",
///         "properties": {
///             "a": { "type": "integer" },
///             "b": { "type": "integer" }
///         },
///         "required": ["a", "b"]
///     }),
///     |args: AddArgs| Box::pin(async move { Ok(AddOut { sum: args.a + args.b }) }),
/// );
/// ```
pub struct FnTool<A, O>
where
    A: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
{
    name: String,
    description: String,
    parameters: Value,
    func: Arc<dyn Fn(A) -> BoxFuture<O> + Send + Sync>,
}

impl<A, O> FnTool<A, O>
where
    A: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
{
    pub fn new<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        func: F,
    ) -> Self
    where
        F: Fn(A) -> BoxFuture<O> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            func: Arc::new(func),
        }
    }
}

#[async_trait]
impl<A, O> Tool for FnTool<A, O>
where
    A: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
{
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }
    async fn run(&self, args: Value) -> Result<Value> {
        let parsed: A = serde_json::from_value(args)
            .map_err(|e| Error::invalid(format!("tool `{}` args: {e}", self.name)))?;
        let out = (self.func)(parsed).await?;
        serde_json::to_value(out).map_err(Error::from)
    }
}
