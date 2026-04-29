//! MCP server — expose litGraph `Tool`s over stdio (line-delimited JSON-RPC)
//! so Claude Desktop, Cursor, Zed, and any MCP-aware host can call them.
//!
//! # What's supported
//!
//! - `initialize` — returns protocol version + capabilities.
//! - `tools/list` — enumerates registered tools with their JSON-Schema params.
//! - `tools/call` — invokes a tool; result is wrapped in MCP's
//!   `{content: [{type:"text", text:...}]}` shape.
//! - `resources/list` — enumerates registered resources (uri/name/mime).
//! - `resources/read` — reads one resource by URI, returns its text content.
//! - `prompts/list` — enumerates registered prompts with their argument specs.
//! - `prompts/get` — renders one prompt by name+args; returns MCP messages.
//! - `notifications/initialized` — accepted + ignored (client courtesy).
//! - `ping` — responds with empty object (keepalive).
//!
//! # What's NOT supported
//!
//! - HTTP transport — stdio covers the desktop-host use case. HTTP variant
//!   would reuse the dispatcher but swap the IO loop.
//! - Server-initiated requests (sampling/roots).
//! - Change notifications (tool/resource/prompt list updates) — our registered
//!   sets are static per process.
//! - Binary resources (blob) — text only; add when a caller asks.
//!
//! # Wire format
//!
//! One JSON-RPC message per line on stdin. Responses on stdout.
//! stderr is free for logging / diagnostics (Claude Desktop shows it in
//! the server-logs panel).
//!
//! # Error codes (JSON-RPC 2.0)
//!
//! - `-32700` parse error
//! - `-32600` invalid request
//! - `-32601` method not found
//! - `-32602` invalid params
//! - `-32603` internal error (tool raised)

use std::sync::Arc;

use litgraph_core::tool::{Tool, ToolSchema};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "litgraph-mcp-server";

/// Synchronously-readable resource handler. Returns the resource's text
/// content; errors surface as JSON-RPC `-32603 internal error`.
pub type ResourceReader =
    Arc<dyn Fn() -> std::result::Result<String, String> + Send + Sync>;

/// Prompt renderer — takes the argument dict supplied by the client and
/// returns the rendered list of MCP messages (role + text content).
pub type PromptRenderer = Arc<
    dyn Fn(&Value) -> std::result::Result<Vec<McpPromptMessage>, String> + Send + Sync,
>;

/// One resource exposed by the server. URI is the stable identifier the
/// client references in `resources/read`. `mime_type` is a hint — clients
/// decide how to render (typical: "text/plain", "text/markdown",
/// "application/json").
#[derive(Clone)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    pub mime_type: String,
    pub reader: ResourceReader,
}

/// Argument declaration for a prompt. `required=true` means the client MUST
/// supply a value; `required=false` allows omission (falls through to the
/// renderer's default handling).
#[derive(Debug, Clone)]
pub struct McpPromptArg {
    pub name: String,
    pub description: String,
    pub required: bool,
}

/// One message in a rendered prompt. MCP prompts return an array of these
/// for the client to splice into its conversation. `role` is "user" |
/// "assistant" | "system" (per MCP spec — MCP does not have a "tool" role
/// in the prompts wire format).
#[derive(Debug, Clone)]
pub struct McpPromptMessage {
    pub role: String,
    pub text: String,
}

/// One prompt exposed by the server. `renderer` receives the client-supplied
/// arguments (a JSON object) and returns the rendered messages.
#[derive(Clone)]
pub struct McpPrompt {
    pub name: String,
    pub description: String,
    pub arguments: Vec<McpPromptArg>,
    pub renderer: PromptRenderer,
}

/// Builds responses for an MCP server. Transport-agnostic — stdio calls
/// `dispatch` per line; HTTP would call it per request.
pub struct McpServer {
    tools: Vec<Arc<dyn Tool>>,
    resources: Vec<McpResource>,
    prompts: Vec<McpPrompt>,
    server_name: String,
    server_version: String,
}

impl McpServer {
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            tools,
            resources: Vec::new(),
            prompts: Vec::new(),
            server_name: SERVER_NAME.to_string(),
            server_version: "0.1.0".to_string(),
        }
    }

    pub fn with_server_name(mut self, name: impl Into<String>) -> Self {
        self.server_name = name.into();
        self
    }

    pub fn with_server_version(mut self, v: impl Into<String>) -> Self {
        self.server_version = v.into();
        self
    }

    /// Register a resource. Duplicate URIs are allowed by MCP but the server
    /// returns the first match on `resources/read`; callers should keep URIs
    /// unique.
    pub fn with_resource(mut self, resource: McpResource) -> Self {
        self.resources.push(resource);
        self
    }

    /// Register a prompt. Duplicate names same rule as resources — first
    /// match wins on `prompts/get`.
    pub fn with_prompt(mut self, prompt: McpPrompt) -> Self {
        self.prompts.push(prompt);
        self
    }

    /// Dispatch one parsed JSON-RPC message. Returns a `Value` response
    /// (empty for notifications). Caller is responsible for writing it
    /// back with trailing `\n`.
    pub async fn dispatch(&self, req: &Value) -> Option<Value> {
        let id = req.get("id").cloned();
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(Value::Null);

        // Notifications (per JSON-RPC 2.0: id absent). Responses MUST NOT
        // be sent for notifications.
        let is_notification = id.is_none();

        let result_or_err: std::result::Result<Value, (i64, String)> = match method {
            "initialize" => Ok(self.handle_initialize(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(self.handle_tools_list()),
            "tools/call" => self.handle_tools_call(&params).await,
            "resources/list" => Ok(self.handle_resources_list()),
            "resources/read" => self.handle_resources_read(&params),
            "prompts/list" => Ok(self.handle_prompts_list()),
            "prompts/get" => self.handle_prompts_get(&params),
            // Notifications we accept but ignore.
            "notifications/initialized"
            | "notifications/cancelled"
            | "notifications/progress" => Ok(Value::Null),
            _ => Err((-32601, format!("method not found: {method}"))),
        };

        if is_notification {
            return None;
        }

        // Build the response envelope.
        let id = id.unwrap_or(Value::Null);
        Some(match result_or_err {
            Ok(v) => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": v,
            }),
            Err((code, message)) => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": code, "message": message},
            }),
        })
    }

    fn handle_initialize(&self, _params: &Value) -> Value {
        // Advertise only what we actually registered — clients filter by
        // capability when deciding what to call. Tools are always advertised
        // (the server-by-default use case), but resources/prompts only if
        // the caller wired any in.
        let mut caps = serde_json::Map::new();
        caps.insert("tools".into(), json!({"listChanged": false}));
        if !self.resources.is_empty() {
            caps.insert("resources".into(), json!({"listChanged": false, "subscribe": false}));
        }
        if !self.prompts.is_empty() {
            caps.insert("prompts".into(), json!({"listChanged": false}));
        }
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": Value::Object(caps),
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version,
            },
        })
    }

    fn handle_tools_list(&self) -> Value {
        let tools: Vec<Value> = self
            .tools
            .iter()
            .map(|t| {
                let s = t.schema();
                tool_schema_to_mcp(&s)
            })
            .collect();
        json!({"tools": tools})
    }

    async fn handle_tools_call(
        &self,
        params: &Value,
    ) -> std::result::Result<Value, (i64, String)> {
        let name = params
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or((-32602, "missing `name`".to_string()))?;
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));

        let tool = self
            .tools
            .iter()
            .find(|t| t.schema().name == name)
            .ok_or_else(|| (-32602, format!("unknown tool `{name}`")))?;

        match tool.run(args).await {
            Ok(v) => {
                // Wrap in MCP's content-block shape. Text content is the
                // universal fallback; clients that want structured data
                // can json.loads(text).
                let text = match &v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                Ok(json!({
                    "content": [{"type": "text", "text": text}],
                    "isError": false,
                }))
            }
            Err(e) => Ok(json!({
                "content": [{"type": "text", "text": format!("tool error: {e}")}],
                "isError": true,
            })),
        }
    }

    fn handle_resources_list(&self) -> Value {
        let resources: Vec<Value> = self
            .resources
            .iter()
            .map(|r| {
                json!({
                    "uri": r.uri,
                    "name": r.name,
                    "description": r.description,
                    "mimeType": r.mime_type,
                })
            })
            .collect();
        json!({"resources": resources})
    }

    fn handle_resources_read(
        &self,
        params: &Value,
    ) -> std::result::Result<Value, (i64, String)> {
        let uri = params
            .get("uri")
            .and_then(|u| u.as_str())
            .ok_or((-32602, "missing `uri`".to_string()))?;
        let res = self
            .resources
            .iter()
            .find(|r| r.uri == uri)
            .ok_or_else(|| (-32602, format!("unknown resource `{uri}`")))?;
        let text = (res.reader)().map_err(|e| (-32603, format!("read failed: {e}")))?;
        Ok(json!({
            "contents": [{
                "uri": res.uri,
                "mimeType": res.mime_type,
                "text": text,
            }],
        }))
    }

    fn handle_prompts_list(&self) -> Value {
        let prompts: Vec<Value> = self
            .prompts
            .iter()
            .map(|p| {
                let args: Vec<Value> = p
                    .arguments
                    .iter()
                    .map(|a| {
                        json!({
                            "name": a.name,
                            "description": a.description,
                            "required": a.required,
                        })
                    })
                    .collect();
                json!({
                    "name": p.name,
                    "description": p.description,
                    "arguments": args,
                })
            })
            .collect();
        json!({"prompts": prompts})
    }

    fn handle_prompts_get(
        &self,
        params: &Value,
    ) -> std::result::Result<Value, (i64, String)> {
        let name = params
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or((-32602, "missing `name`".to_string()))?;
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        let prompt = self
            .prompts
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| (-32602, format!("unknown prompt `{name}`")))?;

        // Validate required args are present before calling the renderer —
        // saves renderers from repeating the same boilerplate.
        for a in &prompt.arguments {
            if a.required {
                let present = args
                    .get(&a.name)
                    .map(|v| !v.is_null())
                    .unwrap_or(false);
                if !present {
                    return Err((-32602, format!("missing required argument `{}`", a.name)));
                }
            }
        }

        let messages = (prompt.renderer)(&args)
            .map_err(|e| (-32603, format!("prompt render failed: {e}")))?;
        let wire: Vec<Value> = messages
            .into_iter()
            .map(|m| {
                json!({
                    "role": m.role,
                    "content": {"type": "text", "text": m.text},
                })
            })
            .collect();
        Ok(json!({
            "description": prompt.description,
            "messages": wire,
        }))
    }
}

fn tool_schema_to_mcp(s: &ToolSchema) -> Value {
    json!({
        "name": s.name,
        "description": s.description,
        "inputSchema": s.parameters,
    })
}

/// Run the server on stdin / stdout. Blocks until stdin closes. Each
/// line on stdin is one JSON-RPC message; each response goes to stdout
/// followed by `\n`. Errors during dispatch are surfaced as JSON-RPC
/// error responses — IO errors terminate the loop.
pub async fn serve_stdio(server: McpServer) -> std::io::Result<()> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    serve_on(server, stdin, stdout).await
}

/// Generic loop — stdio calls `serve_on(stdin, stdout)`; tests can pass
/// `&[u8]` + `Vec<u8>` to assert end-to-end behavior without a real
/// process. Reads line-delimited JSON-RPC; writes `Response\n` for
/// every non-notification.
pub async fn serve_on<R, W>(
    server: McpServer,
    reader: R,
    mut writer: W,
) -> std::io::Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut lines = BufReader::new(reader).lines();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                // Per JSON-RPC 2.0: parse error → id=null, code -32700.
                let err = json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": {"code": -32700, "message": format!("parse error: {e}")},
                });
                writer
                    .write_all(err.to_string().as_bytes())
                    .await?;
                writer.write_all(b"\n").await?;
                writer.flush().await?;
                continue;
            }
        };

        // Batch requests: array of messages. Rare but spec-required.
        let responses = if let Some(arr) = req.as_array() {
            let mut out = Vec::new();
            for item in arr {
                if let Some(r) = server.dispatch(item).await {
                    out.push(r);
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(Value::Array(out))
            }
        } else {
            server.dispatch(&req).await
        };

        if let Some(resp) = responses {
            writer
                .write_all(resp.to_string().as_bytes())
                .await?;
            writer.write_all(b"\n").await?;
            writer.flush().await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::Result;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "echo".to_string(),
                description: "Echoes back the `text` input.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }),
            }
        }
        async fn run(&self, args: Value) -> Result<Value> {
            let t = args
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok(Value::String(t))
        }
    }

    struct FailingTool;

    #[async_trait]
    impl Tool for FailingTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "fails".to_string(),
                description: "Always fails.".to_string(),
                parameters: json!({"type": "object"}),
            }
        }
        async fn run(&self, _args: Value) -> Result<Value> {
            Err(litgraph_core::Error::parse("kaboom"))
        }
    }

    fn server() -> McpServer {
        McpServer::new(vec![Arc::new(EchoTool), Arc::new(FailingTool)])
    }

    #[tokio::test]
    async fn initialize_returns_protocol_version_and_capabilities() {
        let s = server();
        let req = json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}});
        let resp = s.dispatch(&req).await.unwrap();
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert!(resp["result"]["capabilities"]["tools"].is_object());
        assert_eq!(resp["result"]["serverInfo"]["name"], "litgraph-mcp-server");
    }

    #[tokio::test]
    async fn tools_list_includes_every_registered_tool() {
        let s = server();
        let req = json!({"jsonrpc":"2.0","id":2,"method":"tools/list"});
        let resp = s.dispatch(&req).await.unwrap();
        let tools = resp["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"echo"));
        assert!(names.contains(&"fails"));
        // Schema is exposed as `inputSchema` (MCP spec).
        assert!(tools[0]["inputSchema"].is_object());
    }

    #[tokio::test]
    async fn tools_call_echoes_argument() {
        let s = server();
        let req = json!({
            "jsonrpc":"2.0","id":3,"method":"tools/call",
            "params":{"name":"echo","arguments":{"text":"hi there"}}
        });
        let resp = s.dispatch(&req).await.unwrap();
        assert_eq!(resp["result"]["isError"], false);
        let content = resp["result"]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "hi there");
    }

    #[tokio::test]
    async fn tools_call_tool_error_surfaces_isError_true_not_jsonrpc_error() {
        let s = server();
        let req = json!({
            "jsonrpc":"2.0","id":4,"method":"tools/call",
            "params":{"name":"fails","arguments":{}}
        });
        let resp = s.dispatch(&req).await.unwrap();
        // MCP spec: tool errors are REGULAR results with isError=true, not
        // JSON-RPC errors. JSON-RPC errors are for protocol-level failures
        // (unknown method, malformed params).
        assert!(resp.get("error").is_none());
        assert_eq!(resp["result"]["isError"], true);
        assert!(resp["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("kaboom"));
    }

    #[tokio::test]
    async fn unknown_method_returns_jsonrpc_error_code_32601() {
        let s = server();
        let req = json!({"jsonrpc":"2.0","id":5,"method":"bogus/method"});
        let resp = s.dispatch(&req).await.unwrap();
        assert_eq!(resp["error"]["code"], -32601);
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("bogus/method"));
    }

    #[tokio::test]
    async fn tools_call_unknown_tool_returns_jsonrpc_error_code_32602() {
        let s = server();
        let req = json!({
            "jsonrpc":"2.0","id":6,"method":"tools/call",
            "params":{"name":"nope","arguments":{}}
        });
        let resp = s.dispatch(&req).await.unwrap();
        assert_eq!(resp["error"]["code"], -32602);
    }

    #[tokio::test]
    async fn notifications_produce_no_response() {
        let s = server();
        // id absent → notification per JSON-RPC 2.0.
        let req = json!({"jsonrpc":"2.0","method":"notifications/initialized"});
        let resp = s.dispatch(&req).await;
        assert!(resp.is_none());
    }

    #[tokio::test]
    async fn ping_responds_with_empty_result() {
        let s = server();
        let req = json!({"jsonrpc":"2.0","id":7,"method":"ping"});
        let resp = s.dispatch(&req).await.unwrap();
        assert!(resp["result"].is_object());
    }

    /// End-to-end via in-memory byte buffers. Avoids tokio::io::duplex
    /// (which has buffer-deadlock quirks when one end reads sporadically
    /// — we hit this in iter 131). Instead we prepare ALL requests as
    /// one `\n`-delimited blob, feed it to `serve_on` as the reader,
    /// collect the writer output, then parse + assert.
    #[tokio::test]
    async fn serve_on_processes_multiple_requests_and_skips_notifications() {
        let s = server();
        let requests = [
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
            json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
            // Notification — no response.
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            json!({
                "jsonrpc":"2.0","id":3,"method":"tools/call",
                "params":{"name":"echo","arguments":{"text":"ok"}}
            }),
        ];
        let input: String = requests
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";

        let reader = std::io::Cursor::new(input.into_bytes());
        let mut output: Vec<u8> = Vec::new();
        serve_on(s, reader, &mut output).await.unwrap();

        // Parse the output into one Value per line.
        let text = String::from_utf8(output).unwrap();
        let responses: Vec<Value> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        // 4 inputs → 3 responses (notification skipped).
        assert_eq!(responses.len(), 3);
        assert_eq!(responses[0]["id"], 1);
        assert_eq!(responses[0]["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(responses[1]["id"], 2);
        assert_eq!(responses[1]["result"]["tools"].as_array().unwrap().len(), 2);
        assert_eq!(responses[2]["id"], 3);
        assert_eq!(responses[2]["result"]["content"][0]["text"], "ok");
    }

    #[tokio::test]
    async fn serve_on_handles_parse_error_and_continues_processing() {
        let s = server();
        let input = "{not valid json\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n";
        let reader = std::io::Cursor::new(input.as_bytes().to_vec());
        let mut output: Vec<u8> = Vec::new();
        serve_on(s, reader, &mut output).await.unwrap();

        let text = String::from_utf8(output).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        // First: parse error.
        assert_eq!(lines[0]["error"]["code"], -32700);
        // Second: ping response (loop continued past the malformed line).
        assert_eq!(lines[1]["id"], 1);
        assert!(lines[1]["result"].is_object());
    }

    #[tokio::test]
    async fn serve_on_handles_batch_request() {
        // JSON-RPC 2.0 batching: array of messages → array of responses.
        let s = server();
        let batch = json!([
            {"jsonrpc":"2.0","id":1,"method":"ping"},
            {"jsonrpc":"2.0","id":2,"method":"tools/list"}
        ]);
        let input = format!("{}\n", batch);
        let reader = std::io::Cursor::new(input.into_bytes());
        let mut output: Vec<u8> = Vec::new();
        serve_on(s, reader, &mut output).await.unwrap();

        let text = String::from_utf8(output).unwrap();
        let resp: Value = serde_json::from_str(text.trim()).unwrap();
        let arr = resp.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["id"], 1);
        assert_eq!(arr[1]["id"], 2);
    }

    #[tokio::test]
    async fn custom_server_name_and_version_reflected_in_initialize() {
        let s = McpServer::new(vec![])
            .with_server_name("my-agent")
            .with_server_version("2.0.0");
        let req = json!({"jsonrpc":"2.0","id":1,"method":"initialize"});
        let resp = s.dispatch(&req).await.unwrap();
        assert_eq!(resp["result"]["serverInfo"]["name"], "my-agent");
        assert_eq!(resp["result"]["serverInfo"]["version"], "2.0.0");
    }

    // --- Resources tests ---

    fn make_resource(uri: &str, text: &'static str) -> McpResource {
        McpResource {
            uri: uri.to_string(),
            name: format!("res-{uri}"),
            description: "test resource".to_string(),
            mime_type: "text/plain".to_string(),
            reader: Arc::new(move || Ok(text.to_string())),
        }
    }

    #[tokio::test]
    async fn resources_list_enumerates_registered_resources() {
        let s = McpServer::new(vec![])
            .with_resource(make_resource("mem://readme", "hello"))
            .with_resource(make_resource("mem://license", "MIT"));
        let req = json!({"jsonrpc":"2.0","id":1,"method":"resources/list"});
        let resp = s.dispatch(&req).await.unwrap();
        let arr = resp["result"]["resources"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["uri"], "mem://readme");
        assert_eq!(arr[0]["mimeType"], "text/plain");
        assert_eq!(arr[1]["uri"], "mem://license");
    }

    #[tokio::test]
    async fn resources_read_returns_reader_content() {
        let s = McpServer::new(vec![])
            .with_resource(make_resource("mem://config", "FOO=bar"));
        let req = json!({
            "jsonrpc":"2.0","id":1,"method":"resources/read",
            "params": {"uri": "mem://config"}
        });
        let resp = s.dispatch(&req).await.unwrap();
        let contents = resp["result"]["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["uri"], "mem://config");
        assert_eq!(contents[0]["text"], "FOO=bar");
        assert_eq!(contents[0]["mimeType"], "text/plain");
    }

    #[tokio::test]
    async fn resources_read_unknown_uri_returns_invalid_params() {
        let s = McpServer::new(vec![])
            .with_resource(make_resource("mem://a", "x"));
        let req = json!({
            "jsonrpc":"2.0","id":1,"method":"resources/read",
            "params": {"uri": "mem://bogus"}
        });
        let resp = s.dispatch(&req).await.unwrap();
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"].as_str().unwrap().contains("bogus"));
    }

    #[tokio::test]
    async fn resources_read_missing_uri_returns_invalid_params() {
        let s = McpServer::new(vec![]).with_resource(make_resource("mem://a", "x"));
        let req = json!({"jsonrpc":"2.0","id":1,"method":"resources/read","params":{}});
        let resp = s.dispatch(&req).await.unwrap();
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"].as_str().unwrap().contains("uri"));
    }

    #[tokio::test]
    async fn resources_read_reader_error_surfaces_internal_error() {
        let s = McpServer::new(vec![]).with_resource(McpResource {
            uri: "mem://flaky".into(),
            name: "flaky".into(),
            description: "fails on read".into(),
            mime_type: "text/plain".into(),
            reader: Arc::new(|| Err("disk unreachable".to_string())),
        });
        let req = json!({
            "jsonrpc":"2.0","id":1,"method":"resources/read",
            "params": {"uri": "mem://flaky"}
        });
        let resp = s.dispatch(&req).await.unwrap();
        assert_eq!(resp["error"]["code"], -32603);
        assert!(resp["error"]["message"].as_str().unwrap().contains("disk unreachable"));
    }

    #[tokio::test]
    async fn initialize_advertises_resources_capability_only_when_registered() {
        let without = McpServer::new(vec![]);
        let r1 = without.dispatch(&json!({"jsonrpc":"2.0","id":1,"method":"initialize"})).await.unwrap();
        assert!(r1["result"]["capabilities"].get("resources").is_none());

        let with = without.with_resource(make_resource("mem://a", "x"));
        let r2 = with.dispatch(&json!({"jsonrpc":"2.0","id":1,"method":"initialize"})).await.unwrap();
        assert_eq!(r2["result"]["capabilities"]["resources"]["listChanged"], false);
    }

    // --- Prompts tests ---

    fn make_prompt_greet() -> McpPrompt {
        McpPrompt {
            name: "greet".into(),
            description: "Say hello to someone by name".into(),
            arguments: vec![McpPromptArg {
                name: "who".into(),
                description: "who to greet".into(),
                required: true,
            }],
            renderer: Arc::new(|args| {
                let who = args
                    .get("who")
                    .and_then(|v| v.as_str())
                    .unwrap_or("stranger");
                Ok(vec![McpPromptMessage {
                    role: "user".into(),
                    text: format!("Please greet {who} warmly."),
                }])
            }),
        }
    }

    #[tokio::test]
    async fn prompts_list_enumerates_registered_prompts_with_args() {
        let s = McpServer::new(vec![]).with_prompt(make_prompt_greet());
        let req = json!({"jsonrpc":"2.0","id":1,"method":"prompts/list"});
        let resp = s.dispatch(&req).await.unwrap();
        let arr = resp["result"]["prompts"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["name"], "greet");
        let args = arr[0]["arguments"].as_array().unwrap();
        assert_eq!(args[0]["name"], "who");
        assert_eq!(args[0]["required"], true);
    }

    #[tokio::test]
    async fn prompts_get_renders_messages_with_supplied_args() {
        let s = McpServer::new(vec![]).with_prompt(make_prompt_greet());
        let req = json!({
            "jsonrpc":"2.0","id":1,"method":"prompts/get",
            "params": {"name": "greet", "arguments": {"who": "Alice"}}
        });
        let resp = s.dispatch(&req).await.unwrap();
        let msgs = resp["result"]["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"]["type"], "text");
        assert_eq!(msgs[0]["content"]["text"], "Please greet Alice warmly.");
    }

    #[tokio::test]
    async fn prompts_get_missing_required_arg_returns_invalid_params() {
        let s = McpServer::new(vec![]).with_prompt(make_prompt_greet());
        let req = json!({
            "jsonrpc":"2.0","id":1,"method":"prompts/get",
            "params": {"name": "greet", "arguments": {}}
        });
        let resp = s.dispatch(&req).await.unwrap();
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"].as_str().unwrap().contains("who"));
    }

    #[tokio::test]
    async fn prompts_get_unknown_prompt_returns_invalid_params() {
        let s = McpServer::new(vec![]).with_prompt(make_prompt_greet());
        let req = json!({
            "jsonrpc":"2.0","id":1,"method":"prompts/get",
            "params": {"name": "bogus"}
        });
        let resp = s.dispatch(&req).await.unwrap();
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"].as_str().unwrap().contains("bogus"));
    }

    #[tokio::test]
    async fn prompts_get_renderer_error_surfaces_internal_error() {
        let s = McpServer::new(vec![]).with_prompt(McpPrompt {
            name: "fails".into(),
            description: "always fails".into(),
            arguments: vec![],
            renderer: Arc::new(|_| Err("render kaboom".to_string())),
        });
        let req = json!({
            "jsonrpc":"2.0","id":1,"method":"prompts/get",
            "params": {"name": "fails"}
        });
        let resp = s.dispatch(&req).await.unwrap();
        assert_eq!(resp["error"]["code"], -32603);
        assert!(resp["error"]["message"].as_str().unwrap().contains("render kaboom"));
    }

    #[tokio::test]
    async fn prompts_get_optional_arg_ok_when_omitted() {
        let s = McpServer::new(vec![]).with_prompt(McpPrompt {
            name: "optional-args".into(),
            description: "all optional".into(),
            arguments: vec![McpPromptArg {
                name: "who".into(),
                description: "opt".into(),
                required: false,
            }],
            renderer: Arc::new(|_| {
                Ok(vec![McpPromptMessage { role: "user".into(), text: "hi".into() }])
            }),
        });
        let req = json!({
            "jsonrpc":"2.0","id":1,"method":"prompts/get",
            "params": {"name": "optional-args"}
        });
        let resp = s.dispatch(&req).await.unwrap();
        let msgs = resp["result"]["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["content"]["text"], "hi");
    }

    #[tokio::test]
    async fn initialize_advertises_prompts_capability_only_when_registered() {
        let without = McpServer::new(vec![]);
        let r1 = without.dispatch(&json!({"jsonrpc":"2.0","id":1,"method":"initialize"})).await.unwrap();
        assert!(r1["result"]["capabilities"].get("prompts").is_none());

        let with = without.with_prompt(make_prompt_greet());
        let r2 = with.dispatch(&json!({"jsonrpc":"2.0","id":1,"method":"initialize"})).await.unwrap();
        assert_eq!(r2["result"]["capabilities"]["prompts"]["listChanged"], false);
    }
}
