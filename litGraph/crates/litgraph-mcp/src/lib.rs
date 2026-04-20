//! Model Context Protocol (MCP) client for litGraph — stdio transport.
//!
//! MCP is Anthropic's open JSON-RPC 2.0 protocol for connecting LLMs to
//! external tools/resources/prompts. Servers exist for filesystem, github,
//! slack, postgres, sqlite, fetch, brave-search, puppeteer, etc.
//! See <https://modelcontextprotocol.io>.
//!
//! # What's supported (v1)
//! - Stdio transport (spawn child process, line-delimited JSON-RPC).
//! - `initialize` handshake with capability declaration.
//! - `tools/list` — query the server's tools.
//! - `tools/call(name, args)` — execute a tool.
//! - `McpTool` adapter: each MCP tool implements `litgraph_core::Tool` so it
//!   plugs straight into `ReactAgent` alongside our native tools.
//!
//! # Not yet supported
//! - SSE / WebSocket transports.
//! - `resources/*` and `prompts/*` (less commonly used).
//! - Server-initiated requests (sampling, roots, logging notifications).
//!   Notifications from server are silently dropped — they never block
//!   request/response correlation.
//!
//! # Wire format
//!
//! Every request is `{"jsonrpc":"2.0","id":<int>,"method":"...","params":{...}}`.
//! Response: `{"jsonrpc":"2.0","id":<int>,"result":{...}}` or
//! `{"jsonrpc":"2.0","id":<int>,"error":{"code":-32603,"message":"..."}}`.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex as AsyncMutex};
use tracing::{debug, warn};

const PROTOCOL_VERSION: &str = "2024-11-05";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

#[derive(Debug)]
struct PendingTable {
    next_id: u64,
    pending: HashMap<u64, oneshot::Sender<std::result::Result<Value, McpError>>>,
}

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("transport: {0}")]
    Transport(String),
    #[error("protocol: {0}")]
    Protocol(String),
    #[error("server error {code}: {message}")]
    Server { code: i64, message: String },
    #[error("timeout after {0:?}")]
    Timeout(Duration),
}

impl From<McpError> for Error {
    fn from(e: McpError) -> Self {
        Error::other(e.to_string())
    }
}

/// Spawn an MCP server as a child process and talk JSON-RPC over its stdio.
///
/// The client owns:
///   - `child_stdin` for sending requests
///   - a background tokio task that reads `child_stdout` line-by-line and
///     dispatches replies to per-request `oneshot` channels via `id`.
///
/// Drop = no explicit shutdown; the child gets reaped when its stdin closes.
pub struct McpClient {
    stdin: Arc<AsyncMutex<ChildStdin>>,
    table: Arc<Mutex<PendingTable>>,
    request_timeout: Duration,
    /// Keep the Child alive — `Command::spawn` with `.kill_on_drop(true)`
    /// will SIGKILL the process the moment this is dropped, so we MUST hold
    /// it on the client (not the spawn-local). Wrapped in `Mutex<Option>` so
    /// `Drop` can take it for an explicit kill if needed.
    _child: Arc<Mutex<Option<Child>>>,
}

impl McpClient {
    /// Spawn `program` with `args` and run the JSON-RPC `initialize` handshake.
    pub async fn connect_stdio(
        program: impl AsRef<std::ffi::OsStr>,
        args: &[&str],
    ) -> std::result::Result<Self, McpError> {
        let mut cmd = Command::new(program);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::Transport(format!("spawn: {e}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("no stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("no stdout".into()))?;
        // Drain stderr to a tracing target — keeps the buffer from filling
        // and silently blocking the child, AND surfaces server diagnostics.
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    debug!(target: "mcp::server_stderr", "{line}");
                }
            });
        }

        let table = Arc::new(Mutex::new(PendingTable {
            next_id: 1,
            pending: HashMap::new(),
        }));
        let table_for_reader = table.clone();

        // Background reader: parse one JSON object per line, route to id.
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            loop {
                match reader.next_line().await {
                    Ok(Some(line)) => {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        let v: Value = match serde_json::from_str(line) {
                            Ok(v) => v,
                            Err(e) => {
                                warn!(line, error = %e, "mcp: malformed line");
                                continue;
                            }
                        };
                        // Notifications (no `id`) are dropped; we don't ack
                        // server-side initiated requests in v1.
                        let id = match v.get("id").and_then(|i| i.as_u64()) {
                            Some(id) => id,
                            None => {
                                debug!(method = ?v.get("method"), "mcp: notification ignored");
                                continue;
                            }
                        };
                        let mut t = table_for_reader.lock();
                        if let Some(tx) = t.pending.remove(&id) {
                            let result = if let Some(err) = v.get("error") {
                                let code = err.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
                                let msg = err
                                    .get("message")
                                    .and_then(|m| m.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                Err(McpError::Server { code, message: msg })
                            } else {
                                Ok(v.get("result").cloned().unwrap_or(Value::Null))
                            };
                            let _ = tx.send(result);
                        }
                    }
                    Ok(None) => {
                        debug!("mcp: child stdout closed");
                        break;
                    }
                    Err(e) => {
                        warn!(error = %e, "mcp: read error");
                        break;
                    }
                }
            }
            // Wake any in-flight requests with a transport error.
            let mut t = table_for_reader.lock();
            for (_, tx) in t.pending.drain() {
                let _ = tx.send(Err(McpError::Transport("server closed stdout".into())));
            }
        });

        let client = Self {
            stdin: Arc::new(AsyncMutex::new(stdin)),
            table,
            request_timeout: Duration::from_secs(30),
            _child: Arc::new(Mutex::new(Some(child))),
        };

        // Initialize handshake. The server returns capabilities; we don't act
        // on them in v1 but parsing succeeds.
        let _: Value = client
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": { "name": "litgraph-mcp", "version": env!("CARGO_PKG_VERSION") },
                }),
            )
            .await?;
        // Per spec, send `notifications/initialized` (no id, no response).
        client.notify("notifications/initialized", json!({})).await?;
        Ok(client)
    }

    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.request_timeout = t;
        self
    }

    /// Send a request and await the matching response (correlated by id).
    async fn request(&self, method: &str, params: Value) -> std::result::Result<Value, McpError> {
        let (tx, rx) = oneshot::channel();
        let id = {
            let mut t = self.table.lock();
            let id = t.next_id;
            t.next_id += 1;
            t.pending.insert(id, tx);
            id
        };
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let bytes = serde_json::to_vec(&msg)
            .map_err(|e| McpError::Transport(format!("serialize: {e}")))?;
        {
            let mut s = self.stdin.lock().await;
            s.write_all(&bytes)
                .await
                .map_err(|e| McpError::Transport(format!("write: {e}")))?;
            s.write_all(b"\n")
                .await
                .map_err(|e| McpError::Transport(format!("write: {e}")))?;
            s.flush()
                .await
                .map_err(|e| McpError::Transport(format!("flush: {e}")))?;
        }
        match tokio::time::timeout(self.request_timeout, rx).await {
            Ok(Ok(r)) => r,
            Ok(Err(_)) => Err(McpError::Transport("response channel dropped".into())),
            Err(_) => {
                self.table.lock().pending.remove(&id);
                Err(McpError::Timeout(self.request_timeout))
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> std::result::Result<(), McpError> {
        let msg = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        let bytes = serde_json::to_vec(&msg)
            .map_err(|e| McpError::Transport(format!("serialize: {e}")))?;
        let mut s = self.stdin.lock().await;
        s.write_all(&bytes)
            .await
            .map_err(|e| McpError::Transport(format!("write: {e}")))?;
        s.write_all(b"\n")
            .await
            .map_err(|e| McpError::Transport(format!("write: {e}")))?;
        s.flush()
            .await
            .map_err(|e| McpError::Transport(format!("flush: {e}")))?;
        Ok(())
    }

    /// `tools/list` — descriptor for every tool the server exposes.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDescriptor>> {
        let result = self.request("tools/list", json!({})).await?;
        let arr = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| Error::other("mcp: tools/list missing `tools`"))?;
        let mut out = Vec::with_capacity(arr.len());
        for t in arr {
            let td: McpToolDescriptor = serde_json::from_value(t.clone())
                .map_err(|e| Error::other(format!("mcp: bad tool descriptor: {e}")))?;
            out.push(td);
        }
        Ok(out)
    }

    /// `tools/call` — invoke a tool by name with JSON args. Returns the
    /// result content blob (typically `{"content":[{"type":"text",...}]}`).
    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value> {
        let result = self
            .request(
                "tools/call",
                json!({ "name": name, "arguments": args }),
            )
            .await?;
        Ok(result)
    }

    /// Convenience: `tools/list` → wrap each as an `McpTool` ready to hand to
    /// `ReactAgent::new(.., tools=...)`.
    pub async fn into_tools(self: Arc<Self>) -> Result<Vec<Arc<dyn Tool>>> {
        let descriptors = self.list_tools().await?;
        let out: Vec<Arc<dyn Tool>> = descriptors
            .into_iter()
            .map(|d| {
                Arc::new(McpTool { client: self.clone(), descriptor: d }) as Arc<dyn Tool>
            })
            .collect();
        Ok(out)
    }
}

/// Adapter — exposes a single MCP-server-side tool through litGraph's `Tool`
/// trait so `ReactAgent` can call it the same as a native tool.
pub struct McpTool {
    client: Arc<McpClient>,
    descriptor: McpToolDescriptor,
}

#[async_trait]
impl Tool for McpTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.descriptor.name.clone(),
            description: self.descriptor.description.clone().unwrap_or_default(),
            parameters: self.descriptor.input_schema.clone(),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let raw = self.client.call_tool(&self.descriptor.name, args).await?;
        // Servers return {"content":[{"type":"text","text":"..."}], "isError":false}
        // We collapse text-only content into a single string for the agent and
        // surface isError as a tool-side error.
        if raw.get("isError").and_then(|v| v.as_bool()).unwrap_or(false) {
            let msg = raw
                .get("content")
                .and_then(|c| c.as_array())
                .and_then(|a| a.first())
                .and_then(|p| p.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("MCP tool reported isError=true")
                .to_string();
            return Err(Error::other(format!("mcp tool '{}' failed: {msg}", self.descriptor.name)));
        }
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end: spawn `python3 fake_server.py` (a script that speaks the
    /// MCP wire format on stdio) and verify we can list + call its tools.
    /// We script the fake server inline so the test is hermetic.
    fn fake_server_python() -> String {
        // Python source: reads JSON-RPC requests one per line, replies to
        // initialize / tools/list / tools/call("echo", {"text": ...}).
        r#"
import json, sys
def reply(rid, result=None, error=None):
    msg = {"jsonrpc":"2.0","id":rid}
    if error is not None: msg["error"] = error
    else: msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    method, rid, params = req.get("method"), req.get("id"), req.get("params", {})
    if rid is None:  # notification → drop
        continue
    if method == "initialize":
        reply(rid, {
            "protocolVersion":"2024-11-05",
            "capabilities":{"tools":{}},
            "serverInfo":{"name":"fake","version":"0"},
        })
    elif method == "tools/list":
        reply(rid, {"tools":[
            {"name":"echo","description":"Echoes back its `text` arg.",
             "inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},
            {"name":"err","description":"Returns isError=true.",
             "inputSchema":{"type":"object","properties":{}}},
        ]})
    elif method == "tools/call":
        name = params["name"]; args = params.get("arguments", {})
        if name == "echo":
            reply(rid, {"content":[{"type":"text","text": "you said: "+args["text"]}], "isError":False})
        elif name == "err":
            reply(rid, {"content":[{"type":"text","text":"oops"}], "isError":True})
        else:
            reply(rid, error={"code":-32601,"message":"unknown tool"})
    else:
        reply(rid, error={"code":-32601,"message":"unknown method"})
"#.to_string()
    }

    /// Spawn the inline python via `python3 -u -c <SRC>` so we don't depend
    /// on the temp file handle remaining valid across the spawn.
    async fn spawn_fake_server() -> Option<McpClient> {
        McpClient::connect_stdio("python3", &["-u", "-c", &fake_server_python()])
            .await
            .ok()
    }

    #[tokio::test]
    async fn lists_and_calls_tools_against_fake_stdio_server() {
        let Some(client) = spawn_fake_server().await else {
            eprintln!("skipping: python3 not available"); return;
        };
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 2);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"echo"));
        assert!(names.contains(&"err"));

        let res = client.call_tool("echo", json!({"text": "hello"})).await.unwrap();
        let txt = res["content"][0]["text"].as_str().unwrap();
        assert_eq!(txt, "you said: hello");
    }

    #[tokio::test]
    async fn mcp_tool_adapter_surfaces_iserror_as_rust_error() {
        let Some(client) = spawn_fake_server().await else { return; };
        let client = Arc::new(client);
        let tools = client.clone().into_tools().await.unwrap();
        let err_tool = tools.iter().find(|t| t.schema().name == "err").unwrap();
        let result = err_tool.run(json!({})).await;
        let e = result.unwrap_err();
        assert!(format!("{e}").contains("oops"), "got: {e}");
    }

    #[tokio::test]
    async fn unknown_tool_returns_server_error() {
        let Some(client) = spawn_fake_server().await else { return; };
        let err = client.call_tool("nope", json!({})).await.unwrap_err();
        assert!(format!("{err}").contains("unknown tool"));
    }

    #[tokio::test]
    async fn timeout_fires_when_server_does_not_respond() {
        // Server reads but never writes — request must timeout. Spawn
        // directly without going through `connect_stdio` (which would hang
        // on the initialize handshake). Test the raw request path.
        let mut cmd = Command::new("python3");
        cmd.args(["-u", "-c", "import sys\nfor _ in sys.stdin: pass\n"])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(_) => return,
        };
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let table = Arc::new(Mutex::new(PendingTable { next_id: 1, pending: HashMap::new() }));
        let table_r = table.clone();
        tokio::spawn(async move {
            let mut r = BufReader::new(stdout).lines();
            while let Ok(Some(_)) = r.next_line().await {}
            let mut t = table_r.lock();
            for (_, tx) in t.pending.drain() {
                let _ = tx.send(Err(McpError::Transport("closed".into())));
            }
        });
        let client = McpClient {
            stdin: Arc::new(AsyncMutex::new(stdin)),
            table,
            request_timeout: Duration::from_millis(150),
            _child: Arc::new(Mutex::new(Some(child))),
        };
        let err = client.list_tools().await.unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("timeout"), "got: {err}");
    }
}
