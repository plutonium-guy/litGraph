//! Model Context Protocol (MCP) client for litGraph — stdio + HTTP transports.
//!
//! MCP is Anthropic's open JSON-RPC 2.0 protocol for connecting LLMs to
//! external tools/resources/prompts. Servers exist for filesystem, github,
//! slack, postgres, sqlite, fetch, brave-search, puppeteer, etc.
//! See <https://modelcontextprotocol.io>.
//!
//! # Transports
//! - `connect_stdio(program, args)` — spawn child process, line-delimited JSON-RPC.
//! - `connect_http(url, headers)` — Streamable HTTP (per 2025-03-26 spec):
//!   POST one JSON-RPC message, get one JSON response back. Server may set
//!   `Mcp-Session-Id` on the initialize response; we echo it on every
//!   subsequent request to keep stateful sessions alive.
//!
//! # What's supported
//! - `initialize` handshake with capability declaration.
//! - `tools/list` — query the server's tools.
//! - `tools/call(name, args)` — execute a tool.
//! - `McpTool` adapter: each MCP tool implements `litgraph_core::Tool` so it
//!   plugs straight into `ReactAgent` alongside our native tools.
//!
//! # Not yet supported
//! - SSE response bodies (`text/event-stream`) on POST. We accept-header only
//!   `application/json`. Most hosted MCP servers honor this — if a server
//!   strictly returns SSE we'd need to add a streaming parser.
//! - WebSocket transport.
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

/// Transport-specific state. Stdio holds the child + write half (the read
/// half drives a background task that fills the PendingTable). HTTP holds
/// the reqwest client + URL + session id (filled lazily after `initialize`).
enum Transport {
    Stdio {
        stdin: Arc<AsyncMutex<ChildStdin>>,
        /// Keep the Child alive — `Command::spawn` with `.kill_on_drop(true)`
        /// will SIGKILL the process the moment this is dropped, so we MUST hold
        /// it on the client (not the spawn-local). Wrapped in `Mutex<Option>` so
        /// `Drop` can take it for an explicit kill if needed.
        _child: Arc<Mutex<Option<Child>>>,
    },
    Http {
        client: reqwest::Client,
        url: String,
        /// Set from the first response that carries `Mcp-Session-Id` (typically
        /// `initialize`); echoed on every subsequent request.
        session_id: Arc<Mutex<Option<String>>>,
        /// Caller-supplied headers (e.g. `Authorization: Bearer ...`). Sent
        /// on every request.
        default_headers: Vec<(String, String)>,
    },
}

impl Transport {
    /// Send one JSON-RPC message. For stdio, returns `None` because the
    /// reader task will dispatch any reply via the PendingTable. For HTTP,
    /// returns `Some(value)` if the server replied with a JSON body; `None`
    /// if the server returned an empty 2xx (notification ack).
    async fn send(&self, bytes: Vec<u8>) -> std::result::Result<Option<Value>, McpError> {
        match self {
            Transport::Stdio { stdin, .. } => {
                let mut s = stdin.lock().await;
                s.write_all(&bytes)
                    .await
                    .map_err(|e| McpError::Transport(format!("write: {e}")))?;
                s.write_all(b"\n")
                    .await
                    .map_err(|e| McpError::Transport(format!("write: {e}")))?;
                s.flush()
                    .await
                    .map_err(|e| McpError::Transport(format!("flush: {e}")))?;
                Ok(None)
            }
            Transport::Http { client, url, session_id, default_headers } => {
                let mut req = client
                    .post(url)
                    .header("content-type", "application/json")
                    // Accept BOTH JSON and SSE. Hosted MCP servers
                    // (Cloudflare Workers, Cloud Run, etc) often prefer SSE
                    // for keep-alive-friendly streaming — the client has
                    // to opt in via Accept or they'll fall back to 406.
                    .header("accept", "application/json, text/event-stream")
                    .body(bytes);
                if let Some(sid) = session_id.lock().clone() {
                    req = req.header("Mcp-Session-Id", sid);
                }
                for (k, v) in default_headers {
                    req = req.header(k.as_str(), v.as_str());
                }
                let resp = req
                    .send()
                    .await
                    .map_err(|e| McpError::Transport(format!("http: {e}")))?;
                let status = resp.status();
                // Capture session id BEFORE consuming the response body —
                // any subsequent request needs to echo it.
                if let Some(h) = resp.headers().get("Mcp-Session-Id") {
                    if let Ok(s) = h.to_str() {
                        *session_id.lock() = Some(s.to_string());
                    }
                }
                // Grab content-type BEFORE consuming the body — we need to
                // branch on it between SSE vs JSON parsing.
                let content_type = resp
                    .headers()
                    .get("content-type")
                    .and_then(|h| h.to_str().ok())
                    .unwrap_or("")
                    .to_string();
                let body_bytes = resp
                    .bytes()
                    .await
                    .map_err(|e| McpError::Transport(format!("http body: {e}")))?;
                if !status.is_success() {
                    let snippet = String::from_utf8_lossy(&body_bytes);
                    return Err(McpError::Transport(format!(
                        "http {}: {}",
                        status,
                        snippet.chars().take(200).collect::<String>()
                    )));
                }
                if body_bytes.is_empty() {
                    return Ok(None);
                }
                // Route on content-type. `text/event-stream` → SSE parser;
                // else (including missing / `application/json`) → straight JSON.
                if content_type.starts_with("text/event-stream") {
                    match parse_sse_first_json(&body_bytes)? {
                        Some(v) => Ok(Some(v)),
                        None => Ok(None),  // server sent only comments / `[DONE]`.
                    }
                } else {
                    let v: Value = serde_json::from_slice(&body_bytes)
                        .map_err(|e| McpError::Protocol(format!("bad json response: {e}")))?;
                    Ok(Some(v))
                }
            }
        }
    }
}

/// JSON-RPC client over an MCP server. Built via `connect_stdio` or
/// `connect_http`; thereafter both transports look identical to callers.
///
/// Drop = no explicit shutdown; for stdio, the child gets reaped when its
/// stdin closes; for HTTP, the connection pool stays warm in reqwest.
pub struct McpClient {
    transport: Transport,
    table: Arc<Mutex<PendingTable>>,
    request_timeout: Duration,
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
                        dispatch_response(&table_for_reader, v);
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
            transport: Transport::Stdio {
                stdin: Arc::new(AsyncMutex::new(stdin)),
                _child: Arc::new(Mutex::new(Some(child))),
            },
            table,
            request_timeout: Duration::from_secs(30),
        };

        client.handshake().await?;
        Ok(client)
    }

    /// Connect to a hosted MCP server over HTTP. `url` is the JSON-RPC POST
    /// endpoint (e.g. `https://mcp.example.com/v1`). `headers` is an
    /// iterable of `(name, value)` for auth tokens etc; sent on every request.
    pub async fn connect_http(
        url: impl Into<String>,
        headers: impl IntoIterator<Item = (String, String)>,
    ) -> std::result::Result<Self, McpError> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| McpError::Transport(format!("reqwest build: {e}")))?;
        let table = Arc::new(Mutex::new(PendingTable {
            next_id: 1,
            pending: HashMap::new(),
        }));
        let client = Self {
            transport: Transport::Http {
                client,
                url: url.into(),
                session_id: Arc::new(Mutex::new(None)),
                default_headers: headers.into_iter().collect(),
            },
            table,
            request_timeout: Duration::from_secs(30),
        };
        client.handshake().await?;
        Ok(client)
    }

    /// Run `initialize` + `notifications/initialized`. Shared by both transports.
    async fn handshake(&self) -> std::result::Result<(), McpError> {
        let _: Value = self
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
        self.notify("notifications/initialized", json!({})).await?;
        Ok(())
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
        match self.transport.send(bytes).await {
            Ok(Some(v)) => {
                // HTTP path: the response is right here. Dispatch it through
                // the same table so request/response correlation logic stays
                // in one place. (Server might still echo our id, or send back
                // a notification — `dispatch_response` handles both.)
                dispatch_response(&self.table, v);
            }
            Ok(None) => {
                // Stdio: reader task will fill the slot. HTTP empty body would
                // be a protocol violation for a request, but the timeout below
                // catches it.
            }
            Err(e) => {
                // Drop the pending slot so it doesn't leak.
                self.table.lock().pending.remove(&id);
                return Err(e);
            }
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
        // Notification: no id, no response expected. HTTP servers typically
        // return 202 Accepted with empty body; we ignore any body that does
        // come back (no id → can't correlate anyway).
        let _ = self.transport.send(bytes).await?;
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

/// Route one parsed JSON-RPC message to its waiter. Notifications (no `id`)
/// are dropped silently. Used by the stdio reader task AND by HTTP's inline
/// dispatch — both share the same correlation logic.
/// Parse an SSE body (`text/event-stream`) and return the first JSON-RPC
/// payload. Ignores `:comment` lines, empty lines, and `data: [DONE]`
/// sentinels. Multi-line `data:` values (spec-legal — multiple `data:` lines
/// per event concatenate with `\n`) are joined. Returns `Ok(None)` if the
/// body contains only comments / `[DONE]` — that's a valid ack shape for
/// notifications.
fn parse_sse_first_json(bytes: &[u8]) -> std::result::Result<Option<Value>, McpError> {
    let body = std::str::from_utf8(bytes)
        .map_err(|e| McpError::Protocol(format!("sse utf8: {e}")))?;
    // SSE events are separated by blank lines. Within one event, multiple
    // `data:` lines concatenate with `\n` per the spec. We only need the
    // FIRST real-JSON event (MCP over HTTP returns a single response per
    // request) — later events / keep-alives are ignored.
    let mut current_data = String::new();
    for line in body.lines() {
        if line.is_empty() {
            // Event boundary. If we've accumulated a non-DONE payload, try
            // parsing it.
            let trimmed = current_data.trim();
            if !trimmed.is_empty() && trimmed != "[DONE]" {
                return serde_json::from_str::<Value>(trimmed)
                    .map(Some)
                    .map_err(|e| {
                        McpError::Protocol(format!(
                            "sse json parse: {e}\n--- payload ---\n{trimmed}"
                        ))
                    });
            }
            current_data.clear();
            continue;
        }
        if line.starts_with(':') {
            // SSE comment / keep-alive — skip.
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            // The leading space after `data:` is optional per spec; trim
            // exactly one if present.
            let payload = rest.strip_prefix(' ').unwrap_or(rest);
            if !current_data.is_empty() {
                current_data.push('\n');
            }
            current_data.push_str(payload);
        }
        // `event:` / `id:` / `retry:` lines are parsed only for completeness;
        // MCP doesn't use them. Ignore silently.
    }
    // EOF without trailing blank line — also an event boundary.
    let trimmed = current_data.trim();
    if !trimmed.is_empty() && trimmed != "[DONE]" {
        return serde_json::from_str::<Value>(trimmed)
            .map(Some)
            .map_err(|e| {
                McpError::Protocol(format!(
                    "sse json parse: {e}\n--- payload ---\n{trimmed}"
                ))
            });
    }
    Ok(None)
}

fn dispatch_response(table: &Arc<Mutex<PendingTable>>, v: Value) {
    let id = match v.get("id").and_then(|i| i.as_u64()) {
        Some(id) => id,
        None => {
            debug!(method = ?v.get("method"), "mcp: notification ignored");
            return;
        }
    };
    let mut t = table.lock();
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
    use tokio::io::AsyncReadExt;

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
            transport: Transport::Stdio {
                stdin: Arc::new(AsyncMutex::new(stdin)),
                _child: Arc::new(Mutex::new(Some(child))),
            },
            table,
            request_timeout: Duration::from_millis(150),
        };
        let err = client.list_tools().await.unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("timeout"), "got: {err}");
    }

    // ---------------------------------------------------------------- HTTP

    /// Tiny TCP-level fake HTTP server for the MCP HTTP-transport tests.
    /// Reads ONE request, dispatches by JSON-RPC method, writes one JSON
    /// response (with optional `Mcp-Session-Id` header), closes the connection.
    /// Configure with `behavior` callbacks so each test can shape the reply
    /// without copy-pasting the socket plumbing.
    struct FakeHttpServer {
        url: String,
        seen_session_ids: Arc<Mutex<Vec<Option<String>>>>,
        seen_auth: Arc<Mutex<Vec<Option<String>>>>,
        _shutdown: oneshot::Sender<()>,
    }

    fn parse_http_request(buf: &[u8]) -> Option<(String, Vec<u8>)> {
        let split = buf.windows(4).position(|w| w == b"\r\n\r\n")?;
        let head = std::str::from_utf8(&buf[..split]).ok()?.to_string();
        let body = buf[split + 4..].to_vec();
        Some((head, body))
    }

    fn header_value<'a>(head: &'a str, name: &str) -> Option<&'a str> {
        for line in head.lines().skip(1) {
            if let Some((k, v)) = line.split_once(':') {
                if k.trim().eq_ignore_ascii_case(name) {
                    return Some(v.trim());
                }
            }
        }
        None
    }

    async fn read_full_request(
        stream: &mut tokio::net::TcpStream,
    ) -> std::io::Result<(String, Vec<u8>)> {
        let mut buf = Vec::with_capacity(4096);
        let mut chunk = [0u8; 4096];
        let head;
        let mut body;
        loop {
            let n = stream.read(&mut chunk).await?;
            if n == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof"));
            }
            buf.extend_from_slice(&chunk[..n]);
            if let Some((h, b)) = parse_http_request(&buf) {
                head = h;
                body = b;
                break;
            }
        }
        // If body is shorter than Content-Length, read more.
        let cl: usize = header_value(&head, "content-length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        while body.len() < cl {
            let n = stream.read(&mut chunk).await?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..n]);
        }
        Ok((head, body))
    }

    /// Spawn a fake HTTP MCP server. `respond` returns the JSON body for a
    /// given request; the server adds `Mcp-Session-Id: test-sid` on the
    /// initialize reply only (so the client must echo it on later requests).
    async fn fake_http_server() -> FakeHttpServer {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen_session_ids: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_auth: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();

        let session_log = seen_session_ids.clone();
        let auth_log = seen_auth.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    Ok((mut stream, _)) = listener.accept() => {
                        let session_log = session_log.clone();
                        let auth_log = auth_log.clone();
                        tokio::spawn(async move {
                            let (head, body) = match read_full_request(&mut stream).await {
                                Ok(t) => t,
                                Err(_) => return,
                            };
                            session_log.lock().push(
                                header_value(&head, "Mcp-Session-Id").map(|s| s.to_string())
                            );
                            auth_log.lock().push(
                                header_value(&head, "Authorization").map(|s| s.to_string())
                            );
                            let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                            let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
                            let rid = req.get("id").cloned();
                            let mut extra_headers = String::new();
                            let body_value = match method {
                                "initialize" => {
                                    extra_headers.push_str("Mcp-Session-Id: test-sid\r\n");
                                    json!({
                                        "jsonrpc":"2.0", "id": rid,
                                        "result": {
                                            "protocolVersion":"2024-11-05",
                                            "capabilities":{"tools":{}},
                                            "serverInfo":{"name":"fake-http","version":"0"},
                                        }
                                    })
                                }
                                "notifications/initialized" => {
                                    // 202-style empty ack. Write headers + zero body.
                                    let resp = "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n";
                                    let _ = stream.write_all(resp.as_bytes()).await;
                                    return;
                                }
                                "tools/list" => json!({
                                    "jsonrpc":"2.0", "id": rid,
                                    "result": {"tools":[
                                        {"name":"echo","description":"Echoes back.",
                                         "inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},
                                    ]}
                                }),
                                "tools/call" => {
                                    let p = req.get("params").cloned().unwrap_or(Value::Null);
                                    let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                    let args = p.get("arguments").cloned().unwrap_or(Value::Null);
                                    if name == "echo" {
                                        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                        json!({
                                            "jsonrpc":"2.0", "id": rid,
                                            "result": {"content":[{"type":"text","text": format!("you said: {text}")}], "isError": false}
                                        })
                                    } else {
                                        json!({
                                            "jsonrpc":"2.0", "id": rid,
                                            "error": {"code":-32601,"message":"unknown tool"}
                                        })
                                    }
                                }
                                _ => json!({
                                    "jsonrpc":"2.0", "id": rid,
                                    "error": {"code":-32601,"message":"unknown method"}
                                }),
                            };
                            let body_bytes = serde_json::to_vec(&body_value).unwrap();
                            let resp = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}\r\n",
                                body_bytes.len(),
                                extra_headers,
                            );
                            let _ = stream.write_all(resp.as_bytes()).await;
                            let _ = stream.write_all(&body_bytes).await;
                        });
                    }
                }
            }
        });

        FakeHttpServer {
            url: format!("http://127.0.0.1:{port}/mcp"),
            seen_session_ids,
            seen_auth,
            _shutdown: shutdown_tx,
        }
    }

    #[tokio::test]
    async fn http_lists_and_calls_tools_against_fake_server() {
        let server = fake_http_server().await;
        let client = McpClient::connect_http(&server.url, std::iter::empty())
            .await
            .expect("connect_http should succeed");
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        let res = client.call_tool("echo", json!({"text": "world"})).await.unwrap();
        assert_eq!(res["content"][0]["text"].as_str().unwrap(), "you said: world");
    }

    #[tokio::test]
    async fn http_session_id_is_captured_then_echoed() {
        let server = fake_http_server().await;
        let client = McpClient::connect_http(&server.url, std::iter::empty()).await.unwrap();
        // Trigger a couple more requests after handshake.
        client.list_tools().await.unwrap();
        client.call_tool("echo", json!({"text":"hi"})).await.unwrap();
        let log = server.seen_session_ids.lock().clone();
        // Order: initialize (no sid), notifications/initialized (sid set),
        // tools/list (sid), tools/call (sid).
        assert_eq!(log.len(), 4, "got: {:?}", log);
        assert_eq!(log[0], None, "initialize must not carry a session id");
        for (i, sid) in log.iter().enumerate().skip(1) {
            assert_eq!(sid.as_deref(), Some("test-sid"), "request {i}: expected echoed sid");
        }
    }

    #[tokio::test]
    async fn http_default_headers_are_sent_on_every_request() {
        let server = fake_http_server().await;
        let headers = vec![("Authorization".to_string(), "Bearer secret-token".to_string())];
        let client = McpClient::connect_http(&server.url, headers).await.unwrap();
        client.list_tools().await.unwrap();
        let log = server.seen_auth.lock().clone();
        assert!(log.len() >= 2);
        for (i, v) in log.iter().enumerate() {
            assert_eq!(v.as_deref(), Some("Bearer secret-token"), "request {i}: missing auth");
        }
    }

    #[tokio::test]
    async fn http_server_error_surfaces_as_mcp_server_error() {
        let server = fake_http_server().await;
        let client = McpClient::connect_http(&server.url, std::iter::empty()).await.unwrap();
        let err = client.call_tool("nope", json!({})).await.unwrap_err();
        assert!(format!("{err}").contains("unknown tool"), "got: {err}");
    }

    // ---------- SSE response support (iter 97) ----------

    #[test]
    fn parse_sse_first_json_extracts_single_event() {
        let body = b"data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n\n";
        let v = parse_sse_first_json(body).unwrap().unwrap();
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["id"], 1);
        assert_eq!(v["result"]["ok"], true);
    }

    #[test]
    fn parse_sse_first_json_ignores_comments_and_keepalives() {
        // Comment lines start with `:`. Some servers emit periodic
        // `:keep-alive` to hold long-polling connections open.
        let body = b": keep-alive\n\n:another\n\ndata: {\"id\":42,\"result\":1}\n\n";
        let v = parse_sse_first_json(body).unwrap().unwrap();
        assert_eq!(v["id"], 42);
    }

    #[test]
    fn parse_sse_first_json_joins_multiline_data_with_newlines() {
        // Per spec, multiple consecutive `data:` lines in one event
        // concatenate with `\n` — we preserve that so the payload parses
        // correctly when JSON is pretty-printed across lines.
        let body = b"data: {\n\
                     data:   \"id\": 3,\n\
                     data:   \"result\": \"ok\"\n\
                     data: }\n\n";
        let v = parse_sse_first_json(body).unwrap().unwrap();
        assert_eq!(v["id"], 3);
        assert_eq!(v["result"], "ok");
    }

    #[test]
    fn parse_sse_first_json_handles_done_sentinel_and_comments_only() {
        // Body is all keep-alives + a terminal [DONE]. Result: None — caller
        // treats that as "server ack with no reply" (notifications/initialized).
        let body = b": keep\n\ndata: [DONE]\n\n";
        assert!(parse_sse_first_json(body).unwrap().is_none());
    }

    #[test]
    fn parse_sse_first_json_errors_with_payload_in_message_on_bad_json() {
        let body = b"data: not valid json\n\n";
        let err = parse_sse_first_json(body).unwrap_err();
        let msg = format!("{err}");
        // Error includes the raw payload — debugging against a live SSE
        // server that's emitting malformed JSON is hard without it.
        assert!(msg.contains("not valid json"), "got: {msg}");
    }

    #[test]
    fn parse_sse_first_json_tolerates_missing_trailing_blank_line() {
        // Real servers sometimes close without the trailing \n\n — we
        // treat EOF as an event boundary so the response still parses.
        let body = b"data: {\"id\":1,\"result\":7}";
        let v = parse_sse_first_json(body).unwrap().unwrap();
        assert_eq!(v["result"], 7);
    }

    /// Fake HTTP/1.1 server that returns its replies as `text/event-stream`
    /// instead of `application/json`. Otherwise identical to the iter-75
    /// `fake_http_server()` fixture. Tests the client's SSE branch end-to-end.
    async fn fake_http_sse_server() -> FakeHttpServer {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen_session_ids: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_auth: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();

        let session_log = seen_session_ids.clone();
        let auth_log = seen_auth.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    Ok((mut stream, _)) = listener.accept() => {
                        let session_log = session_log.clone();
                        let auth_log = auth_log.clone();
                        tokio::spawn(async move {
                            let (head, body) = match read_full_request(&mut stream).await {
                                Ok(t) => t,
                                Err(_) => return,
                            };
                            session_log.lock().push(
                                header_value(&head, "Mcp-Session-Id").map(|s| s.to_string())
                            );
                            auth_log.lock().push(
                                header_value(&head, "Authorization").map(|s| s.to_string())
                            );
                            // Assert the client actually asked for SSE — this
                            // is what real hosted servers gate on.
                            let accept = header_value(&head, "Accept").unwrap_or("");
                            assert!(
                                accept.contains("text/event-stream"),
                                "client must send Accept: text/event-stream; got: {accept}"
                            );
                            let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                            let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
                            let rid = req.get("id").cloned();
                            let mut extra_headers = String::new();
                            let body_value = match method {
                                "initialize" => {
                                    extra_headers.push_str("Mcp-Session-Id: sse-sid\r\n");
                                    json!({
                                        "jsonrpc":"2.0", "id": rid,
                                        "result": {
                                            "protocolVersion":"2024-11-05",
                                            "capabilities":{"tools":{}},
                                            "serverInfo":{"name":"fake-sse","version":"0"},
                                        }
                                    })
                                }
                                "notifications/initialized" => {
                                    // 202 empty — same as JSON path, no SSE body.
                                    let resp = "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n";
                                    let _ = stream.write_all(resp.as_bytes()).await;
                                    return;
                                }
                                "tools/list" => json!({
                                    "jsonrpc":"2.0", "id": rid,
                                    "result": {"tools":[
                                        {"name":"sse_echo","description":"SSE echo.",
                                         "inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},
                                    ]}
                                }),
                                "tools/call" => {
                                    let p = req.get("params").cloned().unwrap_or(Value::Null);
                                    let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                    let args = p.get("arguments").cloned().unwrap_or(Value::Null);
                                    if name == "sse_echo" {
                                        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                        json!({
                                            "jsonrpc":"2.0", "id": rid,
                                            "result": {"content":[{"type":"text","text": format!("sse said: {text}")}], "isError": false}
                                        })
                                    } else {
                                        json!({
                                            "jsonrpc":"2.0", "id": rid,
                                            "error": {"code":-32601,"message":"unknown tool"}
                                        })
                                    }
                                }
                                _ => json!({
                                    "jsonrpc":"2.0", "id": rid,
                                    "error": {"code":-32601,"message":"unknown method"}
                                }),
                            };
                            // Wrap the JSON-RPC response as an SSE event. Start
                            // with a keep-alive comment to prove the parser
                            // tolerates them.
                            let payload = serde_json::to_string(&body_value).unwrap();
                            let sse = format!(": keep\n\ndata: {payload}\n\n");
                            let resp = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n{}\r\n",
                                sse.len(),
                                extra_headers,
                            );
                            let _ = stream.write_all(resp.as_bytes()).await;
                            let _ = stream.write_all(sse.as_bytes()).await;
                        });
                    }
                }
            }
        });

        FakeHttpServer {
            url: format!("http://127.0.0.1:{port}/mcp"),
            seen_session_ids,
            seen_auth,
            _shutdown: shutdown_tx,
        }
    }

    #[tokio::test]
    async fn http_client_speaks_sse_against_sse_server() {
        // The full production path: hosted server replies with
        // `Content-Type: text/event-stream`, client parses SSE, returns
        // the JSON-RPC result as if it had been plain JSON.
        let server = fake_http_sse_server().await;
        let client = McpClient::connect_http(&server.url, std::iter::empty())
            .await
            .expect("SSE handshake must succeed");
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "sse_echo");
        let res = client.call_tool("sse_echo", json!({"text": "from sse"})).await.unwrap();
        assert_eq!(res["content"][0]["text"].as_str().unwrap(), "sse said: from sse");
    }

    #[tokio::test]
    async fn http_sse_session_id_still_captured_from_sse_response_header() {
        // Mcp-Session-Id lives in HTTP response HEADERS (not the SSE body),
        // so session stickiness must work regardless of body shape.
        let server = fake_http_sse_server().await;
        let client = McpClient::connect_http(&server.url, std::iter::empty()).await.unwrap();
        client.list_tools().await.unwrap();
        let log = server.seen_session_ids.lock().clone();
        assert!(log.len() >= 3);
        assert_eq!(log[0], None, "initialize must not carry a session id");
        for sid in log.iter().skip(1) {
            assert_eq!(sid.as_deref(), Some("sse-sid"));
        }
    }
}
