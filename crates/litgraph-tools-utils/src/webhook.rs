//! WebhookTool — POST agent messages to Slack / Discord / generic
//! incoming webhooks.
//!
//! # Security model
//!
//! Webhook URLs are SECRETS (anyone with one can post to the channel).
//! The URL is hard-coded at construction time, NOT passed as a tool
//! arg — so a prompt-injected agent can't pivot the target channel.
//! Agent only chooses the message text.
//!
//! For multi-channel setups, construct MULTIPLE WebhookTool instances
//! (one per channel) and let the agent pick by tool name. This is the
//! recommended pattern — better than accepting channel names as args
//! because the agent CAN'T be talked into posting to an arbitrary
//! channel it shouldn't reach.
//!
//! # Presets
//!
//! - **Slack incoming webhooks**: `{text: "..."}` simple payload.
//!   Blocks API is out of scope here — use generic preset + full JSON
//!   body if you need blocks.
//! - **Discord**: `{content: "..."}`.
//! - **Generic**: send whatever JSON the agent produces. Useful for
//!   PagerDuty / Opsgenie / self-hosted bots with custom schemas.
//!
//! # Tool args (LLM-facing)
//!
//! ```json
//! {
//!   "message": "Deploy succeeded. Commit: abc123",
//!   "username": "deploy-bot"     // optional, Slack only
//! }
//! ```

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

/// Wire format preset. Determines the JSON payload shape the tool
/// builds from the agent's `message`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebhookPreset {
    /// `{text: "message", username?: "..."}` — Slack incoming webhook.
    Slack,
    /// `{content: "message", username?: "..."}` — Discord webhook.
    Discord,
    /// Agent's message is the ENTIRE request body (must be valid JSON).
    /// Use when the target expects a custom schema (PagerDuty, Opsgenie,
    /// custom bot). Prompt engineering responsibility shifts to the
    /// caller — they must instruct the model what shape to emit.
    Generic,
}

#[derive(Debug, Clone)]
pub struct WebhookConfig {
    pub url: String,
    pub preset: WebhookPreset,
    pub timeout: Duration,
    /// Tool name in the agent's catalog. Defaults to `webhook_notify`
    /// but operators usually want a descriptive name per instance
    /// (e.g. `notify_oncall_channel`).
    pub tool_name: String,
    /// Description presented to the agent. Defaults to a generic string;
    /// operators should override with context (audience, appropriate
    /// triggers, tone) so the agent uses the tool judiciously.
    pub description: String,
}

impl WebhookConfig {
    pub fn slack(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            preset: WebhookPreset::Slack,
            timeout: Duration::from_secs(10),
            tool_name: "slack_notify".into(),
            description:
                "Post a message to a fixed Slack channel. Use SPARINGLY for \
                 important signals (escalations, completed tasks, errors). \
                 Do not spam."
                    .into(),
        }
    }

    pub fn discord(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            preset: WebhookPreset::Discord,
            timeout: Duration::from_secs(10),
            tool_name: "discord_notify".into(),
            description:
                "Post a message to a fixed Discord channel. Use SPARINGLY \
                 for important signals. Do not spam."
                    .into(),
        }
    }

    pub fn generic(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            preset: WebhookPreset::Generic,
            timeout: Duration::from_secs(10),
            tool_name: "webhook_post".into(),
            description:
                "POST a JSON payload to a fixed webhook URL. The `message` \
                 argument must be the FULL JSON body the endpoint expects."
                    .into(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self
    }
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
}

pub struct WebhookTool {
    cfg: Arc<WebhookConfig>,
    http: Client,
}

impl WebhookTool {
    pub fn new(cfg: WebhookConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("webhook build: {e}")))?;
        Ok(Self {
            cfg: Arc::new(cfg),
            http,
        })
    }
}

#[derive(Debug, Deserialize)]
struct WebhookArgs {
    message: String,
    #[serde(default)]
    username: Option<String>,
}

#[async_trait]
impl Tool for WebhookTool {
    fn schema(&self) -> ToolSchema {
        let params = match self.cfg.preset {
            WebhookPreset::Slack | WebhookPreset::Discord => json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Text to send. Keep under ~2000 chars."
                    },
                    "username": {
                        "type": "string",
                        "description": "Optional sender display name."
                    }
                },
                "required": ["message"]
            }),
            WebhookPreset::Generic => json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "FULL JSON body (as a string) for the webhook endpoint. Must be valid JSON — this tool forwards it verbatim as the POST body."
                    }
                },
                "required": ["message"]
            }),
        };
        ToolSchema {
            name: self.cfg.tool_name.clone(),
            description: self.cfg.description.clone(),
            parameters: params,
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let parsed: WebhookArgs = serde_json::from_value(args)
            .map_err(|e| Error::invalid(format!("webhook args: {e}")))?;
        if parsed.message.is_empty() {
            return Err(Error::invalid("webhook: message must be non-empty"));
        }

        let body = match self.cfg.preset {
            WebhookPreset::Slack => {
                let mut b = json!({"text": parsed.message});
                if let Some(u) = parsed.username {
                    b["username"] = Value::String(u);
                }
                b
            }
            WebhookPreset::Discord => {
                let mut b = json!({"content": parsed.message});
                if let Some(u) = parsed.username {
                    b["username"] = Value::String(u);
                }
                b
            }
            WebhookPreset::Generic => serde_json::from_str(&parsed.message)
                .map_err(|e| Error::invalid(format!("webhook generic: body not valid JSON: {e}")))?,
        };

        let resp = self
            .http
            .post(&self.cfg.url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("webhook send: {e}")))?;

        let status = resp.status().as_u16();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::other(format!("webhook read body: {e}")))?;

        // Slack + Discord both return 200 with a short success body.
        // 4xx indicates bad URL or malformed payload — surface as error
        // so the agent knows the notification didn't land.
        if !(200..=299).contains(&status) {
            return Err(Error::provider(format!(
                "webhook {}: {}",
                status,
                truncate(&text, 512)
            )));
        }

        Ok(json!({
            "status_code": status,
            "body": truncate(&text, 512),
        }))
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut out = s[..max].to_string();
    out.push_str("...");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};

    fn spawn_fake(
        status: u16,
        response_body: String,
    ) -> (
        String,
        std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}/hook");
        let captured_body = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let cb = captured_body.clone();
        std::thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                handle_one(stream, &cb, status, &response_body);
            }
        });
        (url, captured_body)
    }

    fn handle_one(
        mut stream: TcpStream,
        captured_body: &std::sync::Mutex<Vec<u8>>,
        status: u16,
        response_body: &str,
    ) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            if line == "\r\n" {
                break;
            }
            if line.to_ascii_lowercase().starts_with("content-length:") {
                content_length = line[15..].trim().parse().unwrap_or(0);
            }
        }
        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body).unwrap();
        }
        captured_body.lock().unwrap().extend_from_slice(&body);

        let resp = format!(
            "HTTP/1.1 {} OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            status,
            response_body.len()
        );
        stream.write_all(resp.as_bytes()).unwrap();
        stream.write_all(response_body.as_bytes()).unwrap();
    }

    #[tokio::test]
    async fn slack_preset_sends_text_field_payload() {
        let (url, body) = spawn_fake(200, "ok".to_string());
        let tool = WebhookTool::new(WebhookConfig::slack(url)).unwrap();
        let out = tool
            .run(json!({"message": "deploy done", "username": "deploy-bot"}))
            .await
            .unwrap();
        assert_eq!(out["status_code"], 200);
        let sent: Value = serde_json::from_slice(&body.lock().unwrap()).unwrap();
        assert_eq!(sent["text"], "deploy done");
        assert_eq!(sent["username"], "deploy-bot");
    }

    #[tokio::test]
    async fn discord_preset_sends_content_field_payload() {
        let (url, body) = spawn_fake(204, "".to_string());
        let tool = WebhookTool::new(WebhookConfig::discord(url)).unwrap();
        let out = tool
            .run(json!({"message": "hello world"}))
            .await
            .unwrap();
        assert_eq!(out["status_code"], 204);
        let sent: Value = serde_json::from_slice(&body.lock().unwrap()).unwrap();
        assert_eq!(sent["content"], "hello world");
        // No username → field absent.
        assert!(sent.get("username").is_none());
    }

    #[tokio::test]
    async fn generic_preset_forwards_arbitrary_json_body() {
        let (url, body) = spawn_fake(200, "ok".to_string());
        let tool = WebhookTool::new(WebhookConfig::generic(url)).unwrap();
        // Agent emits a custom schema.
        let msg = r#"{"event": "page", "severity": "high", "summary": "disk full"}"#;
        let out = tool.run(json!({"message": msg})).await.unwrap();
        assert_eq!(out["status_code"], 200);
        let sent: Value = serde_json::from_slice(&body.lock().unwrap()).unwrap();
        assert_eq!(sent["event"], "page");
        assert_eq!(sent["severity"], "high");
    }

    #[tokio::test]
    async fn generic_preset_rejects_invalid_json() {
        let tool = WebhookTool::new(WebhookConfig::generic("http://127.0.0.1:1")).unwrap();
        let err = tool
            .run(json!({"message": "not json"}))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn empty_message_rejected() {
        let tool = WebhookTool::new(WebhookConfig::slack("http://127.0.0.1:1")).unwrap();
        let err = tool.run(json!({"message": ""})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn upstream_4xx_surfaces_as_error() {
        let (url, _body) = spawn_fake(400, "invalid_payload".to_string());
        let tool = WebhookTool::new(WebhookConfig::slack(url)).unwrap();
        let err = tool
            .run(json!({"message": "hi"}))
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("400"));
        assert!(msg.contains("invalid_payload"));
    }

    #[tokio::test]
    async fn tool_schema_reflects_preset() {
        let slack = WebhookTool::new(WebhookConfig::slack("http://x")).unwrap();
        let s = slack.schema();
        assert_eq!(s.name, "slack_notify");
        assert!(s.parameters["properties"].get("username").is_some());

        let generic = WebhookTool::new(WebhookConfig::generic("http://x")).unwrap();
        let g = generic.schema();
        assert_eq!(g.name, "webhook_post");
        // Generic has NO username field (model picks whole body).
        assert!(g.parameters["properties"].get("username").is_none());
    }

    #[tokio::test]
    async fn with_name_and_description_override_defaults() {
        let cfg = WebhookConfig::slack("http://x")
            .with_name("notify_oncall")
            .with_description("Use only for P1 incidents.");
        let tool = WebhookTool::new(cfg).unwrap();
        let s = tool.schema();
        assert_eq!(s.name, "notify_oncall");
        assert_eq!(s.description, "Use only for P1 incidents.");
    }

    #[tokio::test]
    async fn url_is_not_exposed_in_tool_schema() {
        // Security invariant: URL is a secret, must not leak into the
        // agent's tool catalog (otherwise the LLM can memorize + exfil).
        let cfg = WebhookConfig::slack("https://hooks.slack.com/services/SECRET/TOKEN/abc");
        let tool = WebhookTool::new(cfg).unwrap();
        let s = tool.schema();
        let rendered = serde_json::to_string(&s.parameters).unwrap();
        assert!(!rendered.contains("hooks.slack.com"));
        assert!(!rendered.contains("SECRET"));
        assert!(!s.description.contains("hooks.slack.com"));
    }
}
