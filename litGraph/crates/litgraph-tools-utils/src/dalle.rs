//! DALL-E / image-generation tool — POST to OpenAI-compatible
//! `/images/generations`. Closes the image-OUTPUT modality gap (image
//! INPUT is already covered via `ContentPart::Image` on chat models;
//! audio INPUT via the iter-114 Whisper tool).
//!
//! # Compatibility
//!
//! OpenAI is the reference but the same wire shape is used by:
//! - Self-hosted Stable Diffusion through OpenAI-compatible proxies
//!   (e.g. `comfyui-openai`, `sdwebui-openai`)
//! - Some hosted SD providers (Together's image endpoint, etc.)
//!
//! # Tool args (LLM-facing)
//!
//! ```json
//! {
//!   "prompt":          "a watercolor of a cat sleeping in a sunbeam",
//!   "size":            "1024x1024",        // optional
//!   "quality":         "standard",         // "standard" | "hd"
//!   "n":               1,                  // dall-e-3 only allows 1
//!   "response_format": "url"               // "url" | "b64_json"
//! }
//! ```
//!
//! # Why fixed JSON output shape
//!
//! Whisper (iter 114) returns a string for the simple `json` case. Image
//! gen always returns a list (n images, each with a url OR b64_json).
//! Return shape: `{"images": [{"url": "..."} | {"b64_json": "..."}, ...]}`.
//! Predictable for the LLM to reason about: "look at images[0].url".

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "dall-e-3";

#[derive(Debug, Clone)]
pub struct DalleConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
}

impl DalleConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            timeout: Duration::from_secs(120),
        }
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
}

pub struct DalleImageTool {
    cfg: Arc<DalleConfig>,
    http: Client,
}

impl DalleImageTool {
    pub fn new(cfg: DalleConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("dalle build: {e}")))?;
        Ok(Self {
            cfg: Arc::new(cfg),
            http,
        })
    }
}

#[derive(Debug, Deserialize)]
struct DalleArgs {
    prompt: String,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    quality: Option<String>,
    #[serde(default)]
    n: Option<u8>,
    #[serde(default)]
    response_format: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiError {
    error: OpenAiErrorBody,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorBody {
    message: String,
    #[serde(default)]
    #[allow(dead_code)]
    code: Option<String>,
}

#[async_trait]
impl Tool for DalleImageTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "image_generate".to_string(),
            description: "Generate one or more images from a text prompt using a DALL-E-compatible \
                          endpoint. Required: `prompt` (description of the desired image). Optional: \
                          `size` (e.g. \"1024x1024\", \"1024x1792\", \"1792x1024\"), `quality` \
                          (\"standard\" | \"hd\"), `n` (number of images, dall-e-3 enforces 1), \
                          `response_format` (\"url\" returns hosted CDN URLs, default; \"b64_json\" \
                          returns base64-encoded PNGs inline). Returns `{images: [{url|b64_json: ...}, ...]}`."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "size": {
                        "type": "string",
                        "description": "Output dimensions like \"1024x1024\". DALL-E 3 supports 1024x1024, 1024x1792, 1792x1024.",
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["standard", "hd"],
                        "description": "DALL-E 3 only. \"hd\" costs ~2x more.",
                    },
                    "n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Number of images. dall-e-3 enforces 1.",
                    },
                    "response_format": {
                        "type": "string",
                        "enum": ["url", "b64_json"],
                        "description": "url returns hosted CDN URLs (expire in ~1h). b64_json returns base64 PNGs inline (huge payload).",
                    }
                },
                "required": ["prompt"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let parsed: DalleArgs = serde_json::from_value(args)
            .map_err(|e| Error::invalid(format!("dalle args: {e}")))?;

        let mut body = json!({
            "model": self.cfg.model,
            "prompt": parsed.prompt,
        });
        if let Some(s) = parsed.size {
            body["size"] = Value::String(s);
        }
        if let Some(q) = parsed.quality {
            body["quality"] = Value::String(q);
        }
        if let Some(n) = parsed.n {
            body["n"] = Value::from(n);
        }
        if let Some(f) = parsed.response_format {
            body["response_format"] = Value::String(f);
        }

        let url = format!("{}/images/generations", self.cfg.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("dalle send: {e}")))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::other(format!("dalle read body: {e}")))?;

        if !status.is_success() {
            let msg = serde_json::from_str::<OpenAiError>(&text)
                .map(|e| e.error.message)
                .unwrap_or_else(|_| text.clone());
            return Err(Error::provider(format!(
                "dalle {}: {msg}",
                status.as_u16()
            )));
        }

        // Response shape: {"created": ts, "data": [{"url": ...} | {"b64_json": ...}, ...]}.
        // Normalize to {"images": [{"url"|"b64_json": ...}, ...]} — drop the
        // `created` field (rarely useful to the LLM, adds noise).
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| Error::other(format!("dalle parse json: {e}")))?;
        let data = v
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(json!({ "images": data }))
    }
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
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}/v1");
        let captured_body = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_headers = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let cb = captured_body.clone();
        let ch = captured_headers.clone();

        std::thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                handle_one(stream, &cb, &ch, status, &response_body);
            }
        });

        (url, captured_body, captured_headers)
    }

    fn handle_one(
        mut stream: TcpStream,
        captured_body: &std::sync::Mutex<Vec<u8>>,
        captured_headers: &std::sync::Mutex<Vec<String>>,
        status: u16,
        response_body: &str,
    ) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut headers = Vec::<String>::new();
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
            headers.push(line.trim_end().to_string());
        }
        captured_headers.lock().unwrap().extend(headers);
        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body).unwrap();
        }
        captured_body.lock().unwrap().extend_from_slice(&body);

        let resp_body_bytes = response_body.as_bytes();
        let response = format!(
            "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            status,
            resp_body_bytes.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
        stream.write_all(resp_body_bytes).unwrap();
    }

    #[tokio::test]
    async fn returns_normalized_images_array_for_url_response() {
        let body = r#"{"created": 123, "data": [{"url": "https://cdn/img1.png"}]}"#;
        let (url, captured_body, captured_headers) = spawn_fake(200, body.to_string());
        let cfg = DalleConfig::new("sk-test").with_base_url(&url);
        let tool = DalleImageTool::new(cfg).unwrap();
        let out = tool
            .run(json!({"prompt": "a cat in a sunbeam"}))
            .await
            .unwrap();
        assert_eq!(out["images"][0]["url"], "https://cdn/img1.png");
        // No "created" field bleeds through.
        assert!(out.get("created").is_none());

        // Bearer auth + JSON body sent.
        let headers = captured_headers.lock().unwrap().clone();
        assert!(headers.iter().any(|h| h.to_ascii_lowercase().contains("authorization: bearer sk-test")));
        let req: Value = serde_json::from_slice(&captured_body.lock().unwrap()).unwrap();
        assert_eq!(req["prompt"], "a cat in a sunbeam");
        assert_eq!(req["model"], "dall-e-3");
    }

    #[tokio::test]
    async fn b64_response_passes_through_inline() {
        let body = r#"{"data": [{"b64_json": "aGVsbG8="}]}"#;
        let (url, _b, _h) = spawn_fake(200, body.to_string());
        let cfg = DalleConfig::new("k").with_base_url(&url);
        let tool = DalleImageTool::new(cfg).unwrap();
        let out = tool
            .run(json!({
                "prompt": "x",
                "response_format": "b64_json",
            }))
            .await
            .unwrap();
        assert_eq!(out["images"][0]["b64_json"], "aGVsbG8=");
    }

    #[tokio::test]
    async fn multi_image_request_returns_all_images() {
        let body = r#"{"data": [{"url": "https://cdn/a.png"}, {"url": "https://cdn/b.png"}]}"#;
        let (url, captured_body, _h) = spawn_fake(200, body.to_string());
        let cfg = DalleConfig::new("k").with_base_url(&url).with_model("dall-e-2");
        let tool = DalleImageTool::new(cfg).unwrap();
        let out = tool
            .run(json!({"prompt": "x", "n": 2}))
            .await
            .unwrap();
        let imgs = out["images"].as_array().unwrap();
        assert_eq!(imgs.len(), 2);
        // Verify request body had n=2 and model=dall-e-2.
        let req: Value = serde_json::from_slice(&captured_body.lock().unwrap()).unwrap();
        assert_eq!(req["n"], 2);
        assert_eq!(req["model"], "dall-e-2");
    }

    #[tokio::test]
    async fn size_and_quality_propagated_to_request_body() {
        let body = r#"{"data": [{"url": "https://cdn/x.png"}]}"#;
        let (url, captured_body, _h) = spawn_fake(200, body.to_string());
        let cfg = DalleConfig::new("k").with_base_url(&url);
        let tool = DalleImageTool::new(cfg).unwrap();
        let _ = tool
            .run(json!({
                "prompt": "x",
                "size": "1024x1792",
                "quality": "hd",
            }))
            .await
            .unwrap();
        let req: Value = serde_json::from_slice(&captured_body.lock().unwrap()).unwrap();
        assert_eq!(req["size"], "1024x1792");
        assert_eq!(req["quality"], "hd");
    }

    #[tokio::test]
    async fn upstream_4xx_surfaces_openai_error_message() {
        let body = r#"{"error": {"message": "Your prompt was rejected.", "code": "content_policy_violation"}}"#;
        let (url, _b, _h) = spawn_fake(400, body.to_string());
        let cfg = DalleConfig::new("k").with_base_url(&url);
        let tool = DalleImageTool::new(cfg).unwrap();
        let err = tool.run(json!({"prompt": "x"})).await.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("400"));
        assert!(msg.contains("Your prompt was rejected."));
    }

    #[tokio::test]
    async fn missing_prompt_errors_with_invalid() {
        let cfg = DalleConfig::new("k").with_base_url("http://127.0.0.1:1");
        let tool = DalleImageTool::new(cfg).unwrap();
        let err = tool.run(json!({"size": "1024x1024"})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn schema_has_correct_name_and_required_fields() {
        let cfg = DalleConfig::new("k");
        let tool = DalleImageTool::new(cfg).unwrap();
        let s = tool.schema();
        assert_eq!(s.name, "image_generate");
        let required = s.parameters["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "prompt");
    }
}
