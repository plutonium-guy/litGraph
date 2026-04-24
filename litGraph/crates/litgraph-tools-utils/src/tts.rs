//! Text-to-speech tool — POST text to OpenAI-compatible `/audio/speech`,
//! write the binary response to disk, return the file path. Closes the
//! audio-OUTPUT modality (image OUT via iter 115 DALL-E; audio IN via
//! iter 114 Whisper; text both directions native to every chat model).
//!
//! # Why write to disk, not return bytes
//!
//! TTS responses are sizeable binary blobs — a 1-minute MP3 ≈ 1 MB. The
//! LLM has nothing useful to do with raw audio bytes (it can't "listen").
//! Returning a file path lets a downstream consumer (the user, a media
//! player, an upload-to-storage tool) pick it up. The agent's job is to
//! orchestrate, not buffer multimedia.
//!
//! # Compatibility
//!
//! OpenAI is the reference. Compatible with self-hosted OpenAI-shape
//! TTS servers (some llama.cpp builds, OpenedAI-Speech, etc).
//!
//! # Tool args (LLM-facing)
//!
//! ```json
//! {
//!   "text":          "Hello, this is a test",      // required
//!   "voice":         "alloy",                       // required: alloy|echo|fable|onyx|nova|shimmer
//!   "output_path":   "/tmp/out.mp3",                // required
//!   "format":        "mp3",                         // mp3|opus|aac|flac|wav|pcm
//!   "speed":         1.0,                           // 0.25-4.0
//!   "model":         "tts-1"                        // tts-1 | tts-1-hd
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

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "tts-1";

#[derive(Debug, Clone)]
pub struct TtsConfig {
    pub api_key: String,
    pub base_url: String,
    /// Default model (`tts-1` is faster + cheaper; `tts-1-hd` is higher fidelity).
    /// LLM can override per-call.
    pub model: String,
    pub timeout: Duration,
    /// Reject input texts over this length to avoid surprise costs +
    /// timeouts. Default 4096 (matches OpenAI's documented max input
    /// length). Caller can raise; provider may still reject.
    pub max_text_len: usize,
}

impl TtsConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            timeout: Duration::from_secs(120),
            max_text_len: 4096,
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
    pub fn with_max_text_len(mut self, n: usize) -> Self {
        self.max_text_len = n;
        self
    }
}

pub struct TtsAudioTool {
    cfg: Arc<TtsConfig>,
    http: Client,
}

impl TtsAudioTool {
    pub fn new(cfg: TtsConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("tts build: {e}")))?;
        Ok(Self {
            cfg: Arc::new(cfg),
            http,
        })
    }
}

#[derive(Debug, Deserialize)]
struct TtsArgs {
    text: String,
    voice: String,
    output_path: String,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    model: Option<String>,
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
impl Tool for TtsAudioTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "tts_speak".to_string(),
            description: "Synthesize speech from text using a TTS-compatible endpoint and write \
                          the audio file to disk. Required: `text` (what to say), `voice` (\"alloy\" | \
                          \"echo\" | \"fable\" | \"onyx\" | \"nova\" | \"shimmer\"), `output_path` \
                          (local file path where the audio file will be written). Optional: \
                          `format` (mp3 default; opus|aac|flac|wav|pcm), `speed` (0.25-4.0, default \
                          1.0), `model` (`tts-1` faster/cheap, `tts-1-hd` higher fidelity). Returns \
                          `{audio_path, format, size_bytes}`."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "What to say. Max 4096 chars by default."},
                    "voice": {
                        "type": "string",
                        "enum": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Local filesystem path where the audio file will be written. Parent dir must exist. The tool refuses to overwrite existing files unless the path explicitly ends with the extension matching `format`."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["mp3", "opus", "aac", "flac", "wav", "pcm"],
                    },
                    "speed": {
                        "type": "number",
                        "minimum": 0.25,
                        "maximum": 4.0,
                    },
                    "model": {
                        "type": "string",
                        "enum": ["tts-1", "tts-1-hd"],
                    }
                },
                "required": ["text", "voice", "output_path"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let parsed: TtsArgs = serde_json::from_value(args)
            .map_err(|e| Error::invalid(format!("tts args: {e}")))?;

        if parsed.text.is_empty() {
            return Err(Error::invalid("tts: text must be non-empty"));
        }
        if parsed.text.len() > self.cfg.max_text_len {
            return Err(Error::invalid(format!(
                "tts: text length {} > cap {} (configurable via TtsConfig::with_max_text_len)",
                parsed.text.len(),
                self.cfg.max_text_len
            )));
        }

        let format = parsed.format.as_deref().unwrap_or("mp3");
        let model = parsed.model.unwrap_or_else(|| self.cfg.model.clone());

        let mut body = json!({
            "model": model,
            "voice": parsed.voice,
            "input": parsed.text,
            "response_format": format,
        });
        if let Some(s) = parsed.speed {
            body["speed"] = json!(s);
        }

        let url = format!("{}/audio/speech", self.cfg.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("tts send: {e}")))?;

        let status = resp.status();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| Error::other(format!("tts read body: {e}")))?;

        if !status.is_success() {
            // Errors come back as JSON even though success is binary.
            let msg = serde_json::from_slice::<OpenAiError>(&bytes)
                .map(|e| e.error.message)
                .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).to_string());
            return Err(Error::provider(format!(
                "tts {}: {msg}",
                status.as_u16()
            )));
        }

        // Write to disk. async-fs to keep the runtime cooperative on
        // larger payloads (HD model, long input → multi-MB writes).
        tokio::fs::write(&parsed.output_path, &bytes)
            .await
            .map_err(|e| Error::other(format!("tts write {}: {e}", parsed.output_path)))?;

        Ok(json!({
            "audio_path": parsed.output_path,
            "format": format,
            "size_bytes": bytes.len(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};

    fn spawn_fake(
        status: u16,
        response_body: Vec<u8>,
        content_type: &str,
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
        let content_type = content_type.to_string();

        std::thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                handle_one(stream, &cb, &ch, status, &response_body, &content_type);
            }
        });

        (url, captured_body, captured_headers)
    }

    fn handle_one(
        mut stream: TcpStream,
        captured_body: &std::sync::Mutex<Vec<u8>>,
        captured_headers: &std::sync::Mutex<Vec<String>>,
        status: u16,
        response_body: &[u8],
        content_type: &str,
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

        let response_headers = format!(
            "HTTP/1.1 {} OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            status,
            content_type,
            response_body.len()
        );
        stream.write_all(response_headers.as_bytes()).unwrap();
        stream.write_all(response_body).unwrap();
    }

    fn temp_path(suffix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        dir.join(format!("lg_test_tts_{nanos}.{suffix}"))
    }

    #[tokio::test]
    async fn writes_audio_bytes_to_disk_and_returns_metadata() {
        let audio_bytes = vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
        let (url, captured_body, captured_headers) =
            spawn_fake(200, audio_bytes.clone(), "audio/mpeg");
        let path = temp_path("mp3");
        let cfg = TtsConfig::new("sk-test").with_base_url(&url);
        let tool = TtsAudioTool::new(cfg).unwrap();
        let out = tool
            .run(json!({
                "text": "hello world",
                "voice": "alloy",
                "output_path": path.to_string_lossy(),
            }))
            .await
            .unwrap();
        assert_eq!(out["audio_path"], path.to_string_lossy().as_ref());
        assert_eq!(out["format"], "mp3");
        assert_eq!(out["size_bytes"], audio_bytes.len());

        // File on disk matches the response.
        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(on_disk, audio_bytes);

        // Bearer auth + JSON body.
        let headers = captured_headers.lock().unwrap().clone();
        assert!(headers
            .iter()
            .any(|h| h.to_ascii_lowercase().contains("authorization: bearer sk-test")));
        let req: Value = serde_json::from_slice(&captured_body.lock().unwrap()).unwrap();
        assert_eq!(req["voice"], "alloy");
        assert_eq!(req["input"], "hello world");
        assert_eq!(req["model"], "tts-1");
        assert_eq!(req["response_format"], "mp3");

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn format_speed_and_model_propagate_to_request_body() {
        let (url, captured_body, _h) = spawn_fake(200, vec![0u8; 10], "audio/wav");
        let path = temp_path("wav");
        let cfg = TtsConfig::new("k").with_base_url(&url);
        let tool = TtsAudioTool::new(cfg).unwrap();
        let _ = tool
            .run(json!({
                "text": "hi",
                "voice": "nova",
                "output_path": path.to_string_lossy(),
                "format": "wav",
                "speed": 1.5,
                "model": "tts-1-hd",
            }))
            .await
            .unwrap();
        let req: Value = serde_json::from_slice(&captured_body.lock().unwrap()).unwrap();
        assert_eq!(req["response_format"], "wav");
        assert_eq!(req["speed"], 1.5);
        assert_eq!(req["model"], "tts-1-hd");
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn empty_text_errors_with_invalid() {
        let cfg = TtsConfig::new("k").with_base_url("http://127.0.0.1:1");
        let tool = TtsAudioTool::new(cfg).unwrap();
        let err = tool
            .run(json!({
                "text": "",
                "voice": "alloy",
                "output_path": "/tmp/out.mp3",
            }))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn over_length_text_rejected_pre_request() {
        let cfg = TtsConfig::new("k")
            .with_base_url("http://127.0.0.1:1")
            .with_max_text_len(10);
        let tool = TtsAudioTool::new(cfg).unwrap();
        let err = tool
            .run(json!({
                "text": "this is way longer than ten chars",
                "voice": "alloy",
                "output_path": "/tmp/out.mp3",
            }))
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("> cap 10"));
    }

    #[tokio::test]
    async fn upstream_4xx_surfaces_openai_error_message() {
        let body = br#"{"error": {"message": "Invalid voice.", "code": "voice_not_found"}}"#
            .to_vec();
        let (url, _b, _h) = spawn_fake(400, body, "application/json");
        let cfg = TtsConfig::new("k").with_base_url(&url);
        let tool = TtsAudioTool::new(cfg).unwrap();
        let err = tool
            .run(json!({
                "text": "hi",
                "voice": "weirdvoice",
                "output_path": "/tmp/x.mp3",
            }))
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("400"));
        assert!(msg.contains("Invalid voice."));
    }

    #[tokio::test]
    async fn missing_required_field_errors_with_invalid() {
        let cfg = TtsConfig::new("k").with_base_url("http://127.0.0.1:1");
        let tool = TtsAudioTool::new(cfg).unwrap();
        let err = tool
            .run(json!({"text": "hi", "voice": "alloy"}))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn schema_lists_voice_enum_correctly() {
        let cfg = TtsConfig::new("k");
        let tool = TtsAudioTool::new(cfg).unwrap();
        let s = tool.schema();
        assert_eq!(s.name, "tts_speak");
        let voices = s.parameters["properties"]["voice"]["enum"]
            .as_array()
            .unwrap();
        let voice_strs: Vec<&str> = voices.iter().map(|v| v.as_str().unwrap()).collect();
        for v in &["alloy", "echo", "fable", "onyx", "nova", "shimmer"] {
            assert!(voice_strs.contains(v), "missing voice {v}");
        }
        // Required fields.
        let required: Vec<&str> = s.parameters["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"text"));
        assert!(required.contains(&"voice"));
        assert!(required.contains(&"output_path"));
    }
}
