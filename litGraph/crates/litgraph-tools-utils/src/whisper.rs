//! Whisper transcription tool — POST audio file to OpenAI-compatible
//! `/audio/transcriptions` endpoint. Works with OpenAI directly + every
//! provider that mimics that surface (Groq, Together, OpenRouter,
//! self-hosted whisper.cpp servers, vLLM with whisper).
//!
//! # Why a Tool, not a ChatModel method
//!
//! Whisper is a one-shot speech-to-text — the LLM agent's job is to
//! decide when to transcribe (when given an audio path), call this tool,
//! and reason about the resulting text. That's the Tool pattern. The
//! transcript is text, which the agent already knows how to handle.
//!
//! # Multipart upload, hand-rolled
//!
//! `reqwest` has multipart support but pulling in the `multipart` feature
//! would expand the binary footprint for every workspace user. Whisper
//! requests are small (single file + a few text fields) so we hand-roll
//! the form: explicit boundary, manual field assembly, single body buffer.
//!
//! # Tool args (LLM-facing)
//!
//! ```json
//! {
//!   "audio_path": "/path/to/audio.m4a",   // required
//!   "language":   "en",                    // optional ISO-639-1
//!   "prompt":     "transcript context",    // optional
//!   "response_format": "json"              // "json" | "text" | "verbose_json"
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
const DEFAULT_MODEL: &str = "whisper-1";

/// Configuration for the Whisper tool. Provider-agnostic — any
/// OpenAI-compatible `/audio/transcriptions` endpoint works.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    /// Optional max audio file size in bytes. Default 25 MiB
    /// (matches OpenAI's documented limit). Files larger than this
    /// fail-fast before the upload — saves bandwidth + a confusing
    /// upstream 413.
    pub max_file_size: u64,
}

impl WhisperConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            timeout: Duration::from_secs(120),
            max_file_size: 25 * 1024 * 1024,
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

    pub fn with_max_file_size(mut self, bytes: u64) -> Self {
        self.max_file_size = bytes;
        self
    }
}

pub struct WhisperTranscribeTool {
    cfg: Arc<WhisperConfig>,
    http: Client,
}

impl WhisperTranscribeTool {
    pub fn new(cfg: WhisperConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("whisper build: {e}")))?;
        Ok(Self {
            cfg: Arc::new(cfg),
            http,
        })
    }
}

#[derive(Debug, Deserialize)]
struct WhisperArgs {
    audio_path: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
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
impl Tool for WhisperTranscribeTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "whisper_transcribe".to_string(),
            description: "Transcribe an audio file to text using a Whisper-compatible endpoint. \
                          Pass `audio_path` (local file path). Optional: `language` (ISO-639-1 \
                          like \"en\"), `prompt` (transcript-style hint to bias the recognizer), \
                          `response_format` (\"json\" | \"text\" | \"verbose_json\"). Returns \
                          the transcribed text."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Local filesystem path to the audio file (mp3, m4a, wav, mpeg, mpga, ogg, flac, webm)."
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional ISO-639-1 language code. Improves accuracy when the recording language is known."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional transcript-style hint. Useful for biasing the recognizer toward domain vocabulary or recovering from a prior chunk's tail."
                    },
                    "response_format": {
                        "type": "string",
                        "enum": ["json", "text", "verbose_json"],
                        "description": "Defaults to json. verbose_json includes timestamps."
                    }
                },
                "required": ["audio_path"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let parsed: WhisperArgs = serde_json::from_value(args)
            .map_err(|e| Error::invalid(format!("whisper args: {e}")))?;

        // Read the file. async to keep the runtime cooperative on huge
        // uploads. spawn_blocking would be an alternative but tokio::fs
        // is fine for the typical Whisper file size (<25 MiB).
        let bytes = tokio::fs::read(&parsed.audio_path)
            .await
            .map_err(|e| Error::invalid(format!("whisper read {}: {e}", parsed.audio_path)))?;
        if (bytes.len() as u64) > self.cfg.max_file_size {
            return Err(Error::invalid(format!(
                "whisper: file {} is {} bytes (cap {}); refusing to upload",
                parsed.audio_path,
                bytes.len(),
                self.cfg.max_file_size
            )));
        }
        let filename = std::path::Path::new(&parsed.audio_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("audio")
            .to_string();
        let content_type = guess_audio_mime(&filename);

        // Build the multipart body.
        let boundary = format!("----litgraphwhisper{}", uuid_like());
        let body = build_multipart(
            &boundary,
            &filename,
            &content_type,
            &bytes,
            &self.cfg.model,
            parsed.language.as_deref(),
            parsed.prompt.as_deref(),
            parsed.response_format.as_deref(),
        );

        let url = format!("{}/audio/transcriptions", self.cfg.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(body)
            .send()
            .await
            .map_err(|e| Error::other(format!("whisper send: {e}")))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::other(format!("whisper read body: {e}")))?;

        if !status.is_success() {
            // Try to extract the OpenAI error message; fall back to raw body.
            let msg = serde_json::from_str::<OpenAiError>(&text)
                .map(|e| e.error.message)
                .unwrap_or_else(|_| text.clone());
            return Err(Error::provider(format!(
                "whisper {}: {msg}",
                status.as_u16()
            )));
        }

        // response_format determines body shape:
        //   - "text" → raw transcript text (no JSON)
        //   - "json" → {"text": "..."}
        //   - "verbose_json" → richer JSON; pass through as-is
        let response_format = parsed.response_format.as_deref().unwrap_or("json");
        match response_format {
            "text" => Ok(Value::String(text)),
            _ => {
                // Parse JSON; if it's the simple {"text": "..."} shape, return
                // the inner string for ergonomics. Else return the full Value.
                let v: Value = serde_json::from_str(&text)
                    .map_err(|e| Error::other(format!("whisper parse json: {e}")))?;
                if response_format == "json" {
                    if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
                        return Ok(Value::String(t.to_string()));
                    }
                }
                Ok(v)
            }
        }
    }
}

fn guess_audio_mime(filename: &str) -> &'static str {
    let ext = filename.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "mp3" | "mpga" | "mpeg" => "audio/mpeg",
        "m4a" | "mp4" => "audio/mp4",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "webm" => "audio/webm",
        _ => "application/octet-stream",
    }
}

/// Tiny pseudo-UUID — just for boundary uniqueness, not security.
/// Avoids pulling in the `uuid` crate.
fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:032x}")
}

#[allow(clippy::too_many_arguments)]
fn build_multipart(
    boundary: &str,
    filename: &str,
    content_type: &str,
    file_bytes: &[u8],
    model: &str,
    language: Option<&str>,
    prompt: Option<&str>,
    response_format: Option<&str>,
) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(file_bytes.len() + 1024);
    let push_field = |out: &mut Vec<u8>, name: &str, value: &str| {
        out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        out.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(b"\r\n");
    };

    // file field first
    out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    out.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\
             Content-Type: {content_type}\r\n\r\n"
        )
        .as_bytes(),
    );
    out.extend_from_slice(file_bytes);
    out.extend_from_slice(b"\r\n");

    push_field(&mut out, "model", model);
    if let Some(l) = language {
        push_field(&mut out, "language", l);
    }
    if let Some(p) = prompt {
        push_field(&mut out, "prompt", p);
    }
    if let Some(f) = response_format {
        push_field(&mut out, "response_format", f);
    }

    out.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};

    /// Minimal hand-rolled HTTP server: serves one canned response then
    /// remembers the request body for assertions. Intentionally tiny —
    /// httpmock would be overkill.
    struct FakeServer {
        captured_body: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
        captured_headers: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    fn spawn_fake(
        status: u16,
        response_body: String,
        content_type: &str,
    ) -> (String, FakeServer) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}/v1");
        let captured_body = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_headers = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let cb = captured_body.clone();
        let ch = captured_headers.clone();
        let response_body_owned = response_body;
        let content_type = content_type.to_string();

        std::thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                handle_one(
                    stream,
                    &cb,
                    &ch,
                    status,
                    &response_body_owned,
                    &content_type,
                );
            }
        });

        (
            url,
            FakeServer {
                captured_body,
                captured_headers,
            },
        )
    }

    fn handle_one(
        mut stream: TcpStream,
        captured_body: &std::sync::Mutex<Vec<u8>>,
        captured_headers: &std::sync::Mutex<Vec<String>>,
        status: u16,
        response_body: &str,
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

        let resp_body_bytes = response_body.as_bytes();
        let response = format!(
            "HTTP/1.1 {} OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            status,
            content_type,
            resp_body_bytes.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
        stream.write_all(resp_body_bytes).unwrap();
    }

    fn write_temp_audio(content: &[u8], ext: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "lg_test_audio_{}.{}",
            uuid_like(),
            ext
        ));
        std::fs::write(&path, content).unwrap();
        path
    }

    #[tokio::test]
    async fn json_response_extracts_text_field() {
        let (url, srv) = spawn_fake(200, r#"{"text": "hello world"}"#.to_string(), "application/json");
        let path = write_temp_audio(b"fake audio bytes", "mp3");
        let cfg = WhisperConfig::new("sk-test").with_base_url(&url);
        let tool = WhisperTranscribeTool::new(cfg).unwrap();
        let out = tool
            .run(json!({"audio_path": path.to_string_lossy()}))
            .await
            .unwrap();
        assert_eq!(out, json!("hello world"));

        // Verify the request hit /audio/transcriptions with bearer auth.
        let headers = srv.captured_headers.lock().unwrap().clone();
        let req_line = &headers[0];
        assert!(req_line.contains("/audio/transcriptions"));
        assert!(headers.iter().any(|h| h.to_ascii_lowercase().contains("authorization: bearer sk-test")));
        // And the multipart body contains the model name + filename.
        let body = String::from_utf8_lossy(&srv.captured_body.lock().unwrap()).to_string();
        assert!(body.contains("name=\"model\""));
        assert!(body.contains("whisper-1"));
        assert!(body.contains("filename="));

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn text_response_format_returns_raw_string() {
        let (url, _srv) = spawn_fake(200, "raw transcript text".to_string(), "text/plain");
        let path = write_temp_audio(b"x", "mp3");
        let cfg = WhisperConfig::new("k").with_base_url(&url);
        let tool = WhisperTranscribeTool::new(cfg).unwrap();
        let out = tool
            .run(json!({
                "audio_path": path.to_string_lossy(),
                "response_format": "text",
            }))
            .await
            .unwrap();
        assert_eq!(out, json!("raw transcript text"));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn verbose_json_returns_full_value() {
        let body = r#"{"text": "hi", "segments": [{"id": 0, "text": "hi"}]}"#;
        let (url, _srv) = spawn_fake(200, body.to_string(), "application/json");
        let path = write_temp_audio(b"x", "mp3");
        let cfg = WhisperConfig::new("k").with_base_url(&url);
        let tool = WhisperTranscribeTool::new(cfg).unwrap();
        let out = tool
            .run(json!({
                "audio_path": path.to_string_lossy(),
                "response_format": "verbose_json",
            }))
            .await
            .unwrap();
        assert_eq!(out["text"], "hi");
        assert!(out["segments"].is_array());
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn missing_file_errors_with_invalid() {
        let cfg = WhisperConfig::new("k").with_base_url("http://127.0.0.1:1");
        let tool = WhisperTranscribeTool::new(cfg).unwrap();
        let err = tool
            .run(json!({"audio_path": "/nonexistent-litgraph-file.mp3"}))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn over_size_cap_fails_before_upload() {
        let path = write_temp_audio(&vec![0u8; 100], "mp3");
        let cfg = WhisperConfig::new("k")
            .with_base_url("http://127.0.0.1:1")
            .with_max_file_size(50);
        let tool = WhisperTranscribeTool::new(cfg).unwrap();
        let err = tool
            .run(json!({"audio_path": path.to_string_lossy()}))
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("100 bytes"));
        assert!(msg.contains("cap 50"));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn upstream_4xx_surfaces_error_message_from_openai_body() {
        let err_body = r#"{"error": {"message": "Invalid model.", "code": "model_not_found"}}"#;
        let (url, _srv) = spawn_fake(404, err_body.to_string(), "application/json");
        let path = write_temp_audio(b"x", "mp3");
        let cfg = WhisperConfig::new("k").with_base_url(&url);
        let tool = WhisperTranscribeTool::new(cfg).unwrap();
        let err = tool
            .run(json!({"audio_path": path.to_string_lossy()}))
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("404"));
        assert!(msg.contains("Invalid model."));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn language_and_prompt_propagated_in_multipart_body() {
        let (url, srv) = spawn_fake(200, r#"{"text": "ok"}"#.to_string(), "application/json");
        let path = write_temp_audio(b"x", "mp3");
        let cfg = WhisperConfig::new("k").with_base_url(&url);
        let tool = WhisperTranscribeTool::new(cfg).unwrap();
        let _ = tool
            .run(json!({
                "audio_path": path.to_string_lossy(),
                "language": "en",
                "prompt": "transcript context here",
            }))
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&srv.captured_body.lock().unwrap()).to_string();
        assert!(body.contains("name=\"language\""));
        assert!(body.contains("\r\nen\r\n"));
        assert!(body.contains("name=\"prompt\""));
        assert!(body.contains("transcript context here"));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn schema_has_correct_name_and_required_fields() {
        let cfg = WhisperConfig::new("k");
        let tool = WhisperTranscribeTool::new(cfg).unwrap();
        let s = tool.schema();
        assert_eq!(s.name, "whisper_transcribe");
        let required = s.parameters["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "audio_path");
    }

    #[test]
    fn audio_mime_dispatch_by_extension() {
        assert_eq!(guess_audio_mime("clip.mp3"), "audio/mpeg");
        assert_eq!(guess_audio_mime("clip.wav"), "audio/wav");
        assert_eq!(guess_audio_mime("clip.m4a"), "audio/mp4");
        assert_eq!(guess_audio_mime("clip.unknown"), "application/octet-stream");
    }
}
