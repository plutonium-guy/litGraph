//! End-to-end Bedrock streaming test against a fake server.
//!
//! Uses `BedrockConfig::with_endpoint()` to redirect the request to a local
//! TCP listener. The fake server accepts the SigV4-signed POST and replies
//! with real AWS event-stream binary frames carrying base64'd Anthropic-shaped
//! SSE events. Verifies that `BedrockChat::stream()` parses + decodes + emits
//! `ChatStreamEvent::Delta` / `Done` correctly.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use base64::Engine;
use futures_util::StreamExt;
use litgraph_core::ChatModel;
use litgraph_core::ChatOptions;
use litgraph_core::Message;
use litgraph_core::model::ChatStreamEvent;
use litgraph_providers_bedrock::{AwsCredentials, BedrockChat, BedrockConfig};

/// Build one chunk frame wrapping a base64'd JSON event.
fn make_chunk_frame(inner_json: &str) -> Vec<u8> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(inner_json);
    let payload = format!(r#"{{"bytes":"{b64}"}}"#);
    let payload_bytes = payload.as_bytes();

    let header_name = b":event-type";
    let header_value = b"chunk";
    let mut headers = Vec::new();
    headers.push(header_name.len() as u8);
    headers.extend_from_slice(header_name);
    headers.push(7u8);
    headers.extend_from_slice(&(header_value.len() as u16).to_be_bytes());
    headers.extend_from_slice(header_value);

    let total_len = 12 + headers.len() + payload_bytes.len() + 4;
    let mut frame = Vec::new();
    frame.extend_from_slice(&(total_len as u32).to_be_bytes());
    frame.extend_from_slice(&(headers.len() as u32).to_be_bytes());
    frame.extend_from_slice(&[0u8; 4]); // dummy prelude CRC
    frame.extend_from_slice(&headers);
    frame.extend_from_slice(payload_bytes);
    frame.extend_from_slice(&[0u8; 4]); // dummy message CRC
    frame
}

/// Spawn a TCP server that ignores the incoming request and returns the given
/// response body with `Content-Type: application/vnd.amazon.eventstream`.
fn start_fake_bedrock(response_body: Vec<u8>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        if let Ok((mut s, _)) = listener.accept() {
            // Drain request bytes; not parsed, just consumed.
            let mut buf = [0u8; 8192];
            // Read until we see \r\n\r\n then drain content-length bytes if present.
            let mut total = Vec::new();
            loop {
                let n = match s.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(_) => break,
                };
                total.extend_from_slice(&buf[..n]);
                if let Some(pos) = find_double_crlf(&total) {
                    let content_length = parse_content_length(&total[..pos]).unwrap_or(0);
                    let need = pos + 4 + content_length;
                    while total.len() < need {
                        let n = match s.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => n,
                            Err(_) => break,
                        };
                        total.extend_from_slice(&buf[..n]);
                    }
                    break;
                }
            }

            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/vnd.amazon.eventstream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                response_body.len()
            );
            let _ = s.write_all(header.as_bytes());
            let _ = s.write_all(&response_body);
            let _ = s.flush();
        }
    });
    port
}

fn find_double_crlf(haystack: &[u8]) -> Option<usize> {
    haystack.windows(4).position(|w| w == b"\r\n\r\n")
}

fn parse_content_length(headers: &[u8]) -> Option<usize> {
    let s = std::str::from_utf8(headers).ok()?;
    for line in s.split("\r\n") {
        if let Some(rest) = line.strip_prefix("Content-Length: ") {
            return rest.parse().ok();
        }
        if let Some(rest) = line.strip_prefix("content-length: ") {
            return rest.parse().ok();
        }
    }
    None
}

#[tokio::test]
async fn bedrock_stream_yields_progressive_deltas_against_fake_server() {
    let frames = vec![
        make_chunk_frame(r#"{"type":"message_start","message":{"usage":{"input_tokens":7}}}"#),
        make_chunk_frame(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#),
        make_chunk_frame(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo "}}"#),
        make_chunk_frame(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}"#),
        make_chunk_frame(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}"#),
    ];
    let mut body: Vec<u8> = Vec::new();
    for f in &frames { body.extend_from_slice(f); }
    let port = start_fake_bedrock(body);

    let cfg = BedrockConfig::new(
        AwsCredentials {
            access_key_id: "AKID".into(),
            secret_access_key: "secret".into(),
            session_token: None,
        },
        "us-east-1",
        "anthropic.claude-opus-4-7-v1:0",
    )
    .with_endpoint(format!("http://127.0.0.1:{port}"));

    let chat = BedrockChat::new(cfg).unwrap();
    let mut stream = chat
        .stream(vec![Message::user("hi")], &ChatOptions::default())
        .await
        .unwrap();

    let mut deltas: Vec<String> = Vec::new();
    let mut got_done = false;
    while let Some(ev) = stream.next().await {
        match ev.unwrap() {
            ChatStreamEvent::Delta { text } => deltas.push(text),
            ChatStreamEvent::Done { response } => {
                got_done = true;
                assert_eq!(response.message.text_content(), "Hello world");
                assert_eq!(response.usage.prompt, 7);
                assert_eq!(response.usage.completion, 3);
                assert_eq!(response.usage.total, 10);
            }
            _ => {}
        }
    }
    assert_eq!(deltas, vec!["Hel".to_string(), "lo ".to_string(), "world".to_string()]);
    assert!(got_done, "expected a Done event after the deltas");
}
