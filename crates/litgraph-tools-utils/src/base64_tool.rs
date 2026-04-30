//! `Base64Tool` — encode / decode base64. Real prod use: agents
//! handling JWT tokens, base64-encoded image URLs, API keys,
//! certificate data, or any wire-format that uses base64.
//!
//! # Args
//!
//! - `text: String` — the input.
//! - `mode: "encode" | "decode"` — direction.
//! - `variant: "standard" | "url_safe"` — character alphabet.
//!   Default `standard` (RFC 4648 §4 — `+/=`). `url_safe` swaps
//!   to `-_` for use in URLs and JWT headers (RFC 4648 §5).
//!
//! # Returns
//!
//! - On `encode`: `{mode: "encode", variant, output: "<base64>"}`.
//! - On `decode`: `{mode: "decode", variant, output: "<utf8>"}`.
//!   Decoded bytes are interpreted as UTF-8; non-UTF-8 inputs
//!   return `Error::InvalidInput`. (For binary outputs, decode
//!   in code rather than via this tool.)

use async_trait::async_trait;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct Base64Tool;

impl Base64Tool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for Base64Tool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "base64".into(),
            description: "Encode or decode base64. Variants: 'standard' (RFC 4648 §4, +/= alphabet) \
                and 'url_safe' (RFC 4648 §5, -_ alphabet, no padding — used in JWTs and URL params). \
                Decoded output is UTF-8; non-UTF-8 binary decode is not supported. Common uses: \
                JWT header inspection, decoding base64-encoded image URLs, encoding payloads for \
                an API that requires base64 in the request body."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Input. For 'encode' it's plain text; for 'decode' it's the base64 string."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["encode", "decode"],
                        "description": "Direction. Required."
                    },
                    "variant": {
                        "type": "string",
                        "enum": ["standard", "url_safe"],
                        "description": "Alphabet variant. Default 'standard'."
                    }
                },
                "required": ["text", "mode"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let text = args
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("base64: missing `text`"))?;
        let mode = args
            .get("mode")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("base64: missing `mode`"))?;
        let variant = args
            .get("variant")
            .and_then(|v| v.as_str())
            .unwrap_or("standard");

        let output = match mode {
            "encode" => match variant {
                "standard" => STANDARD.encode(text.as_bytes()),
                "url_safe" => URL_SAFE_NO_PAD.encode(text.as_bytes()),
                other => {
                    return Err(Error::invalid(format!(
                        "base64: unknown variant '{other}'",
                    )))
                }
            },
            "decode" => {
                let bytes = match variant {
                    "standard" => STANDARD.decode(text.as_bytes()).map_err(|e| {
                        Error::invalid(format!("base64: bad input: {e}"))
                    })?,
                    "url_safe" => URL_SAFE_NO_PAD.decode(text.as_bytes()).map_err(|e| {
                        Error::invalid(format!("base64: bad input: {e}"))
                    })?,
                    other => {
                        return Err(Error::invalid(format!(
                            "base64: unknown variant '{other}'",
                        )));
                    }
                };
                String::from_utf8(bytes).map_err(|e| {
                    Error::invalid(format!("base64: decoded bytes are not UTF-8: {e}"))
                })?
            }
            other => {
                return Err(Error::invalid(format!(
                    "base64: unknown mode '{other}' (use 'encode' or 'decode')",
                )))
            }
        };

        Ok(json!({
            "mode": mode,
            "variant": variant,
            "output": output,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn standard_encode_known_vector() {
        let t = Base64Tool::new();
        let v = t
            .run(json!({"text": "hello", "mode": "encode"}))
            .await
            .unwrap();
        assert_eq!(
            v.get("output").and_then(|x| x.as_str()),
            Some("aGVsbG8="),
        );
        assert_eq!(v.get("variant").and_then(|x| x.as_str()), Some("standard"));
    }

    #[tokio::test]
    async fn standard_decode_known_vector() {
        let t = Base64Tool::new();
        let v = t
            .run(json!({"text": "aGVsbG8=", "mode": "decode"}))
            .await
            .unwrap();
        assert_eq!(v.get("output").and_then(|x| x.as_str()), Some("hello"));
    }

    #[tokio::test]
    async fn url_safe_encode_no_padding() {
        let t = Base64Tool::new();
        let v = t
            .run(json!({
                "text": "hello",
                "mode": "encode",
                "variant": "url_safe"
            }))
            .await
            .unwrap();
        // url_safe variant uses -_ alphabet and no = padding.
        let out = v.get("output").and_then(|x| x.as_str()).unwrap();
        assert!(!out.contains('='), "url_safe should not pad: {out}");
        assert!(!out.contains('+'));
        assert!(!out.contains('/'));
    }

    #[tokio::test]
    async fn url_safe_decode_round_trip() {
        let t = Base64Tool::new();
        // Pick a payload with bytes that produce + and / in standard.
        let original = "subjects?";
        let enc = t
            .run(json!({
                "text": original,
                "mode": "encode",
                "variant": "url_safe"
            }))
            .await
            .unwrap();
        let encoded = enc.get("output").and_then(|x| x.as_str()).unwrap().to_string();
        let dec = t
            .run(json!({
                "text": encoded,
                "mode": "decode",
                "variant": "url_safe"
            }))
            .await
            .unwrap();
        assert_eq!(
            dec.get("output").and_then(|x| x.as_str()),
            Some(original),
        );
    }

    #[tokio::test]
    async fn round_trip_is_identity() {
        let t = Base64Tool::new();
        for text in ["", "hello", "the quick brown fox", "🎉 emoji"] {
            let enc = t
                .run(json!({"text": text, "mode": "encode"}))
                .await
                .unwrap();
            let encoded = enc.get("output").and_then(|x| x.as_str()).unwrap().to_string();
            let dec = t
                .run(json!({"text": encoded, "mode": "decode"}))
                .await
                .unwrap();
            assert_eq!(
                dec.get("output").and_then(|x| x.as_str()),
                Some(text),
                "round-trip failed for '{text}'",
            );
        }
    }

    #[tokio::test]
    async fn bad_base64_returns_invalid_input() {
        let t = Base64Tool::new();
        let r = t
            .run(json!({"text": "not-valid-base64!!", "mode": "decode"}))
            .await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn unknown_mode_or_variant_errors() {
        let t = Base64Tool::new();
        assert!(matches!(
            t.run(json!({"text": "x", "mode": "weird"})).await,
            Err(Error::InvalidInput(_)),
        ));
        assert!(matches!(
            t.run(json!({"text": "x", "mode": "encode", "variant": "weird"})).await,
            Err(Error::InvalidInput(_)),
        ));
    }

    #[tokio::test]
    async fn decoded_non_utf8_errors() {
        // Encode arbitrary bytes that aren't valid UTF-8 (e.g. 0xff).
        let invalid_utf8 = STANDARD.encode([0xff, 0xfe, 0xfd]);
        let t = Base64Tool::new();
        let r = t
            .run(json!({"text": invalid_utf8, "mode": "decode"}))
            .await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn missing_required_args_error() {
        let t = Base64Tool::new();
        assert!(matches!(
            t.run(json!({"mode": "encode"})).await,
            Err(Error::InvalidInput(_)),
        ));
        assert!(matches!(
            t.run(json!({"text": "x"})).await,
            Err(Error::InvalidInput(_)),
        ));
    }

    #[tokio::test]
    async fn jwt_header_decode_workflow() {
        // JWT headers are url_safe base64 of `{"alg":"HS256","typ":"JWT"}`.
        // The standard test vector for this JWT header.
        let t = Base64Tool::new();
        let v = t
            .run(json!({
                "text": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
                "mode": "decode",
                "variant": "url_safe"
            }))
            .await
            .unwrap();
        assert_eq!(
            v.get("output").and_then(|x| x.as_str()),
            Some(r#"{"alg":"HS256","typ":"JWT"}"#),
        );
    }
}
