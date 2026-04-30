//! `JwtDecodeTool` — decode a JWT into its header + payload
//! (without verifying the signature).
//!
//! # Why a dedicated tool
//!
//! A JWT is three base64url-no-pad segments joined by `.`. The
//! agent could chain `string.split` + `Base64Tool` + `json_extract`
//! to dissect one, but that's three tool calls per JWT and prone
//! to mistakes (which segment is which? what variant of base64?).
//! `JwtDecodeTool` does the standard decode in one call and
//! returns a structured object the agent can feed directly to
//! downstream tools.
//!
//! # What this is NOT
//!
//! **Signature verification is out of scope.** Verifying a JWT
//! requires the issuer's signing key, which an agent doesn't
//! generally have access to and shouldn't be juggling at the
//! tool level. This tool is for **inspection**, not authentication.
//! If your agent ever uses the result of this tool to authorize
//! anything, that's a bug — verification belongs in the auth
//! layer (typically a server-side middleware before the agent
//! ever sees the token).
//!
//! # Args
//!
//! - `token: String` — the JWT string (`header.payload.signature`).
//!
//! # Returns
//!
//! `{header: <obj>, payload: <obj>, signature_present: bool,
//!   expired: Option<bool>}`. `expired` is computed when the
//! payload contains an `exp` (RFC 7519 §4.1.4) claim — `null`
//! if the field is missing.

use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct JwtDecodeTool;

impl JwtDecodeTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for JwtDecodeTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "jwt_decode".into(),
            description: "Decode a JWT (`header.payload.signature`) into its parsed header \
                and payload objects. Does NOT verify the signature — verification requires \
                the issuer's signing key and belongs in the auth layer, not the agent. Use \
                this for inspection: extracting user_id / claims, debugging OAuth flows, \
                checking token expiry. Returns `{header, payload, signature_present, \
                expired}` where `expired` is null if no `exp` claim is present."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "token": {
                        "type": "string",
                        "description": "JWT string. Must have at least 2 dot-separated segments (header + payload). Signature segment is optional."
                    }
                },
                "required": ["token"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let token = args
            .get("token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("jwt_decode: missing `token`"))?;
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() < 2 {
            return Err(Error::invalid(
                "jwt_decode: token must have at least 2 segments (header.payload)",
            ));
        }
        let header = decode_segment(parts[0], "header")?;
        let payload = decode_segment(parts[1], "payload")?;
        let signature_present = parts.len() >= 3 && !parts[2].is_empty();

        let expired = payload
            .get("exp")
            .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)))
            .map(|exp| {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                now > exp
            });

        Ok(json!({
            "header": header,
            "payload": payload,
            "signature_present": signature_present,
            "expired": expired,
        }))
    }
}

fn decode_segment(b64: &str, label: &str) -> Result<Value> {
    let bytes = URL_SAFE_NO_PAD
        .decode(b64.as_bytes())
        .map_err(|e| Error::invalid(format!("jwt_decode: bad base64 in {label}: {e}")))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| Error::invalid(format!("jwt_decode: bad JSON in {label}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a JWT-shaped string from header + payload JSON.
    /// Signature defaults to "sig" (placeholder, not real).
    fn make_jwt(header: &Value, payload: &Value) -> String {
        let h = URL_SAFE_NO_PAD.encode(serde_json::to_vec(header).unwrap());
        let p = URL_SAFE_NO_PAD.encode(serde_json::to_vec(payload).unwrap());
        format!("{h}.{p}.sig")
    }

    #[tokio::test]
    async fn decodes_canonical_jwt() {
        let token = make_jwt(
            &json!({"alg": "HS256", "typ": "JWT"}),
            &json!({"sub": "user_42", "iat": 1_600_000_000}),
        );
        let t = JwtDecodeTool::new();
        let v = t.run(json!({"token": token})).await.unwrap();
        let header = v.get("header").unwrap();
        assert_eq!(
            header.get("alg").and_then(|x| x.as_str()),
            Some("HS256"),
        );
        let payload = v.get("payload").unwrap();
        assert_eq!(
            payload.get("sub").and_then(|x| x.as_str()),
            Some("user_42"),
        );
        assert_eq!(
            v.get("signature_present").and_then(|x| x.as_bool()),
            Some(true),
        );
        // No `exp` field → expired is null.
        assert!(v.get("expired").unwrap().is_null());
    }

    #[tokio::test]
    async fn detects_expired_token() {
        let token = make_jwt(
            &json!({"alg": "HS256"}),
            &json!({"exp": 1_000_000_000_i64}), // year 2001
        );
        let t = JwtDecodeTool::new();
        let v = t.run(json!({"token": token})).await.unwrap();
        assert_eq!(
            v.get("expired").and_then(|x| x.as_bool()),
            Some(true),
        );
    }

    #[tokio::test]
    async fn detects_unexpired_token() {
        let token = make_jwt(
            &json!({"alg": "HS256"}),
            &json!({"exp": 32_000_000_000_i64}), // year ~3000
        );
        let t = JwtDecodeTool::new();
        let v = t.run(json!({"token": token})).await.unwrap();
        assert_eq!(
            v.get("expired").and_then(|x| x.as_bool()),
            Some(false),
        );
    }

    #[tokio::test]
    async fn signature_present_false_when_only_two_segments() {
        let h = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"HS256\"}");
        let p = URL_SAFE_NO_PAD.encode(b"{\"sub\":\"x\"}");
        let token = format!("{h}.{p}");
        let t = JwtDecodeTool::new();
        let v = t.run(json!({"token": token})).await.unwrap();
        assert_eq!(
            v.get("signature_present").and_then(|x| x.as_bool()),
            Some(false),
        );
    }

    #[tokio::test]
    async fn signature_present_false_when_signature_empty() {
        let token = make_jwt(
            &json!({"alg": "HS256"}),
            &json!({"sub": "x"}),
        );
        // Strip signature: "h.p.sig" → "h.p."
        let stripped = format!(
            "{}.",
            token.rsplitn(2, '.').nth(1).unwrap(),
        );
        let t = JwtDecodeTool::new();
        let v = t.run(json!({"token": stripped})).await.unwrap();
        assert_eq!(
            v.get("signature_present").and_then(|x| x.as_bool()),
            Some(false),
        );
    }

    #[tokio::test]
    async fn one_segment_token_errors() {
        let t = JwtDecodeTool::new();
        let r = t.run(json!({"token": "onesegment"})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn bad_base64_in_header_errors() {
        let t = JwtDecodeTool::new();
        let r = t
            .run(json!({"token": "not-base64!@#.payload.sig"}))
            .await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn bad_json_after_decode_errors() {
        // Valid base64url, invalid JSON.
        let h = URL_SAFE_NO_PAD.encode(b"not json");
        let p = URL_SAFE_NO_PAD.encode(b"{\"sub\":\"x\"}");
        let t = JwtDecodeTool::new();
        let r = t.run(json!({"token": format!("{h}.{p}.sig")})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn missing_token_arg_errors() {
        let t = JwtDecodeTool::new();
        let r = t.run(json!({})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn standard_jwt_test_vector() {
        // RFC 7519 §A.1 example. Header `{"alg":"HS256","typ":"JWT"}`.
        // Payload `{"iss":"joe","exp":1300819380,"http://example.com/is_root":true}`.
        let token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\
                     .eyJpc3MiOiJqb2UiLCJleHAiOjEzMDA4MTkzODAsImh0dHA6Ly9leGFtcGxlLmNvbS9pc19yb290Ijp0cnVlfQ\
                     .dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let t = JwtDecodeTool::new();
        let v = t.run(json!({"token": token})).await.unwrap();
        let header = v.get("header").unwrap();
        assert_eq!(
            header.get("alg").and_then(|x| x.as_str()),
            Some("HS256"),
        );
        let payload = v.get("payload").unwrap();
        assert_eq!(
            payload.get("iss").and_then(|x| x.as_str()),
            Some("joe"),
        );
        // `exp` is in the past → expired=true.
        assert_eq!(
            v.get("expired").and_then(|x| x.as_bool()),
            Some(true),
        );
    }
}
