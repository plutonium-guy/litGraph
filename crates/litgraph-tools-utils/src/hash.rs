//! `HashTool` — compute content hash via blake3 / sha256 /
//! sha512 / md5. Real prod use: content fingerprinting for
//! dedup, integrity checks, agent-generated cache keys.
//!
//! # Args
//!
//! - `text: String` — the input to hash. (Inputs are treated as
//!   UTF-8 strings — for binary inputs, base64-encode upstream
//!   and the agent should pass the encoded form.)
//! - `algorithm: "blake3" | "sha256" | "sha512" | "md5"` —
//!   default `blake3`.
//!
//! # Returns
//!
//! `{algorithm: "<name>", hex: "<lowercase hex digest>"}`.
//!
//! # Algorithm guidance
//!
//! - **blake3**: default. Fast, cryptographically strong, what
//!   most prod workflows want. Use unless interop forces another.
//! - **sha256**: cryptographic, widely-supported, slow-er. Use
//!   when interop with non-Rust services demands it (most
//!   APIs, JWT, file checksums).
//! - **sha512**: stronger SHA, similar interop story to sha256.
//! - **md5**: NOT cryptographically secure. Use only for non-
//!   security fingerprinting (cache keys against legacy
//!   systems, ETag computation).

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{json, Value};
use sha2::{Digest as Sha2Digest, Sha256, Sha512};

#[derive(Debug, Clone, Default)]
pub struct HashTool;

impl HashTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for HashTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "hash".into(),
            description: "Compute a hex digest of `text` via the requested `algorithm`. \
                Algorithms: blake3 (default, fast cryptographic), sha256, sha512, md5 \
                (legacy, non-cryptographic). Useful for content fingerprinting, dedup \
                detection, integrity checks, and agent-generated cache keys."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Input to hash. Treated as UTF-8 bytes."
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": ["blake3", "sha256", "sha512", "md5"],
                        "description": "Hash algorithm. Default 'blake3'."
                    }
                },
                "required": ["text"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let text = args
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("hash: missing `text`"))?;
        let algorithm = args
            .get("algorithm")
            .and_then(|v| v.as_str())
            .unwrap_or("blake3");

        let bytes = text.as_bytes();
        let hex_digest = match algorithm {
            "blake3" => blake3::hash(bytes).to_hex().to_string(),
            "sha256" => {
                let mut h = Sha256::new();
                h.update(bytes);
                hex::encode(h.finalize())
            }
            "sha512" => {
                let mut h = Sha512::new();
                h.update(bytes);
                hex::encode(h.finalize())
            }
            "md5" => {
                let digest = md5::Md5::digest(bytes);
                hex::encode(digest)
            }
            other => {
                return Err(Error::invalid(format!(
                    "hash: unknown algorithm '{other}' (use 'blake3', 'sha256', 'sha512', or 'md5')",
                )));
            }
        };
        Ok(json!({
            "algorithm": algorithm,
            "hex": hex_digest,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn blake3_default_algorithm() {
        let t = HashTool::new();
        let v = t.run(json!({"text": "hello"})).await.unwrap();
        assert_eq!(
            v.get("algorithm").and_then(|x| x.as_str()),
            Some("blake3"),
        );
        let hex = v.get("hex").and_then(|x| x.as_str()).unwrap();
        // blake3("hello") known fixed vector.
        assert_eq!(
            hex,
            "ea8f163db38682925e4491c5e58d4bb3506ef8c14eb78a86e908c5624a67200f"
        );
    }

    #[tokio::test]
    async fn sha256_known_vector() {
        let t = HashTool::new();
        let v = t
            .run(json!({"text": "hello", "algorithm": "sha256"}))
            .await
            .unwrap();
        assert_eq!(
            v.get("hex").and_then(|x| x.as_str()),
            Some("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"),
        );
    }

    #[tokio::test]
    async fn sha512_known_vector() {
        let t = HashTool::new();
        let v = t
            .run(json!({"text": "hello", "algorithm": "sha512"}))
            .await
            .unwrap();
        let hex = v.get("hex").and_then(|x| x.as_str()).unwrap();
        // sha512("hello") known fixed vector.
        assert_eq!(
            hex,
            "9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca72323c3d99ba5c11d7c7acc6e14b8c5da0c4663475c2e5c3adef46f73bcdec043"
        );
    }

    #[tokio::test]
    async fn md5_known_vector() {
        let t = HashTool::new();
        let v = t
            .run(json!({"text": "hello", "algorithm": "md5"}))
            .await
            .unwrap();
        assert_eq!(
            v.get("hex").and_then(|x| x.as_str()),
            Some("5d41402abc4b2a76b9719d911017c592"),
        );
    }

    #[tokio::test]
    async fn empty_string_known_vectors() {
        let t = HashTool::new();
        let v = t
            .run(json!({"text": "", "algorithm": "sha256"}))
            .await
            .unwrap();
        assert_eq!(
            v.get("hex").and_then(|x| x.as_str()),
            Some("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        );
    }

    #[tokio::test]
    async fn deterministic_same_input_same_output() {
        let t = HashTool::new();
        let v1 = t.run(json!({"text": "test"})).await.unwrap();
        let v2 = t.run(json!({"text": "test"})).await.unwrap();
        assert_eq!(v1, v2);
    }

    #[tokio::test]
    async fn unknown_algorithm_errors() {
        let t = HashTool::new();
        let r = t
            .run(json!({"text": "x", "algorithm": "rot13"}))
            .await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn missing_text_errors() {
        let t = HashTool::new();
        let r = t.run(json!({})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn output_hex_is_lowercase() {
        let t = HashTool::new();
        for alg in ["blake3", "sha256", "sha512", "md5"] {
            let v = t
                .run(json!({"text": "ABCdef123", "algorithm": alg}))
                .await
                .unwrap();
            let hex = v.get("hex").and_then(|x| x.as_str()).unwrap();
            assert_eq!(
                hex,
                hex.to_lowercase(),
                "{alg} should output lowercase hex",
            );
        }
    }
}
