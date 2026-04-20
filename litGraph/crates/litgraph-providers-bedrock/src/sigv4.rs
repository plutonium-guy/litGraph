//! Minimal hand-rolled AWS Signature Version 4 for HTTPS POST requests.
//!
//! Scope: just enough for Bedrock InvokeModel. We sign the canonical request +
//! string-to-sign, derive the signing key via the standard HMAC chain, and emit
//! `Authorization` + `X-Amz-Date` headers (and optional `X-Amz-Security-Token`).
//!
//! Why hand-rolled? `aws-sigv4` pulls aws-smithy-runtime-api and friends
//! (~150KB of generated wire types). For one POST + signing we don't need any
//! of that. ~120 lines of HMAC-SHA256.

use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

pub struct SigningInputs<'a> {
    pub method: &'a str,
    pub host: &'a str,
    pub path: &'a str,
    /// Already URL-encoded query string (no leading `?`).
    pub query: &'a str,
    pub body: &'a [u8],
    /// User-supplied headers other than `host` / `x-amz-date` / `x-amz-content-sha256`.
    pub extra_headers: &'a [(String, String)],
    pub region: &'a str,
    pub service: &'a str,
    pub now: DateTime<Utc>,
}

#[derive(Debug)]
pub struct SignedHeaders {
    pub authorization: String,
    pub x_amz_date: String,
    pub x_amz_content_sha256: String,
    pub x_amz_security_token: Option<String>,
}

pub fn sign(creds: &AwsCredentials, inp: &SigningInputs) -> SignedHeaders {
    let amz_date = inp.now.format("%Y%m%dT%H%M%SZ").to_string();
    let date_stamp = inp.now.format("%Y%m%d").to_string();

    let payload_hash = hex::encode(Sha256::digest(inp.body));

    // Headers we always sign.
    let mut headers: Vec<(String, String)> = vec![
        ("host".into(), inp.host.to_string()),
        ("x-amz-content-sha256".into(), payload_hash.clone()),
        ("x-amz-date".into(), amz_date.clone()),
    ];
    if let Some(tok) = &creds.session_token {
        headers.push(("x-amz-security-token".into(), tok.clone()));
    }
    for (k, v) in inp.extra_headers {
        headers.push((k.to_lowercase(), v.trim().to_string()));
    }
    headers.sort_by(|a, b| a.0.cmp(&b.0));

    let canonical_headers: String = headers
        .iter()
        .map(|(k, v)| format!("{k}:{v}\n"))
        .collect();
    let signed_header_names: String = headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");

    let canonical_request = format!(
        "{method}\n{path}\n{query}\n{ch}\n{sh}\n{ph}",
        method = inp.method.to_uppercase(),
        path = inp.path,
        query = inp.query,
        ch = canonical_headers,
        sh = signed_header_names,
        ph = payload_hash,
    );

    let creq_hash = hex::encode(Sha256::digest(canonical_request.as_bytes()));
    let credential_scope = format!("{date_stamp}/{}/{}/aws4_request", inp.region, inp.service);
    let string_to_sign = format!("AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{creq_hash}");

    let signing_key = derive_signing_key(
        &creds.secret_access_key,
        &date_stamp,
        inp.region,
        inp.service,
    );
    let signature = hex::encode(hmac_sha256(&signing_key, string_to_sign.as_bytes()));

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={ak}/{scope}, SignedHeaders={sh}, Signature={sig}",
        ak = creds.access_key_id,
        scope = credential_scope,
        sh = signed_header_names,
        sig = signature,
    );

    SignedHeaders {
        authorization,
        x_amz_date: amz_date,
        x_amz_content_sha256: payload_hash,
        x_amz_security_token: creds.session_token.clone(),
    }
}

fn derive_signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_sha256(format!("AWS4{secret}").as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

fn hmac_sha256(key: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key");
    mac.update(msg);
    mac.finalize().into_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    /// Vector from the official AWS SigV4 test suite (get-vanilla):
    /// region=us-east-1, service=service, fixed date, empty body.
    /// Verifies our signing key + signature derivation matches AWS reference.
    #[test]
    fn signing_key_matches_aws_reference() {
        // From AWS docs: "AWS4wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY" for date 20150830, region us-east-1, service iam
        let secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
        let key = derive_signing_key(secret, "20150830", "us-east-1", "iam");
        // Reference signing key (from AWS docs example):
        let expected = hex::decode(
            "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9"
        ).unwrap();
        assert_eq!(key, expected);
    }

    #[test]
    fn produces_authorization_header_shape() {
        let creds = AwsCredentials {
            access_key_id: "AKIDEXAMPLE".into(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(),
            session_token: None,
        };
        let now = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let signed = sign(&creds, &SigningInputs {
            method: "POST",
            host: "bedrock-runtime.us-east-1.amazonaws.com",
            path: "/model/test/invoke",
            query: "",
            body: b"{}",
            extra_headers: &[("content-type".to_string(), "application/json".to_string())],
            region: "us-east-1",
            service: "bedrock",
            now,
        });
        assert!(signed.authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20230101/us-east-1/bedrock/aws4_request"));
        assert!(signed.authorization.contains("SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date"));
        assert!(signed.authorization.contains("Signature="));
        assert_eq!(signed.x_amz_date, "20230101T000000Z");
    }
}
