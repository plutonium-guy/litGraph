//! `UrlParseTool` — parse a URL into its components. Pairs with
//! the iter-279/280/281 utility tools to give agents a complete
//! "input parsing" toolkit.
//!
//! # Args
//!
//! - `url: String` — the URL to parse.
//!
//! # Returns
//!
//! A JSON object with:
//!
//! - `scheme` — `"https"`, `"http"`, `"ftp"`, etc.
//! - `host` — domain or IP, no port. Null for opaque URLs (e.g.
//!   `mailto:`).
//! - `port` — explicit port if present, else null. Default ports
//!   for the scheme (443 for https, 80 for http) are NOT
//!   inferred — null means "URL didn't specify."
//! - `path` — path component, always present (defaults to `/`).
//! - `query` — query string without the leading `?`, or null if
//!   none.
//! - `query_params` — query params as a `{key: value | [value,
//!   ...]}` object. Keys appearing once map to a single string;
//!   keys appearing multiple times map to a string array.
//! - `fragment` — fragment without the leading `#`, or null.
//! - `username` / `password` — auth components if present, else
//!   null. (Most URLs don't have these.)
//!
//! # Real prod use
//!
//! - **Validate redirect targets**: agent fetches a URL from
//!   user input → parse → confirm `host` is on an allowlist
//!   before redirecting.
//! - **Extract OAuth callback params**: agent receives
//!   `https://app/callback?code=ABC&state=xyz` → parse →
//!   `query_params.code` is what the agent needs.
//! - **Build URL variants**: parse, then template a new URL with
//!   modified path/query.

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use url::Url;

#[derive(Debug, Clone, Default)]
pub struct UrlParseTool;

impl UrlParseTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for UrlParseTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "url_parse".into(),
            description: "Parse a URL into its components (scheme, host, port, path, \
                query, fragment) and return query parameters as a key/value object. \
                Useful for validating redirect hosts, extracting OAuth callback \
                params, or building URL variants. Errors on invalid URL syntax."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to parse. Example: 'https://example.com/path?a=1&b=2#section'."
                    }
                },
                "required": ["url"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let raw = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("url_parse: missing `url`"))?;
        let parsed = Url::parse(raw)
            .map_err(|e| Error::invalid(format!("url_parse: bad URL: {e}")))?;

        // Group query params: single value → string; repeated → array.
        let mut grouped: HashMap<String, Vec<String>> = HashMap::new();
        for (k, v) in parsed.query_pairs() {
            grouped.entry(k.into_owned()).or_default().push(v.into_owned());
        }
        let mut params_obj = Map::new();
        // Stable key ordering for tests.
        let mut keys: Vec<&String> = grouped.keys().collect();
        keys.sort();
        for k in keys {
            let values = &grouped[k];
            let v = if values.len() == 1 {
                Value::String(values[0].clone())
            } else {
                Value::Array(values.iter().map(|s| Value::String(s.clone())).collect())
            };
            params_obj.insert(k.clone(), v);
        }

        let host = parsed.host_str().map(|s| Value::String(s.to_string()))
            .unwrap_or(Value::Null);
        let port = parsed.port().map(|p| Value::from(p)).unwrap_or(Value::Null);
        let query = parsed.query().map(|s| Value::String(s.to_string()))
            .unwrap_or(Value::Null);
        let fragment = parsed.fragment().map(|s| Value::String(s.to_string()))
            .unwrap_or(Value::Null);
        let username = if parsed.username().is_empty() {
            Value::Null
        } else {
            Value::String(parsed.username().to_string())
        };
        let password = parsed
            .password()
            .map(|s| Value::String(s.to_string()))
            .unwrap_or(Value::Null);

        Ok(json!({
            "scheme": parsed.scheme(),
            "host": host,
            "port": port,
            "path": parsed.path(),
            "query": query,
            "query_params": Value::Object(params_obj),
            "fragment": fragment,
            "username": username,
            "password": password,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn parses_basic_https_url() {
        let t = UrlParseTool::new();
        let v = t
            .run(json!({"url": "https://example.com/path?a=1&b=2#section"}))
            .await
            .unwrap();
        assert_eq!(v.get("scheme").and_then(|x| x.as_str()), Some("https"));
        assert_eq!(v.get("host").and_then(|x| x.as_str()), Some("example.com"));
        assert!(v.get("port").unwrap().is_null());
        assert_eq!(v.get("path").and_then(|x| x.as_str()), Some("/path"));
        assert_eq!(v.get("fragment").and_then(|x| x.as_str()), Some("section"));
        let params = v.get("query_params").unwrap();
        assert_eq!(params.get("a").and_then(|x| x.as_str()), Some("1"));
        assert_eq!(params.get("b").and_then(|x| x.as_str()), Some("2"));
    }

    #[tokio::test]
    async fn parses_explicit_port() {
        let t = UrlParseTool::new();
        let v = t
            .run(json!({"url": "https://example.com:8443/api"}))
            .await
            .unwrap();
        assert_eq!(v.get("port").and_then(|x| x.as_u64()), Some(8443));
    }

    #[tokio::test]
    async fn default_port_returns_null() {
        // 443 is the default for https → not surfaced.
        let t = UrlParseTool::new();
        let v = t
            .run(json!({"url": "https://example.com/api"}))
            .await
            .unwrap();
        assert!(v.get("port").unwrap().is_null());
    }

    #[tokio::test]
    async fn repeated_query_params_become_array() {
        let t = UrlParseTool::new();
        let v = t
            .run(json!({"url": "https://e.com/?tag=a&tag=b&tag=c"}))
            .await
            .unwrap();
        let params = v.get("query_params").unwrap();
        assert_eq!(
            params.get("tag").unwrap(),
            &json!(["a", "b", "c"]),
        );
    }

    #[tokio::test]
    async fn no_query_returns_null() {
        let t = UrlParseTool::new();
        let v = t.run(json!({"url": "https://e.com/path"})).await.unwrap();
        assert!(v.get("query").unwrap().is_null());
        assert!(v.get("query_params").unwrap().as_object().unwrap().is_empty());
    }

    #[tokio::test]
    async fn auth_components_extracted() {
        let t = UrlParseTool::new();
        let v = t
            .run(json!({"url": "https://alice:secret@e.com/dashboard"}))
            .await
            .unwrap();
        assert_eq!(
            v.get("username").and_then(|x| x.as_str()),
            Some("alice"),
        );
        assert_eq!(
            v.get("password").and_then(|x| x.as_str()),
            Some("secret"),
        );
    }

    #[tokio::test]
    async fn no_auth_components_return_null() {
        let t = UrlParseTool::new();
        let v = t.run(json!({"url": "https://e.com/"})).await.unwrap();
        assert!(v.get("username").unwrap().is_null());
        assert!(v.get("password").unwrap().is_null());
    }

    #[tokio::test]
    async fn bad_url_returns_invalid_input() {
        let t = UrlParseTool::new();
        let r = t.run(json!({"url": "not a url"})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn missing_url_arg_errors() {
        let t = UrlParseTool::new();
        let r = t.run(json!({})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn percent_encoded_query_decoded() {
        let t = UrlParseTool::new();
        let v = t
            .run(json!({"url": "https://e.com/?q=hello%20world"}))
            .await
            .unwrap();
        let params = v.get("query_params").unwrap();
        assert_eq!(
            params.get("q").and_then(|x| x.as_str()),
            Some("hello world"),
        );
    }

    #[tokio::test]
    async fn fragment_without_leading_hash() {
        let t = UrlParseTool::new();
        let v = t
            .run(json!({"url": "https://e.com/page#anchor-1"}))
            .await
            .unwrap();
        assert_eq!(
            v.get("fragment").and_then(|x| x.as_str()),
            Some("anchor-1"),
        );
    }

    #[tokio::test]
    async fn oauth_callback_workflow() {
        // Realistic prod scenario: agent gets an OAuth callback URL,
        // pulls the auth code.
        let t = UrlParseTool::new();
        let v = t
            .run(json!({
                "url": "https://app.example.com/callback?code=ABC123&state=xyz789&scope=read+write"
            }))
            .await
            .unwrap();
        let params = v.get("query_params").unwrap();
        assert_eq!(
            params.get("code").and_then(|x| x.as_str()),
            Some("ABC123"),
        );
        assert_eq!(
            params.get("state").and_then(|x| x.as_str()),
            Some("xyz789"),
        );
    }
}
