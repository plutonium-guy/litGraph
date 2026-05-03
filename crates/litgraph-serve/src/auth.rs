//! Auth middleware for `litgraph-serve`. Optional: callers wrap the
//! `Router` with these layers in production; out-of-the-box deploys
//! stay open (matches the existing /invoke + /stream behaviour).
//!
//! Two primitives:
//!
//! - [`bearer_layer`] / [`bearer_layer_multi`] — single static
//!   token (or rotation-set). Rejects `Authorization` header
//!   mismatches with 401 + `WWW-Authenticate: Bearer`.
//! - [`forwarded_user_layer`] — extracts `X-Forwarded-User` and
//!   stashes it as a request extension so handlers can read
//!   [`ForwardedUser`] for per-tenant logic without re-parsing.
//!
//! Drop-in for any `Router`:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use litgraph_core::ChatModel;
//! # async fn run(model: Arc<dyn ChatModel>) {
//! use litgraph_serve::{router_for, auth};
//! let app = router_for(model)
//!     .layer(auth::bearer_layer("super-secret"))
//!     .layer(auth::forwarded_user_layer());
//! # }
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::middleware::{from_fn, from_fn_with_state, Next};
use axum::response::{IntoResponse, Response};

/// Identity extracted from `X-Forwarded-User`. Inserted as a request
/// extension by [`forwarded_user_layer`]. Handlers read via
/// `Extension(user): Extension<ForwardedUser>` (axum extractor).
#[derive(Debug, Clone)]
pub struct ForwardedUser(pub String);

/// Bearer-token check against a single static token.
pub fn bearer_layer(token: impl Into<String>) -> _BearerLayer {
    let mut set = HashSet::new();
    set.insert(token.into());
    bearer_layer_multi(set)
}

/// Bearer auth against any of N tokens. Use for rotation: keep the
/// old + new token in the set during the cutover window.
pub fn bearer_layer_multi(tokens: HashSet<String>) -> _BearerLayer {
    from_fn_with_state(Arc::new(tokens), bearer_check)
}

/// Concrete return type for the bearer layer constructors. Re-exported
/// so callers that want to store the layer in a struct can name it.
pub type _BearerLayer = axum::middleware::FromFnLayer<
    fn(
        axum::extract::State<Arc<HashSet<String>>>,
        Request<Body>,
        Next,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>>,
    Arc<HashSet<String>>,
    (axum::extract::State<Arc<HashSet<String>>>,),
>;

fn bearer_check(
    axum::extract::State(tokens): axum::extract::State<Arc<HashSet<String>>>,
    req: Request<Body>,
    next: Next,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>> {
    Box::pin(async move {
        if let Some(supplied) = extract_bearer(req.headers()) {
            if tokens.contains(&supplied) {
                return next.run(req).await;
            }
        }
        let mut resp = (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "missing or invalid bearer token"})),
        )
            .into_response();
        resp.headers_mut().insert(
            axum::http::header::WWW_AUTHENTICATE,
            "Bearer realm=\"litgraph\"".parse().unwrap(),
        );
        resp
    })
}

fn extract_bearer(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(axum::http::header::AUTHORIZATION)?;
    let s = raw.to_str().ok()?;
    let (scheme, value) = s.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    Some(value.trim().to_string())
}

/// Middleware that lifts the `X-Forwarded-User` request header into a
/// [`ForwardedUser`] extension. If the header is missing the
/// extension isn't set — handlers can `Option<Extension<ForwardedUser>>`
/// to handle anonymous requests.
pub fn forwarded_user_layer() -> _ForwardedUserLayer {
    from_fn(forwarded_user_inner)
}

/// Concrete return type for [`forwarded_user_layer`].
pub type _ForwardedUserLayer = axum::middleware::FromFnLayer<
    fn(
        Request<Body>,
        Next,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>>,
    (),
    (),
>;

fn forwarded_user_inner(
    mut req: Request<Body>,
    next: Next,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>> {
    // Pull the header before next.run consumes the request.
    let user_value = req
        .headers()
        .get("x-forwarded-user")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());
    if let Some(s) = user_value {
        req.extensions_mut().insert(ForwardedUser(s));
    }
    Box::pin(async move { next.run(req).await })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_bearer_reads_standard_header() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            "Bearer my-token".parse().unwrap(),
        );
        assert_eq!(extract_bearer(&h), Some("my-token".into()));
    }

    #[test]
    fn extract_bearer_case_insensitive_scheme() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            "bearer xyz".parse().unwrap(),
        );
        assert_eq!(extract_bearer(&h), Some("xyz".into()));
    }

    #[test]
    fn extract_bearer_rejects_basic() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            "Basic dXNlcjpwYXNz".parse().unwrap(),
        );
        assert_eq!(extract_bearer(&h), None);
    }

    #[test]
    fn extract_bearer_handles_missing() {
        let h = HeaderMap::new();
        assert_eq!(extract_bearer(&h), None);
    }

    #[test]
    fn extract_bearer_strips_whitespace() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            "Bearer  spaced  ".parse().unwrap(),
        );
        assert_eq!(extract_bearer(&h), Some("spaced".into()));
    }
}
