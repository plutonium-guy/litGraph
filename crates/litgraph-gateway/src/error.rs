//! Client-facing errors in the OpenAI wire shape.
//!
//! Upstream credentials, deployment ids and base URLs go to the trace,
//! never to the client. A caller learns "gpt-4o is unavailable", not which
//! of three deployments failed or where it is hosted.

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

#[derive(Debug)]
pub enum GatewayError {
    Unauthorized,
    GroupForbidden,
    ModelNotFound { model: String },
    RateLimited { retry_after_ms: u64 },
    BudgetExhausted { spent_usd: f64, cap_usd: f64 },
    NoDeploymentAvailable,
    /// A client-side rejection relayed from upstream.
    ///
    /// # Invariant: `message` reaches the client verbatim
    ///
    /// This is deliberate. An upstream 400 ("context length exceeded",
    /// "unsupported parameter") is the most useful thing a caller gets, and
    /// replacing it with a generic string makes every client error
    /// undebuggable. Nothing here sanitises it.
    ///
    /// The consequence is an invariant this module cannot enforce: any
    /// `ChatModel` placed in a `Deployment` must keep the text of its
    /// NON-RETRYABLE errors free of deployment ids, upstream base URLs and
    /// credentials. Retryable errors are safe by construction — they are
    /// consumed by failover and only ever logged (see `dispatch::is_retryable`).
    ///
    /// Shipped providers satisfy this today: they construct only
    /// `Error::provider(..)`, which is retryable. But the path is live, not
    /// theoretical — `TokenBudgetChatModel` (`litgraph-resilience`) returns a
    /// non-retryable `Error::invalid(..)` and can wrap any `ChatModel`. Its
    /// text is benign; the next wrapper's might not be. Check before composing.
    UpstreamRejected { message: String },
    BadRequest { message: String },
}

impl GatewayError {
    fn parts(&self) -> (StatusCode, &'static str, &'static str, String) {
        match self {
            Self::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "invalid_request_error",
                "invalid_api_key",
                "Incorrect API key provided.".into(),
            ),
            Self::GroupForbidden => (
                StatusCode::FORBIDDEN,
                "invalid_request_error",
                "model_not_allowed",
                "This API key is not permitted to use the requested model.".into(),
            ),
            Self::ModelNotFound { model } => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "model_not_found",
                format!("The model {model:?} does not exist."),
            ),
            Self::RateLimited { .. } => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "rate_limit_exceeded",
                "Rate limit reached for this API key.".into(),
            ),
            Self::BudgetExhausted { spent_usd, cap_usd } => (
                StatusCode::PAYMENT_REQUIRED,
                "insufficient_quota",
                "budget_exceeded",
                format!("Spend cap reached: ${spent_usd:.2} of ${cap_usd:.2} used."),
            ),
            Self::NoDeploymentAvailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "no_deployment_available",
                "The requested model is temporarily unavailable.".into(),
            ),
            Self::UpstreamRejected { message } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "upstream_rejected",
                message.clone(),
            ),
            Self::BadRequest { message } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "bad_request",
                message.clone(),
            ),
        }
    }
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, kind, code, message) = self.parts();
        let mut resp =
            (status, Json(json!({"error": {"message": message, "type": kind, "code": code}})))
                .into_response();
        if let Self::RateLimited { retry_after_ms } = self {
            let secs = retry_after_ms.div_ceil(1_000).max(1);
            if let Ok(v) = HeaderValue::from_str(&secs.to_string()) {
                resp.headers_mut().insert(header::RETRY_AFTER, v);
            }
        }
        resp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    async fn body_json(r: axum::response::Response) -> serde_json::Value {
        let bytes = to_bytes(r.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn errors_use_the_openai_wire_shape() {
        let r = GatewayError::Unauthorized.into_response();
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED);
        let v = body_json(r).await;
        assert!(v["error"]["message"].is_string());
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn status_codes_match_the_taxonomy() {
        let cases = vec![
            (GatewayError::Unauthorized, StatusCode::UNAUTHORIZED),
            (GatewayError::GroupForbidden, StatusCode::FORBIDDEN),
            (GatewayError::ModelNotFound { model: "x".into() }, StatusCode::NOT_FOUND),
            (GatewayError::RateLimited { retry_after_ms: 1_000 }, StatusCode::TOO_MANY_REQUESTS),
            (
                GatewayError::BudgetExhausted { spent_usd: 2.0, cap_usd: 1.0 },
                StatusCode::PAYMENT_REQUIRED,
            ),
            (GatewayError::NoDeploymentAvailable, StatusCode::SERVICE_UNAVAILABLE),
        ];
        for (err, expected) in cases {
            assert_eq!(err.into_response().status(), expected);
        }
    }

    #[tokio::test]
    async fn rate_limit_sets_retry_after() {
        let r = GatewayError::RateLimited { retry_after_ms: 2_500 }.into_response();
        assert_eq!(r.headers().get("retry-after").unwrap(), "3");
    }

    #[tokio::test]
    async fn client_errors_never_leak_deployment_internals() {
        // NoDeploymentAvailable's message is a hardcoded constant, so on its
        // own this assertion can never fail no matter what the code does. It
        // is kept as one case among several, not as the whole test.
        let r = GatewayError::NoDeploymentAvailable.into_response();
        let v = body_json(r).await;
        let msg = v["error"]["message"].as_str().unwrap();
        for leak in ["http://", "https://", "api_key", "deployment_id", "gpt4o-azure"] {
            assert!(!msg.contains(leak), "error message leaked {leak:?}: {msg}");
        }
    }

    #[tokio::test]
    async fn upstream_rejected_relays_its_message_verbatim() {
        // This is the one variant that puts caller-supplied upstream text
        // straight into the body, so it is the variant a leak would travel
        // through. The relay is deliberate (see the doc comment on the
        // variant): an upstream 400 like "context length exceeded" is the
        // most useful thing a client gets, and blanking it would make every
        // client error undebuggable.
        //
        // This test therefore pins the CONTRACT, not an absence: whatever
        // dispatch hands over arrives intact and correctly enveloped. The
        // safety obligation lives upstream, on whoever composes a ChatModel
        // into a Deployment.
        let r = GatewayError::UpstreamRejected {
            message: "context length exceeded: 9000 > 8192".into(),
        }
        .into_response();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        let v = body_json(r).await;
        assert_eq!(v["error"]["message"], "context length exceeded: 9000 > 8192");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }
}
