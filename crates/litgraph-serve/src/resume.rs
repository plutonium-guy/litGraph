//! Webhook-resume HTTP bridge — axum router that wraps a
//! [`litgraph_core::ResumeRegistry`] (iter 201) so external systems
//! can wake paused agent threads via simple HTTP calls.
//!
//! Pattern: an agent run hits `Command(interrupt)` and parks on
//! `registry.register(thread_id).await_resume()`. Some external
//! event — a Slack interactive button, a webhook callback, a human
//! approval workflow — `POST`s the resume payload to this router,
//! which delivers it to the parked thread.
//!
//! Mount alongside the chat router (and optionally the studio
//! router) to expose interrupt-resume controls to UIs:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use litgraph_core::ResumeRegistry;
//! # use axum::Router;
//! # async fn run(model: Arc<dyn litgraph_core::ChatModel>) {
//! let registry = ResumeRegistry::new();
//! let app = litgraph_serve::router_for(model)
//!     .merge(litgraph_serve::resume::resume_router(registry.clone()));
//! # let _ = app;
//! # }
//! ```
//!
//! # Endpoints
//!
//! | Method | Path                          | Body                       | Returns                          |
//! |--------|-------------------------------|----------------------------|----------------------------------|
//! | POST   | `/threads/:thread_id/resume`  | `{ "value": <any-json> }`  | `{ "delivered": true }`          |
//! | DELETE | `/threads/:thread_id/resume`  | (none)                     | `{ "cancelled": true \| false }` |
//! | GET    | `/resumes/pending`            | (none)                     | `{ "pending": [thread_id, ...] }` |
//!
//! All endpoints return JSON. `POST` returns 404 if no thread is
//! currently registered for that id (or it already resumed).
//! Validation errors return 400 with `{ "error": "..." }`.
//!
//! # Auth
//!
//! Out of scope here — these endpoints can wake any registered
//! thread, so wrap the returned [`Router`] with bearer auth or an
//! IP allow-list before exposing publicly.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use litgraph_core::ResumeRegistry;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Build the resume-bridge router. Cheap — `ResumeRegistry` is
/// `Arc`-shared internally.
pub fn resume_router(registry: ResumeRegistry) -> Router {
    Router::new()
        .route("/threads/:thread_id/resume", post(post_resume))
        .route("/threads/:thread_id/resume", delete(delete_resume))
        .route("/resumes/pending", get(get_pending))
        .with_state(registry)
}

#[derive(Debug, Deserialize)]
struct ResumeBody {
    /// Arbitrary JSON value handed to the parked agent's
    /// `await_resume()` future.
    value: Value,
}

#[derive(Debug, Serialize)]
struct ApiErrorBody {
    error: String,
}

async fn post_resume(
    State(registry): State<ResumeRegistry>,
    Path(thread_id): Path<String>,
    Json(body): Json<ResumeBody>,
) -> Result<Json<Value>, ApiError> {
    registry
        .resume(&thread_id, body.value)
        .map_err(|e| {
            // Treat "no pending resume" as 404; everything else as 500.
            let s = e.to_string();
            if s.contains("no pending resume") {
                ApiError::NotFound(s)
            } else {
                ApiError::Other(s)
            }
        })?;
    Ok(Json(json!({ "delivered": true, "thread_id": thread_id })))
}

async fn delete_resume(
    State(registry): State<ResumeRegistry>,
    Path(thread_id): Path<String>,
) -> Json<Value> {
    let cancelled = registry.cancel(&thread_id);
    Json(json!({ "cancelled": cancelled, "thread_id": thread_id }))
}

async fn get_pending(State(registry): State<ResumeRegistry>) -> Json<Value> {
    let mut pending = registry.pending_ids();
    pending.sort();
    Json(json!({ "pending": pending, "count": pending.len() }))
}

enum ApiError {
    NotFound(String),
    Other(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            ApiError::NotFound(m) => (StatusCode::NOT_FOUND, m),
            ApiError::Other(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
        };
        (status, Json(ApiErrorBody { error: msg })).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    async fn body_to_json(resp: Response) -> Value {
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let s = String::from_utf8(bytes.to_vec()).unwrap();
        serde_json::from_str(&s).unwrap_or_else(|e| panic!("not json: {s} ({e})"))
    }

    #[tokio::test]
    async fn post_resume_delivers_value_to_pending_thread() {
        let registry = ResumeRegistry::new();
        let fut = registry.register("t1").unwrap();
        let h = tokio::spawn(async move { fut.await_resume().await });
        tokio::task::yield_now().await;

        let app = resume_router(registry);
        let body = json!({"value": {"approved": true, "note": "looks good"}}).to_string();
        let resp = app
            .oneshot(
                Request::post("/threads/t1/resume")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        assert_eq!(v["delivered"], true);
        assert_eq!(v["thread_id"], "t1");

        let got = h.await.unwrap();
        assert_eq!(got, Some(json!({"approved": true, "note": "looks good"})));
    }

    #[tokio::test]
    async fn post_resume_unknown_thread_returns_404() {
        let registry = ResumeRegistry::new();
        let app = resume_router(registry);
        let body = json!({"value": {}}).to_string();
        let resp = app
            .oneshot(
                Request::post("/threads/nobody/resume")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let v = body_to_json(resp).await;
        assert!(v["error"]
            .as_str()
            .unwrap()
            .contains("no pending resume"));
    }

    #[tokio::test]
    async fn post_resume_malformed_body_returns_400() {
        let registry = ResumeRegistry::new();
        let _f = registry.register("t1").unwrap();
        let app = resume_router(registry);
        let resp = app
            .oneshot(
                Request::post("/threads/t1/resume")
                    .header("content-type", "application/json")
                    .body(Body::from("{not json"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn delete_resume_cancels_pending() {
        let registry = ResumeRegistry::new();
        let fut = registry.register("t1").unwrap();
        let h = tokio::spawn(async move { fut.await_resume().await });
        tokio::task::yield_now().await;

        let app = resume_router(registry);
        let resp = app
            .oneshot(
                Request::delete("/threads/t1/resume")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        assert_eq!(v["cancelled"], true);

        let got = h.await.unwrap();
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn delete_resume_unknown_thread_returns_false() {
        let registry = ResumeRegistry::new();
        let app = resume_router(registry);
        let resp = app
            .oneshot(
                Request::delete("/threads/never-registered/resume")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        assert_eq!(v["cancelled"], false);
    }

    #[tokio::test]
    async fn get_pending_lists_active_threads() {
        let registry = ResumeRegistry::new();
        let _f1 = registry.register("alice").unwrap();
        let _f2 = registry.register("bob").unwrap();
        let _f3 = registry.register("carol").unwrap();

        let app = resume_router(registry);
        let resp = app
            .oneshot(
                Request::get("/resumes/pending")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        assert_eq!(v["count"], 3);
        let pending = v["pending"].as_array().unwrap();
        let ids: Vec<String> = pending
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids, vec!["alice", "bob", "carol"]);
    }

    #[tokio::test]
    async fn get_pending_empty_when_no_threads_registered() {
        let registry = ResumeRegistry::new();
        let app = resume_router(registry);
        let resp = app
            .oneshot(
                Request::get("/resumes/pending")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let v = body_to_json(resp).await;
        assert_eq!(v["count"], 0);
        assert_eq!(v["pending"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn double_resume_first_succeeds_second_404s() {
        let registry = ResumeRegistry::new();
        let fut = registry.register("t1").unwrap();
        let h = tokio::spawn(async move { fut.await_resume().await });
        tokio::task::yield_now().await;

        let app = resume_router(registry.clone());
        let body = json!({"value": "first"}).to_string();
        let resp = app
            .clone()
            .oneshot(
                Request::post("/threads/t1/resume")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = h.await;

        // Second resume: thread no longer registered → 404.
        let body = json!({"value": "second"}).to_string();
        let resp = app
            .oneshot(
                Request::post("/threads/t1/resume")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn registry_clone_shared_across_router_handlers() {
        // The registry passed in stays usable from outside the
        // router; a resume issued via HTTP is observed by an
        // external `pending_count()` check.
        let registry = ResumeRegistry::new();
        let _f = registry.register("t1").unwrap();
        assert_eq!(registry.pending_count(), 1);

        let app = resume_router(registry.clone());
        let body = json!({"value": "ok"}).to_string();
        let _ = app
            .oneshot(
                Request::post("/threads/t1/resume")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        // After resume, the external view also drops to 0.
        assert_eq!(registry.pending_count(), 0);
    }
}
