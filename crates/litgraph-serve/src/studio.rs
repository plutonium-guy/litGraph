//! LangGraph Studio-style debug endpoints over any
//! [`litgraph_graph::Checkpointer`].
//!
//! Mount alongside the chat router to expose thread state + history +
//! time-travel controls to a UI:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use litgraph_graph::{Checkpointer, MemoryCheckpointer};
//! # use axum::Router;
//! # async fn run(model: Arc<dyn litgraph_core::ChatModel>) {
//! let cp: Arc<dyn Checkpointer> = Arc::new(MemoryCheckpointer::default());
//! let app = litgraph_serve::router_for(model)
//!     .merge(litgraph_serve::studio::studio_router(cp));
//! # let _ = app;
//! # }
//! ```
//!
//! # Endpoints
//!
//! | Method | Path                                       | Returns                                |
//! |--------|--------------------------------------------|----------------------------------------|
//! | GET    | `/threads/:thread_id/state`                | summary of latest checkpoint           |
//! | GET    | `/threads/:thread_id/history`              | array of summaries, oldest-first       |
//! | GET    | `/threads/:thread_id/checkpoints/:step`    | full checkpoint w/ base64 state blob   |
//! | POST   | `/threads/:thread_id/rewind`               | body `{step}` → drop later checkpoints |
//! | DELETE | `/threads/:thread_id`                      | wipe all checkpoints for thread        |
//!
//! All endpoints return JSON. Errors → 500 with a JSON body
//! `{error: "..."}` (404-shaped errors come through axum's path
//! routing).
//!
//! # Auth
//!
//! Out of scope here — wrap the returned [`Router`] with a tower
//! middleware (bearer auth, IP allow-list) before exposing publicly.
//! These endpoints can wipe history; treat them as admin-only.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use base64::{engine::general_purpose, Engine as _};
use litgraph_graph::{Checkpoint, Checkpointer};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone)]
struct StudioState {
    cp: Arc<dyn Checkpointer>,
}

/// Build the debug router. Cheap — `Arc` clones inside.
pub fn studio_router(cp: Arc<dyn Checkpointer>) -> Router {
    let state = StudioState { cp };
    Router::new()
        .route("/threads/:thread_id/state", get(get_state))
        .route("/threads/:thread_id/history", get(get_history))
        .route(
            "/threads/:thread_id/checkpoints/:step",
            get(get_checkpoint),
        )
        .route("/threads/:thread_id/rewind", post(rewind))
        .route("/threads/:thread_id", delete(clear_thread))
        .with_state(state)
}

/// Summary view of a checkpoint — same shape used in `state` and
/// `history`. Excludes the raw state bytes so the UI can render a
/// list cheaply; full bytes are available via `/checkpoints/:step`.
#[derive(Debug, Clone, Serialize)]
struct CheckpointSummary {
    thread_id: String,
    step: u64,
    next_nodes: Vec<String>,
    has_pending_interrupt: bool,
    n_pending_sends: usize,
    state_bytes: usize,
    ts_ms: u64,
}

impl From<&Checkpoint> for CheckpointSummary {
    fn from(cp: &Checkpoint) -> Self {
        Self {
            thread_id: cp.thread_id.clone(),
            step: cp.step,
            next_nodes: cp.next_nodes.clone(),
            has_pending_interrupt: cp.pending_interrupt.is_some(),
            n_pending_sends: cp.next_sends.len(),
            state_bytes: cp.state.len(),
            ts_ms: cp.ts_ms,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RewindRequest {
    step: u64,
}

#[derive(Debug, Serialize)]
struct ApiErrorBody {
    error: String,
}

// ---- handlers --------------------------------------------------------------

async fn get_state(
    State(s): State<StudioState>,
    Path(thread_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let latest = s.cp.latest(&thread_id).await.map_err(ApiError::from)?;
    match latest {
        Some(cp) => Ok(Json(json!(CheckpointSummary::from(&cp)))),
        None => Ok(Json(Value::Null)),
    }
}

async fn get_history(
    State(s): State<StudioState>,
    Path(thread_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let mut all = s.cp.list(&thread_id).await.map_err(ApiError::from)?;
    all.sort_by_key(|c| c.step);
    let summaries: Vec<CheckpointSummary> = all.iter().map(CheckpointSummary::from).collect();
    Ok(Json(json!(summaries)))
}

async fn get_checkpoint(
    State(s): State<StudioState>,
    Path((thread_id, step)): Path<(String, u64)>,
) -> Result<Json<Value>, ApiError> {
    let cp = s
        .cp
        .get(&thread_id, step)
        .await
        .map_err(ApiError::from)?
        .ok_or_else(|| ApiError::NotFound(format!("no checkpoint at step {step}")))?;
    // Encode raw state as base64 — clients that need the JSON-decoded
    // form should call back into the runtime since decoding requires
    // the state type, which is graph-specific. Studio UIs typically
    // just display the byte length / cuts of the bytes for debugging.
    let state_b64 = general_purpose::STANDARD.encode(&cp.state);
    Ok(Json(json!({
        "thread_id": cp.thread_id,
        "step": cp.step,
        "next_nodes": cp.next_nodes,
        "n_pending_sends": cp.next_sends.len(),
        "has_pending_interrupt": cp.pending_interrupt.is_some(),
        "state_bytes": cp.state.len(),
        "state_base64": state_b64,
        "ts_ms": cp.ts_ms,
    })))
}

async fn rewind(
    State(s): State<StudioState>,
    Path(thread_id): Path<String>,
    Json(req): Json<RewindRequest>,
) -> Result<Json<Value>, ApiError> {
    let dropped = s
        .cp
        .rewind_to(&thread_id, req.step)
        .await
        .map_err(ApiError::from)?;
    Ok(Json(json!({ "dropped": dropped, "step": req.step })))
}

async fn clear_thread(
    State(s): State<StudioState>,
    Path(thread_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    s.cp.clear_thread(&thread_id)
        .await
        .map_err(ApiError::from)?;
    Ok(Json(json!({ "cleared": thread_id })))
}

// ---- error -----------------------------------------------------------------

enum ApiError {
    Graph(litgraph_graph::GraphError),
    NotFound(String),
}

impl From<litgraph_graph::GraphError> for ApiError {
    fn from(e: litgraph_graph::GraphError) -> Self {
        ApiError::Graph(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            ApiError::NotFound(m) => (StatusCode::NOT_FOUND, m),
            ApiError::Graph(e) => {
                // `rewind_to` on a missing step is a user error → 400,
                // not a 500. Detect via the message; cleaner than a
                // separate variant on the upstream enum.
                let s = e.to_string();
                if s.contains("no checkpoint at step") {
                    (StatusCode::BAD_REQUEST, s)
                } else {
                    (StatusCode::INTERNAL_SERVER_ERROR, s)
                }
            }
        };
        (status, Json(ApiErrorBody { error: msg })).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use litgraph_graph::{Checkpoint, MemoryCheckpointer};
    use tower::ServiceExt;

    /// Async helper. Tests `.await` it inside their own
    /// `#[tokio::test]` so we don't try to spin a nested runtime.
    async fn cp_with_steps(thread: &str, steps: &[u64]) -> Arc<dyn Checkpointer> {
        let mem = Arc::new(MemoryCheckpointer::default()) as Arc<dyn Checkpointer>;
        for (i, &step) in steps.iter().enumerate() {
            let cp = Checkpoint {
                thread_id: thread.into(),
                step,
                state: vec![i as u8; 8],
                next_nodes: vec![format!("node-{step}")],
                next_sends: Vec::new(),
                pending_interrupt: None,
                ts_ms: 1_000_000 + step,
            };
            mem.put(cp).await.unwrap();
        }
        mem
    }

    fn app(cp: Arc<dyn Checkpointer>) -> Router {
        studio_router(cp)
    }

    async fn body_to_json(resp: Response) -> Value {
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let s = String::from_utf8(bytes.to_vec()).unwrap();
        serde_json::from_str(&s).unwrap_or_else(|e| panic!("not json: {s} ({e})"))
    }

    // ---- get_state ----

    #[tokio::test]
    async fn state_returns_latest_checkpoint_summary() {
        let cp = cp_with_steps("t1", &[0, 1, 2, 5]).await;
        let resp = app(cp)
            .oneshot(Request::get("/threads/t1/state").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        assert_eq!(v["thread_id"], "t1");
        assert_eq!(v["step"], 5);
        assert_eq!(v["state_bytes"], 8);
        assert_eq!(v["next_nodes"][0], "node-5");
        assert_eq!(v["has_pending_interrupt"], false);
        assert_eq!(v["n_pending_sends"], 0);
    }

    #[tokio::test]
    async fn state_returns_null_for_unknown_thread() {
        let cp = cp_with_steps("t1", &[0]).await;
        let resp = app(cp)
            .oneshot(Request::get("/threads/nope/state").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        assert!(v.is_null());
    }

    // ---- get_history ----

    #[tokio::test]
    async fn history_returns_summaries_oldest_first() {
        let cp = cp_with_steps("t1", &[3, 1, 2, 0]).await; // inserted out of order
        let resp = app(cp)
            .oneshot(
                Request::get("/threads/t1/history")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let v = body_to_json(resp).await;
        let arr = v.as_array().unwrap();
        let steps: Vec<u64> = arr.iter().map(|c| c["step"].as_u64().unwrap()).collect();
        assert_eq!(steps, vec![0, 1, 2, 3], "steps must come back oldest-first");
    }

    #[tokio::test]
    async fn history_empty_thread_returns_empty_array() {
        let cp = cp_with_steps("t1", &[]).await;
        let resp = app(cp)
            .oneshot(
                Request::get("/threads/t1/history")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let v = body_to_json(resp).await;
        assert_eq!(v.as_array().unwrap().len(), 0);
    }

    // ---- get_checkpoint ----

    #[tokio::test]
    async fn checkpoint_returns_full_blob_base64() {
        let cp = cp_with_steps("t1", &[0, 1, 2]).await;
        let resp = app(cp)
            .oneshot(
                Request::get("/threads/t1/checkpoints/1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        assert_eq!(v["step"], 1);
        let b64 = v["state_base64"].as_str().unwrap();
        let bytes = general_purpose::STANDARD.decode(b64).unwrap();
        // Step 1 was the second insert (i=1 in the loop) so all bytes = 1.
        assert_eq!(bytes, vec![1u8; 8]);
    }

    #[tokio::test]
    async fn checkpoint_missing_step_returns_404() {
        let cp = cp_with_steps("t1", &[0]).await;
        let resp = app(cp)
            .oneshot(
                Request::get("/threads/t1/checkpoints/999")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let v = body_to_json(resp).await;
        assert!(v["error"].as_str().unwrap().contains("step 999"));
    }

    // ---- rewind ----

    #[tokio::test]
    async fn rewind_drops_later_checkpoints_returns_count() {
        let cp = cp_with_steps("t1", &[0, 1, 2, 3, 4]).await;
        let body = json!({"step": 2}).to_string();
        let resp = app(cp.clone())
            .oneshot(
                Request::post("/threads/t1/rewind")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        assert_eq!(v["dropped"], 2);
        assert_eq!(v["step"], 2);
        // After rewind, latest is step 2.
        let latest = cp.latest("t1").await.unwrap().unwrap();
        assert_eq!(latest.step, 2);
    }

    #[tokio::test]
    async fn rewind_to_unknown_step_returns_400() {
        let cp = cp_with_steps("t1", &[0, 1, 2]).await;
        let body = json!({"step": 99}).to_string();
        let resp = app(cp)
            .oneshot(
                Request::post("/threads/t1/rewind")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_to_json(resp).await;
        assert!(v["error"].as_str().unwrap().contains("no checkpoint at step 99"));
    }

    #[tokio::test]
    async fn rewind_with_malformed_json_400() {
        let cp = cp_with_steps("t1", &[0, 1]).await;
        let resp = app(cp)
            .oneshot(
                Request::post("/threads/t1/rewind")
                    .header("content-type", "application/json")
                    .body(Body::from("not json"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ---- clear_thread ----

    #[tokio::test]
    async fn delete_thread_clears_all_checkpoints() {
        let cp = cp_with_steps("t1", &[0, 1, 2]).await;
        let resp = app(cp.clone())
            .oneshot(
                Request::delete("/threads/t1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        assert_eq!(v["cleared"], "t1");
        assert!(cp.latest("t1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_unknown_thread_succeeds_idempotently() {
        let cp = cp_with_steps("t1", &[0]).await;
        let resp = app(cp)
            .oneshot(
                Request::delete("/threads/nope")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // MemoryCheckpointer's clear_thread is a no-op on missing
        // — we surface 200, not 404. (Studio UIs poll, idempotency
        // keeps the path simple.)
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ---- routing ----

    #[tokio::test]
    async fn unknown_route_404() {
        let cp = cp_with_steps("t1", &[0]).await;
        let resp = app(cp)
            .oneshot(Request::get("/nope").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
