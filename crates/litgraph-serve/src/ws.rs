//! WebSocket endpoint for `litgraph-serve`. Adds bidirectional
//! streaming on top of the SSE endpoint:
//!
//! - **SSE (`POST /stream`)** — server pushes events; client cannot
//!   send back-channel signals without a second connection.
//! - **WebSocket (`GET /ws`)** — full duplex; client sends one
//!   `InvokeRequest` JSON message, server streams `ChatStreamEvent`
//!   JSON frames back, client may send `{"action":"cancel"}` at any
//!   time to abort the in-flight stream.
//!
//! # Why a separate module + feature flag
//!
//! WebSocket support pulls `tokio-tungstenite` (~ 30 KB) plus a
//! handful of axum extras. Default users only need SSE (covers ~ 95
//! % of agent UIs), so the WS path is gated behind a Cargo feature
//! `ws`. Enable in `Cargo.toml`:
//!
//! ```toml
//! litgraph-serve = { version = "0.1", features = ["ws"] }
//! ```
//!
//! # Wire format
//!
//! Client sends one text frame:
//!
//! ```json
//! { "messages": [{"role":"user","content":"hi"}], "options": {} }
//! ```
//!
//! Server replies with a sequence of text frames, each a JSON-encoded
//! `ChatStreamEvent`. After the final `Done` event the server sends:
//!
//! ```json
//! { "kind": "done" }
//! ```
//!
//! …then closes the connection. Errors come through as:
//!
//! ```json
//! { "kind": "error", "message": "<text>" }
//! ```
//!
//! Client may at any time send `{"action":"cancel"}` to drop the
//! upstream stream. The server tears down the model stream + closes
//! the socket.

use std::sync::Arc;

use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use futures::{SinkExt, StreamExt};
use litgraph_core::ChatModel;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::oneshot;
use tracing::debug;

#[derive(Clone)]
pub(crate) struct WsState {
    pub model: Arc<dyn ChatModel>,
}

/// Mount the `/ws` route on the given router. Call alongside
/// [`crate::router_for`] when you want both HTTP and WS endpoints
/// sharing the same model.
pub fn router_for_ws(model: Arc<dyn ChatModel>) -> Router {
    let state = WsState { model };
    Router::new().route("/ws", get(ws_upgrade)).with_state(state)
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(s): State<WsState>) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, s.model))
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ClientFrame {
    Invoke(crate::ws_compat::WsInvokeRequest),
    Control { action: String },
}

async fn handle_socket(mut socket: WebSocket, model: Arc<dyn ChatModel>) {
    // First frame must be the invoke request. Anything else closes
    // the connection with an error frame.
    let initial = match socket.recv().await {
        Some(Ok(WsMessage::Text(t))) => t,
        Some(Ok(WsMessage::Close(_))) | None => return,
        Some(Ok(_)) => {
            let _ = send_error(&mut socket, "expected text frame").await;
            return;
        }
        Some(Err(e)) => {
            debug!("ws recv error: {e}");
            return;
        }
    };

    let req: crate::ws_compat::WsInvokeRequest = match serde_json::from_str(&initial) {
        Ok(r) => r,
        Err(e) => {
            let _ = send_error(&mut socket, &format!("bad invoke json: {e}")).await;
            return;
        }
    };

    // Spawn a cancellation watcher: any subsequent client frame ending
    // in `{"action":"cancel"}` flips the oneshot.
    let (cancel_tx, mut cancel_rx) = oneshot::channel::<()>();
    let mut cancel_tx = Some(cancel_tx);
    let (mut sink, mut src) = socket.split();

    tokio::spawn(async move {
        while let Some(frame) = src.next().await {
            let Ok(WsMessage::Text(t)) = frame else {
                break;
            };
            if let Ok(ClientFrame::Control { action }) = serde_json::from_str::<ClientFrame>(&t) {
                if action == "cancel" {
                    if let Some(tx) = cancel_tx.take() {
                        let _ = tx.send(());
                    }
                    break;
                }
            }
        }
    });

    // Run the model stream + forward each event as a text frame.
    let stream_res = model.stream(req.messages, &req.options).await;
    let mut stream = match stream_res {
        Ok(s) => s,
        Err(e) => {
            let _ = sink
                .send(WsMessage::Text(
                    json!({"kind": "error", "message": e.to_string()}).to_string(),
                ))
                .await;
            let _ = sink.send(WsMessage::Close(None)).await;
            return;
        }
    };

    loop {
        tokio::select! {
            _ = &mut cancel_rx => {
                let _ = sink
                    .send(WsMessage::Text(
                        json!({"kind": "cancelled"}).to_string(),
                    ))
                    .await;
                break;
            }
            ev = stream.next() => match ev {
                Some(Ok(event)) => {
                    let body = match serde_json::to_string(&event) {
                        Ok(b) => b,
                        Err(e) => {
                            let _ = sink
                                .send(WsMessage::Text(
                                    json!({"kind": "error", "message": e.to_string()}).to_string(),
                                ))
                                .await;
                            break;
                        }
                    };
                    if sink.send(WsMessage::Text(body)).await.is_err() {
                        break;
                    }
                }
                Some(Err(e)) => {
                    let _ = sink
                        .send(WsMessage::Text(
                            json!({"kind": "error", "message": e.to_string()}).to_string(),
                        ))
                        .await;
                    break;
                }
                None => {
                    let _ = sink
                        .send(WsMessage::Text(
                            json!({"kind": "done"}).to_string(),
                        ))
                        .await;
                    break;
                }
            }
        }
    }

    let _ = sink.send(WsMessage::Close(None)).await;
}

async fn send_error(socket: &mut WebSocket, msg: &str) -> Result<(), axum::Error> {
    socket
        .send(WsMessage::Text(
            json!({"kind": "error", "message": msg}).to_string(),
        ))
        .await
}
