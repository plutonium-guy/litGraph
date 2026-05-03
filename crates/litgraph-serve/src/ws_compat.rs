//! Cross-module shape used by both `lib.rs` (HTTP `InvokeRequest`) and
//! `ws.rs` (WebSocket invoke frame). Kept in its own module so the
//! struct is `pub(crate)` instead of leaking through the public API.

use litgraph_core::{ChatOptions, Message};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct WsInvokeRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub options: ChatOptions,
}
