//! `InstrumentedChatModel` — wraps any `ChatModel`, emits lifecycle events into
//! a `CallbackHandle`. Users who don't want to modify every provider get
//! observability for free by wrapping once at the top level.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Message, Result};

use crate::callback::CallbackHandle;
use crate::event::{Event, Phase};

pub struct InstrumentedChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub events: CallbackHandle,
}

impl InstrumentedChatModel {
    pub fn new(inner: Arc<dyn ChatModel>, events: CallbackHandle) -> Self {
        Self { inner, events }
    }
}

#[async_trait]
impl ChatModel for InstrumentedChatModel {
    fn name(&self) -> &str { self.inner.name() }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let model = self.inner.name().to_string();
        self.events.emit(Event::Llm {
            phase: Phase::Start,
            model: model.clone(),
            usage: None,
            error: None,
            ts_ms: Event::now_ms(),
        });
        match self.inner.invoke(messages, opts).await {
            Ok(resp) => {
                self.events.emit(Event::Llm {
                    phase: Phase::End,
                    model,
                    usage: Some(resp.usage),
                    error: None,
                    ts_ms: Event::now_ms(),
                });
                Ok(resp)
            }
            Err(e) => {
                self.events.emit(Event::Llm {
                    phase: Phase::Error,
                    model,
                    usage: None,
                    error: Some(e.to_string()),
                    ts_ms: Event::now_ms(),
                });
                Err(e)
            }
        }
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        // Stream bypass — emit only Start; downstream stream handler should emit LlmToken
        // + the terminal End.
        self.events.emit(Event::Llm {
            phase: Phase::Start,
            model: self.inner.name().to_string(),
            usage: None,
            error: None,
            ts_ms: Event::now_ms(),
        });
        self.inner.stream(messages, opts).await
    }
}
