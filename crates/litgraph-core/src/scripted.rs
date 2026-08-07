use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::stream;
use parking_lot::Mutex;

use crate::{
    ChatModel, ChatOptions, ChatResponse, ChatStream, ChatStreamEvent, Error, FinishReason,
    Message, Result, TokenUsage,
};

#[derive(Debug, Clone)]
pub enum ScriptedReply {
    Response(ChatResponse),
    Error(String),
}

impl ScriptedReply {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Response(ChatResponse {
            message: Message::assistant(text),
            finish_reason: FinishReason::Stop,
            usage: TokenUsage::default(),
            model: "scripted".into(),
        })
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(message.into())
    }
}

#[derive(Debug, Clone)]
pub struct ScriptedCall {
    pub sequence: usize,
    pub messages: Vec<Message>,
    pub options: ChatOptions,
}

#[derive(Debug)]
pub struct ScriptedChatModel {
    name: String,
    replies: Vec<ScriptedReply>,
    next: AtomicUsize,
    calls: Mutex<Vec<ScriptedCall>>,
    cycle: bool,
    stream_chunk_size: usize,
}

impl ScriptedChatModel {
    pub fn new(replies: Vec<ScriptedReply>) -> Result<Self> {
        if replies.is_empty() {
            return Err(Error::invalid(
                "ScriptedChatModel requires at least one reply",
            ));
        }
        Ok(Self {
            name: "scripted".into(),
            replies,
            next: AtomicUsize::new(0),
            calls: Mutex::new(Vec::new()),
            cycle: false,
            stream_chunk_size: 1,
        })
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_cycle(mut self, cycle: bool) -> Self {
        self.cycle = cycle;
        self
    }

    pub fn with_stream_chunk_size(mut self, stream_chunk_size: usize) -> Result<Self> {
        if stream_chunk_size == 0 {
            return Err(Error::invalid("stream_chunk_size must be positive"));
        }
        self.stream_chunk_size = stream_chunk_size;
        Ok(self)
    }

    pub fn invoke_sync(
        &self,
        messages: Vec<Message>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let sequence = self.next.fetch_add(1, Ordering::SeqCst);
        self.calls.lock().push(ScriptedCall {
            sequence,
            messages,
            options: options.clone(),
        });
        let index = if self.cycle {
            sequence % self.replies.len()
        } else if sequence < self.replies.len() {
            sequence
        } else {
            return Err(Error::provider(format!(
                "scripted replies exhausted after {} call(s); add replies or set cycle=True",
                self.replies.len()
            )));
        };
        match &self.replies[index] {
            ScriptedReply::Response(response) => {
                let mut response = response.clone();
                if response.model == "scripted" {
                    response.model = self.name.clone();
                }
                Ok(response)
            }
            ScriptedReply::Error(message) => Err(Error::provider(message.clone())),
        }
    }

    pub fn calls(&self) -> Vec<ScriptedCall> {
        self.calls.lock().clone()
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().len()
    }

    pub fn remaining(&self) -> Option<usize> {
        if self.cycle {
            None
        } else {
            Some(
                self.replies
                    .len()
                    .saturating_sub(self.next.load(Ordering::SeqCst)),
            )
        }
    }

    pub fn reset(&self, clear_calls: bool) {
        self.next.store(0, Ordering::SeqCst);
        if clear_calls {
            self.calls.lock().clear();
        }
    }
}

#[async_trait]
impl ChatModel for ScriptedChatModel {
    fn name(&self) -> &str {
        &self.name
    }

    async fn invoke(&self, messages: Vec<Message>, options: &ChatOptions) -> Result<ChatResponse> {
        self.invoke_sync(messages, options)
    }

    async fn stream(&self, messages: Vec<Message>, options: &ChatOptions) -> Result<ChatStream> {
        let response = self.invoke_sync(messages, options)?;
        let chars: Vec<char> = response.message.text_content().chars().collect();
        let mut events = Vec::with_capacity(chars.len() / self.stream_chunk_size + 1);
        for chunk in chars.chunks(self.stream_chunk_size) {
            events.push(Ok(ChatStreamEvent::Delta {
                text: chunk.iter().collect(),
            }));
        }
        events.push(Ok(ChatStreamEvent::Done { response }));
        Ok(Box::pin(stream::iter(events)))
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;

    #[tokio::test]
    async fn consumes_replies_and_records_calls() {
        let model =
            ScriptedChatModel::new(vec![ScriptedReply::text("one"), ScriptedReply::text("two")])
                .unwrap();
        let first = model
            .invoke(vec![Message::user("a")], &ChatOptions::default())
            .await
            .unwrap();
        let second = model
            .invoke(vec![Message::user("b")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(first.message.text_content(), "one");
        assert_eq!(second.message.text_content(), "two");
        assert_eq!(model.call_count(), 2);
        assert_eq!(model.calls()[1].messages[0].text_content(), "b");
    }

    #[tokio::test]
    async fn exhaustion_is_explicit() {
        let model = ScriptedChatModel::new(vec![ScriptedReply::text("one")]).unwrap();
        model.invoke(vec![], &ChatOptions::default()).await.unwrap();
        let error = model
            .invoke(vec![], &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("scripted replies exhausted"));
    }

    #[tokio::test]
    async fn cycle_reuses_the_script() {
        let model = ScriptedChatModel::new(vec![ScriptedReply::text("same")])
            .unwrap()
            .with_cycle(true);
        for _ in 0..3 {
            assert_eq!(
                model
                    .invoke(vec![], &ChatOptions::default())
                    .await
                    .unwrap()
                    .message
                    .text_content(),
                "same"
            );
        }
        assert_eq!(model.remaining(), None);
    }

    #[tokio::test]
    async fn stream_chunks_text_and_finishes() {
        let model = ScriptedChatModel::new(vec![ScriptedReply::text("hello")])
            .unwrap()
            .with_stream_chunk_size(2)
            .unwrap();
        let events: Vec<_> = model
            .stream(vec![], &ChatOptions::default())
            .await
            .unwrap()
            .collect()
            .await;
        let deltas: Vec<_> = events
            .iter()
            .filter_map(|event| match event.as_ref().unwrap() {
                ChatStreamEvent::Delta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec!["he", "ll", "o"]);
        assert!(matches!(
            events.last().unwrap().as_ref().unwrap(),
            ChatStreamEvent::Done { .. }
        ));
    }
}
