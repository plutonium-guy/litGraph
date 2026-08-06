use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, KeyedMutex, Message, Result};

/// Wrap any [`ChatModel`] so calls sharing the same key serialize
/// against each other; calls with different keys run in parallel.
/// Bridges the iter-241 [`KeyedMutex`] primitive into the
/// resilience family.
///
/// # Why this exists
///
/// A stateful agent's ReAct loop reads-then-writes shared
/// conversation state per `thread_id`. Two concurrent steps for
/// the same thread interleave their reads/writes and corrupt the
/// state. A single global mutex fixes correctness but serializes
/// EVERYONE — a thousand different threads run one-at-a-time.
///
/// `KeyedSerializedChatModel` lifts the per-thread ReAct-step lock
/// into the model wrapper layer: same `thread_id` queues, different
/// `thread_id`s run concurrently.
///
/// # Key extraction
///
/// The constructor takes a `key_fn: impl Fn(&[Message],
/// &ChatOptions) -> Option<String>`. Returning `None` means "no
/// serialization for this call" (passes through). Typical
/// extractor: `|_, opts| opts.metadata.get("thread_id").cloned()`.
///
/// `String` is fixed as the key type — most prod keys (thread IDs,
/// user IDs, document IDs) stringify naturally and dyn-trait
/// fungibility is more important than generic-K flexibility here.
///
/// # Streaming
///
/// `stream()` acquires the lock, calls the inner `stream()`, and
/// IMPORTANTLY drops the lock when this method returns — NOT
/// when the stream finishes. The caller drives the stream after
/// the handshake, so the lock must release at handshake to allow
/// other same-key calls to queue. If you need per-step exclusion
/// across the whole stream lifetime, hold the lock externally
/// around the `stream` + `drain` cycle.
pub struct KeyedSerializedChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub registry: Arc<KeyedMutex<String>>,
    key_fn: Arc<dyn Fn(&[Message], &ChatOptions) -> Option<String> + Send + Sync>,
}

impl KeyedSerializedChatModel {
    pub fn new<F>(
        inner: Arc<dyn ChatModel>,
        registry: Arc<KeyedMutex<String>>,
        key_fn: F,
    ) -> Self
    where
        F: Fn(&[Message], &ChatOptions) -> Option<String> + Send + Sync + 'static,
    {
        Self {
            inner,
            registry,
            key_fn: Arc::new(key_fn),
        }
    }
}

#[async_trait]
impl ChatModel for KeyedSerializedChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let key = (self.key_fn)(&messages, opts);
        if let Some(k) = key {
            let _guard = self.registry.lock(k).await;
            self.inner.invoke(messages, opts).await
        } else {
            self.inner.invoke(messages, opts).await
        }
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        let key = (self.key_fn)(&messages, opts);
        if let Some(k) = key {
            // Hold the lock only for the handshake (see doc above).
            let _guard = self.registry.lock(k).await;
            self.inner.stream(messages, opts).await
        } else {
            self.inner.stream(messages, opts).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    #[allow(unused_imports)]
    use litgraph_core::tool::Tool as _;
    #[allow(unused_imports)]
    use litgraph_core::{ContentPart, Message, Role};
    #[allow(unused_imports)]
    use std::sync::atomic::{AtomicU32, Ordering};
    #[allow(unused_imports)]
    use litgraph_core::Error;
    #[allow(unused_imports)]
    use std::time::Duration;

    /// Always-succeeds model for closed-path test.
    struct AlwaysOkModel;

    #[async_trait]
    impl ChatModel for AlwaysOkModel {
        fn name(&self) -> &str {
            "always-ok"
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text: "hi".into() }],
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "always-ok".into(),
            })
        }
        async fn stream(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    // ---- KeyedSerializedChatModel tests --------------------------------

    /// Sleeps `delay_ms` per call and tracks peak concurrent calls.
    /// Lets keyed-mutex tests verify same-key serialization vs
    /// different-key parallelism.
    struct DelayChatModel {
        delay_ms: u64,
        in_flight: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl ChatModel for DelayChatModel {
        fn name(&self) -> &str {
            "delay-chat"
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut p = self.peak.load(Ordering::SeqCst);
            while now > p {
                match self.peak.compare_exchange(
                    p,
                    now,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(actual) => p = actual,
                }
            }
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text: "ok".into() }],
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "delay-chat".into(),
            })
        }
        async fn stream(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn key_from_first_user_text(
        msgs: &[Message],
        _opts: &ChatOptions,
    ) -> Option<String> {
        msgs.iter()
            .find(|m| m.role == Role::User)
            .map(|m| m.text_content().to_string())
    }

    #[tokio::test]
    async fn keyed_serialized_same_key_serializes() {
        use std::sync::atomic::AtomicUsize;
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn ChatModel> = Arc::new(DelayChatModel {
            delay_ms: 20,
            in_flight: in_flight.clone(),
            peak: peak.clone(),
        });
        let registry = Arc::new(KeyedMutex::<String>::new());
        let cm = Arc::new(KeyedSerializedChatModel::new(
            inner,
            registry,
            key_from_first_user_text,
        ));
        let mut handles = Vec::new();
        for _ in 0..5 {
            let cm = cm.clone();
            handles.push(tokio::spawn(async move {
                cm.invoke(
                    vec![Message::user("user_a")],
                    &ChatOptions::default(),
                )
                .await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }
        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "same-key didn't serialize",
        );
    }

    #[tokio::test]
    async fn keyed_serialized_different_keys_run_in_parallel() {
        use std::sync::atomic::AtomicUsize;
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn ChatModel> = Arc::new(DelayChatModel {
            delay_ms: 30,
            in_flight: in_flight.clone(),
            peak: peak.clone(),
        });
        let registry = Arc::new(KeyedMutex::<String>::new());
        let cm = Arc::new(KeyedSerializedChatModel::new(
            inner,
            registry,
            key_from_first_user_text,
        ));
        let mut handles = Vec::new();
        for i in 0..5 {
            let cm = cm.clone();
            handles.push(tokio::spawn(async move {
                cm.invoke(
                    vec![Message::user(format!("thread_{i}"))],
                    &ChatOptions::default(),
                )
                .await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }
        assert!(
            peak.load(Ordering::SeqCst) >= 2,
            "different keys never ran concurrently",
        );
    }

    #[tokio::test]
    async fn keyed_serialized_none_key_passes_through() {
        // key_fn returns None → no lock is acquired; calls run
        // fully concurrently regardless of arguments.
        use std::sync::atomic::AtomicUsize;
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn ChatModel> = Arc::new(DelayChatModel {
            delay_ms: 25,
            in_flight: in_flight.clone(),
            peak: peak.clone(),
        });
        let registry = Arc::new(KeyedMutex::<String>::new());
        let cm = Arc::new(KeyedSerializedChatModel::new(
            inner,
            registry,
            |_msgs, _opts| None,
        ));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let cm = cm.clone();
            handles.push(tokio::spawn(async move {
                cm.invoke(
                    vec![Message::user("any")],
                    &ChatOptions::default(),
                )
                .await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }
        assert!(
            peak.load(Ordering::SeqCst) >= 2,
            "None key still serialized: peak {}",
            peak.load(Ordering::SeqCst),
        );
    }

    #[tokio::test]
    async fn keyed_serialized_name_proxies_inner() {
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let registry = Arc::new(KeyedMutex::<String>::new());
        let cm = KeyedSerializedChatModel::new(
            inner,
            registry,
            |_, _| Some("k".into()),
        );
        assert_eq!(cm.name(), "always-ok");
    }

}
