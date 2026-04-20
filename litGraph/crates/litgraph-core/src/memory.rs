//! Conversation memory — a thin abstraction over a growing message list with
//! pluggable eviction policies. The tradeoff space is well-understood:
//!
//! - `BufferMemory` keeps the last N turns. Simplest, predictable cost.
//! - `TokenBufferMemory` keeps messages under a token budget; evicts oldest
//!   when over. Requires a token-counter callback so this crate stays free of
//!   tokenizer deps.
//!
//! Both honor a `system_pin`: messages with `Role::System` (or messages
//! explicitly added via `set_system`) never count toward the eviction budget
//! and are always returned at the head of `messages()`. This matches the
//! production pattern of "system prompt is fixed, conversation rotates."
//!
//! Memories are sync — they don't do I/O. Persistence is the caller's job
//! (use a checkpointer if needed).

use serde::{Deserialize, Serialize};

use crate::{ChatModel, ChatOptions, Error, Message, Result, Role};

/// Counts tokens in a message. Pluggable so this crate stays tokenizer-free.
/// Plug `litgraph-tokenizers::Tokenizer::count` here, or a constant approx.
pub type TokenCounter = std::sync::Arc<dyn Fn(&Message) -> usize + Send + Sync>;

/// Bytes-on-the-wire shape for memory persistence. Versioned so a future
/// schema change can fail loudly instead of silently mis-deserializing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub version: u32,
    pub system: Option<Message>,
    pub history: Vec<Message>,
}

impl MemorySnapshot {
    pub const CURRENT_VERSION: u32 = 1;

    /// Serialize to JSON bytes. Chosen over bincode because `Message` /
    /// `ContentPart` use internally-tagged enums (`#[serde(tag = "type", ...)]`)
    /// which bincode rejects (`deserialize_any` unsupported). JSON also has
    /// the upside of being human-readable when spelunking through a stored
    /// session — a real win for ops at the cost of a few bytes vs bincode.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| Error::other(format!("memory serialize: {e}")))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let snap: Self = serde_json::from_slice(bytes)
            .map_err(|e| Error::other(format!("memory deserialize: {e}")))?;
        if snap.version != Self::CURRENT_VERSION {
            return Err(Error::other(format!(
                "memory snapshot version mismatch: got {}, expected {}",
                snap.version, Self::CURRENT_VERSION
            )));
        }
        Ok(snap)
    }
}

pub trait ConversationMemory: Send + Sync {
    /// Append one message. May trigger eviction.
    fn append(&mut self, m: Message);
    /// Snapshot the conversation in turn order, system-pinned message first.
    fn messages(&self) -> Vec<Message>;
    /// Drop all non-pinned messages.
    fn clear(&mut self);
    /// Replace (or set) the system-pinned prompt. Pass `None` to remove it.
    fn set_system(&mut self, m: Option<Message>);
}

/// Keep the last `max_turns` non-system messages. A "turn" is one message,
/// not a user+assistant pair — this keeps tool-call interleaving sane (a single
/// "logical turn" can be 1 user + 1 assistant + N tool messages).
#[derive(Debug)]
pub struct BufferMemory {
    pub max_messages: usize,
    system: Option<Message>,
    history: std::collections::VecDeque<Message>,
}

impl BufferMemory {
    pub fn new(max_messages: usize) -> Self {
        Self {
            max_messages: max_messages.max(1),
            system: None,
            history: std::collections::VecDeque::new(),
        }
    }
}

impl BufferMemory {
    /// Snapshot the current state for persistence. Pair with
    /// `to_bytes()` for a serialized blob.
    pub fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            version: MemorySnapshot::CURRENT_VERSION,
            system: self.system.clone(),
            history: self.history.iter().cloned().collect(),
        }
    }

    /// Restore from a snapshot. The `max_messages` cap on `self` is preserved;
    /// if the snapshot is larger, oldest messages are evicted.
    pub fn restore(&mut self, snap: MemorySnapshot) {
        self.system = snap.system;
        self.history.clear();
        for m in snap.history {
            self.history.push_back(m);
        }
        while self.history.len() > self.max_messages {
            self.history.pop_front();
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> { self.snapshot().to_bytes() }
    pub fn from_bytes(max_messages: usize, bytes: &[u8]) -> Result<Self> {
        let snap = MemorySnapshot::from_bytes(bytes)?;
        let mut m = Self::new(max_messages);
        m.restore(snap);
        Ok(m)
    }

    /// Summarize the oldest `summarize_count` messages via `model` and replace
    /// them with a single synthetic system message. Idempotent if the history
    /// is shorter than `summarize_count` (no-op + Ok).
    ///
    /// If a system pin is set, it's prepended verbatim to the summary so the
    /// distilled context inherits the original instructions. The new summary
    /// becomes the system pin (the prior pin is folded in).
    pub async fn summarize_and_compact(
        &mut self,
        model: &dyn ChatModel,
        summarize_count: usize,
    ) -> Result<()> {
        if summarize_count == 0 || self.history.len() < summarize_count {
            return Ok(());
        }
        let to_summarize: Vec<Message> = self.history.drain(..summarize_count).collect();
        let prior_summary = self.system.as_ref().map(|m| m.text_content());
        let summary = summarize_conversation(model, &to_summarize, prior_summary.as_deref()).await?;
        // Replace the system pin with the new running summary.
        self.system = Some(Message::system(summary));
        Ok(())
    }
}

/// Distill `messages` into a brief running summary suitable for prepending to
/// future turns as a system message. If `prior_summary` is given, the model
/// is asked to extend it with new information rather than start fresh —
/// preserves earlier context when called repeatedly.
///
/// Caps `max_tokens` at 800 (summaries should be much shorter than the input)
/// and uses temperature 0.0 for determinism.
pub async fn summarize_conversation(
    model: &dyn ChatModel,
    messages: &[Message],
    prior_summary: Option<&str>,
) -> Result<String> {
    if messages.is_empty() {
        return Ok(prior_summary.unwrap_or("").to_string());
    }
    let transcript = messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            format!("{role}: {}", m.text_content())
        })
        .collect::<Vec<_>>()
        .join("\n");

    let user_prompt = if let Some(prior) = prior_summary.filter(|s| !s.is_empty()) {
        format!(
            "Previous summary:\n{prior}\n\n\
             New conversation turns to incorporate:\n{transcript}\n\n\
             Update the summary so it reflects both the prior summary and the new turns. \
             Stay terse — preserve facts, decisions, and unresolved questions. Drop pleasantries."
        )
    } else {
        format!(
            "Conversation transcript:\n{transcript}\n\n\
             Distill into a brief summary that preserves facts, decisions, and unresolved \
             questions so a future assistant can pick up the thread. Stay terse — drop pleasantries."
        )
    };

    let req = vec![
        Message::system(
            "You are a meticulous note-taker. Produce concise factual summaries — no filler, \
             no hedging, no apology. Preserve names, numbers, decisions verbatim."
        ),
        Message::user(user_prompt),
    ];
    let opts = ChatOptions { temperature: Some(0.0), max_tokens: Some(800), ..Default::default() };
    let resp = model.invoke(req, &opts).await?;
    let summary = resp.message.text_content();
    if summary.is_empty() {
        return Err(Error::other("summarize_conversation: empty summary"));
    }
    Ok(summary)
}

impl ConversationMemory for BufferMemory {
    fn append(&mut self, m: Message) {
        // System messages set via append also count as the pin (LangChain
        // parity: dropping a system message into the conversation pins it).
        if matches!(m.role, crate::Role::System) {
            self.system = Some(m);
            return;
        }
        self.history.push_back(m);
        while self.history.len() > self.max_messages {
            self.history.pop_front();
        }
    }
    fn messages(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.history.len() + 1);
        if let Some(s) = &self.system { out.push(s.clone()); }
        out.extend(self.history.iter().cloned());
        out
    }
    fn clear(&mut self) {
        self.history.clear();
    }
    fn set_system(&mut self, m: Option<Message>) {
        self.system = m;
    }
}

/// Keep messages under a total token budget; evict oldest until the running
/// total is within budget. The system-pinned message is always included and
/// counted — but never evicted (so the budget must be larger than its size,
/// otherwise nothing else fits).
pub struct TokenBufferMemory {
    pub max_tokens: usize,
    counter: TokenCounter,
    system: Option<Message>,
    history: std::collections::VecDeque<Message>,
}

impl TokenBufferMemory {
    pub fn new(max_tokens: usize, counter: TokenCounter) -> Self {
        Self {
            max_tokens,
            counter,
            system: None,
            history: std::collections::VecDeque::new(),
        }
    }

    fn total_tokens(&self) -> usize {
        let sys = self.system.as_ref().map(|m| (self.counter)(m)).unwrap_or(0);
        let hist: usize = self.history.iter().map(|m| (self.counter)(m)).sum();
        sys + hist
    }
}

impl TokenBufferMemory {
    pub fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            version: MemorySnapshot::CURRENT_VERSION,
            system: self.system.clone(),
            history: self.history.iter().cloned().collect(),
        }
    }

    /// Restore from a snapshot. The `max_tokens` budget on `self` is enforced
    /// via the same eviction loop `append` uses; if the snapshot exceeds the
    /// budget, oldest history is evicted until it fits.
    pub fn restore(&mut self, snap: MemorySnapshot) {
        self.system = snap.system;
        self.history.clear();
        for m in snap.history {
            self.history.push_back(m);
        }
        while self.total_tokens() > self.max_tokens && !self.history.is_empty() {
            self.history.pop_front();
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> { self.snapshot().to_bytes() }
    pub fn from_bytes(max_tokens: usize, counter: TokenCounter, bytes: &[u8]) -> Result<Self> {
        let snap = MemorySnapshot::from_bytes(bytes)?;
        let mut m = Self::new(max_tokens, counter);
        m.restore(snap);
        Ok(m)
    }
}

impl ConversationMemory for TokenBufferMemory {
    fn append(&mut self, m: Message) {
        if matches!(m.role, crate::Role::System) {
            self.system = Some(m);
            return;
        }
        self.history.push_back(m);
        // Evict from the front until under budget. Stop if history is empty
        // (system alone might already exceed budget — caller's problem).
        while self.total_tokens() > self.max_tokens && !self.history.is_empty() {
            self.history.pop_front();
        }
    }
    fn messages(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.history.len() + 1);
        if let Some(s) = &self.system { out.push(s.clone()); }
        out.extend(self.history.iter().cloned());
        out
    }
    fn clear(&mut self) {
        self.history.clear();
    }
    fn set_system(&mut self, m: Option<Message>) {
        self.system = m;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, Role};

    #[test]
    fn buffer_memory_keeps_last_n_messages() {
        let mut m = BufferMemory::new(3);
        for i in 0..5 {
            m.append(Message::user(format!("u{i}")));
        }
        let got = m.messages();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].text_content(), "u2");
        assert_eq!(got[2].text_content(), "u4");
    }

    #[test]
    fn buffer_memory_pins_system_message() {
        let mut m = BufferMemory::new(2);
        m.append(Message::system("you are helpful"));
        for i in 0..5 {
            m.append(Message::user(format!("u{i}")));
        }
        let got = m.messages();
        // system pinned + 2 most recent user messages.
        assert_eq!(got.len(), 3);
        assert!(matches!(got[0].role, Role::System));
        assert_eq!(got[0].text_content(), "you are helpful");
        assert_eq!(got[1].text_content(), "u3");
        assert_eq!(got[2].text_content(), "u4");
    }

    #[test]
    fn buffer_memory_clear_keeps_system_pin() {
        let mut m = BufferMemory::new(5);
        m.append(Message::system("sys"));
        m.append(Message::user("hi"));
        m.clear();
        let got = m.messages();
        assert_eq!(got.len(), 1);
        assert!(matches!(got[0].role, Role::System));
    }

    #[test]
    fn buffer_memory_set_system_overrides() {
        let mut m = BufferMemory::new(5);
        m.append(Message::system("v1"));
        m.set_system(Some(Message::system("v2")));
        assert_eq!(m.messages()[0].text_content(), "v2");
        m.set_system(None);
        assert!(m.messages().is_empty());
    }

    #[test]
    fn buffer_memory_max_messages_zero_clamps_to_one() {
        let mut m = BufferMemory::new(0);
        m.append(Message::user("a"));
        m.append(Message::user("b"));
        let got = m.messages();
        assert_eq!(got.len(), 1, "must keep at least one");
        assert_eq!(got[0].text_content(), "b");
    }

    #[test]
    fn token_buffer_evicts_oldest_when_over_budget() {
        // Counter: 1 token per character (deterministic, easy math).
        let counter: TokenCounter =
            std::sync::Arc::new(|m: &Message| m.text_content().chars().count());
        let mut m = TokenBufferMemory::new(10, counter);
        m.append(Message::user("aaaaa"));   // 5
        m.append(Message::user("bbb"));     // 8 total
        m.append(Message::user("ccc"));     // 11 → over 10, evict oldest "aaaaa"
        let got = m.messages();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].text_content(), "bbb");
        assert_eq!(got[1].text_content(), "ccc");
    }

    #[test]
    fn token_buffer_counts_system_against_budget_but_doesnt_evict_it() {
        let counter: TokenCounter =
            std::sync::Arc::new(|m: &Message| m.text_content().chars().count());
        let mut m = TokenBufferMemory::new(20, counter);
        m.append(Message::system("system_prompt_here")); // 18 tokens
        m.append(Message::user("hello world")); // 11 → total 29, must evict user
        let got = m.messages();
        // System pin survived; user evicted.
        assert_eq!(got.len(), 1);
        assert!(matches!(got[0].role, Role::System));
    }

    #[test]
    fn buffer_memory_serde_round_trip() {
        let mut m = BufferMemory::new(5);
        m.append(Message::system("you are helpful"));
        m.append(Message::user("hi"));
        m.append(Message::assistant("hello"));
        m.append(Message::user("again"));
        let bytes = m.to_bytes().unwrap();
        let m2 = BufferMemory::from_bytes(5, &bytes).unwrap();
        let got = m2.messages();
        assert_eq!(got.len(), 4);
        assert_eq!(got[0].text_content(), "you are helpful");
        assert_eq!(got[3].text_content(), "again");
    }

    #[test]
    fn buffer_memory_from_bytes_enforces_max_messages_cap() {
        // Snapshot has 10 messages; restore into a 3-message buffer → drops oldest 7.
        let mut big = BufferMemory::new(50);
        for i in 0..10 {
            big.append(Message::user(format!("u{i}")));
        }
        let bytes = big.to_bytes().unwrap();
        let small = BufferMemory::from_bytes(3, &bytes).unwrap();
        let got = small.messages();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].text_content(), "u7");
        assert_eq!(got[2].text_content(), "u9");
    }

    #[test]
    fn token_buffer_serde_round_trip_preserves_pin_and_history() {
        let counter: TokenCounter =
            std::sync::Arc::new(|m: &Message| m.text_content().chars().count());
        let mut m = TokenBufferMemory::new(100, counter.clone());
        m.append(Message::system("sys"));
        m.append(Message::user("hello"));
        m.append(Message::assistant("hi back"));
        let bytes = m.to_bytes().unwrap();
        let m2 = TokenBufferMemory::from_bytes(100, counter, &bytes).unwrap();
        let got = m2.messages();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].role, Role::System);
        assert_eq!(got[2].text_content(), "hi back");
    }

    #[test]
    fn from_bytes_rejects_wrong_version() {
        let snap = MemorySnapshot { version: 9999, system: None, history: vec![] };
        let bad_bytes = serde_json::to_vec(&snap).unwrap();
        let err = BufferMemory::from_bytes(10, &bad_bytes).unwrap_err();
        assert!(format!("{err}").contains("version mismatch"));
    }

    #[test]
    fn from_bytes_rejects_garbage() {
        assert!(BufferMemory::from_bytes(10, b"not valid json").is_err());
    }

    /// Inline fake ChatModel for the summarization tests — capture the prompt
    /// the helper builds, and return a canned summary.
    struct FakeSummarizer {
        captured_prompt: std::sync::Arc<std::sync::Mutex<Option<String>>>,
        canned_summary: String,
    }

    #[async_trait::async_trait]
    impl crate::ChatModel for FakeSummarizer {
        fn name(&self) -> &str { "fake-summarizer" }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &crate::ChatOptions,
        ) -> Result<crate::ChatResponse> {
            // Capture the user prompt (last message) so tests can inspect what
            // the helper actually asked the model.
            *self.captured_prompt.lock().unwrap() =
                messages.last().map(|m| m.text_content());
            Ok(crate::ChatResponse {
                message: Message::assistant(self.canned_summary.clone()),
                finish_reason: crate::FinishReason::Stop,
                usage: crate::TokenUsage::default(),
                model: "fake-summarizer".into(),
            })
        }
        async fn stream(
            &self,
            _messages: Vec<Message>,
            _opts: &crate::ChatOptions,
        ) -> Result<crate::model::ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn summarize_conversation_builds_transcript_and_returns_summary() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let m = FakeSummarizer { captured_prompt: captured.clone(), canned_summary: "user asked about X; agent answered Y".into() };
        let msgs = vec![
            Message::user("what is X?"),
            Message::assistant("X is the answer Y"),
        ];
        let summary = summarize_conversation(&m, &msgs, None).await.unwrap();
        assert_eq!(summary, "user asked about X; agent answered Y");
        let prompt = captured.lock().unwrap().clone().unwrap();
        // Transcript appears verbatim with role: prefixes.
        assert!(prompt.contains("user: what is X?"));
        assert!(prompt.contains("assistant: X is the answer Y"));
        // No prior summary section when none was passed.
        assert!(!prompt.contains("Previous summary:"));
    }

    #[tokio::test]
    async fn summarize_conversation_extends_prior_summary() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let m = FakeSummarizer { captured_prompt: captured.clone(), canned_summary: "extended".into() };
        let msgs = vec![Message::user("more")];
        let _ = summarize_conversation(&m, &msgs, Some("earlier facts")).await.unwrap();
        let prompt = captured.lock().unwrap().clone().unwrap();
        assert!(prompt.contains("Previous summary:\nearlier facts"));
        assert!(prompt.contains("New conversation turns to incorporate:"));
    }

    #[tokio::test]
    async fn summarize_conversation_empty_messages_returns_prior() {
        // No messages → just hand back whatever prior summary you had.
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let m = FakeSummarizer { captured_prompt: captured.clone(), canned_summary: "ignored".into() };
        let out = summarize_conversation(&m, &[], Some("kept")).await.unwrap();
        assert_eq!(out, "kept");
        // Model was NOT called.
        assert!(captured.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn buffer_memory_summarize_and_compact_replaces_old_turns() {
        let m = FakeSummarizer {
            captured_prompt: std::sync::Arc::new(std::sync::Mutex::new(None)),
            canned_summary: "they greeted each other".into(),
        };
        let mut mem = BufferMemory::new(10);
        mem.append(Message::user("hi"));
        mem.append(Message::assistant("hello"));
        mem.append(Message::user("how are you"));
        mem.append(Message::assistant("good thanks"));
        mem.append(Message::user("tell me about Rust"));

        // Compact the first 4 messages → leaves the 5th + a synthetic system pin.
        mem.summarize_and_compact(&m, 4).await.unwrap();
        let got = mem.messages();
        assert_eq!(got.len(), 2, "system pin + remaining user message");
        assert_eq!(got[0].role, Role::System);
        assert_eq!(got[0].text_content(), "they greeted each other");
        assert_eq!(got[1].text_content(), "tell me about Rust");
    }

    #[tokio::test]
    async fn summarize_and_compact_is_noop_when_history_too_short() {
        let m = FakeSummarizer {
            captured_prompt: std::sync::Arc::new(std::sync::Mutex::new(None)),
            canned_summary: "should not run".into(),
        };
        let mut mem = BufferMemory::new(10);
        mem.append(Message::user("only one"));
        mem.summarize_and_compact(&m, 5).await.unwrap();
        // No change.
        let got = mem.messages();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].text_content(), "only one");
    }
}
