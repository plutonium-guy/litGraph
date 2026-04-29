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

/// Rolling buffer with an LLM-distilled running summary of evicted turns.
/// LangChain parity: `ConversationSummaryBufferMemory`. The difference from
/// `BufferMemory` is that evicted messages don't vanish — they get folded
/// into a running summary preserved as a synthetic pre-system message, so
/// long-horizon context survives even when recent-buffer caps kick in.
///
/// # Flow
///
/// 1. Caller appends messages as usual (sync — `append()` never blocks).
/// 2. Before invoking the model, call `compact(model).await?` if needed.
///    If `len(buffer) > max_recent`, the oldest `summarize_chunk` messages
///    are drained + summarized via `summarize_conversation`, and the summary
///    is folded into `running_summary` (extended on each compaction).
/// 3. `messages()` returns `[system_pin?, synthetic_summary_msg?, ...recent]`.
///
/// Summarization is deliberately decoupled from `append` so the sync
/// `ConversationMemory` trait still fits. If the caller skips `compact`
/// the buffer just grows; this is predictable and debuggable.
pub struct SummaryBufferMemory {
    pub max_recent_messages: usize,
    pub summarize_chunk: usize,
    system: Option<Message>,
    running_summary: Option<String>,
    history: std::collections::VecDeque<Message>,
}

impl SummaryBufferMemory {
    /// `max_recent_messages`: soft cap on how many recent messages to keep
    /// verbatim. `summarize_chunk`: how many of the oldest messages to drain
    /// + summarize per `compact()` call. `summarize_chunk` must be ≥ 1; a
    /// typical choice is `max_recent / 2` (amortize LLM cost — compact only
    /// when over budget, and summarize enough to drop well under it).
    pub fn new(max_recent_messages: usize, summarize_chunk: usize) -> Self {
        Self {
            max_recent_messages: max_recent_messages.max(1),
            summarize_chunk: summarize_chunk.max(1),
            system: None,
            running_summary: None,
            history: std::collections::VecDeque::new(),
        }
    }

    /// Current running summary, if any. `None` until `compact()` runs at
    /// least once. Snapshot-friendly — the caller can persist it.
    pub fn running_summary(&self) -> Option<&str> {
        self.running_summary.as_deref()
    }

    /// Current recent buffer size (excludes system pin + summary).
    pub fn recent_len(&self) -> usize {
        self.history.len()
    }

    /// True if `compact()` would evict + summarize on the next call.
    pub fn needs_compact(&self) -> bool {
        self.history.len() > self.max_recent_messages
    }

    /// Drain the oldest `summarize_chunk` messages and fold them into the
    /// running summary. No-op when the buffer is at or under the cap.
    /// Returns `Ok(true)` if compaction ran, `Ok(false)` if no-op.
    pub async fn compact(&mut self, model: &dyn ChatModel) -> Result<bool> {
        if !self.needs_compact() {
            return Ok(false);
        }
        let to_drain = self.summarize_chunk.min(self.history.len());
        let drained: Vec<Message> = self.history.drain(..to_drain).collect();
        let prior = self.running_summary.as_deref();
        let new_summary = summarize_conversation(model, &drained, prior).await?;
        self.running_summary = Some(new_summary);
        Ok(true)
    }

    /// Force a full compact — summarize ALL buffered messages into the
    /// running summary, leaving the recent buffer empty. Useful at session
    /// boundaries or before persisting memory to a cold store.
    pub async fn compact_all(&mut self, model: &dyn ChatModel) -> Result<bool> {
        if self.history.is_empty() {
            return Ok(false);
        }
        let drained: Vec<Message> = self.history.drain(..).collect();
        let prior = self.running_summary.as_deref();
        let new_summary = summarize_conversation(model, &drained, prior).await?;
        self.running_summary = Some(new_summary);
        Ok(true)
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        // Pack the running summary into the system field as a synthetic
        // system message (prefixed) so round-trips through
        // `MemorySnapshot::{to_bytes, from_bytes}` preserve it. Callers
        // using `restore` on a fresh instance get it back.
        let system = match (&self.system, &self.running_summary) {
            (Some(s), Some(sum)) => Some(Message::system(format!(
                "{}\n\n[Conversation summary so far]\n{sum}",
                s.text_content()
            ))),
            (Some(s), None) => Some(s.clone()),
            (None, Some(sum)) => Some(Message::system(format!(
                "[Conversation summary so far]\n{sum}"
            ))),
            (None, None) => None,
        };
        MemorySnapshot {
            version: MemorySnapshot::CURRENT_VERSION,
            system,
            history: self.history.iter().cloned().collect(),
        }
    }

    pub fn restore(&mut self, snap: MemorySnapshot) {
        // Unpack: if the system message has our marker, split the summary
        // back out; else treat it as a plain pin.
        const MARKER: &str = "\n\n[Conversation summary so far]\n";
        self.system = None;
        self.running_summary = None;
        if let Some(sys) = snap.system {
            let text = sys.text_content();
            if let Some(idx) = text.find(MARKER) {
                let (head, tail) = text.split_at(idx);
                let sum = &tail[MARKER.len()..];
                if !head.is_empty() {
                    self.system = Some(Message::system(head));
                }
                if !sum.is_empty() {
                    self.running_summary = Some(sum.to_string());
                }
            } else if let Some(inner) = text.strip_prefix("[Conversation summary so far]\n") {
                self.running_summary = Some(inner.to_string());
            } else {
                self.system = Some(sys);
            }
        }
        self.history.clear();
        for m in snap.history {
            self.history.push_back(m);
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> { self.snapshot().to_bytes() }
    pub fn from_bytes(
        max_recent: usize,
        summarize_chunk: usize,
        bytes: &[u8],
    ) -> Result<Self> {
        let snap = MemorySnapshot::from_bytes(bytes)?;
        let mut m = Self::new(max_recent, summarize_chunk);
        m.restore(snap);
        Ok(m)
    }
}

impl ConversationMemory for SummaryBufferMemory {
    fn append(&mut self, m: Message) {
        if matches!(m.role, crate::Role::System) {
            self.system = Some(m);
            return;
        }
        self.history.push_back(m);
        // Do NOT summarize here — it's async. Caller must `compact().await`.
    }
    fn messages(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.history.len() + 2);
        if let Some(s) = &self.system {
            out.push(s.clone());
        }
        if let Some(sum) = &self.running_summary {
            out.push(Message::system(format!(
                "Summary of earlier turns:\n{sum}"
            )));
        }
        out.extend(self.history.iter().cloned());
        out
    }
    fn clear(&mut self) {
        self.history.clear();
        self.running_summary = None;
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

    // --- SummaryBufferMemory tests ---

    #[tokio::test]
    async fn summary_buffer_no_compact_when_under_cap() {
        let m = FakeSummarizer {
            captured_prompt: std::sync::Arc::new(std::sync::Mutex::new(None)),
            canned_summary: "should not run".into(),
        };
        let mut mem = SummaryBufferMemory::new(10, 4);
        mem.append(Message::user("hi"));
        assert!(!mem.needs_compact());
        let ran = mem.compact(&m).await.unwrap();
        assert!(!ran, "no-op under cap");
        assert!(mem.running_summary().is_none());
    }

    #[tokio::test]
    async fn summary_buffer_compacts_oldest_chunk() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let m = FakeSummarizer {
            captured_prompt: captured.clone(),
            canned_summary: "greetings exchanged".into(),
        };
        let mut mem = SummaryBufferMemory::new(3, 2);
        mem.append(Message::user("hi"));
        mem.append(Message::assistant("hello"));
        mem.append(Message::user("how are you"));
        mem.append(Message::assistant("good"));  // now 4 > cap 3 → compact
        assert!(mem.needs_compact());

        let ran = mem.compact(&m).await.unwrap();
        assert!(ran);
        // Oldest 2 drained + summarized; 2 remaining.
        assert_eq!(mem.recent_len(), 2);
        assert_eq!(mem.running_summary(), Some("greetings exchanged"));

        // messages() surfaces summary as a synthetic system message.
        let got = mem.messages();
        assert_eq!(got.len(), 3, "summary + 2 recent");
        assert_eq!(got[0].role, Role::System);
        assert!(got[0].text_content().contains("greetings exchanged"));
        assert_eq!(got[1].text_content(), "how are you");
        assert_eq!(got[2].text_content(), "good");
    }

    #[tokio::test]
    async fn summary_buffer_extends_prior_summary_across_compactions() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let m = FakeSummarizer {
            captured_prompt: captured.clone(),
            canned_summary: "second-summary".into(),
        };
        let mut mem = SummaryBufferMemory::new(2, 2);
        mem.append(Message::user("m1"));
        mem.append(Message::user("m2"));
        mem.append(Message::user("m3"));  // trigger first compact
        mem.compact(&m).await.unwrap();
        // Fake always returns "second-summary" — overwrite in subsequent compactions.
        mem.append(Message::user("m4"));
        mem.append(Message::user("m5"));  // trigger second compact
        mem.compact(&m).await.unwrap();

        // Second compact's prompt should reference the prior summary.
        let prompt = captured.lock().unwrap().clone().unwrap();
        assert!(prompt.contains("Previous summary:"),
                "second compact must pass prior summary to the LLM");
    }

    #[tokio::test]
    async fn summary_buffer_messages_includes_system_pin_and_summary() {
        let m = FakeSummarizer {
            captured_prompt: std::sync::Arc::new(std::sync::Mutex::new(None)),
            canned_summary: "distilled".into(),
        };
        let mut mem = SummaryBufferMemory::new(1, 2);
        mem.set_system(Some(Message::system("you are helpful")));
        mem.append(Message::user("first"));
        mem.append(Message::user("second"));
        mem.append(Message::user("third"));
        mem.compact(&m).await.unwrap();
        let got = mem.messages();
        // [system_pin, summary, ...recent]
        assert_eq!(got[0].role, Role::System);
        assert_eq!(got[0].text_content(), "you are helpful");
        assert_eq!(got[1].role, Role::System);
        assert!(got[1].text_content().contains("distilled"));
    }

    #[tokio::test]
    async fn summary_buffer_compact_all_drains_everything() {
        let m = FakeSummarizer {
            captured_prompt: std::sync::Arc::new(std::sync::Mutex::new(None)),
            canned_summary: "everything summarized".into(),
        };
        let mut mem = SummaryBufferMemory::new(100, 10);
        mem.append(Message::user("a"));
        mem.append(Message::user("b"));
        mem.append(Message::user("c"));
        let ran = mem.compact_all(&m).await.unwrap();
        assert!(ran);
        assert_eq!(mem.recent_len(), 0);
        assert_eq!(mem.running_summary(), Some("everything summarized"));
    }

    #[tokio::test]
    async fn summary_buffer_clear_removes_summary_too() {
        let m = FakeSummarizer {
            captured_prompt: std::sync::Arc::new(std::sync::Mutex::new(None)),
            canned_summary: "gone".into(),
        };
        let mut mem = SummaryBufferMemory::new(1, 2);
        mem.append(Message::user("x"));
        mem.append(Message::user("y"));
        mem.append(Message::user("z"));
        mem.compact(&m).await.unwrap();
        assert!(mem.running_summary().is_some());
        mem.clear();
        assert_eq!(mem.recent_len(), 0);
        assert!(mem.running_summary().is_none());
    }

    #[tokio::test]
    async fn summary_buffer_system_via_append_sets_pin() {
        let m = FakeSummarizer {
            captured_prompt: std::sync::Arc::new(std::sync::Mutex::new(None)),
            canned_summary: "_".into(),
        };
        let mut mem = SummaryBufferMemory::new(10, 2);
        mem.append(Message::system("pin via append"));
        mem.append(Message::user("x"));
        let got = mem.messages();
        assert_eq!(got[0].role, Role::System);
        assert_eq!(got[0].text_content(), "pin via append");
        let _ = m;
    }

    #[tokio::test]
    async fn summary_buffer_snapshot_roundtrip_preserves_summary() {
        let m = FakeSummarizer {
            captured_prompt: std::sync::Arc::new(std::sync::Mutex::new(None)),
            canned_summary: "state of the world".into(),
        };
        let mut mem = SummaryBufferMemory::new(1, 2);
        mem.set_system(Some(Message::system("op prompt")));
        mem.append(Message::user("a"));
        mem.append(Message::user("b"));
        mem.append(Message::user("c"));
        mem.compact(&m).await.unwrap();
        let bytes = mem.to_bytes().unwrap();
        let restored = SummaryBufferMemory::from_bytes(1, 2, &bytes).unwrap();
        assert_eq!(restored.running_summary(), Some("state of the world"));
        let got = restored.messages();
        assert!(got.iter().any(|m| m.text_content() == "op prompt"));
        assert!(got.iter().any(|m| m.text_content().contains("state of the world")));
    }

    #[tokio::test]
    async fn summary_buffer_compact_chunk_capped_at_history_len() {
        let m = FakeSummarizer {
            captured_prompt: std::sync::Arc::new(std::sync::Mutex::new(None)),
            canned_summary: "single".into(),
        };
        // Chunk is 10, buffer has only 2 over cap.
        let mut mem = SummaryBufferMemory::new(1, 10);
        mem.append(Message::user("a"));
        mem.append(Message::user("b"));
        mem.compact(&m).await.unwrap();
        assert_eq!(mem.recent_len(), 0, "chunk clamps to history len");
        assert_eq!(mem.running_summary(), Some("single"));
    }

    #[tokio::test]
    async fn summary_buffer_construction_clamps_zero_inputs() {
        // `new(0, 0)` should still produce a sane memory — clamp to 1.
        let mut mem = SummaryBufferMemory::new(0, 0);
        assert_eq!(mem.max_recent_messages, 1);
        assert_eq!(mem.summarize_chunk, 1);
        mem.append(Message::user("x"));
        mem.append(Message::user("y"));
        assert!(mem.needs_compact());
    }
}
