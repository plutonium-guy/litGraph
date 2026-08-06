//! Resilient wrappers for `ChatModel`. Wraps any provider with retry + jittered
//! exponential backoff via the `backon` crate.
//!
//! # What gets retried
//!
//! - `Error::RateLimited` (429) — retried, honors the upstream `retry_after_ms`
//!   when present.
//! - `Error::Timeout` — retried.
//! - `Error::Provider(s)` where `s` matches a 5xx status pattern — retried.
//!
//! Everything else (bad request, parse error, invalid input, tool failure,
//! cancellation) is treated as terminal and returned to the caller without
//! retries — replays would just waste tokens / propagate the bug.
//!
//! # Streaming
//!
//! `stream()` is NOT retried (token streams can't restart cleanly mid-stream).
//! For streaming retries, capture the failure at the consumer layer and re-call.

mod budget;
mod bulkhead;
mod cache;
mod cassette;
mod circuit_breaker;
mod fallback;
mod metrics;
mod pii;
mod race;
mod rate_limit;
mod retry;
mod self_consistency;
mod serialize;
mod singleflight;
mod timeout;

pub use budget::{CostCappedChatModel, TokenBudgetChatModel};
pub use bulkhead::{BulkheadChatModel, BulkheadEmbeddings, BulkheadMode};
pub use cache::{CachedChatModel, CachedEmbeddings, PromptCachingChatModel};
pub use cassette::{
    embed_documents_hash, embed_query_hash, exchange_hash, tool_args_hash, Cassette,
    CassetteExchange, EmbedCassette, EmbedExchange, RecordingChatModel, RecordingEmbeddings,
    RecordingTool, ReplayingChatModel, ReplayingEmbeddings, ReplayingTool, ToolCassette,
    ToolExchange,
};
pub use circuit_breaker::{CircuitBreakerChatModel, CircuitBreakerEmbeddings};
pub use fallback::{FallbackChatModel, FallbackEmbeddings};
pub use metrics::{MetricsChatModel, MetricsEmbeddings, MetricsTool, DEFAULT_LATENCY_BUCKETS_SECS};
pub use pii::PiiScrubbingChatModel;
pub use race::{HedgedChatModel, HedgedEmbeddings, RaceChatModel, RaceEmbeddings};
pub use rate_limit::{
    RateLimitConfig, RateLimitedChatModel, RateLimitedEmbeddings, SharedRateLimitedChatModel,
    SharedRateLimitedEmbeddings,
};
pub use retry::{RetryConfig, RetryingChatModel, RetryingEmbeddings};
pub use self_consistency::{
    default_text_voter, extracted_field_voter, ConsistencyVoter, SelfConsistencyChatModel,
};
pub use serialize::KeyedSerializedChatModel;
pub use singleflight::{SingleflightEmbeddings, SingleflightTool};
pub use timeout::{TimeoutChatModel, TimeoutEmbeddings};
