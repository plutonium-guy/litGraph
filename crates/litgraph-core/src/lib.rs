//! litgraph-core — shared types, traits, errors for litGraph.
//!
//! No PyO3. Usable as a pure Rust crate. Python bindings live in `litgraph-py`.

pub mod error;
pub mod message;
pub mod prompt;
pub mod model;
pub mod tool;
pub mod document;
pub mod memory;
pub mod structured;
pub mod xml_parser;
pub mod list_parser;
pub mod react_parser;
pub mod format_instructions;
pub mod evaluators;
pub mod output_fixer;
pub mod partial_json;
pub mod llm_judge;
pub mod pii;
pub mod example_selector;
pub mod eval_harness;
pub mod markdown_table_parser;
pub mod vector_store_memory;
pub mod store;
pub mod middleware;
pub mod skill_loader;
pub mod trajectory;
pub mod pairwise;
pub mod eval_synth;
pub mod tool_offload;
pub mod dataset_version;
pub mod prompt_hub;
pub mod langmem;
pub mod assistants;
pub mod batch;
pub mod embed_batch;
pub mod semantic_store;
pub mod stream_multiplex;
pub mod stream_broadcast;
pub mod progress;
pub mod resume_registry;
pub mod tool_dispatch;
pub mod table_parser;

pub use error::{Error, Result};
pub use structured::StructuredChatModel;
pub use xml_parser::{decode_entities as xml_decode_entities, parse_nested_xml, parse_xml_tags};
pub use list_parser::{
    parse_boolean, parse_comma_list, parse_markdown_list, parse_numbered_list,
};
pub use react_parser::{parse_react_step, ReactStep};
pub use format_instructions::{
    boolean_format_instructions, comma_list_format_instructions,
    markdown_list_format_instructions, numbered_list_format_instructions,
    react_format_instructions, xml_format_instructions,
};
pub use evaluators::{
    contains_all, contains_any, embedding_cosine, exact_match, exact_match_strict,
    jaccard_similarity, json_validity, levenshtein, levenshtein_ratio, regex_match,
};
pub use output_fixer::{fix_with_llm, parse_with_retry};
pub use partial_json::{parse_partial_json, repair_partial_json};
pub use llm_judge::{JudgeScore, LlmJudge};
pub use pii::{luhn_valid, PiiScrubber, Replacement, ScrubResult};
pub use example_selector::{LengthBasedExampleSelector, SemanticSimilarityExampleSelector};
pub use markdown_table_parser::{parse_markdown_tables, MarkdownTable};
pub use vector_store_memory::{RetrievedMessage, VectorStoreMemory};
pub use store::{InMemoryStore, Namespace, SearchFilter, Store, StoreItem};
pub use middleware::{
    AgentMiddleware, LoggingMiddleware, MessageWindowMiddleware, MiddlewareChain,
    MiddlewareChatModel, SystemPromptMiddleware,
};
pub use skill_loader::{load_agents_md, load_skills_dir, Skill, SystemPromptBuilder};
pub use trajectory::{evaluate_trajectory, TrajectoryPolicy, TrajectoryStep};
pub use pairwise::{PairwiseEvaluator, PairwiseResult, Winner};
pub use eval_synth::{parse_synth_response, synthesize_eval_cases};
pub use tool_offload::{
    default_offload_dir, is_offloaded_marker, resolve_handle, FilesystemOffloadBackend,
    InMemoryOffloadBackend, OffloadBackend, OffloadingTool, DEFAULT_PREVIEW_BYTES,
    DEFAULT_THRESHOLD_BYTES,
};
pub use dataset_version::{
    dataset_fingerprint, record_and_check, regression_check, DatasetManifest,
    InMemoryRunStore, JsonlRunStore, RegressionAlert, RunRecord, RunStore,
};
pub use prompt_hub::{CachingPromptHub, FilesystemPromptHub, PromptHub, PromptRef};
pub use langmem::{EpisodicMemory, Memory, MemoryExtractor, DEFAULT_EXTRACTION_SYSTEM_PROMPT};
pub use assistants::{Assistant, AssistantManager, AssistantPatch};
pub use batch::{
    batch_concurrent, batch_concurrent_fail_fast, batch_concurrent_stream,
    batch_concurrent_stream_with_progress, batch_concurrent_with_progress, BatchProgress,
    BatchStreamItem,
};
pub use embed_batch::{
    embed_documents_concurrent, embed_documents_concurrent_stream,
    embed_documents_concurrent_with_progress, EmbedProgress, EmbedStreamItem,
    DEFAULT_EMBED_CHUNK_SIZE, DEFAULT_EMBED_CONCURRENCY,
};
pub use semantic_store::{SemanticHit, SemanticStore};
pub use stream_multiplex::{multiplex_chat_streams, MultiplexEvent, MultiplexStream};
pub use stream_broadcast::{
    broadcast_chat_stream, broadcast_chat_stream_with_main, BroadcastEvent, BroadcastHandle,
    BroadcastSubscriberStream,
};
pub use progress::{Progress, ProgressObserver};
pub use resume_registry::{ResumeFuture, ResumeRegistry};
pub use tool_dispatch::{
    tool_dispatch_concurrent, tool_dispatch_concurrent_fail_fast,
    tool_dispatch_concurrent_stream, tool_dispatch_concurrent_with_progress,
    ToolDispatchProgress, ToolDispatchStreamItem,
};
pub use table_parser::{
    format_instructions as table_format_instructions, parse_table_csv, parse_table_json,
    parse_table_value, Table, TableQuery,
};
pub use eval_harness::{
    run_eval, AggregateScores, ContainsAllScorer, EvalCase, EvalCaseResult, EvalDataset,
    EvalReport, ExactMatchScorer, JaccardScorer, LevenshteinScorer, LlmJudgeScorer, RegexScorer,
    ScoreResult, Scorer,
};
pub use memory::{
    BufferMemory, ConversationMemory, MemorySnapshot, SummaryBufferMemory, TokenBufferMemory,
    TokenCounter, summarize_conversation,
};
pub use message::{Message, Role, ContentPart, ImageSource};
pub use model::{ChatModel, ChatOptions, ChatResponse, ChatStream, ChatStreamEvent, Embeddings, FinishReason, TokenUsage};
pub use prompt::{ChatPromptTemplate, FewShotChatPromptTemplate, PromptValue};
pub use tool::{Tool, ToolCall, ToolResult, ToolSchema};
pub use document::Document;
