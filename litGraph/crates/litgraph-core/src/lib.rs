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
pub use model::{ChatModel, Embeddings, ChatOptions, ChatResponse, TokenUsage, FinishReason};
pub use prompt::{ChatPromptTemplate, FewShotChatPromptTemplate, PromptValue};
pub use tool::{Tool, ToolCall, ToolResult, ToolSchema};
pub use document::Document;
