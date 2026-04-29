//! litgraph-graph — typed StateGraph with parallel Kahn scheduler, conditional edges,
//! checkpointers, and human-in-the-loop interrupts.
//!
//! # Minimal example
//!
//! ```no_run
//! use litgraph_graph::{StateGraph, NodeOutput, Result, END, START};
//!
//! # async fn run() -> Result<()> {
//! #[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
//! struct State { n: u32 }
//!
//! let mut g = StateGraph::<State>::new();
//! g.add_node("inc", |s: State| async move { NodeOutput::update(State { n: s.n + 1 }) });
//! g.add_edge(START, "inc");
//! g.add_edge("inc", END);
//! let compiled = g.compile()?;
//! let final_state = compiled.invoke(State::default(), None).await?;
//! assert_eq!(final_state.n, 1);
//! # Ok(()) }
//! ```

pub mod state;
pub mod node;
pub mod graph;
pub mod scheduler;
pub mod checkpoint;
pub mod event;
pub mod interrupt;
pub mod visualize;

pub use state::{Reducer, merge_append, merge_replace};
pub use node::{NodeOutput, NodeFn};
pub use graph::{StateGraph, CompiledGraph, START, END};
pub use checkpoint::{
    state_history, Checkpoint, Checkpointer, MemoryCheckpointer, StateHistoryEntry,
};
pub use event::GraphEvent;
pub use interrupt::{Interrupt, Command};

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("unknown node `{0}`")]
    UnknownNode(String),

    #[error("cycle detected in graph")]
    Cycle,

    #[error("entry point not set")]
    NoEntry,

    #[error("node `{0}` panicked")]
    Panic(String),

    #[error("interrupted at node `{0}`")]
    Interrupted(String),

    #[error("cancelled")]
    Cancelled,

    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    #[error("serialization: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("bincode: {0}")]
    Bincode(#[from] bincode::Error),

    #[error("rmp-encode: {0}")]
    RmpEncode(#[from] rmp_serde::encode::Error),

    #[error("rmp-decode: {0}")]
    RmpDecode(#[from] rmp_serde::decode::Error),

    #[error(transparent)]
    Core(#[from] litgraph_core::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, GraphError>;
