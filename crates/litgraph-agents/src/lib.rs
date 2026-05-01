//! Agent patterns built on `litgraph-graph`. The flagship is [`react`], a
//! tool-calling ReAct loop: the model proposes tool calls, the dispatcher runs them
//! concurrently (`JoinSet`), results feed back as `Role::Tool` messages until the
//! model stops.
//!
//! This is the 80% use case — native provider tool-calling is faster and more
//! reliable than any regex-parsed ReAct pattern. No regex. No text parsing.

pub mod react;
pub mod supervisor;
pub mod text_react;
pub mod plan_execute;
pub mod subagent;
pub mod map_reduce;
pub mod refine;
pub mod critique_revise;

pub use react::{AgentEvent, AgentEventStream, AgentState, ReactAgent, ReactAgentConfig};
pub use supervisor::{SupervisorAgent, SupervisorConfig};
pub use text_react::{
    StoppedReason, TextReActAgent, TextReactAgentConfig, TextReactEvent, TextReactEventStream,
    TextReactResult, TextReactTurn,
};
pub use plan_execute::{
    PlanAndExecuteAgent, PlanAndExecuteConfig, PlanAndExecuteResult, StepOutcome,
};
pub use subagent::SubagentTool;
pub use map_reduce::MapReduceSummarizer;
pub use refine::RefineSummarizer;
pub use critique_revise::{
    CritiqueReviseAgent, CritiqueReviseIteration, CritiqueReviseResult,
};
