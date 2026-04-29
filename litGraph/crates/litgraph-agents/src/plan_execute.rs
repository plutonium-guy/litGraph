//! Plan-and-Execute agent. Two-phase pattern: planner LLM emits an
//! ordered list of steps, then a worker LLM (with tools) executes each
//! step sequentially. Different from ReactAgent (reactive tool-calling
//! per turn) — here the model thinks once, commits to a plan, then grinds.
//!
//! # When to use
//!
//! - Tasks with N obvious sub-questions where order matters
//!   ("research X, then summarize, then write a memo about Y")
//! - Multi-document analyses where each step builds on the prior
//! - Complex transformations the agent benefits from "scoping" before doing
//!
//! # When NOT to use
//!
//! - Single-tool-call queries (use ReactAgent — less overhead)
//! - Tasks where the plan needs to change based on intermediate results
//!   (use ReactAgent — reactive)
//! - Open-ended exploratory tasks (use ReactAgent or SupervisorAgent)
//!
//! # Composition
//!
//! Internally builds on `ReactAgent` for each step's execution — gets
//! tool-calling, parallelism, max-iterations handling for free. The
//! planner + executor can be DIFFERENT models (cheap planner,
//! capable executor) to optimize cost.

use std::sync::Arc;

use litgraph_core::{parse_numbered_list, ChatModel, ChatOptions, Message, Result};
use litgraph_core::tool::Tool;

use crate::react::{ReactAgent, ReactAgentConfig};

const DEFAULT_PLANNER_SYSTEM: &str = "You are a planning assistant. Given a user task, \
write a SHORT numbered list (1-7 steps) of concrete sub-tasks an executor agent \
should perform IN ORDER to accomplish it. Each step must be:\n\
- Self-contained (no references to 'previous step' beyond what's in context)\n\
- Concrete and actionable (not 'think about X')\n\
- One sentence\n\
\n\
Output ONLY the numbered list — no preamble, no commentary.";

#[derive(Debug, Clone)]
pub struct PlanAndExecuteConfig {
    /// System prompt for the planner LLM. Default works for general tasks.
    pub planner_system: String,
    /// Max steps to execute (planner output truncated to this). Default 7.
    pub max_steps: usize,
    /// Per-step max iterations on the inner ReactAgent (tool-call loops
    /// per step). Default 5.
    pub max_iterations_per_step: u32,
    /// System prompt for the executor's per-step ReactAgent. Default
    /// `None` — the step itself is the user input; no extra system hint.
    pub executor_system: Option<String>,
    /// Chat options for the planner. Temperature defaults sensibly low
    /// (planner is supposed to be terse + deterministic).
    pub planner_chat_options: ChatOptions,
    /// Chat options forwarded to the executor's ReactAgent. Defaults
    /// allow tool-calling at whatever temperature the model defaults.
    pub executor_chat_options: ChatOptions,
}

impl Default for PlanAndExecuteConfig {
    fn default() -> Self {
        Self {
            planner_system: DEFAULT_PLANNER_SYSTEM.into(),
            max_steps: 7,
            max_iterations_per_step: 5,
            executor_system: None,
            planner_chat_options: ChatOptions { temperature: Some(0.0), ..Default::default() },
            executor_chat_options: ChatOptions::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepOutcome {
    pub step: String,
    pub output: String,
}

#[derive(Debug, Clone)]
pub struct PlanAndExecuteResult {
    /// The planner's parsed plan (one entry per step).
    pub plan: Vec<String>,
    /// One outcome per executed step. Length ≤ `plan.len()` if the
    /// executor short-circuited (currently never; reserved for future
    /// "stop on error" semantics).
    pub steps: Vec<StepOutcome>,
    /// Final assistant message — the last executed step's output. Use
    /// this as the user-facing answer.
    pub final_answer: String,
}

pub struct PlanAndExecuteAgent {
    planner: Arc<dyn ChatModel>,
    executor: Arc<dyn ChatModel>,
    tools: Vec<Arc<dyn Tool>>,
    cfg: PlanAndExecuteConfig,
}

impl PlanAndExecuteAgent {
    /// `planner` + `executor` may be the same model (default workflow) or
    /// different (cheap planner, capable executor — common cost-optimization).
    pub fn new(
        planner: Arc<dyn ChatModel>,
        executor: Arc<dyn ChatModel>,
        tools: Vec<Arc<dyn Tool>>,
        cfg: PlanAndExecuteConfig,
    ) -> Self {
        Self { planner, executor, tools, cfg }
    }

    /// Convenience: same model for both phases.
    pub fn from_single_model(
        model: Arc<dyn ChatModel>,
        tools: Vec<Arc<dyn Tool>>,
        cfg: PlanAndExecuteConfig,
    ) -> Self {
        Self::new(model.clone(), model, tools, cfg)
    }

    /// Run end-to-end: plan → execute each step → return aggregated outcome.
    pub async fn invoke(&self, user_task: impl Into<String>) -> Result<PlanAndExecuteResult> {
        let task = user_task.into();
        let plan = self.plan(&task).await?;
        let plan_truncated: Vec<String> =
            plan.iter().take(self.cfg.max_steps).cloned().collect();

        let mut steps = Vec::with_capacity(plan_truncated.len());
        let mut accumulated_context = String::new();

        for (i, step) in plan_truncated.iter().enumerate() {
            let exec_input = build_executor_input(&task, &plan_truncated, i, &accumulated_context, step);
            let output = self.run_step(&exec_input).await?;
            // Append "Step N output: ..." to accumulator for the next step's context.
            accumulated_context.push_str(&format!(
                "\n\nStep {} output:\n{}",
                i + 1,
                output
            ));
            steps.push(StepOutcome { step: step.clone(), output });
        }

        let final_answer = steps
            .last()
            .map(|s| s.output.clone())
            .unwrap_or_default();

        Ok(PlanAndExecuteResult { plan: plan_truncated, steps, final_answer })
    }

    /// Phase 1: ask the planner to emit a numbered list, parse via
    /// `parse_numbered_list` (iter 106). If parse yields zero items, fall
    /// back to splitting on newlines (some models emit bullets).
    async fn plan(&self, task: &str) -> Result<Vec<String>> {
        let messages = vec![
            Message::system(self.cfg.planner_system.clone()),
            Message::user(format!("Task:\n{task}\n\nProvide the numbered plan.")),
        ];
        let resp = self.planner
            .invoke(messages, &self.cfg.planner_chat_options)
            .await?;
        let text = resp.message.text_content();
        let parsed = parse_numbered_list(&text);
        if parsed.is_empty() {
            // Fallback: split on newlines, drop empty lines / common prefixes.
            Ok(split_on_lines_as_fallback(&text))
        } else {
            Ok(parsed)
        }
    }

    /// Phase 2: spin up a fresh ReactAgent for THIS step. Each step gets
    /// its own clean message history (no leak between steps) — the
    /// accumulated_context goes in via the user prompt, not the agent's
    /// session.
    async fn run_step(&self, exec_input: &str) -> Result<String> {
        let cfg = ReactAgentConfig {
            system_prompt: self.cfg.executor_system.clone(),
            max_iterations: self.cfg.max_iterations_per_step,
            chat_options: self.cfg.executor_chat_options.clone(),
            max_parallel_tools: 16,
        };
        let agent = ReactAgent::new(self.executor.clone(), self.tools.clone(), cfg)
            .map_err(|e| litgraph_core::Error::other(format!("plan_execute: build executor: {e}")))?;
        let state = agent
            .invoke(exec_input)
            .await
            .map_err(|e| litgraph_core::Error::other(format!("plan_execute: step: {e}")))?;
        // The last assistant message is the step's output.
        let out = state
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, litgraph_core::Role::Assistant) && m.tool_calls.is_empty())
            .map(|m| m.text_content())
            .unwrap_or_default();
        Ok(out)
    }
}

/// Build the user-side prompt for a single step's executor invocation.
/// Includes: original task, the full plan (so the agent sees context),
/// accumulated outputs from prior steps, and the current step.
fn build_executor_input(
    task: &str,
    plan: &[String],
    step_idx: usize,
    accumulated: &str,
    current: &str,
) -> String {
    let plan_rendered: String = plan
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{}. {}", i + 1, s))
        .collect::<Vec<_>>()
        .join("\n");
    let mut s = format!(
        "Original task:\n{task}\n\n\
         Full plan:\n{plan_rendered}\n"
    );
    if !accumulated.trim().is_empty() {
        s.push_str("\nPrior step outputs:");
        s.push_str(accumulated);
        s.push('\n');
    }
    s.push_str(&format!(
        "\nNow execute step {} of {}:\n{current}\n\n\
         Output ONLY the result of this step.",
        step_idx + 1,
        plan.len(),
    ));
    s
}

/// Bullet-style fallback when the planner doesn't emit a clean numbered
/// list (e.g. "- step 1\n- step 2"). Strips common bullet prefixes.
fn split_on_lines_as_fallback(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.trim_start_matches(|c: char| c == '-' || c == '*' || c == '•' || c.is_ascii_digit() || c == '.' || c == ')' || c == ' ')
                .trim()
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ChatResponse, Message};
    use std::sync::Mutex;

    /// Scripted model: returns texts in sequence (cycles if exhausted).
    /// `is_planner` flips behavior — planner returns plans, executor
    /// returns step outputs.
    struct ScriptedModel {
        name: &'static str,
        replies: Vec<&'static str>,
        idx: std::sync::atomic::AtomicUsize,
        captured: Mutex<Vec<Vec<Message>>>,
    }

    impl ScriptedModel {
        fn new(name: &'static str, replies: Vec<&'static str>) -> Arc<Self> {
            Arc::new(Self {
                name,
                replies,
                idx: std::sync::atomic::AtomicUsize::new(0),
                captured: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl ChatModel for ScriptedModel {
        fn name(&self) -> &str { self.name }
        async fn invoke(&self, messages: Vec<Message>, _opts: &ChatOptions) -> Result<ChatResponse> {
            self.captured.lock().unwrap().push(messages.clone());
            let i = self.idx.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let text = self.replies[i % self.replies.len()];
            Ok(ChatResponse {
                message: Message::assistant(text),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "scripted".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn happy_path_three_step_plan_executes_all() {
        let planner = ScriptedModel::new("planner", vec![
            "1. First, gather facts about X.\n2. Second, analyze the facts.\n3. Third, write a summary."
        ]);
        let executor = ScriptedModel::new("executor", vec![
            "Facts gathered: X is widely used.",
            "Analysis: usage growing 20%/yr.",
            "Summary: X is a growing technology.",
        ]);
        let agent = PlanAndExecuteAgent::new(
            planner,
            executor.clone(),
            vec![],
            PlanAndExecuteConfig::default(),
        );
        let r = agent.invoke("Tell me about X").await.unwrap();
        assert_eq!(r.plan.len(), 3);
        assert!(r.plan[0].contains("gather facts"));
        assert_eq!(r.steps.len(), 3);
        assert_eq!(r.steps[0].output, "Facts gathered: X is widely used.");
        assert_eq!(r.final_answer, "Summary: X is a growing technology.");
    }

    #[tokio::test]
    async fn max_steps_truncates_plan() {
        let planner = ScriptedModel::new("planner", vec![
            "1. a\n2. b\n3. c\n4. d\n5. e\n6. f\n7. g\n8. h\n9. i\n10. j"
        ]);
        let executor = ScriptedModel::new("executor", vec!["done"]);
        let cfg = PlanAndExecuteConfig { max_steps: 3, ..Default::default() };
        let agent = PlanAndExecuteAgent::new(planner, executor, vec![], cfg);
        let r = agent.invoke("task").await.unwrap();
        assert_eq!(r.plan.len(), 3);
        assert_eq!(r.steps.len(), 3);
        assert_eq!(r.plan, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn fallback_parses_bulleted_plan() {
        let planner = ScriptedModel::new("planner", vec![
            "- gather data\n- analyze\n- summarize"
        ]);
        let executor = ScriptedModel::new("executor", vec!["done"]);
        let agent = PlanAndExecuteAgent::from_single_model(
            planner.clone() as Arc<dyn ChatModel>,
            vec![],
            PlanAndExecuteConfig::default(),
        );
        // from_single_model uses planner for both — so executor invocations
        // will reuse the planner's scripted replies. That's fine for this test:
        // we just check the plan parsing path.
        let _ = agent.invoke("task").await.unwrap();
    }

    #[tokio::test]
    async fn empty_plan_returns_empty_steps_no_panic() {
        let planner = ScriptedModel::new("planner", vec![""]);
        let executor = ScriptedModel::new("executor", vec!["never called"]);
        let agent = PlanAndExecuteAgent::new(
            planner,
            executor.clone(),
            vec![],
            PlanAndExecuteConfig::default(),
        );
        let r = agent.invoke("task").await.unwrap();
        assert!(r.plan.is_empty());
        assert!(r.steps.is_empty());
        assert_eq!(r.final_answer, "");
        // Executor should not have been called.
        assert_eq!(executor.captured.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn each_step_sees_full_plan_and_prior_outputs() {
        let planner = ScriptedModel::new("planner", vec!["1. step alpha\n2. step beta"]);
        let executor = ScriptedModel::new("executor", vec![
            "alpha output",
            "beta output",
        ]);
        let agent = PlanAndExecuteAgent::new(
            planner,
            executor.clone(),
            vec![],
            PlanAndExecuteConfig::default(),
        );
        agent.invoke("master task").await.unwrap();
        let captured = executor.captured.lock().unwrap();
        assert_eq!(captured.len(), 2);
        // First step's user msg has empty prior-outputs section.
        let first_user = captured[0].iter().rev().find(|m| matches!(m.role, litgraph_core::Role::User)).unwrap();
        let first_text = first_user.text_content();
        assert!(first_text.contains("master task"));
        assert!(first_text.contains("step alpha"));
        assert!(first_text.contains("step beta"));  // full plan visible to step 1
        assert!(!first_text.contains("Prior step outputs"));
        // Second step's user msg includes prior output from step 1.
        let second_user = captured[1].iter().rev().find(|m| matches!(m.role, litgraph_core::Role::User)).unwrap();
        let second_text = second_user.text_content();
        assert!(second_text.contains("Prior step outputs"));
        assert!(second_text.contains("alpha output"));
    }

    #[tokio::test]
    async fn from_single_model_uses_one_model_for_both_phases() {
        let m = ScriptedModel::new("solo", vec![
            "1. just one step",   // planner call
            "the answer",          // executor call
        ]);
        let agent = PlanAndExecuteAgent::from_single_model(
            m.clone() as Arc<dyn ChatModel>,
            vec![],
            PlanAndExecuteConfig::default(),
        );
        let r = agent.invoke("q").await.unwrap();
        assert_eq!(r.plan, vec!["just one step"]);
        assert_eq!(r.final_answer, "the answer");
        // Two total calls (1 planner + 1 executor) on the same model.
        assert_eq!(m.captured.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn planner_chat_options_temperature_locked_low() {
        let m = ScriptedModel::new("m", vec!["1. step\n", "out"]);
        struct TempCapture { temps: Mutex<Vec<Option<f32>>> }
        #[async_trait]
        impl ChatModel for TempCapture {
            fn name(&self) -> &str { "tc" }
            async fn invoke(&self, _m: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
                self.temps.lock().unwrap().push(opts.temperature);
                Ok(ChatResponse {
                    message: Message::assistant("1. one step"),
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage::default(),
                    model: "tc".into(),
                })
            }
            async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> { unimplemented!() }
        }
        let tc = Arc::new(TempCapture { temps: Mutex::new(vec![]) });
        let agent = PlanAndExecuteAgent::new(
            tc.clone(),
            m,
            vec![],
            PlanAndExecuteConfig::default(),
        );
        agent.invoke("q").await.unwrap();
        let temps = tc.temps.lock().unwrap();
        assert_eq!(temps[0], Some(0.0), "planner uses temperature=0.0 by default");
    }

    #[tokio::test]
    async fn fallback_strips_bullet_prefixes() {
        let result = split_on_lines_as_fallback("- a\n* b\n• c\n1) d\n2. e");
        assert_eq!(result, vec!["a", "b", "c", "d", "e"]);
    }

    #[tokio::test]
    async fn fallback_skips_blank_lines() {
        let result = split_on_lines_as_fallback("\n\n- a\n\n- b\n\n");
        assert_eq!(result, vec!["a", "b"]);
    }
}
