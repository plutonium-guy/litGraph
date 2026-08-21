//! Deployment selection and failover.
//!
//! Failover applies to infrastructure failures only. A client error is
//! identical at every deployment in the group, so retrying it burns quota
//! on all of them and returns the same error more slowly.
//!
//! Admission is delegated entirely to `CircuitBreaker::call` (see
//! `crates/litgraph-core/src/circuit_breaker.rs`). Candidates are NOT
//! pre-filtered on `Deployment::is_available()` — that snapshot reports
//! `Open` even after a breaker's cooldown has elapsed (the `Open ->
//! HalfOpenProbing` transition only happens inside `call`), so filtering on
//! it here would exclude a recovered deployment for the lifetime of the
//! process. `call` itself knows how to admit a half-open probe.

use std::sync::Arc;

use litgraph_core::circuit_breaker::CallError;
use litgraph_core::{ChatOptions, ChatResponse, Error, Message};
use thiserror::Error as ThisError;

use crate::registry::{Deployment, ModelGroup, RoutingStrategy};

#[derive(Debug, ThisError)]
pub enum DispatchError {
    /// A client-side error from upstream. Not retried, surfaced as-is.
    #[error("upstream rejected the request: {message}")]
    Upstream { message: String },
    /// Every deployment was open or failed. Surfaces as 503.
    #[error("no deployment in the group could serve the request")]
    AllDeploymentsUnavailable,
}

/// Does this error mean "try another deployment"?
///
/// Transport failures, timeouts and upstream 429s are infrastructure: they
/// are properties of THIS deployment and another may well succeed.
/// `InvalidInput` is the client's fault and identical at every deployment,
/// so retrying it burns quota everywhere and returns the same error more
/// slowly.
fn is_retryable(err: &Error) -> bool {
    matches!(err, Error::Provider(_) | Error::Timeout | Error::RateLimited { .. })
}

pub async fn dispatch_invoke(
    group: &ModelGroup,
    strategy: &dyn RoutingStrategy,
    messages: Vec<Message>,
    opts: &ChatOptions,
) -> Result<(ChatResponse, Arc<Deployment>), DispatchError> {
    // Do NOT filter on `is_available()` here — see module docs above. Every
    // deployment is a candidate; the breaker decides admission per attempt.
    let mut remaining: Vec<Arc<Deployment>> = group.deployments.clone();

    while !remaining.is_empty() {
        let Some(chosen) = strategy.pick(&remaining) else { break };
        remaining.retain(|d| d.id != chosen.id);

        let attempt_messages = messages.clone();
        match chosen
            .breaker()
            .call(|| async { chosen.model.invoke(attempt_messages, opts).await })
            .await
        {
            Ok(resp) => return Ok((resp, chosen)),
            // Breaker is open (or another caller holds the probe slot): this
            // deployment is not admitting traffic right now. Skip it — it is
            // not a failure of the request.
            Err(CallError::CircuitOpen) => continue,
            Err(CallError::Inner(e)) if is_retryable(&e) => {
                tracing::warn!(deployment = %chosen.id, error = %e, "deployment failed, failing over");
                continue;
            }
            Err(CallError::Inner(e)) => {
                return Err(DispatchError::Upstream { message: e.to_string() });
            }
        }
    }
    Err(DispatchError::AllDeploymentsUnavailable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Deployment, ModelGroup, WeightedRandom};
    use litgraph_core::{
        ChatModel, ChatOptions, ChatResponse, ChatStream, Error, FinishReason, Message, Result,
        Role, TokenUsage,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct Flaky {
        kind: &'static str,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ChatModel for Flaky {
        fn name(&self) -> &str {
            "flaky"
        }
        async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.kind {
                "ok" => Ok(ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content: vec![],
                        tool_calls: vec![],
                        tool_call_id: None,
                        name: None,
                        cache: false,
                    },
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage::default(),
                    model: "flaky".into(),
                }),
                "5xx" => Err(Error::provider("upstream 503 Service Unavailable")),
                "timeout" => Err(Error::Timeout),
                _ => Err(Error::invalid("400 Bad Request: unsupported parameter")),
            }
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unreachable!("this task covers non-streaming dispatch only")
        }
    }

    fn group_of(specs: &[(&str, &'static str, Arc<AtomicUsize>)]) -> ModelGroup {
        ModelGroup {
            name: "g".into(),
            deployments: specs
                .iter()
                .map(|(id, kind, calls)| {
                    Arc::new(Deployment::for_test(
                        id,
                        "g",
                        1,
                        Arc::new(Flaky { kind, calls: calls.clone() }),
                    ))
                })
                .collect(),
        }
    }

    #[tokio::test]
    async fn infrastructure_failure_fails_over_to_the_next_deployment() {
        let bad = Arc::new(AtomicUsize::new(0));
        let good = Arc::new(AtomicUsize::new(0));
        let g = group_of(&[("bad", "5xx", bad.clone()), ("good", "ok", good.clone())]);

        let (_resp, used) =
            dispatch_invoke(&g, &WeightedRandom::seeded(1), vec![], &ChatOptions::default())
                .await
                .expect("should succeed on the healthy deployment");
        assert_eq!(used.id, "good");
        assert_eq!(good.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn client_error_does_not_fail_over() {
        let a = Arc::new(AtomicUsize::new(0));
        let b = Arc::new(AtomicUsize::new(0));
        let g = group_of(&[("a", "4xx", a.clone()), ("b", "4xx", b.clone())]);

        let err = dispatch_invoke(&g, &WeightedRandom::seeded(1), vec![], &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, DispatchError::Upstream { .. }));
        // Exactly one deployment was tried — a 400 is the same everywhere.
        assert_eq!(a.load(Ordering::SeqCst) + b.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn all_deployments_failing_reports_exhausted() {
        let a = Arc::new(AtomicUsize::new(0));
        let b = Arc::new(AtomicUsize::new(0));
        let g = group_of(&[("a", "5xx", a.clone()), ("b", "5xx", b.clone())]);

        let err = dispatch_invoke(&g, &WeightedRandom::seeded(1), vec![], &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, DispatchError::AllDeploymentsUnavailable));
        assert_eq!(a.load(Ordering::SeqCst) + b.load(Ordering::SeqCst), 2, "both tried");
    }

    #[tokio::test]
    async fn open_breakers_are_skipped_before_dispatch() {
        let never = Arc::new(AtomicUsize::new(0));
        let g = group_of(&[("open", "ok", never.clone())]);
        g.deployments[0].trip_for_test();

        let err = dispatch_invoke(&g, &WeightedRandom::seeded(1), vec![], &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, DispatchError::AllDeploymentsUnavailable));
        assert_eq!(never.load(Ordering::SeqCst), 0, "open breaker must not be dispatched to");
    }

    // --- Added per task-5-ruling.md R3: Error::Timeout must fail over like a 5xx. ---
    //
    // Deployment order here is deliberately [good, bad] rather than [bad,
    // good]: with `WeightedRandom::seeded(1)` and two equal-weight
    // candidates, the first `pick` draws the *second* vec element (verified
    // empirically — the brief's own [bad, good] ordering, kept unmodified
    // above for `infrastructure_failure_fails_over_to_the_next_deployment`,
    // happens to succeed on the first pick and never actually exercises the
    // failover path, since it only asserts `good.load == 1`). Putting the
    // failing deployment second here makes it the one drawn first, so this
    // test genuinely exercises "timeout on attempt 1, success on attempt 2".
    #[tokio::test]
    async fn timeout_fails_over_to_the_next_deployment() {
        let good = Arc::new(AtomicUsize::new(0));
        let bad = Arc::new(AtomicUsize::new(0));
        let g = group_of(&[("good", "ok", good.clone()), ("bad", "timeout", bad.clone())]);

        let (_resp, used) =
            dispatch_invoke(&g, &WeightedRandom::seeded(1), vec![], &ChatOptions::default())
                .await
                .expect("timeout should fail over to the healthy deployment");
        assert_eq!(used.id, "good");
        assert_eq!(bad.load(Ordering::SeqCst), 1, "the timeout deployment must have been tried");
        assert_eq!(good.load(Ordering::SeqCst), 1);
    }

    // --- Added per task-5-ruling.md R12: this is the regression the ruling
    // exists to prevent. A breaker that trips must recover once its cooldown
    // elapses — `Deployment::breaker()` is public, and `CircuitBreaker::trip`
    // takes the cooldown as a parameter, so no new test constructor is
    // needed: we just trip with a cooldown short enough to wait out here. ---
    #[tokio::test]
    async fn deployment_recovers_once_cooldown_elapses() {
        let calls = Arc::new(AtomicUsize::new(0));
        let g = group_of(&[("rec", "ok", calls.clone())]);
        g.deployments[0].breaker().trip(Duration::from_millis(20));

        // While open, the deployment is skipped and the model is never invoked.
        let err = dispatch_invoke(&g, &WeightedRandom::seeded(1), vec![], &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, DispatchError::AllDeploymentsUnavailable));
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        tokio::time::sleep(Duration::from_millis(35)).await;

        // Cooldown elapsed: the same deployment must be admitted (as a
        // half-open probe) and serve the request.
        let (_resp, used) =
            dispatch_invoke(&g, &WeightedRandom::seeded(1), vec![], &ChatOptions::default())
                .await
                .expect("cooldown elapsed; deployment should be admitted again");
        assert_eq!(used.id, "rec");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
