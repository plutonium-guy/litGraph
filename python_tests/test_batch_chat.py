"""batch_chat — bounded-concurrency parallel ChatModel.invoke fan-out.

Test approach: build a tiny in-process FunctionEmbeddings-style chat
model wrapper isn't doable in the public API today, so these tests
exercise the binding via OpenAIChat configured against an unreachable
host — the *errors* shape verifies fail-fast vs per-call isolation
without needing a real LLM."""
from litgraph.agents import batch_chat
from litgraph.providers import OpenAIChat


def _broken_model():
    """Point at a port that nothing's listening on; every invoke fails
    fast with a connection error. That's exactly the deterministic
    error path we want for these tests."""
    return OpenAIChat(
        api_key="sk-not-a-real-key",
        model="gpt-4o-mini",
        base_url="http://127.0.0.1:1",
    )


def test_batch_chat_returns_one_result_per_input():
    model = _broken_model()
    inputs = [
        [{"role": "user", "content": "q1"}],
        [{"role": "user", "content": "q2"}],
        [{"role": "user", "content": "q3"}],
    ]
    out = batch_chat(model, inputs, max_concurrency=2)
    assert len(out) == 3


def test_batch_chat_per_call_errors_isolated():
    """Default (fail_fast=False): every input gets its own slot.
    Failed slots have an `error` key, successful slots have `text`."""
    model = _broken_model()
    inputs = [[{"role": "user", "content": f"q{i}"}] for i in range(3)]
    out = batch_chat(model, inputs, max_concurrency=4, fail_fast=False)
    # All three should have errors (broken host), but the *shape* is
    # the contract: a list of dicts with `error` keys, not a thrown
    # exception.
    assert len(out) == 3
    for r in out:
        assert "error" in r, f"expected error dict, got {r!r}"


def test_batch_chat_fail_fast_raises():
    """fail_fast=True must raise on the first failed invocation."""
    model = _broken_model()
    inputs = [[{"role": "user", "content": "q"}]]
    try:
        batch_chat(model, inputs, max_concurrency=2, fail_fast=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError under fail_fast")


def test_batch_chat_empty_inputs_returns_empty():
    model = _broken_model()
    out = batch_chat(model, [], max_concurrency=4)
    assert out == []


def test_batch_chat_rejects_non_list_input():
    model = _broken_model()
    try:
        batch_chat(model, ["not a list of messages"], max_concurrency=4)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("expected ValueError/TypeError on bad input shape")


if __name__ == "__main__":
    fns = [
        test_batch_chat_returns_one_result_per_input,
        test_batch_chat_per_call_errors_isolated,
        test_batch_chat_fail_fast_raises,
        test_batch_chat_empty_inputs_returns_empty,
        test_batch_chat_rejects_non_list_input,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)
