"""TimeoutChat + TimeoutEmbeddings — per-invocation deadline wrappers.

`tokio::time::timeout` runs the inner future and a deadline timer
concurrently; whichever completes first wins. The inner future is
cancelled on timeout, releasing whatever resources it held (HTTP
connection, parse buffer, etc.)."""
import time

from litgraph.embeddings import (
    FunctionEmbeddings, TimeoutEmbeddings,
)
from litgraph.providers import OpenAIChat, TimeoutChat


def _slow_embedder(dim, delay_s):
    """Returns a FunctionEmbeddings that sleeps `delay_s` then emits a
    fixed vector. Lets us verify the deadline path with a deterministic
    inner."""
    def embed(texts):
        time.sleep(delay_s)
        return [[0.1] * dim for _ in texts]
    return FunctionEmbeddings(embed, dimensions=dim, name="slow")


def test_timeout_chat_constructs():
    raw = OpenAIChat(
        api_key="sk-fake",
        model="gpt-4o-mini",
        base_url="http://127.0.0.1:1",
    )
    chat = TimeoutChat(raw, timeout_ms=10)
    assert "TimeoutChat" in repr(chat)


def test_timeout_chat_propagates_inner_failure_within_deadline():
    """Inner points at unreachable host — connection error must surface
    as RuntimeError (NOT a timeout error, since refused connection
    fails fast under the deadline)."""
    raw = OpenAIChat(
        api_key="sk-fake",
        model="gpt-4o-mini",
        base_url="http://127.0.0.1:1",
    )
    chat = TimeoutChat(raw, timeout_ms=2000)
    try:
        chat.invoke([{"role": "user", "content": "hi"}])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


def test_timeout_chat_extractable_via_other_wrappers():
    """TimeoutChat must compose through `extract_chat_model`."""
    from litgraph.providers import TokenBudgetChat
    raw = OpenAIChat(
        api_key="sk-fake",
        model="gpt-4o-mini",
        base_url="http://127.0.0.1:1",
    )
    timed = TimeoutChat(raw, timeout_ms=1000)
    bud = TokenBudgetChat(timed, max_tokens=4096)
    assert bud is not None


def test_timeout_embed_under_deadline_passes_through():
    inner = _slow_embedder(dim=4, delay_s=0.01)
    emb = TimeoutEmbeddings(inner, timeout_ms=500)
    vec = emb.embed_query("hi")
    assert len(vec) == 4


def test_timeout_embed_constructs_with_extractable_inner():
    """Verify TimeoutEmbeddings can wrap any registered embedder.
    (The actual deadline-fires behaviour is covered by the Rust test
    suite — Python's `time.sleep` blocking inside `Python::with_gil`
    on a single-threaded runtime can stall the timer task, so a
    pure-Python end-to-end timeout test is unreliable.)"""
    inner = _slow_embedder(dim=4, delay_s=0)
    emb = TimeoutEmbeddings(inner, timeout_ms=100)
    assert "TimeoutEmbeddings" in repr(emb)


def test_timeout_embed_documents_passthrough_under_deadline():
    """Documents path returns successfully when inner is fast enough."""
    inner = _slow_embedder(dim=4, delay_s=0)
    emb = TimeoutEmbeddings(inner, timeout_ms=2000)
    out = emb.embed_documents(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 4 for v in out)


def test_timeout_embed_dimensions_property():
    inner = _slow_embedder(dim=7, delay_s=0)
    emb = TimeoutEmbeddings(inner, timeout_ms=100)
    assert emb.dimensions == 7


def test_timeout_embed_extractable_via_other_wrappers():
    from litgraph.embeddings import RetryingEmbeddings
    inner = _slow_embedder(dim=4, delay_s=0)
    timed = TimeoutEmbeddings(inner, timeout_ms=200)
    wrapped = RetryingEmbeddings(timed, max_retries=2)
    assert wrapped is not None


if __name__ == "__main__":
    fns = [
        test_timeout_chat_constructs,
        test_timeout_chat_propagates_inner_failure_within_deadline,
        test_timeout_chat_extractable_via_other_wrappers,
        test_timeout_embed_under_deadline_passes_through,
        test_timeout_embed_constructs_with_extractable_inner,
        test_timeout_embed_documents_passthrough_under_deadline,
        test_timeout_embed_dimensions_property,
        test_timeout_embed_extractable_via_other_wrappers,
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
