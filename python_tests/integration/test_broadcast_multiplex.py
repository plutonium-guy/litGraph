"""Live integration: `multiplex_chat_streams` (live) + broadcast (BLOCKED).

- `multiplex_chat_streams` fans IN N models' streams to one tagged
  iterator. We use the same DeepSeek model with two labels (no second
  provider key) to still see N>1 streams interleaved.
- `broadcast_chat_stream`: SKIPPED pending a fresh maturin build.
  The Python binding (`crates/litgraph-py/src/agents.rs`) was patched
  in iter 360 to enter the bridge runtime before calling
  `BroadcastHandle::subscribe()` (which lazily spawns the upstream pump
  via `tokio::spawn`). Before the patch the call panicked from
  synchronous Python:

      pyo3_runtime.PanicException: there is no reactor running, must
      be called from the context of a Tokio 1.x runtime

  Re-enable this test (remove the `@pytest.mark.skip`) after the next
  `maturin develop` / wheel rebuild picks up the binding change.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_multiplex_chat_streams_two_labels(deepseek_chat):
    from litgraph.agents import multiplex_chat_streams

    events = list(multiplex_chat_streams(
        [("ds-a", deepseek_chat), ("ds-b", deepseek_chat)],
        [{"role": "user", "content": "Reply: ok"}],
        max_tokens=10,
    ))
    assert events, "multiplex yielded no events"
    labels = {e["model_label"] for e in events}
    assert "ds-a" in labels
    assert "ds-b" in labels


def test_broadcast_chat_stream_two_subscribers(deepseek_chat):
    """Fixed in iter 360 (binding) + iter 376 (rebuilt). `subscribe()`
    now enters the bridge tokio runtime before spawning the upstream
    pump, so calling from sync Python no longer panics."""
    from litgraph.agents import broadcast_chat_stream

    handle = broadcast_chat_stream(
        deepseek_chat,
        [{"role": "user", "content": "Reply: hello"}],
        capacity=64,
        max_tokens=10,
    )
    sub_a = handle.subscribe()
    sub_b = handle.subscribe()

    a_events = list(sub_a)
    b_events = list(sub_b)
    assert a_events, "subscriber A got no events"
    assert b_events, "subscriber B got no events"
    # Both subscribers should see the same number of events (fan-out).
    assert len(a_events) == len(b_events), (
        f"a={len(a_events)} != b={len(b_events)}"
    )
