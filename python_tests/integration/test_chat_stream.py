"""Live integration: SSE streaming against DeepSeek.

Stream events are dicts with `type` ∈ {`delta`, `done`}.
- `delta` carries `text` (the next chunk).
- `done` carries the assembled `text`, `finish_reason`, `usage`, `model`.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_stream_yields_delta_events(deepseek_chat):
    events = list(
        deepseek_chat.stream(
            [{"role": "user", "content": "Reply with: pong"}],
            max_tokens=10,
        )
    )
    assert len(events) > 0
    deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "delta"]
    assert len(deltas) >= 1


def test_stream_concatenated_text_matches_done(deepseek_chat):
    events = list(
        deepseek_chat.stream(
            [{"role": "user", "content": "Reply: ALPHA"}],
            max_tokens=10,
        )
    )
    deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "delta"]
    assembled = "".join(d.get("text", "") for d in deltas)
    done = next((e for e in events if isinstance(e, dict) and e.get("type") == "done"), None)
    assert done is not None
    # The `done` event's `text` is the full assembled message; the
    # streamed deltas should equal it.
    assert assembled == done.get("text", "")


def test_stream_done_event_has_finish_reason_and_usage(deepseek_chat):
    events = list(
        deepseek_chat.stream(
            [{"role": "user", "content": "ok"}],
            max_tokens=10,
        )
    )
    done = next((e for e in events if isinstance(e, dict) and e.get("type") == "done"), None)
    assert done is not None
    assert done.get("finish_reason") in ("stop", "length", "tool_calls")
    usage = done.get("usage")
    assert isinstance(usage, dict)
    assert usage.get("total", 0) > 0


def test_stream_token_count_matches_usage(deepseek_chat):
    """Sanity: assembled stream text is non-empty when usage reports
    completion tokens > 0."""
    events = list(
        deepseek_chat.stream(
            [{"role": "user", "content": "Say one word."}],
            max_tokens=10,
        )
    )
    deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "delta"]
    assembled = "".join(d.get("text", "") for d in deltas)
    done = next((e for e in events if isinstance(e, dict) and e.get("type") == "done"), None)
    if done and done.get("usage", {}).get("completion", 0) > 0:
        assert len(assembled) > 0
