"""Live integration: `streaming.stream_events` translator over a real stream.

Wraps `OpenAIChat.stream(...)` (which yields native dict events) into
a sync iterator of normalised `{event, name, run_id, data}` records
suitable for client-side consumption.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_stream_events_translator_yields_lcel_records(deepseek_chat):
    """`stream_events` emits LangChain-style `{event, name, run_id, data}`
    records. Today it surfaces lifecycle events (start / end), not the
    individual deltas — those are kept as native dicts on the underlying
    stream. Verify lifecycle events fire with stable keys + matching
    run_id."""
    from litgraph.streaming import stream_events

    raw = deepseek_chat.stream(
        [{"role": "user", "content": "Reply: hello"}],
        max_tokens=10,
    )
    records = list(stream_events(raw, name="test-run"))
    assert records, "translator yielded no records"
    for rec in records:
        assert isinstance(rec, dict)
        for key in ("event", "name", "run_id", "data"):
            assert key in rec, f"missing {key!r} in {rec!r}"
        assert rec["name"] == "test-run"
    run_ids = {r["run_id"] for r in records}
    assert len(run_ids) == 1, f"run_id should be stable across records: {run_ids!r}"
    events = [r["event"] for r in records]
    assert events[0] == "on_chat_model_start"
    assert events[-1] == "on_chat_model_end"
