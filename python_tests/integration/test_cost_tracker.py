"""Live integration: CostTracker accumulates real DeepSeek usage.

API: `model.instrument(tracker)` subscribes the model's event bus
to the tracker. Mutates the model in place; returns `None`. Calls
on the model after `instrument(...)` flow into the tracker.

Drain: the bus is async-flushed; tests wait briefly before reading
the snapshot to let pending events land.
"""
from __future__ import annotations

import time

import pytest


pytestmark = pytest.mark.integration


def _drain():
    """Give the async event-bus drain task a beat to land events."""
    time.sleep(0.5)


def test_cost_tracker_records_calls(deepseek_chat):
    from litgraph.observability import CostTracker

    # DeepSeek prices ($/1M tokens, May 2026 list — cache-miss + output rates).
    tracker = CostTracker({"deepseek-chat": (0.27, 1.10)})
    deepseek_chat.instrument(tracker)

    deepseek_chat.invoke([{"role": "user", "content": "Reply: hi"}], max_tokens=10)
    deepseek_chat.invoke([{"role": "user", "content": "Reply: bye"}], max_tokens=10)
    _drain()

    snap = tracker.snapshot()
    assert snap["calls"] >= 2
    assert snap["prompt_tokens"] > 0
    assert snap["completion_tokens"] > 0
    assert snap["usd"] >= 0.0


def test_cost_tracker_per_model_breakdown(deepseek_chat):
    from litgraph.observability import CostTracker

    tracker = CostTracker({"deepseek-chat": (0.27, 1.10)})
    deepseek_chat.instrument(tracker)
    deepseek_chat.invoke([{"role": "user", "content": "ok"}], max_tokens=10)
    _drain()

    snap = tracker.snapshot()
    per_model = snap.get("per_model", {})
    assert "deepseek-chat" in per_model
    assert per_model["deepseek-chat"]["calls"] >= 1


def test_cost_tracker_usd_helper(deepseek_chat):
    from litgraph.observability import CostTracker

    tracker = CostTracker({"deepseek-chat": (0.27, 1.10)})
    deepseek_chat.instrument(tracker)
    deepseek_chat.invoke([{"role": "user", "content": "ok"}], max_tokens=10)
    _drain()

    usd = tracker.usd()
    assert isinstance(usd, float)
    assert usd >= 0.0
