"""Track real DeepSeek spend via `CostTracker`.

DeepSeek pricing (2026 list, $/1M tokens):
  Input cache miss: $0.27   Input cache hit: $0.07   Output: $1.10

`CostTracker({"model": (input_rate, output_rate)})` accumulates token
counts + computes USD. `model.instrument(tracker)` subscribes the
chat model's event bus to the tracker (mutates in place; returns
None — the *original* `model` reference now feeds the tracker).

Run:
    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_cost_tracker.py
"""
from __future__ import annotations

import os
import time

from litgraph.providers import OpenAIChat
from litgraph.observability import CostTracker


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY")

    chat = OpenAIChat(
        api_key=api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )
    tracker = CostTracker({"deepseek-chat": (0.27, 1.10)})
    chat.instrument(tracker)

    for q in ["Capital of France?", "Capital of Germany?", "Capital of Japan?"]:
        out = chat.invoke([{"role": "user", "content": q}], max_tokens=15)
        print(f"  Q: {q:30s} → {out['text']!r}")

    # Bus is async-flushed; give it a beat before reading.
    time.sleep(0.5)

    snap = tracker.snapshot()
    print()
    print("--- spend so far ---")
    print(f"calls:             {snap['calls']}")
    print(f"prompt tokens:     {snap['prompt_tokens']}")
    print(f"completion tokens: {snap['completion_tokens']}")
    print(f"USD spent:         ${snap['usd']:.6f}")


if __name__ == "__main__":
    main()
