"""Demo: `model.with_cache(...)` saves a roundtrip on the second call.

Run:
    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_caching.py
"""
from __future__ import annotations

import os
import time

from litgraph.providers import OpenAIChat
from litgraph.cache import MemoryCache


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY")

    chat = OpenAIChat(
        api_key=api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )
    chat.with_cache(MemoryCache(max_capacity=128))

    msgs = [{"role": "user", "content": "Reply with: cached-demo"}]

    t0 = time.time()
    a = chat.invoke(msgs, max_tokens=10, temperature=0)
    t_first = time.time() - t0

    t0 = time.time()
    b = chat.invoke(msgs, max_tokens=10, temperature=0)
    t_second = time.time() - t0

    print(f"First  call: {t_first*1000:.1f} ms  → {a['text']!r}")
    print(f"Second call: {t_second*1000:.1f} ms  → {b['text']!r}")
    speedup = t_first / max(t_second, 1e-6)
    print(f"Cache speedup: {speedup:.1f}x")
    assert a["text"] == b["text"], "cache should return identical text"


if __name__ == "__main__":
    main()
