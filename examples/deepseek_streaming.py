"""Stream tokens from DeepSeek and print as they arrive.

Run:
    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_streaming.py
"""
from __future__ import annotations

import os
import sys

from litgraph.providers import OpenAIChat


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY")

    chat = OpenAIChat(
        api_key=api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )

    print("Streaming reply: ", end="", flush=True)
    final = None
    for ev in chat.stream(
        [{"role": "user", "content": "Count from one to five, one per line."}],
        max_tokens=40,
    ):
        if isinstance(ev, dict):
            t = ev.get("type")
            if t == "delta":
                sys.stdout.write(ev.get("text", ""))
                sys.stdout.flush()
            elif t == "done":
                final = ev
    print()
    if final:
        print(f"\n[finish_reason={final.get('finish_reason')!r} "
              f"usage={final.get('usage')}]")


if __name__ == "__main__":
    main()
