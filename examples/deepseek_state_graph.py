"""StateGraph with parallel branches calling DeepSeek concurrently.

The two `fetch_*` branches run on the tokio worker pool — both API
calls are in flight at the same time. The `merge` node aggregates
when both have returned.

Run:
    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_state_graph.py
"""
from __future__ import annotations

import os
import time

from litgraph.providers import OpenAIChat
from litgraph.graph import StateGraph, START, END


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY")

    chat = OpenAIChat(
        api_key=api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )

    def ask_capital(country: str):
        out = chat.invoke(
            [{"role": "user", "content": f"What is the capital of {country}? Reply with one word."}],
            max_tokens=10,
        )
        return out["text"].strip()

    g = StateGraph()

    def france(_state):
        return {"france": ask_capital("France")}

    def germany(_state):
        return {"germany": ask_capital("Germany")}

    def japan(_state):
        return {"japan": ask_capital("Japan")}

    def merge(_state):
        return {}

    for name, fn in [("france", france), ("germany", germany), ("japan", japan)]:
        g.add_node(name, fn)
        g.add_edge(START, name)
        g.add_edge(name, "merge")
    g.add_node("merge", merge)
    g.add_edge("merge", END)

    started = time.time()
    out = g.compile().invoke({})
    elapsed = time.time() - started

    print(f"Capitals (3 parallel API calls in {elapsed:.2f}s):")
    print(f"  France:  {out['france']}")
    print(f"  Germany: {out['germany']}")
    print(f"  Japan:   {out['japan']}")


if __name__ == "__main__":
    main()
