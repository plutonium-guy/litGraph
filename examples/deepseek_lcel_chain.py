"""LCEL-style composition with `Pipe` + `RunnableLambda` + a real model.

Run:
    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_lcel_chain.py

Builds:  prepare_messages | call_model | extract_text

A direct port of LangChain's LCEL pipeline shape, running against
DeepSeek via litGraph's `OpenAIChat`.
"""
from __future__ import annotations

import os

from litgraph.compat import RunnableLambda
from litgraph.lcel import Pipe
from litgraph.providers import OpenAIChat


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY in your environment.")

    chat = OpenAIChat(
        api_key=api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )

    prepare = RunnableLambda(lambda country: [
        {"role": "system", "content": "Reply with one word only."},
        {"role": "user", "content": f"Capital of {country}?"},
    ])

    def call_model(msgs):
        return chat.invoke(msgs, max_tokens=10)

    extract = RunnableLambda(lambda out: out["text"].strip())

    chain = Pipe(prepare) | call_model | extract
    for country in ["France", "Germany", "Italy"]:
        print(f"{country} -> {chain(country)}")


if __name__ == "__main__":
    main()
