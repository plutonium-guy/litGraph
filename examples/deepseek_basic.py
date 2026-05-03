"""DeepSeek hello-world via the OpenAI-compat path.

DeepSeek implements OpenAI's `/v1/chat/completions`, so the existing
`OpenAIChat` adapter works as-is — pass a `base_url` + your DeepSeek
API key and you're done. No new provider crate needed.

Run:
    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_basic.py
"""
from __future__ import annotations

import os

from litgraph.providers import OpenAIChat


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set DEEPSEEK_API_KEY to your key from https://platform.deepseek.com/"
        )

    chat = OpenAIChat(
        api_key=api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )

    out = chat.invoke(
        [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "What's the capital of Japan?"},
        ],
        max_tokens=20,
    )
    print("text:    ", out["text"])
    print("model:   ", out["model"])
    print("usage:   ", out["usage"])
    print("finish:  ", out["finish_reason"])


if __name__ == "__main__":
    main()
