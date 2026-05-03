"""Live integration: lcel `Pipe` composition with a real model.

`Pipe(step) | step | step` builds a chain where each step's output is
fed into the next. Steps can be plain callables, anything with
`.invoke(input)`, or `__call__`. Hits DeepSeek twice through a small
chain.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_pipe_callable_then_model(deepseek_chat):
    from litgraph.lcel import Pipe

    def to_messages(country: str) -> list[dict]:
        return [
            {"role": "system", "content": "Reply with one word only."},
            {"role": "user", "content": f"Capital of {country}?"},
        ]

    def call_model(msgs: list[dict]) -> dict:
        return deepseek_chat.invoke(msgs, max_tokens=10)

    chain = Pipe(to_messages) | call_model
    out = chain("France")
    assert "Paris" in out["text"]


def test_pipe_extract_text(deepseek_chat):
    from litgraph.lcel import Pipe

    def call_model(country: str) -> dict:
        return deepseek_chat.invoke(
            [{"role": "user", "content": f"Capital of {country}? One word."}],
            max_tokens=10,
        )

    def text_only(out: dict) -> str:
        return out["text"].strip()

    chain = Pipe(call_model) | text_only
    text = chain("Germany")
    assert "Berlin" in text
