"""Live integration: LangChain-compat `Runnable*` shims with a real model.

These shims exist so LangChain users can port code with minimal edits.
We exercise the round-trip: `RunnableLambda` and `RunnableParallel`
composed via `Pipe` against DeepSeek.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_runnable_lambda_pipes_into_model(deepseek_chat):
    from litgraph.compat import RunnableLambda
    from litgraph.lcel import Pipe

    to_msgs = RunnableLambda(
        lambda country: [
            {"role": "system", "content": "Reply with one word only."},
            {"role": "user", "content": f"Capital of {country}?"},
        ]
    )

    def call(msgs):
        return deepseek_chat.invoke(msgs, max_tokens=10)

    chain = Pipe(to_msgs) | call
    out = chain("Spain")
    assert "Madrid" in out["text"]


def test_runnable_parallel_dispatch(deepseek_chat):
    """RunnableParallel runs a dict of branches over the same input
    and returns a dict of the same keys."""
    from litgraph.compat import RunnableParallel

    def short(country):
        return deepseek_chat.invoke(
            [{"role": "user", "content": f"Capital of {country}? One word."}],
            max_tokens=10,
        )["text"]

    def loud(country):
        return country.upper()

    par = RunnableParallel({"capital": short, "loud": loud})
    out = par("italy")
    assert isinstance(out, dict)
    assert out["loud"] == "ITALY"
    assert "Rome" in out["capital"]
