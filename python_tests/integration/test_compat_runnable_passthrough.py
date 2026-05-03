"""Live integration: `RunnablePassthrough` identity in a parallel branch.

Pairs `RunnablePassthrough` (identity) with a model-call branch in a
`RunnableParallel` so callers can preserve the original input alongside
the model's transformed output — a common LCEL pattern for context
that needs to flow alongside generation.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_passthrough_in_parallel_with_model(deepseek_chat):
    from litgraph.compat import RunnableParallel, RunnablePassthrough

    def call_model(country: str) -> str:
        return deepseek_chat.invoke(
            [{"role": "user", "content": f"Capital of {country}? One word."}],
            max_tokens=10,
        )["text"].strip()

    par = RunnableParallel({
        "input": RunnablePassthrough(),
        "capital": call_model,
    })
    out = par("France")
    assert out["input"] == "France"
    assert "Paris" in out["capital"]
