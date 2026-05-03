"""Live integration: LangChain-compat `RunnableBranch` over a real model.

`RunnableBranch((pred, branch), ..., default=)` dispatches to the first
branch whose predicate returns truthy. Combined with a model-call, it
gives LangChain users a drop-in port for conditional pipelines.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_runnable_branch_routes_to_correct_chain(deepseek_chat):
    from litgraph.compat import RunnableBranch, RunnableLambda

    def call_capital(country: str) -> str:
        return deepseek_chat.invoke(
            [{"role": "user", "content": f"Capital of {country}? One word."}],
            max_tokens=10,
        )["text"].strip()

    capital_chain = RunnableLambda(call_capital)
    upper_chain = RunnableLambda(lambda s: s.upper())
    default_chain = RunnableLambda(lambda s: f"unhandled: {s}")

    branch = RunnableBranch(
        (lambda s: s.startswith("capital:"), lambda s: capital_chain(s.removeprefix("capital:").strip())),
        (lambda s: s.startswith("loud:"), lambda s: upper_chain(s.removeprefix("loud:").strip())),
        default=default_chain,
    )

    cap = branch.invoke("capital: Spain")
    assert "Madrid" in cap, f"capital branch failed: {cap!r}"

    loud = branch.invoke("loud: hello world")
    assert loud == "HELLO WORLD"

    other = branch.invoke("nothing matches this")
    assert other.startswith("unhandled:")
