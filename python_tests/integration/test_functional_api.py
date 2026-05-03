"""Live integration: `@entrypoint` + `@task` functional API.

API: `@entrypoint()` (with parens) wraps an `async def` workflow into
a `Workflow`. Tasks are `async def` too. Invoke via
`workflow.invoke(...)`.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_entrypoint_workflow_runs_a_model_call(deepseek_chat):
    from litgraph.functional import entrypoint, task

    @task
    async def ask(q: str) -> str:
        out = deepseek_chat.invoke(
            [{"role": "user", "content": q}],
            max_tokens=10,
        )
        return out["text"]

    @entrypoint()
    async def workflow(question: str) -> str:
        return await ask(question)

    text = workflow.invoke("Reply with: tasked")
    assert isinstance(text, str)
    assert text.strip()


def test_entrypoint_workflow_combines_two_tasks(deepseek_chat):
    """Two tasks called in sequence; entrypoint returns a dict."""
    from litgraph.functional import entrypoint, task

    @task
    async def alpha() -> str:
        out = deepseek_chat.invoke(
            [{"role": "user", "content": "Reply with just: ALPHA"}],
            max_tokens=10,
        )
        return out["text"]

    @task
    async def beta() -> str:
        out = deepseek_chat.invoke(
            [{"role": "user", "content": "Reply with just: BETA"}],
            max_tokens=10,
        )
        return out["text"]

    @entrypoint()
    async def workflow(_unused: int) -> dict:
        return {"a": await alpha(), "b": await beta()}

    out = workflow.invoke(0)
    assert isinstance(out, dict)
    assert "ALPHA" in out["a"].upper()
    assert "BETA" in out["b"].upper()
