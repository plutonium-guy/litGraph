"""Tests for the @entrypoint + @task functional API.

These tests exercise the pure-Python decorators in
`python/litgraph/functional.py`. They don't require the native module
to be loaded — the decorators are pure-Python sugar.
"""

import asyncio
import pytest

# Import directly from the source path to keep the test runnable
# without an installed maturin build of the native module. CI that
# builds the wheel can switch to `from litgraph import entrypoint, task`.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from litgraph.functional import entrypoint, task, Workflow  # noqa: E402


# ─── @task ─────────────────────────────────────────────────────────


def test_task_marks_function():
    @task
    async def my_task(x):
        return x + 1

    assert getattr(my_task, "__litgraph_task__", False) is True


def test_task_preserves_callable():
    @task
    async def my_task(x):
        return x * 2

    result = asyncio.run(my_task(5))
    assert result == 10


# ─── @entrypoint ───────────────────────────────────────────────────


def test_entrypoint_returns_workflow():
    @entrypoint()
    async def workflow(x):
        return x + 1

    assert isinstance(workflow, Workflow)


def test_entrypoint_invoke_runs_workflow():
    @entrypoint()
    async def workflow(x):
        return x * 3

    result = workflow.invoke(7)
    assert result == 21


def test_entrypoint_ainvoke_returns_coroutine():
    @entrypoint()
    async def workflow(x):
        return x + 100

    result = asyncio.run(workflow.ainvoke(5))
    assert result == 105


def test_entrypoint_with_task_calls():
    @task
    async def add_one(x):
        return x + 1

    @task
    async def double(x):
        return x * 2

    @entrypoint()
    async def workflow(x):
        a = await add_one(x)
        b = await double(a)
        return b

    # (5 + 1) * 2 = 12
    assert workflow.invoke(5) == 12


def test_entrypoint_rejects_sync_function():
    with pytest.raises(TypeError, match="async function"):

        @entrypoint()
        def workflow(x):  # NOT async — should error.
            return x

        # Force evaluation in case decorator is lazy.
        _ = workflow


def test_entrypoint_checkpointer_stored():
    sentinel_checkpointer = {"backend": "memory"}

    @entrypoint(checkpointer=sentinel_checkpointer)
    async def workflow(x):
        return x

    assert workflow.checkpointer is sentinel_checkpointer


def test_entrypoint_invoke_inside_running_loop_errors_clearly():
    # asyncio.run with a workflow that calls .invoke from inside its
    # own loop should fail with a clear message, not a confusing
    # "asyncio.run() cannot be called from a running event loop"
    # crash deep in the workflow.
    @entrypoint()
    async def inner(x):
        return x

    @entrypoint()
    async def outer(x):
        # Calling .invoke() from inside an event loop should raise.
        return inner.invoke(x)

    with pytest.raises(RuntimeError, match="running event loop"):
        outer.invoke(1)


def test_workflow_repr():
    @entrypoint()
    async def named_wf(x):
        return x

    r = repr(named_wf)
    assert "named_wf" in r
    assert "Workflow" in r


def test_workflow_callable_directly():
    # The Workflow object is callable — returns the underlying coroutine
    # so users can `await workflow(...)` without `.ainvoke(...)`.
    @entrypoint()
    async def workflow(x):
        return x + 1

    coro = workflow(5)
    assert asyncio.iscoroutine(coro)
    assert asyncio.run(coro) == 6


def test_astream_yields_final_chunk():
    @entrypoint()
    async def workflow(x):
        return x * 2

    async def collect():
        chunks = []
        async for chunk in workflow.astream(5):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())
    assert chunks == [{"final": 10}]


def test_workflow_preserves_function_name_and_doc():
    @entrypoint()
    async def documented_workflow(x):
        """A workflow with a docstring."""
        return x

    assert documented_workflow.__name__ == "documented_workflow"
    assert documented_workflow.__doc__ == "A workflow with a docstring."
