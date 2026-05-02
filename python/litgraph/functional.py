"""Functional API — `@task` + `@entrypoint` decorators for workflow authoring.

LangGraph parity layer. The native graph DSL (`litgraph.graph.StateGraph`)
remains the canonical way to build production workflows; this module ships
the LangGraph-style functional decorators for users who prefer the
inline-style code shape, especially for short scripts and notebook work.

# Why these decorators

LangGraph's `@entrypoint` + `@task` lets users write workflows that read
like ordinary async Python:

    @task
    async def fetch(url): ...

    @task
    async def summarize(text): ...

    @entrypoint()
    async def workflow(url):
        page = await fetch(url)
        return await summarize(page)

    result = workflow.invoke("https://example.com")

This is functionally equivalent to a 4-node `StateGraph`, but the inline
form trims boilerplate for one-shot tasks. The decorators here implement
the same surface so LangGraph users transitioning to litgraph can keep
their existing code shape.

# Implementation note

This is a v1 of the functional API. The current shape:

- `@task` is a marker decorator — it tags the function with
  `__litgraph_task__ = True` so future tooling (introspection,
  graph-build helpers) can identify task-functions. The function itself
  is unchanged at runtime.

- `@entrypoint(checkpointer=...)` wraps an async function into a `Workflow`
  object exposing `.invoke()` (sync), `.ainvoke()` (async), and `.astream()`
  (yields the final result as a single chunk).

Future iters can integrate `@task` calls inside an `@entrypoint`-wrapped
function with the native StateGraph runtime — recording each `await
task_fn(...)` as a graph node, applying the configured checkpointer for
mid-workflow durability. For now, the decorators provide the API shape;
the runtime delegates to plain async-await execution.

The checkpointer arg accepts any object with the litgraph Checkpointer
trait shape (in Python: a dict-like or a native checkpointer instance);
v1 does not yet plumb checkpoints through individual task awaits — track
that under the larger StateGraph integration roadmap item.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, AsyncIterator, Callable, Optional, TypeVar

T = TypeVar("T")


def task(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark an async function as a workflow task.

    Currently a tagging decorator — adds `__litgraph_task__ = True` to
    the function so introspection tooling can identify task functions.
    Future iters will integrate task awaits with the StateGraph runtime
    so each `await task_fn(...)` inside an `@entrypoint` body becomes a
    checkpointable graph node.

    Works on both async and sync functions. Async is preferred since
    the native runtime is tokio-backed; sync functions will be wrapped
    in `asyncio.to_thread` when integrated with the runtime.
    """
    fn.__litgraph_task__ = True  # type: ignore[attr-defined]
    return fn


class Workflow:
    """Wraps an async function as an invokable workflow.

    Mirrors LangGraph's runnable shape. Methods:
    - `invoke(*args, **kwargs)` — sync convenience; runs the workflow
      to completion via `asyncio.run`.
    - `ainvoke(*args, **kwargs)` — async; await directly.
    - `astream(*args, **kwargs)` — async iterator yielding `{"final":
      <result>}` as a single chunk in v1. Future versions will yield
      per-task progress events when integrated with StateGraph.

    The wrapped function MUST be async (`async def`). Sync functions are
    rejected at decoration time with a clear error so callers don't
    silently get a wrapped-but-broken workflow.
    """

    # Slots omit `__doc__` because that would shadow the class docstring;
    # `functools.update_wrapper` writes through the instance __dict__ for
    # name/doc/wrapped, which works fine without slots for those.
    __slots__ = ("_fn", "_checkpointer", "__name__", "__wrapped__", "__dict__")

    def __init__(
        self,
        fn: Callable[..., Any],
        checkpointer: Optional[Any] = None,
    ) -> None:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@entrypoint requires an async function (`async def`); "
                f"got {fn.__name__!r} which is sync. Wrap with `async def "
                f"{fn.__name__}(...): ...` or use the StateGraph DSL directly."
            )
        self._fn = fn
        self._checkpointer = checkpointer
        # Preserve metadata so introspection (help(), repr) works.
        functools.update_wrapper(self, fn, updated=())

    @property
    def checkpointer(self) -> Optional[Any]:
        return self._checkpointer

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self._fn(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        # Best-effort sync wrapper. If we're already inside an event
        # loop (e.g. Jupyter), guide the user to use `ainvoke` instead
        # rather than triggering "asyncio.run() cannot be called from a
        # running event loop" deep in the workflow.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._fn(*args, **kwargs))
        raise RuntimeError(
            f"Workflow.invoke({self.__name__}) called from inside a running "
            f"event loop. Use `await workflow.ainvoke(...)` instead, or run "
            f"`workflow.invoke(...)` from synchronous code."
        )

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[dict]:
        """Async-iterate workflow events. v1 yields a single
        `{"final": <result>}` chunk; future versions will yield per-task
        progress events when integrated with StateGraph runtime."""
        result = await self._fn(*args, **kwargs)
        yield {"final": result}

    def __repr__(self) -> str:
        return f"<Workflow {self.__name__!r} checkpointer={self._checkpointer!r}>"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Allow the decorated workflow to be called directly — returns
        # the underlying coroutine for callers who prefer `await
        # workflow(...)` over `await workflow.ainvoke(...)`.
        return self._fn(*args, **kwargs)


def entrypoint(
    checkpointer: Optional[Any] = None,
) -> Callable[[Callable[..., Any]], Workflow]:
    """Mark an async function as a workflow entrypoint.

    Returns a decorator that wraps the function in a `Workflow` object
    with `.invoke()` / `.ainvoke()` / `.astream()` methods. Pass an
    optional `checkpointer` for durability (full integration is roadmap
    work; v1 stores the reference for downstream tooling).

    Usage:

        @entrypoint()
        async def my_workflow(input):
            return await some_task(input)

        result = my_workflow.invoke({"x": 1})
    """

    def decorator(fn: Callable[..., Any]) -> Workflow:
        return Workflow(fn, checkpointer=checkpointer)

    return decorator
