"""LangChain-shape compat shims for code porting from LangChain.

litGraph's design intentionally rejects the LangChain `Runnable`
abstraction (in favour of explicit graphs), but real projects don't
port in one move. This module ships thin shims so a common import:

    from langchain.agents import AgentExecutor
    from langchain_core.runnables import RunnableLambda, RunnableParallel

…can become:

    from litgraph.compat import AgentExecutor, RunnableLambda, RunnableParallel

…without rewriting the call sites. Behaviour is approximate, not
identical. The shims prefer the *modern* litGraph idiom internally —
`AgentExecutor(agent, tools)` builds a `ReactAgent` and forwards
`invoke`; `RunnableParallel({...})` evaluates branches sequentially.
For full LCEL semantics, keep using LangChain.

These shims are NOT a long-term API. Migrate your call sites to
`litgraph.agents.ReactAgent` / `litgraph.graph.StateGraph` and delete
`from litgraph.compat import …` once the port is done.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping


__all__ = [
    "AgentExecutor",
    "RunnableLambda",
    "RunnableParallel",
    "RunnableBranch",
    "RunnablePassthrough",
]


# ---- AgentExecutor compat ----


class AgentExecutor:
    """Minimal stand-in for LangChain's `AgentExecutor`. Builds a
    `litgraph.agents.ReactAgent` under the hood. Accepts the
    most-common LangChain constructor kwargs and ignores the rest.

    Args:
        agent: ignored when `tools` is passed (the modern litGraph
            ReactAgent owns the loop). When `tools` is None,
            `agent.invoke` is forwarded directly.
        tools: tool list to wire into a ReactAgent. If provided, the
            shim builds `ReactAgent(model, tools)` where `model`
            comes from `agent.llm` if available.
        verbose: ignored (use OTel / `on_request` for tracing).
        max_iterations: forwarded as `max_iterations` to ReactAgent
            when supported.
    """

    def __init__(
        self,
        agent: Any = None,
        tools: Iterable[Any] | None = None,
        verbose: bool = False,
        max_iterations: int = 10,
        **_: Any,
    ) -> None:
        # Try to build a ReactAgent only when we have BOTH a real
        # ChatModel-shaped object AND a non-empty tool list. Empty
        # tools or model-less agent → fall through to forwarding the
        # `agent.invoke` LangChain users already wrote.
        tools_list = list(tools) if tools is not None else []
        impl: Any = None
        if tools_list and agent is not None and hasattr(agent, "llm"):
            try:
                from litgraph.agents import ReactAgent  # type: ignore[attr-defined]
                impl = ReactAgent(
                    agent.llm,
                    tools_list,
                    max_iterations=max_iterations,
                )
            except (ValueError, TypeError):
                # Underlying model is not a recognised litGraph chat
                # model. Fall back to the agent's own `invoke`.
                impl = agent
        if impl is None:
            impl = agent
        self._impl = impl
        self.verbose = verbose

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: Any,
        tools: Iterable[Any],
        **kwargs: Any,
    ) -> "AgentExecutor":
        """LangChain's classic factory. Equivalent to the constructor."""
        return cls(agent=agent, tools=tools, **kwargs)

    def invoke(self, input: Mapping[str, Any] | str) -> Any:
        # LangChain agents accept either a string or `{"input": str}`.
        if isinstance(input, Mapping):
            text = input.get("input") or input.get("messages") or ""
        else:
            text = input
        if self._impl is None:
            raise RuntimeError("AgentExecutor has no underlying agent")
        return self._impl.invoke(text)

    def __call__(self, input: Any) -> Any:
        return self.invoke(input)


# ---- Runnable shims ----


class RunnableLambda:
    """Wrap a Python function so it composes with `litgraph.lcel.Pipe`.
    Equivalent to LangChain's `RunnableLambda(f)` for `invoke`. Does
    NOT implement async, batch, or streaming variants — those use
    different idioms in litGraph (graph nodes / `batch_chat`).
    """

    def __init__(self, func: Callable[[Any], Any]) -> None:
        if not callable(func):
            raise TypeError("RunnableLambda requires a callable")
        self._func = func
        self.name = getattr(func, "__name__", "lambda")

    def invoke(self, input: Any) -> Any:
        return self._func(input)

    def __call__(self, input: Any) -> Any:
        return self.invoke(input)

    def __or__(self, other: Any) -> Any:
        from .lcel import Pipe
        return Pipe(self) | other


class RunnableParallel:
    """Run a set of named branches against the same input; return a
    dict of branch-name → result.

    Sequential by design. For real concurrency use
    `litgraph.graph.StateGraph` and let the Kahn scheduler fan branches
    out on the tokio worker pool (drops the GIL).
    """

    def __init__(self, branches: Mapping[str, Any] | None = None, **kw_branches: Any) -> None:
        merged: dict[str, Any] = dict(branches) if branches else {}
        merged.update(kw_branches)
        if not merged:
            raise ValueError("RunnableParallel requires at least one branch")
        self._branches = merged

    def invoke(self, input: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, branch in self._branches.items():
            inv = getattr(branch, "invoke", None)
            out[name] = inv(input) if callable(inv) else branch(input)
        return out

    def __call__(self, input: Any) -> dict[str, Any]:
        return self.invoke(input)


class RunnableBranch:
    """Conditional dispatch. Each `(predicate, branch)` pair is
    checked in order; the first whose predicate returns truthy runs.
    `default` (positional or keyword) handles the no-match case.

    Mirrors LangChain's API:

        RunnableBranch(
            (lambda x: "hello" in x, greeting_chain),
            (lambda x: "bye" in x, farewell_chain),
            default_chain,
        )
    """

    def __init__(self, *branches: Any, default: Any = None) -> None:
        # Last positional with no predicate becomes the default.
        if branches and not isinstance(branches[-1], tuple):
            *branches, default = branches  # type: ignore[assignment]
        self._branches: list[tuple[Callable[[Any], Any], Any]] = []
        for b in branches:
            if not isinstance(b, tuple) or len(b) != 2:
                raise TypeError("each branch must be (predicate, runnable)")
            self._branches.append(b)
        self._default = default

    def invoke(self, input: Any) -> Any:
        for pred, branch in self._branches:
            if pred(input):
                inv = getattr(branch, "invoke", None)
                return inv(input) if callable(inv) else branch(input)
        if self._default is None:
            raise RuntimeError("RunnableBranch: no predicate matched and no default")
        inv = getattr(self._default, "invoke", None)
        return inv(input) if callable(inv) else self._default(input)


class RunnablePassthrough:
    """Identity. Sugar for chains that need a no-op step (e.g. when
    one branch of a `RunnableParallel` should pass the input through
    unchanged)."""

    def invoke(self, input: Any) -> Any:
        return input

    def __call__(self, input: Any) -> Any:
        return input
