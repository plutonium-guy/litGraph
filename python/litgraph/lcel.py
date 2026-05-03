"""LCEL-style pipe operator (`|`) for litGraph.

LangChain's marquee feature is composing primitives with `|`:

    chain = prompt | model | output_parser

litGraph rejects this as the *primary* abstraction (we use explicit
graphs), but provides a thin compat shim for users porting code or
who genuinely prefer LCEL ergonomics.

`Pipe(x)` wraps any callable / `ChatModel` / `Runnable`-shaped object.
Two `Pipe` values combine via `|`; `Pipe(...)(input)` invokes the
chain end-to-end. The wrapper is single-file, dep-free, and < 100
lines — its job is to *not* grow into a Runnable framework.

Example:

    from litgraph.lcel import Pipe
    from litgraph.providers import OpenAIChat
    from litgraph.prompts import ChatPromptTemplate

    chain = Pipe(ChatPromptTemplate.from_template("Tell me about {topic}.")) \
            | Pipe(OpenAIChat(model="gpt-5")) \
            | Pipe(lambda r: r["content"].strip())

    out = chain({"topic": "photosynthesis"})

Important: Pipe is sugar over `invoke`. It does NOT model fan-out,
checkpointing, time-travel, or any of the things `StateGraph` does.
For agents and orchestration, use `litgraph.graph.StateGraph` and
`litgraph.agents.ReactAgent` directly.
"""
from __future__ import annotations

from typing import Any, Callable


__all__ = ["Pipe", "parallel"]


class Pipe:
    """Wrap a callable / model / function so it composes with `|`.

    The wrapped value can be:
    - A bare Python callable `f(input) -> output`.
    - Anything with an `invoke(input)` method (litGraph chat models,
      retrievers, structured-output wrappers).
    - Anything with `__call__(input)` (closures, classes, agents).

    `__or__` is *associative-left* — `a | b | c` builds the chain
    `(a | b) | c`. Calling the chain runs each step's `invoke` (or
    falls back to `__call__`) on the previous step's output.
    """

    __slots__ = ("_steps",)

    def __init__(self, step: Any) -> None:
        # Steps stored as a flat list so chains stay shallow.
        self._steps: list[Any] = [step]

    def __or__(self, other: object) -> "Pipe":
        if isinstance(other, Pipe):
            new = Pipe.__new__(Pipe)
            new._steps = self._steps + other._steps
            return new
        new = Pipe.__new__(Pipe)
        new._steps = self._steps + [other]
        return new

    def __call__(self, input: Any) -> Any:
        out = input
        for step in self._steps:
            out = _invoke(step, out)
        return out

    # `invoke` alias so a Pipe can be wrapped by another Pipe / passed
    # anywhere the litGraph Runnable-shape is expected.
    def invoke(self, input: Any) -> Any:
        return self(input)

    def __repr__(self) -> str:
        names = [_step_name(s) for s in self._steps]
        return f"Pipe({' | '.join(names)})"


def _invoke(step: Any, value: Any) -> Any:
    """Dispatch one step. Tries `invoke` first (litGraph + LangChain
    convention), falls back to `__call__`."""
    inv = getattr(step, "invoke", None)
    if callable(inv):
        return inv(value)
    if callable(step):
        return step(value)
    raise TypeError(
        f"Pipe step is not callable and has no `invoke`: {type(step).__name__}"
    )


def _step_name(step: Any) -> str:
    return (
        getattr(step, "name", None)
        or getattr(step, "__name__", None)
        or type(step).__name__
    )


def parallel(*steps: Any) -> Callable[[Any], list[Any]]:
    """Return a callable that runs every `step` against the same input
    and returns the list of outputs in step order. Mirror of LangChain's
    `RunnableParallel` for users porting LCEL.

    `parallel` does NOT spawn threads — it's sequential by design,
    because the litGraph mental model for parallel work is `StateGraph`
    branches (which use the tokio worker pool + drop the GIL). LCEL
    parallel is sugar; if you need real concurrency, use a graph.

    Example:

        from litgraph.lcel import Pipe, parallel

        embed_then_search = Pipe(parallel(emb_provider_a, emb_provider_b))
        # → calling it with one query gives [hits_from_a, hits_from_b]
    """
    def run(value: Any) -> list[Any]:
        return [_invoke(s, value) for s in steps]
    run.__name__ = f"parallel({len(steps)})"
    return run
