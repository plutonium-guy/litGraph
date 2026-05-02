"""One-call factories for the most common litGraph patterns.

Designed for AI coding assistants. The pattern: when the user says
"build me an X", the agent's first try should be
`litgraph.recipes.X(...)` — if that exists and works, the agent ships
in 5 lines instead of 50.

Currently shipped:

- `eval(target, cases, scorers=None)` — wraps `EvalHarness` with sane
  defaults (exact-match + Levenshtein). Returns a `Report`.
- `serve(graph, port=8080)` — runs `litgraph-serve` against a compiled
  graph. Stub: prints the command to run; full impl pending.

Planned (see AGENT_DX.md §13):

- `rag(corpus_path, model=None)` — ingest + chunk + embed + agent in 1 call.
- `multi_agent(roles=[...])` — supervisor pre-wired.

Each recipe is intentionally opinionated: zero-config for the obvious
case, escape hatches for the rest.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping


__all__ = [
    "eval",
    "serve",
]


def eval(
    target: Callable[[str], str],
    cases: Iterable[Mapping[str, Any]],
    scorers: Iterable[Mapping[str, Any]] | None = None,
    max_parallel: int = 4,
) -> Any:
    """Run an eval over `cases` with default scorers (exact-match +
    levenshtein). Returns a dict `{per_case, aggregate}`.

    Args:
        target: callable taking the input string, returns the model
            output. Exceptions are recorded per-case, don't abort.
        cases: iterable of `{"input": str, "expected": str?,
            "metadata": any?}` dicts.
        scorers: optional list of scorer specs. Defaults to
            `[{"name": "exact_match"}, {"name": "levenshtein"}]`.
            Other built-ins: `jaccard`, `contains_all` (with
            `required: [...]`), `regex_match` (with `pattern: ...`),
            `embedding_cosine`, `json_validity`.
        max_parallel: concurrent target invocations.

    Example:

        from litgraph.testing import MockChatModel
        from litgraph.recipes import eval

        m = MockChatModel(replies=["Paris", "Berlin"])

        def predict(question: str) -> str:
            return m.invoke([{"role": "user", "content": question}])["content"]

        report = eval(predict, [
            {"input": "Capital of France?", "expected": "Paris"},
            {"input": "Capital of Germany?", "expected": "Berlin"},
        ])
        print(report["aggregate"]["means"])    # → {"exact_match": 1.0, ...}
    """
    # Lazy-import the native module so the recipe is importable on a
    # build without the .so (pure-Python sugar).
    from litgraph import evaluators  # type: ignore[attr-defined]

    if scorers is None:
        scorers = [
            {"name": "exact_match"},
            {"name": "levenshtein"},
        ]
    return evaluators.run_eval(
        list(cases),
        target,
        list(scorers),
        max_parallel=max_parallel,
    )


def serve(graph: Any, port: int = 8080, host: str = "0.0.0.0") -> str:
    """Render the shell command that would serve `graph` via
    `litgraph-serve`. Stub: doesn't actually spawn the binary yet —
    the binary lives in a separate Cargo target and we don't ship it
    in the wheel.

    Returns the command-line string so the caller can run it via
    `subprocess` or print it. Full one-call impl is planned (see
    AGENT_DX.md §13); the stub is here so coding-agent autocomplete
    finds the symbol.
    """
    if not hasattr(graph, "graph_id") and not hasattr(graph, "compile"):
        raise TypeError(
            "recipes.serve expects a CompiledGraph or a StateGraph; "
            "got "
            f"{type(graph).__name__}. Pass `g.compile()` or `g`."
        )
    return f"litgraph-serve --graph {graph!r} --host {host} --port {port}"
