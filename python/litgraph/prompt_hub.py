"""Local prompt registry + pointer at the community hub.

LangChain Hub is a hosted, paid service. litGraph's hub is a thin
Python registry: you register prompts in-process (or import them from
your own package) and look them up by name. The community-shared
prompts live as plain Markdown in a GitHub directory (see
`HUB_URL`); pull what you need, register locally.

Example:

    from litgraph.prompt_hub import register, get, search

    register(
        "rag/sql_agent",
        "You are an analyst. Use the SQL tool to answer the user's question.\\n"
        "Always show the query, then the result.",
        tags=["rag", "sql"],
        version="1",
    )

    p = get("rag/sql_agent")        # raises KeyError if missing
    rough = search("rag")           # list of matching names
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


__all__ = [
    "Prompt",
    "register",
    "get",
    "search",
    "list_prompts",
    "clear",
    "HUB_URL",
]


# Community-shared prompts live here. Update if the directory moves.
HUB_URL = "https://github.com/plutonium-guy/litGraph/tree/main/prompts"


@dataclass(frozen=True)
class Prompt:
    """A registered prompt. Immutable so calling code can hold a
    reference without worrying about mutation across modules."""

    name: str
    template: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    version: str = "1"
    description: str = ""

    def render(self, **vars: object) -> str:
        """Render via Python `str.format(**vars)`. Use minijinja
        (via `litgraph.prompts.ChatPromptTemplate`) for full Jinja
        semantics — this is a one-liner for simple substitutions.
        """
        return self.template.format(**vars)


_REGISTRY: dict[str, Prompt] = {}


def register(
    name: str,
    template: str,
    *,
    tags: Iterable[str] = (),
    version: str = "1",
    description: str = "",
    overwrite: bool = False,
) -> Prompt:
    """Register a prompt under `name`. Refuses to overwrite an
    existing entry unless `overwrite=True` so accidental name
    collisions surface loudly."""
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"prompt {name!r} already registered (pass overwrite=True to replace)"
        )
    p = Prompt(
        name=name,
        template=template,
        tags=tuple(tags),
        version=version,
        description=description,
    )
    _REGISTRY[name] = p
    return p


def get(name: str) -> Prompt:
    """Return the registered prompt or raise `KeyError`."""
    return _REGISTRY[name]


def search(query: str) -> list[Prompt]:
    """Substring + tag search. Case-insensitive. Cheap; the registry
    is in-process and small."""
    q = query.lower()
    out: list[Prompt] = []
    for p in _REGISTRY.values():
        if q in p.name.lower() or q in p.description.lower():
            out.append(p)
            continue
        if any(q in t.lower() for t in p.tags):
            out.append(p)
    return out


def list_prompts() -> list[Prompt]:
    """Return every registered prompt, sorted by name."""
    return sorted(_REGISTRY.values(), key=lambda p: p.name)


def clear() -> None:
    """Drop every registered prompt. Mostly for tests."""
    _REGISTRY.clear()
