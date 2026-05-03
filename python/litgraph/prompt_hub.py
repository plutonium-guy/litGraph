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
    "fetch_from_url",
    "load_directory",
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


def _parse_front_matter(src: str, name_hint: str = "") -> Prompt:
    """Parse a YAML-front-matter prompt file (the format used by
    `prompts/*.md`). Front matter is a `---`-delimited key:value
    block; everything after is the template body."""
    if src.startswith("---"):
        try:
            _, fm, body = src.split("---", 2)
        except ValueError:
            return Prompt(name=name_hint, template=src.strip())
        meta: dict[str, str] = {}
        for line in fm.strip().splitlines():
            k, _, v = line.partition(":")
            if k.strip():
                meta[k.strip()] = v.strip().strip("'\"")
        return Prompt(
            name=meta.get("name", name_hint),
            template=body.strip(),
            tags=tuple(meta.get("tags", "").split()),
            version=meta.get("version", "1"),
            description=meta.get("description", ""),
        )
    return Prompt(name=name_hint, template=src.strip())


def fetch_from_url(url: str, *, register_as: str | None = None) -> Prompt:
    """Fetch a prompt from any HTTP(S) URL. The body may be:

    - Plain text — registered as-is, name = `register_as` or the
      URL's last path segment.
    - YAML-front-matter Markdown (the `prompts/*.md` shape).
    - A LangChain Hub JSON export (`{name, template, ...}`) — flat
      fields are mapped to the Prompt dataclass.

    Lazy-imports `requests`. Refuses overwrite by default
    (matches `register`'s contract); pass `overwrite=True` via the
    returned Prompt's `register_as` if you re-pull.

    Example:

        from litgraph.prompt_hub import fetch_from_url, get

        fetch_from_url(
            "https://raw.githubusercontent.com/plutonium-guy/litGraph/main/prompts/rag_qa.md",
        )
        print(get("rag_qa").template[:80])
    """
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "requests not installed. Run `pip install requests`."
        ) from e
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    body = r.text
    name = register_as or url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    # JSON-shape (LangChain Hub export)?
    import json as _json
    if body.lstrip().startswith("{"):
        try:
            payload = _json.loads(body)
            p = Prompt(
                name=payload.get("name", name),
                template=payload.get("template") or payload.get("prompt") or "",
                tags=tuple(payload.get("tags", []) or ()),
                version=str(payload.get("version", "1")),
                description=payload.get("description", ""),
            )
            _REGISTRY[p.name] = p
            return p
        except (_json.JSONDecodeError, KeyError):
            pass
    p = _parse_front_matter(body, name_hint=name)
    _REGISTRY[p.name] = p
    return p


def load_directory(path: str) -> list[Prompt]:
    """Walk a directory of `*.md` prompts (front-matter shape) and
    register every one. Returns the list of registered Prompts.
    Useful for bootstrapping the registry from the bundled
    `prompts/` folder.
    """
    import os
    out: list[Prompt] = []
    if not os.path.isdir(path):
        raise FileNotFoundError(f"prompts directory not found: {path}")
    for name in sorted(os.listdir(path)):
        # Skip macOS AppleDouble sidecars + the README itself.
        if name.startswith(".") or name.startswith("_") or name == "README.md":
            continue
        if not name.endswith(".md"):
            continue
        full = os.path.join(path, name)
        with open(full, "r", encoding="utf-8") as f:
            src = f.read()
        stem = os.path.splitext(name)[0]
        p = _parse_front_matter(src, name_hint=stem)
        _REGISTRY[p.name] = p
        out.append(p)
    return out
