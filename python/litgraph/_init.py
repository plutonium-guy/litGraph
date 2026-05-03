"""`litgraph init <template> <dir>` — scaffold a minimal project.

Templates ship as in-memory string dicts so they work in a wheel
install (no example files on disk required). Each template:

- Drops the right files into `<dir>` (creates `<dir>` if needed).
- Includes a `pyproject.toml`, a `.env.example`, a `README.md`, and a
  runnable hello-world that uses `litgraph.testing` mocks so
  `python <entrypoint>` exits 0 with no API key.
- Includes a pytest test file that passes out of the box.

Available templates:
    chat-agent     A ReactAgent with one tool. Tests with MockChatModel.
    rag            Vector-store + retriever + RAG node. Mock embeddings.
    eval-suite     EvalHarness with a tiny golden set + scoreboard.

Usage:
    litgraph init chat-agent ./my-app
    cd my-app
    pip install -e .
    pytest
"""
from __future__ import annotations

import os
import sys


__all__ = ["main", "TEMPLATES"]


_PYPROJECT = """\
[project]
name = "{slug}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.9"
dependencies = ["litgraph"]

[project.optional-dependencies]
dev = ["pytest"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["{slug}"]
"""


_ENV_EXAMPLE = """\
# Copy to .env and fill in. The hello-world runs without these
# (uses MockChatModel), but real providers need them.
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
"""


_README = """\
# {name}

Scaffolded by `litgraph init {template}`.

## Quickstart

```bash
pip install -e .
pytest                    # tests pass with no API key
python -m {slug}          # runs the hello-world (mocked)
```

## Real provider

Set the right `*_API_KEY` env var, then edit `{slug}/__main__.py` to
swap `MockChatModel(...)` for `OpenAIChat(model="gpt-5")` (or any
other litGraph provider). The rest of the code is unchanged.

## See also

- [`litgraph` PyPI](https://pypi.org/project/litgraph/)
- [USAGE.md](https://github.com/plutonium-guy/litGraph/blob/main/USAGE.md) — subsystem how-to
- [AGENT_DX.md](https://github.com/plutonium-guy/litGraph/blob/main/AGENT_DX.md) — DX features for AI coding agents
"""


_AGENTS_MD = """\
# Agent rules for {name}

Coding agent: read this first.

## Build / test

```bash
pip install -e .
pytest                # all tests must pass
python -m {slug}      # runs the hello-world
```

## Where things live

- `{slug}/__init__.py` — exports the public agent factory.
- `{slug}/__main__.py` — runnable hello-world (mocked, no API key).
- `tests/test_agent.py` — locks the agent's contract; uses
  `litgraph.testing.MockChatModel` for determinism.

## Conventions

- Don't add a real provider call to `__main__.py` — that's a demo.
  Real provider goes in user code that calls into `{slug}.build_agent(...)`.
- New tests live in `tests/test_<feature>.py`.
- Use `litgraph.testing.MockChatModel`, `MockEmbeddings`, `MockTool` in
  tests — never burn credits.
"""


_CHAT_AGENT_INIT = """\
\"\"\"{name} — a ReactAgent with one tool, scaffolded by `litgraph init`.\"\"\"
from __future__ import annotations

from typing import Any


def add(args: dict[str, Any]) -> dict[str, Any]:
    \"\"\"Tool: add two integers. Returns {{'sum': int}}.\"\"\"
    return {{"sum": int(args["a"]) + int(args["b"])}}


def add_tool() -> Any:
    \"\"\"Build the `add` FunctionTool. Lazy import so the module is
    importable on a build without the litGraph native module.\"\"\"
    from litgraph.tools import FunctionTool
    return FunctionTool(
        "add", "Add two integers.",
        {{"type": "object",
          "properties": {{"a": {{"type": "integer"}}, "b": {{"type": "integer"}}}},
          "required": ["a", "b"]}},
        add,
    )


def build_agent(model: Any, **agent_kwargs: Any) -> Any:
    \"\"\"Wire model + tools + system prompt. Pass any ChatModel.

    Example:

        from litgraph.providers import OpenAIChat
        from {slug} import build_agent

        agent = build_agent(OpenAIChat(model="gpt-5"))
        print(agent.invoke("17 + 25?"))
    \"\"\"
    from litgraph.agents import ReactAgent
    return ReactAgent(
        model,
        [add_tool()],
        system_prompt=agent_kwargs.pop("system_prompt", "Be terse."),
        **agent_kwargs,
    )
"""


_CHAT_AGENT_MAIN = """\
\"\"\"Hello-world entry point. Runs offline with MockChatModel.

Run:    python -m {slug}
\"\"\"
from litgraph.testing import MockChatModel

from {slug} import add


def main() -> None:
    # Mock returns a tool-call followed by the final answer. Replace
    # with `OpenAIChat(model="gpt-5")` (or any ChatModel) once
    # `OPENAI_API_KEY` is set.
    m = MockChatModel(replies=[
        # First reply: pretend to call the `add` tool.
        {{"role": "assistant",
          "content": "I'll add those.",
          "tool_calls": [{{"id": "1", "name": "add",
                           "args": {{"a": 17, "b": 25}}}}],
          "usage": {{"prompt": 0, "completion": 0, "total": 0}}}},
        # Second reply: final answer using the tool result.
        {{"role": "assistant",
          "content": "17 + 25 = 42",
          "tool_calls": [],
          "usage": {{"prompt": 0, "completion": 0, "total": 0}}}},
    ])

    # Without the native ReactAgent loop wired up here we just
    # demonstrate the tool + the model. Real usage:
    #     agent = build_agent(real_model); agent.invoke("17 + 25?")
    print("model says:", m.invoke([{{"role": "user", "content": "17+25?"}}])["content"])
    print("add tool:", add({{"a": 17, "b": 25}}))


if __name__ == "__main__":
    main()
"""


_CHAT_AGENT_TEST = """\
\"\"\"Tests for the {slug} hello-world.\"\"\"
from litgraph.testing import MockChatModel, MockTool

from {slug} import add, add_tool


def test_add_returns_sum():
    assert add({{"a": 17, "b": 25}}) == {{"sum": 42}}


def test_add_tool_has_schema():
    t = add_tool()
    assert t.name == "add"
    # Schema declares two integer args.
    schema = getattr(t, "schema", None) or getattr(t, "_schema", None)
    if schema is not None:
        assert "a" in str(schema)
        assert "b" in str(schema)


def test_mock_model_is_callable():
    m = MockChatModel(replies=["ok"])
    assert m.invoke([{{"role": "user", "content": "x"}}])["content"] == "ok"


def test_mock_tool_records_calls():
    t = MockTool("noop", returns=42)
    assert t.invoke({{}}) == 42
    assert t.calls == [{{}}]
"""


# Templates are dicts: relative-path → file content. Format strings
# resolve against `{name, slug, template, description}`.
_RAG_INIT = """\
\"\"\"{name} — a RAG agent scaffolded by `litgraph init`.\"\"\"
from __future__ import annotations

from typing import Any


SAMPLE_DOCS = [
    {{"page_content": "The capital of France is Paris.", "metadata": {{"source": "geo"}}}},
    {{"page_content": "Photosynthesis converts light into chemical energy.",
     "metadata": {{"source": "bio"}}}},
    {{"page_content": "TCP guarantees in-order delivery; UDP does not.",
     "metadata": {{"source": "net"}}}},
]


def build_agent(model: Any, embeddings: Any, documents: Any = None) -> Any:
    \"\"\"Build a RAG agent over `documents` (defaults to SAMPLE_DOCS).

    Example with real providers:

        from litgraph.providers import OpenAIChat
        from litgraph.embeddings import OpenAIEmbeddings
        from {slug} import build_agent

        agent = build_agent(OpenAIChat(model="gpt-5"), OpenAIEmbeddings())
        print(agent.invoke("What is photosynthesis?"))
    \"\"\"
    from litgraph.recipes import rag
    docs = list(documents) if documents is not None else SAMPLE_DOCS
    return rag(documents=docs, model=model, embeddings=embeddings)
"""

_RAG_MAIN = """\
\"\"\"Hello-world entry point. Runs offline with mocks.

Run:    python -m {slug}
\"\"\"
from litgraph.testing import MockChatModel, MockEmbeddings

from {slug} import SAMPLE_DOCS, build_agent


def main() -> None:
    m = MockChatModel(replies=["The capital of France is Paris (per the context)."])
    emb = MockEmbeddings(dim=8)
    agent = build_agent(m, emb)
    out = agent.invoke("What is the capital of France?")
    print("answer:", out["answer"])
    print("hits:", len(out["hits"]))


if __name__ == "__main__":
    main()
"""

_RAG_TEST = """\
\"\"\"Tests for the {slug} RAG agent.\"\"\"
from litgraph.testing import MockChatModel, MockEmbeddings

from {slug} import SAMPLE_DOCS, build_agent


def test_sample_docs_have_expected_shape():
    assert len(SAMPLE_DOCS) >= 1
    for d in SAMPLE_DOCS:
        assert "page_content" in d


def test_rag_agent_returns_answer_and_hits():
    m = MockChatModel(replies=["mocked rag answer"])
    emb = MockEmbeddings(dim=8)
    agent = build_agent(m, emb)
    out = agent.invoke("What is the capital of France?")
    assert out["answer"] == "mocked rag answer"
    assert isinstance(out["hits"], list)
"""


_EVAL_INIT = """\
\"\"\"{name} — eval suite scaffolded by `litgraph init`.\"\"\"
from __future__ import annotations

from typing import Any, Callable, Sequence


SAMPLE_CASES: list[dict[str, str]] = [
    {{"input": "Capital of France?", "expected": "Paris"}},
    {{"input": "Capital of Germany?", "expected": "Berlin"}},
    {{"input": "Capital of Japan?", "expected": "Tokyo"}},
]


def run_eval(
    target: Callable[[str], str],
    cases: Sequence[dict[str, str]] = SAMPLE_CASES,
) -> Any:
    \"\"\"Run the eval against any target callable. Returns a report dict
    with `per_case` + `aggregate.means` (exact_match, levenshtein).

    Example:

        from litgraph.providers import OpenAIChat
        from {slug} import run_eval

        m = OpenAIChat(model="gpt-5")
        report = run_eval(lambda q: m.invoke([{{"role":"user","content":q}}])["content"])
        print(report["aggregate"]["means"])
    \"\"\"
    from litgraph.recipes import eval as _eval
    return _eval(target, cases)
"""

_EVAL_MAIN = """\
\"\"\"Hello-world entry point. Runs offline with a MockChatModel.

Run:    python -m {slug}
\"\"\"
from litgraph.testing import MockChatModel

from {slug} import run_eval


def main() -> None:
    m = MockChatModel(replies=["Paris", "Berlin", "Tokyo"])

    def predict(q: str) -> str:
        return m.invoke([{{"role": "user", "content": q}}])["content"]

    report = run_eval(predict)
    print("=== aggregate ===")
    for k, v in report["aggregate"]["means"].items():
        print(f"  {{k}}: {{v:.3f}}")
    print(f"  n_cases: {{report['aggregate']['n_cases']}}")


if __name__ == "__main__":
    main()
"""

_EVAL_TEST = """\
\"\"\"Tests for the {slug} eval suite.\"\"\"
from litgraph.testing import MockChatModel

from {slug} import SAMPLE_CASES, run_eval


def test_sample_cases_have_expected_shape():
    for c in SAMPLE_CASES:
        assert "input" in c
        assert "expected" in c


def test_run_eval_returns_aggregate():
    m = MockChatModel(replies=["Paris", "Berlin", "Tokyo"])
    report = run_eval(
        lambda q: m.invoke([{{"role": "user", "content": q}}])["content"]
    )
    assert report["aggregate"]["n_cases"] == len(SAMPLE_CASES)
    assert report["aggregate"]["means"]["exact_match"] == 1.0
"""


TEMPLATES: dict[str, dict[str, str]] = {
    "chat-agent": {
        "pyproject.toml": _PYPROJECT,
        ".env.example": _ENV_EXAMPLE,
        "README.md": _README,
        "AGENTS.md": _AGENTS_MD,
        "{slug}/__init__.py": _CHAT_AGENT_INIT,
        "{slug}/__main__.py": _CHAT_AGENT_MAIN,
        "tests/__init__.py": "",
        "tests/test_agent.py": _CHAT_AGENT_TEST,
    },
    "rag": {
        "pyproject.toml": _PYPROJECT,
        ".env.example": _ENV_EXAMPLE,
        "README.md": _README,
        "AGENTS.md": _AGENTS_MD,
        "{slug}/__init__.py": _RAG_INIT,
        "{slug}/__main__.py": _RAG_MAIN,
        "tests/__init__.py": "",
        "tests/test_agent.py": _RAG_TEST,
    },
    "eval-suite": {
        "pyproject.toml": _PYPROJECT,
        ".env.example": _ENV_EXAMPLE,
        "README.md": _README,
        "AGENTS.md": _AGENTS_MD,
        "{slug}/__init__.py": _EVAL_INIT,
        "{slug}/__main__.py": _EVAL_MAIN,
        "tests/__init__.py": "",
        "tests/test_agent.py": _EVAL_TEST,
    },
}


def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower() or "app"


def _render(content: str, ctx: dict[str, str]) -> str:
    """Render `{name}`-style placeholders. Doubled braces in templates
    (`{{x}}`) survive as literal `{x}` for code blocks."""
    return content.format(**ctx)


def _materialise(template_name: str, target_dir: str) -> int:
    if template_name not in TEMPLATES:
        print(f"litgraph init: unknown template {template_name!r}", file=sys.stderr)
        print(f"available: {', '.join(sorted(TEMPLATES))}", file=sys.stderr)
        return 2
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"litgraph init: target {target_dir!r} exists and is not empty",
              file=sys.stderr)
        return 1

    name = os.path.basename(os.path.abspath(target_dir))
    slug = _slugify(name)
    ctx = {
        "name": name,
        "slug": slug,
        "template": template_name,
        "description": f"litGraph {template_name} scaffolded by `litgraph init`",
    }

    files = TEMPLATES[template_name]
    written = 0
    for rel_path, content in files.items():
        rel = rel_path.format(**ctx)
        dest = os.path.join(target_dir, rel)
        os.makedirs(os.path.dirname(dest) or target_dir, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(_render(content, ctx))
        written += 1

    print(f"litgraph init: wrote {written} file(s) into {target_dir}/")
    print()
    print(f"  cd {target_dir}")
    print("  pip install -e '.[dev]'")
    print("  pytest")
    print(f"  python -m {slug}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("usage: litgraph init <template> <dir>")
        print(f"templates: {', '.join(sorted(TEMPLATES))}")
        return 0
    if len(argv) < 2:
        print("usage: litgraph init <template> <dir>", file=sys.stderr)
        return 2
    template, target = argv[0], argv[1]
    return _materialise(template, target)


if __name__ == "__main__":
    sys.exit(main())
