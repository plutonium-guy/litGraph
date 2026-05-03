# litGraph community prompts

Curated, copy-paste-ready prompts for common agent patterns. Pull
what you need into your project and register via
`litgraph.prompt_hub.register(name, template, ...)`.

This is a flat folder of `*.md` files; each one starts with a YAML
front-matter block giving `name`, `tags`, `version`, `description`.
Anything below `---\n` is the prompt body.

## Why a folder, not a hosted hub?

LangChain Hub is a managed, paid service. The litGraph design call is
"prompts are source code" — track them in your VCS, version them
alongside your agent, no third-party fetch on cold-start. This folder
is the *seed*; fork-and-curate is the workflow.

## Adding a prompt

1. Drop a new `*.md` file with the YAML front matter.
2. Open a PR.
3. Maintain it like any other source file.

## Index

| File | Tags | Use |
|---|---|---|
| `react_terse.md` | react, agent | terse ReAct-style system prompt |
| `rag_qa.md` | rag, qa | strict-grounding QA over a context passage |
| `sql_agent.md` | rag, sql | SQL-tool-using analyst |
| `summarise_long_doc.md` | summarise | long-document map-reduce summary |

To use:

```python
from litgraph.prompt_hub import register

# Read + register at startup. Cheap; stays in-process.
import pathlib, re
for p in pathlib.Path("prompts").glob("*.md"):
    src = p.read_text()
    if src.startswith("---"):
        # Strip front matter for the body
        _, fm, body = src.split("---", 2)
        # Parse very simple key: value YAML — fork to PyYAML for richer.
        meta = {}
        for line in fm.strip().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip("'\"")
        register(meta["name"], body.strip(),
                 tags=tuple(meta.get("tags", "").split()),
                 version=meta.get("version", "1"),
                 description=meta.get("description", ""))
```
