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
    "rag",
    "multi_agent",
    "summarize",
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


# ---- One-call RAG ----


def rag(
    documents: Iterable[Mapping[str, Any]] | None = None,
    *,
    model: Any = None,
    embeddings: Any = None,
    store: Any = None,
    retriever_k: int = 5,
    system_prompt: str | None = None,
) -> Any:
    """Return a ready-to-invoke RAG agent.

    Wires `embeddings → store.add(docs) → MMR retriever → ReactAgent`
    in one call. Sensible defaults so a user types:

        from litgraph.providers import OpenAIChat
        from litgraph.embeddings import OpenAIEmbeddings
        from litgraph.recipes import rag

        agent = rag(
            documents=my_docs,                   # list of {page_content, metadata?}
            model=OpenAIChat(model="gpt-5"),
            embeddings=OpenAIEmbeddings(),
        )
        print(agent.invoke("What does the corpus say about X?"))

    Args:
        documents: list of doc dicts to ingest. If None, the user
            will fill the store themselves before calling
            `agent.invoke`.
        model: any litGraph ChatModel.
        embeddings: any embeddings provider with `embed(texts)`.
        store: optional pre-built VectorStore. If None, a
            `MemoryStore` is created with the embeddings dim.
        retriever_k: top-k to fetch per query.
        system_prompt: override the default RAG system prompt.

    Returns: an object with `.invoke(query)` that runs the full RAG
    loop. Raises if model + embeddings aren't supplied.
    """
    if model is None:
        raise ValueError("rag(model=...) is required")
    if embeddings is None:
        raise ValueError("rag(embeddings=...) is required")

    # Build the store lazily so users without a store provider get
    # an in-memory default.
    if store is None:
        try:
            from litgraph.stores import MemoryStore  # type: ignore[attr-defined]
            # Memory store needs dim; pull from a probe embedding.
            probe = embeddings.embed(["__probe__"])[0]
            store = MemoryStore(dim=len(probe))
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                "no `store` passed and could not auto-build MemoryStore "
                f"({e}). Pass `store=<VectorStore>`."
            ) from e

    if documents is not None:
        docs_list = list(documents)
        if docs_list:
            vecs = embeddings.embed([d.get("page_content", "") for d in docs_list])
            store.add(docs_list, vecs)

    # Build retriever + agent.
    from litgraph.retrieval import VectorRetriever, MMRRetriever  # type: ignore[attr-defined]
    base = VectorRetriever(embeddings, store, k=retriever_k)
    retriever = MMRRetriever(base, fetch_k=retriever_k * 4, k=retriever_k, lambda_mult=0.5)

    default_prompt = (
        "You answer questions using the provided context. If the answer isn't "
        "in the context, say so plainly. Cite specific snippets you use."
    )

    return _RagAgent(
        model=model,
        retriever=retriever,
        system_prompt=system_prompt or default_prompt,
    )


class _RagAgent:
    """Tiny dispatcher that retrieves on every invoke + assembles a
    context-aware prompt. Kept inside `recipes` so the import surface
    of `litgraph.agents` stays unchanged."""

    def __init__(self, model: Any, retriever: Any, system_prompt: str) -> None:
        self.model = model
        self.retriever = retriever
        self.system_prompt = system_prompt

    def invoke(self, query: str) -> dict[str, Any]:
        hits = self.retriever.retrieve(query)
        # Format hits as numbered context entries.
        ctx_parts = []
        for i, h in enumerate(hits, 1):
            content = h.get("page_content", "") if isinstance(h, dict) else getattr(h, "page_content", "")
            ctx_parts.append(f"[{i}] {content}")
        context = "\n\n".join(ctx_parts) if ctx_parts else "(no context)"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
        response = self.model.invoke(messages)
        return {
            "answer": response.get("content", "") if isinstance(response, dict) else str(response),
            "hits": hits,
            "messages": messages + [response if isinstance(response, dict) else {"role": "assistant", "content": str(response)}],
        }


# ---- One-call multi-agent supervisor ----


def multi_agent(
    workers: Mapping[str, Any],
    *,
    supervisor_model: Any,
    system_prompt: str | None = None,
) -> Any:
    """Return a supervisor-style multi-agent ready to dispatch.

    The supervisor reads the user's request + each worker's
    description, picks one to handle the turn, runs it, and returns
    the result. For richer flows (handoffs, debate) use `SwarmAgent`
    or `DebateAgent` directly.

    Args:
        workers: dict of `{role_name: agent}`. Each agent must
            implement `invoke(input)`. The role_name doubles as the
            description for routing — pick descriptive names like
            "billing_specialist" / "tech_support".
        supervisor_model: ChatModel for routing. The supervisor sends
            it the user query + the role list and expects a
            single-word reply naming the chosen role.
        system_prompt: override the routing prompt.

    Returns: object with `.invoke(query)` that returns
    `{"chosen_role": str, "result": Any}`.
    """
    if not workers:
        raise ValueError("workers must be non-empty")
    return _SupervisorAgent(
        workers=dict(workers),
        supervisor_model=supervisor_model,
        system_prompt=system_prompt,
    )


class _SupervisorAgent:
    def __init__(
        self,
        workers: Mapping[str, Any],
        supervisor_model: Any,
        system_prompt: str | None,
    ) -> None:
        self.workers = dict(workers)
        self.model = supervisor_model
        self.system_prompt = system_prompt or (
            "You are a router. Given a user query and a list of available "
            "agents, reply with EXACTLY one agent name from the list — no "
            "explanation, no punctuation. The agent that best matches the "
            "query handles it."
        )

    def invoke(self, query: str) -> dict[str, Any]:
        roles = list(self.workers.keys())
        list_block = "\n".join(f"- {r}" for r in roles)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Available agents:\n{list_block}\n\n"
                    f"User query: {query}\n\n"
                    "Reply with the single agent name."
                ),
            },
        ]
        decision = self.model.invoke(messages)
        text = decision.get("content", "") if isinstance(decision, dict) else str(decision)
        # Pick the role whose name appears in the model's reply.
        chosen = next((r for r in roles if r in text), roles[0])
        worker = self.workers[chosen]
        result = worker.invoke(query)
        return {"chosen_role": chosen, "result": result}


# ---- One-call map-reduce summariser ----


def summarize(
    text: str,
    *,
    model: Any,
    chunk_size: int = 4_000,
    chunk_overlap: int = 200,
    map_prompt: str | None = None,
    reduce_prompt: str | None = None,
) -> dict[str, Any]:
    """Map-reduce summarise a long document.

    Splits `text` into overlapping chunks, summarises each chunk
    independently (map), then summarises the concatenated chunk
    summaries (reduce). For documents that fit in the model's context
    window this is overkill — `summarize` exists for the long-doc
    case where one-shot summarisation truncates input.

    Args:
        text: source document.
        model: any litGraph ChatModel.
        chunk_size: max bytes per chunk before splitting.
        chunk_overlap: bytes of overlap between adjacent chunks (helps
            preserve cross-chunk references).
        map_prompt: override the per-chunk system prompt.
        reduce_prompt: override the final-merge system prompt.

    Returns: `{summary: str, chunk_summaries: list[str]}`.

    Example:

        from litgraph.providers import OpenAIChat
        from litgraph.recipes import summarize

        text = open("long.md").read()
        result = summarize(text, model=OpenAIChat(model="gpt-5"))
        print(result["summary"])
    """
    if not text or not text.strip():
        return {"summary": "", "chunk_summaries": []}
    if model is None:
        raise ValueError("summarize(model=...) is required")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    # Lazy-import the native splitter; recipes are importable on a
    # build without the .so but `summarize` itself needs it.
    try:
        from litgraph.splitters import RecursiveCharacterSplitter  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        # Fallback: dumb byte-window split. Good enough when the
        # native module isn't loaded (tests use this path).
        chunks = _dumb_chunk(text, chunk_size, chunk_overlap)
    else:
        sp = RecursiveCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = sp.split_text(text) if hasattr(sp, "split_text") else _dumb_chunk(text, chunk_size, chunk_overlap)

    map_sys = map_prompt or (
        "Summarise the following passage in 2-3 sentences. "
        "Preserve named entities, numbers, and dates. No fluff."
    )
    reduce_sys = reduce_prompt or (
        "You'll receive a sequence of summaries from contiguous "
        "passages of one document. Merge them into a single coherent "
        "summary, ~5-8 sentences. Preserve key entities + numbers."
    )

    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        msgs = [
            {"role": "system", "content": map_sys},
            {"role": "user", "content": f"Passage {i+1}/{len(chunks)}:\n\n{chunk}"},
        ]
        resp = model.invoke(msgs)
        chunk_summaries.append(_content_of(resp))

    if len(chunk_summaries) == 1:
        return {"summary": chunk_summaries[0], "chunk_summaries": chunk_summaries}

    joined = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(chunk_summaries))
    final_msgs = [
        {"role": "system", "content": reduce_sys},
        {"role": "user", "content": joined},
    ]
    final_resp = model.invoke(final_msgs)
    return {
        "summary": _content_of(final_resp),
        "chunk_summaries": chunk_summaries,
    }


def _content_of(resp: Any) -> str:
    if isinstance(resp, dict):
        return str(resp.get("content", ""))
    if hasattr(resp, "content"):
        return str(getattr(resp, "content"))
    return str(resp)


def _dumb_chunk(text: str, size: int, overlap: int) -> list[str]:
    """Pure-Python overlapping window splitter. Used as fallback when
    the native splitter isn't importable (test mode)."""
    if not text:
        return []
    out: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(pos + size, len(text))
        out.append(text[pos:end])
        if end >= len(text):
            break
        pos = end - overlap
    return out
