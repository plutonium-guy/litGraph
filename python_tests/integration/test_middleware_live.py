"""Live integration: `MiddlewareChat` wrapping DeepSeek inside a `ReactAgent`.

`MiddlewareChat` is an opaque chat-protocol wrapper — it does NOT expose
`.invoke()` on the Python surface. It plugs into `ReactAgent` /
`SupervisorAgent` (Rust-side chat protocol). To exercise the wrapped
behaviour we drive it through `ReactAgent`.

Tested middleware:
- `SystemPromptMiddleware` — injects a system prompt
- `LoggingMiddleware` — observability tap (no behaviour change)
- `MessageWindowMiddleware` — truncates to last N messages
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_middleware_chain_construction():
    """Smoke test the fluent builder. `with_(...)` returns self;
    `names()` lists the registered middleware in order."""
    from litgraph.middleware import (
        LoggingMiddleware,
        MessageWindowMiddleware,
        MiddlewareChain,
        SystemPromptMiddleware,
    )

    chain = (
        MiddlewareChain()
        .with_(SystemPromptMiddleware("Be terse."))
        .with_(LoggingMiddleware())
        .with_(MessageWindowMiddleware(8))
    )
    names = chain.names()
    assert len(names) == 3
    assert len(chain) == 3


def test_middleware_chat_in_react_agent(deepseek_chat):
    """Wrap DeepSeek with a system-prompt middleware, drive through ReactAgent
    to verify the wrapped model satisfies the agent loop's chat protocol."""
    from litgraph.agents import ReactAgent
    from litgraph.middleware import (
        MiddlewareChain,
        MiddlewareChat,
        SystemPromptMiddleware,
    )

    chain = MiddlewareChain().with_(
        SystemPromptMiddleware("Reply with exactly the word: pong")
    )
    wrapped = MiddlewareChat(deepseek_chat, chain)
    agent = ReactAgent(wrapped, [], max_iterations=2)
    state = agent.invoke("ping")
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else ""
    assert text.strip(), f"agent produced no text: {final!r}"

