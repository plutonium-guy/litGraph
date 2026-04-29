"""LangChain 1.0-style middleware: before/after-model hooks composed in order.

Smoke tests verify Python class registration + chain composition. Behavioural
tests run on the Rust side (`crates/litgraph-core/src/middleware.rs`).
"""

import pytest

litgraph = pytest.importorskip("litgraph")
from litgraph.middleware import (  # noqa: E402
    LoggingMiddleware,
    MessageWindowMiddleware,
    MiddlewareChain,
    SystemPromptMiddleware,
)


def test_chain_starts_empty():
    chain = MiddlewareChain()
    assert len(chain) == 0
    assert chain.names() == []


def test_chain_append_and_names_in_order():
    chain = MiddlewareChain()
    chain.append(SystemPromptMiddleware("be terse"))
    chain.append(MessageWindowMiddleware(5))
    chain.append(LoggingMiddleware())
    assert len(chain) == 3
    assert chain.names() == ["system_prompt", "message_window", "logging"]


def test_chain_repr_includes_names():
    chain = MiddlewareChain()
    chain.append(LoggingMiddleware())
    chain.append(MessageWindowMiddleware(3))
    r = repr(chain)
    assert "logging" in r
    assert "message_window" in r


def test_message_window_rejects_zero_keep_last_silently():
    # Underlying impl clamps to >=1; this test just confirms construction
    # never raises for valid inputs.
    MessageWindowMiddleware(0)
    MessageWindowMiddleware(1)
    MessageWindowMiddleware(100)


def test_system_prompt_repr_truncates_long_input():
    long = "x" * 200
    r = repr(SystemPromptMiddleware(long))
    assert "SystemPromptMiddleware" in r
    # The implementation truncates to ~40 chars.
    assert len(r) < 100


def test_chain_composes_with_chat_model():
    """Chain must be passable to MiddlewareChat which is, in turn, accepted by
    ReactAgent. We can't run a real model here without an API key, so just
    verify the construction path."""
    from litgraph.middleware import MiddlewareChat
    from litgraph.providers import OpenAIChat

    chain = MiddlewareChain()
    chain.append(SystemPromptMiddleware("you are helpful"))
    chain.append(MessageWindowMiddleware(10))
    base = OpenAIChat(api_key="sk-fake", model="gpt-4o-mini")
    wrapped = MiddlewareChat(base, chain)
    assert "MiddlewareChat" in repr(wrapped)


def test_middleware_chat_plugs_into_react_agent():
    """MiddlewareChat must be acceptable wherever a ChatModel is."""
    from litgraph.agents import ReactAgent
    from litgraph.middleware import MiddlewareChat
    from litgraph.providers import OpenAIChat
    from litgraph.tools import CalculatorTool

    chain = MiddlewareChain()
    chain.append(SystemPromptMiddleware("math only"))
    base = OpenAIChat(api_key="sk-fake", model="gpt-4o-mini")
    wrapped = MiddlewareChat(base, chain)
    # Construction must succeed without making an HTTP call.
    ReactAgent(wrapped, [CalculatorTool()], max_iterations=1)
