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


# ---- ToolBudgetMiddleware ----
#
# This is a `ToolMiddleware` (hooks tool dispatch), not an `AgentMiddleware`
# (hooks the chat model) like the classes above — it plugs into
# `ReactAgent(..., tool_middleware=[...])` rather than `MiddlewareChain`.
# The counting logic lives in Rust (`litgraph-agents::middleware::
# ToolBudgetMiddleware`); these tests only exercise the binding.

def test_tool_budget_middleware_starts_at_zero_calls():
    from litgraph.middleware import ToolBudgetMiddleware

    budget = ToolBudgetMiddleware(3)
    assert budget.calls() == 0


def test_tool_budget_middleware_repr_reports_calls():
    from litgraph.middleware import ToolBudgetMiddleware

    budget = ToolBudgetMiddleware(5)
    assert "ToolBudgetMiddleware" in repr(budget)


def test_tool_budget_middleware_reset_is_a_noop_when_unused():
    from litgraph.middleware import ToolBudgetMiddleware

    budget = ToolBudgetMiddleware(1)
    budget.reset()
    assert budget.calls() == 0


def _fake_llm_always_calls_tool(tool_name: str):
    """A local HTTP server standing in for an OpenAI-compatible endpoint.
    Always replies with a tool_call for `tool_name`, never a final answer —
    used to drive the agent past its tool budget deterministically, with no
    real LLM credential needed."""
    import http.server
    import json as _json

    class FakeLLM(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            payload = {
                "id": "1", "object": "chat.completion", "model": "m",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant", "content": None,
                        "tool_calls": [{
                            "id": "t1",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": "{}"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            out = _json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a, **kw):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    return srv


def test_tool_budget_middleware_caps_calls_through_react_agent():
    """End-to-end: the native middleware chain — not the pure-Python
    `litgraph.tool_hooks.ToolBudget` — actually gates tool dispatch inside
    ReactAgent's native tool-call loop."""
    import threading

    from litgraph.agents import ReactAgent
    from litgraph.middleware import ToolBudgetMiddleware
    from litgraph.providers import OpenAIChat
    from litgraph.tools import tool

    @tool
    def ping() -> str:
        """Return a fixed pong reply."""
        return "pong"

    budget = ToolBudgetMiddleware(2)
    srv = _fake_llm_always_calls_tool("ping")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        chat = OpenAIChat(api_key="k", model="gpt-x",
                          base_url=f"http://127.0.0.1:{port}/v1")
        agent = ReactAgent(
            model=chat, tools=[ping], max_iterations=4,
            tool_middleware=[budget],
        )
        out = agent.invoke("go")
        tool_msgs = [m for m in out["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 4
        # First two calls succeed (within the cap of 2)...
        assert [m["content"] for m in tool_msgs[:2]] == ["pong", "pong"]
        # ...every call past the cap is short-circuited by the middleware,
        # surfaced as a tool-result error rather than a fatal invoke() error.
        for m in tool_msgs[2:]:
            assert "tool budget exceeded" in m["content"]
        assert budget.calls() == 2
    finally:
        srv.shutdown()


def test_tool_budget_middleware_does_not_reset_between_invoke_calls():
    """The per-turn counter is *shared* Rust-side state (an `Arc` clone into
    the agent's middleware chain) — it does NOT reset automatically between
    separate `invoke()` calls on the same agent, mirroring the Rust
    `ToolBudgetMiddleware::reset()` contract exactly. Callers must reset()
    explicitly between turns if that's the behaviour they want."""
    import threading

    from litgraph.agents import ReactAgent
    from litgraph.middleware import ToolBudgetMiddleware
    from litgraph.providers import OpenAIChat
    from litgraph.tools import tool

    @tool
    def ping() -> str:
        """Return a fixed pong reply."""
        return "pong"

    budget = ToolBudgetMiddleware(1)
    srv = _fake_llm_always_calls_tool("ping")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        chat = OpenAIChat(api_key="k", model="gpt-x",
                          base_url=f"http://127.0.0.1:{port}/v1")
        agent = ReactAgent(
            model=chat, tools=[ping], max_iterations=2,
            tool_middleware=[budget],
        )
        agent.invoke("first turn")
        assert budget.calls() == 1

        # A second invoke() on the SAME agent inherits the already-exhausted
        # counter: the very first tool call of the new turn is rejected.
        out2 = agent.invoke("second turn")
        tool_msgs2 = [m for m in out2["messages"] if m["role"] == "tool"]
        assert "tool budget exceeded" in tool_msgs2[0]["content"]

        # Explicit reset() restores a fresh budget for the next turn.
        budget.reset()
        assert budget.calls() == 0
        out3 = agent.invoke("third turn")
        tool_msgs3 = [m for m in out3["messages"] if m["role"] == "tool"]
        assert tool_msgs3[0]["content"] == "pong"
    finally:
        srv.shutdown()
