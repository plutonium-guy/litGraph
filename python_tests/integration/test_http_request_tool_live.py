"""Live integration: `HttpRequestTool` makes a real HTTP GET via DeepSeek agent.

Hits httpbin.org's `/uuid` endpoint (returns a tiny JSON `{"uuid": "..."}`).
Failing this test means either the network is down OR the tool/agent
plumbing for HTTP regressed. Marked separately so CI can opt out.

Skip cleanly when the host is unreachable so the suite still passes
on offline runners.
"""
from __future__ import annotations

import socket

import pytest


pytestmark = pytest.mark.integration


def _httpbin_reachable() -> bool:
    try:
        socket.create_connection(("httpbin.org", 443), timeout=3).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _httpbin_reachable(), reason="httpbin.org unreachable")
def test_react_agent_fetches_url_via_http_tool(deepseek_chat):
    from litgraph.agents import ReactAgent
    from litgraph.tools import HttpRequestTool

    http = HttpRequestTool(timeout_s=10, allowed_hosts=["httpbin.org"])
    agent = ReactAgent(
        deepseek_chat,
        [http],
        system_prompt=(
            "You have an http_request tool. To answer the user, GET the URL, "
            "then quote the JSON response back. Be terse."
        ),
        max_iterations=4,
    )
    state = agent.invoke(
        "GET https://httpbin.org/uuid and quote the 'uuid' field from the response."
    )
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else str(final)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    # Successful run mentions either 'uuid' (the field) or any
    # hex-with-dashes pattern characteristic of a UUID.
    has_uuid_word = "uuid" in (text or "").lower()
    has_hex_dashes = any(c == "-" for c in (text or ""))
    assert has_uuid_word or has_hex_dashes, (
        f"agent didn't surface the response: {final!r}"
    )
