"""Live integration: HtmlHeaderSplitter + DuckDuckGo + WebFetchTool.

- `HtmlHeaderSplitter` walks an HTML doc into per-heading chunks
- `DuckDuckGoSearchTool` (no key) called via DeepSeek `ReactAgent`
- `WebFetchTool` (no key) does HTML strip on a known-good static URL

Network-gated tests skip cleanly offline.
"""
from __future__ import annotations

import socket

import pytest


pytestmark = pytest.mark.integration


_HTML = """
<!doctype html>
<html><body>
<h1>Top</h1>
<p>intro under top</p>
<h2>Sub A</h2>
<p>content of sub A</p>
<h2>Sub B</h2>
<p>content of sub B</p>
</body></html>
"""


def _online() -> bool:
    try:
        socket.create_connection(("duckduckgo.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


def test_html_header_splitter_walks_headings(deepseek_chat):
    from litgraph.splitters import HtmlHeaderSplitter

    splitter = HtmlHeaderSplitter(max_depth=2)
    chunks = splitter.split_text(_HTML)
    assert chunks, "html splitter returned no chunks"


@pytest.mark.skipif(not _online(), reason="duckduckgo.com unreachable")
def test_duckduckgo_search_tool_via_agent(deepseek_chat):
    from litgraph.agents import ReactAgent
    from litgraph.tools import DuckDuckGoSearchTool

    ddg = DuckDuckGoSearchTool(timeout_s=10)
    agent = ReactAgent(
        deepseek_chat,
        [ddg],
        system_prompt=(
            "You have a duckduckgo_search tool. Use it ONCE to answer the "
            "user's factual question, then summarise in one sentence."
        ),
        max_iterations=4,
    )
    state = agent.invoke("What is the capital of France? Use duckduckgo to confirm.")
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else str(final)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    assert "Paris" in (text or ""), f"agent missed Paris: {final!r}"


@pytest.mark.skipif(not _online(), reason="example.com unreachable")
def test_web_fetch_tool_via_agent(deepseek_chat):
    from litgraph.agents import ReactAgent
    from litgraph.tools import WebFetchTool

    fetcher = WebFetchTool(timeout_s=10, default_max_chars=2000)
    agent = ReactAgent(
        deepseek_chat,
        [fetcher],
        system_prompt=(
            "You have a web_fetch tool that returns clean-text from a URL. "
            "Fetch the URL, then summarise the content in one sentence."
        ),
        max_iterations=4,
    )
    state = agent.invoke(
        "Fetch https://example.com and tell me the page's main title."
    )
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else str(final)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    # example.com's content mentions "Example Domain" — accept either word.
    text_lower = (text or "").lower()
    assert "example" in text_lower, f"agent didn't surface page content: {final!r}"
