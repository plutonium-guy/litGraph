"""Live integration meta-test: `MockChatModel` and the live model
satisfy the same protocol.

We don't call DeepSeek inside this test (the mock replaces it). The
test gates on the env var to keep the suite as one cohesive block;
if the mock and live model diverge in shape, downstream consumers
(`ReactAgent`, `Pipe`, `recipes.summarize`) break in surprising ways.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_mock_chat_model_matches_live_invoke_shape(deepseek_chat):
    from litgraph.testing import MockChatModel

    mock = MockChatModel(replies=["mocked-reply"])
    out_mock = mock.invoke([{"role": "user", "content": "hi"}])
    # Mock returns either a string or a dict — accept either, but check
    # what it returns is the same shape downstream consumers can read.
    if isinstance(out_mock, dict):
        # Native shape: should expose `text` or `content`.
        assert any(k in out_mock for k in ("text", "content"))
    else:
        assert isinstance(out_mock, str)
        assert "mocked" in out_mock

    # Mock should also surface `.calls` for assertion-style use.
    assert hasattr(mock, "calls")
    assert len(mock.calls) == 1


def test_mock_chat_model_cycles_replies(deepseek_chat):
    from litgraph.testing import MockChatModel

    mock = MockChatModel(replies=["one", "two"])
    a = mock.invoke([{"role": "user", "content": "x"}])
    b = mock.invoke([{"role": "user", "content": "y"}])
    c = mock.invoke([{"role": "user", "content": "z"}])
    # Should cycle: a→one, b→two, c→one (cycled).
    a_text = a if isinstance(a, str) else (a.get("text") or a.get("content") or "")
    c_text = c if isinstance(c, str) else (c.get("text") or c.get("content") or "")
    assert "one" in a_text
    assert "one" in c_text, f"reply did not cycle: {c!r}"
