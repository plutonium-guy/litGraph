"""Pattern for unit-testing your own agent with `litgraph.testing`.

Common need: write tests that don't burn API credits. The mocks in
`litgraph.testing` give deterministic chat / embedding / tool stubs
that mirror the real shapes.

Run:  python examples/mock_test_pattern.py
"""
from litgraph.testing import MockChatModel, MockEmbeddings, MockTool


def my_agent(model, search_tool, query: str) -> dict:
    """Toy agent: ask the model, optionally call a tool, return result."""
    out = model.invoke([{"role": "user", "content": query}])
    if "search" in out["content"].lower():
        hits = search_tool.invoke({"q": query})
        return {"answer": out["content"], "hits": hits}
    return {"answer": out["content"], "hits": []}


def test_agent_path_no_tool():
    m = MockChatModel(replies=["Direct answer: 42"])
    tool = MockTool("search", returns=[])
    out = my_agent(m, tool, "What's 6 * 7?")
    assert out["answer"] == "Direct answer: 42"
    assert out["hits"] == []
    assert tool.calls == []                # tool was never called
    assert len(m.calls) == 1               # model called once
    print("ok: no-tool path")


def test_agent_path_with_tool():
    m = MockChatModel(replies=["I should search the web."])
    tool = MockTool("search", returns=[{"url": "https://x", "title": "y"}])
    out = my_agent(m, tool, "Latest news?")
    assert "search" in out["answer"].lower()
    assert out["hits"] == [{"url": "https://x", "title": "y"}]
    assert tool.calls == [{"q": "Latest news?"}]
    print("ok: tool path")


def test_embeddings_deterministic():
    e = MockEmbeddings(dim=8)
    a = e.embed(["hello"])[0]
    b = e.embed(["hello"])[0]
    assert a == b                          # determinism — same in CI as local
    assert abs(sum(x * x for x in a) ** 0.5 - 1.0) < 1e-6   # L2-normalised
    print("ok: embeddings deterministic + normalised")


if __name__ == "__main__":
    test_agent_path_no_tool()
    test_agent_path_with_tool()
    test_embeddings_deterministic()
    print("\nall pass")
