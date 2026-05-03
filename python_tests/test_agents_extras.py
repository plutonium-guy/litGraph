"""Tests for litgraph.agents_extras — Swarm + BigTool helpers."""
from __future__ import annotations

import pytest

from litgraph.agents_extras import SwarmAgent, BigToolAgent, Handoff
from litgraph.testing import MockEmbeddings, MockTool


# ---- Fake agent ----


class _FakeAgent:
    """Minimal agent matching ReactAgent's `invoke` shape."""

    def __init__(self, name: str, reply: str = "", handoff: Handoff | None = None):
        self.name = name
        self._reply = reply
        self._handoff = handoff
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        out = {
            "messages": list(messages) + [{"role": "assistant", "content": self._reply}],
        }
        if self._handoff is not None:
            out["handoff"] = self._handoff
        return out


# ---- SwarmAgent ----


def test_swarm_entry_must_exist_in_agents():
    a = _FakeAgent("alpha")
    with pytest.raises(ValueError, match="not in agents"):
        SwarmAgent({"alpha": a}, entry="beta")


def test_swarm_max_handoffs_must_be_positive():
    a = _FakeAgent("alpha")
    with pytest.raises(ValueError):
        SwarmAgent({"alpha": a}, entry="alpha", max_handoffs=0)


def test_swarm_no_handoff_returns_first_agent_result():
    alpha = _FakeAgent("alpha", reply="done")
    swarm = SwarmAgent({"alpha": alpha}, entry="alpha")
    out = swarm.invoke("hi")
    assert out["final_agent"] == "alpha"
    assert out["handoff_chain"] == ["alpha"]
    assert out["messages"][-1]["content"] == "done"


def test_swarm_handoff_routes_to_target():
    beta = _FakeAgent("beta", reply="from beta")
    alpha = _FakeAgent("alpha", reply="alpha here", handoff=Handoff("beta"))
    swarm = SwarmAgent({"alpha": alpha, "beta": beta}, entry="alpha")
    out = swarm.invoke("hi")
    assert out["final_agent"] == "beta"
    assert out["handoff_chain"] == ["alpha", "beta"]


def test_swarm_unknown_handoff_target_raises():
    alpha = _FakeAgent("alpha", reply="x", handoff=Handoff("ghost"))
    swarm = SwarmAgent({"alpha": alpha}, entry="alpha")
    with pytest.raises(ValueError, match="unknown target"):
        swarm.invoke("hi")


def test_swarm_max_handoffs_caps_loops():
    # Two agents handing off to each other indefinitely.
    alpha = _FakeAgent("alpha", reply="a", handoff=Handoff("beta"))
    beta = _FakeAgent("beta", reply="b", handoff=Handoff("alpha"))
    swarm = SwarmAgent({"alpha": alpha, "beta": beta}, entry="alpha", max_handoffs=3)
    out = swarm.invoke("hi")
    assert out.get("stopped") == "max_handoffs_exceeded"
    # Loop runs max_handoffs+1 = 4 iterations; chain has 5 entries
    # (entry + 4 follow-ons).
    assert len(out["handoff_chain"]) == 5


def test_swarm_on_handoff_callback_fires():
    seen: list = []
    beta = _FakeAgent("beta", reply="ok")
    alpha = _FakeAgent("alpha", reply="x", handoff=Handoff("beta", payload={"reason": "billing"}))
    swarm = SwarmAgent(
        {"alpha": alpha, "beta": beta},
        entry="alpha",
        on_handoff=lambda f, t, p: seen.append((f, t, p)),
    )
    swarm.invoke("hi")
    assert seen == [("alpha", "beta", {"reason": "billing"})]


# ---- BigToolAgent ----


def _factory_capturing(captured: list[list]) -> object:
    def f(tools):
        captured.append(list(tools))
        return _FakeAgent("inner", reply="ok")
    return f


def test_bigtool_empty_tool_list_rejected():
    emb = MockEmbeddings(dim=8)
    with pytest.raises(ValueError):
        BigToolAgent(lambda tools: _FakeAgent("x"), tools=[], embeddings=emb)


def test_bigtool_zero_k_rejected():
    emb = MockEmbeddings(dim=8)
    t = MockTool("a", returns=1)
    with pytest.raises(ValueError):
        BigToolAgent(lambda tools: _FakeAgent("x"), tools=[t], embeddings=emb, k=0)


def test_bigtool_selects_k_tools_per_invoke():
    emb = MockEmbeddings(dim=8)
    tools = [MockTool(f"tool_{i}", returns=i, description=f"does thing {i}") for i in range(20)]
    captured: list[list] = []
    big = BigToolAgent(_factory_capturing(captured), tools=tools, embeddings=emb, k=4)
    big.invoke("any query")
    assert len(captured) == 1
    assert len(captured[0]) == 4


def test_bigtool_factory_receives_only_subset():
    emb = MockEmbeddings(dim=8)
    tools = [MockTool(f"t{i}", returns=i) for i in range(10)]
    captured: list[list] = []
    big = BigToolAgent(_factory_capturing(captured), tools=tools, embeddings=emb, k=3)
    big.invoke("query")
    selected = captured[0]
    assert all(t in tools for t in selected)
    assert len(selected) == 3


def test_handoff_dataclass_immutable():
    h = Handoff("target", {"k": 1})
    with pytest.raises((AttributeError, TypeError)):
        h.target = "x"  # type: ignore[misc]
