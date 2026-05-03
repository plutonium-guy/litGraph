"""Multi-agent helpers: Swarm-style handoffs + BigTool selection.

Native agents (`litgraph.agents`) cover ReAct, Supervisor, Plan-Execute,
Debate, Critique-Revise, SelfConsistency. This module adds:

- [`SwarmAgent`] — handoff-topology multi-agent. Each agent declares
  the agents it can hand off to; control flows like a state machine.
  Mirrors `langgraph-swarm`'s shape, in pure Python on top of the
  existing `ReactAgent`.
- [`BigToolAgent`] — large-scale tool selection via embedding-based
  retrieval. Embed every tool's `(name, description)` once, retrieve
  top-k for the user's query, hand only those to a `ReactAgent`. Cuts
  prompt size + tool-confusion from 100s of tools.

Both are pure Python — they compose existing native agents under the
hood, so they get all the perf wins (Rust HTTP / SSE parse / tokenize)
without re-implementing the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


__all__ = ["SwarmAgent", "BigToolAgent", "Handoff"]


@dataclass(frozen=True)
class Handoff:
    """Marker returned from an agent's `invoke` to switch control to
    another agent in the swarm. The swarm router reads this and
    dispatches the next turn to `target` with the same conversation
    state. `payload` is optional context the target should see."""
    target: str
    payload: Mapping[str, Any] | None = None


class SwarmAgent:
    """Handoff-topology multi-agent.

    Construct with a dict of `{name: agent}` pairs and a starting
    `entry` agent. Each agent must implement `invoke(messages) ->
    {messages, ...}` (the native `ReactAgent` shape). To hand off,
    an agent's invoke must return a dict with a `handoff` key set to
    a [`Handoff`] instance.

    Args:
        agents: name → agent mapping.
        entry: name of the agent that handles the first turn.
        max_handoffs: cap on consecutive handoffs to avoid loops.
        on_handoff: optional callback fired on every handoff (for
            audit / tracing).

    Example:

        from litgraph.agents import ReactAgent
        from litgraph.agents_extras import SwarmAgent, Handoff

        triage = ReactAgent(model, [sentiment_tool], system_prompt="...")
        billing = ReactAgent(model, [refund_tool], system_prompt="...")
        tech = ReactAgent(model, [debug_tool], system_prompt="...")

        swarm = SwarmAgent(
            agents={"triage": triage, "billing": billing, "tech": tech},
            entry="triage",
        )
        out = swarm.invoke("My subscription was double-charged.")
    """

    def __init__(
        self,
        agents: Mapping[str, Any],
        entry: str,
        max_handoffs: int = 5,
        on_handoff: Callable[[str, str, Mapping[str, Any] | None], None] | None = None,
    ) -> None:
        if entry not in agents:
            raise ValueError(f"entry agent {entry!r} not in agents map")
        if max_handoffs <= 0:
            raise ValueError("max_handoffs must be positive")
        self.agents = dict(agents)
        self.entry = entry
        self.max_handoffs = max_handoffs
        self.on_handoff = on_handoff

    def invoke(self, user_input: str | Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        messages: list[Mapping[str, Any]]
        if isinstance(user_input, str):
            messages = [{"role": "user", "content": user_input}]
        else:
            messages = list(user_input)

        current = self.entry
        history: list[str] = [current]
        for _ in range(self.max_handoffs + 1):
            agent = self.agents[current]
            result = agent.invoke(messages)
            # Native ReactAgent returns dict with `messages` key; update.
            new_msgs = result.get("messages") if isinstance(result, dict) else None
            if new_msgs:
                messages = list(new_msgs)
            handoff = result.get("handoff") if isinstance(result, dict) else None
            if not isinstance(handoff, Handoff):
                return {
                    "messages": messages,
                    "handoff_chain": history,
                    "final_agent": current,
                }
            if handoff.target not in self.agents:
                raise ValueError(
                    f"agent {current!r} handed off to unknown target {handoff.target!r}"
                )
            if self.on_handoff:
                self.on_handoff(current, handoff.target, handoff.payload)
            current = handoff.target
            history.append(current)
        return {
            "messages": messages,
            "handoff_chain": history,
            "final_agent": current,
            "stopped": "max_handoffs_exceeded",
        }


class BigToolAgent:
    """Large-scale tool selection. Embed each tool's `(name +
    description)`, retrieve top-`k` by cosine similarity to the user's
    query, and only pass *those* to the inner agent.

    Cuts prompt size from O(N tools) to O(k) and reduces the model's
    "wrong tool, similar name" failure mode that surfaces beyond ~30
    tools.

    Args:
        agent_factory: callable `(tools) -> agent` that builds an
            agent given the selected tool list. Typically
            `lambda tools: ReactAgent(model, tools, system_prompt=...)`.
        tools: full tool catalogue.
        embeddings: any `litgraph.embeddings.*` (or duck-shape with
            `embed(texts) -> list[list[float]]`).
        k: how many tools to retrieve per turn.
    """

    def __init__(
        self,
        agent_factory: Callable[[list[Any]], Any],
        tools: Iterable[Any],
        embeddings: Any,
        k: int = 8,
    ) -> None:
        self._factory = agent_factory
        self._tools = list(tools)
        if not self._tools:
            raise ValueError("tools must be non-empty")
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        # Embed every tool's "<name>: <description>" once.
        descriptors = [
            f"{getattr(t, 'name', type(t).__name__)}: {getattr(t, 'description', '')}"
            for t in self._tools
        ]
        self._tool_vecs = embeddings.embed(descriptors)
        self._embeddings = embeddings

    def _select(self, query: str) -> list[Any]:
        import math
        q_vec = self._embeddings.embed_query(query) if hasattr(self._embeddings, "embed_query") else self._embeddings.embed([query])[0]
        # Cosine similarity (assume L2-normalised vectors → dot is fine,
        # but we normalise defensively here).
        def cos(a: list[float], b: list[float]) -> float:
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return sum(x * y for x, y in zip(a, b)) / (na * nb)
        scored = [
            (cos(list(q_vec), list(v)), tool)
            for v, tool in zip(self._tool_vecs, self._tools)
        ]
        scored.sort(key=lambda p: p[0], reverse=True)
        return [t for _, t in scored[: self.k]]

    def invoke(self, user_input: str) -> Any:
        selected = self._select(user_input)
        agent = self._factory(selected)
        return agent.invoke(user_input)
