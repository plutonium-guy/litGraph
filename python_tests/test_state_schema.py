"""Implicit Pydantic-style state coercion on native StateGraph."""
from __future__ import annotations

from litgraph.graph import END, StateGraph


class TypedState:
    def __init__(self, count: int, label: str = "start") -> None:
        self.count = int(count)
        self.label = label

    @classmethod
    def model_validate(cls, value):
        if isinstance(value, cls):
            return value
        return cls(**value)

    def model_dump(self):
        return {"count": self.count, "label": self.label}


def test_state_schema_coerces_node_input_and_final_output():
    seen = []
    graph = StateGraph(state_schema=TypedState)

    def increment(state):
        seen.append(state)
        return {"count": state.count + 1, "label": "done"}

    graph.add_node("increment", increment)
    graph.set_entry("increment")
    graph.add_edge("increment", END)

    result = graph.compile().invoke({"count": "41"})

    assert isinstance(seen[0], TypedState)
    assert isinstance(result, TypedState)
    assert result.count == 42
    assert result.label == "done"


def test_state_schema_accepts_model_instance_as_initial_state():
    graph = StateGraph(state_schema=TypedState)
    graph.add_node("identity", lambda state: state)
    graph.set_entry("identity")
    graph.add_edge("identity", END)

    result = graph.compile().invoke(TypedState(count=7, label="typed"))

    assert isinstance(result, TypedState)
    assert result.count == 7
    assert result.label == "typed"


def test_dict_state_remains_backwards_compatible_without_schema():
    graph = StateGraph()
    graph.add_node("increment", lambda state: {"count": state["count"] + 1})
    graph.set_entry("increment")
    graph.add_edge("increment", END)

    assert graph.compile().invoke({"count": 1}) == {"count": 2}


def test_dict_state_schema_is_an_explicit_no_op():
    graph = StateGraph(state_schema=dict)
    graph.add_node("increment", lambda state: {"count": state["count"] + 1})
    graph.set_entry("increment")
    graph.add_edge("increment", END)

    assert graph.compile().invoke({"count": 1}) == {"count": 2}
