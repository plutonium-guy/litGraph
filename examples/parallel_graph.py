"""StateGraph parallel-fanout demo.

Three branches each "fetch" a slow result. Static edges from START fire all
three concurrently in one super-step; their updates merge through the default
`merge_append` reducer. The whole graph completes in roughly the time of the
*slowest* branch — not the sum.

Run:  python examples/parallel_graph.py
"""
import time
from litgraph.graph import StateGraph, START, END


def slow_a(_state):
    time.sleep(0.2)
    return {"items": [{"branch": "a", "value": 1}]}

def slow_b(_state):
    time.sleep(0.2)
    return {"items": [{"branch": "b", "value": 2}]}

def slow_c(_state):
    time.sleep(0.2)
    return {"items": [{"branch": "c", "value": 3}]}

def join(_state):
    return {}


g = StateGraph()
g.add_node("a", slow_a)
g.add_node("b", slow_b)
g.add_node("c", slow_c)
g.add_node("join", join)
g.add_edge(START, "a")
g.add_edge(START, "b")
g.add_edge(START, "c")
g.add_edge("a", "join")
g.add_edge("b", "join")
g.add_edge("c", "join")
g.add_edge("join", END)

compiled = g.compile()

t = time.time()
state = compiled.invoke({"items": []})
elapsed = time.time() - t

print(f"merged items: {state['items']}")
print(f"elapsed: {elapsed*1000:.0f} ms (≈ 200ms = 1 branch, NOT 600ms = sequential)")
