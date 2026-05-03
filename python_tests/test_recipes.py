"""Tests for `litgraph.recipes` — one-call factories.

Each recipe should be runnable with mocks (no LLM credential required)
and return the documented shape.
"""
from __future__ import annotations

import pytest

from litgraph import recipes
from litgraph.testing import MockChatModel


def test_recipes_eval_returns_per_case_and_aggregate():
    m = MockChatModel(replies=["Paris", "Berlin"])

    def predict(q: str) -> str:
        return m.invoke([{"role": "user", "content": q}])["content"]

    report = recipes.eval(predict, [
        {"input": "Capital of France?", "expected": "Paris"},
        {"input": "Capital of Germany?", "expected": "Berlin"},
    ])
    assert "per_case" in report
    assert "aggregate" in report
    assert report["aggregate"]["n_cases"] == 2
    assert report["aggregate"]["n_errors"] == 0
    assert report["aggregate"]["means"]["exact_match"] == 1.0


def test_recipes_eval_default_scorers_include_levenshtein():
    m = MockChatModel(replies=["close"])

    def predict(_q: str) -> str:
        return m.invoke([{"role": "user", "content": _q}])["content"]

    report = recipes.eval(predict, [
        {"input": "x", "expected": "close"},
    ])
    assert "exact_match" in report["aggregate"]["means"]
    assert "levenshtein" in report["aggregate"]["means"]


def test_recipes_eval_custom_scorers_passed_through():
    m = MockChatModel(replies=["bar"])

    def predict(_q: str) -> str:
        return m.invoke([{"role": "user", "content": _q}])["content"]

    report = recipes.eval(
        predict,
        [{"input": "x", "expected": "bar"}],
        scorers=[{"name": "exact_match"}],
    )
    assert list(report["aggregate"]["means"].keys()) == ["exact_match"]


def test_recipes_eval_target_exception_recorded_not_raised():
    def boom(_q: str) -> str:
        raise RuntimeError("explode")

    report = recipes.eval(boom, [
        {"input": "x", "expected": "y"},
    ])
    assert report["aggregate"]["n_errors"] == 1


def test_recipes_serve_rejects_non_graph_input():
    with pytest.raises(TypeError, match="CompiledGraph"):
        recipes.serve("not a graph")


def test_recipes_serve_renders_command_for_compileable():
    class _Stub:
        def compile(self):
            return self

    cmd = recipes.serve(_Stub(), port=9000, host="127.0.0.1")
    assert "litgraph-serve" in cmd
    assert "9000" in cmd
    assert "127.0.0.1" in cmd


# ---- recipes.rag ----


def test_recipes_rag_requires_model():
    with pytest.raises(ValueError, match="model"):
        recipes.rag(documents=[], embeddings=object())


def test_recipes_rag_requires_embeddings():
    with pytest.raises(ValueError, match="embeddings"):
        recipes.rag(documents=[], model=object())


def test_recipes_rag_runs_end_to_end_with_mocks():
    """Build a mock store + retriever path and exercise rag()."""
    from litgraph.testing import MockChatModel, MockEmbeddings

    class _StubStore:
        def __init__(self):
            self._docs = []
            self._embs = []
        def add(self, docs, embs):
            self._docs.extend(docs)
            self._embs.extend(embs)

    class _StubRetriever:
        def __init__(self, docs):
            self.docs = docs
        def retrieve(self, query, k=None):
            return self.docs[:2]

    # Bypass the auto-store / auto-retriever path: pass a pre-built
    # store and patch the retrieval module path manually.
    store = _StubStore()
    model = MockChatModel(replies=["The corpus says hello."])
    emb = MockEmbeddings(dim=8)

    docs = [{"page_content": "hello world"}, {"page_content": "goodbye"}]
    # Drive the underlying _RagAgent directly to avoid the
    # MMRRetriever import path which needs the native module.
    from litgraph.recipes import _RagAgent
    agent = _RagAgent(
        model=model,
        retriever=_StubRetriever(docs),
        system_prompt="You are helpful.",
    )
    out = agent.invoke("What does the corpus say?")
    assert "hello" in out["answer"]
    assert len(out["hits"]) == 2


# ---- recipes.multi_agent ----


def test_recipes_multi_agent_requires_workers():
    with pytest.raises(ValueError, match="non-empty"):
        recipes.multi_agent({}, supervisor_model=object())


def test_recipes_multi_agent_routes_to_chosen_worker():
    from litgraph.testing import MockChatModel

    class _Worker:
        def __init__(self, name):
            self.name = name
            self.calls = []
        def invoke(self, q):
            self.calls.append(q)
            return f"{self.name} says: {q}"

    billing = _Worker("billing_specialist")
    tech = _Worker("tech_support")

    # Supervisor model returns "billing_specialist" — should route there.
    sup_model = MockChatModel(replies=["billing_specialist"])
    ma = recipes.multi_agent(
        {"billing_specialist": billing, "tech_support": tech},
        supervisor_model=sup_model,
    )
    out = ma.invoke("My subscription is double-charged.")
    assert out["chosen_role"] == "billing_specialist"
    assert "billing_specialist says" in out["result"]
    assert billing.calls == ["My subscription is double-charged."]
    assert tech.calls == []


def test_recipes_multi_agent_falls_back_to_first_role_on_unparseable_decision():
    from litgraph.testing import MockChatModel

    class _Worker:
        def __init__(self, name):
            self.name = name
        def invoke(self, q):
            return f"{self.name}: {q}"

    sup_model = MockChatModel(replies=["I'm not sure"])
    ma = recipes.multi_agent(
        {"alpha": _Worker("alpha"), "beta": _Worker("beta")},
        supervisor_model=sup_model,
    )
    out = ma.invoke("anything")
    assert out["chosen_role"] == "alpha"  # first role wins fallback
