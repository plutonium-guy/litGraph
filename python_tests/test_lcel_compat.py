"""Tests for litgraph.lcel + litgraph.compat — LangChain shim layer."""
from __future__ import annotations

import pytest

from litgraph.lcel import Pipe, parallel
from litgraph.compat import (
    AgentExecutor,
    RunnableLambda,
    RunnableParallel,
    RunnableBranch,
    RunnablePassthrough,
)


# ---- Pipe ----


def test_pipe_single_step_calls_function():
    p = Pipe(lambda x: x * 2)
    assert p(3) == 6


def test_pipe_chain_runs_in_order():
    chain = Pipe(lambda x: x + 1) | Pipe(lambda x: x * 2) | Pipe(lambda x: -x)
    assert chain(5) == -((5 + 1) * 2)


def test_pipe_invoke_alias():
    p = Pipe(lambda x: x + 1)
    assert p.invoke(10) == 11


def test_pipe_step_with_invoke_method():
    class M:
        def invoke(self, x):
            return x.upper()
    chain = Pipe(M())
    assert chain("hi") == "HI"


def test_pipe_step_with_call_only():
    class C:
        def __call__(self, x):
            return f"<{x}>"
    chain = Pipe(C())
    assert chain("x") == "<x>"


def test_pipe_step_neither_invoke_nor_callable_raises():
    chain = Pipe(42)  # bare int — not callable
    with pytest.raises(TypeError, match="not callable"):
        chain("x")


def test_pipe_combines_pipes_flat():
    a = Pipe(lambda x: x + 1)
    b = Pipe(lambda x: x * 2) | Pipe(lambda x: -x)
    chain = a | b
    assert chain(1) == -((1 + 1) * 2)


def test_parallel_runs_branches_in_order():
    p = parallel(lambda x: x + 1, lambda x: x * 2, lambda x: x - 1)
    assert p(10) == [11, 20, 9]


def test_pipe_repr_contains_step_names():
    chain = Pipe(lambda x: x).__or__(Pipe(lambda x: x))
    s = repr(chain)
    assert "Pipe(" in s


# ---- compat: RunnableLambda ----


def test_runnable_lambda_invoke():
    f = RunnableLambda(lambda x: x ** 2)
    assert f.invoke(5) == 25
    assert f(5) == 25


def test_runnable_lambda_rejects_non_callable():
    with pytest.raises(TypeError):
        RunnableLambda(42)  # type: ignore[arg-type]


def test_runnable_lambda_pipes_via_or():
    f = RunnableLambda(lambda x: x + 1)
    chain = f | RunnableLambda(lambda x: x * 2)
    assert chain(3) == 8


# ---- compat: RunnableParallel ----


def test_runnable_parallel_runs_named_branches():
    p = RunnableParallel({"plus": lambda x: x + 1, "double": lambda x: x * 2})
    out = p.invoke(5)
    assert out == {"plus": 6, "double": 10}


def test_runnable_parallel_kwarg_branches():
    p = RunnableParallel(plus=lambda x: x + 1, double=lambda x: x * 2)
    out = p.invoke(5)
    assert out == {"plus": 6, "double": 10}


def test_runnable_parallel_empty_rejected():
    with pytest.raises(ValueError):
        RunnableParallel()


# ---- compat: RunnableBranch ----


def test_runnable_branch_dispatches_first_match():
    rb = RunnableBranch(
        (lambda x: "hello" in x, lambda x: "greeting"),
        (lambda x: "bye" in x, lambda x: "farewell"),
        lambda x: "fallback",
    )
    assert rb.invoke("hello world") == "greeting"
    assert rb.invoke("bye now") == "farewell"
    assert rb.invoke("anything else") == "fallback"


def test_runnable_branch_no_match_no_default_raises():
    rb = RunnableBranch((lambda x: False, lambda x: "no"))
    with pytest.raises(RuntimeError, match="no predicate matched"):
        rb.invoke("x")


def test_runnable_passthrough_returns_input():
    pp = RunnablePassthrough()
    assert pp.invoke("x") == "x"
    assert pp({"k": 1}) == {"k": 1}


# ---- compat: AgentExecutor ----


class _ModelStub:
    def invoke(self, _msgs):
        return {"role": "assistant", "content": "ok"}


class _AgentWithLLM:
    """Stand-in for a LangChain agent — has `.llm` attribute."""
    llm = _ModelStub()


def test_agent_executor_forwards_to_invoke_when_no_tools():
    class A:
        def invoke(self, x):
            return f"echo:{x}"
    ae = AgentExecutor(agent=A())
    assert ae.invoke("hi") == "echo:hi"
    assert ae("hi") == "echo:hi"


def test_agent_executor_invoke_accepts_dict_input():
    class A:
        def invoke(self, x):
            return f"echo:{x}"
    ae = AgentExecutor(agent=A())
    assert ae.invoke({"input": "hello"}) == "echo:hello"


def test_agent_executor_no_impl_raises():
    ae = AgentExecutor(agent=None)
    with pytest.raises(RuntimeError):
        ae.invoke("hi")


def test_agent_executor_from_factory_method():
    class A:
        def invoke(self, x):
            return f"X:{x}"
    ae = AgentExecutor.from_agent_and_tools(agent=A(), tools=[])
    # tools=[] without a model → falls through to forwarding `agent.invoke`
    # We at least exercise the constructor.
    assert ae is not None
