"""Batteries-included agent construction, execution, tracing, and evaluation."""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping


__all__ = ["AgentHarness", "AgentRun", "create_agent"]


@dataclass(frozen=True)
class AgentRun:
    """Normalized result from one agent invocation."""

    run_id: str
    input: str
    output: str
    state: Any
    elapsed_ms: float
    success: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentHarness:
    """A small development harness around any object with ``invoke``.

    Use :meth:`build` for litGraph's batteries-included ReAct agent, or pass
    an existing agent to the constructor. Runs expose a stable result shape;
    streams, evaluations, JSONL traces, and event hooks use the same harness.
    """

    def __init__(
        self,
        agent: Any,
        *,
        trace_path: str | Path | None = None,
        on_event: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not callable(getattr(agent, "invoke", None)):
            raise TypeError("AgentHarness requires an agent with invoke(input)")
        self.agent = agent
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.on_event = on_event
        self.last_run: AgentRun | None = None
        self._trace_lock = threading.Lock()

    @classmethod
    def build(
        cls,
        model: Any,
        *,
        tools: Iterable[Any] = (),
        instructions: str | None = None,
        agents_md_path: str | Path | None = None,
        skills_dir: str | Path | None = None,
        max_iterations: int = 15,
        planning: bool = True,
        virtual_filesystem: bool = True,
        trace_path: str | Path | None = None,
        on_event: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> "AgentHarness":
        """Build a ReAct agent with planning and scratch storage enabled.

        The model is always explicit. Provider credentials may still use the
        provider's normal environment-variable support.
        """
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        from litgraph.deep_agent import create_deep_agent

        agent = create_deep_agent(
            model,
            tools=list(tools),
            system_prompt=instructions,
            agents_md_path=(
                str(agents_md_path) if agents_md_path is not None else None
            ),
            skills_dir=str(skills_dir) if skills_dir is not None else None,
            max_iterations=max_iterations,
            with_planning=planning,
            with_vfs=virtual_filesystem,
        )
        return cls(agent, trace_path=trace_path, on_event=on_event)

    def run(self, user_input: str, *, raise_errors: bool = True) -> AgentRun:
        """Invoke the agent and return a normalized :class:`AgentRun`."""
        run_id = uuid.uuid4().hex
        started = time.perf_counter()
        self._emit({"type": "run_start", "run_id": run_id, "input": user_input})
        try:
            state = self.agent.invoke(user_input)
            result = AgentRun(
                run_id=run_id,
                input=user_input,
                output=_extract_output(state),
                state=state,
                elapsed_ms=(time.perf_counter() - started) * 1_000,
                success=True,
                error=None,
            )
        except Exception as exc:
            result = AgentRun(
                run_id=run_id,
                input=user_input,
                output="",
                state=None,
                elapsed_ms=(time.perf_counter() - started) * 1_000,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            self.last_run = result
            self._emit({"type": "run_end", **result.to_dict()})
            if raise_errors:
                raise
            return result
        self.last_run = result
        self._emit({"type": "run_end", **result.to_dict()})
        return result

    def stream(self, user_input: str, *, tokens: bool = True) -> Iterator[dict[str, Any]]:
        """Yield native agent events while also forwarding them to tracing."""
        method_name = "stream_tokens" if tokens else "stream"
        method = getattr(self.agent, method_name, None)
        if not callable(method) and tokens:
            method = getattr(self.agent, "stream", None)
        if not callable(method):
            raise TypeError("agent does not expose stream() or stream_tokens()")

        run_id = uuid.uuid4().hex
        started = time.perf_counter()
        self._emit({"type": "run_start", "run_id": run_id, "input": user_input})
        try:
            for raw_event in method(user_input):
                event = dict(raw_event)
                self._emit({"type": "agent_event", "run_id": run_id, "event": event})
                yield event
        except Exception as exc:
            self._emit(
                {
                    "type": "run_error",
                    "run_id": run_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        self._emit(
            {
                "type": "run_end",
                "run_id": run_id,
                "input": user_input,
                "elapsed_ms": (time.perf_counter() - started) * 1_000,
                "success": True,
            }
        )

    def evaluate(
        self,
        cases: Iterable[Mapping[str, Any]],
        *,
        scorers: Iterable[Mapping[str, Any]] | None = None,
        max_parallel: int = 4,
    ) -> Any:
        """Evaluate this agent with litGraph's concurrent evaluation runner."""
        from litgraph.recipes import eval as run_eval

        return run_eval(
            lambda prompt: self.run(prompt).output,
            list(cases),
            scorers=scorers,
            max_parallel=max_parallel,
        )

    def _emit(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("timestamp", time.time())
        if self.on_event is not None:
            self.on_event(payload)
        if self.trace_path is None:
            return
        line = json.dumps(payload, default=_json_default, ensure_ascii=False)
        with self._trace_lock:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def create_agent(
    model: Any,
    *,
    tools: Iterable[Any] = (),
    instructions: str | None = None,
    agents_md_path: str | Path | None = None,
    skills_dir: str | Path | None = None,
    max_iterations: int = 15,
    planning: bool = True,
    virtual_filesystem: bool = True,
    trace_path: str | Path | None = None,
    on_event: Callable[[Mapping[str, Any]], None] | None = None,
) -> AgentHarness:
    """Create a batteries-included agent and return its development harness."""
    return AgentHarness.build(
        model,
        tools=tools,
        instructions=instructions,
        agents_md_path=agents_md_path,
        skills_dir=skills_dir,
        max_iterations=max_iterations,
        planning=planning,
        virtual_filesystem=virtual_filesystem,
        trace_path=trace_path,
        on_event=on_event,
    )


def _extract_output(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        for key in ("output", "answer", "content"):
            value = state.get(key)
            if value is not None:
                return _content_to_text(value)
        messages = state.get("messages")
        if isinstance(messages, (list, tuple)) and messages:
            last = messages[-1]
            if isinstance(last, Mapping):
                return _content_to_text(last.get("content", ""))
            return _content_to_text(last)
    return _content_to_text(state)


def _content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, Mapping) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return repr(value)
