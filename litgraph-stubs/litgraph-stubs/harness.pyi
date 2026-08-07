from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping


class AgentRun:
    run_id: str
    input: str
    output: str
    state: Any
    elapsed_ms: float
    success: bool
    error: str | None
    def to_dict(self) -> dict[str, Any]: ...


class AgentHarness:
    agent: Any
    trace_path: Path | None
    on_event: Callable[[Mapping[str, Any]], None] | None
    last_run: AgentRun | None

    def __init__(
        self,
        agent: Any,
        *,
        trace_path: str | Path | None = ...,
        on_event: Callable[[Mapping[str, Any]], None] | None = ...,
    ) -> None: ...
    @classmethod
    def build(
        cls,
        model: Any,
        *,
        tools: Iterable[Any] = ...,
        instructions: str | None = ...,
        agents_md_path: str | Path | None = ...,
        skills_dir: str | Path | None = ...,
        max_iterations: int = ...,
        planning: bool = ...,
        virtual_filesystem: bool = ...,
        trace_path: str | Path | None = ...,
        on_event: Callable[[Mapping[str, Any]], None] | None = ...,
    ) -> "AgentHarness": ...
    def run(self, user_input: str, *, raise_errors: bool = ...) -> AgentRun: ...
    def stream(self, user_input: str, *, tokens: bool = ...) -> Iterator[dict[str, Any]]: ...
    def evaluate(
        self,
        cases: Iterable[Mapping[str, Any]],
        *,
        scorers: Iterable[Mapping[str, Any]] | None = ...,
        max_parallel: int = ...,
    ) -> Any: ...


def create_agent(
    model: Any,
    *,
    tools: Iterable[Any] = ...,
    instructions: str | None = ...,
    agents_md_path: str | Path | None = ...,
    skills_dir: str | Path | None = ...,
    max_iterations: int = ...,
    planning: bool = ...,
    virtual_filesystem: bool = ...,
    trace_path: str | Path | None = ...,
    on_event: Callable[[Mapping[str, Any]], None] | None = ...,
) -> AgentHarness: ...
