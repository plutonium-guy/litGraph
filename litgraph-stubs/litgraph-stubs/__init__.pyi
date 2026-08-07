from . import (
    agents as agents,
    agents_extras as agents_extras,
    cache as cache,
    cache_extras as cache_extras,
    compat as compat,
    deep_agent as deep_agent,
    embeddings as embeddings,
    embeddings_extras as embeddings_extras,
    evaluators as evaluators,
    graph as graph,
    harness as harness,
    lcel as lcel,
    loaders as loaders,
    loaders_extras as loaders_extras,
    mcp as mcp,
    memory as memory,
    memory_extras as memory_extras,
    middleware as middleware,
    observability as observability,
    parsers as parsers,
    prompt_hub as prompt_hub,
    prompts as prompts,
    providers as providers,
    providers_extras as providers_extras,
    recipes as recipes,
    retrieval as retrieval,
    serve as serve,
    splitters as splitters,
    splitters_extras as splitters_extras,
    store as store,
    stores_extras as stores_extras,
    streaming as streaming,
    stream_parts as stream_parts,
    testing as testing,
    tokenizers as tokenizers,
    tool_hooks as tool_hooks,
    tools as tools,
    tracing as tracing,
)
from .harness import AgentHarness, AgentRun, create_agent
from .stream_parts import StreamPart, stream_part

__all__: list[str]
__version__: str

def sum_as_string(a: int, b: int) -> str: ...


# ─── Typed mirror of ChatStreamEvent dict events (iter 379) ────────
#
# Re-exported at the top of the package from `_stream_part.py`; lives
# here in `__init__.pyi` because the runtime file is private (underscore
# prefix) and check_stubs walks only public submodules.

from dataclasses import dataclass as _dataclass
from typing import (
    AsyncIterable as _AsyncIterable,
    AsyncIterator as _AsyncIterator,
    Iterable as _Iterable,
    Iterator as _Iterator,
    Optional as _Optional,
    Union as _Union,
)


@_dataclass(frozen=True)
class Delta:
    text: str
    type: str = ...


@_dataclass(frozen=True)
class ToolCallDelta:
    index: int
    id: _Optional[str] = ...
    name: _Optional[str] = ...
    arguments_delta: _Optional[str] = ...
    type: str = ...


@_dataclass(frozen=True)
class Done:
    text: str
    finish_reason: str
    model: str
    usage: dict = ...
    type: str = ...


def parse_stream_part(event: dict) -> _Union[Delta, ToolCallDelta, Done]: ...


def parse_stream_parts(
    stream: _Iterable[dict],
) -> _Iterator[_Union[Delta, ToolCallDelta, Done]]: ...


async def aparse_stream_parts(
    stream: _AsyncIterable[dict],
) -> _AsyncIterator[_Union[Delta, ToolCallDelta, Done]]: ...
