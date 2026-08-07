"""litGraph — Rust-core LangChain/LangGraph alternative with Python bindings.

The native module is built by maturin (`crates/litgraph-py`) and re-exported
here. Pure-Python sugar (decorators, helpers) lives in sibling modules of
this package.
"""

# Native module — built by maturin, lives next to this file at runtime as
# `litgraph.litgraph`. Wrapped in try/except so the package can be
# imported in dev environments without a built wheel (pure-Python sugar
# like `functional.py` works without the native module). Production
# installs always have the native module present.
try:
    from .litgraph import *  # type: ignore[no-redef]  # noqa: F401,F403
    from . import litgraph as _native  # type: ignore[no-redef]  # noqa: F401

    __doc__ = _native.__doc__
    if hasattr(_native, "__all__"):
        __all__ = list(_native.__all__)
    else:
        __all__ = []
except ImportError as _native_err:  # pragma: no cover — only in dev w/o build
    import warnings as _warnings

    _warnings.warn(
        f"litgraph: native module not built ({_native_err}). "
        f"Pure-Python sugar (functional API) still works; "
        f"run `maturin develop` for the full native API.",
        ImportWarning,
        stacklevel=2,
    )
    __all__ = []

# Pure-Python sugar shipped alongside the native module.
from .functional import entrypoint, task, Workflow  # noqa: E402
from .coerce import coerce_one, coerce_stream  # noqa: E402
# Schema-aware StateGraph/CompiledGraph overlay (iter 378). Shadows the
# native classes of the same name so `litgraph.StateGraph(state_schema=...)`
# auto-coerces invoke/resume input + output. `state_schema=None` is the
# pre-iter-378 dict-in / dict-out behaviour. The original native classes
# remain reachable via `litgraph.litgraph.StateGraph` for callers that
# need the raw boundary.
from ._state_graph import StateGraph, CompiledGraph  # noqa: E402,F811
# Typed mirror of native ChatStreamEvent dict events (iter 379). Closes
# the StreamPart half of Tier-1 #7. Variants are dataclasses so `match`
# narrowing works without a runtime validator dep.
from ._stream_part import (  # noqa: E402
    Delta,
    Done,
    ToolCallDelta,
    aparse_stream_parts,
    parse_stream_part,
    parse_stream_parts,
)
from .harness import AgentHarness, AgentRun, create_agent  # noqa: E402
from .stream_parts import StreamPart, stream_part  # noqa: E402
from . import stream_parts  # noqa: E402,F401
from . import recipes  # noqa: E402,F401
from . import testing  # noqa: E402,F401
from . import tool_hooks  # noqa: E402,F401
from . import streaming  # noqa: E402,F401
from . import prompt_hub  # noqa: E402,F401
# `*_extras` modules import optional third-party libs lazily (inside
# methods), so importing the module itself is cheap. Adapter classes
# are only constructed when the user opts in.
from . import embeddings_extras  # noqa: E402,F401
from . import providers_extras  # noqa: E402,F401
from . import cache_extras  # noqa: E402,F401
from . import loaders_extras  # noqa: E402,F401
from . import splitters_extras  # noqa: E402,F401
from . import memory_extras  # noqa: E402,F401
from . import stores_extras  # noqa: E402,F401
from . import agents_extras  # noqa: E402,F401
from . import lcel  # noqa: E402,F401
from . import compat  # noqa: E402,F401

__all__.extend([
    "entrypoint",
    "task",
    "Workflow",
    "coerce_one",
    "coerce_stream",
    "StateGraph",
    "CompiledGraph",
    "Delta",
    "Done",
    "StreamPart",
    "ToolCallDelta",
    "aparse_stream_parts",
    "parse_stream_part",
    "parse_stream_parts",
    "AgentHarness",
    "AgentRun",
    "create_agent",
    "StreamPart",
    "stream_part",
    "harness",
    "recipes",
    "testing",
    "tool_hooks",
    "streaming",
    "stream_parts",
    "prompt_hub",
    "embeddings_extras",
    "providers_extras",
    "cache_extras",
    "loaders_extras",
    "splitters_extras",
    "memory_extras",
    "stores_extras",
    "agents_extras",
    "lcel",
    "compat",
])
