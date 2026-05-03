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

__all__.extend([
    "entrypoint",
    "task",
    "Workflow",
    "coerce_one",
    "coerce_stream",
    "recipes",
    "testing",
    "tool_hooks",
    "streaming",
    "prompt_hub",
    "embeddings_extras",
    "providers_extras",
    "cache_extras",
])
