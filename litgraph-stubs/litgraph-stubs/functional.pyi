"""Type stubs for `litgraph.functional` — `@entrypoint` + `@task`."""

from typing import Any, AsyncIterator, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


def task(fn: Callable[..., Any]) -> Callable[..., Any]: ...


class Workflow:
    __name__: str
    __doc__: Optional[str]
    __wrapped__: Callable[..., Any]

    @property
    def checkpointer(self) -> Optional[Any]: ...
    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any: ...
    def invoke(self, *args: Any, **kwargs: Any) -> Any: ...
    def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[dict]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Awaitable[Any]: ...


def entrypoint(
    checkpointer: Optional[Any] = ...,
) -> Callable[[Callable[..., Any]], Workflow]: ...
