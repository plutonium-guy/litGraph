from typing import Any, Awaitable, Callable

class FunctionTool:
    name: str
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | str,
        func: Callable[[dict[str, Any]], Any | Awaitable[Any]],
    ) -> None: ...

class BraveSearchTool:
    name: str
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout_s: int = 20,
    ) -> None: ...

class TavilySearchTool:
    name: str
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout_s: int = 30,
        search_depth: str = "basic",
    ) -> None: ...

class DuckDuckGoSearchTool:
    name: str
    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: int = 20,
    ) -> None: ...

class CalculatorTool:
    name: str
    def __init__(self) -> None: ...

class HttpRequestTool:
    name: str
    def __init__(
        self,
        timeout_s: int = 20,
        allowed_methods: list[str] | None = None,
        allowed_hosts: list[str] | None = None,
    ) -> None: ...

class ReadFileTool:
    name: str
    def __init__(self, sandbox_root: str, max_bytes: int = 1_048_576) -> None: ...

class WriteFileTool:
    name: str
    def __init__(self, sandbox_root: str) -> None: ...

class ListDirectoryTool:
    name: str
    def __init__(self, sandbox_root: str) -> None: ...

class ShellTool:
    name: str
    def __init__(
        self,
        working_dir: str,
        allowed_commands: list[str],
        timeout_s: int = 30,
        max_output_bytes: int = 65_536,
    ) -> None: ...
