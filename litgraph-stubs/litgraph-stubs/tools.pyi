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

def tool(
    func: Callable[..., Any],
    name: str | None = None,
) -> FunctionTool: ...

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

class SqliteQueryTool:
    name: str
    def __init__(
        self,
        db_path: str,
        allowed_tables: list[str],
        read_only: bool = True,
        max_rows: int = 1000,
        max_output_bytes: int = 262_144,
        allow_pragma: bool = False,
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

class PlanningTool:
    name: str
    def __init__(self) -> None: ...
    def snapshot(self) -> list[dict[str, Any]]: ...
    def clear(self) -> int: ...

class VirtualFilesystemTool:
    name: str
    def __init__(self, max_total_bytes: int = 0) -> None: ...
    def snapshot(self) -> dict[str, str]: ...
    def total_bytes(self) -> int: ...

class SubagentTool:
    name: str
    def __init__(self, name: str, description: str, agent: Any) -> None: ...
