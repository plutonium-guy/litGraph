from typing import Any

class McpTool:
    name: str
    description: str

class McpClient:
    @staticmethod
    def connect_stdio(
        program: str,
        args: list[str] = ...,
        timeout_s: int = 30,
    ) -> "McpClient": ...
    def list_tools(self) -> list[dict[str, Any]]: ...
    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...
    def tools(self) -> list[McpTool]: ...
