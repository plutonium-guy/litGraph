from typing import Any

from .providers import OpenAIChat, AnthropicChat, GeminiChat, BedrockChat, CohereChat
from .tools import FunctionTool

ChatProvider = OpenAIChat | AnthropicChat | GeminiChat | BedrockChat | CohereChat

class ReactAgent:
    def __init__(
        self,
        model: ChatProvider,
        tools: list[FunctionTool],
        system_prompt: str | None = None,
        max_iterations: int = 10,
    ) -> None: ...
    def invoke(self, user: str) -> dict[str, Any]: ...

class SupervisorAgent:
    def __init__(
        self,
        model: ChatProvider,
        workers: dict[str, ReactAgent],
        system_prompt: str | None = None,
        max_hops: int = 6,
    ) -> None: ...
    def invoke(self, user: str) -> dict[str, Any]: ...
    def worker_names(self) -> list[str]: ...
