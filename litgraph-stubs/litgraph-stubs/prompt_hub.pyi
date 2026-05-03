"""Local prompt registry + community hub URL."""
from __future__ import annotations
from typing import Iterable


HUB_URL: str


class Prompt:
    name: str
    template: str
    tags: tuple[str, ...]
    version: str
    description: str
    def render(self, **vars: object) -> str: ...


def register(
    name: str,
    template: str,
    *,
    tags: Iterable[str] = ...,
    version: str = "1",
    description: str = "",
    overwrite: bool = False,
) -> Prompt: ...


def get(name: str) -> Prompt: ...
def search(query: str) -> list[Prompt]: ...
def list_prompts() -> list[Prompt]: ...
def clear() -> None: ...
def fetch_from_url(url: str, *, register_as: str | None = ...) -> Prompt: ...
def load_directory(path: str) -> list[Prompt]: ...
