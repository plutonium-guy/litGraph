"""Stubs for the `litgraph.serve` PyO3 submodule (iter 384)."""
from __future__ import annotations

from typing import Any


class ServeHandle:
    """Handle to a running litgraph-serve axum instance."""

    def address(self) -> str: ...
    def url(self) -> str: ...
    def model(self) -> str: ...
    def shutdown(self) -> None: ...


def spawn_chat(
    model: Any,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> ServeHandle: ...
