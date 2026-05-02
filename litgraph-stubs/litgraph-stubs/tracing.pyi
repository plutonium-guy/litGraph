"""OpenTelemetry / LangSmith bootstrappers. Each `init_*` returns a
guard that you must keep alive for the duration of the program;
dropping it drains pending spans before exit."""
from __future__ import annotations
from typing import Mapping


def init_langsmith(api_key: str, project_name: str = "litgraph") -> None: ...
def init_otlp(
    endpoint: str = "http://localhost:4317",
    service_name: str = "litgraph",
) -> None: ...
def init_otlp_http(
    endpoint: str,
    service_name: str,
    headers: Mapping[str, str],
) -> None: ...
def init_stdout(service_name: str = "litgraph") -> None: ...
def shutdown() -> None: ...
