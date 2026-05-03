"""OpenAI-compat constructors for non-OpenAI providers."""
from __future__ import annotations
from typing import Any


def huggingface_tgi_chat(
    endpoint_url: str,
    model: str | None = ...,
    api_key: str | None = ...,
    **kwargs: Any,
) -> Any: ...


def nim_chat(
    model: str,
    api_key: str | None = ...,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    **kwargs: Any,
) -> Any: ...


def watsonx_chat(
    model: str,
    project_id: str,
    api_key: str | None = ...,
    region: str = "us-south",
    **kwargs: Any,
) -> Any: ...


def snowflake_cortex_chat(
    model: str,
    account: str,
    api_key: str | None = ...,
    **kwargs: Any,
) -> Any: ...


def databricks_chat(
    model: str,
    workspace_url: str,
    api_key: str | None = ...,
    **kwargs: Any,
) -> Any: ...


def replicate_chat(
    model: str,
    api_key: str | None = ...,
    base_url: str = "https://openai-proxy.replicate.com/v1",
    **kwargs: Any,
) -> Any: ...
