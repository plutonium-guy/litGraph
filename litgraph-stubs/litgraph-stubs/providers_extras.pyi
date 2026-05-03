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


def mistralrs_chat(
    model: str,
    base_url: str = "http://127.0.0.1:1234/v1",
    api_key: str = "no-key",
    **kwargs: Any,
) -> Any: ...


def vllm_chat(
    model: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "no-key",
    **kwargs: Any,
) -> Any: ...


def llamacpp_chat(
    model: str = "llama",
    base_url: str = "http://127.0.0.1:8080/v1",
    api_key: str = "no-key",
    **kwargs: Any,
) -> Any: ...
