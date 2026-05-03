"""One-line constructors for chat models that speak OpenAI-compat
but live behind non-OpenAI URLs.

Native providers (OpenAI / Anthropic / Gemini / Bedrock / Cohere) are
in `litgraph.providers`. This module covers stragglers that route via
OpenAI-compat: HuggingFace TGI, NVIDIA NIM, IBM watsonx (preview),
Snowflake Cortex, Databricks Model Serving. Each helper picks sane
defaults for `base_url` and `api_key` env var so the user writes one
line and gets a working `ChatModel`.

Example:

    from litgraph.providers_extras import (
        huggingface_tgi_chat, nim_chat, snowflake_cortex_chat,
    )

    m = huggingface_tgi_chat(
        endpoint_url="https://my-tgi.company.com/v1",
        model="meta-llama/Meta-Llama-3-70B-Instruct",
    )
    out = m.invoke([{"role": "user", "content": "hi"}])
"""
from __future__ import annotations

import os
from typing import Any


__all__ = [
    "huggingface_tgi_chat",
    "nim_chat",
    "watsonx_chat",
    "snowflake_cortex_chat",
    "databricks_chat",
    "replicate_chat",
]


def _openai_compat(model: str, *, api_key: str, base_url: str, **kwargs: Any) -> Any:
    """Build an `OpenAIChat` against the given base URL. Lazy-imports
    so the module loads on builds without the native provider."""
    from litgraph.providers import OpenAIChat  # type: ignore[attr-defined]
    return OpenAIChat(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def huggingface_tgi_chat(
    endpoint_url: str,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """Chat against HuggingFace Text-Generation-Inference (TGI).
    TGI exposes `/v1/chat/completions` — drop a `/v1` suffix if your
    endpoint already includes it.

    Args:
        endpoint_url: TGI endpoint root (e.g. "https://x.huggingface.cloud").
            `/v1` is appended if missing.
        model: model name (TGI ignores in single-model deploys; pass
            anything if the endpoint pins one).
        api_key: HF token (or `HF_TOKEN` / `HF_API_TOKEN` env).
    """
    base = endpoint_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN", "")
    return _openai_compat(model or "tgi", api_key=key, base_url=base, **kwargs)


def nim_chat(
    model: str,
    api_key: str | None = None,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    **kwargs: Any,
) -> Any:
    """Chat against NVIDIA NIM (`integrate.api.nvidia.com`). NIM is
    OpenAI-compat by default. Set `NVIDIA_API_KEY`."""
    key = api_key or os.environ.get("NVIDIA_API_KEY", "")
    return _openai_compat(model, api_key=key, base_url=base_url, **kwargs)


def watsonx_chat(
    model: str,
    project_id: str,
    api_key: str | None = None,
    region: str = "us-south",
    **kwargs: Any,
) -> Any:
    """Chat against IBM watsonx.ai. Uses the OpenAI-compatible
    Foundation Models endpoint `/ml/v1/text/chat`. Set `WATSONX_API_KEY`.
    `project_id` is required by IBM auth."""
    key = api_key or os.environ.get("WATSONX_API_KEY", "")
    base = f"https://{region}.ml.cloud.ibm.com/ml/v1"
    return _openai_compat(
        model,
        api_key=key,
        base_url=base,
        # watsonx requires project_id as an extra header / body param;
        # OpenAIChat exposes `extra_headers` for this.
        extra_headers={"X-Watson-Project-Id": project_id},
        **kwargs,
    )


def snowflake_cortex_chat(
    model: str,
    account: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """Chat against Snowflake Cortex Inference REST. The `account`
    is the URL-prefix shown in the Snowflake console (e.g.
    `myorg-account123`). Set `SNOWFLAKE_PAT` (Programmatic Access
    Token) — these expire, rotate them via Snowflake."""
    key = api_key or os.environ.get("SNOWFLAKE_PAT", "")
    base = f"https://{account}.snowflakecomputing.com/api/v2/cortex/inference/v1"
    return _openai_compat(model, api_key=key, base_url=base, **kwargs)


def databricks_chat(
    model: str,
    workspace_url: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """Chat against Databricks Model Serving. `workspace_url` is the
    Databricks workspace root (e.g. https://dbc-xxx.cloud.databricks.com).
    Set `DATABRICKS_TOKEN`."""
    key = api_key or os.environ.get("DATABRICKS_TOKEN", "")
    base = workspace_url.rstrip("/") + f"/serving-endpoints/{model}/invocations"
    # Databricks's serving endpoint is *almost* OpenAI-compat; for the
    # /chat/completions equivalent we point at the v1 alias:
    if "/serving-endpoints/" in base and "/invocations" in base:
        base = workspace_url.rstrip("/") + "/serving-endpoints"
    return _openai_compat(model, api_key=key, base_url=base, **kwargs)


def replicate_chat(
    model: str,
    api_key: str | None = None,
    base_url: str = "https://openai-proxy.replicate.com/v1",
    **kwargs: Any,
) -> Any:
    """Chat against Replicate via its OpenAI-compat proxy. Set
    `REPLICATE_API_TOKEN`."""
    key = api_key or os.environ.get("REPLICATE_API_TOKEN", "")
    return _openai_compat(model, api_key=key, base_url=base_url, **kwargs)
