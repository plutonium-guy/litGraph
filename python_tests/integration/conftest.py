"""pytest fixtures + CLI flag for live-API integration tests.

Every test in this folder is `@pytest.mark.integration`. They:

- Skip cleanly when no live endpoint is configured.
- Skip when `--no-deepseek` is passed (CI opt-out).
- Use small `max_tokens` to keep cost low.
- Build their own `OpenAIChat` instance via the `deepseek_chat`
  fixture below.

The suite targets any OpenAI-compatible `/chat/completions` endpoint.
DeepSeek stays the default; point it elsewhere (Ollama, vLLM, LM
Studio, Together, ...) with:

    LITGRAPH_TEST_BASE_URL=http://localhost:11434/v1
    LITGRAPH_TEST_MODEL=qwen3:30b-a3b
    LITGRAPH_TEST_API_KEY=ollama        # optional
    LITGRAPH_TEST_TIMEOUT_S=300         # optional; local models are slower

When `LITGRAPH_TEST_BASE_URL` is set, `DEEPSEEK_API_KEY` is not
required -- local servers do not authenticate.
"""
from __future__ import annotations

import os

import pytest

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"


def _base_url() -> str:
    return os.environ.get("LITGRAPH_TEST_BASE_URL") or DEEPSEEK_BASE_URL


def _model() -> str:
    return os.environ.get("LITGRAPH_TEST_MODEL") or DEEPSEEK_MODEL


def _api_key():
    """Key for the configured endpoint.

    Explicit `LITGRAPH_TEST_API_KEY` wins. Against a custom base URL a
    local server needs no real key, so fall back to a placeholder.
    Against DeepSeek the real `DEEPSEEK_API_KEY` is mandatory.
    """
    explicit = os.environ.get("LITGRAPH_TEST_API_KEY")
    if explicit:
        return explicit
    if os.environ.get("LITGRAPH_TEST_BASE_URL"):
        return os.environ.get("DEEPSEEK_API_KEY") or "ollama"
    return os.environ.get("DEEPSEEK_API_KEY")


def pytest_addoption(parser):
    parser.addoption(
        "--no-deepseek",
        action="store_true",
        default=False,
        help="Skip live-API integration tests even when a live "
        "endpoint is configured.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--no-deepseek") and _api_key():
        return
    skip_marker = pytest.mark.skip(
        reason="no live endpoint configured (set DEEPSEEK_API_KEY or "
        "LITGRAPH_TEST_BASE_URL) or --no-deepseek passed"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def deepseek_api_key() -> str:
    key = _api_key()
    if not key:
        pytest.skip("no live endpoint configured")
    return key


@pytest.fixture(scope="session")
def live_model() -> str:
    """Model id the live endpoint is configured with.

    Tests that assert on the echoed model name, or that build a
    price table keyed by model, must use this instead of hardcoding
    `deepseek-chat` — the suite runs against other providers too.
    """
    return _model()


@pytest.fixture
def deepseek_chat(deepseek_api_key: str):
    """An `OpenAIChat` pointed at the configured OpenAI-compat
    endpoint. Per-test instance so middleware / wrappers don't
    leak across tests.

    Defaults: DeepSeek `deepseek-chat`. Override endpoint and model
    with `LITGRAPH_TEST_BASE_URL` / `LITGRAPH_TEST_MODEL`. Tests keep
    `temperature=0` and small `max_tokens` at the call site to hold
    cost / non-determinism down.
    """
    from litgraph.providers import OpenAIChat
    return OpenAIChat(
        api_key=deepseek_api_key,
        model=_model(),
        base_url=_base_url(),
        timeout_s=int(os.environ.get("LITGRAPH_TEST_TIMEOUT_S", "120")),
    )
