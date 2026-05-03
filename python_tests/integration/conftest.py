"""pytest fixtures + CLI flag for live-API integration tests.

Every test in this folder is `@pytest.mark.integration`. They:

- Skip cleanly when `DEEPSEEK_API_KEY` env is unset.
- Skip when `--no-deepseek` is passed (CI opt-out).
- Use small `max_tokens` to keep cost low.
- Build their own `OpenAIChat` instance pointing at DeepSeek
  via the `deepseek_chat` fixture below.
"""
from __future__ import annotations

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-deepseek",
        action="store_true",
        default=False,
        help="Skip DeepSeek live-API integration tests even when "
        "DEEPSEEK_API_KEY is set.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--no-deepseek") and os.environ.get("DEEPSEEK_API_KEY"):
        return
    skip_marker = pytest.mark.skip(
        reason="DEEPSEEK_API_KEY not set or --no-deepseek passed"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def deepseek_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return key


@pytest.fixture
def deepseek_chat(deepseek_api_key: str):
    """A `OpenAIChat` configured for DeepSeek's OpenAI-compat
    endpoint. Per-test instance so middleware / wrappers don't
    leak across tests.

    Defaults: `deepseek-chat` model, `temperature=0`,
    `max_tokens=50` — keep cost / non-determinism low. Tests that
    need the reasoner override via `model="deepseek-reasoner"`.
    """
    from litgraph.providers import OpenAIChat
    return OpenAIChat(
        api_key=deepseek_api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )
