"""Provider capability flags for the live integration suite.

Some tests exercise features that not every OpenAI-compatible endpoint
implements. Gate them on capability, not on a hardcoded provider name,
so the suite reports real coverage wherever it is pointed.

Resolved at import time from the same env vars `conftest.py` reads.
"""
from __future__ import annotations

import os

DEEPSEEK_HOST = "api.deepseek.com"


def _base_url() -> str:
    return os.environ.get("LITGRAPH_TEST_BASE_URL") or f"https://{DEEPSEEK_HOST}/v1"


def _is_deepseek() -> bool:
    return DEEPSEEK_HOST in _base_url()


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no")


#: `response_format={"type": "json_schema"}` (strict schema mode).
#: DeepSeek answers `400 "This response_format type is unavailable now"`.
#: Ollama, vLLM, OpenAI and LM Studio all implement it.
SUPPORTS_JSON_SCHEMA = _flag("LITGRAPH_TEST_JSON_SCHEMA", not _is_deepseek())

#: An embeddings model reachable on the same endpoint. DeepSeek exposes
#: none. Set `LITGRAPH_TEST_EMBED_MODEL` to enable embedding-backed tests.
EMBED_MODEL = os.environ.get("LITGRAPH_TEST_EMBED_MODEL", "")
SUPPORTS_EMBEDDINGS = bool(EMBED_MODEL)


def embed_base_url() -> str:
    """Endpoint serving the embeddings model. Defaults to the chat endpoint —
    with Ollama or vLLM one server serves both."""
    return os.environ.get("LITGRAPH_TEST_EMBED_BASE_URL") or _base_url()


def embed_dimensions() -> int:
    """Vector width of `EMBED_MODEL`. `OpenAIEmbeddings` requires it up front.
    Defaults to 768 (`nomic-embed-text`); override per model."""
    return int(os.environ.get("LITGRAPH_TEST_EMBED_DIMENSIONS", "768"))

NO_JSON_SCHEMA_REASON = (
    "endpoint does not support response_format=json_schema "
    "(set LITGRAPH_TEST_JSON_SCHEMA=1 to force)"
)
NO_EMBEDDINGS_REASON = (
    "no embeddings model configured (set LITGRAPH_TEST_EMBED_MODEL)"
)
