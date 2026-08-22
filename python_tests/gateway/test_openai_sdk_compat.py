"""Run openai-python unchanged against a live Ollama-backed gateway."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

GATEWAY_URL = os.environ.get("LITGRAPH_GATEWAY_URL")
GATEWAY_KEY = os.environ.get("LITGRAPH_GATEWAY_KEY", "")
GATEWAY_MODEL = os.environ.get("LITGRAPH_GATEWAY_MODEL", "ollama")


@pytest.mark.skipif(not GATEWAY_URL, reason="LITGRAPH_GATEWAY_URL not set")
def test_openai_sdk_non_streaming():
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=GATEWAY_URL, api_key=GATEWAY_KEY)
    response = client.chat.completions.create(
        model=GATEWAY_MODEL,
        messages=[{"role": "user", "content": "Reply with just: ok"}],
        max_tokens=10,
    )
    assert response.choices[0].message.content
    assert response.usage.total_tokens > 0


@pytest.mark.skipif(not GATEWAY_URL, reason="LITGRAPH_GATEWAY_URL not set")
def test_openai_sdk_streaming():
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=GATEWAY_URL, api_key=GATEWAY_KEY)
    chunks = list(
        client.chat.completions.create(
            model=GATEWAY_MODEL,
            messages=[{"role": "user", "content": "Count to three."}],
            max_tokens=20,
            stream=True,
        )
    )
    assert chunks
    assert any(chunk.choices and chunk.choices[0].delta.content for chunk in chunks)


@pytest.mark.skipif(not GATEWAY_URL, reason="LITGRAPH_GATEWAY_URL not set")
def test_openai_sdk_rejects_bad_key():
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=GATEWAY_URL, api_key="lg-sk-deadbeef.bogus")
    with pytest.raises(openai.AuthenticationError):
        client.chat.completions.create(
            model=GATEWAY_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
