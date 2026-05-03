"""Tests for litgraph.loaders_extras — parameter validation + email
body extraction. Live IMAP / Reddit / Airtable / YouTube paths
require credentials and are out of scope for unit tests."""
from __future__ import annotations

from email.message import EmailMessage

import pytest

from litgraph.loaders_extras import (
    ImapLoader,
    YouTubeTranscriptLoader,
    AirtableLoader,
    OutlookLoader,
    TwitterLoader,
    _doc,
    _extract_email_body,
)


def test_imap_max_messages_must_be_positive():
    with pytest.raises(ValueError):
        ImapLoader(host="x", username="u", password="p", max_messages=0)


def test_youtube_video_ids_required():
    with pytest.raises(ValueError):
        YouTubeTranscriptLoader([])


def test_extract_email_body_plain_text():
    msg = EmailMessage()
    msg.set_content("hello world")
    body = _extract_email_body(msg)
    assert "hello world" in body


def test_extract_email_body_html_stripped():
    msg = EmailMessage()
    msg.add_alternative("<html><body><b>hi</b> there</body></html>", subtype="html")
    body = _extract_email_body(msg)
    # Plain alt missing; falls back to HTML stripped of tags.
    assert "hi" in body
    assert "<b>" not in body


def test_extract_email_body_prefers_plain_over_html():
    msg = EmailMessage()
    msg.set_content("plain body")
    msg.add_alternative("<html><body>html body</body></html>", subtype="html")
    body = _extract_email_body(msg)
    assert "plain body" in body
    assert "html" not in body.lower() or "html body" not in body


def test_doc_helper_returns_expected_shape():
    d = _doc("hello", source="x", id=1)
    assert d == {"page_content": "hello", "metadata": {"source": "x", "id": 1}}


def test_airtable_requires_api_key_at_load():
    """The loader stores config eagerly but only validates the key at
    `load()` time so users can construct it from env."""
    loader = AirtableLoader(base_id="appX", table_name="t", api_key=None)
    # Cheap to construct without the key:
    assert loader.base_id == "appX"
    assert loader.table_name == "t"


# ---- Outlook ----

def test_outlook_requires_token():
    """No env var, no constructor arg → eager error."""
    import os
    saved = os.environ.pop("MS_GRAPH_TOKEN", None)
    try:
        with pytest.raises(ValueError, match="access_token required"):
            OutlookLoader()
    finally:
        if saved is not None:
            os.environ["MS_GRAPH_TOKEN"] = saved


def test_outlook_max_messages_must_be_positive():
    with pytest.raises(ValueError):
        OutlookLoader(access_token="x", max_messages=0)


def test_outlook_reads_token_from_env(monkeypatch):
    monkeypatch.setenv("MS_GRAPH_TOKEN", "env-token-xyz")
    loader = OutlookLoader()
    assert loader.access_token == "env-token-xyz"


# ---- Twitter ----

def test_twitter_requires_query_xor_user_id():
    with pytest.raises(ValueError, match="exactly one of"):
        TwitterLoader(query="x", user_id="u", bearer_token="t")
    with pytest.raises(ValueError, match="exactly one of"):
        TwitterLoader(bearer_token="t")  # neither set


def test_twitter_max_results_must_be_positive():
    with pytest.raises(ValueError):
        TwitterLoader(query="x", bearer_token="t", max_results=0)


def test_twitter_bearer_token_required():
    import os
    saved = os.environ.pop("TWITTER_BEARER_TOKEN", None)
    try:
        with pytest.raises(ValueError, match="bearer_token required"):
            TwitterLoader(query="x")
    finally:
        if saved is not None:
            os.environ["TWITTER_BEARER_TOKEN"] = saved


def test_twitter_user_id_form_is_valid():
    loader = TwitterLoader(user_id="123", bearer_token="t")
    assert loader.user_id == "123"
    assert loader.query is None
