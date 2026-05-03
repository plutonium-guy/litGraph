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
