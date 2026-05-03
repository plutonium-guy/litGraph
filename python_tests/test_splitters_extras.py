"""Tests for litgraph.splitters_extras — pure-Python aggregator paths.

The NLTK / SpaCy adapters lazy-import; the tests cover the parts that
work without those libs (the aggregator + parameter validation)."""
from __future__ import annotations

import pytest

from litgraph.splitters_extras import NltkSentenceSplitter, _aggregate


def test_aggregate_no_chunking_returns_each_sentence_joined():
    out = _aggregate(["a.", "b.", "c."], cap=100, overlap=0)
    assert out == ["a. b. c."]


def test_aggregate_splits_when_cap_exceeded():
    out = _aggregate(["aaaaa", "bbbbb", "ccccc"], cap=8, overlap=0)
    # Each sentence is 5 bytes + 1 separator = 6 in the running budget.
    # First chunk gets one sentence; second gets next; etc.
    assert len(out) >= 2


def test_aggregate_respects_overlap():
    sents = ["one.", "two.", "three.", "four."]
    out = _aggregate(sents, cap=12, overlap=6)
    # Overlap means the next chunk starts with tail sentences.
    assert any("two" in c for c in out[1:])


def test_aggregate_handles_empty_input():
    assert _aggregate([], cap=10, overlap=0) == []


def test_nltk_invalid_chunk_size_rejected():
    with pytest.raises(ValueError):
        NltkSentenceSplitter(chunk_size=0)


def test_nltk_overlap_must_be_less_than_chunk_size():
    with pytest.raises(ValueError):
        NltkSentenceSplitter(chunk_size=10, chunk_overlap=10)


def test_nltk_negative_overlap_rejected():
    with pytest.raises(ValueError):
        NltkSentenceSplitter(chunk_size=10, chunk_overlap=-1)


def test_nltk_construction_succeeds_without_nltk_installed():
    # Construction should not import nltk; only `split_text` does.
    s = NltkSentenceSplitter(chunk_size=100, chunk_overlap=10)
    assert s.chunk_size == 100
    assert s.chunk_overlap == 10
