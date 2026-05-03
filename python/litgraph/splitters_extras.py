"""NLTK / SpaCy sentence splitters.

Native splitters (`litgraph.splitters`) cover recursive char,
markdown / HTML headers, JSON, token-based, semantic, and code
tree-sitter splits. NLTK + SpaCy ship in this module because:

- their corpora / pretrained models are heavyweight Python
  ecosystems we don't want to re-implement in Rust;
- splitter choice rarely matters on the hot path (it's one-time
  ingestion), so the GIL cost is fine.

Each adapter implements `split_text(text)` and `split_documents(docs)`
matching the rest of the splitter API.
"""
from __future__ import annotations

from typing import Any, Iterable


__all__ = [
    "NltkSentenceSplitter",
    "SpacySentenceSplitter",
]


def _doc(content: str, **metadata: Any) -> dict[str, Any]:
    return {"page_content": content, "metadata": dict(metadata)}


class NltkSentenceSplitter:
    """Sentence-tokenise via NLTK's Punkt model. Lazy-imports `nltk`;
    install with `pip install nltk` and run
    `python -c "import nltk; nltk.download('punkt')"` once.

    Args:
        language: model language (default "english").
        chunk_size: optional cap. If set, sentences are aggregated
            until the byte total would exceed `chunk_size`, then a
            new chunk starts. Useful for token-budget downstreaming.
        chunk_overlap: bytes carried over between adjacent chunks.
            Ignored when `chunk_size` is None.
    """

    def __init__(
        self,
        language: str = "english",
        chunk_size: int | None = None,
        chunk_overlap: int = 0,
    ) -> None:
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or (chunk_size is not None and chunk_overlap >= chunk_size):
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.language = language
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _sentences(self, text: str) -> list[str]:
        try:
            import nltk  # type: ignore[import-not-found]
            from nltk.tokenize import sent_tokenize  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "nltk not installed. "
                "Run `pip install nltk` and `python -c \"import nltk; "
                "nltk.download('punkt')\"` to use this splitter."
            ) from e
        # NLTK throws LookupError if the punkt corpus isn't downloaded.
        # Surface a user-friendly message.
        try:
            return list(sent_tokenize(text, language=self.language))
        except LookupError as e:  # pragma: no cover — env-dependent
            raise RuntimeError(
                "NLTK 'punkt' corpus not downloaded. "
                "Run `python -c \"import nltk; nltk.download('punkt')\"`."
            ) from e

    def split_text(self, text: str) -> list[str]:
        sents = self._sentences(text)
        if self.chunk_size is None:
            return sents
        return _aggregate(sents, self.chunk_size, self.chunk_overlap)

    def split_documents(self, docs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in docs:
            for chunk in self.split_text(d.get("page_content", "")):
                out.append(_doc(chunk, **d.get("metadata", {})))
        return out


class SpacySentenceSplitter:
    """Sentence-tokenise via SpaCy. Higher-quality boundary detection
    than NLTK's regex-based punkt, at the cost of larger model
    download and slower inference.

    Lazy-imports `spacy`; install with `pip install spacy` and
    `python -m spacy download en_core_web_sm` (or another model).

    Args:
        model_name: SpaCy pipeline (default "en_core_web_sm").
        chunk_size: optional aggregation cap. See `NltkSentenceSplitter`.
        chunk_overlap: bytes carried over between chunks.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        chunk_size: int | None = None,
        chunk_overlap: int = 0,
    ) -> None:
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or (chunk_size is not None and chunk_overlap >= chunk_size):
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _nlp(self):
        try:
            import spacy  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "spacy not installed. Run `pip install spacy` "
                f"and `python -m spacy download {self.model_name}`."
            ) from e
        try:
            return spacy.load(self.model_name)
        except OSError as e:  # pragma: no cover — env-dependent
            raise RuntimeError(
                f"SpaCy model {self.model_name!r} not installed. "
                f"Run `python -m spacy download {self.model_name}`."
            ) from e

    def split_text(self, text: str) -> list[str]:
        nlp = self._nlp()
        sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        if self.chunk_size is None:
            return sents
        return _aggregate(sents, self.chunk_size, self.chunk_overlap)

    def split_documents(self, docs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in docs:
            for chunk in self.split_text(d.get("page_content", "")):
                out.append(_doc(chunk, **d.get("metadata", {})))
        return out


def _aggregate(sents: list[str], cap: int, overlap: int) -> list[str]:
    """Combine sentences into byte-budgeted chunks with optional
    overlap. Pure Python; behaviour matches recursive-char splitter
    for sentence-aligned input."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sents:
        s_len = len(s.encode("utf-8")) + 1  # +1 for the space separator
        if cur and cur_len + s_len > cap:
            chunks.append(" ".join(cur).strip())
            if overlap > 0:
                # Keep tail sentences whose total bytes ≤ overlap.
                tail: list[str] = []
                tail_len = 0
                for prev in reversed(cur):
                    p_len = len(prev.encode("utf-8")) + 1
                    if tail_len + p_len > overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += p_len
                cur = tail
                cur_len = tail_len
            else:
                cur = []
                cur_len = 0
        cur.append(s)
        cur_len += s_len
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks
