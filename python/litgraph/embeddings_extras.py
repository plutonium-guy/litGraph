"""Local embedding adapters that wrap third-party Python libs.

Native embeddings live in `litgraph.embeddings` (FastEmbed / OpenAI /
Cohere / Voyage / Jina / Bedrock / Gemini). This module is the escape
hatch for libs whose Python ecosystem is mature enough that
re-implementing them in Rust isn't worth it: SentenceTransformers,
HuggingFace `Inference` SDK, NVIDIA NIM (HTTP).

Each adapter implements the `Embeddings` duck-shape used by litGraph
retrievers: `embed(texts)` and `embed_query(text)`.

Example:

    from litgraph.embeddings_extras import SentenceTransformersEmbeddings

    emb = SentenceTransformersEmbeddings("all-MiniLM-L6-v2")
    vecs = emb.embed(["hello", "world"])

    # Use with any litGraph store:
    from litgraph.stores import HnswStore
    store = HnswStore(dim=emb.dim)
    store.add(docs, vecs)
"""
from __future__ import annotations

from typing import Any, Iterable


__all__ = [
    "SentenceTransformersEmbeddings",
    "HuggingFaceInferenceEmbeddings",
    "NimEmbeddings",
    "InstructorEmbeddings",
    "E5Embeddings",
]


class SentenceTransformersEmbeddings:
    """Wrap `sentence-transformers` so any local ST model plugs into
    the litGraph retrieval stack. Lazy import — `sentence-transformers`
    is an optional dep; install with `pip install sentence-transformers`.

    Defaults are normalisation-on (cosine == dot) and CPU device, the
    settings most retrievers expect.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run `pip install sentence-transformers` to use this adapter."
            ) from e
        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        self._batch_size = batch_size
        # Cache the dim once — ST exposes `get_sentence_embedding_dimension`.
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        ts = list(texts)
        if not ts:
            return []
        vecs = self._model.encode(
            ts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HuggingFaceInferenceEmbeddings:
    """Wrap the HuggingFace Inference API for embedding endpoints.
    Lazy-import `huggingface_hub`; install with `pip install
    huggingface-hub`.

    Args:
        model: model name on the Hub (e.g. "BAAI/bge-large-en-v1.5").
        token: HF token. Falls back to `HF_TOKEN` env.
        endpoint_url: optional dedicated-Inference Endpoint URL; if
            provided, overrides Hub routing.
    """

    def __init__(
        self,
        model: str,
        token: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        import os
        try:
            from huggingface_hub import InferenceClient  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "huggingface_hub not installed. "
                "Run `pip install huggingface-hub` to use this adapter."
            ) from e
        self._client = InferenceClient(
            model=endpoint_url or model,
            token=token or os.environ.get("HF_TOKEN"),
        )
        self._model = model
        self.dim: int | None = None  # populated on first call

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        ts = list(texts)
        if not ts:
            return []
        # InferenceClient.feature_extraction returns numpy arrays for
        # batches and a single array for a string.
        out: list[list[float]] = []
        for t in ts:
            v = self._client.feature_extraction(t, model=self._model)
            # v may be 1D (sentence) or 2D (token-level); take mean
            # if 2D so we always return a sentence vector.
            try:
                if hasattr(v, "ndim") and v.ndim == 2:
                    v = v.mean(axis=0)
                lst = list(map(float, v))
            except (TypeError, ValueError):
                lst = list(map(float, v))
            if self.dim is None:
                self.dim = len(lst)
            out.append(lst)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class NimEmbeddings:
    """NVIDIA NIM (Inference Microservices) embeddings, called via the
    OpenAI-compatible `/v1/embeddings` endpoint. NIM speaks OpenAI
    natively so this is a thin wrapper around `OpenAIEmbeddings` with
    a base_url override.

    Args:
        model: NIM model id (e.g. "NV-Embed-v1").
        api_key: NIM API key (or set `NVIDIA_API_KEY`).
        base_url: NIM endpoint root (default integrate.api.nvidia.com).
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
    ) -> None:
        import os
        from litgraph.embeddings import OpenAIEmbeddings  # type: ignore[attr-defined]
        self._inner = OpenAIEmbeddings(
            model=model,
            api_key=api_key or os.environ.get("NVIDIA_API_KEY", ""),
            base_url=base_url,
        )
        self.dim: int | None = None

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        out = self._inner.embed(list(texts))
        if out and self.dim is None:
            self.dim = len(out[0])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class InstructorEmbeddings:
    """Instructor (Su et al., 2022) — task-conditioned embeddings.
    Wraps `InstructorEmbedding` (which itself sits on
    sentence-transformers) so each `embed` call carries an instruction
    prefix that biases the vector toward the downstream task.

    Lazy-imports `InstructorEmbedding`; install with
    `pip install InstructorEmbedding sentence-transformers`.

    Args:
        model_name: e.g. "hkunlp/instructor-large" or "...-base".
        instruction: prompt prefixed to every input, e.g.
            "Represent the science paragraph for retrieval:".
        device: "cpu" or "cuda".
    """

    def __init__(
        self,
        model_name: str = "hkunlp/instructor-base",
        instruction: str = "Represent the document for retrieval:",
        device: str = "cpu",
    ) -> None:
        try:
            from InstructorEmbedding import INSTRUCTOR  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "InstructorEmbedding not installed. "
                "Run `pip install InstructorEmbedding sentence-transformers` "
                "to use this adapter."
            ) from e
        self._model = INSTRUCTOR(model_name, device=device)
        self._instruction = instruction
        self.dim: int | None = None

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        ts = list(texts)
        if not ts:
            return []
        pairs = [[self._instruction, t] for t in ts]
        vecs = self._model.encode(pairs)
        out = [list(map(float, v)) for v in vecs]
        if out and self.dim is None:
            self.dim = len(out[0])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class E5Embeddings:
    """E5 family ("intfloat/e5-base-v2", "...-large-v2", multilingual).
    Built on sentence-transformers but requires a `query: ` /
    `passage: ` prefix to differentiate retrieval inputs — this
    wrapper applies the prefix automatically.

    Lazy-imports `sentence-transformers`. Install with
    `pip install sentence-transformers`.

    Args:
        model_name: e.g. "intfloat/e5-base-v2".
        device: "cpu" or "cuda".
        normalize: L2-normalise output (E5 paper recommends).
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run `pip install sentence-transformers` to use this adapter."
            ) from e
        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        prefixed = [f"{prefix}: {t}" for t in texts]
        vecs = self._model.encode(
            prefixed,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vecs]

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        ts = list(texts)
        if not ts:
            return []
        return self._encode(ts, "passage")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], "query")[0]


# Convenience: catch the most common typos / aliases for `from_env`.
def from_env(name: str, /, **overrides: Any) -> Any:
    """Pick an embeddings adapter by name. Convenience for recipes
    that read config from env / yaml.

    Names: "sentence-transformers", "hf-inference", "nim",
    "instructor", "e5".
    """
    n = name.replace("_", "-").lower()
    if n in ("sentence-transformers", "st", "sbert"):
        return SentenceTransformersEmbeddings(**overrides)
    if n in ("hf-inference", "huggingface-inference", "hf"):
        return HuggingFaceInferenceEmbeddings(**overrides)
    if n in ("nim", "nvidia-nim"):
        return NimEmbeddings(**overrides)
    if n in ("instructor",):
        return InstructorEmbeddings(**overrides)
    if n in ("e5",):
        return E5Embeddings(**overrides)
    raise ValueError(f"unknown embeddings adapter: {name!r}")
