---
layout: default
title: Retrieval and memory
description: Ingest documents, search with vector and lexical retrieval, compose RAG pipelines, and preserve conversation or long-term memory.
eyebrow: Ground agents
---

# Retrieval and memory

litGraph separates ingestion, embeddings, storage, retrieval, reranking, and memory. Each layer has a small contract, making it possible to change an index or provider without rebuilding the entire RAG pipeline.

## Ingestion pipeline

```python
from litgraph.loaders import DirectoryLoader
from litgraph.splitters import RecursiveCharacterSplitter

documents = DirectoryLoader("./knowledge", glob="**/*.md").load()
chunks = RecursiveCharacterSplitter(
    chunk_size=1_000,
    chunk_overlap=100,
).split_documents(documents)
```

Loaders cover local text, Markdown, JSONL, directories, CSV, PDF, DOCX, notebooks, HTML and sitemaps, along with optional service adapters. Splitters include recursive text, Markdown and HTML headers, JSON, tokens, semantic chunks, and code-aware tree-sitter variants.

Preserve source identifiers, section names, timestamps, and access-control metadata at ingestion. Retrieval quality and safe filtering depend on metadata that cannot be recovered later.

## Embed and store

```python
from litgraph.embeddings import OpenAIEmbeddings
from litgraph.stores import HnswStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
store = HnswStore(dim=3072)
store.add_documents(chunks, embeddings=embeddings)

hits = store.similarity_search("How are agent runs traced?", k=5)
```

Vector stores include memory, native HNSW, Qdrant, pgvector, Chroma, and Weaviate. They implement one vector-store contract; choose based on durability, scale, filtering, operations, and deployment constraints.

| Store | Good fit |
|---|---|
| Memory | Tests, prototypes, and short-lived processes. |
| HNSW | Fast embedded search without running a service. |
| Qdrant | Dedicated vector service with filtering and operations tooling. |
| pgvector | Teams already operating Postgres and relational metadata. |
| Chroma / Weaviate | Existing deployments or ecosystem-specific features. |

The embedding dimension must match the index. Record the embedding model and normalization strategy with the collection so an upgrade cannot silently mix incompatible vectors.

## Lexical, vector, and hybrid retrieval

- `BM25Retriever` is strong for exact terms, identifiers, and rare words.
- Vector retrieval captures semantic similarity and paraphrases.
- `HybridRetriever` combines ranked lists with reciprocal-rank fusion.
- `MMRRetriever` trades some relevance for diversity.
- `EnsembleRetriever` combines multiple retrieval strategies.
- `RaceRetriever` returns the first acceptable backend response.

For most production knowledge bases, start by measuring BM25 and vector search independently, then add hybrid fusion when their failure modes are complementary.

## Advanced RAG patterns

```python
from litgraph.retrieval import (
    ContextualCompressionRetriever,
    HyDERetriever,
    MMRRetriever,
    MultiQueryRetriever,
)

diverse = MMRRetriever(base=store, fetch_k=20, k=5, lambda_mult=0.5)
compressed = ContextualCompressionRetriever(
    base_retriever=diverse,
    llm=model,
)
```

Other included patterns:

- **Multi-query** generates several search formulations and merges results.
- **HyDE** embeds a hypothetical answer to bridge question/document vocabulary.
- **Parent-document** retrieves small chunks but returns the larger source context.
- **Multi-vector** indexes several representations of one source.
- **Self-query** turns structured request constraints into metadata filters.
- **Time-weighted** balances semantic relevance with recency.
- **Contextual compression** filters or condenses retrieved context before generation.

Add these only after an evaluation set identifies the failure mode they address. More retrieval stages increase latency, cost, and the number of parameters that can overfit a small test set.

## Reranking

Rerank a moderately sized candidate set after cheap retrieval. Adapters include Cohere, Voyage, Jina, FastEmbed cross-encoders, and ensemble reranking.

<div class="flow"><span>query</span><i>→</i><span>retrieve 20–100</span><i>→</i><span>rerank</span><i>→</i><span>select 3–10</span><i>→</i><span>generate</span></div>

Track retrieval recall separately from final answer quality. A generator cannot cite evidence that the retrieval layer never returned.

## Conversation memory

```python
from litgraph.memory import TokenBufferMemory

memory = TokenBufferMemory(max_tokens=4_000, model_name="gpt-5")
memory.set_system("You are concise.")
memory.add_user("Remember that the deployment region is ap-south-1.")
memory.add_ai("Understood.")

messages = memory.messages()
```

Token-buffer memory bounds the conversation by model-aware token count. Summary-buffer memory asks a model to compress older context when the limit is reached. Backends include in-process, SQLite, Postgres, and Redis.

Conversation history is not the same as durable knowledge. Keep an auditable source of truth outside the prompt, and use memory to select or summarize what the agent needs for the current turn.

## Long-term facts

The LangMem-style fact extractor distills durable facts from longer conversations into the shared store abstraction. Define what is eligible to remember, how a user can inspect or delete it, and when a fact expires.

<div class="callout warning"><strong>Privacy.</strong> Long-term memory can turn transient input into durable personal data. Apply retention, access control, redaction, and deletion policies before enabling extraction.</div>

## Evaluate retrieval

Create a small, versioned set of queries with relevant document identifiers. Track at least recall@k, ranking quality, latency, and the proportion of answers whose claims are supported by retrieved evidence. Re-run the set whenever chunking, embeddings, filters, stores, fusion, or reranking change.
