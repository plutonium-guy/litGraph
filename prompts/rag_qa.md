---
name: rag_qa
tags: rag qa
version: 1
description: Strict-grounding QA over a retrieved context passage.
---

Answer the question using ONLY the context below. If the answer
isn't in the context, reply "I don't know based on the provided
context." Don't guess. Don't extrapolate.

Cite specific sentences from the context that support each claim.
Use bracketed citation markers like [1], [2] tied to the order of
appearance in the context.

Context:
---
{context}
---

Question: {question}

Answer:
