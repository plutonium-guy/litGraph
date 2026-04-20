"""End-to-end RAG agent demo.

Runs entirely on a tiny in-memory corpus + a fake OpenAI-compatible HTTP
server, so it exercises the full pipeline without a live API key:

  1. Loaders   — synthetic docs (substitute DirectoryLoader/WebLoader for real)
  2. Splitters — chunk into ~120 chars
  3. Embeddings — toy bag-of-words (substitute your real embedder)
  4. Vector store — HNSW (instant-distance, pure Rust)
  5. Retriever — VectorRetriever
  6. Tool      — wraps retriever as a `lookup` FunctionTool
  7. Agent     — ReactAgent + OpenAIChat -> fake server
  8. Cost      — CostTracker accumulates USD as the agent runs
  9. Cache     — MemoryCache deduplicates identical model calls

Run:  python examples/rag_agent.py
"""
import http.server
import json
import threading

from litgraph.providers import OpenAIChat
from litgraph.agents import ReactAgent
from litgraph.tools import FunctionTool
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import HnswVectorStore, VectorRetriever
from litgraph.splitters import RecursiveCharacterSplitter
from litgraph.observability import CostTracker
from litgraph.cache import MemoryCache


# ---------- 1. Corpus ----------

CORPUS = [
    {"content": "Rust has no garbage collector and uses RAII for memory.", "id": "rust1"},
    {"content": "Python uses reference counting plus a cycle collector.", "id": "py1"},
    {"content": "Tokio is the de-facto async runtime in Rust; uses work-stealing.", "id": "tok1"},
    {"content": "asyncio is Python's standard async runtime; single-threaded by default.", "id": "asy1"},
    {"content": "litGraph runs Rust under PyO3 and releases the GIL during work.", "id": "lg1"},
]


# ---------- 2-5. RAG plumbing ----------

splitter = RecursiveCharacterSplitter(chunk_size=120, chunk_overlap=20)
chunks = splitter.split_documents(CORPUS)
print(f"split {len(CORPUS)} docs -> {len(chunks)} chunks")

VOCAB = ["rust", "python", "tokio", "asyncio", "gil", "memory", "async", "litgraph"]

def bow_embed(texts):
    out = []
    for t in texts:
        low = t.lower()
        out.append([float(low.count(w)) for w in VOCAB])
    return out

embeddings = FunctionEmbeddings(bow_embed, dimensions=len(VOCAB), name="bow")
store = HnswVectorStore()
store.add(chunks, bow_embed([c["content"] for c in chunks]))
retriever = VectorRetriever(embeddings, store)


# ---------- 6. lookup tool wrapping the retriever ----------

def lookup(args):
    hits = retriever.retrieve(args["query"], k=3)
    return {"results": [{"id": h["id"], "snippet": h["content"]} for h in hits]}

lookup_tool = FunctionTool(
    "lookup",
    "Search the knowledge base. Returns top-3 relevant snippets.",
    {"type": "object",
     "properties": {"query": {"type": "string"}},
     "required": ["query"]},
    lookup,
)


# ---------- 7. Fake OpenAI server (instead of a real API key) ----------

class FakeOAI(http.server.BaseHTTPRequestHandler):
    SCRIPT = []
    CALLS = [0]
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        idx = FakeOAI.CALLS[0]
        FakeOAI.CALLS[0] += 1
        body = json.dumps(FakeOAI.SCRIPT[idx % len(FakeOAI.SCRIPT)]).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a, **kw): pass


def assistant_tool_call(name, args):
    return {
        "choices": [{"index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }]}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        "model": "gpt-fake",
    }

def assistant_text(text):
    return {
        "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110},
        "model": "gpt-fake",
    }

FakeOAI.SCRIPT = [
    assistant_tool_call("lookup", {"query": "tokio runtime in rust"}),
    assistant_text("Tokio is the de-facto async runtime in Rust, work-stealing."),
]
FakeOAI.CALLS[0] = 0
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOAI)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()


# ---------- 8-9. Provider with cache + cost tracking ----------

model = OpenAIChat(api_key="fake", model="gpt-fake",
                   base_url=f"http://127.0.0.1:{port}/v1")
cache = MemoryCache(max_capacity=100)
tracker = CostTracker({"gpt-fake": (2.0, 10.0)})
model.with_cache(cache)
model.instrument(tracker)


# ---------- 7 (final). Agent ----------

agent = ReactAgent(
    model, [lookup_tool],
    system_prompt="Answer using `lookup` to fetch facts; be concise.",
    max_iterations=4,
)

state = agent.invoke("Tell me about Tokio.")

print("\n--- agent transcript ---")
for m in state["messages"]:
    body = m.get("content") or m.get("tool_calls") or ""
    print(f"  [{m['role']}] {body}")

# Give the cost-tracker bus 50ms to flush its drain task.
import time; time.sleep(0.1)
snap = tracker.snapshot()
print(f"\ncalls={snap['calls']} prompt_toks={snap['prompt_tokens']} usd=${snap['usd']:.4f}")

srv.shutdown()
print("done.")
