"""Tests for SupervisorAgent + SemanticCache Python bindings."""
import http.server
import json
import threading

from litgraph.providers import OpenAIChat, ollama_chat
from litgraph.agents import ReactAgent, SupervisorAgent
from litgraph.tools import FunctionTool
from litgraph.cache import SemanticCache
from litgraph.embeddings import FunctionEmbeddings


class FakeOpenAI(http.server.BaseHTTPRequestHandler):
    """Returns canned responses scripted by call count."""
    SCRIPT = []
    CALLS = [0]

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        idx = FakeOpenAI.CALLS[0]
        FakeOpenAI.CALLS[0] += 1
        body = json.dumps(FakeOpenAI.SCRIPT[idx % len(FakeOpenAI.SCRIPT)]).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw): pass


def _start():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _msg(text="ok", finish="stop", tool_calls=None):
    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        "model": "test",
    }


def _tc(id_, name, args):
    return {
        "id": id_, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_supervisor_routes_via_python_binding():
    FakeOpenAI.SCRIPT = [
        # Supervisor: handoff to math
        _msg(text="", finish="tool_calls", tool_calls=[
            _tc("h1", "handoff", {"worker": "math", "message": "compute 2+2"})
        ]),
        # Math worker: returns plain text
        _msg(text="The answer is 4."),
        # Supervisor: finish
        _msg(text="", finish="tool_calls", tool_calls=[
            _tc("f1", "finish", {"answer": "Done: 4"})
        ]),
        # Supervisor: stop
        _msg(text="all done"),
    ]
    FakeOpenAI.CALLS[0] = 0
    srv = _start()
    try:
        port = srv.server_address[1]
        sup_model = OpenAIChat(api_key="x", model="gpt-test",
                               base_url=f"http://127.0.0.1:{port}/v1")
        math_model = OpenAIChat(api_key="x", model="gpt-test",
                                base_url=f"http://127.0.0.1:{port}/v1")
        math_worker = ReactAgent(math_model, [], max_iterations=3)
        sup = SupervisorAgent(sup_model, {"math": math_worker}, max_hops=4)

        state = sup.invoke("compute 2+2")
        assert state["iterations"] >= 1
        assert "math" in sup.worker_names()
    finally:
        srv.shutdown()


def test_semantic_cache_constructs_and_clears():
    def embed(texts):
        # 2-dim BOW for "cat"/"dog"
        out = []
        for t in texts:
            l = t.lower()
            out.append([float(l.count("cat")), float(l.count("dog"))])
        return out
    e = FunctionEmbeddings(embed, dimensions=2, name="bow")
    cache = SemanticCache(e, threshold=0.95, max_entries=100)
    assert len(cache) == 0
    cache.clear()
    assert "SemanticCache" in repr(cache)


def test_provider_with_semantic_cache_chain():
    """OpenAIChat.with_semantic_cache() composes correctly."""
    def embed(texts):
        return [[float(len(t))] for t in texts]
    e = FunctionEmbeddings(embed, dimensions=1, name="charlen")
    cache = SemanticCache(e, threshold=0.99)

    model = OpenAIChat(api_key="x", model="gpt-test")
    # Wraps inner ChatModel with SemanticCachedModel — no error means
    # the chain composed (Arc<dyn ChatModel> swap succeeded).
    model.with_semantic_cache(cache)
    assert "OpenAIChat" in repr(model)


def test_ollama_chat_helper():
    """`ollama_chat()` returns a pre-configured OpenAIChat for Ollama's
    OpenAI-compat server. We verify only construction here — no network."""
    m = ollama_chat("llama3.2")
    assert isinstance(m, OpenAIChat)
    assert "llama3.2" in repr(m)
    # Custom base_url override.
    m2 = ollama_chat("qwen2.5", base_url="http://10.0.0.5:11434/v1")
    assert isinstance(m2, OpenAIChat)


if __name__ == "__main__":
    fns = [
        test_supervisor_routes_via_python_binding,
        test_semantic_cache_constructs_and_clears,
        test_provider_with_semantic_cache_chain,
        test_ollama_chat_helper,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)
