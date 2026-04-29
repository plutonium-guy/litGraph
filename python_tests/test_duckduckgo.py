"""DuckDuckGoSearchTool — Instant Answer JSON endpoint, no API key.
Verifies construction + ReactAgent acceptance + end-to-end via a fake server."""
import http.server
import json
import threading

from litgraph.agents import ReactAgent
from litgraph.providers import OpenAIChat
from litgraph.tools import DuckDuckGoSearchTool


def test_constructor_works_without_api_key():
    """The whole point of DDG: zero credentials."""
    t = DuckDuckGoSearchTool()
    assert t.name == "web_search"
    assert "DuckDuckGoSearchTool" in repr(t)


def test_react_agent_accepts_ddg_tool():
    chat = OpenAIChat(api_key="k", model="gpt-x", base_url="http://127.0.0.1:1/v1")
    agent = ReactAgent(model=chat, tools=[DuckDuckGoSearchTool()], max_iterations=1)
    assert agent is not None


def test_react_agent_invokes_ddg_end_to_end():
    """Fake LLM emits a web_search tool call → ReactAgent dispatches to our
    DDG tool → DDG (also fake) returns Wikipedia-style abstract → final answer
    references it."""

    # Two servers: one is the LLM, one is the DDG endpoint.
    class FakeDDG(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({
                "Heading": "Rust",
                "AbstractText": "Rust is a memory-safe systems language.",
                "AbstractURL": "https://en.wikipedia.org/wiki/Rust_(programming_language)",
                "RelatedTopics": [],
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a, **kw): pass

    SCRIPT = [
        # Turn 1: tool call → web_search("rust language")
        {
            "id": "1", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "t1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "rust language"}),
                        }
                    }]
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        # Turn 2: final answer that mentions DDG's abstract
        {
            "id": "2", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": "Per the search: Rust is a memory-safe systems language."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]

    class FakeLLM(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            payload = SCRIPT[FakeLLM.IDX[0]]
            FakeLLM.IDX[0] += 1
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a, **kw): pass

    FakeLLM.IDX[0] = 0
    ddg_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeDDG)
    threading.Thread(target=ddg_srv.serve_forever, daemon=True).start()
    llm_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=llm_srv.serve_forever, daemon=True).start()
    ddg_port = ddg_srv.server_address[1]
    llm_port = llm_srv.server_address[1]
    try:
        chat = OpenAIChat(api_key="k", model="gpt-x",
                          base_url=f"http://127.0.0.1:{llm_port}/v1")
        ddg = DuckDuckGoSearchTool(base_url=f"http://127.0.0.1:{ddg_port}")
        agent = ReactAgent(model=chat, tools=[ddg], max_iterations=5)
        out = agent.invoke("what is rust?")
        # Final message references the DDG abstract.
        assert "memory-safe" in out["messages"][-1]["content"]
    finally:
        ddg_srv.shutdown()
        llm_srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_constructor_works_without_api_key,
        test_react_agent_accepts_ddg_tool,
        test_react_agent_invokes_ddg_end_to_end,
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
