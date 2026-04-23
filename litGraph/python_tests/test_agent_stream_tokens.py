"""ReactAgent.stream_tokens() — token-level streaming for chat-UI agents.

Builds on iter 76's per-turn event stream by adding `token_delta` events
emitted as the LLM generates characters. Use case: chat interfaces where
the user sees the assistant's reply progressively typed out instead of
waiting for the full message.

Same event shape as `stream()` PLUS the new `token_delta` event. Verified
against a fake OpenAI SSE server that streams chunked JSON deltas."""
import http.server
import json
import threading

from litgraph.agents import ReactAgent
from litgraph.providers import OpenAIChat
from litgraph.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class _FakeOpenAISSE(http.server.BaseHTTPRequestHandler):
    """Streams ONE SSE response per POST. The script controls what comes back
    on each call (in order)."""
    SCRIPT: list[list[bytes]] = []
    IDX = [0]

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        chunks = self.SCRIPT[self.IDX[0]]
        self.IDX[0] += 1
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for c in chunks:
            self.wfile.write(c)
            self.wfile.flush()

    def log_message(self, *a, **kw): pass


def _spawn(script):
    _FakeOpenAISSE.SCRIPT = script
    _FakeOpenAISSE.IDX[0] = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAISSE)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _text_chunks(text):
    """Build one SSE response that emits each character as a delta then DONE."""
    chunks = []
    for ch in text:
        body = json.dumps({
            "choices": [{"index": 0, "delta": {"content": ch}}]
        }).encode()
        chunks.append(b"data: " + body + b"\n\n")
    # Final stop chunk + [DONE].
    final = json.dumps({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }).encode()
    chunks.append(b"data: " + final + b"\n\n")
    chunks.append(b"data: [DONE]\n\n")
    return chunks


def test_stream_tokens_emits_token_delta_events_for_final_answer():
    text = "Hello!"
    srv, port = _spawn([_text_chunks(text)])
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        agent = ReactAgent(model=chat, tools=[add])
        events = list(agent.stream_tokens("hi"))
    finally:
        srv.shutdown()

    # One token_delta per character, in order.
    deltas = [e["text"] for e in events if e["type"] == "token_delta"]
    assert "".join(deltas) == text, f"got: {deltas}"
    assert len(deltas) == len(text)

    # Then llm_message + final, in order.
    types = [e["type"] for e in events]
    assert types[0] == "iteration_start"
    assert "token_delta" in types
    # llm_message comes AFTER all token_deltas of that turn.
    last_token_idx = max(i for i, t in enumerate(types) if t == "token_delta")
    llm_msg_idx = types.index("llm_message")
    assert llm_msg_idx > last_token_idx
    # Final last.
    assert types[-1] == "final"


def test_stream_tokens_handles_tool_call_turn_with_zero_token_deltas():
    """First turn is a tool call (empty content → 0 deltas). Then the agent
    runs the tool and streams the final answer.

    Tool-call turn streamed as: empty delta (no content, just tool_calls)
    followed by Done. Final-answer turn streams "Result: 9" as 9 deltas."""
    # Turn 1: tool_call (no content). OpenAI streams tool_call_deltas + Done.
    turn1 = [
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1","type":"function","function":{"name":"add","arguments":""}}]}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\":4,\\"b\\":5}"}}]}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    turn2 = _text_chunks("Result: 9")

    srv, port = _spawn([turn1, turn2])
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        agent = ReactAgent(model=chat, tools=[add])
        events = list(agent.stream_tokens("what is 4+5"))
    finally:
        srv.shutdown()

    # Count token_deltas per iteration.
    counts = {}
    cur = None
    for e in events:
        if e["type"] == "iteration_start":
            cur = e["iteration"]
            counts[cur] = 0
        elif e["type"] == "token_delta":
            counts[cur] = counts.get(cur, 0) + 1
    assert counts == {1: 0, 2: len("Result: 9")}, f"got: {counts}"

    # Tool ran successfully.
    tool_results = [e for e in events if e["type"] == "tool_call_result"]
    assert len(tool_results) == 1
    assert "9" in tool_results[0]["result"]
    assert tool_results[0]["is_error"] is False


if __name__ == "__main__":
    fns = [
        test_stream_tokens_emits_token_delta_events_for_final_answer,
        test_stream_tokens_handles_tool_call_turn_with_zero_token_deltas,
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
