"""Stream tokens from a chat model and print as they arrive.

Uses a tiny fake OpenAI-compatible SSE server so this runs offline. Swap
`base_url` to a real provider URL + drop the api_key=fake to use OpenAI / vLLM
/ Ollama / Together / Groq. The streaming code itself does not change.

Run:  python examples/streaming_chat.py
"""
import http.server
import threading

from litgraph.providers import OpenAIChat


class FakeSSE(http.server.BaseHTTPRequestHandler):
    CHUNKS = [
        b'data: {"choices":[{"index":0,"delta":{"content":"Streaming "}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"is "}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"all "}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"in "}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"Rust"},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for c in FakeSSE.CHUNKS:
            self.wfile.write(c)
            self.wfile.flush()
    def log_message(self, *a, **kw): pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeSSE)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

model = OpenAIChat(api_key="fake", model="gpt-fake",
                   base_url=f"http://127.0.0.1:{port}/v1")

print("Tell me one sentence: ", end="", flush=True)
for ev in model.stream([{"role": "user", "content": "Tell me one sentence."}]):
    if ev["type"] == "delta":
        print(ev["text"], end="", flush=True)
    elif ev["type"] == "done":
        print(f"\n[done — {ev['usage']['total']} tokens, finish={ev['finish_reason']}]")

srv.shutdown()
