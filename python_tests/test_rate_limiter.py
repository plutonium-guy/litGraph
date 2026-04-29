"""Token-bucket rate limit decorator — verify burst + steady-state throttling.

Wall-clock-bound test: 4 calls @ 120 RPM with burst=1 → ~1.5s total
(0ms, 500ms, 1000ms, 1500ms). Tolerance is generous to handle CI jitter.
"""
import http.server
import json
import threading
import time

from litgraph.providers import OpenAIChat


class FakeOpenAI(http.server.BaseHTTPRequestHandler):
    HITS = [0]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        FakeOpenAI.HITS[0] += 1
        body = json.dumps({
            "id": "x",
            "object": "chat.completion",
            "model": "gpt-fake",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a, **kw): pass


def _spawn():
    FakeOpenAI.HITS[0] = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_rate_limit_burst_then_throttle():
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="gpt-fake",
                          base_url=f"http://127.0.0.1:{port}/v1")
        # 60 RPM = 1 RPS, burst=2 → first 2 instant, 3rd waits ~1s.
        chat.with_rate_limit(60, burst=2)
        msgs = [{"role": "user", "content": "hi"}]

        t0 = time.monotonic()
        chat.invoke(msgs)
        chat.invoke(msgs)
        burst_elapsed = time.monotonic() - t0
        assert burst_elapsed < 0.3, f"burst should be near-instant, took {burst_elapsed:.2f}s"

        chat.invoke(msgs)  # 3rd call must wait ~1s
        total = time.monotonic() - t0
        assert total >= 0.9, f"3rd call should wait ~1s, total {total:.2f}s"
        assert FakeOpenAI.HITS[0] == 3
    finally:
        srv.shutdown()


def test_rate_limit_strict_burst_one_paces_evenly():
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="gpt-fake",
                          base_url=f"http://127.0.0.1:{port}/v1")
        # 240 RPM = 4 RPS, burst=1 → every 250ms strictly.
        chat.with_rate_limit(240, burst=1)
        msgs = [{"role": "user", "content": "hi"}]

        t0 = time.monotonic()
        for _ in range(3):
            chat.invoke(msgs)
        # 3 calls @ 4 RPS w/ burst=1 → 0ms, 250ms, 500ms = ~500ms total.
        elapsed = time.monotonic() - t0
        assert 0.4 <= elapsed < 1.0, \
            f"3 calls @ 4 RPS should take ~500ms, got {elapsed:.2f}s"
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_rate_limit_burst_then_throttle,
        test_rate_limit_strict_burst_one_paces_evenly,
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
