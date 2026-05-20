"""Tests for `litgraph.recipes.serve` actually spawning the binary
(iter 384). Pre-iter-384 this returned a fake shell-command string."""

import sys
import urllib.request
import urllib.error
import json
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

litgraph = pytest.importorskip("litgraph")
if not hasattr(litgraph, "serve") or not hasattr(litgraph.serve, "spawn_chat"):
    pytest.skip(
        "native serve binding not built; run `maturin develop`",
        allow_module_level=True,
    )

from litgraph import recipes  # noqa: E402
from litgraph.providers import OpenAIChat  # noqa: E402


def _stub_model():
    """Native `OpenAIChat` instance pointed at an unreachable base URL.
    Used purely for the bind / shutdown / endpoint-shape tests where
    the model itself is never invoked — the chat completion endpoint
    is exercised separately under provider-keyed integration tests."""
    return OpenAIChat(
        api_key="test-only",
        base_url="http://127.0.0.1:1",
        model="stub-model",
    )


def _wait_for_health(url: str, timeout: float = 3.0) -> dict:
    """Poll `/health` until 200 or timeout. Returns parsed JSON."""
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=0.5) as resp:
                if resp.status == 200:
                    return json.loads(resp.read())
        except (urllib.error.URLError, ConnectionError) as e:
            last_err = e
            time.sleep(0.05)
    raise AssertionError(f"server did not become healthy: {last_err}")


def test_serve_chat_model_binds_and_serves_health():
    model = _stub_model()
    # port=0 → OS picks a free port. Avoids flake on busy CI.
    handle = recipes.serve(model, port=0)
    try:
        url = handle.url()
        assert url.startswith("http://localhost:") or url.startswith("http://127.0.0.1:")
        body = _wait_for_health(url)
        assert body == {"status": "ok"}
    finally:
        handle.shutdown()


def test_serve_info_endpoint_returns_model_metadata():
    model = _stub_model()
    handle = recipes.serve(model, port=0)
    try:
        url = handle.url()
        _wait_for_health(url)
        with urllib.request.urlopen(f"{url}/info", timeout=1.0) as resp:
            payload = json.loads(resp.read())
        assert "name" in payload
        assert "endpoints" in payload
        # Endpoints list must cover invoke / stream / batch — the
        # surface every downstream HTTP client expects.
        ep_paths = {e if isinstance(e, str) else e.get("path") for e in payload["endpoints"]}
        assert "/invoke" in ep_paths
        assert "/stream" in ep_paths
        assert "/batch" in ep_paths
    finally:
        handle.shutdown()


def test_serve_handle_address_and_url_match():
    model = _stub_model()
    handle = recipes.serve(model, port=0)
    try:
        addr = handle.address()
        # address() returns "host:port"; url() returns "http://...".
        assert ":" in addr
        port = addr.rsplit(":", 1)[1]
        assert port in handle.url()
    finally:
        handle.shutdown()


def test_serve_shutdown_is_idempotent():
    model = _stub_model()
    handle = recipes.serve(model, port=0)
    handle.shutdown()
    # Second call must not raise; documented contract.
    handle.shutdown()


def test_serve_rejects_graph_with_clear_message():
    # `CompiledGraph` would have `.compile` missing but `.invoke`
    # present — simulate that surface so the test doesn't need the
    # full graph wiring.
    class FakeGraph:
        def invoke(self, *a, **kw):
            return {}

    with pytest.raises(NotImplementedError, match="recipes.serve\\(graph\\)"):
        recipes.serve(FakeGraph(), port=0)


def test_serve_rejects_garbage_input():
    with pytest.raises(TypeError, match="ChatModel"):
        recipes.serve("not a model", port=0)


def test_serve_port_in_use_raises_oserror():
    """Bind once, then try to bind the same port — second call must
    surface as OSError, not silent failure."""
    model = _stub_model()
    handle = recipes.serve(model, port=0)
    try:
        addr = handle.address()
        port = int(addr.rsplit(":", 1)[1])
        # Second bind on the explicit port should fail.
        with pytest.raises(OSError):
            recipes.serve(_stub_model(), port=port).shutdown()
    finally:
        handle.shutdown()
