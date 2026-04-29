"""LangSmith OTel shim — preconfigured OTLP/HTTP exporter pointing at
LangSmith's ingest endpoint with the right auth headers. For users
migrating from LangChain's LangSmith tracing to litGraph without
re-plumbing their observability stack.

These smoke tests verify the config-glue surface; full end-to-end
verification would require a real LangSmith endpoint (skipped).
Init doesn't contact the endpoint synchronously — the batch processor
defers connection until the first export."""
from litgraph.tracing import init_langsmith, init_otlp_http, shutdown


def test_init_langsmith_roundtrip():
    init_langsmith(api_key="lsv2_fake_key", project_name="test-project")
    shutdown()


def test_init_langsmith_default_project_name():
    init_langsmith(api_key="lsv2_fake_key")
    shutdown()


def test_init_otlp_http_with_custom_headers():
    init_otlp_http(
        endpoint="http://127.0.0.1:9999/v1/traces",
        service_name="honeycomb-test",
        headers={"x-honeycomb-team": "fake-team-id", "x-honeycomb-dataset": "dev"},
    )
    shutdown()


def test_init_otlp_http_empty_headers_ok():
    init_otlp_http(
        endpoint="http://127.0.0.1:9999/v1/traces",
        service_name="empty-headers",
        headers={},
    )
    shutdown()


def test_reinit_langsmith_replaces_prior_guard():
    init_langsmith("key1", "project-1")
    init_langsmith("key2", "project-2")
    shutdown()


def test_shutdown_idempotent_with_langsmith():
    init_langsmith("k", "p")
    shutdown()
    shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_init_langsmith_roundtrip,
        test_init_langsmith_default_project_name,
        test_init_otlp_http_with_custom_headers,
        test_init_otlp_http_empty_headers_ok,
        test_reinit_langsmith_replaces_prior_guard,
        test_shutdown_idempotent_with_langsmith,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)
