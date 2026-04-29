"""OpenTelemetry OTLP exporter bindings. Smoke-tests the Python-facing
config surface: init_stdout / init_otlp / shutdown don't panic and can
be called multiple times.

Full span-export integration is tested upstream (tracing-opentelemetry +
opentelemetry_sdk); our binding layer's value is the config glue +
guard lifetime management."""
from litgraph.tracing import init_otlp, init_stdout, shutdown


def test_init_stdout_and_shutdown_roundtrip():
    """Stdout exporter installs globally, shutdown is clean."""
    init_stdout(service_name="test-stdout")
    shutdown()


def test_shutdown_without_init_is_noop():
    """Calling shutdown() when no provider was installed must not raise."""
    shutdown()


def test_shutdown_is_idempotent():
    init_stdout(service_name="t")
    shutdown()
    shutdown()
    shutdown()  # repeated calls safe


def test_init_otlp_accepts_endpoint_and_service_name():
    """OTLP exporter init doesn't require a live collector at init time —
    the batch processor defers connection until the first export. So we
    can pass a fake endpoint and init must still succeed synchronously."""
    init_otlp(
        endpoint="http://127.0.0.1:9999",  # nothing listening
        service_name="test-otlp",
    )
    shutdown()


def test_init_otlp_defaults():
    """Default args: endpoint=localhost:4317, service_name=litgraph."""
    init_otlp()
    shutdown()


def test_init_stdout_defaults():
    init_stdout()
    shutdown()


def test_reinit_replaces_prior_guard():
    """Calling init a second time replaces the stored guard, doesn't leak."""
    init_stdout(service_name="first")
    init_stdout(service_name="second")
    shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_init_stdout_and_shutdown_roundtrip,
        test_shutdown_without_init_is_noop,
        test_shutdown_is_idempotent,
        test_init_otlp_accepts_endpoint_and_service_name,
        test_init_otlp_defaults,
        test_init_stdout_defaults,
        test_reinit_replaces_prior_guard,
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
