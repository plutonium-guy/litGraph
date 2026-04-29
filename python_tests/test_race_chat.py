"""RaceChat — invoke N chat models concurrently, first success wins.

Tests use OpenAIChat configured against an unreachable host so every
inner errors fast and deterministically; that lets us verify the
aggregate-error path. Latency-win behaviour is covered by the Rust
test suite — testing it from Python adds flake risk without
strengthening the contract."""
from litgraph.providers import OpenAIChat, RaceChat


def _broken(label: str):
    return OpenAIChat(
        api_key=f"sk-{label}",
        model="gpt-4o-mini",
        base_url="http://127.0.0.1:1",
    )


def test_race_chat_constructs_repr():
    r = RaceChat([_broken("a"), _broken("b")])
    assert "race(" in repr(r)


def test_race_chat_rejects_empty_list():
    try:
        RaceChat([])
    except ValueError as e:
        assert "at least one" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_race_chat_aggregates_when_all_fail():
    r = RaceChat([_broken("x"), _broken("y"), _broken("z")])
    try:
        r.invoke([{"role": "user", "content": "hi"}])
    except RuntimeError as e:
        msg = str(e)
        assert "all 3 inners failed" in msg, f"got: {msg}"
    else:
        raise AssertionError("expected RuntimeError")


def test_race_chat_single_inner_passthrough_errors():
    """Single inner: race degenerates to a direct call. Still surfaces
    the failure (broken host) as RuntimeError, not the aggregate."""
    r = RaceChat([_broken("only")])
    try:
        r.invoke([{"role": "user", "content": "hi"}])
    except RuntimeError as e:
        msg = str(e)
        # Single-inner path skips the aggregate format.
        assert "all 1 inners failed" not in msg
    else:
        raise AssertionError("expected RuntimeError")


def test_race_chat_extractable_as_chat_model():
    """RaceChat must compose with other wrappers via extract_chat_model.
    Wrap one in a TokenBudgetChat to verify."""
    from litgraph.providers import TokenBudgetChat
    inner = RaceChat([_broken("a"), _broken("b")])
    # Just constructing the wrapper exercises the extraction path.
    bud = TokenBudgetChat(inner, max_tokens=4096)
    assert bud is not None


if __name__ == "__main__":
    fns = [
        test_race_chat_constructs_repr,
        test_race_chat_rejects_empty_list,
        test_race_chat_aggregates_when_all_fail,
        test_race_chat_single_inner_passthrough_errors,
        test_race_chat_extractable_as_chat_model,
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
