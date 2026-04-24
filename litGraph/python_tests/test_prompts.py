"""ChatPromptTemplate — Jinja `{{ var }}` interpolation, role-tagged
parts, and named MessagesPlaceholder slots for splicing pre-built history.

Direct LangChain `ChatPromptTemplate` parity. Production motivation: every
agent app needs `[system | history | user]` shaped prompts; rebuilding
this by string-concatenating in user code is a footgun farm."""
from litgraph.prompts import ChatPromptTemplate


def test_from_messages_with_jinja_interpolation():
    tmpl = ChatPromptTemplate.from_messages([
        ("system", "You are {{ persona }}."),
        ("user", "Translate '{{ word }}' to {{ language }}."),
    ])
    pv = tmpl.format({"persona": "a polyglot", "word": "hello", "language": "Spanish"})
    msgs = pv.to_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are a polyglot."
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Translate 'hello' to Spanish."


def test_placeholder_splices_messages_in_correct_position():
    """The standard chat-with-memory pattern: system prompt, history
    placeholder, current user input. Verify history lands BETWEEN system
    and user, in the order it was passed."""
    tmpl = ChatPromptTemplate.from_messages([
        ("system", "You are {{ persona }}."),
        ("placeholder", "history"),
        ("user", "{{ input }}"),
    ])
    pv = tmpl.format({"persona": "terse", "input": "what's up?"})
    pv = pv.with_placeholder("history", [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    msgs = pv.to_messages()
    assert len(msgs) == 4
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"] == "You are terse."
    assert msgs[1]["content"] == "hi"
    assert msgs[2]["content"] == "hello"
    assert msgs[3]["content"] == "what's up?"


def test_required_placeholder_unfilled_errors_at_to_messages():
    """A required placeholder that's never filled must error — silent drop
    of expected memory/history is the kind of bug that sits in prod
    forever before someone notices the LLM lost context."""
    tmpl = ChatPromptTemplate.from_messages([
        ("system", "You are helpful."),
        ("placeholder", "history"),
        ("user", "now"),
    ])
    pv = tmpl.format({})
    try:
        pv.to_messages()
    except RuntimeError as e:
        assert "history" in str(e)
    else:
        raise AssertionError("expected RuntimeError on unfilled required placeholder")


def test_optional_placeholder_unfilled_silently_dropped():
    """`optional_placeholder` rows are skipped when not filled — useful for
    'examples' / few-shot slots that are sometimes empty."""
    tmpl = ChatPromptTemplate.from_messages([
        ("system", "You are helpful."),
        ("optional_placeholder", "examples"),
        ("user", "{{ q }}"),
    ])
    msgs = tmpl.format({"q": "now"}).to_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "now"


def test_with_placeholder_unknown_name_errors():
    tmpl = ChatPromptTemplate.from_messages([
        ("system", "s"),
        ("placeholder", "history"),
        ("user", "u"),
    ])
    pv = tmpl.format({})
    try:
        pv.with_placeholder("not_a_slot", [])
    except RuntimeError as e:
        assert "not_a_slot" in str(e)
    else:
        raise AssertionError("expected RuntimeError on unknown placeholder name")


def test_chained_builder_api():
    """Alternative to from_messages — chain builder methods."""
    tmpl = (
        ChatPromptTemplate()
        .system("You are {{ persona }}.")
        .placeholder("history")
        .user("{{ input }}")
    )
    assert tmpl.placeholder_names() == ["history"]
    msgs = (
        tmpl.format({"persona": "terse", "input": "hi"})
        .with_placeholder("history", [{"role": "user", "content": "earlier"}])
        .to_messages()
    )
    assert len(msgs) == 3


def test_human_and_ai_aliases_for_user_and_assistant():
    """LangChain users will write `('human', ...)` / `('ai', ...)` —
    accept both as aliases for compatibility."""
    tmpl = ChatPromptTemplate.from_messages([
        ("human", "{{ q }}"),
        ("ai", "I think the answer is {{ a }}."),
    ])
    msgs = tmpl.format({"q": "x", "a": "42"}).to_messages()
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_format_without_vars_uses_empty_dict():
    """No-var templates should format without requiring an empty dict."""
    tmpl = ChatPromptTemplate.from_messages([("system", "constant text")])
    msgs = tmpl.format().to_messages()
    assert len(msgs) == 1
    assert msgs[0]["content"] == "constant text"


def test_empty_history_filling_required_placeholder_succeeds():
    """Filling a required placeholder with [] is valid — slot was filled,
    just splices nothing. (Different from never filled, which errors.)"""
    tmpl = ChatPromptTemplate.from_messages([
        ("system", "s"),
        ("placeholder", "history"),
        ("user", "u"),
    ])
    msgs = tmpl.format({}).with_placeholder("history", []).to_messages()
    assert len(msgs) == 2


if __name__ == "__main__":
    fns = [
        test_from_messages_with_jinja_interpolation,
        test_placeholder_splices_messages_in_correct_position,
        test_required_placeholder_unfilled_errors_at_to_messages,
        test_optional_placeholder_unfilled_silently_dropped,
        test_with_placeholder_unknown_name_errors,
        test_chained_builder_api,
        test_human_and_ai_aliases_for_user_and_assistant,
        test_format_without_vars_uses_empty_dict,
        test_empty_history_filling_required_placeholder_succeeds,
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
