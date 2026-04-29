"""ChatPromptTemplate.from_json + from_dict + to_json + to_dict — load
prompts from version-controlled .json (or .yaml via PyYAML upstream)
files. Round-trip serialization."""
import json

from litgraph.prompts import ChatPromptTemplate


def test_from_json_loads_minimal_template():
    spec = json.dumps({
        "messages": [
            {"role": "system", "template": "You are {{ persona }}."},
            {"role": "user", "template": "{{ q }}"},
        ],
    })
    tmpl = ChatPromptTemplate.from_json(spec)
    pv = tmpl.format({"persona": "terse", "q": "Why?"})
    msgs = pv.to_messages()
    assert len(msgs) == 2
    assert msgs[0]["content"] == "You are terse."
    assert msgs[1]["content"] == "Why?"


def test_from_json_with_placeholder_and_partials():
    spec = json.dumps({
        "messages": [
            {"role": "system", "template": "You are {{ persona }}."},
            {"role": "placeholder", "name": "history", "optional": True},
            {"role": "user", "template": "{{ q }}"},
        ],
        "partials": {"persona": "helpful"},
    })
    tmpl = ChatPromptTemplate.from_json(spec)
    # `persona` pre-bound via partials → only need to supply `q`.
    pv = tmpl.format({"q": "hi"})
    msgs = pv.to_messages()
    # Optional placeholder unfilled → just system + user.
    assert len(msgs) == 2
    assert "helpful" in msgs[0]["content"]


def test_from_json_unknown_role_raises_value_error():
    spec = json.dumps({"messages": [{"role": "robot", "template": "x"}]})
    try:
        ChatPromptTemplate.from_json(spec)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "robot" in str(e)


def test_from_json_invalid_json_raises_value_error():
    try:
        ChatPromptTemplate.from_json("not json")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_from_dict_accepts_python_dict_directly():
    """Mirror real-world usage: load YAML upstream via PyYAML, pass dict."""
    spec = {
        "messages": [
            {"role": "user", "template": "from yaml: {{ x }}"},
        ],
    }
    tmpl = ChatPromptTemplate.from_dict(spec)
    msgs = tmpl.format({"x": "ok"}).to_messages()
    assert msgs[0]["content"] == "from yaml: ok"


def test_to_json_round_trips():
    original = (
        ChatPromptTemplate()
        .system("You are {{ persona }}.")
        .placeholder("history")
        .user("{{ question }}")
    )
    serialized = original.to_json()
    restored = ChatPromptTemplate.from_json(serialized)

    pv = restored.format({"persona": "polite", "question": "How?"})
    pv = pv.with_placeholder("history",
                             [{"role": "user", "content": "earlier"}])
    msgs = pv.to_messages()
    assert msgs[0]["content"] == "You are polite."
    assert msgs[1]["content"] == "earlier"
    assert msgs[2]["content"] == "How?"


def test_to_dict_returns_serializable_python_dict():
    tmpl = ChatPromptTemplate().system("sys").user("{{ x }}")
    d = tmpl.to_dict()
    assert isinstance(d, dict)
    assert "messages" in d
    # Round-trip through `from_dict`.
    restored = ChatPromptTemplate.from_dict(d)
    msgs = restored.format({"x": "y"}).to_messages()
    assert msgs[0]["content"] == "sys"
    assert msgs[1]["content"] == "y"


def test_to_json_omits_optional_false_for_cleanliness():
    """Required placeholder shouldn't have `optional: false` in JSON —
    cleaner output for code review."""
    tmpl = ChatPromptTemplate().placeholder("h")
    blob = tmpl.to_json()
    assert "h" in blob
    # JSON shouldn't include a literal "optional": false (default).
    assert '"optional": false' not in blob
    assert '"optional":false' not in blob


def test_to_json_includes_optional_true_when_set():
    tmpl = ChatPromptTemplate().optional_placeholder("h")
    blob = tmpl.to_json()
    assert '"optional"' in blob
    assert "true" in blob


def test_yaml_workflow_via_pyyaml_upstream():
    """The headline use case: store prompts in version-controlled YAML."""
    try:
        import yaml
    except ImportError:
        # PyYAML not installed in the test venv; skip without failing.
        return
    yaml_text = """
messages:
  - role: system
    template: "You are {{ persona }}."
  - role: user
    template: "{{ question }}"
partials:
  persona: helpful
"""
    spec = yaml.safe_load(yaml_text)
    tmpl = ChatPromptTemplate.from_dict(spec)
    msgs = tmpl.format({"question": "Why?"}).to_messages()
    assert "helpful" in msgs[0]["content"]
    assert msgs[1]["content"] == "Why?"


if __name__ == "__main__":
    import traceback
    fns = [
        test_from_json_loads_minimal_template,
        test_from_json_with_placeholder_and_partials,
        test_from_json_unknown_role_raises_value_error,
        test_from_json_invalid_json_raises_value_error,
        test_from_dict_accepts_python_dict_directly,
        test_to_json_round_trips,
        test_to_dict_returns_serializable_python_dict,
        test_to_json_omits_optional_false_for_cleanliness,
        test_to_json_includes_optional_true_when_set,
        test_yaml_workflow_via_pyyaml_upstream,
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
