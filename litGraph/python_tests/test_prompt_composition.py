"""ChatPromptTemplate composition (iter 152) — extend() + + operator.
Layer prompts (base + role + task) loaded from separate files."""
from litgraph.prompts import ChatPromptTemplate


def test_extend_appends_other_parts():
    base = ChatPromptTemplate().system("You are {{ persona }}.")
    task = ChatPromptTemplate().user("{{ q }}")
    combined = base.extend(task)
    msgs = combined.format({"persona": "terse", "q": "Why?"}).to_messages()
    assert len(msgs) == 2
    assert msgs[0]["content"] == "You are terse."
    assert msgs[1]["content"] == "Why?"


def test_plus_operator_concats_without_mutating():
    a = ChatPromptTemplate().system("A").user("{{ x }}")
    b = ChatPromptTemplate().assistant("B")
    c = a + b
    # Originals untouched.
    assert len(a) == 2
    assert len(b) == 1
    assert len(c) == 3
    msgs = c.format({"x": "x-val"}).to_messages()
    assert msgs[0]["content"] == "A"
    assert msgs[1]["content"] == "x-val"
    assert msgs[2]["content"] == "B"


def test_three_way_layered_compose_base_role_task():
    """The headline pattern: base persona + role-specific + task-specific."""
    base = ChatPromptTemplate().system("You are a {{ persona }}.")
    role = ChatPromptTemplate().system("Specialty: {{ role }}.")
    task = ChatPromptTemplate().user("{{ task }}")
    layered = base + role + task
    msgs = layered.format({
        "persona": "polite assistant",
        "role": "code reviewer",
        "task": "review this PR",
    }).to_messages()
    assert len(msgs) == 3
    assert msgs[0]["content"] == "You are a polite assistant."
    assert msgs[1]["content"] == "Specialty: code reviewer."
    assert msgs[2]["content"] == "review this PR"


def test_extend_preserves_placeholder_position():
    pre = ChatPromptTemplate().system("sys").placeholder("hist")
    post = ChatPromptTemplate().user("{{ q }}")
    combined = pre.extend(post)
    pv = combined.format({"q": "current"})
    pv = pv.with_placeholder("hist", [{"role": "user", "content": "prior"}])
    msgs = pv.to_messages()
    assert len(msgs) == 3
    assert msgs[0]["content"] == "sys"
    assert msgs[1]["content"] == "prior"
    assert msgs[2]["content"] == "current"


def test_extend_with_empty_template_is_noop():
    a = ChatPromptTemplate().system("only")
    combined = a.extend(ChatPromptTemplate())
    assert len(combined) == len(a)


def test_compose_loaded_from_file_specs():
    """Compose templates loaded from separate JSON specs — the canonical
    iter-150 + iter-152 workflow."""
    import json
    base_spec = {"messages": [
        {"role": "system", "template": "You are {{ persona }}."}
    ]}
    role_spec = {"messages": [
        {"role": "system", "template": "Domain: {{ domain }}."}
    ]}
    task_spec = {"messages": [
        {"role": "user", "template": "{{ q }}"}
    ]}
    base = ChatPromptTemplate.from_dict(base_spec)
    role = ChatPromptTemplate.from_dict(role_spec)
    task = ChatPromptTemplate.from_dict(task_spec)

    full = base + role + task
    msgs = full.format({"persona": "expert", "domain": "rust", "q": "explain ownership"}).to_messages()
    assert msgs[0]["content"] == "You are expert."
    assert msgs[1]["content"] == "Domain: rust."
    assert msgs[2]["content"] == "explain ownership"


def test_len_reports_part_count():
    t = ChatPromptTemplate()
    assert len(t) == 0
    t = t.system("a").placeholder("h").user("{{ x }}")
    assert len(t) == 3


if __name__ == "__main__":
    import traceback
    fns = [
        test_extend_appends_other_parts,
        test_plus_operator_concats_without_mutating,
        test_three_way_layered_compose_base_role_task,
        test_extend_preserves_placeholder_position,
        test_extend_with_empty_template_is_noop,
        test_compose_loaded_from_file_specs,
        test_len_reports_part_count,
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
