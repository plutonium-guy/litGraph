"""ReAct text-mode output parser — for non-tool-calling models (Ollama,
local models, base-completion checkpoints) that emit Thought/Action/
Action Input/Final Answer prose. Complements the iter-13 ReactAgent
which uses provider-native tool-calling APIs."""
from litgraph.parsers import parse_react_step


def _raises_value_error(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError:
        return True
    return False


def test_action_with_json_input():
    text = (
        "Thought: I need to check the weather.\n"
        "Action: get_weather\n"
        "Action Input: {\"city\": \"Paris\"}"
    )
    step = parse_react_step(text)
    assert step["kind"] == "action"
    assert step["thought"] == "I need to check the weather."
    assert step["tool"] == "get_weather"
    assert step["input"] == {"city": "Paris"}


def test_action_with_string_input_when_not_json():
    """Some models (llama2-base) emit raw strings, not JSON."""
    text = "Action: search\nAction Input: Paris weather today"
    step = parse_react_step(text)
    assert step["kind"] == "action"
    assert step["input"] == "Paris weather today"


def test_action_input_strips_code_fence():
    """Instruction-tuned models wrap JSON in ```json...``` fences."""
    text = (
        "Action: get_weather\n"
        "Action Input: ```json\n"
        '{"city": "Paris"}\n'
        "```"
    )
    step = parse_react_step(text)
    assert step["input"] == {"city": "Paris"}


def test_final_answer_terminates_reasoning():
    text = "Thought: I have what I need.\nFinal Answer: It's 15°C and raining."
    step = parse_react_step(text)
    assert step["kind"] == "final"
    assert step["thought"] == "I have what I need."
    assert step["answer"] == "It's 15°C and raining."


def test_final_answer_wins_when_llm_emits_both():
    """LLM hedges — Final Answer is authoritative over Action."""
    text = (
        "Thought: maybe search\n"
        "Action: search\n"
        "Action Input: foo\n"
        "Final Answer: Actually I know — 42."
    )
    step = parse_react_step(text)
    assert step["kind"] == "final"
    assert step["answer"] == "Actually I know — 42."


def test_markdown_bold_labels_accepted():
    """Some models emit **Action:** instead of Action:."""
    text = (
        "**Thought:** check weather\n"
        "**Action:** get_weather\n"
        '**Action Input:** {"city": "Paris"}'
    )
    step = parse_react_step(text)
    assert step["kind"] == "action"
    assert step["tool"] == "get_weather"
    assert step["input"] == {"city": "Paris"}


def test_missing_thought_still_parses():
    """Some models skip Thought and go straight to Action."""
    text = "Action: search\nAction Input: foo"
    step = parse_react_step(text)
    assert step["thought"] is None
    assert step["tool"] == "search"


def test_case_insensitive_labels():
    text = "thought: hmm\naction: search\naction input: foo"
    step = parse_react_step(text)
    assert step["kind"] == "action"
    assert step["tool"] == "search"


def test_no_action_and_no_final_raises_value_error():
    assert _raises_value_error(parse_react_step, "I'm thinking about this...")


def test_empty_action_value_raises():
    assert _raises_value_error(parse_react_step, "Action:\nAction Input: foo")


def test_multiline_thought_preserved():
    text = (
        "Thought: Let me think.\n"
        "First, check weather.\n"
        "Then decide what to wear.\n"
        "Action: get_weather\n"
        "Action Input: {}"
    )
    step = parse_react_step(text)
    t = step["thought"]
    assert "First, check" in t
    assert "Then decide" in t


def test_real_world_trace_with_observations_still_parses_final():
    """After the runner feeds back Observations, the LLM continues with
    Thought + Final Answer. Parser picks out the Final Answer step."""
    text = (
        "Thought: first\n"
        "Action: search\n"
        "Action Input: Paris\n"
        "Observation: results about Paris\n"
        "Thought: now I know\n"
        "Final Answer: Paris is the capital of France."
    )
    step = parse_react_step(text)
    assert step["kind"] == "final"
    assert step["answer"] == "Paris is the capital of France."


if __name__ == "__main__":
    import traceback

    fns = [
        test_action_with_json_input,
        test_action_with_string_input_when_not_json,
        test_action_input_strips_code_fence,
        test_final_answer_terminates_reasoning,
        test_final_answer_wins_when_llm_emits_both,
        test_markdown_bold_labels_accepted,
        test_missing_thought_still_parses,
        test_case_insensitive_labels,
        test_no_action_and_no_final_raises_value_error,
        test_empty_action_value_raises,
        test_multiline_thought_preserved,
        test_real_world_trace_with_observations_still_parses_final,
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
