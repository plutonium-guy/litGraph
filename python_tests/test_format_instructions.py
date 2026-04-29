"""Format-instructions helpers — prompt snippets paired with each output
parser. LangChain `get_format_instructions()` parity: user concatenates
the snippet into their prompt, LLM outputs parseable format, parser
decodes it."""
from litgraph.parsers import (
    boolean_format_instructions,
    comma_list_format_instructions,
    markdown_list_format_instructions,
    numbered_list_format_instructions,
    parse_boolean,
    parse_comma_list,
    parse_markdown_list,
    parse_numbered_list,
    parse_react_step,
    parse_xml_tags,
    react_format_instructions,
    xml_format_instructions,
)


def test_comma_list_instructions_paired_with_parser():
    """The instruction string plus a simulated-LLM output round-trips."""
    instr = comma_list_format_instructions()
    assert "comma-separated" in instr
    # Simulated LLM output following the instruction.
    simulated = "apple, banana, cherry"
    assert parse_comma_list(simulated) == ["apple", "banana", "cherry"]


def test_numbered_list_instructions_paired_with_parser():
    instr = numbered_list_format_instructions()
    assert "1." in instr
    simulated = "1. first\n2. second\n3. third"
    assert parse_numbered_list(simulated) == ["first", "second", "third"]


def test_markdown_list_instructions_paired_with_parser():
    instr = markdown_list_format_instructions()
    assert "- first item" in instr
    simulated = "- first\n- second"
    assert parse_markdown_list(simulated) == ["first", "second"]


def test_boolean_instructions_paired_with_parser():
    instr = boolean_format_instructions()
    assert "yes or no" in instr.lower()
    assert parse_boolean("yes") is True
    assert parse_boolean("no") is False


def test_xml_instructions_list_each_requested_tag():
    tags = ["thinking", "answer"]
    instr = xml_format_instructions(tags)
    assert "<thinking>...</thinking>" in instr
    assert "<answer>...</answer>" in instr
    # Entity-escape guidance present.
    assert "&lt;" in instr
    # Round-trip with parser.
    simulated = "<thinking>reasoning here</thinking><answer>42</answer>"
    parsed = parse_xml_tags(simulated, tags)
    assert parsed["answer"] == "42"


def test_react_instructions_list_tools_and_grammar():
    tools = [
        "get_weather: fetch weather for a city",
        "web_search: search the web for a query",
    ]
    instr = react_format_instructions(tools)
    assert "get_weather" in instr
    assert "web_search" in instr
    assert "Thought:" in instr
    assert "Action:" in instr
    assert "Action Input:" in instr
    assert "Final Answer:" in instr
    # Round-trip with the parser on a simulated response.
    simulated = (
        "Thought: check the weather\n"
        "Action: get_weather\n"
        'Action Input: {"city": "Paris"}'
    )
    step = parse_react_step(simulated)
    assert step["tool"] == "get_weather"


def test_react_instructions_handles_no_tools():
    instr = react_format_instructions([])
    assert "no tools available" in instr.lower()
    assert "Final Answer:" in instr


def test_instructions_are_pure_strings_no_mutation():
    """Each helper is deterministic — two calls return the same string."""
    assert comma_list_format_instructions() == comma_list_format_instructions()
    assert xml_format_instructions(["a"]) == xml_format_instructions(["a"])


if __name__ == "__main__":
    import traceback

    fns = [
        test_comma_list_instructions_paired_with_parser,
        test_numbered_list_instructions_paired_with_parser,
        test_markdown_list_instructions_paired_with_parser,
        test_boolean_instructions_paired_with_parser,
        test_xml_instructions_list_each_requested_tag,
        test_react_instructions_list_tools_and_grammar,
        test_react_instructions_handles_no_tools,
        test_instructions_are_pure_strings_no_mutation,
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
