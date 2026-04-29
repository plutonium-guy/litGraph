"""parse_markdown_tables — extract markdown tables → list of dicts.
LLM output parsing primitive."""
from litgraph.parsers import parse_markdown_tables


def test_basic_three_column():
    text = """
| name | score | rank |
|------|-------|------|
| alice | 95 | 1 |
| bob   | 80 | 2 |
"""
    tables = parse_markdown_tables(text)
    assert len(tables) == 1
    t = tables[0]
    assert t["headers"] == ["name", "score", "rank"]
    assert len(t["rows"]) == 2
    assert t["rows"][0] == {"name": "alice", "score": "95", "rank": "1"}
    assert t["rows"][1]["rank"] == "2"


def test_no_tables_in_text_returns_empty_list():
    assert parse_markdown_tables("just plain text") == []
    assert parse_markdown_tables("") == []


def test_multiple_tables_in_one_text():
    text = """
First report:

| a | b |
|---|---|
| 1 | 2 |

Some prose.

| x | y |
|---|---|
| u | v |
| w | z |
"""
    tables = parse_markdown_tables(text)
    assert len(tables) == 2
    assert tables[0]["headers"] == ["a", "b"]
    assert len(tables[0]["rows"]) == 1
    assert tables[1]["headers"] == ["x", "y"]
    assert len(tables[1]["rows"]) == 2


def test_table_without_outer_pipes():
    text = """
name | score
-----|------
alice | 95
bob | 80
"""
    tables = parse_markdown_tables(text)
    assert len(tables) == 1
    assert tables[0]["rows"][0]["name"] == "alice"


def test_alignment_separator_accepted():
    text = """
| h1 | h2 |
|:---|:---:|
| a | b |
"""
    tables = parse_markdown_tables(text)
    assert tables[0]["rows"][0] == {"h1": "a", "h2": "b"}


def test_pipes_inside_backticks_preserved():
    text = """
| code | desc |
|------|------|
| `a|b` | the combined |
"""
    tables = parse_markdown_tables(text)
    assert tables[0]["rows"][0]["code"] == "`a|b`"
    assert tables[0]["rows"][0]["desc"] == "the combined"


def test_ragged_rows_extras_dropped_missing_become_empty():
    text = """
| a | b |
|---|---|
| 1 | 2 | 3 |
| only |
"""
    tables = parse_markdown_tables(text)
    rows = tables[0]["rows"]
    assert rows[0]["a"] == "1"
    assert rows[0]["b"] == "2"
    # Extra cell "3" dropped (no header).
    assert rows[1]["a"] == "only"
    assert rows[1]["b"] == ""  # missing → empty string


def test_header_only_table_zero_rows():
    text = """
| h |
|---|
"""
    tables = parse_markdown_tables(text)
    assert len(tables) == 1
    assert tables[0]["headers"] == ["h"]
    assert tables[0]["rows"] == []


def test_header_without_separator_not_a_table():
    text = """
| a | b |
| c | d |
"""
    assert parse_markdown_tables(text) == []


def test_invalid_separator_with_letters_not_a_table():
    text = """
| a | b |
| not | a sep |
| 1 | 2 |
"""
    assert parse_markdown_tables(text) == []


def test_cell_whitespace_trimmed():
    text = """
|   col1   |   col2   |
|---|---|
|   alpha   |   beta   |
"""
    t = parse_markdown_tables(text)[0]
    assert t["headers"] == ["col1", "col2"]
    assert t["rows"][0]["col1"] == "alpha"


def test_numeric_strings_preserved_no_coercion():
    """Caller does type coercion downstream — we never auto-parse."""
    text = """
| n |
|---|
| 42 |
| 3.14 |
| true |
| null |
"""
    rows = parse_markdown_tables(text)[0]["rows"]
    assert rows[0]["n"] == "42"
    assert rows[1]["n"] == "3.14"
    assert rows[2]["n"] == "true"
    assert rows[3]["n"] == "null"


def test_realistic_llm_output_format():
    """LLMs often emit prose around tables. Parser should find the table
    in the middle without choking on surrounding markdown."""
    text = """
Here are the top results from your query:

| product | price | in_stock |
|---------|-------|----------|
| Widget  | $10   | yes      |
| Gadget  | $25   | no       |
| Gizmo   | $5    | yes      |

Note: prices exclude shipping. Let me know if you'd like me to filter
these in any particular way.
"""
    tables = parse_markdown_tables(text)
    assert len(tables) == 1
    assert len(tables[0]["rows"]) == 3
    assert tables[0]["rows"][0]["product"] == "Widget"
    assert tables[0]["rows"][2]["in_stock"] == "yes"


if __name__ == "__main__":
    import traceback
    fns = [
        test_basic_three_column,
        test_no_tables_in_text_returns_empty_list,
        test_multiple_tables_in_one_text,
        test_table_without_outer_pipes,
        test_alignment_separator_accepted,
        test_pipes_inside_backticks_preserved,
        test_ragged_rows_extras_dropped_missing_become_empty,
        test_header_only_table_zero_rows,
        test_header_without_separator_not_a_table,
        test_invalid_separator_with_letters_not_a_table,
        test_cell_whitespace_trimmed,
        test_numeric_strings_preserved_no_coercion,
        test_realistic_llm_output_format,
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
