"""load_concurrent — bounded-concurrency multi-loader fan-out.

Each loader's blocking `load()` runs on the Tokio blocking pool;
output is aligned to input order and per-loader failures don't tank
the whole batch by default."""
import os
import tempfile

from litgraph.loaders import (
    CsvLoader, JsonLinesLoader, JsonLoader, MarkdownLoader, TextLoader,
    load_concurrent,
)


def _write(dir_, name, content):
    p = os.path.join(dir_, name)
    with open(p, "w") as f:
        f.write(content)
    return p


def test_load_concurrent_returns_per_loader_results():
    with tempfile.TemporaryDirectory() as d:
        a = _write(d, "a.txt", "alpha")
        b = _write(d, "b.txt", "beta")
        c = _write(d, "c.txt", "gamma")
        loaders = [TextLoader(a), TextLoader(b), TextLoader(c)]
        results = load_concurrent(loaders, max_concurrency=2)
    assert len(results) == 3
    contents = [r[0]["content"] for r in results]
    assert contents == ["alpha", "beta", "gamma"]


def test_load_concurrent_alignment_with_mixed_loader_types():
    with tempfile.TemporaryDirectory() as d:
        t = _write(d, "x.txt", "from text")
        m = _write(d, "y.md", "# title\nbody")
        loaders = [TextLoader(t), MarkdownLoader(m)]
        results = load_concurrent(loaders, max_concurrency=4)
    assert len(results) == 2
    assert "from text" in results[0][0]["content"]
    # MarkdownLoader returns docs containing the body.
    assert results[1]


def test_load_concurrent_isolates_per_loader_errors():
    """A non-existent file errors; the other loader still produces docs.
    Output slot for the failed loader is a single-element list with
    `{"error": "..."}`."""
    with tempfile.TemporaryDirectory() as d:
        good = _write(d, "g.txt", "content")
        bad = os.path.join(d, "missing.txt")  # never created
        loaders = [TextLoader(good), TextLoader(bad)]
        results = load_concurrent(loaders, max_concurrency=4)
    assert len(results) == 2
    assert results[0][0]["content"] == "content"
    assert "error" in results[1][0]


def test_load_concurrent_fail_fast_raises():
    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, "missing.txt")
        loaders = [TextLoader(bad)]
        try:
            load_concurrent(loaders, max_concurrency=2, fail_fast=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")


def test_load_concurrent_empty_list_returns_empty():
    out = load_concurrent([], max_concurrency=4)
    assert out == []


def test_load_concurrent_supports_jsonl_and_csv():
    with tempfile.TemporaryDirectory() as d:
        jl = _write(d, "x.jsonl", '{"content": "row1"}\n{"content": "row2"}\n')
        csv_path = _write(d, "y.csv", "col1,col2\nv1,v2\n")
        loaders = [JsonLinesLoader(jl, content_field="content"), CsvLoader(csv_path)]
        results = load_concurrent(loaders, max_concurrency=4)
    assert len(results) == 2
    assert len(results[0]) == 2  # 2 jsonl rows
    # CSV produces 1 doc (header + 1 row).
    assert len(results[1]) >= 1


def test_load_concurrent_rejects_unsupported_loader():
    """Network-loader types aren't accepted yet — should raise."""
    with tempfile.TemporaryDirectory() as d:
        good = _write(d, "g.txt", "content")
        try:
            load_concurrent([TextLoader(good), "not a loader"], max_concurrency=4)
        except (RuntimeError, TypeError):
            pass
        else:
            raise AssertionError("expected RuntimeError/TypeError")


def test_load_concurrent_supports_json_loader():
    with tempfile.TemporaryDirectory() as d:
        j = _write(d, "x.json", '{"content": "from json"}')
        loaders = [JsonLoader(j, content_field="content")]
        results = load_concurrent(loaders, max_concurrency=2)
    assert len(results) == 1


if __name__ == "__main__":
    fns = [
        test_load_concurrent_returns_per_loader_results,
        test_load_concurrent_alignment_with_mixed_loader_types,
        test_load_concurrent_isolates_per_loader_errors,
        test_load_concurrent_fail_fast_raises,
        test_load_concurrent_empty_list_returns_empty,
        test_load_concurrent_supports_jsonl_and_csv,
        test_load_concurrent_rejects_unsupported_loader,
        test_load_concurrent_supports_json_loader,
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
