"""JsonLoader: bare object / array / nested-via-pointer modes."""
import json
import os
import tempfile

from litgraph.loaders import JsonLoader


def _write(suffix: str, content: str) -> str:
    fd, path = tempfile.mkstemp(prefix="json-", suffix=suffix)
    os.write(fd, content.encode())
    os.close(fd)
    return path


def test_array_with_content_field_yields_one_doc_per_element():
    p = _write(".json",
        json.dumps([
            {"title": "a", "body": "alpha"},
            {"title": "b", "body": "beta"},
        ]))
    try:
        docs = JsonLoader(p, content_field="body").load()
        assert [d["content"] for d in docs] == ["alpha", "beta"]
        assert docs[0]["metadata"]["title"] == "a"
        # docs_to_pylist round-trips numeric metadata as strings (existing helper).
        assert docs[0]["metadata"]["index"] == "0"
    finally:
        os.unlink(p)


def test_array_without_content_field_serializes_each_row():
    p = _write(".json", json.dumps([{"k": 1}, {"k": 2}]))
    try:
        docs = JsonLoader(p).load()
        assert json.loads(docs[0]["content"]) == {"k": 1}
        assert json.loads(docs[1]["content"]) == {"k": 2}
    finally:
        os.unlink(p)


def test_nested_array_via_pointer():
    p = _write(".json", json.dumps({
        "meta": {"v": 1},
        "results": {"items": [{"x": "first"}, {"x": "second"}]},
    }))
    try:
        docs = JsonLoader(p, pointer="results.items", content_field="x").load()
        assert [d["content"] for d in docs] == ["first", "second"]
    finally:
        os.unlink(p)


def test_pointer_supports_array_index():
    p = _write(".json", json.dumps({
        "pages": [{"items": ["a", "b", "c"]}, {"items": ["d"]}],
    }))
    try:
        # pages[1].items → ["d"]
        docs = JsonLoader(p, pointer="pages.1.items").load()
        assert len(docs) == 1
        assert docs[0]["content"] == '"d"'
    finally:
        os.unlink(p)


def test_unknown_pointer_raises():
    p = _write(".json", json.dumps({"a": 1}))
    try:
        try:
            JsonLoader(p, pointer="nope.deeper").load()
        except RuntimeError as e:
            assert "did not resolve" in str(e)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        os.unlink(p)


def test_bare_object_yields_one_doc():
    p = _write(".json", json.dumps({"name": "X", "weight": 1.5}))
    try:
        docs = JsonLoader(p).load()
        assert len(docs) == 1
        assert "X" in docs[0]["content"]
        assert docs[0]["metadata"]["name"] == "X"
    finally:
        os.unlink(p)


if __name__ == "__main__":
    fns = [
        test_array_with_content_field_yields_one_doc_per_element,
        test_array_without_content_field_serializes_each_row,
        test_nested_array_via_pointer,
        test_pointer_supports_array_index,
        test_unknown_pointer_raises,
        test_bare_object_yields_one_doc,
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
