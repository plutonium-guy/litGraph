"""JsonSplitter — recursive structural chunking with path preservation.
Each chunk must remain valid JSON and carry a `_path` provenance marker."""
import json

from litgraph.splitters import JsonSplitter


def test_small_object_one_chunk():
    s = JsonSplitter(max_chunk_size=200)
    chunks = s.split_text(json.dumps({"a": 1, "b": "hello"}))
    assert len(chunks) == 1
    v = json.loads(chunks[0])
    assert v["_path"] == ""
    assert v["_root"]["a"] == 1


def test_large_object_splits_at_keys_and_each_chunk_is_valid_json():
    s = JsonSplitter(max_chunk_size=120)
    big = {f"k{i}": "x" * 60 for i in range(4)}
    chunks = s.split_text(json.dumps(big))
    assert len(chunks) >= 2
    # Each chunk parses as valid JSON.
    for c in chunks:
        v = json.loads(c)
        assert "_path" in v
    # All four keys preserved.
    all_keys = []
    for c in chunks:
        v = json.loads(c)
        all_keys.extend(k for k in v.keys() if not k.startswith("_"))
    for k in ["k0", "k1", "k2", "k3"]:
        assert k in all_keys


def test_nested_path_carries_through():
    s = JsonSplitter(max_chunk_size=80)
    nested = {"outer": {"inner": {"leaf1": "x" * 40, "leaf2": "y" * 40}}}
    chunks = s.split_text(json.dumps(nested))
    paths = [json.loads(c)["_path"] for c in chunks]
    assert any("outer.inner" in p for p in paths)


def test_array_splits_on_element_boundaries():
    s = JsonSplitter(max_chunk_size=60)
    big = {"items": ["a" * 30, "b" * 30, "c" * 30, "d" * 30]}
    chunks = s.split_text(json.dumps(big))
    assert len(chunks) >= 2
    # Combined item count across chunks = 4 (no element lost).
    total_items = 0
    for c in chunks:
        v = json.loads(c)
        for k, val in v.items():
            if k.startswith("_"): continue
            if isinstance(val, list):
                total_items += len(val)
            elif isinstance(val, str):
                total_items += 1
    assert total_items == 4


def test_oversized_scalar_kept_intact():
    """A 500-char essay with 100-char budget must NOT be truncated."""
    s = JsonSplitter(max_chunk_size=100)
    big = {"essay": "z" * 500}
    chunks = s.split_text(json.dumps(big))
    essay_chunks = [c for c in chunks if "zzzzz" in c]
    assert len(essay_chunks) == 1
    v = json.loads(essay_chunks[0])
    # Find the essay value among the non-meta keys.
    essay_val = next(val for k, val in v.items() if not k.startswith("_"))
    assert isinstance(essay_val, str)
    assert len(essay_val) == 500


def test_invalid_json_passthrough():
    s = JsonSplitter(max_chunk_size=50)
    chunks = s.split_text("not { valid json,")
    assert chunks == ["not { valid json,"]


def test_min_chunk_size_clamped_to_64():
    """Pathological max_chunk_size=0 still produces working output."""
    s = JsonSplitter(max_chunk_size=0)
    assert "JsonSplitter" in repr(s)
    # Small input still works.
    chunks = s.split_text('{"a": 1}')
    assert len(chunks) == 1


if __name__ == "__main__":
    fns = [
        test_small_object_one_chunk,
        test_large_object_splits_at_keys_and_each_chunk_is_valid_json,
        test_nested_path_carries_through,
        test_array_splits_on_element_boundaries,
        test_oversized_scalar_kept_intact,
        test_invalid_json_passthrough,
        test_min_chunk_size_clamped_to_64,
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
