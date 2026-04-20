"""CsvLoader: real CSV/TSV files → Documents with content + metadata."""
import os
import tempfile

from litgraph.loaders import CsvLoader


def _write(name: str, content: str) -> str:
    fd, path = tempfile.mkstemp(prefix="csv-", suffix=name)
    os.write(fd, content.encode())
    os.close(fd)
    return path


def test_csv_loader_with_content_column_uses_that_value():
    p = _write(".csv",
        "id,title,body,author\n"
        "1,Hello,This is the first doc,alice\n"
        "2,World,Second one,bob\n")
    try:
        docs = CsvLoader(p, content_column="body").load()
        assert len(docs) == 2
        assert docs[0]["content"] == "This is the first doc"
        assert docs[1]["content"] == "Second one"
        assert docs[0]["metadata"]["title"] == "Hello"
        assert docs[0]["metadata"]["author"] == "alice"
        assert "body" not in docs[0]["metadata"]  # content column excluded
        assert docs[0]["id"].endswith("#0")
        assert docs[1]["id"].endswith("#1")
    finally:
        os.unlink(p)


def test_csv_loader_no_content_column_joins_kv_pairs():
    p = _write(".csv", "k,v\nfoo,1\nbar,2\n")
    try:
        docs = CsvLoader(p).load()
        assert len(docs) == 2
        assert docs[0]["content"] == "k=foo\nv=1"
        assert docs[1]["content"] == "k=bar\nv=2"
    finally:
        os.unlink(p)


def test_csv_loader_max_rows():
    p = _write(".csv", "n\n1\n2\n3\n4\n5\n")
    try:
        docs = CsvLoader(p, max_rows=2).load()
        assert len(docs) == 2
    finally:
        os.unlink(p)


def test_csv_loader_tsv_delimiter():
    p = _write(".tsv", "a\tb\n1\t2\n")
    try:
        docs = CsvLoader(p, content_column="b", delimiter="\t").load()
        assert len(docs) == 1
        assert docs[0]["content"] == "2"
        assert docs[0]["metadata"]["a"] == "1"
    finally:
        os.unlink(p)


def test_csv_loader_unknown_content_column_raises():
    p = _write(".csv", "a,b\n1,2\n")
    try:
        try:
            CsvLoader(p, content_column="nope").load()
        except RuntimeError as e:
            assert "'nope' not found" in str(e)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        os.unlink(p)


def test_csv_loader_invalid_multibyte_delimiter_raises():
    try:
        CsvLoader("/tmp/whatever.csv", delimiter=",,")
    except ValueError as e:
        assert "single byte" in str(e)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    fns = [
        test_csv_loader_with_content_column_uses_that_value,
        test_csv_loader_no_content_column_joins_kv_pairs,
        test_csv_loader_max_rows,
        test_csv_loader_tsv_delimiter,
        test_csv_loader_unknown_content_column_raises,
        test_csv_loader_invalid_multibyte_delimiter_raises,
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
