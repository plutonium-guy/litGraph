"""JupyterNotebookLoader — load .ipynb files. Per-cell by default;
optional concat-into-one-doc, cell-type filtering, output inclusion."""
import json
import os
import tempfile

from litgraph.loaders import JupyterNotebookLoader


SAMPLE = {
    "metadata": {"kernelspec": {"name": "python3", "language": "python"}},
    "cells": [
        {"cell_type": "markdown", "source": ["# Title\n", "intro text"]},
        {"cell_type": "code", "source": "import os\nprint(os.getcwd())\n",
         "outputs": [{"output_type": "stream", "text": "/Users/x\n"}]},
        {"cell_type": "code", "source": ["x = 1\n", "x + 1\n"],
         "outputs": [{"output_type": "execute_result",
                      "data": {"text/plain": "2"}}]},
        {"cell_type": "raw", "source": "raw cell skipped"},
    ],
}


def _write_nb(content):
    fd, path = tempfile.mkstemp(suffix=".ipynb")
    os.write(fd, json.dumps(content).encode())
    os.close(fd)
    return path


def test_loads_per_cell_by_default():
    path = _write_nb(SAMPLE)
    try:
        docs = JupyterNotebookLoader(path).load()
        # markdown + 2 code = 3 (raw skipped).
        assert len(docs) == 3
        types = [d["metadata"]["cell_type"] for d in docs]
        assert types == ["markdown", "code", "code"]
    finally:
        os.unlink(path)


def test_outputs_excluded_by_default():
    path = _write_nb(SAMPLE)
    try:
        docs = JupyterNotebookLoader(path).load()
        for d in docs:
            assert "--- output ---" not in d["content"]
            assert "/Users/x" not in d["content"]
    finally:
        os.unlink(path)


def test_outputs_included_when_opt_in():
    path = _write_nb(SAMPLE)
    try:
        docs = JupyterNotebookLoader(path, include_outputs=True).load()
        code1 = next(d for d in docs if "import os" in d["content"])
        assert "--- output ---" in code1["content"]
        assert "/Users/x" in code1["content"]
    finally:
        os.unlink(path)


def test_cell_index_metadata_preserved():
    path = _write_nb(SAMPLE)
    try:
        docs = JupyterNotebookLoader(path).load()
        # Document metadata values are stringified during the Python
        # conversion (existing loader binding behavior). Compare as str.
        assert str(docs[0]["metadata"]["cell_index"]) == "0"
        assert str(docs[1]["metadata"]["cell_index"]) == "1"
        assert str(docs[2]["metadata"]["cell_index"]) == "2"
    finally:
        os.unlink(path)


def test_cell_types_filter():
    path = _write_nb(SAMPLE)
    try:
        docs = JupyterNotebookLoader(path, cell_types=["markdown"]).load()
        assert len(docs) == 1
        assert docs[0]["metadata"]["cell_type"] == "markdown"
    finally:
        os.unlink(path)


def test_concat_mode_emits_single_doc():
    path = _write_nb(SAMPLE)
    try:
        docs = JupyterNotebookLoader(path, concat_into_one_doc=True).load()
        assert len(docs) == 1
        assert str(docs[0]["metadata"]["n_cells"]) == "3"
        assert "[cell 0 — markdown]" in docs[0]["content"]
        assert "[cell 1 — code]" in docs[0]["content"]
    finally:
        os.unlink(path)


def test_language_metadata_extracted():
    path = _write_nb(SAMPLE)
    try:
        docs = JupyterNotebookLoader(path).load()
        for d in docs:
            assert d["metadata"]["language"] == "python"
    finally:
        os.unlink(path)


def test_document_id_includes_cell_index():
    path = _write_nb(SAMPLE)
    try:
        docs = JupyterNotebookLoader(path).load()
        for d in docs:
            assert f"#cell-{d['metadata']['cell_index']}" in d["id"]
    finally:
        os.unlink(path)


def test_empty_notebook_returns_empty_list():
    path = _write_nb({"cells": [], "metadata": {}})
    try:
        docs = JupyterNotebookLoader(path).load()
        assert docs == []
    finally:
        os.unlink(path)


def test_invalid_json_raises_runtime_error():
    fd, path = tempfile.mkstemp(suffix=".ipynb")
    os.write(fd, b"not valid json")
    os.close(fd)
    try:
        try:
            JupyterNotebookLoader(path).load()
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            pass
    finally:
        os.unlink(path)


def test_error_output_surfaces_as_text():
    nb = {
        "metadata": {},
        "cells": [
            {"cell_type": "code", "source": "1/0",
             "outputs": [{"output_type": "error",
                          "ename": "ZeroDivisionError",
                          "evalue": "division by zero",
                          "traceback": ["..."]}]},
        ],
    }
    path = _write_nb(nb)
    try:
        docs = JupyterNotebookLoader(path, include_outputs=True).load()
        assert "<error: ZeroDivisionError: division by zero>" in docs[0]["content"]
    finally:
        os.unlink(path)


if __name__ == "__main__":
    import traceback
    fns = [
        test_loads_per_cell_by_default,
        test_outputs_excluded_by_default,
        test_outputs_included_when_opt_in,
        test_cell_index_metadata_preserved,
        test_cell_types_filter,
        test_concat_mode_emits_single_doc,
        test_language_metadata_extracted,
        test_document_id_includes_cell_index,
        test_empty_notebook_returns_empty_list,
        test_invalid_json_raises_runtime_error,
        test_error_output_surfaces_as_text,
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
