"""Map-reduce summarise a long document with DeepSeek via
`litgraph.recipes.summarize`.

Run:
    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_summarize.py
"""
from __future__ import annotations

import os

from litgraph.providers import OpenAIChat
from litgraph.recipes import summarize


SAMPLE_DOC = """\
The mitochondrion is a double-membrane-bound organelle found in most
eukaryotic cells. Mitochondria use aerobic respiration to generate
adenosine triphosphate (ATP), which is the primary source of chemical
energy that supplies most cellular activities. They are sometimes
described as "the powerhouse of the cell".

Photosynthesis is the process by which green plants and some other
organisms use sunlight to synthesise foods from carbon dioxide and
water. Photosynthesis in plants generally involves the green pigment
chlorophyll and generates oxygen as a byproduct.

The cell cycle is a series of events that take place in a cell as it
grows and divides. The cycle has two main phases: interphase (G1, S, G2)
and the M (mitotic) phase. Errors in cell-cycle regulation can lead to
cancer.
"""


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY")

    chat = OpenAIChat(
        api_key=api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )

    out = summarize(SAMPLE_DOC, model=chat, chunk_size=400, chunk_overlap=50)
    print(f"Final summary ({len(out['summary'])} chars):")
    print(out["summary"])
    print()
    print(f"Per-chunk summaries: {len(out['chunk_summaries'])}")


if __name__ == "__main__":
    main()
