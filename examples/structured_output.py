"""Coerce a chat model's output into a typed shape.

`with_structured_output(Schema)` wraps a `ChatModel` so the response
is validated + decoded into your schema (Pydantic / dataclass /
TypedDict / raw JSON Schema). This example uses MockChatModel for
offline determinism — swap in any real ChatModel and the rest of
the code is unchanged.

Run:  python examples/structured_output.py
"""
from dataclasses import dataclass

from litgraph.testing import MockChatModel


@dataclass
class Verdict:
    answer: str
    confidence: float


def main() -> None:
    # Mock returns the JSON the schema expects. With a real model the
    # wrapper enforces this shape via tool-call / response-format.
    m = MockChatModel(replies=[
        {"role": "assistant",
         "content": '{"answer": "yes", "confidence": 0.92}',
         "tool_calls": [],
         "usage": {"prompt": 0, "completion": 0, "total": 0}},
    ])

    structured = m.with_structured_output(Verdict)
    raw = structured.invoke([{"role": "user", "content": "Is the sky blue?"}])

    # The mock doesn't actually decode; with a real ChatModel
    # `with_structured_output` returns a typed instance. We parse the
    # mock's content here to demonstrate the shape:
    import json
    decoded = Verdict(**json.loads(raw["content"]))
    print(f"answer={decoded.answer!r}, confidence={decoded.confidence}")
    assert decoded.answer == "yes"
    assert decoded.confidence == 0.92


if __name__ == "__main__":
    main()
