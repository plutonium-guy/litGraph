"""Run an eval over a tiny golden set with `litgraph.recipes.eval`.

Uses MockChatModel so this runs offline with no API key. Swap in a
real ChatModel (`OpenAIChat(model="gpt-5")`) to evaluate a real
provider — the harness call doesn't change.

Run:  python examples/eval_harness.py
"""
from litgraph.recipes import eval as run_eval
from litgraph.testing import MockChatModel


def main() -> None:
    # In real life: `model = OpenAIChat(model="gpt-5")`.
    model = MockChatModel(replies=["Paris", "Berlin", "Tokyo"])

    def predict(question: str) -> str:
        return model.invoke([{"role": "user", "content": question}])["content"]

    cases = [
        {"input": "Capital of France?", "expected": "Paris"},
        {"input": "Capital of Germany?", "expected": "Berlin"},
        {"input": "Capital of Japan?", "expected": "Tokyo"},
    ]

    report = run_eval(predict, cases)

    print("=== aggregate ===")
    for k, v in report["aggregate"]["means"].items():
        print(f"  {k}: {v:.3f}")
    print(f"  n_cases: {report['aggregate']['n_cases']}")
    print(f"  n_errors: {report['aggregate']['n_errors']}")

    print("\n=== per case ===")
    for case in report["per_case"]:
        scores = ", ".join(
            f"{k}={v['score']:.2f}" for k, v in case["scores"].items()
        )
        print(f"  {case['input']!r:40s} → {case['output']!r:10s}  [{scores}]")


if __name__ == "__main__":
    main()
