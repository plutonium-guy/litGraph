"""Eval harness — `run_eval(cases, target, scorers, max_parallel)`.
Concurrent dataset runner; per-case + aggregate report; pluggable scorers."""
from litgraph.evaluators import run_eval


CASES = [
    {"input": "what is 2+2?", "expected": "4"},
    {"input": "capital of france?", "expected": "Paris"},
    {"input": "largest planet?", "expected": "Jupiter"},
]


def test_perfect_target_scores_one():
    lookup = {"what is 2+2?": "4", "capital of france?": "Paris", "largest planet?": "Jupiter"}
    def target(q):
        return lookup[q]
    report = run_eval(
        cases=CASES,
        target=target,
        scorers=[{"name": "exact_match"}],
        max_parallel=4,
    )
    assert report["aggregate"]["n_cases"] == 3
    assert report["aggregate"]["n_errors"] == 0
    assert report["aggregate"]["means"]["exact_match"] == 1.0


def test_wrong_target_scores_zero():
    def target(q):
        return "wrong answer"
    report = run_eval(
        cases=CASES,
        target=target,
        scorers=[{"name": "exact_match"}],
    )
    assert report["aggregate"]["means"]["exact_match"] == 0.0


def test_target_exception_recorded_per_case_does_not_abort():
    def target(q):
        if "planet" in q:
            raise RuntimeError("upstream timeout")
        return "ok"
    report = run_eval(
        cases=CASES,
        target=target,
        scorers=[{"name": "jaccard"}],
    )
    assert report["aggregate"]["n_cases"] == 3
    assert report["aggregate"]["n_errors"] == 1
    errored = [c for c in report["per_case"] if c.get("error")]
    assert len(errored) == 1
    assert "upstream timeout" in errored[0]["error"]
    assert errored[0]["output"] is None


def test_per_case_ordering_preserved():
    def target(q):
        return q
    report = run_eval(cases=CASES, target=target, scorers=[{"name": "jaccard"}])
    inputs = [c["input"] for c in report["per_case"]]
    assert inputs == [c["input"] for c in CASES]


def test_multiple_scorers_aggregate_separately():
    def target(q):
        return "Paris"
    report = run_eval(
        cases=CASES,
        target=target,
        scorers=[{"name": "exact_match"}, {"name": "jaccard"}],
    )
    assert "exact_match" in report["aggregate"]["means"]
    assert "jaccard" in report["aggregate"]["means"]
    em = report["aggregate"]["means"]["exact_match"]
    assert abs(em - 1.0/3.0) < 1e-6  # only "Paris" case matches


def test_contains_all_scorer_with_required_substrings():
    def target(q):
        return "the answer is 42 percent"
    cases = [{"input": "q"}]
    report = run_eval(
        cases=cases,
        target=target,
        scorers=[{"name": "contains_all", "required": ["42", "percent"]}],
    )
    assert report["aggregate"]["means"]["contains_all"] == 1.0


def test_regex_scorer_with_pattern():
    def target(q):
        return "answer: 42"
    cases = [{"input": "q"}]
    report = run_eval(
        cases=cases,
        target=target,
        scorers=[{"name": "regex", "pattern": r"\d+"}],
    )
    assert report["aggregate"]["means"]["regex"] == 1.0


def test_levenshtein_scorer_emits_continuous_score():
    def target(q):
        return "Paris, France"
    cases = [{"input": "q", "expected": "Paris"}]
    report = run_eval(
        cases=cases,
        target=target,
        scorers=[{"name": "levenshtein"}],
    )
    s = report["aggregate"]["means"]["levenshtein"]
    assert 0.0 < s < 1.0


def test_unknown_scorer_raises_value_error():
    try:
        run_eval(
            cases=CASES,
            target=lambda q: q,
            scorers=[{"name": "made_up"}],
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "made_up" in str(e)


def test_metadata_round_trips_to_per_case():
    cases = [{"input": "q1", "expected": "a", "metadata": {"tag": "easy", "id": 7}}]
    report = run_eval(
        cases=cases,
        target=lambda q: "a",
        scorers=[{"name": "exact_match"}],
    )
    assert report["per_case"][0]["metadata"] == {"tag": "easy", "id": 7}


def test_empty_dataset_returns_empty_report():
    report = run_eval(cases=[], target=lambda q: "x", scorers=[{"name": "exact_match"}])
    assert report["per_case"] == []
    assert report["aggregate"]["n_cases"] == 0


def test_runs_to_completion_with_many_cases_and_concurrency():
    """8 cases with max_parallel=4 → all complete + correct.
    NOTE: Python `time.sleep` holds the GIL, so true wall-clock parallelism
    requires the target to call into GIL-releasing Rust code (e.g.
    chat.invoke()). For unit-test-level "ran without deadlock," the
    correctness assertion is what matters."""
    cases = [{"input": f"q{i}", "expected": f"q{i}"} for i in range(8)]
    def target(q):
        return q
    report = run_eval(cases=cases, target=target, scorers=[{"name": "exact_match"}], max_parallel=4)
    assert report["aggregate"]["n_cases"] == 8
    assert report["aggregate"]["means"]["exact_match"] == 1.0
    # Per-case ordering preserved despite concurrency.
    assert [c["input"] for c in report["per_case"]] == [f"q{i}" for i in range(8)]


if __name__ == "__main__":
    import traceback
    fns = [
        test_perfect_target_scores_one,
        test_wrong_target_scores_zero,
        test_target_exception_recorded_per_case_does_not_abort,
        test_per_case_ordering_preserved,
        test_multiple_scorers_aggregate_separately,
        test_contains_all_scorer_with_required_substrings,
        test_regex_scorer_with_pattern,
        test_levenshtein_scorer_emits_continuous_score,
        test_unknown_scorer_raises_value_error,
        test_metadata_round_trips_to_per_case,
        test_empty_dataset_returns_empty_report,
        test_runs_to_completion_with_many_cases_and_concurrency,
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
