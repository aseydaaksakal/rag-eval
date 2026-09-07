"""Run the evaluation as a pytest case.

    pytest examples/test_rag_regression.py

The same check `rag-eval compare` performs, expressed as a test so it lands in
the suite your team already runs. Regenerate the baseline deliberately, in its
own commit, when a change is meant to move the numbers:

    rag-eval run --dataset examples/questions.jsonl \
        --pipeline examples.pipeline:build --out results.json
    rag-eval baseline save results.json --to examples/baseline.json
"""

from pathlib import Path

import pytest

from rag_eval import compare, evaluate, load_baseline, load_dataset, save_baseline, to_text
from rag_eval.baseline import load_thresholds

HERE = Path(__file__).parent
BASELINE = HERE / "baseline.json"
METRICS = ["recall@3", "mrr@3", "groundedness", "hallucination", "refusal_rate"]


@pytest.fixture(scope="module")
def run():
    from pipeline import build

    return evaluate(load_dataset(HERE / "questions.jsonl"), build(), metrics=METRICS)


def test_no_example_failed_to_run(run):
    assert not run.failures, [f.error for f in run.failures]


def test_absolute_thresholds_hold(run):
    result = compare(run, None, load_thresholds(HERE / "rag-eval.toml"))
    assert result.passed, "\n" + to_text(run, result)


def test_no_regression_against_the_baseline(run):
    if not BASELINE.exists():
        save_baseline(run, BASELINE, label="bootstrap")
        pytest.skip(f"wrote a first baseline to {BASELINE}; re-run to compare")
    result = compare(run, load_baseline(BASELINE), load_thresholds(HERE / "rag-eval.toml"))
    assert result.passed, "\n" + to_text(run, result)


def test_a_generator_that_ignores_its_context_is_caught(run):
    """The regression this harness exists to catch.

    Retrieval is unchanged, so recall and MRR do not move. Only groundedness
    does — which is why a retrieval-only dashboard misses this failure.
    """
    from pipeline import build_broken

    broken = evaluate(load_dataset(HERE / "questions.jsonl"), build_broken(), metrics=METRICS)

    assert broken.aggregate()["recall@3"] == run.aggregate()["recall@3"]
    result = compare(broken, {"aggregate": run.aggregate()}, tolerance=0.01)
    assert not result.passed
    assert any(f.metric == "groundedness" for f in result.findings)
